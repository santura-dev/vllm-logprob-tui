package vllm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Client talks to a vLLM OpenAI-compatible server.
type Client struct {
	baseURL string
	http    *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 5 * time.Minute},
	}
}

// StreamCompletion sends a streaming completion request. For each text delta it
// calls onEvent with the text and (when available) the token's logprob data.
func (c *Client) StreamCompletion(ctx context.Context, req CompletionRequest, onEvent func(StreamEvent)) (*CompletionResult, error) {
	req.Stream = true
	if req.StreamOptions == nil {
		req.StreamOptions = &StreamOptions{IncludeUsage: true}
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("connect to vLLM server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return nil, fmt.Errorf("vLLM returned %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}

	result := &CompletionResult{}
	start := time.Now()
	var ttft time.Duration
	firstToken := false

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}

		var chunk StreamingChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return nil, fmt.Errorf("decode stream chunk: %w", err)
		}
		// usage-only final chunk has no choices
		if chunk.Usage != nil {
			result.Usage = *chunk.Usage
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]

		var tp *TokenProb
		if choice.Logprobs != nil && len(choice.Logprobs.TopLogprobs) > 0 && choice.Text != "" {
			// one top_logprobs entry per token in this chunk's text
			tp = parseTopLogprobs(choice.Logprobs.TopLogprobs[0], choice.Text)
			result.TokenLogprobs = append(result.TokenLogprobs, *tp)
		}

		if choice.Text != "" {
			if !firstToken {
				ttft = time.Since(start)
				firstToken = true
			}
			result.Text += choice.Text
			if onEvent != nil {
				onEvent(StreamEvent{Text: choice.Text, Token: tp})
			}
		}

		if choice.FinishReason != nil && *choice.FinishReason != "" {
			result.FinishReason = *choice.FinishReason
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read stream: %w", err)
	}
	if !firstToken {
		return nil, fmt.Errorf("no tokens in stream")
	}

	result.TTFT = ttft.Seconds()
	result.TotalTime = time.Since(start).Seconds()
	tokens := result.Usage.CompletionTokens
	if tokens <= 0 {
		tokens = len(result.TokenLogprobs)
	}
	result.TokensPerSec = float64(tokens) / result.TotalTime
	return result, nil
}

// parseTopLogprobs builds the chosen token's entry from one top_logprobs map.
// The chosen token is usually a key; if not (tokenizer mismatch), the closest
// approximation is the highest-probability entry rather than a random one.
func parseTopLogprobs(alternatives map[string]json.RawMessage, token string) *TokenProb {
	tp := &TokenProb{Token: token}

	for tok, raw := range alternatives {
		lp, ok := parseLogprobValue(raw)
		if !ok {
			continue
		}
		if tok == token || tok == strings.TrimSpace(token) {
			tp.LogProb = lp
			continue
		}
		tp.Alts = append(tp.Alts, Alt{Token: tok, LogProb: lp})
	}
	sortAlts(tp.Alts)

	if tp.LogProb == 0 {
		// chosen token not among alternatives: use the top entry as the
		// closest available estimate, and don't duplicate it in the list
		if len(tp.Alts) > 0 {
			tp.LogProb = tp.Alts[0].LogProb
			tp.Alts = tp.Alts[1:]
		}
	}
	return tp
}

func sortAlts(alts []Alt) {
	for i := 1; i < len(alts); i++ {
		for j := i; j > 0 && alts[j].LogProb > alts[j-1].LogProb; j-- {
			alts[j], alts[j-1] = alts[j-1], alts[j]
		}
	}
}

// Perplexity returns exp(mean(-logprob)) over the generated tokens.
func Perplexity(tokens []TokenProb) float64 {
	if len(tokens) == 0 {
		return 0
	}
	var sum float64
	for _, tp := range tokens {
		sum += tp.LogProb
	}
	return math.Exp(-sum / float64(len(tokens)))
}

func parseLogprobValue(raw json.RawMessage) (float64, bool) {
	var f float64
	if err := json.Unmarshal(raw, &f); err == nil {
		return f, true
	}
	var obj struct {
		Logprob *float64 `json:"logprob"`
		Logprog *float64 `json:"logprog"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil {
		switch {
		case obj.Logprob != nil:
			return *obj.Logprob, true
		case obj.Logprog != nil:
			return *obj.Logprog, true
		}
	}
	return 0, false
}

// FetchMetrics scrapes the four batch gauges from the Prometheus endpoint.
func (c *Client) FetchMetrics(ctx context.Context) (BatchMetrics, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/metrics", nil)
	if err != nil {
		return BatchMetrics{}, err
	}
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return BatchMetrics{}, fmt.Errorf("fetch metrics: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return BatchMetrics{}, fmt.Errorf("metrics endpoint returned %s", resp.Status)
	}

	var m BatchMetrics
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		name, value, ok := splitMetric(line)
		if !ok {
			continue
		}
		switch name {
		case "vllm:num_requests_running":
			m.Running, _ = strconv.Atoi(value)
		case "vllm:num_requests_waiting":
			m.Waiting, _ = strconv.Atoi(value)
		case "vllm:num_requests_swapped":
			m.Swapped, _ = strconv.Atoi(value)
		case "vllm:gpu_cache_usage_perc":
			m.CachePerc, _ = strconv.ParseFloat(value, 64)
		}
	}
	return m, scanner.Err()
}

// splitMetric extracts the metric name (before any labels) and its value.
func splitMetric(line string) (string, string, bool) {
	if strings.HasPrefix(line, "#") {
		return "", "", false
	}
	if idx := strings.Index(line, "{"); idx >= 0 {
		end := strings.Index(line, "}")
		if end < 0 {
			return "", "", false
		}
		line = line[:idx] + line[end+1:]
	}
	parts := strings.Fields(strings.TrimSpace(line))
	if len(parts) != 2 {
		return "", "", false
	}
	return parts[0], parts[1], true
}
