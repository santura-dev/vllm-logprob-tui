package vllm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
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

// StreamCompletion sends a streaming completion request and calls onChunk for
// each delta. It returns the accumulated result. onChunk is optional (may be nil).
func (c *Client) StreamCompletion(ctx context.Context, req CompletionRequest, onChunk func(text string)) (*CompletionResult, error) {
	req.Stream = true
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
		if chunk.Usage != nil {
			result.Usage = *chunk.Usage
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]

		if choice.Logprobs != nil && len(choice.Logprobs.TopLogprobs) > 0 && choice.Text != "" {
			// one top_logprobs entry per token in this chunk's text
			alternatives := choice.Logprobs.TopLogprobs[0]
			if lp, ok := extractLogprob(alternatives, choice.Text); ok {
				result.TokenLogprobs = append(result.TokenLogprobs, TokenProb{Token: choice.Text, LogProb: lp})
			}
		}

		if choice.Text != "" {
			if !firstToken {
				ttft = time.Since(start)
				firstToken = true
			}
			result.Text += choice.Text
			if onChunk != nil {
				onChunk(choice.Text)
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
	result.TokensPerSec = float64(len(result.TokenLogprobs)) / result.TotalTime
	return result, nil
}

// extractLogprob finds the logprob for the chosen token, accepting float values
// or {"logprob": ...}/{"logprog": ...} objects, and falls back to any entry.
func extractLogprob(alternatives map[string]json.RawMessage, token string) (float64, bool) {
	for _, key := range []string{token, strings.TrimSpace(token)} {
		if raw, ok := alternatives[key]; ok {
			return parseLogprobValue(raw)
		}
	}
	for _, raw := range alternatives {
		if lp, ok := parseLogprobValue(raw); ok {
			return lp, true
		}
	}
	return 0, false
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
