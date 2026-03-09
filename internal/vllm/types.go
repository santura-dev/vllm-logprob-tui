package vllm

import "encoding/json"

// CompletionRequest is the payload for /v1/completions.
type CompletionRequest struct {
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"`
	Logprobs  int    `json:"logprobs"`
	Stream    bool   `json:"stream"`
}

// StreamingChunk is one SSE data payload from a streamed completion.
type StreamingChunk struct {
	Choices []chunkChoice `json:"choices"`
}

type chunkChoice struct {
	Text         string     `json:"text"`
	FinishReason *string    `json:"finish_reason"`
	Logprobs     *choiceLps `json:"logprobs"`
}

type choiceLps struct {
	TopLogprobs []map[string]json.RawMessage `json:"top_logprobs"`
}

// Usage is the token accounting reported with the completion.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// TokenProb is one generated token with its log probability.
type TokenProb struct {
	Token   string
	LogProb float64
}

// CompletionResult is the accumulated outcome of a streamed completion.
type CompletionResult struct {
	Text          string
	FinishReason  string
	Usage         Usage
	TTFT          float64
	TotalTime     float64
	TokensPerSec  float64
	TokenLogprobs []TokenProb
}

// BatchMetrics holds gauges scraped from the vLLM /metrics endpoint.
type BatchMetrics struct {
	Running   int
	Waiting   int
	Swapped   int
	CachePerc float64
}
