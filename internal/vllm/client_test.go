package vllm

import (
	"encoding/json"
	"testing"
)

func TestSplitMetric(t *testing.T) {
	tests := []struct {
		line  string
		name  string
		value string
		ok    bool
	}{
		{"vllm:num_requests_running 3", "vllm:num_requests_running", "3", true},
		{`vllm:gpu_cache_usage_perc{model="foo"} 0.42`, "vllm:gpu_cache_usage_perc", "0.42", true},
		{"# HELP vllm:num_requests_running count", "", "", false},
		{"garbage", "", "", false},
	}
	for _, tt := range tests {
		name, value, ok := splitMetric(tt.line)
		if name != tt.name || value != tt.value || ok != tt.ok {
			t.Errorf("splitMetric(%q) = (%q, %q, %v), want (%q, %q, %v)", tt.line, name, value, ok, tt.name, tt.value, tt.ok)
		}
	}
}

func TestExtractLogprob(t *testing.T) {
	floatForm := map[string]json.RawMessage{}
	json.Unmarshal([]byte(`{"hello": -0.693, "world": -2.3}`), &floatForm)
	objForm := map[string]json.RawMessage{}
	json.Unmarshal([]byte(`{"hello": {"logprob": -1.2}, "world": {"logprog": -3.4}}`), &objForm)

	tests := []struct {
		name   string
		alts   map[string]json.RawMessage
		tok    string
		want   float64
		wantOK bool
	}{
		{"float form, chosen token", floatForm, "hello", -0.693, true},
		{"float form, fallback first", floatForm, "missing", -0.693, true},
		{"object logprob key", objForm, "hello", -1.2, true},
		{"object logprog key", objForm, "world", -3.4, true},
		{"empty alternatives", map[string]json.RawMessage{}, "x", 0, false},
	}
	for _, tt := range tests {
		got, ok := extractLogprob(tt.alts, tt.tok)
		if ok != tt.wantOK || (ok && got != tt.want) {
			t.Errorf("%s: extractLogprob = (%v, %v), want (%v, %v)", tt.name, got, ok, tt.want, tt.wantOK)
		}
	}
}

func TestStreamChunkDecode(t *testing.T) {
	payload := `{"choices":[{"text":" foo","logprobs":{"top_logprobs":[{" foo": -0.5, "bar": -1.5}]}}]}`
	var chunk StreamingChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(chunk.Choices) != 1 || chunk.Choices[0].Text != " foo" {
		t.Fatalf("unexpected choices: %+v", chunk.Choices)
	}
	lp, ok := extractLogprob(chunk.Choices[0].Logprobs.TopLogprobs[0], " foo")
	if !ok || lp != -0.5 {
		t.Errorf("extractLogprob = (%v, %v), want (-0.5, true)", lp, ok)
	}
}
