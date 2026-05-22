package vllm

import (
	"encoding/json"
	"math"
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

func TestParseTopLogprobs(t *testing.T) {
	floatForm := map[string]json.RawMessage{}
	json.Unmarshal([]byte(`{"hello": -0.693, "world": -2.3}`), &floatForm)
	objForm := map[string]json.RawMessage{}
	json.Unmarshal([]byte(`{"hello": {"logprob": -1.2}, "world": {"logprog": -3.4}}`), &objForm)

	tests := []struct {
		name       string
		alts       map[string]json.RawMessage
		tok        string
		wantLP     float64
		wantAltCnt int
		wantFirst  string // top alternative token, "" if unchecked
	}{
		{"float form, chosen present", floatForm, "hello", -0.693, 1, "world"},
		{"chosen missing: take max, dedupe", floatForm, "missing", -0.693, 1, "world"},
		{"object logprob key", objForm, "hello", -1.2, 1, "world"},
		{"object logprog key", objForm, "world", -3.4, 1, "hello"},
		{"empty alternatives", map[string]json.RawMessage{}, "x", 0, 0, ""},
	}
	for _, tt := range tests {
		tp := parseTopLogprobs(tt.alts, tt.tok)
		if tp.LogProb != tt.wantLP {
			t.Errorf("%s: LogProb = %v, want %v", tt.name, tp.LogProb, tt.wantLP)
		}
		if len(tp.Alts) != tt.wantAltCnt {
			t.Errorf("%s: alt count = %d, want %d", tt.name, len(tp.Alts), tt.wantAltCnt)
			continue
		}
		if tt.wantFirst != "" && tp.Alts[0].Token != tt.wantFirst {
			t.Errorf("%s: top alt = %q, want %q", tt.name, tp.Alts[0].Token, tt.wantFirst)
		}
	}
}

func TestParseTopLogprobsSorted(t *testing.T) {
	alts := map[string]json.RawMessage{}
	json.Unmarshal([]byte(`{"a": -5.0, "b": -1.0, "c": -3.0}`), &alts)
	tp := parseTopLogprobs(alts, "b")
	if tp.LogProb != -1.0 || len(tp.Alts) != 2 || tp.Alts[0].Token != "c" || tp.Alts[1].Token != "a" {
		t.Errorf("parseTopLogprobs = %+v, want chosen b, alts [c, a]", tp)
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
	tp := parseTopLogprobs(chunk.Choices[0].Logprobs.TopLogprobs[0], " foo")
	if tp.LogProb != -0.5 {
		t.Errorf("LogProb = %v, want -0.5", tp.LogProb)
	}
}

func TestPerplexity(t *testing.T) {
	// uniform over 4 tokens: perplexity = 4
	tokens := make([]TokenProb, 4)
	for i := range tokens {
		tokens[i].LogProb = math.Log(0.25)
	}
	if got := Perplexity(tokens); math.Abs(got-4.0) > 1e-9 {
		t.Errorf("Perplexity = %v, want 4.0", got)
	}
	if got := Perplexity(nil); got != 0 {
		t.Errorf("Perplexity(nil) = %v, want 0", got)
	}
}
