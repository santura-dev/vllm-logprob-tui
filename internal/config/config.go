package config

import (
	"flag"
	"fmt"
	"os"
	"time"
)

// Config holds all runtime settings, resolved from CLI flags with env fallbacks.
type Config struct {
	Server          string
	Model           string
	MaxTokens       int
	TopK            int
	MetricsInterval time.Duration
}

// Load parses flags; flags without explicit values fall back to env vars, then defaults.
func Load() (*Config, error) {
	cfg := &Config{}

	flag.StringVar(&cfg.Server, "server", envOr("VLLM_SERVER", "http://localhost:8000"), "vLLM server base URL (env: VLLM_SERVER)")
	flag.StringVar(&cfg.Model, "model", envOr("VLLM_MODEL", "facebook/opt-125m"), "model name (env: VLLM_MODEL)")
	flag.IntVar(&cfg.MaxTokens, "max-tokens", 100, "max tokens to generate")
	flag.IntVar(&cfg.TopK, "top-k", 5, "number of top logprobs to request")
	interval := flag.Duration("metrics-interval", time.Second, "stats/metrics refresh interval")
	flag.Parse()

	cfg.MetricsInterval = *interval
	if cfg.Server == "" {
		return nil, fmt.Errorf("--server is required (or set VLLM_SERVER)")
	}
	return cfg, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
