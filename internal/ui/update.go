package ui

import (
	"context"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/santura-dev/vllm-logprob-tui/internal/config"
	"github.com/santura-dev/vllm-logprob-tui/internal/system"
	"github.com/santura-dev/vllm-logprob-tui/internal/vllm"
)

// responseMsg carries a finished (or failed) completion.
type responseMsg struct {
	result *vllm.CompletionResult
	err    error
}

// statsMsg carries a system + batch metrics snapshot.
type statsMsg struct {
	system    system.Stats
	metrics   vllm.BatchMetrics
	metricsOK bool
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		if !m.ready {
			m.viewport = newViewport(msg.Width, msg.Height)
			m.ready = true
		} else {
			m.viewport.Width = msg.Width
			m.viewport.Height = msg.Height
		}
		m.viewport.SetContent(m.renderContent())
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "esc":
			return m, tea.Quit
		case "tab":
			m.inputFocused = !m.inputFocused
			if m.inputFocused {
				m.textInput.Focus()
			} else {
				m.textInput.Blur()
			}
			return m, nil
		case "enter":
			if m.inputFocused && !m.loading {
				query := strings.TrimSpace(m.textInput.Value())
				if query == "" {
					return m, nil
				}
				m.loading = true
				m.response = "Generating..."
				m.err = nil
				m.viewport.SetContent(m.renderContent())
				return m, queryCmd(m.client, m.cfg, query)
			}
		}

	case responseMsg:
		m.loading = false
		if msg.err != nil {
			m.err = msg.err
		} else {
			m.response = msg.result.Text
			m.logprobs = msg.result.TokenLogprobs
			m.genTime = msg.result.TotalTime
			m.ttft = msg.result.TTFT
			m.tokensPerSec = msg.result.TokensPerSec
			m.finishReason = msg.result.FinishReason
			m.usage = msg.result.Usage
		}
		m.viewport.SetContent(m.renderContent())
		return m, nil

	case statsMsg:
		m.systemStats = formatStats(msg.system)
		m.batchMetrics = msg.metrics
		m.metricsOK = msg.metricsOK
		m.viewport.SetContent(m.renderContent())
		return m, tickStats(m.cfg.MetricsInterval)

	case tickStatsMsg:
		return m, tea.Batch(fetchStatsCmd(m.client), tickStats(m.cfg.MetricsInterval))
	}

	var cmd tea.Cmd
	if m.inputFocused {
		m.textInput, cmd = m.textInput.Update(msg)
	} else {
		m.viewport, cmd = m.viewport.Update(msg)
	}
	return m, cmd
}

func queryCmd(client *vllm.Client, cfg *config.Config, query string) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		req := vllm.CompletionRequest{
			Model:     cfg.Model,
			Prompt:    query,
			MaxTokens: cfg.MaxTokens,
			Logprobs:  cfg.TopK,
		}
		res, err := client.StreamCompletion(ctx, req, nil)
		return responseMsg{result: res, err: err}
	}
}

func fetchStatsCmd(client *vllm.Client) tea.Cmd {
	return func() tea.Msg {
		stats := system.Collect()
		metrics, err := client.FetchMetrics(context.Background())
		return statsMsg{system: stats, metrics: metrics, metricsOK: err == nil}
	}
}

func formatStats(s system.Stats) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("GPU: %s\n", s.GPU))
	b.WriteString(fmt.Sprintf("CPU: %.1f%%\n", s.CPUPercent))
	b.WriteString(fmt.Sprintf("RAM: %d/%d MB (%.1f%%)", s.RAMUsedMB, s.RAMTotalMB, s.RAMPercent))
	return b.String()
}
