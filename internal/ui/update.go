package ui

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/viewport"
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

// chunkMsg carries one streamed delta while generation is in flight.
type chunkMsg struct {
	text  string
	token *vllm.TokenProb
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
			m.respViewport = viewport.New(msg.Width, msg.Height-2)
			m.ready = true
		} else {
			m.viewport.Width = msg.Width
			m.viewport.Height = msg.Height
			m.respViewport.Width = msg.Width
		}
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

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
		case "up", "down":
			if m.inputFocused && !m.loading {
				m.browseHistory(msg.String() == "up")
				return m, nil
			}
		case "enter":
			if m.inputFocused && !m.loading {
				query := strings.TrimSpace(m.textInput.Value())
				if query == "" {
					return m, nil
				}
				m.pushHistory(query)
				m.loading = true
				m.response = ""
				m.logprobs = nil
				m.err = nil
				m.finishReason = ""
				m.viewport.SetContent(m.renderContent())
				m.activeStream = newStream()
				produce := produceCmd(m.client, m.cfg, query, m.activeStream)
				return m, tea.Batch(produce, m.activeStream.listen(), m.spinner.Tick)
			}
		}

	case chunkMsg:
		if m.loading {
			m.response += msg.text
			if msg.token != nil {
				m.logprobs = append(m.logprobs, *msg.token)
			}
			m.respViewport.GotoBottom()
		}
		return m, m.activeStream.listen()

	case responseMsg:
		m.loading = false
		m.activeStream = nil
		if msg.err != nil {
			m.err = msg.err
			m.serverUp = false
		} else {
			m.response = msg.result.Text
			m.logprobs = msg.result.TokenLogprobs
			m.genTime = msg.result.TotalTime
			m.ttft = msg.result.TTFT
			m.tokensPerSec = msg.result.TokensPerSec
			m.finishReason = msg.result.FinishReason
			m.usage = msg.result.Usage
			m.serverUp = true
		}
		m.viewport.SetContent(m.renderContent())
		return m, nil

	case statsMsg:
		m.systemStats = formatStats(msg.system)
		m.batchMetrics = msg.metrics
		m.metricsOK = msg.metricsOK
		m.viewport.SetContent(m.renderContent())
		return m, nil

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

func (m *Model) pushHistory(q string) {
	if len(m.history) == 0 || m.history[len(m.history)-1] != q {
		m.history = append(m.history, q)
	}
	m.historyIdx = -1
}

func (m *Model) browseHistory(older bool) {
	if len(m.history) == 0 {
		return
	}
	if older {
		if m.historyIdx == -1 {
			m.historyIdx = len(m.history) - 1
		} else if m.historyIdx > 0 {
			m.historyIdx--
		}
	} else {
		if m.historyIdx == -1 {
			return
		}
		m.historyIdx++
		if m.historyIdx >= len(m.history) {
			m.historyIdx = -1
			m.textInput.SetValue("")
			return
		}
	}
	m.textInput.SetValue(m.history[m.historyIdx])
}

// activeStream carries chunks from the producer goroutine into Update.
// A tea.Cmd returns exactly one Msg, so a channel + re-armed listener is the
// standard pattern for continuous streams.
type streamChan struct {
	ch     chan tea.Msg
	closed bool
}

func newStream() *streamChan {
	return &streamChan{ch: make(chan tea.Msg, 256)}
}

func (s *streamChan) send(msg tea.Msg) {
	if !s.closed {
		s.ch <- msg
	}
}

func (s *streamChan) close() {
	s.closed = true
}

// listen returns the next message; nil when the stream is closed.
func (s *streamChan) listen() tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-s.ch
		if !ok {
			return nil
		}
		return msg
	}
}

func produceCmd(client *vllm.Client, cfg *config.Config, query string, s *streamChan) tea.Cmd {
	return func() tea.Msg {
		defer s.close()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		req := vllm.CompletionRequest{
			Model:     cfg.Model,
			Prompt:    query,
			MaxTokens: cfg.MaxTokens,
			Logprobs:  cfg.TopK,
		}
		res, err := client.StreamCompletion(ctx, req, func(ev vllm.StreamEvent) {
			s.send(chunkMsg{text: ev.Text, token: ev.Token})
		})
		return responseMsg{result: res, err: err}
	}
}

func fetchStatsCmd(client *vllm.Client) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		stats := system.Collect()
		metrics, err := client.FetchMetrics(ctx)
		return statsMsg{
			system:    stats,
			metrics:   metrics,
			metricsOK: err == nil,
		}
	}
}

func formatStats(s system.Stats) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("GPU: %s\n", s.GPU))
	b.WriteString(fmt.Sprintf("CPU: %.1f%%\n", s.CPUPercent))
	b.WriteString(fmt.Sprintf("RAM: %d/%d MB (%.1f%%)", s.RAMUsedMB, s.RAMTotalMB, s.RAMPercent))
	return b.String()
}
