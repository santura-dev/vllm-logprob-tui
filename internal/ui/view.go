package ui

import (
	"fmt"
	"math"
	"strings"

	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/lipgloss"

	"github.com/santura-dev/vllm-logprob-tui/internal/vllm"
)

var (
	boxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("238")).
			Padding(0, 1)

	titleStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("39")).Bold(true)
	dimStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("243"))
	valueStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("252"))
	accentStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("141")) // soft violet

	// probability heat: indigo -> cyan -> mint, no traffic lights
	pHigh = lipgloss.NewStyle().Foreground(lipgloss.Color("158")) // mint
	pMid  = lipgloss.NewStyle().Foreground(lipgloss.Color("117")) // sky
	pLow  = lipgloss.NewStyle().Foreground(lipgloss.Color("99"))  // violet

	okDot    = lipgloss.NewStyle().Foreground(lipgloss.Color("158"))
	errStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("174")) // muted rose
)

func newViewport(width, height int) viewport.Model {
	vp := viewport.New(width, height-2)
	vp.Style = boxStyle
	return vp
}

// Layout: fixed-height panels (query, request, server) + response panel that
// takes all remaining terminal lines, scrolling internally. Nothing overflows.
func (m Model) View() string {
	if !m.ready {
		return "Initializing..."
	}

	query := boxStyle.Render(m.renderPrompt())
	request := boxStyle.Render(m.renderRequestStats())
	server := boxStyle.Render(m.renderServerStats())

	fixed := lipgloss.Height(query) + lipgloss.Height(request) + lipgloss.Height(server)
	// +3 for borders/padding slack, -1 status bar
	respHeight := m.viewport.Height - fixed - 2
	if respHeight < 4 {
		respHeight = 4
	}

	respView := m.renderResponsePanel(respHeight)
	return lipgloss.JoinVertical(lipgloss.Left, query, respView, request, server) + "\n" + m.statusBar()
}

// renderResponsePanel renders response + logprob table inside its own
// viewport, height-capped to fit the terminal.
func (m *Model) renderResponsePanel(height int) string {
	vp := m.respViewport
	vp.Height = height
	vp.Width = m.viewport.Width
	vp.SetContent(m.renderResponse())
	return vp.View()
}

func (m Model) statusBar() string {
	server := errStyle.Render("● offline")
	if m.serverUp {
		server = okDot.Render("● connected")
	} else if m.metricsOK {
		server = okDot.Render("● connected")
	}

	parts := []string{server}
	if m.loading {
		parts = append(parts, m.spinner.View()+" generating")
	}
	parts = append(parts, m.focusHint())
	return dimStyle.Render(strings.Join(parts, "  "))
}

func (m Model) focusHint() string {
	if m.inputFocused {
		return "tab scroll · enter send · ↑↓ history · ctrl+c quit"
	}
	return "tab edit · pgup/pgdn scroll · ↑↓ history · ctrl+c quit"
}

func (m *Model) renderContent() string {
	var sections []string

	sections = append(sections, boxStyle.Render(m.renderPrompt()))
	sections = append(sections, boxStyle.Render(m.renderResponse()))
	sections = append(sections, boxStyle.Render(m.renderRequestStats()))
	sections = append(sections, boxStyle.Render(m.renderServerStats()))

	return lipgloss.JoinVertical(lipgloss.Left, sections...)
}

func (m *Model) renderPrompt() string {
	focus := ""
	if !m.inputFocused {
		focus = dimStyle.Render("  (scrolling — tab to edit)")
	}
	return titleStyle.Render("PROMPT") + focus + "\n" + valueStyle.Render(m.textInput.View())
}

func (m *Model) renderResponse() string {
	if m.err != nil {
		return titleStyle.Render("RESPONSE") + "\n" + errStyle.Render("✕ "+m.err.Error())
	}
	if m.loading && m.response == "" {
		return titleStyle.Render("RESPONSE") + "\n" + dimStyle.Render(m.spinner.View()+" waiting for first token...")
	}
	if m.response == "" {
		return titleStyle.Render("RESPONSE") + "\n" + dimStyle.Render("press enter to generate")
	}

	var b strings.Builder
	b.WriteString(titleStyle.Render("RESPONSE") + "\n")
	b.WriteString(valueStyle.Render(m.response))
	if len(m.logprobs) > 0 {
		ppl := vllm.Perplexity(m.logprobs)
		header := titleStyle.Render("TOKENS")
		if ppl > 0 {
			header += dimStyle.Render(fmt.Sprintf("  perplexity %.2f", ppl))
		}
		b.WriteString("\n\n" + header + "\n")
		b.WriteString(m.renderLogprobTable())
	}
	return b.String()
}

func (m *Model) renderLogprobTable() string {
	var b strings.Builder
	for _, tp := range m.logprobs {
		p := math.Exp(tp.LogProb)
		style := probStyle(p)
		b.WriteString(fmt.Sprintf("  %s %s %s\n",
			style.Render(fmt.Sprintf("%-12s", cleanToken(tp.Token))),
			style.Render(probabilityBar(p)),
			dimStyle.Render(fmt.Sprintf("%5.1f%%", p*100)),
		))
		if len(tp.Alts) > 0 {
			b.WriteString(dimStyle.Render("   ↳ "+renderAlts(tp.Alts)) + "\n")
		}
	}
	return b.String()
}

func renderAlts(alts []vllm.Alt) string {
	var parts []string
	for i, a := range alts {
		if i >= 3 {
			break
		}
		p := math.Exp(a.LogProb)
		parts = append(parts, fmt.Sprintf("%s %.0f%%", cleanToken(a.Token), p*100))
	}
	return strings.Join(parts, " · ")
}

// cleanToken renders BPE markers readably.
func cleanToken(tok string) string {
	r := strings.NewReplacer("Ġ", "·", "Ċ", "⏎", "▁", "·")
	return r.Replace(tok)
}

func probStyle(p float64) lipgloss.Style {
	switch {
	case p >= 0.9:
		return pHigh
	case p >= 0.5:
		return pMid
	default:
		return pLow
	}
}

// probabilityBar uses a fine gradient block for a smoother look.
func probabilityBar(p float64) string {
	const width = 18
	filled := int(p * float64(width))
	return strings.Repeat("─", 0) + strings.Repeat("▰", filled) + strings.Repeat("▱", width-filled)
}

func (m *Model) renderRequestStats() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("REQUEST") + "\n")
	if m.finishReason == "" {
		b.WriteString(dimStyle.Render("  no request yet"))
		return b.String()
	}
	b.WriteString(valueStyle.Render(fmt.Sprintf("  ttft %.2fs   total %.2fs   %.1f tok/s", m.ttft, m.genTime, m.tokensPerSec)) + "\n")
	b.WriteString(dimStyle.Render(fmt.Sprintf("  finish %s   tokens %d prompt / %d completion / %d total",
		m.finishReason, m.usage.PromptTokens, m.usage.CompletionTokens, m.usage.TotalTokens)))
	return b.String()
}

func (m *Model) renderServerStats() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render("SERVER") + "\n")
	b.WriteString(dimStyle.Render(m.systemStats))
	if m.metricsOK {
		b.WriteString("\n" + valueStyle.Render(fmt.Sprintf("vLLM  %d running   %d queued   %d swapped   KV cache %.0f%%",
			m.batchMetrics.Running, m.batchMetrics.Waiting, m.batchMetrics.Swapped, m.batchMetrics.CachePerc*100)))
	} else {
		b.WriteString("\n" + dimStyle.Render("vLLM metrics unavailable"))
	}
	return b.String()
}
