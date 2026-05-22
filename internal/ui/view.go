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
			Padding(0, 1)

	probHigh = lipgloss.NewStyle().Foreground(lipgloss.Color("46"))
	probMid  = lipgloss.NewStyle().Foreground(lipgloss.Color("220"))
	probLow  = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))

	dimStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("243"))
	headerStyle = lipgloss.NewStyle().Bold(true)
)

func newViewport(width, height int) viewport.Model {
	vp := viewport.New(width, height-2)
	vp.Style = boxStyle
	return vp
}

func (m Model) View() string {
	if !m.ready {
		return "Initializing..."
	}
	return m.viewport.View() + "\n" + m.statusBar()
}

func (m Model) statusBar() string {
	server := probLow.Render("● server down")
	if m.serverUp {
		server = probHigh.Render("● server up")
	} else if m.metricsOK {
		server = probHigh.Render("● server up")
	}

	parts := []string{server}
	if m.loading {
		parts = append(parts, m.spinner.View()+" generating")
	}
	parts = append(parts, m.focusHint())
	return dimStyle.Render(strings.Join(parts, " · "))
}

func (m Model) focusHint() string {
	if m.inputFocused {
		return "tab: scroll · enter: send · ↑/↓: history · ctrl+c: quit"
	}
	return "tab: edit · enter/pgup/pgdn: scroll · ↑/↓: history · ctrl+c: quit"
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
		focus = dimStyle.Render(" (scrolling — tab to edit)")
	}
	return fmt.Sprintf("%s\n%s", headerStyle.Render("Query"), m.textInput.View()) + focus
}

func (m *Model) renderResponse() string {
	if m.err != nil {
		return headerStyle.Render("Response") + fmt.Sprintf("\n%s", probLow.Render("error: "+m.err.Error()))
	}
	if m.loading && m.response == "" {
		return headerStyle.Render("Response") + "\n" + m.spinner.View() + " waiting for first token..."
	}
	if m.response == "" {
		return headerStyle.Render("Response") + "\n(press enter to generate)"
	}

	var b strings.Builder
	b.WriteString(headerStyle.Render("Response") + "\n")
	b.WriteString(m.response)
	if len(m.logprobs) > 0 {
		b.WriteString("\n\n" + headerStyle.Render("Token probabilities"))
		ppl := vllm.Perplexity(m.logprobs)
		if ppl > 0 {
			b.WriteString(dimStyle.Render(fmt.Sprintf("  (perplexity %.2f)", ppl)))
		}
		b.WriteString("\n")
		b.WriteString(m.renderLogprobTable())
	}
	return b.String()
}

func (m *Model) renderLogprobTable() string {
	var b strings.Builder
	for _, tp := range m.logprobs {
		p := math.Exp(tp.LogProb)
		bar := probabilityBar(p)
		style := probStyle(p)
		b.WriteString(fmt.Sprintf("  %s %s %s\n",
			style.Render(fmt.Sprintf("%-10s", cleanToken(tp.Token))),
			bar,
			style.Render(fmt.Sprintf("%5.1f%%", p*100)),
		))
		if len(tp.Alts) > 0 {
			b.WriteString(dimStyle.Render("           alts: "+renderAlts(tp.Alts)) + "\n")
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

// cleanToken renders BPE space/newline markers readably.
func cleanToken(tok string) string {
	r := strings.NewReplacer("Ġ", "␣", "Ċ", "⏎", "▁", "␣")
	return r.Replace(tok)
}

func probStyle(p float64) lipgloss.Style {
	switch {
	case p >= 0.9:
		return probHigh
	case p >= 0.5:
		return probMid
	default:
		return probLow
	}
}

func probabilityBar(p float64) string {
	filled := int(p * 20)
	return "[" + strings.Repeat("█", filled) + strings.Repeat("░", 20-filled) + "]"
}

func (m *Model) renderRequestStats() string {
	var b strings.Builder
	b.WriteString(headerStyle.Render("Request") + "\n")
	if m.finishReason == "" {
		b.WriteString("  no request yet")
		return b.String()
	}
	b.WriteString(fmt.Sprintf("  ttft %.2fs · total %.2fs · %.1f tok/s\n", m.ttft, m.genTime, m.tokensPerSec))
	b.WriteString(fmt.Sprintf("  finish: %s · tokens: %d prompt / %d completion / %d total",
		m.finishReason, m.usage.PromptTokens, m.usage.CompletionTokens, m.usage.TotalTokens))
	return b.String()
}

func (m *Model) renderServerStats() string {
	var b strings.Builder
	b.WriteString(headerStyle.Render("Server") + "\n")
	b.WriteString(m.systemStats)
	if m.metricsOK {
		b.WriteString(fmt.Sprintf("\nvLLM: %d running · %d queued · %d swapped · KV cache %.1f%%",
			m.batchMetrics.Running, m.batchMetrics.Waiting, m.batchMetrics.Swapped, m.batchMetrics.CachePerc*100))
	} else {
		b.WriteString("\nvLLM metrics: unavailable")
	}
	return b.String()
}
