package ui

import (
	"fmt"
	"math"
	"strings"

	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/lipgloss"
)

var (
	boxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			Padding(0, 1)

	probHigh = lipgloss.NewStyle().Foreground(lipgloss.Color("46"))
	probMid  = lipgloss.NewStyle().Foreground(lipgloss.Color("220"))
	probLow  = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
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
	return m.viewport.View() + "\n" + m.focusHint()
}

func (m Model) focusHint() string {
	if m.inputFocused {
		return "tab: scroll output · enter: send · ctrl+c: quit"
	}
	return "tab: edit prompt · enter/pgup/pgdn: scroll · ctrl+c: quit"
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
		focus = " (scrolling — tab to edit)"
	}
	return fmt.Sprintf("Query:%s\n%s", focus, m.textInput.View())
}

func (m *Model) renderResponse() string {
	if m.err != nil {
		return fmt.Sprintf("Response:\nerror: %v", m.err)
	}
	if m.loading {
		return "Response:\nGenerating..."
	}
	if m.response == "" {
		return "Response:\n(press enter to generate)"
	}

	var b strings.Builder
	b.WriteString("Response:\n")
	b.WriteString(m.response)
	if len(m.logprobs) > 0 {
		b.WriteString("\n\nToken probabilities:\n")
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
		b.WriteString(fmt.Sprintf("  %s %s %.1f%%\n", style.Render(fmt.Sprintf("%-12q", tp.Token)), bar, p*100))
	}
	return b.String()
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
	b.WriteString("Request:\n")
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
	b.WriteString("Server:\n")
	b.WriteString(m.systemStats)
	if m.metricsOK {
		b.WriteString(fmt.Sprintf("\nvLLM: %d running · %d queued · %d swapped · KV cache %.1f%%",
			m.batchMetrics.Running, m.batchMetrics.Waiting, m.batchMetrics.Swapped, m.batchMetrics.CachePerc*100))
	} else {
		b.WriteString("\nvLLM metrics: unavailable")
	}
	return b.String()
}
