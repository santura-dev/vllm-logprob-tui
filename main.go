package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
)

type model struct {
	textInput       textinput.Model
	response        string
	systemStats     string
	tokenStats      string
	generationTime  float64
	finishReason    string
	viewport        viewport.Model
	inputFocused    bool
	loading         bool
	ready           bool
	contentCache    string
	responseChanged bool
	statsChanged    bool
}

type responseMsg struct {
	response       string
	tokenUsage     map[string]interface{}
	generationTime float64
	finishReason   string
}

func initialModel() model {
	ti := textinput.New()
	ti.Placeholder = "Enter your query..."
	ti.Focus()

	vp := viewport.New(80, 20)
	vp.SetContent("Initializing...")

	return model{
		textInput:       ti,
		response:        "",
		systemStats:     "Loading...",
		tokenStats:      "Loading...",
		generationTime:  0,
		finishReason:    "",
		viewport:        vp,
		inputFocused:    true,
		loading:         false,
		ready:           false,
		contentCache:    "",
		responseChanged: false,
		statsChanged:    false,
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(textinput.Blink, updateStats())
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	var vCmd tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		if !m.ready {
			m.viewport = viewport.New(msg.Width, msg.Height-2)
			m.ready = true
		} else {
			m.viewport.Width = msg.Width
			m.viewport.Height = msg.Height - 2
		}
		m.updateViewportContent()

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
				query := m.textInput.Value()
				m.loading = true
				m.response = "Generating response..."
				m.updateViewportContent()
				return m, sendQuery(query)
			}
		}

	case responseMsg:
		m.response = msg.response
		m.generationTime = msg.generationTime
		m.finishReason = msg.finishReason
		if usage, ok := msg.tokenUsage["total_tokens"]; ok {
			m.tokenStats = fmt.Sprintf("Total Tokens: %v", usage)
		}
		m.loading = false
		m.responseChanged = true
		return m, updateStats()

	case statsMsg:
		m.systemStats = string(msg)
		m.statsChanged = true
		return m, updateStats()
	}

	if m.inputFocused {
		m.textInput, cmd = m.textInput.Update(msg)
	} else {
		// Pass keyboard events to viewport when not focused on input
		m.viewport, vCmd = m.viewport.Update(msg)
	}

	return m, tea.Batch(cmd, vCmd)
}

func (m *model) updateViewportContent() {
	maxWidth := m.viewport.Width - 4
	if maxWidth < 20 {
		maxWidth = 20
	}

	style := lipgloss.NewStyle().
		Padding(1).
		Border(lipgloss.RoundedBorder()).
		Width(maxWidth)

	focusMsg := "(Tab to toggle focus; ↑↓/PgUp/PgDown to scroll when unfocused)"
	if m.inputFocused {
		focusMsg = "(Tab to unfocus for scrolling)"
	}

	inputView := style.Render(fmt.Sprintf("Query: %s\n%s", m.textInput.View(), focusMsg))
	response := m.response
	if len(response) > 500 {
		response = response[:500] + "..."
	}
	responseView := style.Render(fmt.Sprintf("Response:\n%s", response))
	analysisView := style.Render(fmt.Sprintf("Response Analysis:\nGeneration Time: %.2fs\nFinish Reason: %s", m.generationTime, m.finishReason))
	systemView := style.Render(fmt.Sprintf("System Stats:\n%s", m.systemStats))
	tokenView := style.Render(fmt.Sprintf("Token Stats:\n%s", m.tokenStats))

	content := lipgloss.JoinVertical(lipgloss.Left, inputView, responseView, analysisView, systemView, tokenView)
	m.viewport.SetContent(content)
}

func (m model) View() string {
	if !m.ready {
		return "Initializing..."
	}

	if !m.responseChanged && !m.statsChanged {
		scrollInfo := fmt.Sprintf("\n%d%% scrolled", int(m.viewport.ScrollPercent()*100))
		return m.contentCache + scrollInfo
	}

	// Rebuild content
	style := lipgloss.NewStyle().Padding(1).Border(lipgloss.RoundedBorder()).Width(m.viewport.Width - 4)

	focusMsg := "(Tab to toggle focus; ↑↓/PgUp/PgDown to scroll when unfocused)"
	if m.inputFocused {
		focusMsg = "(Tab to unfocus for scrolling)"
	}

	inputView := style.Render(fmt.Sprintf("Query: %s\n%s", m.textInput.View(), focusMsg))
	response := m.response
	if len(response) > 500 {
		response = response[:500] + "..."
	}
	responseView := style.Render(fmt.Sprintf("Response:\n%s", response))
	analysisView := style.Render(fmt.Sprintf("Response Analysis:\nGeneration Time: %.2fs\nFinish Reason: %s", m.generationTime, m.finishReason))
	systemView := style.Render(fmt.Sprintf("System Stats:\n%s", m.systemStats))
	tokenView := style.Render(fmt.Sprintf("Token Stats:\n%s", m.tokenStats))

	content := lipgloss.JoinVertical(lipgloss.Left, inputView, responseView, analysisView, systemView, tokenView)
	m.viewport.SetContent(content)
	m.contentCache = content
	m.responseChanged = false
	m.statsChanged = false

	scrollInfo := fmt.Sprintf("\n%d%% scrolled", int(m.viewport.ScrollPercent()*100))
	return m.viewport.View() + scrollInfo
}

type statsMsg string

func updateStats() tea.Cmd {
	return tea.Tick(time.Second*10, func(t time.Time) tea.Msg {
		var stats strings.Builder

		// GPU stats
		cmd := exec.Command("nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits")
		output, err := cmd.Output()
		if err == nil {
			lines := strings.Split(strings.TrimSpace(string(output)), "\n")
			if len(lines) > 0 {
				parts := strings.Split(lines[0], ", ")
				if len(parts) >= 5 {
					used, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
					total, _ := strconv.Atoi(strings.TrimSpace(parts[1]))
					util, _ := strconv.Atoi(strings.TrimSpace(parts[2]))
					temp, _ := strconv.Atoi(strings.TrimSpace(parts[3]))
					power, _ := strconv.ParseFloat(strings.TrimSpace(parts[4]), 64)
					stats.WriteString(fmt.Sprintf("GPU Memory: %d/%d MB\nGPU Utilization: %d%%\nGPU Temp: %d°C\nGPU Power: %.1f W\n", used, total, util, temp, power))
				}
			}
		} else {
			stats.WriteString("GPU Stats: Not Available\n")
		}

		// CPU stats
		cpuPercent, _ := cpu.Percent(0, false)
		if len(cpuPercent) > 0 {
			stats.WriteString(fmt.Sprintf("CPU Utilization: %.1f%%\n", cpuPercent[0]))
		}

		// RAM stats
		memInfo, _ := mem.VirtualMemory()
		stats.WriteString(fmt.Sprintf("RAM Used: %d/%d MB (%.1f%%)\n", memInfo.Used/1024/1024, memInfo.Total/1024/1024, memInfo.UsedPercent))

		return statsMsg(stats.String())
	})
}

func sendQuery(query string) tea.Cmd {
	return func() tea.Msg {
		start := time.Now()

		data := map[string]interface{}{
			"model":      "facebook/opt-125m",
			"prompt":     query,
			"max_tokens": 100,
			"logprobs":   5,
		}

		jsonData, _ := json.Marshal(data)
		resp, err := http.Post("http://localhost:8000/v1/completions", "application/json", bytes.NewBuffer(jsonData))
		if err != nil {
			return responseMsg{response: "Error: " + err.Error()}
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		var result map[string]interface{}
		json.Unmarshal(body, &result)

		if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if text, ok := choice["text"].(string); ok {
					duration := time.Since(start).Seconds()
					usage := result["usage"].(map[string]interface{})
					totalTokens := usage["total_tokens"].(float64)
					speed := totalTokens / duration
					finishReason := choice["finish_reason"].(string)
					return responseMsg{
						response:       text,
						tokenUsage:     map[string]interface{}{"total_tokens": totalTokens, "speed": speed},
						generationTime: duration,
						finishReason:   finishReason,
					}
				}
			}
		}
		return responseMsg{response: "Error parsing response"}
	}
}

func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		log.Fatal(err)
	}
}
