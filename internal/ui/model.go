package ui

import (
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/santura-dev/vllm-logprob-tui/internal/config"
	"github.com/santura-dev/vllm-logprob-tui/internal/vllm"
)

// Model is the bubbletea application model.
type Model struct {
	cfg       *config.Config
	client    *vllm.Client
	textInput textinput.Model
	viewport  viewport.Model
	ready     bool

	inputFocused bool
	loading      bool

	response     string
	logprobs     []vllm.TokenProb
	genTime      float64
	ttft         float64
	tokensPerSec float64
	finishReason string
	usage        vllm.Usage
	err          error

	systemStats  string
	batchMetrics vllm.BatchMetrics
	metricsOK    bool
}

// New builds the initial model from config.
func New(cfg *config.Config) Model {
	ti := textinput.New()
	ti.Placeholder = "Enter your query..."
	ti.Focus()

	return Model{
		cfg:       cfg,
		client:    vllm.NewClient(cfg.Server),
		textInput: ti,
	}
}

// Init starts the blink loop and the stats ticker.
func (m Model) Init() tea.Cmd {
	return tea.Batch(textinput.Blink, tickStats(m.cfg.MetricsInterval))
}

// tickStatsMsg drives the periodic system + metrics refresh.
type tickStatsMsg struct{}

func tickStats(interval time.Duration) tea.Cmd {
	return tea.Tick(interval, func(time.Time) tea.Msg { return tickStatsMsg{} })
}
