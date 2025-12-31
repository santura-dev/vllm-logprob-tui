# vLLM TUI Chat Interface

A beautiful terminal-based chat interface for interacting with vLLM inference servers. Features real-time streaming, system monitoring, and educational token probability visualization.

## Features

- **Interactive Chat**: Stream real-time responses from vLLM models
- **System Monitoring**: Live GPU/CPU/RAM stats
- **Token Probabilities**: Educational visualization of token selection
- **Response Analysis**: Generation time and finish reason tracking
- **Scrollable Interface**: Full navigation with keyboard shortcuts

## Screenshots

[Add screenshots here]

## Prerequisites

- Go 1.19+
- vLLM server running locally (default: http://localhost:8000)

## Installation

```bash
git clone https://github.com/yourusername/vllm-tui.git
cd vllm-tui
go mod tidy
go build -o vllm-tui
```

## Usage

1. Start your vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server --model microsoft/phi-1_5 --host 0.0.0.0 --port 8000
```

2. Run the TUI:
```bash
./vllm-tui
```

## Controls

- **Enter**: Send message
- **Tab**: Toggle between input and scrolling
- **Ctrl+C**: Exit
- **Ctrl+L**: Toggle token probabilities panel

## Architecture

Built with:
- [Bubbletea](https://github.com/charmbracelet/bubbletea) - Terminal UI framework
- [Bubbles](https://github.com/charmbracelet/bubbles) - UI components
- [Lipgloss](https://github.com/charmbracelet/lipgloss) - Styling
- [gopsutil](https://github.com/shirou/gopsutil) - System monitoring

## Configuration

The TUI connects to `http://localhost:8000` by default. Modify the server URL in the code for different endpoints.

## Educational Features

- **Token Probabilities**: See why the model chose specific tokens
- **Generation Stats**: Understand response timing and completion reasons
- **System Metrics**: Learn about hardware utilization during inference

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Related Projects

- [go-kubectl-tui](https://github.com/yourusername/kubectl-tui) - kubectl interface
- [local-inference-operator](https://github.com/yourusername/local-inference-operator) - K8s operator
- [operator-tui](https://github.com/yourusername/operator-tui) - Operator management interface