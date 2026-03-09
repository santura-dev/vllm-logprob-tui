# vllm-logprob-tui

![Go](https://img.shields.io/badge/go-%2300ADD8.svg?style=flat&logo=go&logoColor=white) ![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg) ![vLLM](https://img.shields.io/badge/vLLM-compatible-green)

Terminal UI for vLLM logprobs, token statistics, and inference metrics in real-time.

## The problem

vLLM exposes logprob data through its API. Reading raw JSON from `curl` is not useful when you are trying to understand why a model generated bad output. You need token probabilities, alternatives the model considered, batch statistics, and per-request breakdowns, and you need them updating live.

## The idea

Connect to a vLLM server and display all of that in a navigable terminal interface. Useful for diagnosing model confidence (why does the model say "port 5432" with 0.3 probability?), understanding batch behavior under load, and monitoring inference performance over time.

## Features

- **real-time token probabilities**: top-k tokens and probabilities per generation step
- **batch statistics**: throughput, latency, queue depth, active requests
- **per-request breakdown**: token counts, time to first token, generation speed
- **alternative tokens**: what else the model considered, with probabilities. Useful for spotting uncertainty.

## Run

```bash
go run ./cmd/vllm-tui --server http://localhost:8000
```

or build:

```bash
go build -o vllm-tui ./cmd/vllm-tui
./vllm-tui --server http://localhost:8000
```

## Config

| Flag | Env | Default | Description |
|---|---|---|---|
| `--server` | `VLLM_SERVER` | `http://localhost:8000` | vLLM server base URL |
| `--model` | `VLLM_MODEL` | `facebook/opt-125m` | model name sent to the API |
| `--max-tokens` | — | `100` | max tokens to generate |
| `--top-k` | — | `5` | top-k logprobs to request per token |
| `--metrics-interval` | — | `1s` | system/vLLM metrics refresh interval |

## Keys

- `enter` — send query
- `tab` — toggle between prompt and output scrolling
- `ctrl+c` / `esc` — quit

## Related

- [inference-operator-tui](https://github.com/santura-dev/inference-operator-tui) - TUI for managing the K8s operator
- [kubectl-tui](https://github.com/santura-dev/kubectl-tui) - general Kubernetes TUI

## License

MIT
