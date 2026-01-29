# PocketCoder

AI-powered coding assistant that works with any LLM — local or cloud.

## Features

- **Any LLM**: Works with Ollama, OpenAI, Claude, or any OpenAI-compatible API
- **Session Memory**: Continues where you left off
- **RepoMap**: Understands your project structure
- **Optimized for Local Models**: Works great with 7B-14B parameter models

## Quick Start

```bash
pip install pocketcoder
pocketcoder
```

On first run, a setup wizard will help you configure your LLM provider.

## Supported Providers

| Provider | Models | Setup |
|----------|--------|-------|
| **Ollama** | qwen2.5-coder, llama3, etc. | `ollama serve` |
| **OpenAI** | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| **Anthropic** | claude-sonnet, claude-haiku | `ANTHROPIC_API_KEY` |
| **Any OpenAI-compatible** | vLLM, LM Studio, etc. | Custom base URL |

## Commands

```
/help      - Show help
/model     - Change model
/files     - List added files
/add       - Add file to context
/drop      - Remove file from context
/undo      - Undo last change
/clear     - Clear conversation
/quit      - Exit
```

## Configuration

Config is stored in `~/.pocketcoder/config.yaml`:

```yaml
provider:
  type: ollama
  base_url: http://localhost:11434
  default_model: qwen2.5-coder:7b

thinking:
  mode: smart
  show_reasoning: true
```

## Development

```bash
git clone https://github.com/Chashchin-Dmitry/pocketcoder.git
cd pocketcoder
pip install -e .
pytest
```

## License

MIT
