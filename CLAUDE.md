# CLAUDE.md — PocketCoder

## Quick Start

```bash
cd pocketcoder && pip install -e . && pocketcoder .
```

## Project Structure

```
pocketcoder/
├── __init__.py          # Version
├── __main__.py          # Entry point
├── core/
│   ├── coder.py         # Main orchestrator, agent loop
│   ├── parser.py        # SEARCH/REPLACE + XML tools
│   ├── applier.py       # Apply file changes
│   ├── project_context.py  # Session state, TODO, file tracking
│   ├── content_preview.py  # Universal preview (gearbox)
│   ├── episodes.py      # Episodic memory
│   ├── summary.py       # SESSION_CONTEXT builder
│   └── memory/          # Memory management
├── providers/
│   ├── ollama.py        # Ollama API
│   ├── openai.py        # OpenAI API
│   ├── anthropic.py     # Claude API
│   └── openai_compat.py # Any OpenAI-compatible
├── tools/
│   ├── files.py         # read_file, write_file, list_files, etc.
│   ├── shell.py         # execute_command
│   ├── agent.py         # add_todo, mark_done, ask_question
│   └── history.py       # checkpoint_progress, search_history
├── ui/
│   ├── cli.py           # Terminal interface
│   └── web.py           # Gradio Web UI
└── config/
    └── settings.py      # Config management
```

## Architecture

```
User Input
    │
    ▼
ProjectContext.set_task() — saves goal
    │
    ▼
LLM receives SESSION_CONTEXT (task, files, todo, history)
    │
    ▼
LLM generates: [tool1, tool2, tool3]
    │
    ▼
_apply_reconnaissance_rule()
    ├─► READ + WRITE → only READ, defer WRITE
    └─► WRITE x N → only first WRITE, defer rest
    │
    ▼
Execute tool → ContentPreview → LLM sees result
    │
    ▼
FileTracker.track_write/read() — metadata
    │
    ▼
TodoStateMachine — add_todo/mark_done
    │
    ▼
attempt_completion → DONE
```

## Key Components

| Component | File | Description |
|-----------|------|-------------|
| **ProjectContext** | `project_context.py` | Session state |
| **ContentPreview** | `content_preview.py` | Universal preview (gearbox) |
| **TodoStateMachine** | `project_context.py` | add_todo, mark_done, remove_todo |
| **EpisodeManager** | `episodes.py` | Episodic memory |
| **Parser** | `parser.py` | KNOWN_TOOLS = set(TOOLS.keys()) |

## Tools

### File Operations
- `read_file(path, start_line?, end_line?)` — Read file or line range
- `write_file(path, content)` — Create/overwrite file
- `list_files(path, recursive)` — Show structure
- `find_file(filename)` — Find file by name
- `search_files(pattern, path)` — Search content

### Agent Control
- `add_todo(task)` — Add task to plan
- `mark_done(task)` — Mark as completed
- `remove_todo(task)` — Remove task
- `ask_question(question, options)` — Ask user
- `attempt_completion(result)` — Finish task

### Memory
- `remember_fact(key, value)` — Store fact
- `recall_fact(key)` — Retrieve fact
- `checkpoint_progress(done, remaining)` — Save progress

## CLI Commands

```
/help      — Show help
/model     — Change model
/files     — List added files
/add       — Add file to context
/drop      — Remove file
/undo      — Undo last change
/clear     — Clear conversation
/quit      — Exit
```

## Configuration

```yaml
# ~/.pocketcoder/config.yaml

provider:
  type: ollama          # or: openai, anthropic
  default_model: qwen2.5-coder:7b

thinking:
  mode: smart
  show_reasoning: true
```

## Testing

```bash
pytest tests/
```

## Philosophy: Gearbox

```
Add tool in tools/__init__.py
    ↓
Parser automatically sees it (KNOWN_TOOLS = set(TOOLS.keys()))
    ↓
LLM uses tool
    ↓
ContentPreview formats result
    ↓
ProjectContext saves metadata
```

One mechanism — everything works automatically.
