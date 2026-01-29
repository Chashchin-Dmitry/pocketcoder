"""
Entry point for running PocketCoder as a module: python -m pocketcoder
"""

from __future__ import annotations

import sys

import click

from pocketcoder import __version__


@click.command()
@click.argument("files", nargs=-1, type=click.Path())
@click.option("--provider", "-p", help="LLM provider name (ollama, vllm, lm-studio)")
@click.option("--model", "-m", help="Model name to use")
@click.option("--web", is_flag=True, help="Start web UI instead of CLI")
@click.option("--port", default=7860, help="Port for web UI (default: 7860)")
@click.option("--config", "-c", type=click.Path(), help="Path to config file")
@click.option("--setup", is_flag=True, help="Run initial setup wizard")
@click.option("--debug", "-d", is_flag=True, help="Show debug output (iterations, parsing)")
@click.version_option(version=__version__, prog_name="pocketcoder")
def main(
    files: tuple[str, ...],
    provider: str | None,
    model: str | None,
    web: bool,
    port: int,
    config: str | None,
    setup: bool,
    debug: bool,
) -> int:
    """
    PocketCoder - AI-powered coding assistant for local LLMs.

    Add files to the chat and describe what changes you want to make.

    Examples:

        pocketcoder main.py              # Start with main.py in chat

        pocketcoder src/*.py --model qwen2.5:7b  # Use specific model

        pocketcoder --web                # Start web UI
    """
    from pocketcoder.config.settings import load_config, run_wizard, DEFAULT_CONFIG
    from pocketcoder.core.coder import Coder
    from pathlib import Path

    # Load configuration
    if setup:
        cfg = run_wizard()
        if cfg is None:
            click.echo("Setup cancelled.")
            return 1
    else:
        cfg = load_config(config) if config else load_config()
        if cfg is None:
            click.echo("Welcome to PocketCoder! Let's set up.")
            cfg = run_wizard()
            if cfg is None:
                click.echo("Setup cancelled.")
                return 1

    # Override from CLI args
    if provider:
        cfg["provider"]["name"] = provider
    if model:
        cfg["provider"]["model"] = model

    # Create Coder instance
    try:
        coder = Coder(cfg)
    except ConnectionError as e:
        click.echo(f"Cannot connect to LLM: {e}", err=True)
        return 1

    # Add files from CLI
    for f in files:
        path = Path(f)
        if path.exists():
            coder.add_file(path)
        else:
            click.echo(f"Warning: File not found: {f}", err=True)

    # Start interface
    if web:
        from pocketcoder.ui.web import start_web

        start_web(coder, port=port)
    else:
        return coder.run_interactive(debug=debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
