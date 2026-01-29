"""
Configuration management for PocketCoder.

Handles loading, saving, and default configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Default configuration
DEFAULT_CONFIG: dict[str, Any] = {
    # Active profile name
    "active_profile": "default",

    # Saved profiles (provider + model together)
    "profiles": {
        "default": {
            "type": "ollama",
            "base_url": "http://localhost:11434",
            "model": "qwen2.5-coder:7b",
        },
    },

    # Recent profiles (for quick switching)
    "recent_profiles": [],

    # Legacy provider settings (for backwards compat)
    "provider": {
        "name": "ollama",
        "type": "ollama",
        "base_url": "http://localhost:11434",
        "default_model": "qwen2.5-coder:7b",
    },
    # Thinking mode
    "thinking": {
        "mode": "smart",  # direct | reasoning | clarify | smart
        "show_reasoning": True,
    },
    # Apply settings
    "apply": {
        "mode": "interactive",  # interactive | auto | strict
        "show_diff": True,
    },
    # Shell settings
    "shell": {
        "enabled": True,
        "confirm": True,
        "timeout": 300,
    },
    # RepoMap settings
    "repomap": {
        "enabled": True,
        "backend": "simple",  # simple | treesitter
        "max_tokens": 2000,
    },
    # UI settings
    "ui": {
        "default": "cli",  # cli | web
        "web_port": 7860,
    },
    # Limits
    "max_file_lines": 500,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "max_context_tokens": 8000,
    "history_limit": 10,
}


def get_config_paths() -> list[Path]:
    """Get list of config file paths in priority order."""
    return [
        Path.cwd() / "pocketcoder.yaml",
        Path.cwd() / ".pocketcoder" / "config.yaml",
        Path.home() / ".pocketcoder" / "config.yaml",
    ]


def load_config(path: str | Path | None = None) -> dict[str, Any] | None:
    """
    Load configuration from file.

    Priority:
    1. Explicit path if provided
    2. ./pocketcoder.yaml (project-local)
    3. ./.pocketcoder/config.yaml (project-local)
    4. ~/.pocketcoder/config.yaml (global)
    5. Returns None if not found (triggers setup wizard)
    """
    if path:
        config_path = Path(path)
        if config_path.exists():
            return _load_yaml(config_path)
        return None

    for config_path in get_config_paths():
        if config_path.exists():
            return _load_yaml(config_path)

    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse YAML config file."""
    try:
        content = path.read_text()
        config = yaml.safe_load(content) or {}

        # Merge with defaults
        return _deep_merge(DEFAULT_CONFIG.copy(), config)
    except yaml.YAMLError as e:
        print(f"Warning: Invalid config {path}: {e}")
        print("Using defaults. Run 'pocketcoder --setup' to reconfigure.")
        return DEFAULT_CONFIG.copy()


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: dict[str, Any], path: Path | None = None) -> Path:
    """
    Save configuration to file.

    If no path provided, saves to ~/.pocketcoder/config.yaml
    """
    if path is None:
        path = Path.home() / ".pocketcoder" / "config.yaml"

    # Create directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return path


def get_profile(config: dict[str, Any], name: str | None = None) -> dict[str, Any] | None:
    """
    Get a profile by name or the active profile.

    Args:
        config: Config dictionary
        name: Profile name (None for active)

    Returns:
        Profile dict or None if not found
    """
    profiles = config.get("profiles", {})

    if name is None:
        name = config.get("active_profile", "default")

    return profiles.get(name)


def list_profiles(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """
    List all saved profiles.

    Returns:
        List of (name, profile_dict) tuples
    """
    profiles = config.get("profiles", {})
    recent = config.get("recent_profiles", [])

    # Sort: recent first, then alphabetically
    def sort_key(item):
        name = item[0]
        if name in recent:
            return (0, recent.index(name))
        return (1, name)

    return sorted(profiles.items(), key=sort_key)


def save_profile(config: dict[str, Any], name: str, profile: dict[str, Any]) -> None:
    """
    Save a profile to config.

    Args:
        config: Config dictionary (modified in place)
        name: Profile name
        profile: Profile settings
    """
    if "profiles" not in config:
        config["profiles"] = {}

    config["profiles"][name] = profile


def add_to_recent(config: dict[str, Any], name: str, max_recent: int = 5) -> None:
    """
    Add profile to recent list.

    Args:
        config: Config dictionary (modified in place)
        name: Profile name
        max_recent: Max items in recent list
    """
    recent = config.get("recent_profiles", [])

    # Remove if already in list
    if name in recent:
        recent.remove(name)

    # Add to front
    recent.insert(0, name)

    # Trim
    config["recent_profiles"] = recent[:max_recent]


def set_active_profile(config: dict[str, Any], name: str) -> bool:
    """
    Set the active profile.

    Args:
        config: Config dictionary (modified in place)
        name: Profile name

    Returns:
        True if profile exists and was set
    """
    if name in config.get("profiles", {}):
        config["active_profile"] = name
        add_to_recent(config, name)
        return True
    return False


def run_wizard() -> dict[str, Any] | None:
    """
    Run interactive setup wizard.

    Returns configuration dict or None if cancelled.
    """
    print("\n" + "=" * 50)
    print("  PocketCoder Setup Wizard")
    print("=" * 50 + "\n")

    config = DEFAULT_CONFIG.copy()

    # Step 1: Choose provider
    print("Which LLM provider do you use?\n")
    print("  [1] Ollama (local, free)")
    print("  [2] OpenAI (GPT-4o, GPT-4o-mini)")
    print("  [3] Anthropic (Claude)")
    print("  [4] vLLM (specify URL)")
    print("  [5] LM Studio (localhost:1234)")
    print("  [6] Other OpenAI-compatible")
    print("  [q] Quit\n")

    choice = input("Your choice [1-6, q]: ").strip().lower()

    if choice == "q":
        return None

    if choice == "1":
        config["provider"] = {
            "name": "ollama",
            "type": "ollama",
            "base_url": "http://localhost:11434",
            "default_model": "",
        }
    elif choice == "2":
        # OpenAI
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("\n💡 OpenAI API key not found in environment.")
            print("   Get one at: https://platform.openai.com/api-keys\n")
            api_key = input("Enter API key (sk-...): ").strip()
            if api_key:
                print(f"\n💡 To save permanently, add to ~/.zshrc or ~/.bashrc:")
                print(f'   export OPENAI_API_KEY="{api_key}"')
                os.environ["OPENAI_API_KEY"] = api_key
        config["provider"] = {
            "name": "openai",
            "type": "openai",
            "default_model": "gpt-4o-mini",
            "api_key": api_key,
        }
    elif choice == "3":
        # Anthropic
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("\n💡 Anthropic API key not found in environment.")
            print("   Get one at: https://console.anthropic.com/\n")
            api_key = input("Enter API key (sk-ant-...): ").strip()
            if api_key:
                print(f"\n💡 To save permanently, add to ~/.zshrc or ~/.bashrc:")
                print(f'   export ANTHROPIC_API_KEY="{api_key}"')
                os.environ["ANTHROPIC_API_KEY"] = api_key
        config["provider"] = {
            "name": "anthropic",
            "type": "anthropic",
            "default_model": "claude-sonnet-4-20250514",
            "api_key": api_key,
        }
    elif choice == "4":
        url = input("Enter vLLM URL [http://localhost:8000/v1]: ").strip()
        config["provider"] = {
            "name": "vllm",
            "type": "openai_compat",
            "base_url": url or "http://localhost:8000/v1",
            "default_model": "",
        }
    elif choice == "5":
        config["provider"] = {
            "name": "lm-studio",
            "type": "openai_compat",
            "base_url": "http://localhost:1234/v1",
            "default_model": "",
        }
    elif choice == "6":
        url = input("Enter API URL: ").strip()
        if not url:
            print("URL is required")
            return None
        config["provider"] = {
            "name": "custom",
            "type": "openai_compat",
            "base_url": url,
            "default_model": "",
        }
    else:
        config["provider"] = {
            "name": "ollama",
            "type": "ollama",
            "base_url": "http://localhost:11434",
            "default_model": "",
        }

    # Step 2: Test connection and list models
    print("\nTesting connection...")

    from pocketcoder.providers import create_provider

    try:
        provider = create_provider(config["provider"])
        ok, msg = provider.check_connection()

        if not ok:
            print(f"\n⚠ Cannot connect to LLM: {msg}")

            # Show installation help based on provider
            provider_type = config["provider"].get("type", "ollama")
            if provider_type == "ollama":
                print("\n💡 Ollama not running? Here's how to set it up:")
                print("   Mac:     Download from https://ollama.com/download")
                print("            or: brew install ollama")
                print("   Linux:   curl -fsSL https://ollama.com/install.sh | sh")
                print("   Windows: Download from https://ollama.com/download")
                print("\n   Then run:")
                print("     ollama serve              # start server")
                print("     ollama pull qwen2.5-coder:7b  # download model")
            else:
                print(f"\n💡 Make sure your LLM server is running at:")
                print(f"   {config['provider'].get('base_url', 'unknown')}")

            cont = input("\nContinue anyway? [y/n]: ").strip().lower()
            if cont != "y":
                return None
        else:
            print("Connected!")

            # List models
            try:
                models = provider.list_models()
                if models:
                    print("\nAvailable models:")
                    for i, m in enumerate(models[:10], 1):
                        print(f"  [{i}] {m}")
                    if len(models) > 10:
                        print(f"  ... and {len(models) - 10} more")

                    choice = input("\nSelect model [1] or enter name: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(models):
                        config["provider"]["default_model"] = models[int(choice) - 1]
                    elif choice:
                        config["provider"]["default_model"] = choice
                    else:
                        config["provider"]["default_model"] = models[0]
            except Exception:
                model = input("\nEnter model name: ").strip()
                config["provider"]["default_model"] = model

    except Exception as e:
        print(f"\n⚠ Connection error: {e}")

        # Show installation help
        provider_type = config["provider"].get("type", "ollama")
        if provider_type == "ollama":
            print("\n💡 Ollama not installed? Here's how:")
            print("   Mac:     Download from https://ollama.com/download")
            print("            or: brew install ollama")
            print("   Linux:   curl -fsSL https://ollama.com/install.sh | sh")
            print("   Windows: Download from https://ollama.com/download")
            print("\n   Then run:")
            print("     ollama serve              # start server")
            print("     ollama pull qwen2.5-coder:7b  # download model")

        cont = input("\nContinue anyway? [y/n]: ").strip().lower()
        if cont != "y":
            return None

    # Step 3: Save config
    print("\nWhere to save configuration?")
    print("  [1] ~/.pocketcoder/config.yaml (global)")
    print("  [2] ./pocketcoder.yaml (project-local)")

    choice = input("Your choice [1]: ").strip() or "1"

    if choice == "2":
        path = Path.cwd() / "pocketcoder.yaml"
    else:
        path = Path.home() / ".pocketcoder" / "config.yaml"

    save_config(config, path)
    print(f"\nConfiguration saved to {path}")
    print("\nYou're all set! Run 'pocketcoder' to start.\n")

    return config
