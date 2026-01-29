"""
Configuration module.

Handles loading, saving, and managing PocketCoder configuration.
"""

from pocketcoder.config.settings import (
    load_config,
    save_config,
    DEFAULT_CONFIG,
)

__all__ = ["load_config", "save_config", "DEFAULT_CONFIG"]
