"""
PocketCoder - AI-powered coding assistant optimized for local LLMs.

A lightweight, extensible coding assistant that works with any local LLM
(Ollama, vLLM, LM Studio, llama.cpp, etc.) through OpenAI-compatible APIs.
"""

__version__ = "1.0.3"
__author__ = "PocketCoder Team"

from pocketcoder.core.models import Edit, Message, FileContext
from pocketcoder.core.coder import Coder

__all__ = [
    "__version__",
    "Coder",
    "Edit",
    "Message",
    "FileContext",
]
