"""
Tests for LLM providers.
"""

import pytest
import responses
import json

from pocketcoder.core.models import Message
from pocketcoder.providers.ollama import OllamaProvider
from pocketcoder.providers.openai_compat import OpenAICompatProvider


class TestOllamaProvider:
    """Tests for Ollama provider."""

    @responses.activate
    def test_chat_success(self):
        responses.add(
            responses.POST,
            "http://localhost:11434/api/chat",
            json={
                "message": {"content": "Hello!"},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

        provider = OllamaProvider()
        response = provider.chat(
            [Message("user", "Hi")],
            model="test-model",
        )

        assert response.content == "Hello!"
        assert response.finish_reason == "stop"

    @responses.activate
    def test_check_connection_success(self):
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={"models": []},
        )

        provider = OllamaProvider()
        ok, msg = provider.check_connection()

        assert ok is True
        assert msg == "OK"

    @responses.activate
    def test_check_connection_failure(self):
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=Exception("Connection refused"),
        )

        provider = OllamaProvider()
        ok, msg = provider.check_connection()

        assert ok is False

    @responses.activate
    def test_list_models(self):
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={
                "models": [
                    {"name": "llama2"},
                    {"name": "codellama"},
                ]
            },
        )

        provider = OllamaProvider()
        models = provider.list_models()

        assert "llama2" in models
        assert "codellama" in models


class TestOpenAICompatProvider:
    """Tests for OpenAI-compatible provider."""

    @responses.activate
    def test_chat_success(self):
        responses.add(
            responses.POST,
            "http://localhost:8000/v1/chat/completions",
            json={
                "choices": [
                    {
                        "message": {"content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

        provider = OpenAICompatProvider(base_url="http://localhost:8000/v1")
        response = provider.chat(
            [Message("user", "Hi")],
            model="test-model",
        )

        assert response.content == "Hello!"
        assert response.finish_reason == "stop"
        assert response.prompt_tokens == 10

    @responses.activate
    def test_chat_with_api_key(self):
        def check_auth(request):
            assert "Authorization" in request.headers
            assert "Bearer test-key" in request.headers["Authorization"]
            return (
                200,
                {},
                json.dumps({
                    "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]
                }),
            )

        responses.add_callback(
            responses.POST,
            "http://localhost:8000/v1/chat/completions",
            callback=check_auth,
            content_type="application/json",
        )

        provider = OpenAICompatProvider(
            base_url="http://localhost:8000/v1",
            api_key="test-key",
        )
        response = provider.chat([Message("user", "Hi")], model="test")

        assert response.content == "OK"

    @responses.activate
    def test_check_connection_success(self):
        responses.add(
            responses.GET,
            "http://localhost:8000/v1/models",
            json={"data": []},
        )

        provider = OpenAICompatProvider(base_url="http://localhost:8000/v1")
        ok, msg = provider.check_connection()

        assert ok is True

    @responses.activate
    def test_check_connection_404_ok(self):
        """Some servers don't have /models but still work."""
        responses.add(
            responses.GET,
            "http://localhost:8000/v1/models",
            status=404,
        )

        provider = OpenAICompatProvider(base_url="http://localhost:8000/v1")
        ok, msg = provider.check_connection()

        assert ok is True  # 404 on /models is acceptable

    @responses.activate
    def test_list_models(self):
        responses.add(
            responses.GET,
            "http://localhost:8000/v1/models",
            json={
                "data": [
                    {"id": "gpt-4"},
                    {"id": "gpt-3.5-turbo"},
                ]
            },
        )

        provider = OpenAICompatProvider(base_url="http://localhost:8000/v1")
        models = provider.list_models()

        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models
