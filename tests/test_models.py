"""
Tests for core data models.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile

from pocketcoder.core.models import (
    Message,
    Edit,
    FileContext,
    ChatResponse,
    ParsedResponse,
    Change,
)


class TestMessage:
    def test_create_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self):
        msg = Message(role="assistant", content="Hi there")
        d = msg.to_dict()
        assert d == {"role": "assistant", "content": "Hi there"}


class TestEdit:
    def test_create_edit(self):
        edit = Edit(
            filename="test.py",
            search="old code",
            replace="new code",
        )
        assert edit.filename == "test.py"
        assert edit.search == "old code"
        assert edit.replace == "new code"

    def test_is_new_file(self):
        # New file has empty search
        new_edit = Edit(filename="new.py", search="", replace="content")
        assert new_edit.is_new_file is True

        # Existing file edit has non-empty search
        existing_edit = Edit(filename="old.py", search="old", replace="new")
        assert existing_edit.is_new_file is False

    def test_is_delete_file(self):
        # Delete has empty replace and non-empty search
        delete_edit = Edit(filename="old.py", search="all content", replace="")
        assert delete_edit.is_delete_file is True

        # Normal edit
        normal_edit = Edit(filename="old.py", search="old", replace="new")
        assert normal_edit.is_delete_file is False


class TestFileContext:
    def test_create_from_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            path = Path(f.name)

        try:
            ctx = FileContext.from_path(path)
            assert ctx.lines == 3
            assert "line1" in ctx.content
            assert ctx.mtime > 0
            assert ctx.is_partial is False
        finally:
            path.unlink()

    def test_partial_file(self):
        ctx = FileContext(
            content="partial content",
            lines=10,
            is_partial=True,
            partial_range=(1, 100),
        )
        assert ctx.is_partial is True
        assert ctx.partial_range == (1, 100)


class TestChatResponse:
    def test_create_response(self):
        resp = ChatResponse(
            content="Hello!",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert resp.content == "Hello!"
        assert resp.finish_reason == "stop"
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5

    def test_empty_usage(self):
        resp = ChatResponse(content="Hi", finish_reason="stop")
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0


class TestParsedResponse:
    def test_create_parsed_response(self):
        edit = Edit(filename="test.py", search="old", replace="new")
        parsed = ParsedResponse(
            raw="some response",
            edits=[edit],
            thinking="I'm thinking...",
        )
        assert len(parsed.edits) == 1
        assert parsed.thinking == "I'm thinking..."
        assert parsed.is_question is False

    def test_question_response(self):
        parsed = ParsedResponse(
            raw="What do you want?",
            is_question=True,
            question_text="What do you want?",
            options={"a": "Option A", "b": "Option B"},
        )
        assert parsed.is_question is True
        assert len(parsed.options) == 2


class TestChange:
    def test_create_change(self):
        change = Change(
            file=Path("/test/file.py"),
            old_content="old",
            new_content="new",
        )
        assert change.file == Path("/test/file.py")
        assert change.old_content == "old"
        assert change.new_content == "new"
        assert isinstance(change.timestamp, datetime)
