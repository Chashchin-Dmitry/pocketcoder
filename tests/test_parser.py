"""
Tests for SEARCH/REPLACE parser.
"""

import pytest

from pocketcoder.core.parser import (
    parse_edits,
    parse_edits_with_recovery,
    parse_thinking,
    parse_commands,
    parse_options,
    is_question_response,
    parse_response,
)


class TestParseEdits:
    """Tests for main SEARCH/REPLACE parsing."""

    def test_single_edit(self):
        response = """Here's the fix:

main.py
<<<<<<< SEARCH
def hello():
    pass
=======
def hello():
    print("Hello!")
>>>>>>> REPLACE
"""
        edits = parse_edits(response)
        assert len(edits) == 1
        assert edits[0].filename == "main.py"
        assert edits[0].search == "def hello():\n    pass"
        assert edits[0].replace == 'def hello():\n    print("Hello!")'

    def test_multiple_edits(self):
        response = """Making two changes:

file1.py
<<<<<<< SEARCH
old1
=======
new1
>>>>>>> REPLACE

file2.py
<<<<<<< SEARCH
old2
=======
new2
>>>>>>> REPLACE
"""
        edits = parse_edits(response)
        assert len(edits) == 2
        assert edits[0].filename == "file1.py"
        assert edits[1].filename == "file2.py"

    def test_new_file(self):
        response = """Creating new file:

new_file.py
<<<<<<< SEARCH
=======
def new_function():
    pass
>>>>>>> REPLACE
"""
        edits = parse_edits(response)
        assert len(edits) == 1
        assert edits[0].is_new_file is True
        assert edits[0].search == ""
        assert "def new_function" in edits[0].replace

    def test_no_edits(self):
        response = "Just some text without any edits."
        edits = parse_edits(response)
        assert len(edits) == 0

    def test_malformed_no_crash(self):
        """Parser should not crash on malformed input."""
        bad_inputs = [
            "no markers at all",
            "<<<<<<< SEARCH\nno end",
            "=======\n>>>>>>> REPLACE",
            "file.py\n<<<<<<< SEARCH\nmissing replace",
        ]
        for bad in bad_inputs:
            edits = parse_edits(bad)
            assert isinstance(edits, list)


class TestParseEditsWithRecovery:
    """Tests for recovery parsing with alternative markers."""

    def test_standard_markers(self):
        response = """
test.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
"""
        edits, warnings = parse_edits_with_recovery(response)
        assert len(edits) == 1
        assert len(warnings) == 0

    def test_alternative_markers(self):
        response = """
test.py
<<<<<<< ORIGINAL
old
=======
new
>>>>>>> MODIFIED
"""
        edits, warnings = parse_edits_with_recovery(response)
        assert len(edits) == 1
        assert len(warnings) == 1
        assert "alternative" in warnings[0].lower()

    def test_broken_markers_warning(self):
        response = "test.py\n<<<<<<< SEARCH\nold\n======="
        edits, warnings = parse_edits_with_recovery(response)
        assert len(edits) == 0
        assert any("broken" in w.lower() for w in warnings)


class TestParseThinking:
    """Tests for thinking/reasoning extraction."""

    def test_thinking_section(self):
        response = """## My thoughts:
I need to fix the bug in line 10.

main.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
"""
        thinking = parse_thinking(response)
        assert "fix the bug" in thinking

    def test_no_thinking(self):
        response = """main.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
"""
        thinking = parse_thinking(response)
        assert thinking == ""


class TestParseCommands:
    """Tests for shell command extraction."""

    def test_bash_block(self):
        response = """Run this:

```bash
pytest tests/
```
"""
        commands = parse_commands(response)
        assert len(commands) == 1
        assert "pytest" in commands[0]

    def test_shell_block(self):
        response = """
<<<<<<< SHELL
npm install
>>>>>>> SHELL
"""
        commands = parse_commands(response)
        assert len(commands) == 1
        assert "npm install" in commands[0]

    def test_multiple_commands(self):
        response = """
```bash
pip install pytest
```

```shell
pytest -v
```
"""
        commands = parse_commands(response)
        assert len(commands) == 2


class TestParseOptions:
    """Tests for option/choice extraction."""

    def test_options(self):
        response = """Which approach?

[a] Use database
[b] Use file storage
[c] Use memory cache
"""
        options = parse_options(response)
        assert len(options) == 3
        assert "database" in options["a"]
        assert "file" in options["b"]
        assert "memory" in options["c"]

    def test_no_options(self):
        response = "Just regular text"
        options = parse_options(response)
        assert len(options) == 0


class TestIsQuestionResponse:
    """Tests for question detection."""

    def test_question(self):
        response = "Which method should I use? There are several options."
        assert is_question_response(response) is True

    def test_code_response(self):
        response = """
main.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
"""
        assert is_question_response(response) is False

    def test_long_text(self):
        response = "?" * 3000  # Very long text
        assert is_question_response(response) is False


class TestParseResponse:
    """Tests for full response parsing."""

    def test_full_parse(self):
        response = """## Thinking:
I'll add error handling.

main.py
<<<<<<< SEARCH
def func():
    pass
=======
def func():
    try:
        pass
    except Exception:
        pass
>>>>>>> REPLACE

Then run:

```bash
pytest
```
"""
        parsed = parse_response(response)

        assert parsed.raw == response
        assert "error handling" in parsed.thinking
        assert len(parsed.edits) == 1
        assert len(parsed.commands) == 1
        assert parsed.is_question is False

    def test_question_parse(self):
        response = """I have a question.

Which database should we use?

[a] PostgreSQL
[b] MySQL
[c] SQLite
"""
        parsed = parse_response(response)

        assert parsed.is_question is True
        assert len(parsed.options) == 3
        assert len(parsed.edits) == 0
