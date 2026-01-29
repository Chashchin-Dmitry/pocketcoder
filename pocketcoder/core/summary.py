"""
Summary generation for SESSION_CONTEXT architecture.

v2.0.0: Builds SESSION_CONTEXT XML that is injected into every LLM request.
This enables "continue" to work without keyword detection.

v2.6.0: Added repo_map (code structure overview) to SESSION_CONTEXT.

Functions:
- build_session_context_xml(): builds XML for injection
- generate_full_summary(): generates complete summary on session end
- extract_decisions(): LLM extracts decisions from conversation
- extract_current_vector(): LLM extracts current state
- auto_grep_history(): grep raw.jsonl by keywords
- extract_keywords(): extract keywords from text for grep
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pocketcoder.core.project_context import ProjectContext
    from pocketcoder.core.episodes import EpisodeManager
    from pocketcoder.providers.base import BaseProvider


# =============================================================================
# Keyword Extraction
# =============================================================================

# Common words to ignore when extracting keywords
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "and", "but", "if", "or", "because", "until", "while", "this",
    "that", "these", "those", "what", "which", "who", "whom", "i", "me",
    "my", "myself", "we", "our", "ours", "you", "your", "yours", "he",
    "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "please", "thanks", "thank", "ok", "okay", "yes", "no",
    "file", "code", "make", "create", "show", "get", "put", "use"
}


def extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """
    Extract meaningful keywords from text for grep search.

    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return

    Returns:
        List of keywords
    """
    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())

    # Filter stop words and short words
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    # Count frequency
    freq = {}
    for w in keywords:
        freq[w] = freq.get(w, 0) + 1

    # Sort by frequency and return top N
    sorted_keywords = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)

    return sorted_keywords[:max_keywords]


# =============================================================================
# Auto-grep History
# =============================================================================

def auto_grep_history(
    project_dir: Path,
    keywords: list[str],
    max_matches: int = 5
) -> list[dict]:
    """
    Grep through raw.jsonl files for relevant history.

    Args:
        project_dir: .pocketcoder directory path
        keywords: Keywords to search for
        max_matches: Maximum number of matches to return

    Returns:
        List of matches with user/assistant content
    """
    if not keywords:
        return []

    sessions_dir = project_dir / "sessions"
    if not sessions_dir.exists():
        return []

    matches = []
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)

    # Search through all session raw.jsonl files
    for raw_file in sessions_dir.glob("*/raw.jsonl"):
        try:
            with open(raw_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                try:
                    entry = json.loads(line)
                    content = entry.get("content", "")

                    if pattern.search(content):
                        # Get context (previous and next message if available)
                        match_data = {
                            "timestamp": entry.get("timestamp", ""),
                            "role": entry.get("role", ""),
                            "content": content[:500],  # Limit length
                            "session": raw_file.parent.name
                        }

                        # Try to get the response if this is a user message
                        if entry.get("role") == "user" and i + 1 < len(lines):
                            try:
                                next_entry = json.loads(lines[i + 1])
                                if next_entry.get("role") == "assistant":
                                    match_data["response"] = next_entry.get("content", "")[:500]
                            except Exception:
                                pass

                        matches.append(match_data)

                except json.JSONDecodeError:
                    continue

        except Exception:
            continue

    # Sort by timestamp (most recent first) and limit
    matches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return matches[:max_matches]


def grep_history_to_xml(matches: list[dict], query: str) -> str:
    """
    Convert grep matches to XML for SESSION_CONTEXT.

    Args:
        matches: List of match dicts from auto_grep_history
        query: Original search query

    Returns:
        XML string for <relevant_history> block
    """
    if not matches:
        return '<relevant_history query="" matches="0"/>'

    lines = [f'<relevant_history query="{query}" matches="{len(matches)}">']

    for match in matches:
        timestamp = match.get("timestamp", "")
        lines.append(f'  <match timestamp="{timestamp}">')

        if match.get("role") == "user":
            lines.append(f'    <user>{_escape_xml(match.get("content", ""))}</user>')
            if match.get("response"):
                lines.append(f'    <assistant>{_escape_xml(match.get("response", ""))}</assistant>')
        else:
            lines.append(f'    <{match.get("role", "unknown")}>{_escape_xml(match.get("content", ""))}</{match.get("role", "unknown")}>')

        lines.append('  </match>')

    lines.append('</relevant_history>')
    return "\n".join(lines)


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# =============================================================================
# Decisions Extraction
# =============================================================================

EXTRACT_DECISIONS_PROMPT = """Analyze this conversation and extract:
1. Constraints: limitations set by user (e.g., "no Docker", "local only", "Python 3.10+")
2. Decisions: technical choices made (e.g., "using Flask", "SQLite for storage")
3. Q&A: questions asked and answered

Conversation:
{conversation}

Respond in this exact JSON format:
{{
  "constraints": ["constraint1", "constraint2"],
  "decisions": ["decision1", "decision2"],
  "qa": [{{"q": "question", "a": "answer"}}]
}}

JSON response:"""


def extract_decisions(
    conversation: list[dict],
    provider: Optional["BaseProvider"] = None
) -> dict:
    """
    Extract decisions, constraints, and Q&A from conversation using LLM.

    Args:
        conversation: List of message dicts
        provider: LLM provider

    Returns:
        Dict with constraints, decisions, qa lists
    """
    if not provider:
        return {"constraints": [], "decisions": [], "qa": []}

    # Format conversation for prompt
    conv_text = ""
    for msg in conversation[-20:]:  # Last 20 messages
        role = msg.get("role", "user")
        content = msg.get("content", "")[:500]
        conv_text += f"{role}: {content}\n"

    try:
        from pocketcoder.core.models import Message

        prompt = EXTRACT_DECISIONS_PROMPT.format(conversation=conv_text)
        messages = [Message("user", prompt)]

        response = provider.chat(messages, max_tokens=500)

        # Parse JSON response
        json_match = re.search(r'\{[^}]+\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

    except Exception:
        pass

    return {"constraints": [], "decisions": [], "qa": []}


def decisions_to_xml(decisions: dict) -> str:
    """Convert decisions dict to XML for SESSION_CONTEXT."""
    lines = ["<decisions>"]

    for constraint in decisions.get("constraints", []):
        lines.append(f'  <constraint>{_escape_xml(constraint)}</constraint>')

    for decision in decisions.get("decisions", []):
        lines.append(f'  <decision>{_escape_xml(decision)}</decision>')

    for qa in decisions.get("qa", []):
        q = _escape_xml(qa.get("q", ""))
        a = _escape_xml(qa.get("a", ""))
        lines.append(f'  <qa q="{q}" a="{a}"/>')

    lines.append("</decisions>")
    return "\n".join(lines)


# =============================================================================
# Current Vector Extraction
# =============================================================================

EXTRACT_CURRENT_PROMPT = """Based on this conversation, determine:
1. What is being done RIGHT NOW (current action)
2. What is the NEXT step after current action
3. Is anything blocking progress? If yes, what?

Conversation (recent):
{conversation}

Respond in this exact JSON format:
{{
  "doing": "current action description",
  "next": "next step description",
  "blocked_by": "nothing" or "what is blocking"
}}

JSON response:"""


def extract_current_vector(
    conversation: list[dict],
    provider: Optional["BaseProvider"] = None
) -> dict:
    """
    Extract current vector (doing, next, blocked_by) from conversation.

    Args:
        conversation: List of message dicts
        provider: LLM provider

    Returns:
        Dict with doing, next, blocked_by
    """
    if not provider:
        return {"doing": "", "next": "", "blocked_by": "nothing"}

    # Format recent conversation
    conv_text = ""
    for msg in conversation[-10:]:  # Last 10 messages
        role = msg.get("role", "user")
        content = msg.get("content", "")[:300]
        conv_text += f"{role}: {content}\n"

    try:
        from pocketcoder.core.models import Message

        prompt = EXTRACT_CURRENT_PROMPT.format(conversation=conv_text)
        messages = [Message("user", prompt)]

        response = provider.chat(messages, max_tokens=300)

        # Parse JSON response
        json_match = re.search(r'\{[^}]+\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

    except Exception:
        pass

    return {"doing": "", "next": "", "blocked_by": "nothing"}


def current_vector_to_xml(vector: dict) -> str:
    """Convert current vector dict to XML for SESSION_CONTEXT."""
    lines = ["<current>"]
    lines.append(f'  <doing>{_escape_xml(vector.get("doing", ""))}</doing>')
    lines.append(f'  <next>{_escape_xml(vector.get("next", ""))}</next>')
    lines.append(f'  <blocked_by>{_escape_xml(vector.get("blocked_by", "nothing"))}</blocked_by>')
    lines.append("</current>")
    return "\n".join(lines)


# =============================================================================
# TODO to XML
# =============================================================================

def todo_to_xml(todo_list: list) -> str:
    """
    Convert TODO list to XML for SESSION_CONTEXT.

    Args:
        todo_list: List of TodoItem or dicts

    Returns:
        XML string for <todo> block
    """
    if not todo_list:
        return "<todo/>"

    lines = ["<todo>"]

    for item in todo_list:
        # Handle both TodoItem and dict
        if hasattr(item, "task"):
            task = item.task
            status = item.status
        elif isinstance(item, dict):
            task = item.get("task", item.get("content", ""))
            status = item.get("status", "pending")
        else:
            continue

        # Map status to XML format
        if status == "completed":
            xml_status = "done"
        elif status == "in_progress":
            xml_status = "current"
        else:
            xml_status = "pending"

        lines.append(f'  <item status="{xml_status}">{_escape_xml(task)}</item>')

    lines.append("</todo>")
    return "\n".join(lines)


# =============================================================================
# Main Functions
# =============================================================================

def build_session_context_xml(
    project_context: "ProjectContext",
    current_todo: list,
    user_input: str = "",
    conversation: list[dict] = None,
    provider: Optional["BaseProvider"] = None,
    include_soft_data: bool = False,
    episode_manager: Optional["EpisodeManager"] = None
) -> str:
    """
    Build complete SESSION_CONTEXT XML for injection into LLM prompt.

    Args:
        project_context: ProjectContext instance
        current_todo: Current TODO list
        user_input: Current user input (for auto-grep keywords)
        conversation: Conversation history (for soft data extraction)
        provider: LLM provider (for soft data extraction)
        include_soft_data: Whether to include LLM-extracted decisions
        episode_manager: EpisodeManager for conversation history (v2.3.0)

    Returns:
        Complete SESSION_CONTEXT XML string

    v2.1.0 Changes:
        - <current> now ALWAYS comes from ProjectContext.current_to_xml()
        - No LLM call needed for current vector
        - Only decisions use include_soft_data flag

    v2.3.0 Changes:
        - Added <conversation_history> from EpisodeManager (Episodic Memory)

    v2.6.0 Changes:
        - Added <repo_map> for code structure overview (LLM sees project structure)
    """
    lines = ["<SESSION_CONTEXT>"]

    # 1. Task and Project (from ProjectContext)
    lines.append(project_context.to_xml())

    # 2. Repo Map (v2.6.0: Code structure overview)
    # LLM sees project structure without reading all files
    lines.append(project_context.get_repo_map())

    # 3. Conversation History (v2.3.0: Episodic Memory)
    # Shows previous user requests and their outcomes
    if episode_manager:
        lines.append(episode_manager.build_history_xml(max_episodes=20))
    else:
        lines.append("<conversation_history/>")

    # 4. Files (from FileTracker)
    lines.append(project_context.files.to_xml())

    # 5. Terminal (from TerminalHistory)
    lines.append(project_context.terminal.to_xml())

    # 6. Decisions (soft data - optional, expensive LLM call)
    if include_soft_data and conversation and provider:
        decisions = extract_decisions(conversation, provider)
        lines.append(decisions_to_xml(decisions))
    else:
        lines.append("<decisions/>")

    # 7. Knowledge Base
    lines.append(project_context.knowledge_base_to_xml())

    # 8. TODO
    lines.append(todo_to_xml(current_todo))

    # 9. Current Vector (v2.1.0: ALWAYS from tracker, no LLM call!)
    # This enables follow-up requests like "can you run it?" to work
    lines.append(project_context.current_to_xml())

    # 10. Relevant History (auto-grep)
    if user_input:
        keywords = extract_keywords(user_input)
        if keywords:
            matches = auto_grep_history(project_context.project_dir, keywords)
            query = " ".join(keywords)
            lines.append(grep_history_to_xml(matches, query))
        else:
            lines.append('<relevant_history query="" matches="0"/>')
    else:
        lines.append('<relevant_history query="" matches="0"/>')

    lines.append("</SESSION_CONTEXT>")

    return "\n".join(lines)


def generate_full_summary(
    project_context: "ProjectContext",
    current_todo: list,
    conversation: list[dict],
    provider: Optional["BaseProvider"] = None
) -> dict:
    """
    Generate complete summary for session end (save to summary.json).

    Args:
        project_context: ProjectContext instance
        current_todo: Current TODO list
        conversation: Full conversation history
        provider: LLM provider for soft data extraction

    Returns:
        Summary dict for JSON serialization
    """
    # Extract soft data if provider available
    decisions = {}
    current_vector = {}

    if provider:
        decisions = extract_decisions(conversation, provider)
        current_vector = extract_current_vector(conversation, provider)

    summary = {
        "session_id": project_context.get_current_session_id(),
        "timestamp": datetime.now().isoformat(),
        "task": project_context.task.__dict__ if project_context.task else None,
        "project": {
            "name": project_context.identity.name,
            "path": project_context.identity.path,
            "entry_point": project_context.identity.entry_point
        },
        "files": project_context.files.to_dict(),
        "terminal": project_context.terminal.to_dict(),
        "decisions": decisions,
        "knowledge_base": project_context.knowledge_base,
        "todo_final": [
            {"task": getattr(t, "task", t.get("task", "")),
             "status": getattr(t, "status", t.get("status", "pending"))}
            for t in current_todo
        ] if current_todo else [],
        "current_vector": current_vector
    }

    # Save to session directory
    session_dir = project_context.project_dir / "sessions" / project_context.get_current_session_id()
    session_dir.mkdir(parents=True, exist_ok=True)
    summary_file = session_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary
