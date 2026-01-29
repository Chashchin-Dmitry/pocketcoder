"""
Session persistence for PocketCoder.

Saves and restores chat sessions to/from SQLite database.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from pocketcoder.core.models import Message


# Default database path (legacy)
DEFAULT_DB_PATH = Path.home() / ".pocketcoder" / "sessions.db"


def get_session_db_path(per_project: bool = True) -> Path:
    """
    Get session database path.

    v2.0.0: Changed from global ~/.pocketcoder/sessions.db
            to per-project .pocketcoder/sessions.db

    Args:
        per_project: If True, use .pocketcoder/ in cwd (v2.0.0 default)

    Returns:
        Path to sessions.db
    """
    if per_project:
        db_path = Path.cwd() / ".pocketcoder" / "sessions.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path
    else:
        return DEFAULT_DB_PATH


class SessionManager:
    """
    Manages session persistence in SQLite.

    v2.0.0: Changed from global to per-project storage.

    Sessions contain:
    - Chat history
    - Files in context
    - Active profile
    - Metadata (created, updated, title)
    """

    def __init__(self, db_path: Path | None = None, per_project: bool = True):
        """
        Initialize session manager.

        Args:
            db_path: Path to SQLite database
            per_project: If True, use .pocketcoder/ in cwd (v2.0.0 default)
        """
        self.db_path = db_path or get_session_db_path(per_project)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.current_session_id: str | None = None

    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    profile TEXT,
                    working_dir TEXT,
                    created_at REAL,
                    updated_at REAL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    path TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id)
            """)
            conn.commit()

    def create_session(
        self,
        title: str | None = None,
        profile: str | None = None,
        working_dir: str | None = None,
    ) -> str:
        """
        Create a new session.

        Args:
            title: Session title (auto-generated if not provided)
            profile: Active profile name
            working_dir: Working directory path

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())[:8]
        now = time.time()

        if not title:
            title = f"Session {session_id}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, title, profile, working_dir, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, title, profile, working_dir, now, now, "{}"),
            )
            conn.commit()

        self.current_session_id = session_id
        return session_id

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Save a message to the session.

        Args:
            session_id: Session ID
            role: Message role (user/assistant/system)
            content: Message content
        """
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()

    def save_files(self, session_id: str, file_paths: list[str]) -> None:
        """
        Save file paths for the session.

        Args:
            session_id: Session ID
            file_paths: List of file paths
        """
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing files
            conn.execute(
                "DELETE FROM session_files WHERE session_id = ?",
                (session_id,),
            )
            # Add new files
            for path in file_paths:
                conn.execute(
                    "INSERT INTO session_files (session_id, path) VALUES (?, ?)",
                    (session_id, path),
                )
            conn.commit()

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Load a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session data dict or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get session
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()

            if not row:
                return None

            # Get messages
            messages = conn.execute(
                """
                SELECT role, content, timestamp FROM messages
                WHERE session_id = ? ORDER BY timestamp
                """,
                (session_id,),
            ).fetchall()

            # Get files
            files = conn.execute(
                "SELECT path FROM session_files WHERE session_id = ?",
                (session_id,),
            ).fetchall()

            return {
                "id": row["id"],
                "title": row["title"],
                "profile": row["profile"],
                "working_dir": row["working_dir"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "messages": [
                    Message(role=m["role"], content=m["content"])
                    for m in messages
                ],
                "files": [f["path"] for f in files],
            }

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        List recent sessions.

        Args:
            limit: Max number of sessions to return

        Returns:
            List of session summaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT id, title, profile, working_dir, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "profile": row["profile"],
                    "working_dir": row["working_dir"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    def update_title(self, session_id: str, title: str) -> None:
        """Update session title."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )
            conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its messages.

        Args:
            session_id: Session ID

        Returns:
            True if deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_files WHERE session_id = ?", (session_id,))
            result = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return result.rowcount > 0

    def auto_title(self, session_id: str, first_message: str) -> str:
        """
        Generate a title from the first user message.

        Args:
            session_id: Session ID
            first_message: First user message

        Returns:
            Generated title
        """
        # Take first 50 chars, clean up
        title = first_message[:50].strip()
        if len(first_message) > 50:
            title += "..."

        # Remove newlines
        title = title.replace("\n", " ")

        self.update_title(session_id, title)
        return title
