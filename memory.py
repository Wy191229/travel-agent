import sqlite3
from pathlib import Path
from typing import Dict, List

DB_PATH = Path(__file__).with_name("memory.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)"
        )


def save_message(session_id: str, role: str, content: str):
    if not session_id or not content:
        return

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def save_turn(session_id: str, user_message: str, assistant_answer: str):
    save_message(session_id, "user", user_message)
    save_message(session_id, "assistant", assistant_answer)


def get_recent_messages(session_id: str, limit: int = 8) -> List[Dict[str, str]]:
    if not session_id:
        return []

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    rows = list(reversed(rows))
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def clear_session(session_id: str):
    if not session_id:
        return

    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
