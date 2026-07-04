
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "chats.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    TEXT NOT NULL,
                role       TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)


def create_chat(title: str = "Nuevo chat") -> dict:
    chat_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
            (chat_id, title, now),
        )
    return {"id": chat_id, "title": title, "created_at": now}


def list_chats() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM chats ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_chat(chat_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, created_at FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_chat(chat_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def add_message(chat_id: str, role: str, content: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, now),
        )


def get_messages(chat_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def rename_chat_if_default(chat_id: str, first_message: str) -> None:
    snippet = first_message.strip()[:50]
    with get_conn() as conn:
        row = conn.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if row and row["title"] == "Nuevo chat" and snippet:
            conn.execute("UPDATE chats SET title = ? WHERE id = ?", (snippet, chat_id))
