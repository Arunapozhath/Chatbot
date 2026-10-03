"""
RLHF feedback storage (SQLite) and DPO training data export.
"""
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import FEEDBACK_DB_PATH


def get_conn():
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            user_id     TEXT,
            question    TEXT NOT NULL,
            answer      TEXT NOT NULL,
            rating      INTEGER NOT NULL,   -- 1=thumbs_up, -1=thumbs_down, 0=neutral
            correction  TEXT,               -- user's improved answer (optional)
            source_type TEXT,               -- "rag", "sql", "hybrid", "web"
            sources     TEXT,               -- JSON list of source filenames
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dpo_pairs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt      TEXT NOT NULL,
            chosen      TEXT NOT NULL,      -- preferred answer
            rejected    TEXT NOT NULL,      -- rejected answer
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_rating   ON feedback(rating);
        CREATE INDEX IF NOT EXISTS idx_feedback_user     ON feedback(user_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_session  ON feedback(session_id);
    """)
    conn.commit()
    conn.close()


def save_feedback(
    session_id: str,
    user_id: str,
    question: str,
    answer: str,
    rating: int,
    correction: Optional[str] = None,
    source_type: str = "rag",
    sources: list = None,
):
    conn = get_conn()
    conn.execute(
        """INSERT INTO feedback
           (session_id, user_id, question, answer, rating, correction, source_type, sources, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id, user_id, question, answer, rating,
            correction, source_type,
            json.dumps(sources or []),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()

    # auto-build DPO pair when a correction is provided with a thumbs-down
    if rating == -1 and correction:
        conn.execute(
            "INSERT INTO dpo_pairs (prompt, chosen, rejected, created_at) VALUES (?, ?, ?, ?)",
            (question, correction, answer, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    conn.close()


def get_stats() -> dict:
    conn = get_conn()
    stats = {}
    stats["total"]        = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    stats["thumbs_up"]    = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating=1").fetchone()[0]
    stats["thumbs_down"]  = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating=-1").fetchone()[0]
    stats["corrections"]  = conn.execute("SELECT COUNT(*) FROM feedback WHERE correction IS NOT NULL").fetchone()[0]
    stats["dpo_pairs"]    = conn.execute("SELECT COUNT(*) FROM dpo_pairs").fetchone()[0]

    top = conn.execute(
        "SELECT question, COUNT(*) as cnt FROM feedback GROUP BY question ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    stats["top_questions"] = [{"question": r[0], "count": r[1]} for r in top]

    conn.close()
    return stats


def export_dpo_pairs() -> list[dict]:
    """Export all DPO training pairs as list of {prompt, chosen, rejected}."""
    conn = get_conn()
    rows = conn.execute("SELECT prompt, chosen, rejected FROM dpo_pairs").fetchall()
    conn.close()
    return [{"prompt": r[0], "chosen": r[1], "rejected": r[2]} for r in rows]


def get_recent_feedback(limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# initialise on import
init_db()
