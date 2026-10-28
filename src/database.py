"""SQLite demo data. Replace demo PIN checks with your real identity provider in production."""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from .config import DB_PATH


def _connect():
    path = Path(DB_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users ( 
            customer_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            email TEXT NOT NULL,
            demo_pin TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TEXT NOT NULL,
            FOREIGN KEY(customer_id) REFERENCES users(customer_id)
        );
        CREATE TABLE IF NOT EXISTS escalations (
            escalation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            reason TEXT NOT NULL,
            summary TEXT NOT NULL,
            conversation TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending_human',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            route TEXT NOT NULL,
            user_message TEXT NOT NULL,
            response TEXT NOT NULL,
            summary TEXT,
            escalation_id INTEGER,
            created_at TEXT NOT NULL
        );
        """)
        conn.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?)",
                     ("CUST-1001", "Alex Morgan", "alex@example.com", "2468"))
        conn.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?)",
                     ("CUST-1002", "Sam Lee", "sam@example.com", "1357"))
        conn.execute("""INSERT INTO tickets(customer_id, subject, description, status, created_at)
                        SELECT ?, ?, ?, ?, ? WHERE NOT EXISTS
                        (SELECT 1 FROM tickets WHERE customer_id=? AND subject=?)""",
                     ("CUST-1001", "Unable to sign in", "Login page shows an error.", "in_progress",
                      _now(), "CUST-1001", "Unable to sign in"))
        # Ensure any historical escalations are also indexed in interactions
        conn.execute("""
            INSERT INTO interactions (customer_id, route, user_message, response, summary, escalation_id, created_at)
            SELECT customer_id, 'escalation', reason, 'Forwarded to human support.', summary, escalation_id, created_at
            FROM escalations e
            WHERE NOT EXISTS (
                SELECT 1 FROM interactions i WHERE i.escalation_id = e.escalation_id
            )
        """)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def authenticate_customer(customer_id: str, pin: str) -> dict | None:
    if not customer_id or not pin:
        return None
    with _connect() as conn:
        row = conn.execute("SELECT customer_id, display_name, email FROM users WHERE customer_id=? AND demo_pin=?",
                           (customer_id.strip().upper(), pin.strip())).fetchone()
    return dict(row) if row else None


def create_ticket(customer_id: str, subject: str, description: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO tickets(customer_id, subject, description, status, created_at) VALUES (?, ?, ?, 'open', ?)",
            (customer_id, subject[:120], description[:4000], _now()),
        )
        return int(cur.lastrowid)


def get_ticket(ticket_id: int, customer_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT ticket_id, subject, description, status, created_at FROM tickets WHERE ticket_id=? AND customer_id=?",
                           (ticket_id, customer_id)).fetchone()
    return dict(row) if row else None


def list_tickets(customer_id: str, limit: int = 10) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT ticket_id, subject, status, created_at FROM tickets WHERE customer_id=? ORDER BY ticket_id DESC LIMIT ?",
                            (customer_id, limit)).fetchall()
    return [dict(r) for r in rows]


def create_escalation(customer_id: str | None, reason: str, summary: str, conversation: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO escalations(customer_id, reason, summary, conversation, status, created_at) VALUES (?, ?, ?, ?, 'pending_human', ?)",
            (customer_id, reason[:300], summary[:2000], conversation[:12000], _now()),
        )
        return int(cur.lastrowid)


def list_escalations(status: str | None = None, limit: int = 50) -> list[dict]:
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT escalation_id, customer_id, reason, summary, conversation, status, created_at FROM escalations WHERE status=? ORDER BY escalation_id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT escalation_id, customer_id, reason, summary, conversation, status, created_at FROM escalations ORDER BY escalation_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def update_escalation_status(escalation_id: int, status: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE escalations SET status=? WHERE escalation_id=?",
            (status, escalation_id),
        )
        return cur.rowcount > 0


def list_all_tickets(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ticket_id, customer_id, subject, description, status, created_at FROM tickets ORDER BY ticket_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def log_interaction(
    customer_id: str | None,
    route: str,
    user_message: str,
    response: str,
    summary: str = "",
    escalation_id: int | None = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO interactions (customer_id, route, user_message, response, summary, escalation_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (customer_id, route, user_message[:2000], response[:4000], summary[:4000], escalation_id, _now()),
        )
        return int(cur.lastrowid)


def list_interactions(route: str | None = None, limit: int = 100) -> list[dict]:
    with _connect() as conn:
        if route and route.lower() != "all":
            rows = conn.execute(
                """SELECT interaction_id, customer_id, route, user_message, response, summary, escalation_id, created_at
                   FROM interactions WHERE route=? ORDER BY interaction_id DESC LIMIT ?""",
                (route, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT interaction_id, customer_id, route, user_message, response, summary, escalation_id, created_at
                   FROM interactions ORDER BY interaction_id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]

