"""
SQLite persistence for API keys, budgets, and request history.

Why SQLite (from DECISIONS.md): zero extra infrastructure, durable across
restarts unlike an in-memory dict, and openable by hand with any sqlite client
during debugging. A real product might graduate to Postgres or Redis; that
solves a scale problem this project does not have.

Why every function here is SYNCHRONOUS
--------------------------------------
The `sqlite3` module is blocking. Calling it directly inside an `async def`
handler would stall the entire event loop — exactly the mistake avoided in
Stage 3 by choosing httpx over requests. Callers in main.py therefore wrap
these in `asyncio.to_thread`. Keeping this module plainly synchronous makes
that boundary explicit rather than hiding it behind a fake-async wrapper.

Connections are opened per call rather than shared. SQLite connections are not
safe to move between threads, and `asyncio.to_thread` gives no guarantee about
which thread runs the call. Opening a connection costs microseconds against a
local file, so pooling would be optimising the wrong thing.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("GATEWAY_DB_PATH", "gateway/usage.db"))

# SCHEMA NOTES
# -----------
# Two tables and one index. Deliberately small - the alternative designs were
# considered and rejected for reasons worth recording:
#
#   * No `sessions`/`conversations` table. The API is stateless: clients resend
#     the whole conversation each turn, which is what makes horizontal scaling
#     a question about the ledger alone.
#   * No rollup/summary table for the dashboard. It would need maintaining in
#     step with every write, and a drifting summary on a BILLING system is a
#     worse failure than a slow page. Measured at 200,000 requests, the
#     dashboard's aggregate costs ~30 ms - invisible to a human, and the
#     project's own rule is that an optimisation is only meaningful against a
#     stated objective.
#   * `created_at` is ISO-8601 TEXT in UTC, not an integer epoch. Sortable
#     lexicographically, readable in any sqlite client by hand, and this is not
#     a time-series workload.
#
# The one index exists because every dashboard query filters or groups by
# api_key. It is a COVERING index for the per-key request count, which is why
# that correlated subquery in list_keys() measures FASTER than the grouped-join
# rewrite that looks tidier (17.05 ms vs 20.93 ms at 200,000 rows).
SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    key           TEXT PRIMARY KEY,
    name          TEXT    NOT NULL,
    token_budget  INTEGER NOT NULL,
    tokens_used   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key           TEXT    NOT NULL,
    model             TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    status_code       INTEGER NOT NULL,
    streamed          INTEGER NOT NULL DEFAULT 0,
    duration_ms       INTEGER,
    created_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_key_time
    ON requests (api_key, created_at DESC);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    # WAL lets the dashboard read while requests are being written, instead of
    # readers and writers blocking each other. The default rollback journal
    # would make a dashboard refresh contend with live traffic.
    conn.execute("PRAGMA journal_mode=WAL")

    # NO FOREIGN KEY EXISTS ON requests.api_key, AND THAT IS DELIBERATE.
    #
    # The obvious schema would reference api_keys(key). It would also be wrong:
    # `requests` is an append-only billing ledger, and a ledger has to outlive
    # the thing it bills. Deleting a key must not cascade away its history, and
    # it must not be blocked by that history either - so neither ON DELETE
    # CASCADE nor RESTRICT is the behaviour wanted. The join in recent_requests()
    # is therefore a LEFT JOIN: a request whose key is gone still shows, with a
    # null name.
    #
    # The pragma stays on regardless. SQLite disables foreign keys PER
    # CONNECTION by default, so a constraint added here later would silently not
    # be enforced - a footgun worth disarming in advance rather than debugging.
    conn.execute("PRAGMA foreign_keys=ON")

    # Durability is left at the SQLite default (FULL) rather than the usual WAL
    # tuning of NORMAL. NORMAL can lose the last transactions on power loss;
    # this database decides who gets billed, and the project already argued
    # that accounting fails CLOSED. A few milliseconds per write is the right
    # side of that trade.
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db() -> None:
    """Create tables if absent. Safe to call on every startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# Keys and budgets
# --------------------------------------------------------------------------


def get_key(key: str) -> sqlite3.Row | None:
    """Look up one API key. Returns None if it does not exist."""
    with _connect() as conn:
        return conn.execute(
            "SELECT key, name, token_budget, tokens_used FROM api_keys WHERE key = ?",
            (key,),
        ).fetchone()


def upsert_key(key: str, name: str, token_budget: int) -> None:
    """Create a key, or update its name and budget if it already exists.

    Deliberately does NOT reset tokens_used — re-running the seed script to
    raise someone's budget should not silently erase their usage history.
    """
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO api_keys (key, name, token_budget, tokens_used, created_at)
            VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(key) DO UPDATE SET name = excluded.name,
                                           token_budget = excluded.token_budget
            """,
            (key, name, token_budget, _now()),
        )


def list_keys() -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            """
            SELECT key, name, token_budget, tokens_used, created_at,
                   (token_budget - tokens_used) AS tokens_remaining,
                   (SELECT COUNT(*) FROM requests r WHERE r.api_key = api_keys.key)
                       AS request_count
            FROM api_keys
            ORDER BY name
            """
        ).fetchall()


# --------------------------------------------------------------------------
# Request recording
# --------------------------------------------------------------------------


def record_request(
    api_key: str,
    model: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    status_code: int,
    streamed: bool,
    duration_ms: int,
) -> None:
    """Log one request and deduct its tokens, in a single transaction.

    Both statements share one transaction deliberately. If they were separate,
    a crash between them would either bill tokens that were never logged or log
    a request that was never billed — and the Stage 4 dashboard would then show
    numbers that disagree with the budget it is enforcing.

    The UPDATE is written as `tokens_used = tokens_used + ?` rather than
    read-modify-write in Python, so two concurrent requests cannot both read
    the same starting value and lose one another's increment. Under the Stage 9
    load test there will be many concurrent requests against one key, and a
    read-modify-write would silently undercount.
    """
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO requests (api_key, model, prompt_tokens, completion_tokens,
                                  total_tokens, status_code, streamed, duration_ms,
                                  created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                api_key,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                status_code,
                1 if streamed else 0,
                duration_ms,
                _now(),
            ),
        )
        if total_tokens:
            conn.execute(
                "UPDATE api_keys SET tokens_used = tokens_used + ? WHERE key = ?",
                (total_tokens, api_key),
            )


def recent_requests(limit: int = 50) -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            """
            SELECT r.id, r.api_key, k.name AS key_name, r.model,
                   r.prompt_tokens, r.completion_tokens, r.total_tokens,
                   r.status_code, r.streamed, r.duration_ms, r.created_at
            FROM requests r
            LEFT JOIN api_keys k ON k.key = r.api_key
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def totals() -> sqlite3.Row:
    with _connect() as conn:
        return conn.execute(
            """
            SELECT COUNT(*)                              AS request_count,
                   COALESCE(SUM(prompt_tokens), 0)       AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0)   AS completion_tokens,
                   COALESCE(SUM(total_tokens), 0)        AS total_tokens,
                   COALESCE(SUM(status_code >= 400), 0)  AS error_count
            FROM requests
            """
        ).fetchone()
