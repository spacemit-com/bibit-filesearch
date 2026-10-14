"""扫描状态持久化（重启可继续）。

scan_state 表：scan_status + 遍历位置（目录栈）+ 时间戳。
"""
from __future__ import annotations

import json
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_state (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


class ScanState:
    def __init__(self, db):
        self.db = db
        self.db.conn.execute(_SCHEMA)
        self.db.conn.commit()

    def get(self, key: str, default=None):
        r = self.db.conn.execute("SELECT value FROM scan_state WHERE key=?", (key,)).fetchone()
        if r is None:
            return default
        try:
            return json.loads(r[0])
        except (json.JSONDecodeError, TypeError):
            return r[0]

    def set(self, key: str, value):
        self.db.conn.execute(
            "INSERT INTO scan_state (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.db.conn.commit()

    def delete(self, key: str):
        self.db.conn.execute("DELETE FROM scan_state WHERE key=?", (key,))
        self.db.conn.commit()


def now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def monotonic() -> float:
    return time.monotonic()
