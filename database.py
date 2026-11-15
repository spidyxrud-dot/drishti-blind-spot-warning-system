"""SQLite event history and analytics for the blind-spot monitor."""
from __future__ import annotations

import csv
import sqlite3
from datetime import datetime
from pathlib import Path


class DetectionStore:
    def __init__(self, path: str | Path = "data/detections.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                object_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                event_type TEXT NOT NULL,
                warning_status TEXT NOT NULL,
                source TEXT NOT NULL,
                snapshot_path TEXT DEFAULT ''
            )""")
            try:
                conn.execute("ALTER TABLE detection_events ADD COLUMN snapshot_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON detection_events(timestamp)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=8)

    def add_event(self, session_id: str, track_id: int, object_name: str,
                  confidence: float, event_type: str, warning_status: str,
                  source: str, snapshot_path: str = "") -> None:
        with self._connect() as conn:
            conn.execute("""INSERT INTO detection_events
                (timestamp,session_id,track_id,object_name,confidence,event_type,warning_status,source,snapshot_path)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (datetime.now().astimezone().isoformat(timespec="seconds"), session_id,
                 int(track_id), object_name, float(confidence), event_type, warning_status, source, snapshot_path))

    def recent_events(self, limit: int = 100) -> list[tuple]:
        with self._connect() as conn:
            return conn.execute("""SELECT timestamp,object_name,confidence,event_type,warning_status,source,snapshot_path
                FROM detection_events ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()

    def alert_total(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM detection_events WHERE event_type='BLIND-SPOT ENTRY'").fetchone()[0])

    def class_totals(self, days: int = 30) -> list[tuple[str, int]]:
        with self._connect() as conn:
            return conn.execute("""SELECT object_name, COUNT(*) FROM detection_events
                WHERE event_type='OBJECT DETECTED' AND date(timestamp) >= date('now', ?)
                GROUP BY object_name ORDER BY COUNT(*) DESC""", (f"-{int(days)} days",)).fetchall()

    def daily_event_totals(self, days: int = 30) -> list[tuple[str, str, int]]:
        with self._connect() as conn:
            return conn.execute("""SELECT substr(timestamp,1,10), event_type, COUNT(*)
                FROM detection_events WHERE date(timestamp) >= date('now', ?)
                GROUP BY substr(timestamp,1,10), event_type ORDER BY substr(timestamp,1,10)""",
                (f"-{int(days)} days",)).fetchall()

    def export_csv(self, path: str | Path) -> int:
        rows = self.recent_events(1_000_000)
        with open(path, "w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(("Timestamp", "Object", "Confidence", "Event", "Warning status", "Source", "Snapshot File"))
            writer.writerows(rows)
        return len(rows)
