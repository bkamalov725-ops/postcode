"""Persistent assessment history; images are never kept in the database."""
from pathlib import Path
import sqlite3


class AssessmentStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            with connection:
                connection.execute("""
                CREATE TABLE IF NOT EXISTS assessments (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    assessment_id TEXT NOT NULL UNIQUE,
                    shipment_id TEXT NOT NULL,
                    load_pct REAL NOT NULL CHECK (load_pct >= 0 AND load_pct <= 100),
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    processing_ms REAL NOT NULL,
                    image_sha256 TEXT NOT NULL
                )
                """)
                connection.execute("""
                CREATE INDEX IF NOT EXISTS assessments_shipment_sequence
                ON assessments (shipment_id, sequence DESC)
                """)
        finally:
            connection.close()

    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def add(self, assessment):
        columns = (
            "assessment_id", "shipment_id", "load_pct", "model_version",
            "created_at", "processing_ms", "image_sha256",
        )
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO assessments (" + ", ".join(columns)
                    + ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    tuple(assessment[column] for column in columns),
                )
        finally:
            connection.close()

    def history(self, shipment_id, limit=100):
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT assessment_id, shipment_id, load_pct, model_version, "
                "created_at, processing_ms, image_sha256 FROM assessments "
                "WHERE shipment_id = ? ORDER BY sequence DESC LIMIT ?",
                (shipment_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def latest(self, shipment_id):
        rows = self.history(shipment_id, limit=1)
        return rows[0] if rows else None
