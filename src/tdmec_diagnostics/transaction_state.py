"""Single-transaction authority for resumable Phase 2 diagnostics.

The SQLite file is an unsealed working artifact.  A committed transaction
contains deduplication grouping changes, the matching accumulator snapshot,
and the matching checkpoint snapshot.  JSON checkpoint files are mirrors for
inspection and final sealed artifacts; they are never the authority while the
transaction database exists.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tdmec_diagnostics.checkpoint import ConfigIncompatibleError


class TransactionStateError(RuntimeError):
    """The transactional working state is absent, corrupt, or inconsistent."""


class TransactionalRunState:
    """Durable SQLite transaction coordinating progress and accumulator state."""

    FILENAME = ".diagnostics_transaction.sqlite"
    SCHEMA_VERSION = "tdmec-phase2-transaction-v1"

    def __init__(self, root: Path, config_hash: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILENAME
        self.config_hash = config_hash
        self._closed = False
        self.connection = sqlite3.connect(
            str(self.path),
            isolation_level=None,
        )
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA journal_mode = DELETE")
        self.connection.execute("PRAGMA temp_store = FILE")
        self.connection.execute("PRAGMA cache_size = -8192")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transaction_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transaction_snapshots (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                generation INTEGER NOT NULL,
                checkpoint_json TEXT NOT NULL,
                accumulator_json TEXT NOT NULL
            )
            """
        )
        stored_schema = self._metadata("schema_version")
        stored_hash = self._metadata("config_hash")
        if stored_schema is None and stored_hash is None:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.executemany(
                    """
                    INSERT INTO transaction_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    (
                        ("schema_version", self.SCHEMA_VERSION),
                        ("config_hash", self.config_hash),
                    ),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        else:
            if stored_schema != self.SCHEMA_VERSION:
                raise TransactionStateError(
                    "transaction state schema is incompatible"
                )
            if stored_hash != self.config_hash:
                raise ConfigIncompatibleError(
                    "transaction state config_hash mismatch; "
                    "use resume_mode='restart'"
                )

    def _metadata(self, key: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT value FROM transaction_metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row[0])

    @property
    def has_snapshot(self) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM transaction_snapshots WHERE singleton = 1"
        ).fetchone()
        return row is not None

    @property
    def generation(self) -> int:
        row = self.connection.execute(
            "SELECT generation FROM transaction_snapshots WHERE singleton = 1"
        ).fetchone()
        return 0 if row is None else int(row[0])

    def load_snapshot(self) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        row = self.connection.execute(
            """
            SELECT checkpoint_json, accumulator_json, generation
            FROM transaction_snapshots
            WHERE singleton = 1
            """
        ).fetchone()
        if row is None:
            raise TransactionStateError(
                "transaction state has no committed snapshot"
            )
        try:
            checkpoint = json.loads(str(row[0]))
            accumulators = json.loads(str(row[1]))
        except Exception as exc:
            raise TransactionStateError(
                "transaction snapshot JSON is corrupt"
            ) from exc
        generation = int(row[2])
        if checkpoint.get("transaction_generation") != generation:
            raise TransactionStateError(
                "checkpoint transaction generation is inconsistent"
            )
        if accumulators.get("transaction_generation") != generation:
            raise TransactionStateError(
                "accumulator transaction generation is inconsistent"
            )
        return checkpoint, accumulators, generation

    def begin(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def rollback(self) -> None:
        if self.connection.in_transaction:
            self.connection.rollback()

    def commit_snapshot(
        self,
        checkpoint_payload: Dict[str, Any],
        accumulator_payload: Dict[str, Any],
    ) -> int:
        if not self.connection.in_transaction:
            raise TransactionStateError(
                "snapshot commit requires an active transaction"
            )
        generation = self.generation + 1
        checkpoint_payload["transaction_generation"] = generation
        accumulator_payload["transaction_generation"] = generation
        checkpoint_json = json.dumps(
            checkpoint_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        accumulator_json = json.dumps(
            accumulator_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            self.connection.execute(
                """
                INSERT INTO transaction_snapshots(
                    singleton,
                    generation,
                    checkpoint_json,
                    accumulator_json
                ) VALUES (1, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    generation = excluded.generation,
                    checkpoint_json = excluded.checkpoint_json,
                    accumulator_json = excluded.accumulator_json
                """,
                (generation, checkpoint_json, accumulator_json),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return generation

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def remove(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)
        for suffix in ("-journal", "-wal", "-shm"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
