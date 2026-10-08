"""Durable state.

One SQLite database holds everything the engine needs to resume: objectives,
hypotheses, campaigns, actions, execution records, observations, provenance,
decisions, strategies and claims.

Design points that matter:

* **Resumability.**  Every long-lived object is written as it changes, so a campaign
  interrupted mid-run can be reloaded and continued rather than restarted.
* **Append-only decision log.**  Decisions and state transitions are never updated in
  place; "why did the engine run this?" must remain answerable after the fact.
* **Typed round-trips.**  Rows go back out as the same pydantic models that went in,
  so a schema drift shows up as a validation error rather than a silent ``KeyError``
  three layers away.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import (
    Action,
    ActionStatus,
    ExecutionRecord,
    Hypothesis,
    Objective,
    Observation,
    utc_now,
)
from polymer_engine.core.provenance import Artifact, ProvenanceGraph

logger = get_logger("db.store")

SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS objectives (
    id         TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id           TEXT PRIMARY KEY,
    objective_id TEXT,
    status       TEXT NOT NULL,
    payload      TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hypotheses_objective ON hypotheses(objective_id);

CREATE TABLE IF NOT EXISTS campaigns (
    id           TEXT PRIMARY KEY,
    objective_id TEXT,
    polymer_id   TEXT,
    status       TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns(status);

CREATE TABLE IF NOT EXISTS actions (
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT,
    hypothesis_id TEXT,
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL,
    payload       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status);
CREATE INDEX IF NOT EXISTS idx_actions_campaign ON actions(campaign_id);

CREATE TABLE IF NOT EXISTS execution_records (
    id          TEXT PRIMARY KEY,
    campaign_id TEXT,
    action_id   TEXT,
    kind        TEXT NOT NULL,
    state       TEXT NOT NULL,
    payload     TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_execution_state ON execution_records(state);

CREATE TABLE IF NOT EXISTS observations (
    id          TEXT PRIMARY KEY,
    action_id   TEXT NOT NULL,
    campaign_id TEXT,
    metric      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observations_metric ON observations(metric);
CREATE INDEX IF NOT EXISTS idx_observations_action ON observations(action_id);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_edges (
    child  TEXT NOT NULL,
    parent TEXT NOT NULL,
    PRIMARY KEY (child, parent)
);
CREATE INDEX IF NOT EXISTS idx_edges_parent ON artifact_edges(parent);

CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT,
    decision    TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_campaign ON decisions(campaign_id);

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outcomes_strategy ON strategy_outcomes(strategy_id);

CREATE TABLE IF NOT EXISTS claims (
    claim_id   TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
"""


def _dumps(payload: Any) -> str:
    return json.dumps(payload, default=_default, sort_keys=True)


def _default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (datetime, Path)):
        return str(value)
    if isinstance(value, set):
        return sorted(value, key=str)
    return str(value)


class Store:
    """SQLite-backed engine state.

    Safe to share across threads: a lock serialises writes and the connection is
    opened with ``check_same_thread=False``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA)
            row = self.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
                )
            elif int(row["value"]) != SCHEMA_VERSION:
                raise PolymerEngineError(
                    "Database schema version mismatch",
                    path=str(self.path),
                    found=row["value"],
                    expected=SCHEMA_VERSION,
                )
            self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Objectives and hypotheses
    # ------------------------------------------------------------------
    def save_objective(self, objective: Objective) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO objectives(id, payload, created_at) VALUES (?, ?, ?)",
                (objective.id, objective.model_dump_json(), objective.created_at.isoformat()),
            )

    def get_objective(self, objective_id: str) -> Objective | None:
        row = self.conn.execute("SELECT payload FROM objectives WHERE id = ?", (objective_id,)).fetchone()
        return Objective.model_validate_json(row["payload"]) if row else None

    def list_objectives(self) -> list[Objective]:
        rows = self.conn.execute("SELECT payload FROM objectives ORDER BY created_at").fetchall()
        return [Objective.model_validate_json(r["payload"]) for r in rows]

    def save_hypothesis(self, hypothesis: Hypothesis) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hypotheses(id, objective_id, status, payload, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    hypothesis.id,
                    hypothesis.objective_id,
                    hypothesis.status.value,
                    hypothesis.model_dump_json(),
                    utc_now().isoformat(),
                ),
            )

    def list_hypotheses(self, objective_id: str | None = None) -> list[Hypothesis]:
        if objective_id:
            rows = self.conn.execute(
                "SELECT payload FROM hypotheses WHERE objective_id = ? ORDER BY updated_at", (objective_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT payload FROM hypotheses ORDER BY updated_at").fetchall()
        return [Hypothesis.model_validate_json(r["payload"]) for r in rows]

    # ------------------------------------------------------------------
    # Campaigns
    # ------------------------------------------------------------------
    def save_campaign(self, campaign_id: str, payload: dict[str, Any], *, status: str,
                      objective_id: str | None = None, polymer_id: str | None = None) -> None:
        now = utc_now().isoformat()
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT created_at FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
            created = existing["created_at"] if existing else now
            conn.execute(
                "INSERT OR REPLACE INTO campaigns(id, objective_id, polymer_id, status, payload, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (campaign_id, objective_id, polymer_id, status, _dumps(payload), created, now),
            )

    def get_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT payload FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_campaigns(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self.conn.execute(
                "SELECT id, status, payload, updated_at FROM campaigns WHERE status = ? ORDER BY updated_at",
                (status,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, status, payload, updated_at FROM campaigns ORDER BY updated_at"
            ).fetchall()
        return [
            {"id": r["id"], "status": r["status"], "updated_at": r["updated_at"], **json.loads(r["payload"])}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def save_action(self, action: Action) -> None:
        now = utc_now().isoformat()
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO actions(id, campaign_id, hypothesis_id, kind, status, payload, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, "
                "COALESCE((SELECT created_at FROM actions WHERE id = ?), ?), ?)",
                (
                    action.id, action.campaign_id, action.hypothesis_id, action.kind,
                    action.status.value, action.model_dump_json(), action.id, now, now,
                ),
            )

    def get_action(self, action_id: str) -> Action | None:
        row = self.conn.execute("SELECT payload FROM actions WHERE id = ?", (action_id,)).fetchone()
        return Action.model_validate_json(row["payload"]) if row else None

    def list_actions(
        self, *, campaign_id: str | None = None, statuses: Sequence[ActionStatus] | None = None
    ) -> list[Action]:
        query = "SELECT payload FROM actions"
        clauses: list[str] = []
        params: list[Any] = []
        if campaign_id:
            clauses.append("campaign_id = ?")
            params.append(campaign_id)
        if statuses:
            clauses.append(f"status IN ({','.join('?' * len(statuses))})")
            params.extend(s.value for s in statuses)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at"
        rows = self.conn.execute(query, params).fetchall()
        return [Action.model_validate_json(r["payload"]) for r in rows]

    # ------------------------------------------------------------------
    # Execution records
    # ------------------------------------------------------------------
    def save_execution_record(
        self, record: ExecutionRecord, *, campaign_id: str | None = None, action_id: str | None = None
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO execution_records(id, campaign_id, action_id, kind, state, "
                "payload, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id, campaign_id, action_id, record.kind, record.state.value,
                    record.model_dump_json(), record.updated_at.isoformat(),
                ),
            )

    def get_execution_record(self, record_id: str) -> ExecutionRecord | None:
        row = self.conn.execute(
            "SELECT payload FROM execution_records WHERE id = ?", (record_id,)
        ).fetchone()
        return ExecutionRecord.model_validate_json(row["payload"]) if row else None

    def list_execution_records(self, *, campaign_id: str | None = None) -> list[ExecutionRecord]:
        if campaign_id:
            rows = self.conn.execute(
                "SELECT payload FROM execution_records WHERE campaign_id = ? ORDER BY updated_at",
                (campaign_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT payload FROM execution_records ORDER BY updated_at"
            ).fetchall()
        return [ExecutionRecord.model_validate_json(r["payload"]) for r in rows]

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def save_observation(self, observation: Observation) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO observations(id, action_id, campaign_id, metric, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    observation.id, observation.action_id, observation.campaign_id,
                    observation.metric, observation.model_dump_json(),
                    observation.created_at.isoformat(),
                ),
            )

    def list_observations(
        self, *, campaign_id: str | None = None, metric: str | None = None, action_id: str | None = None
    ) -> list[Observation]:
        query = "SELECT payload FROM observations"
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("campaign_id", campaign_id), ("metric", metric), ("action_id", action_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at"
        rows = self.conn.execute(query, params).fetchall()
        return [Observation.model_validate_json(r["payload"]) for r in rows]

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    def save_artifact(self, artifact: Artifact) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts(artifact_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                (artifact.artifact_id, artifact.kind, artifact.model_dump_json(), artifact.created_at.isoformat()),
            )
            for parent in artifact.parents:
                conn.execute(
                    "INSERT OR IGNORE INTO artifact_edges(child, parent) VALUES (?, ?)",
                    (artifact.artifact_id, parent),
                )

    def save_provenance_graph(self, graph: ProvenanceGraph) -> None:
        for artifact in graph:
            self.save_artifact(artifact)

    def load_provenance_graph(self) -> ProvenanceGraph:
        rows = self.conn.execute("SELECT payload FROM artifacts ORDER BY created_at").fetchall()
        return ProvenanceGraph(Artifact.model_validate_json(r["payload"]) for r in rows)

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        row = self.conn.execute(
            "SELECT payload FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        return Artifact.model_validate_json(row["payload"]) if row else None

    # ------------------------------------------------------------------
    # Decisions and events (append-only)
    # ------------------------------------------------------------------
    def record_decision(self, payload: dict[str, Any], *, campaign_id: str | None = None) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO decisions(campaign_id, decision, payload, created_at) VALUES (?, ?, ?, ?)",
                (
                    campaign_id,
                    str(payload.get("decision", "unspecified")),
                    _dumps(payload),
                    utc_now().isoformat(),
                ),
            )
            return int(cursor.lastrowid or 0)

    def list_decisions(self, *, campaign_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT id, campaign_id, decision, payload, created_at FROM decisions"
        params: list[Any] = []
        if campaign_id:
            query += " WHERE campaign_id = ?"
            params.append(campaign_id)
        query += " ORDER BY id"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [
            {"id": r["id"], "campaign_id": r["campaign_id"], "created_at": r["created_at"], **json.loads(r["payload"])}
            for r in rows
        ]

    def log_event(self, kind: str, payload: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO events(kind, payload, created_at) VALUES (?, ?, ?)",
                (kind, _dumps(payload), utc_now().isoformat()),
            )

    def list_events(self, kind: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if kind:
            rows = self.conn.execute(
                "SELECT kind, payload, created_at FROM events WHERE kind = ? ORDER BY id DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT kind, payload, created_at FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"kind": r["kind"], "created_at": r["created_at"], "payload": json.loads(r["payload"])} for r in rows]

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------
    def save_strategy(self, strategy_id: str, payload: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO strategies(strategy_id, payload, updated_at) VALUES (?, ?, ?)",
                (strategy_id, _dumps(payload), utc_now().isoformat()),
            )

    def get_strategy(self, strategy_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT payload FROM strategies WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_strategies(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT payload FROM strategies ORDER BY strategy_id").fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def record_strategy_outcome(self, strategy_id: str, payload: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO strategy_outcomes(strategy_id, payload, created_at) VALUES (?, ?, ?)",
                (strategy_id, _dumps(payload), utc_now().isoformat()),
            )

    def list_strategy_outcomes(self, strategy_id: str | None = None) -> list[dict[str, Any]]:
        if strategy_id:
            rows = self.conn.execute(
                "SELECT strategy_id, payload, created_at FROM strategy_outcomes WHERE strategy_id = ? ORDER BY id",
                (strategy_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT strategy_id, payload, created_at FROM strategy_outcomes ORDER BY id"
            ).fetchall()
        return [{"strategy_id": r["strategy_id"], "created_at": r["created_at"], **json.loads(r["payload"])} for r in rows]

    # ------------------------------------------------------------------
    # Claims
    # ------------------------------------------------------------------
    def save_claim(self, claim_id: str, status: str, payload: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO claims(claim_id, status, payload, updated_at) VALUES (?, ?, ?, ?)",
                (claim_id, status, _dumps(payload), utc_now().isoformat()),
            )

    def get_claim(self, claim_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT payload FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_claims(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self.conn.execute(
                "SELECT payload FROM claims WHERE status = ? ORDER BY claim_id", (status,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT payload FROM claims ORDER BY claim_id").fetchall()
        return [json.loads(r["payload"]) for r in rows]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, int]:
        tables = (
            "objectives", "hypotheses", "campaigns", "actions", "execution_records",
            "observations", "artifacts", "decisions", "strategies", "claims", "events",
        )
        return {
            table: int(self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
            for table in tables
        }


__all__ = ["SCHEMA_VERSION", "Store"]
