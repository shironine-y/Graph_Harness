from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


class GraphReadyBuild:
    REQUIRED_FILES = (
        "_SUCCESS",
        "graph_schema.json",
        "build_manifest.json",
        "audit/probability_validation.json",
        "indexes/tasks.jsonl",
        "indexes/nodes.jsonl",
        "indexes/relationships.jsonl",
        "indexes/transition_statistics.jsonl",
    )

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        missing = [name for name in self.REQUIRED_FILES if not (self.root / name).is_file()]
        if missing:
            raise ValueError(f"Incomplete graph-ready build: missing {', '.join(missing)}")
        self.schema = self._read_json("graph_schema.json")
        self.manifest = self._read_json("build_manifest.json")
        self.validation = self._read_json("audit/probability_validation.json")
        if self.schema.get("schema_version") != "4.0.0":
            raise ValueError("Expected graph-ready schema 4.0.0")
        if self.manifest.get("schema_version") != "4.0.0":
            raise ValueError("Build manifest schema does not match graph schema")
        if self.manifest.get("status") != "complete" or self.manifest.get("validation") != "passed":
            raise ValueError("Graph-ready build is not complete and validated")
        if self.validation.get("valid") is not True or self.validation.get("problems"):
            raise ValueError("Graph-ready probability validation did not pass")
        scope = self.schema.get("probability_scope") or {}
        if scope.get("estimator_scope") != "exact_task":
            raise ValueError("Only exact_task graph-ready statistics are supported")
        if scope.get("transition_probability_unit") != "occurrence":
            raise ValueError("Unexpected transition probability counting unit")
        if scope.get("outcome_and_coverage_unit") != "unique_trace":
            raise ValueError("Unexpected outcome counting unit")
        self.graph_version = str(self.manifest.get("build_id") or "")
        if not self.graph_version:
            raise ValueError("Graph-ready build ID is missing")
        self.manifest_sha256 = hashlib.sha256(
            (self.root / "build_manifest.json").read_bytes()
        ).hexdigest()

        self.tasks: dict[str, dict[str, Any]] = {}
        for row in self._rows("indexes/tasks.jsonl"):
            task_id = str(row.get("task_id") or "")
            if not task_id or task_id in self.tasks:
                raise ValueError("Graph-ready tasks contain a missing or duplicate task ID")
            if row.get("dataset") not in {"train", "dev"}:
                raise ValueError(f"Graph-ready task {task_id} is outside train/dev")
            self.tasks[task_id] = row
        expected = (self.manifest.get("summary") or {}).get("task_count")
        if expected != len(self.tasks):
            raise ValueError(f"Graph-ready task count mismatch: {len(self.tasks)} != {expected}")

    def _read_json(self, name: str) -> dict[str, Any]:
        value = json.loads((self.root / name).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {name}")
        return value

    def _rows(self, name: str):
        with (self.root / name).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Expected object in {name}:{number}")
                    yield value

    @property
    def split_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(row["dataset"] for row in self.tasks.values()).items()))

    @property
    def summary(self) -> dict[str, Any]:
        counts = self.manifest["summary"]
        return {
            "graph_version": self.graph_version,
            "manifest_sha256": self.manifest_sha256,
            "schema_version": "4.0.0",
            "build_path": str(self.root),
            "task_count": len(self.tasks),
            "trace_count": counts.get("trace_count"),
            "node_count": counts.get("node_count"),
            "relationship_count": counts.get("relationship_count"),
            "transition_count": counts.get("transition_count"),
            "split_counts": self.split_counts,
            "estimator_scope": "exact_task",
            "cross_task_gate_calibrated": False,
        }

    def transitions_for_task(self, task_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return bounded exact-task evidence without exporting raw node properties."""
        if task_id not in self.tasks:
            return []
        task_uid = self.tasks[task_id].get("task_uid")
        if not task_uid:
            raise ValueError(f"Graph-ready task {task_id} has no task_uid")
        rows = []
        for row in self._rows("indexes/transition_statistics.jsonl"):
            if row.get("task_uid") != task_uid:
                continue
            if row.get("estimator_scope") != "exact_task":
                raise ValueError("Transition estimator scope differs from graph schema")
            rows.append({
                "source_event_uid": row.get("source_event_uid"),
                "target_event_uid": row.get("target_event_uid"),
                "transition_probability": row.get("transition_probability"),
                "conditional_success_rate": row.get("conditional_success_rate"),
                "conditional_partial_rate": row.get("conditional_partial_rate"),
                "conditional_failure_rate": row.get("conditional_failure_rate"),
                "success_trace_coverage": row.get("success_trace_coverage"),
                "occurrence_count": row.get("occurrence_count"),
                "trace_support_count": row.get("trace_support_count"),
                "sample_size": row.get("sample_size"),
            })
        rows.sort(key=lambda row: (-(row["sample_size"] or 0), str(row["source_event_uid"])))
        return rows[:limit]
