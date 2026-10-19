from __future__ import annotations

from collections import Counter, defaultdict
import ast
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from harness.probability import ProbabilityAssessment, ProbabilityProvider


DATA_STATUS = "APPWORLD_TRACE_PILOT_NOT_GATE_GRADE"
EMPIRICAL_DATA_STATUS = "APPWORLD_TRAIN_EMPIRICAL_FIVE_RUNS"


class AppWorldAction(BaseModel):
    category: str
    method: str
    route: str
    post_attributes: dict[str, str | bool | int] = Field(default_factory=dict)


class AppWorldTraceRecord(BaseModel):
    trace_id: str
    task_id: str
    experiment_id: str
    success: bool
    actions: list[AppWorldAction]
    source_sha256: str


class AppWorldTraceCell(BaseModel):
    state: str
    action: str
    post_attributes: dict[str, str | bool | int] = Field(default_factory=dict)
    next_state_counts: dict[str, int]
    run_count: int
    unique_task_count: int
    successful_run_count: int = 0
    failed_run_count: int = 0
    successful_task_count: int
    failed_task_count: int
    evidence_refs: list[str]
    representative_routes: list[str] = Field(default_factory=list)


class AppWorldTraceGraphSpec(BaseModel):
    schema_version: str = "1.2"
    graph_version: str
    data_status: str = DATA_STATUS
    source: str = "AppWorld labeled api_calls traces"
    probability_kind: str = "P_FINAL_TASK_SUCCESS_AFTER_VERIFIED_STATE_ACTION"
    records: list[AppWorldTraceRecord]
    cells: list[AppWorldTraceCell]
    policy: dict[str, str]
    min_unique_tasks_for_gate: int = 10
    gate_eligible: bool = False
    limitations: list[str] = Field(default_factory=list)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _route_template(url: str) -> str:
    path = urlsplit(url).path
    pieces = []
    for piece in path.split("/"):
        if not piece:
            continue
        # API documentation uses descriptive placeholder names such as
        # ``{task_id}``, while AppWorld's runtime request tracker may expose the
        # same route as ``{id}``.  Placeholder names are documentation details,
        # not different endpoints, so canonicalize both forms before comparing
        # the checked proposal with the calls that actually ran.
        if re.fullmatch(r"\{[^/{}]+\}", piece) or piece.isdigit() or re.fullmatch(
            r"[0-9a-fA-F]{8,}|[0-9a-fA-F-]{32,}", piece
        ):
            pieces.append("{id}")
        else:
            pieces.append(piece)
    return "/" + "/".join(pieces)


def _classify(method: str, route: str) -> str:
    method = method.upper()
    if route == "/supervisor/message":
        return "COMPLETE_TASK"
    if route in {"/supervisor/profile", "/supervisor/account_passwords"}:
        return "LOAD_IDENTITY_OR_CREDENTIALS"
    if route.endswith("/auth/token"):
        return "AUTHENTICATE"
    if method == "GET":
        return "READ_RESOURCE"
    if method in {"POST", "PUT", "PATCH", "DELETE"}:
        return "WRITE_RESOURCE"
    return "OTHER_API_CALL"


def _state_after(category: str) -> str:
    return f"AFTER_{category}"


SAFE_POST_ATTRIBUTE_KEYS = {
    "execution_status",
    "response_shape",
    "result_count_bucket",
    "nonempty_result",
    "error_class",
    "authentication_succeeded",
}

SAFE_POST_ATTRIBUTE_ENUMS = {
    "execution_status": {"success", "failure"},
    "response_shape": {"list", "object", "boolean", "number", "null", "empty", "error", "text"},
    "result_count_bucket": {"0", "1", "2-5", "6-20", "21+", "unknown"},
    "error_class": {
        "none", "timeout", "authentication", "authorization", "not_found",
        "validation", "execution",
    },
}
SAFE_POST_ATTRIBUTE_BOOLEANS = {"nonempty_result", "authentication_succeeded"}


def validate_safe_post_attributes(attributes: Mapping[str, Any]) -> bool:
    if set(attributes) - SAFE_POST_ATTRIBUTE_KEYS:
        return False
    for key, value in attributes.items():
        if key in SAFE_POST_ATTRIBUTE_ENUMS:
            if not isinstance(value, str) or value not in SAFE_POST_ATTRIBUTE_ENUMS[key]:
                return False
        elif key in SAFE_POST_ATTRIBUTE_BOOLEANS:
            if not isinstance(value, bool):
                return False
        else:
            return False
    return True


def _count_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    return "21+"


def derive_safe_post_attributes(
    output: Any, category: str
) -> dict[str, str | bool | int]:
    """Derive bounded non-sensitive features without retaining tool output values."""
    if output is None:
        output_text = ""
    elif isinstance(output, str):
        output_text = output
    elif isinstance(output, bytes):
        output_text = output.decode("utf-8", errors="replace")
    else:
        output_text = str(output)
    stripped = output_text.strip()
    lowered = stripped.lower()
    failed = stripped.startswith("Execution failed. Traceback:")
    if "timed out" in lowered or "timeout" in lowered:
        error_class = "timeout"
    elif "unauthorized" in lowered or "authentication" in lowered:
        error_class = "authentication"
    elif "forbidden" in lowered or "permission" in lowered:
        error_class = "authorization"
    elif "not found" in lowered:
        error_class = "not_found"
    elif "validation" in lowered or "invalid" in lowered:
        error_class = "validation"
    elif failed:
        error_class = "execution"
    else:
        error_class = "none"

    parsed: Any = None
    parsed_ok = False
    if stripped and len(stripped) <= 1_000_000 and not failed:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
                parsed_ok = True
                break
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue

    if parsed_ok and isinstance(parsed, list):
        response_shape = "list"
        result_count = len(parsed)
    elif parsed_ok and isinstance(parsed, dict):
        response_shape = "object"
        result_count = None
        for key in ("results", "items", "data", "records"):
            value = parsed.get(key)
            if isinstance(value, list):
                result_count = len(value)
                break
        if result_count is None:
            result_count = len(parsed)
    elif parsed_ok and isinstance(parsed, bool):
        response_shape = "boolean"
        result_count = int(parsed)
    elif parsed_ok and isinstance(parsed, (int, float)):
        response_shape = "number"
        result_count = 1
    elif parsed_ok and parsed is None:
        response_shape = "null"
        result_count = 0
    elif not stripped or stripped == "Execution successful.":
        response_shape = "empty"
        result_count = 0
    elif failed:
        response_shape = "error"
        result_count = 0
    else:
        response_shape = "text"
        result_count = None

    attributes: dict[str, str | bool | int] = {
        "execution_status": "failure" if failed else "success",
        "response_shape": response_shape,
        "result_count_bucket": (
            _count_bucket(result_count) if result_count is not None else "unknown"
        ),
        "nonempty_result": bool(result_count) if result_count is not None else bool(stripped),
        "error_class": error_class,
    }
    if category == "AUTHENTICATE":
        attributes["authentication_succeeded"] = not failed and error_class == "none"
    return attributes


def _attribute_chain(node: ast.AST) -> list[str] | None:
    names: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        names.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    names.append(current.id)
    return list(reversed(names))


_DIRECT_APPWORLD_API_NAME = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*__[A-Za-z][A-Za-z0-9_]*$"
)


def _is_appworld_api_call(node: ast.Call) -> bool:
    """Recognize both AppWorld API calling conventions without reading arguments."""

    chain = _attribute_chain(node.func)
    if chain and len(chain) == 3 and chain[0] == "apis":
        return True
    return (
        isinstance(node.func, ast.Name)
        and _DIRECT_APPWORLD_API_NAME.fullmatch(node.func.id) is not None
    )


def _outputs_by_api_call(environment_io_path: Path) -> list[str | None]:
    if not environment_io_path.is_file():
        return []
    content = environment_io_path.read_text(encoding="utf-8")
    blocks = re.findall(
        r"```python\n(.*?)\n```\n\n```\n(.*?)\n```", content, flags=re.DOTALL
    )
    outputs: list[str | None] = []
    for code, output in blocks:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        call_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_appworld_api_call(node):
                call_count += 1
        if call_count == 1:
            outputs.append(output)
        elif call_count > 1:
            outputs.extend([None] * call_count)
    return outputs


class AppWorldTraceGraphBuilder:
    """Read evaluation labels plus api_calls.jsonl without copying request data."""

    def __init__(self, outputs_root: str | Path, *, min_unique_tasks_for_gate: int = 10):
        self.outputs_root = Path(outputs_root).resolve()
        self.min_unique_tasks_for_gate = min_unique_tasks_for_gate

    def _actions(
        self, api_calls_path: Path, environment_io_path: Path | None = None
    ) -> list[AppWorldAction]:
        actions = []
        outputs = (
            _outputs_by_api_call(environment_io_path)
            if environment_io_path is not None
            else []
        )
        for index, line in enumerate(api_calls_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            value = json.loads(line)
            method = str(value.get("method", "UNKNOWN")).upper()
            route = _route_template(str(value.get("url", "/unknown")))
            category = _classify(method, route)
            output = outputs[index] if index < len(outputs) else None
            actions.append(AppWorldAction(
                category=category,
                method=method,
                route=route,
                post_attributes=(
                    derive_safe_post_attributes(output, category)
                    if output is not None
                    else {}
                ),
            ))
        return actions

    def discover_records(self) -> list[AppWorldTraceRecord]:
        records = []
        seen = set()
        for evaluation_path in sorted(self.outputs_root.glob("**/evaluations/*.json")):
            try:
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            run_root = evaluation_path.parent.parent
            experiment_id = str(run_root.relative_to(self.outputs_root)).replace("\\", "/")
            for task_id, result in sorted(evaluation.get("individual", {}).items()):
                if not isinstance(result, dict) or "success" not in result:
                    continue
                api_calls_path = run_root / "tasks" / task_id / "logs" / "api_calls.jsonl"
                if not api_calls_path.is_file() or api_calls_path.stat().st_size == 0:
                    continue
                raw = api_calls_path.read_bytes()
                environment_io_path = run_root / "tasks" / task_id / "logs" / "environment_io.md"
                environment_raw = (
                    environment_io_path.read_bytes()
                    if environment_io_path.is_file()
                    else b""
                )
                actions = self._actions(api_calls_path, environment_io_path)
                if not actions:
                    continue
                source_sha256 = _sha256_bytes(raw + b"\0" + environment_raw)
                identity = f"{experiment_id}:{task_id}:{source_sha256}"
                if identity in seen:
                    continue
                seen.add(identity)
                records.append(
                    AppWorldTraceRecord(
                        trace_id=hashlib.sha256(identity.encode()).hexdigest()[:20],
                        task_id=str(task_id),
                        experiment_id=experiment_id,
                        success=bool(result["success"]),
                        actions=actions,
                        source_sha256=source_sha256,
                    )
                )
        return records

    def build(self) -> AppWorldTraceGraphSpec:
        records = self.discover_records()
        if not records:
            raise ValueError("No AppWorld traces with both API calls and success labels found")

        return build_graph_from_records(
            records,
            min_unique_tasks_for_gate=self.min_unique_tasks_for_gate,
        )


def build_graph_from_records(
    records: list[AppWorldTraceRecord],
    *,
    min_unique_tasks_for_gate: int = 10,
    data_status: str = DATA_STATUS,
    source: str = "AppWorld labeled api_calls traces",
    gate_eligible: bool = False,
    limitations: list[str] | None = None,
) -> AppWorldTraceGraphSpec:
    """Build the sanitized graph from already selected, provenance-bearing records."""
    if not records:
        raise ValueError("At least one AppWorld trace record is required")

    cell_transitions: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    cell_runs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    cell_tasks: dict[tuple[str, str, str], dict[str, set[bool]]] = defaultdict(
        lambda: defaultdict(set)
    )
    cell_run_outcomes: dict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)
    cell_routes: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    policy_counts: dict[str, Counter] = defaultdict(Counter)

    for record in records:
        previous = "START"
        seen_run_cells = set()
        for index, action in enumerate(record.actions):
            if not validate_safe_post_attributes(action.post_attributes):
                raise ValueError("Trace contains unsafe or invalid post-step attributes")
            attributes_signature = json.dumps(
                action.post_attributes, sort_keys=True, separators=(",", ":")
            )
            key = (previous, action.category, attributes_signature)
            next_state = (
                "SUCCESS" if record.success else "FAILURE"
            ) if index == len(record.actions) - 1 else _state_after(action.category)
            cell_transitions[key][next_state] += 1
            cell_routes[key][f"{action.method} {action.route}"] += 1
            policy_counts[previous][action.category] += 1
            if key not in seen_run_cells:
                cell_runs[key].add(record.trace_id)
                cell_tasks[key][record.task_id].add(record.success)
                cell_run_outcomes[key][record.trace_id] = record.success
                seen_run_cells.add(key)
            previous = _state_after(action.category)

    cells = []
    for key in sorted(cell_transitions):
        task_outcomes = cell_tasks[key]
        successful = sum(all(outcomes) for outcomes in task_outcomes.values())
        failed = len(task_outcomes) - successful
        run_outcomes = cell_run_outcomes[key]
        successful_runs = sum(run_outcomes.values())
        failed_runs = len(run_outcomes) - successful_runs
        cells.append(
            AppWorldTraceCell(
                state=key[0],
                action=key[1],
                post_attributes=json.loads(key[2]),
                next_state_counts=dict(sorted(cell_transitions[key].items())),
                run_count=len(cell_runs[key]),
                unique_task_count=len(task_outcomes),
                successful_run_count=successful_runs,
                failed_run_count=failed_runs,
                successful_task_count=successful,
                failed_task_count=failed,
                evidence_refs=sorted(cell_runs[key]),
                representative_routes=[
                    route for route, _ in cell_routes[key].most_common(5)
                ],
            )
        )

    policy = {
        state: counts.most_common(1)[0][0]
        for state, counts in sorted(policy_counts.items())
    }
    probability_kind = "P_FINAL_TASK_SUCCESS_AFTER_VERIFIED_STATE_ACTION"
    provisional = {
        "probability_kind": probability_kind,
        "records": [record.model_dump() for record in records],
        "cells": [cell.model_dump() for cell in cells],
        "policy": policy,
    }
    graph_version = "appworld-pilot-" + hashlib.sha256(
        json.dumps(provisional, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    unique_tasks = len({record.task_id for record in records})
    default_limitations = [
        f"Only {unique_tasks} unique labeled task(s) were discovered.",
        "Repeated runs of one task remain one independent task for evidence gating.",
        "The graph is observational and is not a causal action-effect model.",
        "Request parameters and raw tool outputs are intentionally excluded; only validated bounded post-step attributes are retained.",
    ]
    if not gate_eligible:
        default_limitations.append(
            "Pilot estimates are for scoring tests only and cannot authorize tools."
        )
    return AppWorldTraceGraphSpec(
        graph_version=graph_version,
        data_status=data_status,
        source=source,
        probability_kind=probability_kind,
        records=records,
        cells=cells,
        policy=policy,
        min_unique_tasks_for_gate=min_unique_tasks_for_gate,
        gate_eligible=gate_eligible,
        limitations=limitations or default_limitations,
    )


class AppWorldTraceProbabilityProvider(ProbabilityProvider):
    """Estimate final success after a state/action step has completed."""

    def __init__(self, graph: AppWorldTraceGraphSpec, *, prior_alpha: float = 0.5,
                 draws: int = 2000, seed: int = 17,
                 hierarchical_fallback: bool = False):
        if prior_alpha <= 0 or draws < 100:
            raise ValueError("Positive Beta prior and at least 100 draws required")
        self.graph = graph
        self.prior_alpha = prior_alpha
        self.draws = draws
        self.seed = seed
        self.hierarchical_fallback = hierarchical_fallback
        self.cells = {
            (
                cell.state,
                cell.action,
                json.dumps(cell.post_attributes, sort_keys=True, separators=(",", ":")),
            ): cell
            for cell in graph.cells
        }

    def _state_action_evidence(
        self, state: str, action: str
    ) -> tuple[int, int, int, list[str]]:
        task_ids: set[str] = set()
        refs: set[str] = set()
        outcomes_by_ref: dict[str, bool] = {}
        records_by_ref = {record.trace_id: record for record in self.graph.records}
        for cell in self.graph.cells:
            if cell.state != state or cell.action != action:
                continue
            refs.update(cell.evidence_refs)
        for ref in refs:
            record = records_by_ref[ref]
            task_ids.add(record.task_id)
            outcomes_by_ref[ref] = record.success
        success = sum(outcomes_by_ref.values())
        failure = len(outcomes_by_ref) - success
        return success, failure, len(task_ids), sorted(refs)

    def _global_action_evidence(
        self, action: str
    ) -> tuple[int, int, int, list[str]]:
        task_ids: set[str] = set()
        refs: set[str] = set()
        success = failure = 0
        for record in self.graph.records:
            if any(item.category == action for item in record.actions):
                task_ids.add(record.task_id)
                refs.add(record.trace_id)
                success += int(record.success)
                failure += int(not record.success)
        return success, failure, len(task_ids), sorted(refs)

    def evaluate(self, task_state, candidate_action, verified_attributes,
                 remaining_horizon, policy_context) -> ProbabilityAssessment:
        safe_attributes = {
            key: value
            for key, value in dict(verified_attributes or {}).items()
            if key in SAFE_POST_ATTRIBUTE_KEYS
            and isinstance(value, (str, bool, int))
        }
        attributes_signature = json.dumps(
            safe_attributes, sort_keys=True, separators=(",", ":")
        )
        cell = self.cells.get((task_state, candidate_action, attributes_signature))
        fallback_level = "exact_sanitized_post_state"
        if cell is None or (
            self.hierarchical_fallback
            and cell.unique_task_count < self.graph.min_unique_tasks_for_gate
        ):
            if self.hierarchical_fallback:
                success, failure, unique_tasks, evidence_refs = self._state_action_evidence(
                    task_state, candidate_action
                )
                if unique_tasks >= self.graph.min_unique_tasks_for_gate:
                    cell = None
                    fallback_level = "sanitized_state_action_without_post_attributes"
                else:
                    success, failure, unique_tasks, evidence_refs = self._global_action_evidence(
                        candidate_action
                    )
                    if unique_tasks >= self.graph.min_unique_tasks_for_gate:
                        cell = None
                        fallback_level = "global_sanitized_action_category"
                    else:
                        success = failure = 0
                        unique_tasks = 0
                        evidence_refs = []
            else:
                success = failure = 0
                unique_tasks = 0
                evidence_refs = []
        else:
            success = cell.successful_run_count
            failure = cell.failed_run_count
            unique_tasks = cell.unique_task_count
            evidence_refs = cell.evidence_refs
        if success + failure == 0:
            return ProbabilityAssessment(
                calculable=False,
                data_status=self.graph.data_status,
                graph_version=self.graph.graph_version,
                policy_version=f"{self.graph.graph_version}:observed-policy",
                remaining_horizon=remaining_horizon,
                reason="No sufficiently supported state/action or action-category evidence",
                gate_eligible=False,
            )
        rng = random.Random(self.seed)
        samples = sorted(
            rng.betavariate(success + self.prior_alpha, failure + self.prior_alpha)
            for _ in range(self.draws)
        )
        mean = sum(samples) / len(samples)
        return ProbabilityAssessment(
            calculable=True,
            probability_kind=self.graph.probability_kind,
            probabilities={"SUCCESS": mean, "PARTIAL": 0.0, "FAILURE": 1 - mean},
            success_interval=(
                samples[int(0.025 * self.draws)],
                samples[min(self.draws - 1, int(0.975 * self.draws))],
            ),
            sample_count=success + failure,
            unique_task_count=unique_tasks,
            denominator=success + failure,
            prior={"beta_alpha": self.prior_alpha, "beta_beta": self.prior_alpha},
            posterior={"successful_runs": success, "failed_runs": failure},
            fallback_level=fallback_level,
            evidence_refs=evidence_refs,
            data_status=self.graph.data_status,
            graph_version=self.graph.graph_version,
            policy_version=f"{self.graph.graph_version}:observed-policy",
            remaining_horizon=remaining_horizon,
            reason=(
                "Empirical run-level final-success estimate from repeated train traces"
                if self.graph.data_status == EMPIRICAL_DATA_STATUS
                else "Pilot trace estimate"
            ),
            gate_eligible=self.graph.gate_eligible,
            is_mock=False,
        )


def write_appworld_graph(graph: AppWorldTraceGraphSpec, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(graph.model_dump_json(indent=2), encoding="utf-8")


def score_graph(graph: AppWorldTraceGraphSpec, *, draws: int = 2000,
                seed: int = 17) -> list[dict[str, Any]]:
    provider = AppWorldTraceProbabilityProvider(graph, draws=draws, seed=seed)
    rows = []
    for cell in graph.cells:
        assessment = provider.evaluate(cell.state, cell.action, {}, 1, graph.policy)
        rows.append({
            "state": cell.state,
            "action": cell.action,
            "assessment": assessment.model_dump(),
            "decision": "OBSERVE_ONLY_INSUFFICIENT_GATE_EVIDENCE",
        })
    return rows
