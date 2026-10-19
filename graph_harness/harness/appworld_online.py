from __future__ import annotations

import ast
from dataclasses import dataclass, field as dataclass_field
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import traceback
from types import SimpleNamespace
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field
from dataset_selection import load_dataset_ids, select_online_tasks

from harness.appworld import (
    AppWorldAction,
    AppWorldTraceGraphSpec,
    AppWorldTraceProbabilityProvider,
    AppWorldTraceRecord,
    EMPIRICAL_DATA_STATUS,
    SAFE_POST_ATTRIBUTE_KEYS,
    _classify,
    _route_template,
    _state_after,
    build_graph_from_records,
    derive_safe_post_attributes,
    validate_safe_post_attributes,
)
from harness.graph_ready import GraphReadyBuild
from harness.graph_guidance import format_guidance
from harness.graph_guidance_provider import GraphGuidanceProvider, query_graph_guidance
from harness.failure_ledger import (
    OperationContext,
    TaskFailureLedger,
    classify_failure,
    operation_contexts,
)
from harness.proposal_contracts import (compile_proposal_contracts, check_output,
    check_write_fields, finding_status, completion_protocol_guidance,
    failures as proposal_contract_failures)
from harness.task_targets import (CompletionVerifier, TaskTargetTracker, verified_read_payload,
    CompletionEvidencePending, final_semantic_preflight)
from harness.tool_evidence import ToolEvidenceCapture, RuntimeApiBudgetError
from harness.execution_paths import CheckedActions, compile_paths
from harness.runtime_checkpoints import RuntimeCheckpoints
from harness.task_contracts import sequential_stop_conditions


PROHIBITED_IMPORTS = {
    "os", "pathlib", "requests", "httpx", "socket", "subprocess", "shutil", "sys"
}
MAX_API_CALLS_PER_STEP = 5


class ModelSettings(BaseModel):
    api_key: str
    base_url: str
    model: str
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = 20
    min_p: float | None = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float | None = 1.0
    enable_thinking: bool = True
    generation_seed: int | None = None


class CodeProposal(BaseModel):
    code: str
    raw_content: str
    reasoning_sha256: str | None = None


class StepDecision(BaseModel):
    allowed: bool
    reason: str
    gate_mode: str = "enforce"
    risk_tier: str = "unknown"
    category: str | None = None
    probability_mean: float | None = None
    probability_interval: tuple[float, float] | None = None
    unique_task_count: int = 0
    fallback_level: str | None = None
    graph_version: str


class TaskRunSummary(BaseModel):
    dataset: str
    task_id: str
    started_at: str
    finished_at: str
    duration_seconds: float = Field(ge=0)
    success: bool = False
    completed: bool = False
    evaluation_pass_count: int = Field(default=0, ge=0)
    evaluation_fail_count: int = Field(default=0, ge=0)
    evaluation_total_count: int = Field(default=0, ge=0)
    evaluation_pass_rate: float | None = Field(default=None, ge=0, le=1)
    evaluation_failure_reasons: list[str] = Field(default_factory=list)
    evaluation_failure_details: list[dict[str, str]] = Field(default_factory=list)
    steps_proposed: int = 0
    steps_executed: int = 0
    api_calls_executed: int = 0
    rejected_steps: int = 0
    checkpointed_steps: int = 0
    precheck_rejections: int = 0
    execution_failures: int = 0
    no_mutation_failures: int = 0
    postcondition_failures: int = 0
    repeated_decision_failures: int = 0
    failure_classification_count: int = 0
    failure_evidence_count: int = 0
    active_failure_constraint_count: int = 0
    rollback_attempts: int = 0
    rollback_successes: int = 0
    rollback_failures: int = 0
    rollback_budget_exhausted: bool = False
    decision_rollback_budget_exhausted: bool = False
    task_terminated_due_to_rollback_budget: bool = False
    rollback_cause_counts: dict[str, int] = Field(default_factory=dict)
    decisions_opened: int = 0
    explicit_decisions_opened: int = 0
    implicit_decisions_opened: int = 0
    decision_commits: int = 0
    action_rollbacks: int = 0
    decision_rollbacks: int = 0
    agent_requested_decision_rollbacks: int = 0
    automatic_decision_rollbacks: int = 0
    repeated_rolled_back_proposals: int = 0
    completion_coverage_checks: int = 0
    completion_coverage_rejections: int = 0
    completion_coverage_advisories: int = 0
    completion_format_checks: int = 0
    completion_format_rejections: int = 0
    semantic_write_checks: int = 0
    semantic_write_rejections: int = 0
    semantic_completion_checks: int = 0
    semantic_completion_rejections: int = 0
    harness_metadata_advisories: int = 0
    post_step_probability_checks: int = 0
    guidance_queries: int = 0
    guidance_hits: int = 0
    guidance_errors: int = 0
    probability_ranked_guidance_hits: int = 0
    mean_success_guidance_score: float | None = Field(default=None, ge=0, le=1)
    stop_reason: str
    appworld_output: str
    harness_audit: str
    error: str | None = None
    technical_failure_stage: str | None = None
    technical_failure_location: str | None = None
    task_target_tracking_enabled: bool = False
    completion_verification_enabled: bool = False
    task_targets_created: int = 0
    task_target_constraints: int = 0
    target_provenance_user_instruction: int = 0
    target_provenance_verified_observation: int = 0
    target_provenance_verified_state: int = 0
    candidate_sets_seen: int = 0
    candidate_sets_promoted: int = 0
    completion_verification_checks: int = 0
    completion_satisfied: int = 0
    completion_violated: int = 0
    completion_unknown: int = 0
    completion_blocks: int = 0
    missing_target_count: int = 0
    unexpected_target_count: int = 0
    quantity_mismatch_count: int = 0
    field_mismatch_count: int = 0
    constraint_violation_count: int = 0
    completion_verification_unknown: bool = False
    task_contract_parsed_clauses: int = 0
    task_contract_unresolved_clauses: int = 0
    target_evidence_captured_reads: int = 0
    target_evidence_rejected_batches: int = 0
    completion_feedback_sent: int = 0
    completion_feedback_deduplicated: int = 0
    partial_collection_reads: int = 0
    complete_paginated_reads: int = 0
    subject_scope_conflicts: int = 0
    completion_checks: int = 0
    completion_final_status: str | None = None
    completion_unknown_reason_counts: dict[str, int] = Field(default_factory=dict)
    completion_evidence_pending_episodes: int = 0
    completion_evidence_reads: int = 0
    completion_evidence_budget_exhausted: int = 0
    repair_episodes: list[dict[str, Any]] = Field(default_factory=list)
    repair_started: int = 0
    repair_resolved: int = 0
    confirmed_violation_episodes: int = 0
    rollback_scope_feedback_events: int = 0
    final_preflight_checks: int = 0
    final_preflight_violations: int = 0
    final_preflight_unknown: int = 0
    final_reconciliation_checks: int = 0


def evaluation_test_counts(tracker: object) -> tuple[int, int, int, float | None]:
    """Extract AppWorld's per-requirement result without exposing test details."""

    payload: Mapping[str, Any] = {}
    to_dict = getattr(tracker, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
        except Exception:
            value = None
        if isinstance(value, Mapping):
            payload = value

    def count(name: str, collection_name: str) -> int | None:
        value = getattr(tracker, name, None)
        if type(value) is int and value >= 0:
            return value
        value = payload.get(name)
        if type(value) is int and value >= 0:
            return value
        collection = payload.get(collection_name, getattr(tracker, collection_name, None))
        return len(collection) if isinstance(collection, (list, tuple)) else None

    passed = count("pass_count", "passes")
    failed = count("fail_count", "failures")
    if failed is None:  # Compatibility with older/custom tracker payloads.
        failed = count("fail_count", "fails")
    passed = passed if passed is not None else 0
    failed = failed if failed is not None else 0
    total = passed + failed
    return passed, failed, total, (passed / total if total else None)


def evaluation_failure_reasons(tracker: object) -> list[str]:
    """Return bounded official failure requirements for the task result report.

    These are produced only after AppWorld has evaluated the finished sandbox.  They
    make semantic and answer-format failures visible in task_results.jsonl instead of
    reducing every failure to a boolean.  Traces and ground-truth values are excluded.
    """
    failures = getattr(tracker, "failures", None)
    if not isinstance(failures, (list, tuple)):
        to_dict = getattr(tracker, "to_dict", None)
        try:
            payload = to_dict() if callable(to_dict) else {}
        except Exception:
            payload = {}
        failures = payload.get("failures", payload.get("fails", [])) if isinstance(
            payload, Mapping
        ) else []
    reasons: list[str] = []
    for failure in failures if isinstance(failures, (list, tuple)) else []:
        requirement = failure.get("requirement") if isinstance(failure, Mapping) else None
        if isinstance(requirement, str):
            bounded = " ".join(requirement.split())[:500]
            if bounded and bounded not in reasons:
                reasons.append(bounded)
    return reasons


def evaluation_failure_details(tracker: object) -> list[dict[str, str]]:
    """Classify official failures so answer-format errors are explicit."""
    details = []
    for requirement in evaluation_failure_reasons(tracker):
        is_ignored_order = bool(re.search(r"\bignore\s+order\b", requirement, re.I))
        is_answer_format = bool(
            not is_ignored_order
            and re.search(r"\b(?:order|format|schema|structure)\b", requirement, re.I)
            and re.search(r"\b(?:answer|output|month|months|key|keys)\b", requirement, re.I)
        )
        if is_answer_format:
            error_type = "answer_format_error"
        elif re.search(r"\banswers?\s+match\b", requirement, re.I):
            error_type = "answer_mismatch"
        else:
            error_type = "task_requirement_failure"
        details.append({"error_type": error_type, "requirement": requirement})
    return details


def read_env(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_model_settings(path: str | Path) -> ModelSettings:
    values = read_env(path)
    missing = [
        key for key in ("TAU3_API_KEY", "TAU3_BASE_URL", "TAU3_MODEL")
        if not values.get(key)
    ]
    if missing:
        raise ValueError(f"Missing model settings: {', '.join(missing)}")

    def optional_float(name: str, default: float | None) -> float | None:
        value = values.get(name)
        return default if value in (None, "", "auto") else float(value)

    def optional_int(name: str, default: int | None) -> int | None:
        value = values.get(name)
        return default if value in (None, "", "auto") else int(value)

    return ModelSettings(
        api_key=values["TAU3_API_KEY"],
        base_url=values["TAU3_BASE_URL"],
        model=values["TAU3_MODEL"].removeprefix("openai/"),
        temperature=optional_float("TAU3_TEMPERATURE", 1.0),
        top_p=optional_float("TAU3_TOP_P", 0.95) or 0.95,
        top_k=optional_int("TAU3_TOP_K", 20),
        min_p=optional_float("TAU3_MIN_P", 0.0),
        presence_penalty=optional_float("TAU3_PRESENCE_PENALTY", 0.0) or 0.0,
        repetition_penalty=optional_float("TAU3_REPETITION_PENALTY", 1.0),
        generation_seed=optional_int("TAU3_GENERATION_SEED", None),
        enable_thinking=values.get("TAU3_ENABLE_THINKING", "true").lower()
        in {"1", "true", "yes", "on"},
    )



def load_empirical_trace_export(
    path: str | Path,
    *,
    expected_tasks: int = 20,
    expected_runs_per_task: int = 5,
) -> list[AppWorldTraceRecord]:
    """Load a strictly sanitized 20-task x 5-run train export.

    The loader rejects reference/ground-truth data, incomplete repeat groups, duplicate
    runs, and unexpected fields that could contain API parameters or tool outputs.
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "2.0":
        raise ValueError(
            "Trace export schema 2.0 is required; rebuild it with safe post-step attributes"
        )
    if payload.get("source_kind") != "evaluated_appworld_train_runs":
        raise ValueError(
            "Trace export must contain evaluated AppWorld train runs; "
            "ground-truth/reference traces are not accepted"
        )
    selection = payload.get("selection") or {}
    if selection.get("task_count") != expected_tasks:
        raise ValueError(f"Trace export must contain {expected_tasks} selected tasks")
    if selection.get("runs_per_task") != expected_runs_per_task:
        raise ValueError(
            f"Trace export must contain {expected_runs_per_task} runs per task"
        )
    privacy = payload.get("privacy") or {}
    for field in (
        "request_parameters_included", "tool_outputs_included",
        "passwords_or_tokens_included",
    ):
        if privacy.get(field) is not False:
            raise ValueError(f"Unsafe or unverified trace export privacy field: {field}")
    if privacy.get("derived_post_attributes_included") is not True:
        raise ValueError("Trace export must include verified derived post-step attributes")

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("Trace export records must be a list")
    allowed_record_fields = {
        "trace_id", "task_id", "experiment_id", "success", "actions",
        "source_sha256",
    }
    allowed_action_fields = {"category", "method", "route", "post_attributes"}
    records: list[AppWorldTraceRecord] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Trace record {index} is not an object")
        unexpected = set(raw_record) - allowed_record_fields
        if unexpected:
            raise ValueError(f"Unexpected trace fields at record {index}: {sorted(unexpected)}")
        for action in raw_record.get("actions") or []:
            if not isinstance(action, dict):
                raise ValueError(f"Trace record {index} contains an invalid action")
            unexpected_action = set(action) - allowed_action_fields
            if unexpected_action:
                raise ValueError(
                    f"Unexpected action fields at record {index}: {sorted(unexpected_action)}"
                )
            post_attributes = action.get("post_attributes")
            if not isinstance(post_attributes, dict):
                raise ValueError(f"Trace record {index} action lacks post_attributes")
            unexpected_attributes = set(post_attributes) - SAFE_POST_ATTRIBUTE_KEYS
            if unexpected_attributes:
                raise ValueError(
                    f"Unsafe post-step attributes at record {index}: "
                    f"{sorted(unexpected_attributes)}"
                )
            if not validate_safe_post_attributes(post_attributes):
                raise ValueError(f"Invalid or unsafe post-step attributes at record {index}")
        record = AppWorldTraceRecord.model_validate(raw_record)
        if "ground_truth" in record.experiment_id.lower():
            raise ValueError("Ground-truth/reference traces are not accepted")
        records.append(record)

    expected_total = expected_tasks * expected_runs_per_task
    if len(records) != expected_total:
        raise ValueError(f"Trace export must contain exactly {expected_total} records")
    if len({record.trace_id for record in records}) != len(records):
        raise ValueError("Trace export contains duplicate trace IDs")
    counts: dict[str, int] = {}
    for record in records:
        counts[record.task_id] = counts.get(record.task_id, 0) + 1
    if len(counts) != expected_tasks or set(counts.values()) != {expected_runs_per_task}:
        raise ValueError(
            f"Trace export must contain exactly {expected_runs_per_task} runs for each "
            f"of {expected_tasks} tasks"
        )
    return records


def build_empirical_graph(
    records: list[AppWorldTraceRecord], *, min_unique_tasks: int = 3
) -> AppWorldTraceGraphSpec:
    return build_graph_from_records(
        records,
        min_unique_tasks_for_gate=min_unique_tasks,
        data_status=EMPIRICAL_DATA_STATUS,
        source="AppWorld evaluated train traces: 20 tasks x 5 model runs",
        gate_eligible=True,
        limitations=[
            "The graph contains 5 evaluated model runs for each of 20 train tasks.",
            "Beta posteriors use run-level final success/failure outcomes.",
            "A completed state/action plus bounded derived attributes is a coarse proxy for post-step state; raw tool output values are not exported.",
            "Only bounded output shape, count bucket, error class, and authentication status features are retained.",
            "Unique-task support is checked separately so repeats are not treated as independent tasks.",
            "The estimates are observational and model/service/settings specific.",
            "No test task, test trace, request parameter, token, password, or tool output enters the graph.",
            "Gate authority is limited to disposable AppWorld sandbox tasks in this pilot.",
        ],
    )



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


def normalize_route_path(route: str) -> str:
    """Normalize a route without erasing descriptive placeholder semantics."""
    path = urlsplit(str(route)).path or "/"
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1:
        path = path.rstrip("/")
    return path or "/"


def match_route(expected_template: str, observed_route: str) -> tuple[bool, str | None]:
    """Match one observed route against an expected route template.

    Placeholders are wildcards only on the expected side.  This preserves fixed
    enum/path segments while allowing AppWorld's tracker to expose either a concrete
    value or a generic ``{id}`` for an expected parameter.
    """
    expected = normalize_route_path(expected_template)
    observed = normalize_route_path(observed_route)
    expected_parts = [part for part in expected.split("/") if part]
    observed_parts = [part for part in observed.split("/") if part]
    if len(expected_parts) != len(observed_parts):
        return False, (
            f"path segment count differs: expected {len(expected_parts)}, "
            f"observed {len(observed_parts)}"
        )
    for index, (expected_part, observed_part) in enumerate(
        zip(expected_parts, observed_parts), start=1
    ):
        if re.fullmatch(r"\{[^/{}]+\}", expected_part):
            if not observed_part:
                return False, f"observed path segment {index} is empty"
            continue
        if expected_part != observed_part:
            return False, (
                f"fixed path segment {index} differs: expected {expected_part!r}, "
                f"observed {observed_part!r}; expected route {expected!r}, "
                f"observed route {observed!r}"
            )
    return True, None


def _bounded_iteration_count(
    node: ast.AST,
    namespace: Mapping[str, Any] | None = None,
) -> int | None:
    """Return a statically provable loop length, otherwise None."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts)
    if isinstance(node, ast.Name) and namespace is not None:
        value = namespace.get(node.id)
        if type(value) in {list, tuple}:
            return len(value)
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "range" or node.keywords:
        return None
    values: list[int] = []
    for argument in node.args:
        if not isinstance(argument, ast.Constant) or type(argument.value) is not int:
            return None
        values.append(argument.value)
    if not 1 <= len(values) <= 3:
        return None
    try:
        return len(range(*values))
    except (TypeError, ValueError):
        return None


def inspect_code(
    code: str,
    api_docs: Mapping[str, Any],
    namespace: Mapping[str, Any] | None = None,
) -> tuple[list[AppWorldAction], list[str]]:
    """Statically identify a bounded direct API batch and reject bypass patterns."""
    failures: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [], [f"Python syntax error: {exc.msg}"]
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in node.names] if isinstance(
                node, ast.Import
            ) else [(node.module or "").split(".")[0]]
            if PROHIBITED_IMPORTS.intersection(modules):
                failures.append("System/network module import is outside AppWorld scope")

    calls: list[tuple[ast.Call, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if chain and chain[0] == "apis":
                calls.append((node, chain))
    calls.sort(key=lambda item: (getattr(item[0], "lineno", 0),
                                 getattr(item[0], "col_offset", 0)))

    direct_api_name_nodes = set()
    for call, chain in calls:
        current: ast.AST = call.func
        while isinstance(current, ast.Attribute):
            current = current.value
        direct_api_name_nodes.add(current)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "apis" and node not in direct_api_name_nodes:
            failures.append("Indirect or stored access to apis is not allowed")
            break

    unsupported = (ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp,
                   ast.DictComp, ast.GeneratorExp)
    deferred = (ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                ast.ClassDef, ast.Match, ast.BoolOp, ast.IfExp)
    for call, _ in calls:
        loops = 0
        current = parent.get(call)
        while current is not None:
            if isinstance(current, deferred):
                failures.append("AppWorld API calls cannot be conditional or deferred in "
                                "try/function/lambda/short-circuit expressions")
            if isinstance(current, unsupported):
                failures.append("AppWorld API loops must use one statically bounded for-loop")
            if isinstance(current, ast.For):
                loops += 1
            if isinstance(current, ast.If) and call in ast.walk(current.test):
                failures.append("Read the API result before using it as a branch condition")
            current = parent.get(current)
        if loops > 1:
            failures.append("Nested AppWorld API loops are not allowed")
    actions_by_site = {}
    for call, chain in calls:
        if len(chain) != 3:
            failures.append("AppWorld calls must use direct apis.<app>.<api>(...) syntax")
            continue
        _, app_name, api_name = chain
        try:
            doc = api_docs[app_name][api_name]
            method = str(doc["method"]).upper()
            route = normalize_route_path(str(doc["path"]))
        except (KeyError, TypeError):
            failures.append(f"Unknown AppWorld API: {app_name}.{api_name}")
            continue
        actions_by_site[(call.lineno, call.col_offset)] = AppWorldAction(
            category=_classify(method, route), method=method, route=route)
    # Only literals are interpreted; generated conditions and expressions are not evaluated.
    bounds = {}

    def collect_bounds(statements, values):
        values = dict(values)
        for statement in statements:
            if isinstance(statement, ast.For):
                bounds[id(statement.iter)] = _bounded_iteration_count(statement.iter, values)
            if isinstance(statement, ast.Assign):
                for name in statement.targets:
                    if isinstance(name, ast.Name):
                        values.pop(name.id, None)
                        try:
                            values[name.id] = ast.literal_eval(statement.value)
                        except (ValueError, TypeError):
                            pass
            elif isinstance(statement, (ast.For, ast.If)):
                collect_bounds(statement.body, values)
                collect_bounds(statement.orelse, values)
                for child in ast.walk(statement):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        values.pop(child.id, None)
        return values

    collect_bounds(tree.body, namespace or {})
    try:
        paths = compile_paths(tree, actions_by_site,
                              lambda node: bounds.get(id(node)),
                              max_calls=MAX_API_CALLS_PER_STEP)
        actions = CheckedActions([[actions_by_site[site] for site in path] for path in paths])
    except ValueError as exc:
        failures.append(str(exc))
        actions = list(actions_by_site.values())
    if len(actions) > 1 and any(
        action.category == "COMPLETE_TASK" for action in actions
    ):
        failures.append("Task completion must be the only AppWorld API call in its step")
    return actions, list(dict.fromkeys(failures))


_ACTION_REQUEST_PATTERN = re.compile(
    r"\b(?:add|book|buy|cancel|capitalize|change|convert|correct|create|delete|"
    r"edit|follow|format|invite|like|mark|modify|move|normalize|order|pay|post|"
    r"purchase|remove|rename|request|reserve|schedule|send|share|transform|"
    r"transfer|update|upload)\b",
    flags=re.IGNORECASE,
)








def _completion_answer_node(code: str) -> ast.AST | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _attribute_chain(node.func) != ["apis", "supervisor", "complete_task"]:
            continue
        for keyword in node.keywords:
            if keyword.arg == "answer":
                return keyword.value
        return node.args[0] if node.args else ast.Constant(value=None)
    return None


def _safe_answer_value(node: ast.AST, namespace: Mapping[str, Any] | None) -> tuple[bool, Any]:
    """Resolve a completion answer without evaluating model-supplied code."""
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        pass
    if isinstance(node, ast.Name) and namespace is not None and node.id in namespace:
        value = namespace[node.id]
        if value is None or isinstance(value, (str, int, float, list, tuple, dict)):
            return True, value
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        found, value = _safe_answer_value(node.args[0], namespace)
        if not found:
            return False, None
        if isinstance(node.func, ast.Name) and node.func.id in {"str", "repr"}:
            return True, str(value) if node.func.id == "str" else repr(value)
        if _attribute_chain(node.func) == ["json", "dumps"]:
            return True, json.dumps(value, ensure_ascii=False)
    return False, None












def completion_answer_failures(code, instruction, namespace=None, api_docs=None):
    contracts = compile_proposal_contracts(instruction, api_docs)
    return proposal_contract_failures([f for f in check_output(code, contracts, api_docs, namespace)
                                      if f.check == "ANSWER_PRESENCE"])


def completion_format_failures(code, instruction, namespace=None, api_docs=None):
    contracts = compile_proposal_contracts(instruction, api_docs)
    return proposal_contract_failures([f for f in check_output(code, contracts, api_docs, namespace)
                                      if f.check == "ANSWER_ORDER"])


def semantic_write_failures(code, instruction, namespace=None, api_docs=None):
    return proposal_contract_failures(check_write_fields(code,
        compile_proposal_contracts(instruction, api_docs), api_docs, namespace))


def semantic_completion_failure(
    actions: list[AppWorldAction],
    instruction: str,
    *,
    verified_write_count: int,
    last_verified_write_step: int | None,
    last_verified_read_step: int | None,
) -> str | None:
    """Require a post-write verification read for destructive multi-target tasks."""
    if not any(action.category == "COMPLETE_TASK" for action in actions):
        return None
    destructive = bool(re.search(r"\b(?:delete|remove)\b", instruction, re.I))
    multi_target = bool(
        verified_write_count > 1
        or re.search(r"\b(?:all|each|every|tasks|items|records)\b", instruction, re.I)
    )
    if not destructive or not multi_target or last_verified_write_step is None:
        return None
    if (
        last_verified_read_step is not None
        and last_verified_read_step > last_verified_write_step
    ):
        return None
    return (
        "Semantic write error: this destructive multi-target task has no verified read "
        "after its most recent write. Re-query the affected collection, verify the "
        "remaining and removed target IDs and the fields of retained records against the "
        "user instruction, correct any mismatch, then retry complete_task."
    )


def bulk_completion_requested(instruction: str) -> bool:
    """Identify an explicit request to apply an action to a whole collection."""
    return bool(
        _ACTION_REQUEST_PATTERN.search(instruction)
        and re.search(r"\b(?:all|each|every)\b", instruction, flags=re.IGNORECASE)
    )


def explicit_instruction_target_count(instruction: str) -> int | None:
    """Count an explicit list or an explicit number of target objects."""
    listed = sum(
        bool(re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", line))
        for line in instruction.splitlines()
    )
    if listed >= 2:
        return listed
    numbered = re.search(
        r"\b(\d{1,3})\s+(?:distinct\s+)?"
        r"(?:items?|records?|tasks?|notes?|files?|contacts?|songs?|messages?|"
        r"entries|entities|objects?|recipients?|targets?)\b",
        instruction,
        flags=re.IGNORECASE,
    )
    if numbered:
        value = int(numbered.group(1))
        return value if value > 0 else None
    return None


def normalize_execution_output(output: Any) -> str:
    """Normalize AppWorld's execution result at the runtime boundary.

    AppWorld normally returns rendered text, but local-only code and compatibility
    versions may return ``None``.  Treat that as an empty successful output instead
    of letting downstream verification fail while hashing, measuring, or parsing it.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def collection_result_count(output: Any) -> int | None:
    """Count a returned collection ephemerally without retaining its values."""
    text = normalize_execution_output(output).strip()
    if not text or text.startswith("Execution failed. Traceback:"):
        return None
    if text.startswith("Munch(") and text.endswith(")"):
        text = text[6:-1]
    value: Any
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return None


def _singular_collection_term(value: str) -> str:
    """Return a small, deterministic singular form for route/goal matching."""
    term = value.strip().lower().replace("-", "_")
    if term.endswith("ies") and len(term) > 3:
        return term[:-3] + "y"
    if term.endswith("s") and not term.endswith("ss") and len(term) > 1:
        return term[:-1]
    return term


@dataclass(frozen=True)
class CoverageEvidence:
    verified_target_count: int | None
    candidate_collection_count: int | None
    source: str
    confidence: str
    gate_enforced: bool
    reason: str


def target_collection_evidence(
    instruction: str,
    actions: list[AppWorldAction],
    output: Any,
) -> CoverageEvidence:
    """Record collection counts as candidates, never as completion proof.

    Even a matching route noun cannot establish filters, entity identity or the
    number of calls needed to process the collection. Actual task-bound targets
    and current-state predicates are handled by TaskTargetTracker separately.
    """
    candidate_count = collection_result_count(output)
    if len(actions) != 1 or actions[0].category != "READ_RESOURCE":
        return CoverageEvidence(
            None, candidate_count, "non_read_result", "none", False,
            "Only a verified read can provide collection-size evidence.",
        )
    target_matches = re.findall(
        r"\b(?:all|each|every)\s+(?:of\s+)?"
        r"(?:(?:my|the|these|those|your)\s+)?([A-Za-z][A-Za-z0-9_-]*)",
        instruction,
        flags=re.IGNORECASE,
    )
    target_terms = {_singular_collection_term(term) for term in target_matches}
    if not target_terms:
        return CoverageEvidence(
            None, candidate_count, "candidate_collection", "low", False,
            "The instruction does not identify this returned collection as the target set.",
        )
    route_terms: set[str] = set()
    for piece in _route_template(actions[0].route).split("/"):
        if not piece or piece == "{id}":
            continue
        route_terms.update(
            _singular_collection_term(part)
            for part in piece.split("_") if part
        )
    if target_terms.isdisjoint(route_terms):
        return CoverageEvidence(
            None, candidate_count, "candidate_collection", "low", False,
            "The read route names a broader or different collection than the instruction target.",
        )
    route = actions[0].route.lower()
    if any(token in route for token in ("/search", "search_", "/query")):
        return CoverageEvidence(
            None, candidate_count, "search_result_collection", "low", False,
            "Search results may be partial or require additional filtering.",
        )
    return CoverageEvidence(
        None,
        candidate_count,
        "candidate_matching_collection",
        "low" if candidate_count is not None else "none",
        False,
        "A matching collection route does not establish required target identity, "
        "filter coverage or one-write-per-entity semantics; count is advisory only.",
    )


def target_collection_result_count(
    instruction: str,
    actions: list[AppWorldAction],
    output: str,
) -> int | None:
    """Compatibility wrapper returning only enforceable target counts."""
    evidence = target_collection_evidence(instruction, actions, output)
    return evidence.verified_target_count if evidence.gate_enforced else None


def completion_coverage_failure(
    actions: list[AppWorldAction],
    instruction: str,
    target_count: int | None,
    verified_write_count: int,
) -> str | None:
    """Compatibility hook: scalar counts cannot prove a completion failure.

    Keep the old entry point for callers, but never return a blocking reason.
    Use completion_coverage_advisory for diagnostics and CompletionVerifier for
    confirmed task-target violations.
    """
    return None


def completion_coverage_advisory(
    actions: list[AppWorldAction],
    instruction: str,
    target_count: int | None,
    verified_write_count: int,
) -> str | None:
    """Describe uncertain count evidence without claiming missing targets."""
    if not any(action.category == "COMPLETE_TASK" for action in actions):
        return None
    if not (
        bulk_completion_requested(instruction)
        or explicit_instruction_target_count(instruction) is not None
    ) or target_count is None:
        return None
    if verified_write_count >= target_count:
        return None
    return (
        f"Count-only advisory: {verified_write_count} verified write call(s) and "
        f"{target_count} instruction-listed item(s). API call count is not processed "
        "entity count; this comparison cannot confirm a missing target and does "
        "not block completion or consume a rejection budget."
    )


def sequential_batch_failures(code, actions, instruction, api_docs):
    """Prevent overshooting explicit public stop conditions; independent batches stay legal."""
    conditions = sequential_stop_conditions(instruction, api_docs)
    if not conditions:
        return []
    paths = actions.paths if isinstance(actions, CheckedActions) else (tuple(actions),)
    for direction, entity, app in conditions:
        routes = set()
        for name, doc in api_docs.get(app, {}).items():
            if not isinstance(doc, Mapping):
                continue
            words = name.split("_")
            description = str(doc.get("description", "")).lower()
            if ((direction in words and (entity in words or entity + "s" in words))
                    or re.search(r"\b" + direction + r"\s+" + re.escape(entity) + r"\b", description)):
                routes.add((str(doc.get("method", "")).upper(),
                            normalize_route_path(str(doc.get("path", "")))))
        if any(sum((a.method, a.route) in routes for a in path) > 1 for path in paths):
            return ["Sequential stop-condition check required: advance one item and inspect "
                    "its result before advancing again. Independent operations may still be batched."]
    return []


def extract_code(content: str) -> str:
    blocks = re.findall(r"```python\s*\n(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if len(blocks) != 1:
        raise ValueError(f"Expected exactly one Python code block, found {len(blocks)}")
    code = blocks[0].strip()
    if not code:
        raise ValueError("The Python code block is empty")
    return code


class OpenAICompatibleCodeAgent:
    def __init__(self, settings: ModelSettings, *, max_output_tokens: int = 8192,
                 timeout_seconds: int = 900, retries: int = 2):
        self.settings = settings
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.messages: list[dict[str, Any]] = []

    def initialize(self, world: Any, prompt_path: str | Path) -> None:
        template = Path(prompt_path).read_text(encoding="utf-8").lstrip()
        supervisor = world.task.supervisor
        replacements = {
            "{{ main_user.first_name }}": str(supervisor.first_name),
            "{{ main_user.last_name }}": str(supervisor.last_name),
            "{{ main_user.email }}": str(supervisor.email),
            "{{ main_user.phone_number }}": str(supervisor.phone_number),
            "{{ instruction }}": world.task.instruction,
            "{{ app_descriptions }}": json.dumps(
                world.task.app_descriptions, ensure_ascii=False, indent=2
            ),
        }
        for old, new in replacements.items():
            template = template.replace(old, new)
        self.messages = self._text_to_messages(template)
        self.messages.insert(0, {"role": "system", "content": (
            "Operate only inside the disposable AppWorld task sandbox. Return exactly one "
            "python code block per turn and make at most five direct apis.<app>.<api> calls. "
            "A batch may use explicit calls or one for-loop over a literal collection or "
            "constant range, with no more than five total calls. "
            "If branches and break/continue are allowed inside bounded loops; every possible "
            "path must remain within five calls. Read an API result before testing it. "
            "Do not put API calls in try/functions/lambdas or short-circuit expressions. "
            "For 'until' or 'first matching' tasks, advance one item, inspect the new state, "
            "and test the stop condition before advancing again. Fully enumerate paginated "
            "collections before comparing whole sets; a search result is only a candidate. "
            "For a state-changing proposal, you may assign a literal HARNESS_META dictionary in the same "
            "code block. Its only fields are node_type, decision_id, action_id, and "
            "recovery_intent. Use node_type='decision_action' for the first action of a new "
            "plan and node_type='action' with the same decision_id for later actions. Use "
            "recovery_intent='retry_action' to keep the plan or 'replan_decision' to abandon "
            "it. A replan_decision value is only a request: keep the active decision_id, "
            "first try a different parameter or resource, and let the Harness decide whether "
            "the decision checkpoint may be restored. Never place card data, credentials, "
            "or checkpoint IDs in HARNESS_META. "
            "Validate resource "
            "expiry and other validity fields against the current task state before use. "
            "Do not use indirect API aliases or system/network/file modules. "
            + completion_protocol_guidance(world.task.api_docs) +
            "When an information task specifies an output order or format, preserve that "
            "order exactly in the final answer; before complete_task, keep the answer in a "
            "plain string or dictionary variable so the Harness can validate it. For an "
            "update, change only fields explicitly requested by the instruction. After the "
            "last delete or other destructive multi-target write, re-read the affected "
            "collection and verify target IDs and retained fields before complete_task."
        )})

    @staticmethod
    def _text_to_messages(text: str) -> list[dict[str, str]]:
        matches = list(re.finditer(r"^(USER|ASSISTANT|SYSTEM):\s*$", text,
                                   flags=re.MULTILINE | re.IGNORECASE))
        messages = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            messages.append({
                "role": match.group(1).lower(),
                "content": text[start:end].strip(),
            })
        if not messages:
            raise ValueError("AppWorld prompt contains no role-delimited messages")
        return messages

    def _request(self) -> dict[str, Any]:
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": self.messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "presence_penalty": self.settings.presence_penalty,
            "max_tokens": self.max_output_tokens,
            "chat_template_kwargs": {"enable_thinking": self.settings.enable_thinking},
        }
        if self.settings.top_k is not None:
            body["top_k"] = self.settings.top_k
        if self.settings.min_p is not None:
            body["min_p"] = self.settings.min_p
        if self.settings.repetition_penalty is not None:
            body["repetition_penalty"] = self.settings.repetition_penalty
        if self.settings.generation_seed is not None:
            body["seed"] = self.settings.generation_seed
        request = Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def propose(self) -> CodeProposal:
        for attempt in range(self.retries + 1):
            try:
                response = self._request()
            except Exception as exc:
                if attempt == self.retries:
                    raise RuntimeError(
                        "Model request failed after "
                        f"{self.retries + 1} attempts: "
                        f"{type(exc).__name__}: {safe_exception_text(exc)}"
                    ) from exc
                time.sleep(2 ** attempt)
                continue

            try:
                message = response["choices"][0]["message"]
                content = message.get("content") or ""
                reasoning = message.get("reasoning") or message.get("reasoning_content")
                code = extract_code(content)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"Model response format failed after {self.retries + 1} attempts: {exc}"
                    ) from exc
                if isinstance(response, dict):
                    choices = response.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        failed_message = choices[0].get("message")
                        if isinstance(failed_message, dict):
                            failed_content = failed_message.get("content")
                            if isinstance(failed_content, str) and failed_content:
                                self.messages.append({
                                    "role": "assistant", "content": failed_content
                                })
                self.messages.append({"role": "user", "content": (
                    "Your previous response could not be executed because it did not "
                    "contain exactly one non-empty ```python``` code block. Return exactly "
                    "one Python code block with at most five direct apis.<app>.<api> calls."
                )})
                continue

            self.messages.append({"role": "assistant", "content": content})
            return CodeProposal(
                code=code,
                raw_content=content,
                reasoning_sha256=(hashlib.sha256(str(reasoning).encode()).hexdigest()
                                  if reasoning else None),
            )
        raise AssertionError("unreachable")

    def observe(self, output: str) -> None:
        self.messages.append({"role": "user", "content": f"Output:\n```\n{output}\n```"})

    def guide(self, hint_text: str) -> None:
        if hint_text:
            self.messages.append({"role": "user", "content": hint_text})

    def reject(self, reason: str) -> None:
        self.messages.append({"role": "user", "content": (
            "[HARNESS_FEEDBACK]\n"
            "status: REJECTED\n"
            "failure_type: PRECHECK_REJECTED\n"
            "state_changed: false\n"
            "rollback: NOT_REQUIRED\n"
            "reason: " + reason + "\n"
            "guidance: The proposal was not executed. Correct the stated precheck "
            "problem and generate a different bounded AppWorld API step."
        )})

    def recover(
        self,
        reason: str,
        *,
        rollback_level: str = "action",
        decision_id: str | None = None,
        action_id: str | None = None,
        failure_type: str = "EXECUTION_WITH_MUTATION",
        state_changed: bool | None = True,
        rollback_required: bool = True,
        rollback_verified: bool | None = True,
        api_signature: str | None = None,
        failure_evidence_feedback: str | None = None,
    ) -> None:
        if rollback_required:
            rollback = (
                "DECISION_CHECKPOINT" if rollback_level == "decision"
                else "ACTION_CHECKPOINT"
            )
            guidance = (
                "The repeated decision path was restored. Do not repeat the same strategy; "
                "choose a different resource or open a new plan."
                if rollback_level == "decision" else
                "The environment was restored to immediately before the failed action. "
                "Correct the parameters or resource before retrying."
            )
        else:
            rollback = "NOT_REQUIRED"
            guidance = (
                "The failed action did not modify the environment. Do not repeat the same "
                "invalid resource; revise its parameters, choose another resource, or replan."
            )
        state_text = (
            "unknown" if state_changed is None else str(state_changed).lower()
        )
        rollback_verified_text = (
            "not_applicable" if rollback_verified is None
            else str(rollback_verified).lower()
        )
        content = (
            "[HARNESS_FEEDBACK]\n"
            f"decision_id: {decision_id or 'none'}\n"
            f"action_id: {action_id or 'unknown'}\n"
            f"api: {api_signature or 'unknown'}\n"
            "status: FAILED\n"
            f"failure_type: {failure_type}\n"
            f"reason: {reason}\n"
            f"state_changed: {state_text}\n"
            f"rollback: {rollback}\n"
            f"rollback_verified: {rollback_verified_text}\n"
            f"guidance: {guidance}"
        )
        if failure_evidence_feedback:
            content += "\n\n" + failure_evidence_feedback
        self.messages.append({"role": "user", "content": content})

    def recover_observation(
        self,
        output: str,
        reason: str,
        *,
        failure_evidence_feedback: str | None = None,
    ) -> None:
        self.observe(output)
        content = (
            "[HARNESS_FEEDBACK]\n"
            "status: FAILED\n"
            "failure_type: POSTCONDITION_FAILED\n"
            "state_changed: false\n"
            "rollback: NOT_REQUIRED\n"
            "reason: " + reason + "\n"
            "guidance: The read-only step did not mutate the sandbox. Correct the API or "
            "arguments and continue without restoring state."
        )
        if failure_evidence_feedback:
            content += "\n\n" + failure_evidence_feedback
        self.messages.append({"role": "user", "content": content})


class OnlineProbabilityGate:
    OBSERVE_ONLY_CATEGORIES = {
        "READ_RESOURCE",
        "LOAD_IDENTITY_OR_CREDENTIALS",
        "AUTHENTICATE",
        "COMPLETE_TASK",
    }

    def __init__(self, graph: AppWorldTraceGraphSpec, *, threshold: float = 0.8,
                 min_unique_tasks: int = 3, seed: int = 17):
        self.graph = graph
        self.threshold = threshold
        self.min_unique_tasks = min_unique_tasks
        self.provider = AppWorldTraceProbabilityProvider(
            graph, seed=seed, hierarchical_fallback=True
        )

    def decide(self, state: str, actions: list[AppWorldAction], failures: list[str],
               remaining_horizon: int) -> StepDecision:
        """Perform only the pre-execution structural/risk check.

        Empirical success is intentionally not enforced here. It is evaluated after
        the proposed step executes and passes environment verification.
        """
        if failures:
            return StepDecision(
                allowed=False, reason="; ".join(failures),
                gate_mode="static_block", risk_tier="invalid",
                graph_version=self.graph.graph_version,
            )
        if not actions:
            return StepDecision(
                allowed=True, reason="Local computation only; no AppWorld API call",
                gate_mode="observe_only", risk_tier="local",
                category=None, graph_version=self.graph.graph_version,
            )
        categories = [action.category for action in actions]
        if all(category in self.OBSERVE_ONLY_CATEGORIES for category in categories):
            category = categories[0] if len(set(categories)) == 1 else "BATCH"
            return StepDecision(
                allowed=True,
                reason="Pre-execution checks passed; score after verified execution",
                gate_mode="post_execution_observe_pending",
                risk_tier=(
                    "terminal" if "COMPLETE_TASK" in categories else "low"
                ),
                category=category,
                graph_version=self.graph.graph_version,
            )
        if any(category not in self.OBSERVE_ONLY_CATEGORIES | {"WRITE_RESOURCE"}
               for category in categories):
            return StepDecision(
                allowed=False,
                reason="Unsupported action category cannot be safely checkpointed",
                gate_mode="static_block",
                risk_tier="unknown",
                category="BATCH" if len(actions) > 1 else actions[0].category,
                graph_version=self.graph.graph_version,
            )
        return StepDecision(
            allowed=True,
            reason="Pre-execution checks passed; write will be checkpointed and scored after verification",
            gate_mode="post_execution_enforce_pending",
            risk_tier="high",
            category="BATCH" if len(actions) > 1 else actions[0].category,
            graph_version=self.graph.graph_version,
        )

    def evaluate_completed_step(
        self,
        prior_state: str,
        action: AppWorldAction,
        remaining_horizon: int,
        post_attributes: Mapping[str, str | bool | int],
    ) -> StepDecision:
        """Score final-task success after an executed step has been verified."""
        assessment = self.provider.evaluate(
            prior_state,
            action.category,
            dict(post_attributes),
            remaining_horizon,
            self.graph.policy,
        )
        mean = (assessment.probabilities or {}).get("SUCCESS")
        enforce = action.category == "WRITE_RESOURCE"
        evidence_sufficient = bool(
            assessment.calculable
            and assessment.gate_eligible
            and assessment.unique_task_count >= self.min_unique_tasks
            and mean is not None
        )
        allowed = (not enforce) or bool(evidence_sufficient and mean >= self.threshold)
        if not assessment.calculable:
            reason = assessment.reason
        elif assessment.unique_task_count < self.min_unique_tasks:
            reason = "Insufficient unique empirical train tasks"
        elif mean is None or mean < self.threshold:
            reason = "Post-step final-task success estimate is below the configured threshold"
        else:
            reason = "Post-step final-task success estimate passed"
        if not enforce:
            reason = (
                "Verified step committed; post-step final-task success estimate is observational"
            )
        return StepDecision(
            allowed=allowed,
            reason=reason,
            gate_mode="post_execution_enforce" if enforce else "post_execution_observe",
            risk_tier="high" if enforce else "low",
            category=action.category,
            probability_mean=mean,
            probability_interval=assessment.success_interval,
            unique_task_count=assessment.unique_task_count,
            fallback_level=assessment.fallback_level,
            graph_version=self.graph.graph_version,
        )


class GraphReadyOnlineGate(OnlineProbabilityGate):
    """Use a schema-4 graph for provenance while withholding uncalibrated gating.

    Its transition statistics are scoped to an exact task and no event-to-live-step
    alignment or held-out cross-task calibration is available yet. Static checks,
    execution verification and rollback on runtime failure remain active.
    """

    def __init__(self, graph: GraphReadyBuild):
        self.graph = graph
        self.graph_ready = graph

    def decide(self, state: str, actions: list[AppWorldAction], failures: list[str],
               remaining_horizon: int) -> StepDecision:
        decision = super().decide(state, actions, failures, remaining_horizon)
        if decision.allowed and decision.gate_mode == "post_execution_enforce_pending":
            decision.gate_mode = "post_execution_observe_pending"
            decision.reason = (
                "Pre-execution checks passed; schema-4 exact-task statistics "
                "are advisory until cross-task success is calibrated"
            )
        return decision

    def evaluate_completed_step(
        self,
        prior_state: str,
        action: AppWorldAction,
        remaining_horizon: int,
        post_attributes: Mapping[str, str | bool | int],
    ) -> StepDecision:
        return StepDecision(
            allowed=True,
            reason=(
                "Verified step committed; graph-ready exact-task transition rates "
                "cannot predict final success for an unseen task"
            ),
            gate_mode="post_execution_observe",
            risk_tier="high" if action.category == "WRITE_RESOURCE" else "low",
            category=action.category,
            graph_version=self.graph.graph_version,
        )


class AdvisoryOnlineGate(OnlineProbabilityGate):
    """Keep structural checks and rollback; never gate on graph observations."""

    def __init__(self, graph_version: str):
        self.graph = SimpleNamespace(graph_version=graph_version)

    def decide(self, state: str, actions: list[AppWorldAction], failures: list[str],
               remaining_horizon: int) -> StepDecision:
        decision = super().decide(state, actions, failures, remaining_horizon)
        if decision.allowed and decision.gate_mode == "post_execution_enforce_pending":
            decision.gate_mode = "post_execution_observe_pending"
            decision.reason = "Pre-execution checks passed; Neo4j path statistics are advisory"
        return decision

    def evaluate_completed_step(
        self, prior_state: str, action: AppWorldAction, remaining_horizon: int,
        post_attributes: Mapping[str, str | bool | int],
    ) -> StepDecision:
        return StepDecision(
            allowed=True,
            reason="Verified step committed; exact-task graph rates are observational",
            gate_mode="post_execution_observe",
            risk_tier="high" if action.category == "WRITE_RESOURCE" else "low",
            category=action.category,
            graph_version=self.graph.graph_version,
        )


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def safe_exception_text(exc: BaseException, *, limit: int = 400) -> str:
    """Return a bounded, redacted exception message for operational diagnostics."""
    message = " ".join(str(exc).split())[:limit]
    message = re.sub(
        r"(?i)\b(access[_ -]?token|password|secret|api[_ -]?key|cvv|card[_ -]?number)"
        r"\b\s*[:=]\s*[^,;\s]+",
        r"\1=<redacted>",
        message,
    )
    return re.sub(
        r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)",
        "<redacted_number>",
        message,
    )


def technical_exception_location(exc: BaseException) -> str | None:
    """Return the final stack location without exposing an absolute server path."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return None
    frame = frames[-1]
    return f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}"


def _read_api_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _runtime_api_calls(world: Any, path: Path) -> list[dict[str, Any]]:
    """Read the active request tracker, falling back to its persisted log.

    AppWorld.load_state() closes and recreates the local requester. Its in-memory
    request tracker therefore restarts at zero, while api_calls.jsonl still contains
    the previous branch until the next execute() call rewrites it. Comparing the
    persisted file lengths across that boundary can report zero new calls even when
    the restored branch successfully executed a batch. The active tracker is the
    authoritative source for a local environment.
    """
    if not getattr(world, "remote_environment_url", None):
        requester = getattr(world, "requester", None)
        requests = getattr(requester, "requests", None)
        if isinstance(requests, list) and all(isinstance(row, dict) for row in requests):
            return [dict(row) for row in requests]
    return _read_api_log(path)


def _new_runtime_api_calls(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return calls from the current execution even if AppWorld reset its tracker.

    A checkpoint restore recreates the requester.  In the normal case ``before`` is
    a prefix of ``after``.  If it is not, the active tracker belongs to a new branch,
    so treating its complete content as new is safer than slicing it with a stale
    pre-restore length (which previously produced a false ``observed 0``).
    """
    if not isinstance(before, list) or not isinstance(after, list):
        raise TypeError("Runtime API call snapshots must both be lists")
    if len(after) >= len(before) and after[:len(before)] == before:
        return after[len(before):]
    return after


def requires_checkpoint(actions: list[AppWorldAction]) -> bool:
    """Checkpoint only business-state writes, not auth/terminal POST requests."""
    return any(action.category == "WRITE_RESOURCE" for action in actions)


def world_state_fingerprint(world: Any) -> str:
    """Hash AppWorld's model hashes without exposing database contents."""
    models = getattr(world, "models", None)
    if models is None or not hasattr(models, "model_hashes"):
        raise RuntimeError("AppWorld model hashes are unavailable; rollback cannot be verified")
    model_hashes = models.model_hashes()
    canonical = json.dumps(model_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def restore_checkpoint(
    world: Any, checkpoint_id: str, expected_fingerprint: str
) -> tuple[bool, str | None, str | None]:
    """Restore a checkpoint and verify the recovered database fingerprint."""
    try:
        world.load_state(checkpoint_id)
        # AppWorld.load_state() calls AppWorld.close_all(), which stops and removes
        # the world's outer time freezer but leaves its wrapper pointing at the
        # already-stopped freezegun object. AppWorld.close() would otherwise stop it
        # a second time and fail on some supported freezegun versions.
        registry = getattr(type(world), "id_to_time_freezer", None)
        freezer_id = getattr(world, "time_freezer_id", None)
        time_freezer = getattr(world, "time_freezer", None)
        if (
            isinstance(registry, dict)
            and freezer_id not in registry
            and time_freezer is not None
            and hasattr(time_freezer, "_freezer")
        ):
            time_freezer._freezer = None
        restored_fingerprint = world_state_fingerprint(world)
        if restored_fingerprint != expected_fingerprint:
            return False, restored_fingerprint, "Restored state fingerprint does not match checkpoint"
        runtime = getattr(world, "_harness_runtime_checkpoints", None)
        if runtime is not None:
            runtime.restore(checkpoint_id)
        return True, restored_fingerprint, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def write_rollback_event(
    audit_path: str | Path,
    *,
    dataset: str,
    task_id: str,
    step: int,
    checkpoint_id: str,
    expected_fingerprint: str,
    restored: bool,
    restored_fingerprint: str | None,
    cause: str,
    reason: str,
    error: str | None,
    rollback_scope: str = "current_step",
    decision_id: str | None = None,
    action_id: str | None = None,
    consecutive_action_failures: int = 0,
    automatic_escalation: bool = False,
    agent_requested: bool = False,
) -> None:
    append_jsonl(audit_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "ROLLBACK",
        "dataset": dataset,
        "task_id": task_id,
        "step": step,
        "checkpoint_id": checkpoint_id,
        "attempted": True,
        "restored": restored,
        "expected_state_fingerprint": expected_fingerprint,
        "restored_state_fingerprint": restored_fingerprint,
        "cause": cause,
        "reason": reason,
        "error": error,
        "rollback_scope": rollback_scope,
        "decision_id": decision_id,
        "action_id": action_id,
        "consecutive_action_failures": consecutive_action_failures,
        "automatic_escalation": automatic_escalation,
        "agent_requested": agent_requested,
        "physical_restore_performed": restored,
        "rollback_verified": restored,
        "environment_safe_to_continue": restored,
    })


def write_failure_event(
    audit_path: str | Path,
    *,
    dataset: str,
    task_id: str,
    step: int,
    failure_type: str,
    reason: str,
    state_changed: bool | None,
    rollback_required: bool,
    rollback_level: str | None,
    rollback_verified: bool | None,
    decision_id: str | None,
    action_id: str | None,
    api_signature: str | None,
    failure_signature: str | None,
    rollback_attempts: int,
    rollback_requested_by_agent: bool = False,
    rollback_request_accepted: bool | None = None,
    rollback_request_rejection_reason: str | None = None,
    rollback_budget_exhausted: bool = False,
    task_terminated_due_to_rollback_budget: bool = False,
    environment_safe_to_continue: bool | None = None,
    meaningful_progress_since_failure: bool = False,
    resource_substitution_available: bool | None = None,
    physical_restore_performed: bool = False,
) -> None:
    """Write a value-free recovery decision that is easy to aggregate."""
    append_jsonl(audit_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "FAILURE_CLASSIFICATION",
        "dataset": dataset,
        "task_id": task_id,
        "step": step,
        "failure_type": failure_type,
        "reason": reason,
        "state_changed": state_changed,
        "rollback_required": rollback_required,
        "rollback_level": rollback_level,
        "rollback_verified": rollback_verified,
        "decision_id": decision_id,
        "action_id": action_id,
        "api_signature": api_signature,
        "failure_signature": failure_signature,
        "rollback_attempts": rollback_attempts,
        "rollback_requested_by_agent": rollback_requested_by_agent,
        "rollback_request_accepted": rollback_request_accepted,
        "rollback_request_rejection_reason": rollback_request_rejection_reason,
        "rollback_budget_exhausted": rollback_budget_exhausted,
        "environment_safe_to_continue": (
            environment_safe_to_continue
            if environment_safe_to_continue is not None
            else False if rollback_required else (
                None if state_changed is None else True
            )
        ),
        "meaningful_progress_since_failure": meaningful_progress_since_failure,
        "resource_substitution_available": resource_substitution_available,
        "physical_restore_performed": physical_restore_performed,
        "task_terminated_due_to_rollback_budget": (
            task_terminated_due_to_rollback_budget
        ),
    })


def write_decision_event(
    audit_path: str | Path,
    *,
    kind: str,
    dataset: str,
    task_id: str,
    step: int,
    decision_id: str,
    action_id: str | None,
    reason: str,
    metadata_explicit: bool,
) -> None:
    append_jsonl(audit_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "dataset": dataset,
        "task_id": task_id,
        "step": step,
        "decision_id": decision_id,
        "action_id": action_id,
        "reason": reason,
        "metadata_explicit": metadata_explicit,
    })


def record_failure_evidence(
    ledger: TaskFailureLedger,
    audit_path: str | Path,
    *,
    dataset: str,
    task_id: str,
    step: int,
    decision_id: str | None,
    action_id: str | None,
    operations: list[OperationContext],
    executed_call_count: int,
    reason: str,
    cause: str,
    state_changed: bool | None,
    rollback_required: bool,
    rollback_level: str | None,
    rollback_verified: bool | None,
) -> str:
    """Record trusted runtime evidence and return bounded Agent feedback."""
    if operations and all(operation.runtime_witnessed for operation in operations):
        # Failed requests may be absent from the environment's successful-call log.
        # The boundary nevertheless witnessed the attempted final invocation.
        executed_call_count = len(operations)
    operation = None
    if operations and executed_call_count:
        operation = operations[min(executed_call_count - 1, len(operations) - 1)]
    evidence = ledger.record_failure(
        step=step,
        decision_id=decision_id,
        action_id=action_id,
        operation=operation,
        reason=reason,
        cause=cause,
        api_executed=executed_call_count > 0,
        state_changed=state_changed,
        rollback_required=rollback_required,
        rollback_level=rollback_level,
        rollback_verified=rollback_verified,
    )
    if evidence is not None:
        append_jsonl(audit_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": "FAILURE_EVIDENCE",
            "dataset": dataset,
            "task_id": task_id,
            "step": step,
            "evidence": evidence.audit_dict(),
        })
    return ledger.format_feedback(relevant_operations=operations)


def runtime_action_matches(action, call):
    return (action.method.upper() == str(call.get("method", "UNKNOWN")).upper()
            and match_route(normalize_route_path(action.route),
                            normalize_route_path(str(call.get("url", "/unknown"))))[0])


def route_verification_records(
    actions: list[AppWorldAction],
    new_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return value-free ordered route comparisons for audit and verification."""
    if isinstance(actions, CheckedActions):
        resolved = actions.resolve(new_calls, runtime_action_matches)
        if resolved is not None:
            actions = resolved
    records: list[dict[str, Any]] = []
    length = max(len(actions), len(new_calls))
    for index in range(length):
        action = actions[index] if index < len(actions) else None
        call = new_calls[index] if index < len(new_calls) else None
        expected_method = action.method.upper() if action is not None else None
        observed_method = (
            str(call.get("method", "UNKNOWN")).upper() if call is not None else None
        )
        expected_route = (
            normalize_route_path(action.route) if action is not None else None
        )
        observed_route = (
            normalize_route_path(str(call.get("url", "/unknown")))
            if call is not None else None
        )
        if action is None:
            route_matches, mismatch = False, "unexpected extra runtime API call"
        elif call is None:
            route_matches, mismatch = False, "expected runtime API call was not observed"
        elif expected_method != observed_method:
            route_matches, mismatch = False, (
                f"HTTP method differs: expected {expected_method}, observed {observed_method}"
            )
        else:
            route_matches, mismatch = match_route(expected_route, observed_route)
        records.append({
            "call_index": index,
            "expected_method": expected_method,
            "observed_method": observed_method,
            "expected_route_template": expected_route,
            "observed_route": observed_route,
            "route_match_result": route_matches,
            "route_mismatch_reason": mismatch,
        })
    return records


def execution_verification(
    output: Any,
    actions: list[AppWorldAction],
    new_calls: list[dict[str, Any]],
) -> tuple[bool, list[str], list[str], str | None]:
    """Verify that runtime calls match the checked proposal and execution succeeded."""
    output_text = normalize_execution_output(output)
    legal_path = True
    if isinstance(actions, CheckedActions):
        resolved = actions.resolve(new_calls, runtime_action_matches)
        legal_path = resolved is not None
        if resolved is not None:
            actions = resolved
    observed = []
    for call in new_calls:
        method = str(call.get("method", "UNKNOWN")).upper()
        route = normalize_route_path(str(call.get("url", "/unknown")))
        observed.append(_classify(method, route))
    expected = [action.category for action in actions]
    route_records = route_verification_records(actions, new_calls)
    if output_text.lstrip().startswith("Execution failed. Traceback:"):
        return False, expected, observed, safe_execution_failure_summary(output_text)
    if not legal_path or len(actions) != len(new_calls) or not all(
        record["route_match_result"] for record in route_records
    ):
        mismatches = [
            record["route_mismatch_reason"] for record in route_records
            if not record["route_match_result"]
        ]
        return False, expected, observed, (
            "Executed API calls did not match the checked proposal: "
            f"expected {len(expected)} call(s) in categories {expected}, observed "
            f"{len(new_calls)} call(s) in categories {observed}; "
            + "; ".join(str(value) for value in mismatches if value)
        )
    return True, expected, observed, None


def safe_execution_failure_summary(output: str) -> str:
    """Extract a bounded API error summary without retaining code or credentials."""
    status_match = re.search(r"Response status code is\s+(\d{3})", output)
    message_match = re.search(
        r'\{"message"\s*:\s*"((?:[^"\\]|\\.)*)"\}', output
    )
    parts = ["AppWorld execution returned a failure traceback"]
    if status_match:
        parts.append(f"HTTP {status_match.group(1)}")
    if message_match:
        try:
            message = json.loads('"' + message_match.group(1) + '"')
        except json.JSONDecodeError:
            message = message_match.group(1)
        message = " ".join(str(message).split())[:240]
        message = re.sub(
            r"(?i)\b(access[_ -]?token|password|secret|api[_ -]?key)\b\s*[:=]\s*\S+",
            r"\1=<redacted>",
            message,
        )
        message = re.sub(
            r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)",
            "<redacted_number>",
            message,
        )
        message = re.sub(
            r"(?i)\b(cvv|security[_ -]?code)\b\s*[:=]\s*\S+",
            r"\1=<redacted>",
            message,
        )
        if message:
            parts.append(message)
    return ": ".join(parts)


def safe_runtime_exception_summary(exc: Exception) -> str:
    """Return a bounded, redacted runtime exception without code or credentials."""
    message = " ".join(str(exc).split())[:400]
    message = re.sub(
        r"(?i)\b(access[_ -]?token|password|secret|api[_ -]?key|cvv|card[_ -]?number)"
        r"\b\s*[:=]\s*[^,;\s]+",
        r"\1=<redacted>",
        message,
    )
    message = re.sub(
        r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)",
        "<redacted_number>",
        message,
    )
    status_match = re.search(r"(?:status code is|http)\s*(\d{3})", message, re.I)
    prefix = f"HTTP {status_match.group(1)}: " if status_match else ""
    return f"AppWorld execution exception: {prefix}{message}"


def verification_failure_cause(output: Any) -> str:
    """Classify a failed post-execution check without retaining traceback content."""
    if normalize_execution_output(output).lstrip().startswith(
        "Execution failed. Traceback:"
    ):
        return "execution_error"
    return "verification_mismatch"


def normalized_failure_type(reason: str, cause: str) -> str:
    """Collapse API messages into stable failure families without retaining values."""
    failure_class, _, _ = classify_failure(reason, cause)
    return failure_class


_SENSITIVE_ARGUMENT_NAMES = {
    "access_token", "token", "password", "secret", "api_key", "cvv_number",
    "card_number",
}


def failure_strategy_hash(code: str) -> str:
    """Hash API names and non-secret argument expressions for repeat detection."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return hashlib.sha256(code.encode()).hexdigest()[:16]
    calls: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        # Valid Python call targets are not always a simple attribute chain.
        # Examples include ``handlers[name]()`` and a callable returned by
        # another call.  ``_attribute_chain`` deliberately returns None for
        # those shapes, so ignore them here instead of turning an ordinary
        # model proposal into a Harness technical failure.
        if not chain or len(chain) != 3 or chain[0] != "apis":
            continue
        arguments = [ast.dump(value, include_attributes=False) for value in node.args]
        keywords = []
        for keyword in node.keywords:
            name = keyword.arg or "**kwargs"
            if name.lower() in _SENSITIVE_ARGUMENT_NAMES:
                # Preserve equality/difference without retaining the secret value.
                raw = ast.dump(keyword.value, include_attributes=False)
                value = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()[:12]
            else:
                value = ast.dump(keyword.value, include_attributes=False)
            keywords.append((name, value))
        calls.append({
            "api": ".".join(chain[1:]),
            "args": arguments,
            "keywords": sorted(keywords),
        })
    canonical = json.dumps(calls, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def action_api_signature(actions: list[AppWorldAction]) -> str:
    return ",".join(
        f"{action.method.upper()} {normalize_route_path(action.route)}" for action in actions
    ) or "NO_API"


def action_failure_signature(
    *,
    decision_id: str,
    actions: list[AppWorldAction],
    code: str,
    reason: str,
    cause: str,
) -> tuple[str, str, str, str]:
    return (
        decision_id,
        action_api_signature(actions),
        normalized_failure_type(reason, cause),
        failure_strategy_hash(code),
    )


def failure_signature_digest(signature: tuple[str, str, str, str]) -> str:
    canonical = json.dumps(signature, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def failure_needs_restore(
    *,
    state_changed: bool | None,
    current_fingerprint: str | None,
    target: RollbackTarget,
) -> bool:
    """Return whether recovery requires a physical state restore.

    Unknown action state is handled conservatively. A decision-level replan only needs
    a restore when the live state differs from the decision checkpoint.
    """
    if target.rollback_level == "decision":
        return current_fingerprint != target.checkpoint_fingerprint
    return state_changed is not False


class RollbackBudget:
    """Bound independent failed actions, excluding recovery-scope changes."""

    def __init__(self, total_limit: int):
        if total_limit < 1:
            raise ValueError("Rollback limit must be positive")
        self.total_limit = total_limit
        self.by_cause: Counter[str] = Counter()

    def record(self, cause: str) -> bool:
        """Record a physical restore and return whether another restore is available."""
        self.by_cause[cause] += 1
        return self.available

    @property
    def used(self) -> int:
        return sum(self.by_cause.values())

    @property
    def available(self) -> bool:
        return self.used < self.total_limit

    @property
    def exhausted(self) -> bool:
        return not self.available


class ConsecutiveRejectionBudget:
    """Stop only after consecutive rejected proposals; a committed step resets it."""

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("Rejection limit must be positive")
        self.limit = limit
        self.current = 0
        self.maximum = 0

    def reject(self) -> bool:
        """Record a rejection and return whether another proposal is allowed."""
        self.current += 1
        self.maximum = max(self.maximum, self.current)
        return self.current < self.limit

    def commit(self) -> None:
        self.current = 0


def _identifier_field(name: str | None) -> bool:
    """Return whether a field can carry a cross-step resource identifier."""
    if not name:
        return False
    normalized = name.lower().strip()
    return normalized == "id" or normalized.endswith(("_id", "_ids"))


def _identifier_tokens(field_name: str, value: Any) -> set[tuple[str, str]]:
    """Convert identifier values to field-scoped hashes; never retain raw IDs."""
    tokens: set[tuple[str, str]] = set()
    if isinstance(value, (list, tuple, set)):
        for item in value:
            tokens.update(_identifier_tokens(field_name, item))
        return tokens
    if value is None or isinstance(value, bool) or not isinstance(value, (str, int)):
        return tokens
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    tokens.add((field_name.lower().strip(), digest))
    return tokens


def _parse_structured_output(output: Any) -> Any | None:
    text = normalize_execution_output(output).strip()
    if not text or text.startswith("Execution failed. Traceback:"):
        return None
    if text.startswith("Munch(") and text.endswith(")"):
        text = text[6:-1]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None


def output_identifier_tokens(output: Any) -> frozenset[tuple[str, str]]:
    """Extract only hashed ID-like fields from a verified tool result."""
    value = _parse_structured_output(output)
    tokens: set[tuple[str, str]] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                field_name = str(key)
                if _identifier_field(field_name):
                    tokens.update(_identifier_tokens(field_name, nested))
                if isinstance(nested, (Mapping, list, tuple, set)):
                    visit(nested)
        elif isinstance(item, (list, tuple, set)):
            for nested in item:
                visit(nested)

    visit(value)
    return frozenset(tokens)


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_assigned_names(item))
        return names
    return set()


def _derived_variable_names(code: str, source_names: set[str]) -> set[str]:
    """Propagate active lineage through deterministic local assignments."""
    if not source_names:
        return set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    known = set(source_names)
    added: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            targets: set[str] = set()
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                value = node.value
                for target in node.targets:
                    targets.update(_assigned_names(target))
            elif isinstance(node, ast.AnnAssign):
                value = node.value
                targets.update(_assigned_names(node.target))
            elif isinstance(node, ast.NamedExpr):
                value = node.value
                targets.update(_assigned_names(node.target))
            if value is None or not targets:
                continue
            referenced = {
                child.id for child in ast.walk(value)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            }
            if referenced & known and not targets <= known:
                known.update(targets)
                added.update(targets)
                changed = True
    return added


@dataclass(frozen=True)
class StepProvenance:
    """Privacy-preserving dependency evidence found in a proposed write step."""

    produced_variables: frozenset[str] = frozenset()
    referenced_variables: frozenset[str] = frozenset()
    input_identifier_tokens: frozenset[tuple[str, str]] = frozenset()


def step_provenance(code: str) -> StepProvenance:
    """Extract variable lineage and hashed literal resource IDs from API calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return StepProvenance()
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    produced: set[str] = set()
    referenced: set[str] = set()
    identifiers: set[tuple[str, str]] = set()
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        chain = _attribute_chain(call.func)
        if not chain or chain[0] != "apis":
            continue
        for argument in list(call.args) + [keyword.value for keyword in call.keywords]:
            referenced.update(
                node.id for node in ast.walk(argument)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )
        for keyword in call.keywords:
            if not _identifier_field(keyword.arg):
                continue
            try:
                literal = ast.literal_eval(keyword.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            identifiers.update(_identifier_tokens(str(keyword.arg), literal))

        current: ast.AST = call
        while current in parents:
            parent = parents[current]
            if isinstance(parent, ast.Assign) and current is parent.value:
                for target in parent.targets:
                    produced.update(_assigned_names(target))
                break
            if isinstance(parent, ast.AnnAssign) and current is parent.value:
                produced.update(_assigned_names(parent.target))
                break
            if isinstance(parent, ast.NamedExpr) and current is parent.value:
                produced.update(_assigned_names(parent.target))
                break
            if isinstance(parent, (ast.Expr, ast.Return)):
                break
            current = parent
    return StepProvenance(
        produced_variables=frozenset(produced),
        referenced_variables=frozenset(referenced),
        input_identifier_tokens=frozenset(identifiers),
    )


def _identifier_token_overlap(
    left: set[tuple[str, str]] | frozenset[tuple[str, str]],
    right: set[tuple[str, str]] | frozenset[tuple[str, str]],
) -> bool:
    """Match hashed resource IDs while keeping unrelated identifier kinds separate."""
    if set(left) & set(right):
        return True
    for left_field, left_hash in left:
        left_kind = left_field.removesuffix("_ids").removesuffix("_id")
        for right_field, right_hash in right:
            if left_hash != right_hash:
                continue
            right_kind = right_field.removesuffix("_ids").removesuffix("_id")
            if left_kind and right_kind and (
                left_kind in right_kind or right_kind in left_kind
            ):
                return True
    return False


@dataclass
class TransactionFrame:
    root_checkpoint_id: str
    root_fingerprint: str
    start_step: int
    steps: list[int] = dataclass_field(default_factory=list)
    produced_variables: set[str] = dataclass_field(default_factory=set)
    resource_tokens: set[tuple[str, str]] = dataclass_field(default_factory=set)


@dataclass(frozen=True)
class PreparedTransactionStep:
    step: int
    checkpoint_id: str
    checkpoint_fingerprint: str
    provenance: StepProvenance
    dependent: bool
    dependency_kinds: tuple[str, ...]
    rollback_checkpoint_id: str
    rollback_fingerprint: str
    transaction_start_step: int
    transaction_steps: tuple[int, ...]
    metadata_explicit: bool = False
    committed_previous_steps: tuple[int, ...] = ()

    @property
    def rollback_scope(self) -> str:
        return "dependency_chain" if self.dependent else "current_step"


class CrossStepTransactionManager:
    """Track a generic write dependency chain and select its rollback root."""

    def __init__(self) -> None:
        self.active: TransactionFrame | None = None
        self.opened = 0
        self.explicit_opened = 0
        self.implicit_opened = 0
        self.extensions = 0
        self.commits = 0
        self.dependency_rollbacks = 0

    def prepare_write(
        self,
        *,
        step: int,
        code: str,
        checkpoint_id: str,
        checkpoint_fingerprint: str,
        metadata_explicit: bool = False,
    ) -> PreparedTransactionStep:
        provenance = step_provenance(code)
        dependency_kinds: list[str] = []
        if self.active is not None:
            if provenance.referenced_variables & self.active.produced_variables:
                dependency_kinds.append("variable_lineage")
            if _identifier_token_overlap(
                provenance.input_identifier_tokens, self.active.resource_tokens
            ):
                dependency_kinds.append("resource_identifier")
        dependent = bool(dependency_kinds)
        committed_previous_steps: tuple[int, ...] = ()
        if self.active is not None and not dependent:
            committed_previous_steps = tuple(self.active.steps)
            self.active = None
            self.commits += 1

        if dependent:
            assert self.active is not None
            rollback_checkpoint_id = self.active.root_checkpoint_id
            rollback_fingerprint = self.active.root_fingerprint
            transaction_start_step = self.active.start_step
            transaction_steps = tuple([*self.active.steps, step])
        else:
            rollback_checkpoint_id = checkpoint_id
            rollback_fingerprint = checkpoint_fingerprint
            transaction_start_step = step
            transaction_steps = (step,)
        return PreparedTransactionStep(
            step=step,
            checkpoint_id=checkpoint_id,
            checkpoint_fingerprint=checkpoint_fingerprint,
            provenance=provenance,
            dependent=dependent,
            dependency_kinds=tuple(dependency_kinds),
            rollback_checkpoint_id=rollback_checkpoint_id,
            rollback_fingerprint=rollback_fingerprint,
            transaction_start_step=transaction_start_step,
            transaction_steps=transaction_steps,
            metadata_explicit=metadata_explicit,
            committed_previous_steps=committed_previous_steps,
        )

    def record_success(
        self, prepared: PreparedTransactionStep, output: Any
    ) -> tuple[str, TransactionFrame]:
        resource_tokens = set(prepared.provenance.input_identifier_tokens)
        resource_tokens.update(output_identifier_tokens(output))
        if prepared.dependent:
            assert self.active is not None
            self.active.steps.append(prepared.step)
            self.active.produced_variables.update(prepared.provenance.produced_variables)
            self.active.resource_tokens.update(resource_tokens)
            self.extensions += 1
            return "TRANSACTION_EXTENDED", self.active
        self.active = TransactionFrame(
            root_checkpoint_id=prepared.checkpoint_id,
            root_fingerprint=prepared.checkpoint_fingerprint,
            start_step=prepared.step,
            steps=[prepared.step],
            produced_variables=set(prepared.provenance.produced_variables),
            resource_tokens=resource_tokens,
        )
        self.opened += 1
        if prepared.metadata_explicit:
            self.explicit_opened += 1
        else:
            self.implicit_opened += 1
        return "TRANSACTION_OPENED", self.active

    def record_rollback(self, prepared: PreparedTransactionStep) -> None:
        if prepared.dependent:
            self.active = None
            self.dependency_rollbacks += 1

    def observe_non_write_success(self, code: str, output: Any) -> None:
        """Carry lineage through read/local steps without widening rollback scope."""
        if self.active is None:
            return
        provenance = step_provenance(code)
        dependent = bool(
            provenance.referenced_variables & self.active.produced_variables
            or _identifier_token_overlap(
                provenance.input_identifier_tokens, self.active.resource_tokens
            )
        )
        aliases = _derived_variable_names(code, self.active.produced_variables)
        self.active.produced_variables.update(aliases)
        if dependent:
            self.active.produced_variables.update(provenance.produced_variables)
            self.active.resource_tokens.update(output_identifier_tokens(output))

    def commit_active(self) -> tuple[int, ...]:
        if self.active is None:
            return ()
        steps = tuple(self.active.steps)
        self.active = None
        self.commits += 1
        return steps


_HARNESS_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_HARNESS_NODE_TYPES = {"decision_action", "action", "observation"}
_RECOVERY_INTENTS = {"retry_action", "replan_decision"}


@dataclass(frozen=True)
class HarnessStepMetadata:
    node_type: str
    decision_id: str | None
    action_id: str | None
    recovery_intent: str
    explicit: bool = True


def parse_harness_step_metadata(
    code: str,
) -> tuple[HarnessStepMetadata | None, list[str]]:
    """Read a literal HARNESS_META assignment without executing model code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, []
    assignments: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "HARNESS_META"
            for target in node.targets
        ):
            assignments.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "HARNESS_META"
            and node.value is not None
        ):
            assignments.append(node.value)
    if not assignments:
        return None, []
    if len(assignments) != 1:
        return None, ["Exactly one literal HARNESS_META assignment is allowed"]
    try:
        payload = ast.literal_eval(assignments[0])
    except (ValueError, TypeError, SyntaxError):
        return None, ["HARNESS_META must be a literal dictionary"]
    if not isinstance(payload, dict):
        return None, ["HARNESS_META must be a literal dictionary"]
    allowed = {"node_type", "decision_id", "action_id", "recovery_intent"}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        return None, [f"Unsupported HARNESS_META fields: {unexpected}"]
    node_type = payload.get("node_type")
    if node_type not in _HARNESS_NODE_TYPES:
        return None, [
            "HARNESS_META.node_type must be decision_action, action, or observation"
        ]
    recovery_intent = payload.get("recovery_intent", "retry_action")
    if recovery_intent not in _RECOVERY_INTENTS:
        return None, [
            "HARNESS_META.recovery_intent must be retry_action or replan_decision"
        ]
    decision_id = payload.get("decision_id")
    action_id = payload.get("action_id")
    if node_type in {"decision_action", "action"}:
        if not isinstance(decision_id, str) or not _HARNESS_ID_PATTERN.fullmatch(
            decision_id
        ):
            return None, ["A bounded decision_id is required for action metadata"]
        if not isinstance(action_id, str) or not _HARNESS_ID_PATTERN.fullmatch(action_id):
            return None, ["A bounded action_id is required for action metadata"]
    else:
        if decision_id is not None and (
            not isinstance(decision_id, str)
            or not _HARNESS_ID_PATTERN.fullmatch(decision_id)
        ):
            return None, ["Invalid decision_id in observation metadata"]
        if action_id is not None:
            return None, ["Observation metadata cannot declare action_id"]
    return HarnessStepMetadata(
        node_type=node_type,
        decision_id=decision_id,
        action_id=action_id,
        recovery_intent=recovery_intent,
    ), []


def invalid_harness_metadata_requests_replan(code: str) -> bool:
    """Detect a literal decision-restore request even when other metadata is invalid.

    Invalid grouping metadata can safely fall back to Harness-generated decision/action
    identifiers.  A restore request is different: it asks the Harness to change the live
    state, so malformed metadata must not silently turn that request into an executable
    action.  This helper reads literals only and never executes model-supplied code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "HARNESS_META"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "HARNESS_META"
        ):
            value = node.value
        if value is None:
            continue
        try:
            payload = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("recovery_intent") == "replan_decision"
        ):
            return True
    return False


@dataclass
class DecisionFrame:
    decision_id: str
    checkpoint_id: str
    checkpoint_fingerprint: str
    opened_step: int
    abstract_state: str
    metadata_explicit: bool
    consecutive_action_failures: int = 0
    failure_pending: bool = False
    last_failure_signature: tuple[str, str, str, str] | None = None


@dataclass(frozen=True)
class RollbackTarget:
    checkpoint_id: str
    checkpoint_fingerprint: str
    rollback_level: str
    decision_id: str
    action_id: str
    consecutive_action_failures: int
    automatic_escalation: bool
    restored_abstract_state: str | None = None


class DecisionRollbackManager:
    """Manage explicit decision boundaries and two-level rollback escalation."""

    def __init__(self, *, max_action_retries: int, max_decision_rollbacks: int):
        if max_action_retries < 0:
            raise ValueError("Action retry limit cannot be negative")
        if max_decision_rollbacks < 1:
            raise ValueError("Decision rollback limit must be positive")
        self.max_action_retries = max_action_retries
        self.max_decision_rollbacks = max_decision_rollbacks
        self.active: DecisionFrame | None = None
        self.opened = 0
        self.explicit_opened = 0
        self.implicit_opened = 0
        self.commits = 0
        self.action_rollbacks = 0
        self.decision_rollbacks = 0
        self.agent_requested_decision_rollbacks = 0
        self.automatic_decision_rollbacks = 0

    def effective_metadata(
        self,
        metadata: HarnessStepMetadata | None,
        *,
        step: int,
    ) -> HarnessStepMetadata:
        if metadata is not None:
            return metadata
        if self.active is not None and self.active.failure_pending:
            return HarnessStepMetadata(
                node_type="action",
                decision_id=self.active.decision_id,
                action_id=f"implicit_action_{step}",
                recovery_intent="retry_action",
                explicit=False,
            )
        return HarnessStepMetadata(
            node_type="decision_action",
            decision_id=f"implicit_decision_{step}",
            action_id=f"implicit_action_{step}",
            recovery_intent="retry_action",
            explicit=False,
        )

    def open(
        self,
        *,
        decision_id: str,
        checkpoint_id: str,
        checkpoint_fingerprint: str,
        step: int,
        abstract_state: str,
        metadata_explicit: bool,
    ) -> DecisionFrame:
        self.active = DecisionFrame(
            decision_id=decision_id,
            checkpoint_id=checkpoint_id,
            checkpoint_fingerprint=checkpoint_fingerprint,
            opened_step=step,
            abstract_state=abstract_state,
            metadata_explicit=metadata_explicit,
        )
        self.opened += 1
        if metadata_explicit:
            self.explicit_opened += 1
        else:
            self.implicit_opened += 1
        return self.active

    def commit_active(self) -> DecisionFrame | None:
        frame = self.active
        if frame is not None:
            self.active = None
            self.commits += 1
        return frame

    def needs_decision_restore(
        self,
        metadata: HarnessStepMetadata,
        *,
        proposed_strategy_hash: str | None = None,
    ) -> bool:
        return (
            metadata.recovery_intent == "replan_decision"
            and self.decision_restore_request_rejection_reason(
                metadata, proposed_strategy_hash=proposed_strategy_hash
            ) is None
        )

    @property
    def decision_rollback_available(self) -> bool:
        return self.decision_rollbacks < self.max_decision_rollbacks

    def decision_restore_request_rejection_reason(
        self,
        metadata: HarnessStepMetadata,
        *,
        proposed_strategy_hash: str | None = None,
    ) -> str | None:
        if metadata.recovery_intent != "replan_decision":
            return None
        if self.active is None:
            return "No active decision checkpoint is available"
        if metadata.decision_id != self.active.decision_id:
            return (
                "A decision rollback request must reference the active decision_id "
                f"{self.active.decision_id!r}"
            )
        if not self.active.failure_pending:
            return "Decision rollback is allowed only after an action failure"
        if self.active.consecutive_action_failures <= self.max_action_retries:
            return (
                "Action-level retry or resource substitution is still available; "
                "decision rollback requires a repeated equivalent failure"
            )
        if (
            proposed_strategy_hash is not None
            and self.active.last_failure_signature is not None
            and proposed_strategy_hash != self.active.last_failure_signature[3]
        ):
            return (
                "The proposal already changes the resource or parameters. Execute that "
                "substitution from the current safe state with recovery_intent="
                "'retry_action' instead of restoring the decision checkpoint"
            )
        if not self.decision_rollback_available:
            return (
                "Decision rollback budget is exhausted; continue from the current safe "
                "state with different parameters, resource, or strategy"
            )
        return None

    def requested_decision_target(
        self, metadata: HarnessStepMetadata
    ) -> RollbackTarget:
        if self.active is None:
            raise RuntimeError("No active decision is available for replanning")
        return RollbackTarget(
            checkpoint_id=self.active.checkpoint_id,
            checkpoint_fingerprint=self.active.checkpoint_fingerprint,
            rollback_level="decision",
            decision_id=self.active.decision_id,
            action_id=metadata.action_id or "unknown_action",
            consecutive_action_failures=self.active.consecutive_action_failures,
            automatic_escalation=False,
            restored_abstract_state=self.active.abstract_state,
        )

    def failure_target(
        self,
        *,
        action_checkpoint_id: str,
        action_checkpoint_fingerprint: str,
        action_id: str,
        failure_signature: tuple[str, str, str, str],
    ) -> RollbackTarget:
        if self.active is None:
            raise RuntimeError("A state-changing action has no active decision")
        if self.active.last_failure_signature == failure_signature:
            self.active.consecutive_action_failures += 1
        else:
            self.active.last_failure_signature = failure_signature
            self.active.consecutive_action_failures = 1
        self.active.failure_pending = True
        automatic = (
            self.active.consecutive_action_failures > self.max_action_retries
            and self.decision_rollback_available
        )
        return RollbackTarget(
            checkpoint_id=(
                self.active.checkpoint_id if automatic else action_checkpoint_id
            ),
            checkpoint_fingerprint=(
                self.active.checkpoint_fingerprint
                if automatic else action_checkpoint_fingerprint
            ),
            rollback_level="decision" if automatic else "action",
            decision_id=self.active.decision_id,
            action_id=action_id,
            consecutive_action_failures=self.active.consecutive_action_failures,
            automatic_escalation=automatic,
            restored_abstract_state=(
                self.active.abstract_state if automatic else None
            ),
        )

    def record_restore(self, target: RollbackTarget, *, restored: bool) -> bool:
        if not restored:
            return True
        if target.rollback_level == "action":
            self.action_rollbacks += 1
            return True
        self.decision_rollbacks += 1
        if target.automatic_escalation:
            self.automatic_decision_rollbacks += 1
        else:
            self.agent_requested_decision_rollbacks += 1
        self.active = None
        return self.decision_rollback_available

    def record_action_success(self) -> None:
        if self.active is not None:
            self.active.consecutive_action_failures = 0
            self.active.failure_pending = False
            self.active.last_failure_signature = None

    def abandon_active_without_restore(self) -> DecisionFrame | None:
        """Close a failed decision when its checkpoint already equals live state."""
        frame = self.active
        self.active = None
        return frame


def run_online_task(
    *,
    appworld_root: Path,
    dataset: str,
    task_id: str,
    experiment_name: str,
    settings: ModelSettings,
    gate: OnlineProbabilityGate,
    max_steps: int,
    max_rejections: int,
    max_rollbacks: int,
    max_output_tokens: int,
    max_action_retries: int = 1,
    max_decision_rollbacks: int = 2,
    graph_guidance: GraphGuidanceProvider | None = None,
    task_target_tracking: bool = False,
    completion_verification: bool = False,
    completion_evidence_read_budget: int = 2,
) -> TaskRunSummary:
    """Run one real AppWorld sandbox task; imports AppWorld only on the server path."""
    import os

    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    from appworld import AppWorld, evaluate_task, update_root

    update_root(str(appworld_root))
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    prompt_path = appworld_root / "experiments" / "prompts" / "react_code_agent" / "instructions.txt"
    agent = OpenAICompatibleCodeAgent(settings, max_output_tokens=max_output_tokens)
    state = "START"
    proposed = executed = rejected = api_calls_executed = 0
    checkpointed = rollback_attempts = rollback_successes = post_step_checks = 0
    precheck_rejections = execution_failures = no_mutation_failures = 0
    postcondition_failures = repeated_decision_failures = rollback_failures = 0
    completion_coverage_checks = completion_coverage_rejections = 0
    completion_coverage_advisories = 0
    completion_format_checks = completion_format_rejections = 0
    semantic_write_checks = semantic_write_rejections = 0
    semantic_completion_checks = semantic_completion_rejections = 0
    harness_metadata_advisories = 0
    guidance_queries = guidance_hits = guidance_errors = 0
    probability_ranked_guidance_hits = 0
    success_guidance_scores: list[float] = []
    guidance_available = graph_guidance is not None
    rollback_budget = RollbackBudget(max_rollbacks)
    rejection_budget = ConsecutiveRejectionBudget(max_rejections)
    decision_manager = DecisionRollbackManager(
        max_action_retries=max_action_retries,
        max_decision_rollbacks=max_decision_rollbacks,
    )
    rolled_back_codes: set[str] = set()
    repeated_rolled_back_proposals = 0
    verified_tools: list[str] = []
    bulk_target_count: int | None = None
    coverage_evidence_source = "none"
    coverage_evidence_confidence = "none"
    candidate_collection_count: int | None = None
    coverage_gate_enforced = False
    coverage_gate_reason = "No target-count evidence has been established."
    verified_write_count = 0
    last_verified_write_step: int | None = None
    last_verified_read_step: int | None = None
    previous_guidance = ""
    stop_reason = "Maximum steps reached"
    output_dir = appworld_root / "experiments" / "outputs" / experiment_name / "tasks" / task_id
    audit_path = output_dir / "logs" / "harness_events.jsonl"
    completed = False
    error: str | None = None
    current_runtime_stage = "environment_setup"
    current_step = 0
    technical_failure_stage: str | None = None
    technical_failure_location: str | None = None
    task_terminated_due_to_rollback_budget = False
    failure_ledger = TaskFailureLedger(task_id)
    target_tracker: TaskTargetTracker | None = None
    completion_verifier = CompletionVerifier() if completion_verification else None
    acquisition = CompletionEvidencePending(completion_evidence_read_budget)
    restore_scope_history = []
    active_tool_evidence = None
    completion_candidate_status = None

    try:
        current_runtime_stage = "environment_open"
        with AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            max_interactions=max_steps,
            max_api_calls_per_interaction=MAX_API_CALLS_PER_STEP,
            raise_on_extra_parameters=True,
            raise_on_unsafe_syntax=True,
            raise_on_unsafe_execution=True,
        ) as world:
            current_runtime_stage = "agent_initialize"
            agent.initialize(world, prompt_path)
            proposal_contracts = compile_proposal_contracts(world.task.instruction, world.task.api_docs)
            append_jsonl(audit_path, {"kind": "PROPOSAL_CONTRACTS_INITIALIZED", "dataset": dataset,
                "task_id": task_id, "answer_mode": proposal_contracts.output.answer_mode,
                "answer_basis": proposal_contracts.output.answer_basis,
                "answer_intent_unresolved_reason": proposal_contracts.output.request_unresolved_reason,
                "public_completion_protocol_bound": proposal_contracts.output.answer_port is not None,
                "answer_order_compiled": proposal_contracts.output.order is not None,
                "field_contract_count": len(proposal_contracts.writes),
                "source": "20261013_answer_protocol_revision2_train_public_and_api_docs",
                "order_basis": proposal_contracts.output.order_basis,
                "extraction_issues": [{"check": c, "reason": r, "start": a, "end": b}
                    for c, r, a, b in proposal_contracts.extraction_issues]})
            if task_target_tracking or completion_verification:
                target_tracker = TaskTargetTracker(
                    world.task.instruction if task_target_tracking else "",
                    world.task.api_docs,
                )
                append_jsonl(audit_path, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": "TASK_TARGETS_INITIALIZED",
                    "task_target_tracking_enabled": task_target_tracking,
                    "completion_verification_enabled": completion_verification,
                    "constraints": target_tracker.audit_constraints(),
                    "contract": target_tracker.audit_contract(),
                    **target_tracker.audit_metrics(),
                })
            def snapshot_runtime_context():
                return {
                    "restore_scope_index": len(restore_scope_history),
                    "targets": target_tracker.checkpoint_state() if target_tracker else None,
                    "verified_tools": list(verified_tools),
                    "verified_write_count": verified_write_count,
                    "last_verified_write_step": last_verified_write_step,
                    "last_verified_read_step": last_verified_read_step,
                }

            def restore_runtime_context(context):
                nonlocal verified_write_count, last_verified_write_step, last_verified_read_step
                verified_tools[:] = context["verified_tools"]
                verified_write_count = context["verified_write_count"]
                last_verified_write_step = context["last_verified_write_step"]
                last_verified_read_step = context["last_verified_read_step"]
                if target_tracker is not None:
                    target_tracker.restore_checkpoint_state(context["targets"])
                relevant = restore_scope_history[context.get("restore_scope_index", len(restore_scope_history)):]
                if active_tool_evidence is not None:
                    relevant = relevant + active_tool_evidence.attempts
                writes = [{"operation": r["operation"], "returned_successfully": r["returned"]}
                          for r in relevant if r.get("category") == "WRITE_RESOURCE"]
                failed = next((r["operation"] for r in reversed(relevant) if not r["returned"]), None)
                scope_feedback = {"operations_in_restored_scope": writes, "failed_operation": failed,
                    "restore_level": world._harness_runtime_checkpoints.last_restore.get("restore_level"),
                    "checkpoint_restored": True, "state_evidence_invalidated": True,
                    "instruction": ("The whole action-step checkpoint was restored. "
                        if world._harness_runtime_checkpoints.last_restore.get("restore_level") == "action"
                        else "The whole decision checkpoint scope was restored. ") + "Do not assume successful WRITE calls "
                        "in this restored scope still have effect. Re-read relevant state or reconstruct "
                        "prerequisites before retrying dependent actions."}
                del restore_scope_history[context.get("restore_scope_index", len(restore_scope_history)):]
                if target_tracker is not None:
                    agent.observe("[HARNESS_RESTORE_SCOPE]\n" + json.dumps(scope_feedback))
                    target_tracker.metrics["rollback_scope_feedback_events"] += 1
                    append_jsonl(audit_path, {"kind": "ROLLBACK_SCOPE_FEEDBACK", "step": current_step,
                        **scope_feedback, "ordinary_rejection_budget_consumed": False})
                agent.observe("Checkpoint restored. Python values now match that checkpoint; "
                              "Unsupported Python objects were discarded; re-create them if needed. "
                              "discard assumptions about resources created on the abandoned branch. "
                              "Re-read relevant state before using resource IDs or completing the task. "
                              "The failure information remains available for replanning.")
                append_jsonl(audit_path, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": "RUNTIME_CHECKPOINT_RESTORED", "step": current_step,
                    "evidence_cache_invalidated": True,
                    "python_namespace_restored": True,
                    "failure_evidence_preserved": True,
                    **world._harness_runtime_checkpoints.last_restore,
                })

            world._harness_runtime_checkpoints = RuntimeCheckpoints(
                world, snapshot_runtime_context, restore_runtime_context)
            bulk_target_count = explicit_instruction_target_count(world.task.instruction)
            if bulk_target_count is not None:
                coverage_evidence_source = "instruction_explicit_count_or_list"
                coverage_evidence_confidence = "high"
                coverage_gate_enforced = False
                coverage_gate_reason = (
                    "The instruction count is diagnostic only; write-call counts "
                    "cannot prove target completion."
                )
            graph_ready = getattr(gate, "graph_ready", None)
            if graph_ready is not None:
                append_jsonl(audit_path, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": "GRAPH_CONTEXT",
                    "dataset": dataset,
                    "task_id": task_id,
                    "graph_version": graph_ready.graph_version,
                    "estimator_scope": "exact_task",
                    "exact_task_present": task_id in graph_ready.tasks,
                    "cross_task_probability_gate_enabled": False,
                })
            api_log = Path(world.output_logs_directory) / "api_calls.jsonl"
            def flush_evidence_events(step):
                if target_tracker is not None:
                    for event in target_tracker.drain_evidence_events():
                        append_jsonl(audit_path, {"dataset": dataset, "task_id": task_id,
                            "step": step, **event})

            def check_completion(step, phase, version, answer_code=None):
                target_tracker.synchronize_state_version(version if version != "state_unknown" else None)
                if answer_code is not None:
                    target_tracker.observe_completion_answer(answer_code, getattr(getattr(world, "shell", None), "user_ns", None))
                result = completion_verifier.verify(target_tracker.task_target, target_tracker.current_verified_state)
                if answer_code is not None:
                    output_checks = check_output(answer_code, proposal_contracts, world.task.api_docs,
                        getattr(getattr(world, "shell", None), "user_ns", None))
                    uncheckable = any(f.status == "UNKNOWN" and f.reason == "answer_uncheckable" for f in output_checks)
                    if uncheckable and not any(r.get("reason") == "answer_uncheckable" for r in result.unknown_reasons):
                        result.unknown_count += 1
                        result.unknown_reasons.append({"reason": "answer_uncheckable", "missing_evidence_type": "answer_uncheckable",
                            "suggested_verification_scope": "$answer", "evidence_source_operation": None})
                        if not result.blocks_completion:
                            result.status = "UNKNOWN"
                plan = target_tracker.evidence_plan(result) if result.status == "UNKNOWN" else []
                target_tracker.record_verification(result, count_block=phase == "final_reconciliation")
                target_tracker.record_requirement_outcomes(completion_verifier)
                flush_evidence_events(step)
                episodes = target_tracker.update_repair_episodes(result, completion_verifier, state_version=version)
                target_tracker.metrics["completion_final_status"] = "VIOLATED" if result.blocks_completion else None
                feedback = target_tracker.feedback_once(result)
                if feedback:
                    agent.observe(feedback)
                append_jsonl(audit_path, {"timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": "COMPLETION_VERIFICATION", "dataset": dataset, "task_id": task_id,
                    "step": step, "phase": phase, **result.audit_dict(), "feedback_sent": bool(feedback),
                    "completion_attempt_blocked": phase == "final_reconciliation" and result.blocks_completion,
                    "repair_episodes": episodes, "rollback_required": False,
                    "ordinary_rejection_budget_consumed": False, "rollback_budget_consumed": False})
                return result, plan

            def assess_acquisition(result, plan, version, step):
                status = acquisition.assess(result, plan, version)
                target_tracker.metrics["completion_evidence_pending_episodes"] = acquisition.episodes
                target_tracker.metrics["completion_evidence_budget_exhausted"] = acquisition.exhausted
                if status == "COMPLETION_EVIDENCE_PENDING":
                    pending = acquisition.pending
                    agent.observe("[COMPLETION_EVIDENCE_PENDING]\n" + json.dumps({
                        "missing_evidence": result.unknown_reasons, "relevant_reads": plan,
                        "read_allowance_remaining": min(pending["read_allowance"]-pending["attempts_used"],
                            acquisition.budget-acquisition.reads),
                        "instruction": "Obtain the named evidence before completing. Only relevant READs/local "
                            "computation are evidence acquisition. A WRITE exits this mode for ordinary repair. "
                            "Do not repeat complete_task without obtaining evidence."}, default=str))
                append_jsonl(audit_path, {"kind": status, "step": step,
                    "evidence_gap_key": acquisition.pending["evidence_gap_key"] if acquisition.pending else None,
                    "attempts_used": acquisition.reads, "read_budget": acquisition.budget,
                    "ordinary_rejection_budget_consumed": False, "rollback_required": False})
                return status

            for step in range(1, max_steps + 1):
                current_step = step
                # A failed READ can return through recovery before the commit hook.
                # Reconcile an exhausted allowance before asking for another proposal,
                # so the Agent is not told to obtain zero additional READs indefinitely.
                if (completion_verifier is not None and target_tracker is not None
                        and acquisition.pending and acquisition.reads >= acquisition.budget):
                    exhausted_result, exhausted_plan = check_completion(
                        step, "evidence_budget_exhausted", target_tracker._state_version)
                    exhausted_status = assess_acquisition(exhausted_result, exhausted_plan,
                        target_tracker._state_version, step)
                    if exhausted_status == "COMPLETED_WITH_UNCERTAINTY":
                        agent.observe("[COMPLETION_EVIDENCE_BUDGET_EXHAUSTED] No additional evidence READ is required. "
                            "You may propose complete_task from the current safe state with uncertainty; "
                            "any confirmed violation must still be repaired. No rollback was requested.")
                if guidance_available and graph_guidance is not None:
                    current_runtime_stage = "graph_guidance"
                    hints, guidance_error = query_graph_guidance(
                        graph_guidance,
                        world.task.instruction, world.task.api_docs, verified_tools
                    )
                    guidance_queries += 1
                    if guidance_error:
                        guidance_errors += 1
                        guidance_available = False
                    guidance_hits += int(bool(hints))
                    scores = [hint.get("success_guidance_score") for hint in hints
                              if isinstance(hint.get("success_guidance_score"), (int, float))]
                    if scores:
                        probability_ranked_guidance_hits += 1
                        success_guidance_scores.append(float(scores[0]))
                    hint_text = format_guidance(hints)
                    if hint_text and hint_text != previous_guidance:
                        agent.guide(hint_text)
                    previous_guidance = hint_text
                    append_jsonl(audit_path, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "GRAPH_GUIDANCE",
                        "dataset": dataset,
                        "task_id": task_id,
                        "step": step,
                        "source_split": "train",
                        "graph_version": graph_guidance.graph_version,
                        "matched": bool(hints),
                        "hints": hints,
                        "error_type": guidance_error,
                        "cross_task_probability_gate_enabled": False,
                    })
                current_runtime_stage = "model_proposal"
                proposal = agent.propose()
                proposed += 1
                code_digest = hashlib.sha256(proposal.code.encode()).hexdigest()
                if code_digest in rolled_back_codes:
                    rejected += 1
                    precheck_rejections += 1
                    repeated_rolled_back_proposals += 1
                    append_jsonl(audit_path, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "REPEATED_ROLLED_BACK_PROPOSAL",
                        "dataset": dataset,
                        "task_id": task_id,
                        "step": step,
                        "code_sha256": code_digest,
                    })
                    agent.reject(
                        "This exact code was already rolled back from the current state. "
                        "Choose a different API or correct the call before retrying."
                    )
                    write_failure_event(
                        audit_path,
                        dataset=dataset,
                        task_id=task_id,
                        step=step,
                        failure_type="PRECHECK_REJECTED",
                        reason="Exact rolled-back proposal was repeated",
                        state_changed=False,
                        rollback_required=False,
                        rollback_level=None,
                        rollback_verified=None,
                        decision_id=(
                            decision_manager.active.decision_id
                            if decision_manager.active else None
                        ),
                        action_id=None,
                        api_signature=None,
                        failure_signature=None,
                        rollback_attempts=rollback_attempts,
                    )
                    if not rejection_budget.reject():
                        stop_reason = "Repeated proposal rejection budget exhausted"
                        break
                    continue
                shell = getattr(world, "shell", None)
                proposal_namespace = getattr(shell, "user_ns", None)
                if not isinstance(proposal_namespace, Mapping):
                    proposal_namespace = None
                current_runtime_stage = "static_precheck"
                actions, failures = inspect_code(
                    proposal.code, world.task.api_docs, proposal_namespace
                )
                failures.extend(sequential_batch_failures(proposal.code, actions,
                    world.task.instruction, world.task.api_docs))
                try:
                    proposal_context_signature = world_state_fingerprint(world)
                except Exception:
                    proposal_context_signature = "state_unknown"
                if (completion_verifier is not None and acquisition.pending and not failures
                        and acquisition.pending["state_version"] != proposal_context_signature):
                    refreshed, refreshed_plan = check_completion(step, "pending_state_changed", proposal_context_signature)
                    assess_acquisition(refreshed, refreshed_plan, proposal_context_signature, step)
                if completion_verifier is not None and acquisition.pending and not failures:
                    if any(a.category == "WRITE_RESOURCE" for a in actions):
                        acquisition.exit_for_write()
                        append_jsonl(audit_path, {"kind": "COMPLETION_EVIDENCE_REPAIR_ACTION", "step": step,
                            "evidence_reads_consumed": 0, "ordinary_rejection_budget_consumed": False})
                    elif not any(a.category == "COMPLETE_TASK" for a in actions):
                        allowed_routes = {(str(world.task.api_docs[p["operation"].split(".")[0]][p["operation"].split(".")[1]]["path"]), "GET")
                                          for p in acquisition.pending["plan"]}
                        remaining = min(acquisition.budget-acquisition.reads,
                            acquisition.pending["read_allowance"]-acquisition.pending["attempts_used"])
                        if any((a.route, a.method) not in allowed_routes for a in actions) or len(actions) > remaining:
                            agent.observe("[COMPLETION_EVIDENCE_PENDING] Use only relevant READs, within the remaining "
                                f"{remaining} actual calls. Split the batch; no operation was executed.")
                            append_jsonl(audit_path, {"kind": "COMPLETION_EVIDENCE_PROPOSAL_DEFERRED", "step": step,
                                "ordinary_rejection_budget_consumed": False, "rollback_required": False})
                            continue
                proposal_operations = operation_contexts(
                    proposal.code,
                    actions,
                    context_signature=proposal_context_signature,
                )
                # A syntactic branch/loop site is not a witnessed invocation.
                # Use boundary fingerprints for its actual success/failure below;
                # never reject an alternative solely because another branch failed.
                control_flow = any(isinstance(n, (ast.If, ast.For))
                                   for n in ast.walk(ast.parse(proposal.code))) if not failures else False
                if control_flow:
                    proposal_operations = []
                repeated_failure_rejection = failure_ledger.equivalent_retry_rejection(
                    proposal_operations
                )
                # A bounded acquisition READ may fail or repeat without new evidence.
                # Charge the actual READ attempt to its dedicated budget instead of
                # consuming the ordinary rejection budget in the failure ledger.
                acquisition_read = (completion_verifier is not None and acquisition.pending
                    and actions and all(a.method == "GET" for a in actions))
                if repeated_failure_rejection and not acquisition_read:
                    failures.append(repeated_failure_rejection)
                step_metadata, metadata_failures = parse_harness_step_metadata(
                    proposal.code
                )
                if metadata_failures:
                    if invalid_harness_metadata_requests_replan(proposal.code):
                        failures.extend(
                            "Invalid decision rollback metadata: " + diagnostic
                            for diagnostic in metadata_failures
                        )
                    else:
                        # HARNESS_META groups checkpoints and recovery attempts; it is not
                        # an execution-safety boundary.  Invalid advisory metadata falls
                        # back to a narrow Harness-generated decision/action pair.  The
                        # actual code still passes every ordinary static and runtime check.
                        step_metadata = None
                        harness_metadata_advisories += 1
                        append_jsonl(audit_path, {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "kind": "HARNESS_META_ADVISORY",
                            "dataset": dataset,
                            "task_id": task_id,
                            "step": step,
                            "metadata_applied": False,
                            "diagnostics": metadata_failures,
                            "fallback": "harness_generated_narrow_implicit_decision",
                            "rejection_budget_consumed": False,
                            "rollback_requested_by_agent": False,
                            "rollback_request_accepted": None,
                            "rollback_request_rejection_reason": None,
                        })
                if requires_checkpoint(actions) and step_metadata is not None:
                    if step_metadata.node_type == "observation":
                        failures.append(
                            "A state-changing API cannot be marked as an observation"
                        )
                    rollback_request_rejection = (
                        decision_manager.decision_restore_request_rejection_reason(
                            step_metadata,
                            proposed_strategy_hash=failure_strategy_hash(proposal.code),
                        )
                    )
                    if rollback_request_rejection is not None:
                        failures.append(
                            "Decision rollback request rejected: "
                            + rollback_request_rejection
                        )
                proposal_namespace = getattr(getattr(world, "shell", None), "user_ns", None)
                output_findings = check_output(proposal.code, proposal_contracts, world.task.api_docs,
                    proposal_namespace if isinstance(proposal_namespace, Mapping) else None)
                presence_findings = [f for f in output_findings if f.check == "ANSWER_PRESENCE"]
                failures.extend(proposal_contract_failures(presence_findings))
                if presence_findings:
                    append_jsonl(audit_path, {"kind": "ANSWER_CONTRACT_CHECK", "dataset": dataset,
                        "task_id": task_id, "step": step, "status": finding_status(presence_findings),
                        "allowed": not proposal_contract_failures(presence_findings),
                        "answer_basis": proposal_contracts.output.answer_basis,
                        "findings": [f.audit() for f in presence_findings], "ground_truth_used": False})
                if any(action.category == "WRITE_RESOURCE" for action in actions):
                    semantic_write_checks += 1
                    shell = getattr(world, "shell", None)
                    semantic_namespace = getattr(shell, "user_ns", None)
                    if not isinstance(semantic_namespace, Mapping):
                        semantic_namespace = None
                    field_findings = check_write_fields(proposal.code, proposal_contracts,
                        world.task.api_docs, semantic_namespace)
                    semantic_failures = proposal_contract_failures(field_findings)
                    if semantic_failures:
                        semantic_write_rejections += 1
                        failures.extend(semantic_failures)
                    append_jsonl(audit_path, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "SEMANTIC_WRITE_CHECK",
                        "mechanism": "public_documented_field_contract",
                        "status": finding_status(field_findings),
                        "findings": [f.audit() for f in field_findings],
                        "dataset": dataset,
                        "task_id": task_id,
                        "step": step,
                        "allowed": not semantic_failures,
                        "error_type": (
                            "semantic_write_error" if semantic_failures else None
                        ),
                        "violations": semantic_failures,
                        "ground_truth_used": False,
                    })
                if completion_verifier is not None and target_tracker is not None and not failures:
                    target_tracker.synchronize_state_version(proposal_context_signature if proposal_context_signature != "state_unknown" else None)
                    preflight = final_semantic_preflight(target_tracker, completion_verifier, actions)
                    deferred = False
                    for check in preflight:
                        target_tracker.metrics["final_preflight_checks"] += 1
                        result = check.pop("result", None)
                        constraint_views = check.pop("constraints", None)
                        episodes = target_tracker.update_repair_episodes(result, completion_verifier,
                            state_version=proposal_context_signature, constraints=constraint_views) if result else []
                        target_tracker.metrics["final_preflight_violations"] += int(check["status"] == "VIOLATED")
                        target_tracker.metrics["final_preflight_unknown"] += int(check["status"] in {"UNKNOWN", "EVIDENCE_PENDING"})
                        append_jsonl(audit_path, {"kind": "FINAL_SEMANTIC_PREFLIGHT", "step": step, **check,
                            "verification": result.audit_dict() if result else None, "repair_episodes": episodes,
                            "ordinary_rejection_budget_consumed": False, "rollback_required": False})
                        if result and result.blocks_completion:
                            target_tracker.metrics["completion_final_status"] = "VIOLATED"
                            agent.observe(result.feedback())
                            deferred = True
                        elif check["status"] == "EVIDENCE_PENDING":
                            agent.observe("[FINAL_SEMANTIC_PREFLIGHT] Separate mutation and finalize into different "
                                "steps and READ the resulting collection. Pre-step evidence cannot prove post-write state.")
                            deferred = True
                        elif check["status"] == "UNKNOWN":
                            agent.observe("[FINAL_SEMANTIC_PREFLIGHT] Verification is UNKNOWN; no confirmed violation. "
                                "Read the relevant scope when possible; this advisory does not block the operation.")
                    if deferred:
                        continue
                if any(action.category == "COMPLETE_TASK" for action in actions):
                    completion_format_checks += 1
                    shell = getattr(world, "shell", None)
                    namespace = getattr(shell, "user_ns", None)
                    if not isinstance(namespace, Mapping):
                        namespace = None
                    order_findings = [f for f in output_findings if f.check == "ANSWER_ORDER"]
                    format_failures = proposal_contract_failures(order_findings)
                    format_status = finding_status(order_findings)
                    if format_failures:
                        completion_format_rejections += 1
                        failures.extend(format_failures)
                    append_jsonl(audit_path, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "COMPLETION_FORMAT_CHECK",
                        "status": format_status,
                        "unknown_reason": [f.reason for f in order_findings if f.status == "UNKNOWN"] or None,
                        "mechanism": "generic_public_answer_order",
                        "findings": [f.audit() for f in order_findings],
                        "dataset": dataset,
                        "task_id": task_id,
                        "step": step,
                        "allowed": not format_failures,
                        "error_type": (
                            "answer_format_error" if format_failures else None
                        ),
                        "violations": format_failures,
                        "ground_truth_used": False,
                    })
                    semantic_completion_checks += 1
                    semantic_completion_error = semantic_completion_failure(
                        actions,
                        world.task.instruction,
                        verified_write_count=verified_write_count,
                        last_verified_write_step=last_verified_write_step,
                        last_verified_read_step=last_verified_read_step,
                    )
                    if semantic_completion_error and completion_verifier is None:
                        semantic_completion_rejections += 1
                        failures.append(semantic_completion_error)
                    append_jsonl(audit_path, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "SEMANTIC_COMPLETION_CHECK",
                        "advisory_only": completion_verifier is not None,
                        "dataset": dataset,
                        "task_id": task_id,
                        "step": step,
                        "allowed": semantic_completion_error is None or completion_verifier is not None,
                        "error_type": (
                            "semantic_write_error"
                            if semantic_completion_error else None
                        ),
                        "reason": semantic_completion_error,
                        "verified_write_count": verified_write_count,
                        "last_verified_write_step": last_verified_write_step,
                        "last_verified_read_step": last_verified_read_step,
                        "ground_truth_used": False,
                    })
                    completion_coverage_checks += 1
                    coverage_advisory = completion_coverage_advisory(
                        actions,
                        world.task.instruction,
                        bulk_target_count,
                        verified_write_count,
                    )
                    if coverage_advisory or (
                        bulk_completion_requested(world.task.instruction)
                        and not coverage_gate_enforced
                    ):
                        completion_coverage_advisories += 1
                    append_jsonl(audit_path, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "COMPLETION_COVERAGE_CHECK",
                        "dataset": dataset,
                        "task_id": task_id,
                        "step": step,
                        "bulk_requested": bulk_completion_requested(
                            world.task.instruction
                        ),
                        "observed_target_count": bulk_target_count,
                        "coverage_evidence_source": coverage_evidence_source,
                        "coverage_evidence_confidence": coverage_evidence_confidence,
                        "candidate_collection_count": candidate_collection_count,
                        "verified_target_count": bulk_target_count,
                        "coverage_gate_enforced": coverage_gate_enforced,
                        "coverage_gate_reason": coverage_gate_reason,
                        "verified_write_count": verified_write_count,
                        "allowed": True,
                        "status": "ADVISORY_ONLY",
                        "reason": coverage_advisory,
                        "ordinary_rejection_budget_consumed": False,
                        "rollback_required": False,
                        "rollback_budget_consumed": False,
                    })
                if (completion_verifier is not None and target_tracker is not None and not failures
                        and any(action.category == "COMPLETE_TASK" for action in actions)):
                    current_runtime_stage = "completion_verification"
                    target_tracker.metrics["final_reconciliation_checks"] += 1
                    verification, plan = check_completion(step, "final_reconciliation", proposal_context_signature, proposal.code)
                    completion_candidate_status = assess_acquisition(verification, plan, proposal_context_signature, step)
                    if completion_candidate_status in {"VIOLATED", "COMPLETION_EVIDENCE_PENDING"}:
                        append_jsonl(audit_path, {"kind": "STEP_DECISION", "dataset": dataset,
                            "task_id": task_id, "step": step, "code_sha256": code_digest,
                            "decision": {"allowed": False, "reason": completion_candidate_status,
                                "gate_mode": "completion_verification"},
                            "ordinary_rejection_budget_consumed": False, "rollback_required": False})
                        if verification.blocks_completion:
                            target_tracker.metrics["completion_final_status"] = "VIOLATED"
                        continue
                decision = gate.decide(state, actions, failures, max_steps - step + 1)
                append_jsonl(audit_path, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": "STEP_DECISION",
                    "dataset": dataset,
                    "task_id": task_id,
                    "step": step,
                    "code_sha256": hashlib.sha256(proposal.code.encode()).hexdigest(),
                    "reasoning_sha256": proposal.reasoning_sha256,
                    "harness_metadata": (
                        {
                            "node_type": step_metadata.node_type,
                            "decision_id": step_metadata.decision_id,
                            "action_id": step_metadata.action_id,
                            "recovery_intent": step_metadata.recovery_intent,
                        }
                        if step_metadata is not None else None
                    ),
                    "decision": decision.model_dump(),
                })
                if not decision.allowed:
                    rejected += 1
                    precheck_rejections += 1
                    agent.reject(decision.reason)
                    write_failure_event(
                        audit_path,
                        dataset=dataset,
                        task_id=task_id,
                        step=step,
                        failure_type="PRECHECK_REJECTED",
                        reason=decision.reason,
                        state_changed=False,
                        rollback_required=False,
                        rollback_level=None,
                        rollback_verified=None,
                        decision_id=(step_metadata.decision_id if step_metadata else None),
                        action_id=(step_metadata.action_id if step_metadata else None),
                        api_signature=action_api_signature(actions),
                        failure_signature=None,
                        rollback_attempts=rollback_attempts,
                        rollback_requested_by_agent=bool(
                            step_metadata is not None
                            and step_metadata.recovery_intent == "replan_decision"
                        ),
                        rollback_request_accepted=(
                            False
                            if step_metadata is not None
                            and step_metadata.recovery_intent == "replan_decision"
                            else None
                        ),
                        rollback_request_rejection_reason=(
                            decision.reason
                            if step_metadata is not None
                            and step_metadata.recovery_intent == "replan_decision"
                            else None
                        ),
                        rollback_budget_exhausted=rollback_budget.exhausted,
                    )
                    if not rejection_budget.reject():
                        stop_reason = "Harness rejection budget exhausted"
                        break
                    continue

                checkpoint_id: str | None = None
                checkpoint_fingerprint: str | None = None
                effective_metadata: HarnessStepMetadata | None = None
                if requires_checkpoint(actions):
                    if target_tracker is not None:
                        target_tracker.invalidate_state()
                    current_runtime_stage = "checkpoint_create"
                    effective_metadata = decision_manager.effective_metadata(
                        step_metadata, step=step
                    )
                    assert effective_metadata.decision_id is not None
                    assert effective_metadata.action_id is not None

                    if decision_manager.needs_decision_restore(
                        effective_metadata,
                        proposed_strategy_hash=failure_strategy_hash(proposal.code),
                    ):
                        requested_target = decision_manager.requested_decision_target(
                            effective_metadata
                        )
                        repeated_decision_failures += 1
                        current_fingerprint = world_state_fingerprint(world)
                        replan_state_changed = (
                            current_fingerprint
                            != requested_target.checkpoint_fingerprint
                        )
                        if not replan_state_changed:
                            decision_manager.abandon_active_without_restore()
                            write_failure_event(
                                audit_path,
                                dataset=dataset,
                                task_id=task_id,
                                step=step,
                                failure_type="REPEATED_DECISION_FAILURE",
                                reason=(
                                    "Agent abandoned the failed decision; live state already "
                                    "matched its checkpoint"
                                ),
                                state_changed=False,
                                rollback_required=False,
                                rollback_level=None,
                                rollback_verified=None,
                                decision_id=requested_target.decision_id,
                                action_id=requested_target.action_id,
                                api_signature=None,
                                failure_signature=None,
                                rollback_attempts=rollback_attempts,
                                rollback_requested_by_agent=True,
                                rollback_request_accepted=True,
                                rollback_budget_exhausted=rollback_budget.exhausted,
                            )
                        else:
                            if rollback_budget.exhausted:
                                rejected += 1
                                precheck_rejections += 1
                                reason = (
                                    "Decision rollback request rejected: total rollback "
                                    "budget is exhausted, but the current environment is "
                                    "still safe. Continue from it with a different resource, "
                                    "parameters, or strategy."
                                )
                                agent.reject(reason)
                                write_failure_event(
                                    audit_path,
                                    dataset=dataset,
                                    task_id=task_id,
                                    step=step,
                                    failure_type="PRECHECK_REJECTED",
                                    reason=reason,
                                    state_changed=True,
                                    rollback_required=False,
                                    rollback_level=None,
                                    rollback_verified=None,
                                    decision_id=requested_target.decision_id,
                                    action_id=requested_target.action_id,
                                    api_signature=None,
                                    failure_signature=None,
                                    rollback_attempts=rollback_attempts,
                                    rollback_requested_by_agent=True,
                                    rollback_request_accepted=False,
                                    rollback_request_rejection_reason=reason,
                                    rollback_budget_exhausted=True,
                                    task_terminated_due_to_rollback_budget=False,
                                )
                                continue
                            rollback_attempts += 1
                            restored, restored_fingerprint, rollback_error = (
                                restore_checkpoint(
                                    world,
                                    requested_target.checkpoint_id,
                                    requested_target.checkpoint_fingerprint,
                                )
                            )
                            if restored:
                                rollback_successes += 1
                                if requested_target.restored_abstract_state is not None:
                                    state = requested_target.restored_abstract_state
                            else:
                                rollback_failures += 1
                            within_decision_budget = decision_manager.record_restore(
                                requested_target, restored=restored
                            )
                            write_rollback_event(
                                audit_path,
                                dataset=dataset,
                                task_id=task_id,
                                step=step,
                                checkpoint_id=requested_target.checkpoint_id,
                                expected_fingerprint=requested_target.checkpoint_fingerprint,
                                restored=restored,
                                restored_fingerprint=restored_fingerprint,
                                cause="agent_replan",
                                reason="Agent requested a new decision after an action failure",
                                error=rollback_error,
                                rollback_scope="decision",
                                decision_id=requested_target.decision_id,
                                action_id=requested_target.action_id,
                                consecutive_action_failures=(
                                    requested_target.consecutive_action_failures
                                ),
                                agent_requested=True,
                            )
                            if restored:
                                rollback_budget.record("agent_replan")
                            write_failure_event(
                                audit_path,
                                dataset=dataset,
                                task_id=task_id,
                                step=step,
                                failure_type=(
                                    "REPEATED_DECISION_FAILURE"
                                    if restored else "ROLLBACK_FAILED"
                                ),
                                reason="Agent requested a new decision after an action failure",
                                state_changed=True,
                                rollback_required=True,
                                rollback_level="decision",
                                rollback_verified=restored,
                                decision_id=requested_target.decision_id,
                                action_id=requested_target.action_id,
                                api_signature=None,
                                failure_signature=None,
                                rollback_attempts=rollback_attempts,
                                rollback_requested_by_agent=True,
                                rollback_request_accepted=True,
                                rollback_budget_exhausted=rollback_budget.exhausted,
                            )
                            if not restored:
                                error = f"Decision rollback failed: {rollback_error}"
                                stop_reason = "Rollback failure"
                                break
                            # Reaching either rollback limit disables future restores. The
                            # successfully restored environment remains safe to continue.
                            _ = within_decision_budget

                    if (
                        decision_manager.active is not None
                        and effective_metadata.decision_id
                        != decision_manager.active.decision_id
                    ):
                        committed = decision_manager.commit_active()
                        assert committed is not None
                        write_decision_event(
                            audit_path,
                            kind="DECISION_COMMITTED",
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            decision_id=committed.decision_id,
                            action_id=None,
                            reason="Agent opened a new decision after verified progress",
                            metadata_explicit=committed.metadata_explicit,
                        )

                    if decision_manager.active is None:
                        decision_checkpoint_id = f"harness_decision_{step:03d}_before"
                        decision_fingerprint = world_state_fingerprint(world)
                        saved_id = world._harness_runtime_checkpoints.save(decision_checkpoint_id, "decision")
                        if saved_id != decision_checkpoint_id:
                            raise RuntimeError(
                                "AppWorld did not confirm decision checkpoint "
                                f"{decision_checkpoint_id!r}"
                            )
                        checkpointed += 1
                        decision_manager.open(
                            decision_id=effective_metadata.decision_id,
                            checkpoint_id=decision_checkpoint_id,
                            checkpoint_fingerprint=decision_fingerprint,
                            step=step,
                            abstract_state=state,
                            metadata_explicit=effective_metadata.explicit,
                        )
                        write_decision_event(
                            audit_path,
                            kind="DECISION_OPENED",
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            decision_id=effective_metadata.decision_id,
                            action_id=effective_metadata.action_id,
                            reason="Decision checkpoint saved before its first action",
                            metadata_explicit=effective_metadata.explicit,
                        )

                    checkpoint_id = f"harness_step_{step:03d}_before"
                    checkpoint_fingerprint = world_state_fingerprint(world)
                    saved_id = world._harness_runtime_checkpoints.save(checkpoint_id, "action")
                    if saved_id != checkpoint_id:
                        raise RuntimeError(
                            f"AppWorld did not confirm checkpoint {checkpoint_id!r}"
                        )
                    checkpointed += 1
                    append_jsonl(audit_path, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "CHECKPOINT_CREATED",
                        "dataset": dataset,
                        "task_id": task_id,
                        "step": step,
                        "checkpoint_id": checkpoint_id,
                        "state_fingerprint": checkpoint_fingerprint,
                        "methods": [action.method for action in actions],
                        "checkpoint_level": "action",
                        "decision_id": effective_metadata.decision_id,
                        "action_id": effective_metadata.action_id,
                    })

                before = _runtime_api_calls(world, api_log)
                target_mutation_before = (
                    target_tracker.mutation_attempted if target_tracker is not None else False
                )
                executed += 1
                tool_evidence = ToolEvidenceCapture(
                    world, enabled=task_target_tracking, max_calls=MAX_API_CALLS_PER_STEP,
                    context_signature=proposal_context_signature)
                active_tool_evidence = tool_evidence
                evidence_dispatches = []
                if completion_verifier is not None and acquisition.pending:
                    def validate_evidence_attempt(attempt):
                        if not acquisition.related(attempt["operation"], attempt["arguments"]):
                            raise RuntimeApiBudgetError("Completion evidence request is outside the pending scope; use the documented bound arguments")
                        if acquisition.reads >= acquisition.budget:
                            raise RuntimeApiBudgetError("Completion evidence READ budget exhausted; no extra READ dispatched")
                    def consume_evidence_dispatch(attempt):
                        # Memory only: this runs inside world.execute, never writes files.
                        consumed = acquisition.consume([attempt])
                        target_tracker.metrics["completion_evidence_reads"] = acquisition.reads
                        if consumed:
                            evidence_dispatches.append({"kind": "COMPLETION_EVIDENCE_READ", "step": step,
                                "stage": "dispatched", "operation": attempt["operation"],
                                "attempts_used": acquisition.reads,
                                "evidence_gap_key": acquisition.pending["evidence_gap_key"],
                                "ordinary_rejection_budget_consumed": False, "rollback_required": False})
                    tool_evidence.on_attempt = validate_evidence_attempt
                    tool_evidence.on_dispatch = consume_evidence_dispatch
                try:
                    current_runtime_stage = "action_execution"
                    try:
                        with tool_evidence:
                            output = normalize_execution_output(world.execute(proposal.code))
                    finally:
                        # The executor has returned/raised. Audit side effects are outside it.
                        for event in evidence_dispatches:
                            append_jsonl(audit_path, event)
                        if target_tracker is not None:
                            for attempt in tool_evidence.attempts:
                                target_tracker.record_read_attempt(attempt)
                            flush_evidence_events(step)
                except Exception as exc:
                    if control_flow:
                        proposal_operations = list(tool_evidence.operation_contexts)
                    current_runtime_stage = "action_failure_recovery"
                    execution_failures += 1
                    execution_error = safe_runtime_exception_summary(exc)
                    reason = execution_error
                    after = _runtime_api_calls(world, api_log)
                    new_calls = _new_runtime_api_calls(before, after)
                    api_calls_executed += len(new_calls)
                    if checkpoint_id is None:
                        no_mutation_failures += 1
                        try:
                            current_nonwrite_fingerprint = world_state_fingerprint(world)
                            nonwrite_state_changed = (
                                None if proposal_context_signature == "state_unknown"
                                else current_nonwrite_fingerprint != proposal_context_signature
                            )
                        except Exception:
                            nonwrite_state_changed = None
                        evidence_feedback = record_failure_evidence(
                            failure_ledger, audit_path,
                            dataset=dataset, task_id=task_id, step=step,
                            decision_id=None, action_id=None,
                            operations=proposal_operations,
                            executed_call_count=len(new_calls),
                            reason=reason, cause="execution_error",
                            state_changed=nonwrite_state_changed,
                            rollback_required=False,
                            rollback_level=None, rollback_verified=None,
                        )
                        write_failure_event(
                            audit_path,
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            failure_type="EXECUTION_NO_MUTATION",
                            reason=reason,
                            state_changed=nonwrite_state_changed,
                            rollback_required=False,
                            rollback_level=None,
                            rollback_verified=None,
                            decision_id=None,
                            action_id=None,
                            api_signature=action_api_signature(actions),
                            failure_signature=None,
                            rollback_attempts=rollback_attempts,
                            rollback_budget_exhausted=rollback_budget.exhausted,
                        )
                        agent.recover(
                            reason,
                            rollback_level="none",
                            failure_type="EXECUTION_NO_MUTATION",
                            state_changed=nonwrite_state_changed,
                            rollback_required=False,
                            rollback_verified=None,
                            api_signature=action_api_signature(actions),
                            failure_evidence_feedback=evidence_feedback,
                        )
                        continue
                    assert checkpoint_fingerprint is not None
                    assert effective_metadata is not None
                    assert effective_metadata.action_id is not None
                    try:
                        post_execution_fingerprint = world_state_fingerprint(world)
                        state_changed = (
                            post_execution_fingerprint != checkpoint_fingerprint
                        )
                    except Exception:
                        post_execution_fingerprint = None
                        state_changed = None
                    signature = action_failure_signature(
                        decision_id=effective_metadata.decision_id or "unknown_decision",
                        actions=actions,
                        code=proposal.code,
                        reason=reason,
                        cause="execution_error",
                    )
                    signature_digest = failure_signature_digest(signature)
                    rollback_target = decision_manager.failure_target(
                        action_checkpoint_id=checkpoint_id,
                        action_checkpoint_fingerprint=checkpoint_fingerprint,
                        action_id=effective_metadata.action_id,
                        failure_signature=signature,
                    )
                    if rollback_target.automatic_escalation:
                        repeated_decision_failures += 1
                    rollback_required = failure_needs_restore(
                        state_changed=state_changed,
                        current_fingerprint=post_execution_fingerprint,
                        target=rollback_target,
                    )
                    failure_type = (
                        "REPEATED_DECISION_FAILURE"
                        if rollback_target.automatic_escalation else
                        "EXECUTION_NO_MUTATION"
                        if state_changed is False else
                        "EXECUTION_WITH_MUTATION"
                    )
                    if not rollback_required:
                        no_mutation_failures += 1
                        if rollback_target.rollback_level == "decision":
                            decision_manager.abandon_active_without_restore()
                        write_failure_event(
                            audit_path,
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            failure_type=failure_type,
                            reason=reason,
                            state_changed=False,
                            rollback_required=False,
                            rollback_level=None,
                            rollback_verified=None,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            api_signature=action_api_signature(actions),
                            failure_signature=signature_digest,
                            rollback_attempts=rollback_attempts,
                            rollback_budget_exhausted=rollback_budget.exhausted,
                        )
                        evidence_feedback = record_failure_evidence(
                            failure_ledger, audit_path,
                            dataset=dataset, task_id=task_id, step=step,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            operations=proposal_operations,
                            executed_call_count=len(new_calls),
                            reason=reason, cause="execution_error",
                            state_changed=False, rollback_required=False,
                            rollback_level=None, rollback_verified=None,
                        )
                        agent.recover(
                            reason,
                            rollback_level="none",
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            failure_type=failure_type,
                            state_changed=False,
                            rollback_required=False,
                            rollback_verified=None,
                            api_signature=action_api_signature(actions),
                            failure_evidence_feedback=evidence_feedback,
                        )
                        continue
                    if rollback_budget.exhausted:
                        task_terminated_due_to_rollback_budget = True
                        error = (
                            "Environment cannot be safely continued: execution failed "
                            "after a state mutation and rollback budget is exhausted"
                        )
                        stop_reason = "Unsafe mutated state with rollback unavailable"
                        write_failure_event(
                            audit_path,
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            failure_type="EXECUTION_WITH_MUTATION",
                            reason=reason,
                            state_changed=state_changed,
                            rollback_required=True,
                            rollback_level=rollback_target.rollback_level,
                            rollback_verified=False,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            api_signature=action_api_signature(actions),
                            failure_signature=signature_digest,
                            rollback_attempts=rollback_attempts,
                            rollback_budget_exhausted=True,
                            task_terminated_due_to_rollback_budget=True,
                            environment_safe_to_continue=False,
                        )
                        record_failure_evidence(
                            failure_ledger, audit_path,
                            dataset=dataset, task_id=task_id, step=step,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            operations=proposal_operations,
                            executed_call_count=len(new_calls),
                            reason=reason, cause="execution_error",
                            state_changed=state_changed, rollback_required=True,
                            rollback_level=rollback_target.rollback_level,
                            rollback_verified=False,
                        )
                        break
                    rollback_attempts += 1
                    restored, restored_fingerprint, rollback_error = restore_checkpoint(
                        world,
                        rollback_target.checkpoint_id,
                        rollback_target.checkpoint_fingerprint,
                    )
                    if restored:
                        rollback_successes += 1
                        if rollback_target.restored_abstract_state is not None:
                            state = rollback_target.restored_abstract_state
                    else:
                        rollback_failures += 1
                    within_decision_budget = decision_manager.record_restore(
                        rollback_target, restored=restored
                    )
                    write_rollback_event(
                        audit_path,
                        dataset=dataset,
                        task_id=task_id,
                        step=step,
                        checkpoint_id=rollback_target.checkpoint_id,
                        expected_fingerprint=rollback_target.checkpoint_fingerprint,
                        restored=restored,
                        restored_fingerprint=restored_fingerprint,
                        cause="execution_error",
                        reason="AppWorld raised during execution: " + execution_error,
                        error=rollback_error,
                        rollback_scope=rollback_target.rollback_level,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        consecutive_action_failures=(
                            rollback_target.consecutive_action_failures
                        ),
                        automatic_escalation=rollback_target.automatic_escalation,
                    )
                    if not restored:
                        error = f"Execution failed ({execution_error}); rollback failed ({rollback_error})"
                        stop_reason = "Rollback failure"
                        write_failure_event(
                            audit_path,
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            failure_type="ROLLBACK_FAILED",
                            reason=reason,
                            state_changed=state_changed,
                            rollback_required=True,
                            rollback_level=rollback_target.rollback_level,
                            rollback_verified=False,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            api_signature=action_api_signature(actions),
                            failure_signature=signature_digest,
                            rollback_attempts=rollback_attempts,
                            rollback_budget_exhausted=rollback_budget.exhausted,
                            environment_safe_to_continue=False,
                        )
                        record_failure_evidence(
                            failure_ledger, audit_path,
                            dataset=dataset, task_id=task_id, step=step,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            operations=proposal_operations,
                            executed_call_count=len(new_calls),
                            reason=reason, cause="execution_error",
                            state_changed=state_changed, rollback_required=True,
                            rollback_level=rollback_target.rollback_level,
                            rollback_verified=False,
                        )
                        break
                    rolled_back_codes.add(code_digest)
                    rollback_budget.record("execution_error")
                    write_failure_event(
                        audit_path,
                        dataset=dataset,
                        task_id=task_id,
                        step=step,
                        failure_type=failure_type,
                        reason=reason,
                        state_changed=state_changed,
                        rollback_required=True,
                        rollback_level=rollback_target.rollback_level,
                        rollback_verified=True,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        api_signature=action_api_signature(actions),
                        failure_signature=signature_digest,
                        rollback_attempts=rollback_attempts,
                        rollback_budget_exhausted=rollback_budget.exhausted,
                        environment_safe_to_continue=True,
                        physical_restore_performed=True,
                    )
                    evidence_feedback = record_failure_evidence(
                        failure_ledger, audit_path,
                        dataset=dataset, task_id=task_id, step=step,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        operations=proposal_operations,
                        executed_call_count=len(new_calls),
                        reason=reason, cause="execution_error",
                        state_changed=state_changed, rollback_required=True,
                        rollback_level=rollback_target.rollback_level,
                        rollback_verified=True,
                    )
                    _ = within_decision_budget
                    agent.recover(
                        reason,
                        rollback_level=rollback_target.rollback_level,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        failure_type=failure_type,
                        state_changed=state_changed,
                        rollback_required=True,
                        rollback_verified=True,
                        api_signature=action_api_signature(actions),
                        failure_evidence_feedback=evidence_feedback,
                    )
                    continue
                current_runtime_stage = "runtime_verification"
                if control_flow:
                    proposal_operations = list(tool_evidence.operation_contexts)
                after = _runtime_api_calls(world, api_log)
                new_calls = _new_runtime_api_calls(before, after)
                api_calls_executed += len(new_calls)
                verified, expected, observed, verification_error = execution_verification(
                    output, actions, new_calls
                )
                route_records = route_verification_records(actions, new_calls)
                if verified and isinstance(actions, CheckedActions):
                    actions = actions.resolve(new_calls, runtime_action_matches) or []
                post_execution_fingerprint: str | None = None
                state_changed: bool | None = None
                if checkpoint_fingerprint is not None:
                    post_execution_fingerprint = world_state_fingerprint(world)
                    state_changed = post_execution_fingerprint != checkpoint_fingerprint
                    # Successful idempotent/no-op writes are valid. A fingerprint
                    # reports mutation, not whether the task goal is satisfied.
                else:
                    try:
                        post_execution_fingerprint = world_state_fingerprint(world)
                        state_changed = (
                            None if proposal_context_signature == "state_unknown"
                            else post_execution_fingerprint != proposal_context_signature
                        )
                    except Exception:
                        state_changed = None
                append_jsonl(audit_path, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": "ENVIRONMENT_VERIFICATION",
                    "dataset": dataset,
                    "task_id": task_id,
                    "step": step,
                    "expected_categories": expected,
                    "observed_categories": observed,
                    "expected_route_template": [
                        record["expected_route_template"] for record in route_records
                    ],
                    "observed_route": [
                        record["observed_route"] for record in route_records
                    ],
                    "route_match_result": [
                        record["route_match_result"] for record in route_records
                    ],
                    "route_mismatch_reason": [
                        record["route_mismatch_reason"] for record in route_records
                    ],
                    "route_verification": route_records,
                    "verified": verified,
                    "verification_error": verification_error,
                    "checkpoint_id": checkpoint_id,
                    "pre_execution_state_fingerprint": checkpoint_fingerprint,
                    "post_execution_state_fingerprint": post_execution_fingerprint,
                    "state_changed": state_changed,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "output_chars": len(output),
                })
                if not verified:
                    failure_reason = verification_error or "Environment verification failed"
                    rollback_cause = verification_failure_cause(output)
                    if rollback_cause == "execution_error":
                        execution_failures += 1
                    else:
                        postcondition_failures += 1
                    if checkpoint_id is None:
                        no_mutation_failures += 1
                        evidence_feedback = record_failure_evidence(
                            failure_ledger, audit_path,
                            dataset=dataset, task_id=task_id, step=step,
                            decision_id=None, action_id=None,
                            operations=proposal_operations,
                            executed_call_count=len(new_calls),
                            reason=failure_reason, cause=rollback_cause,
                            state_changed=state_changed, rollback_required=False,
                            rollback_level=None, rollback_verified=None,
                        )
                        append_jsonl(audit_path, {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "kind": "RECOVERY_DECISION",
                            "dataset": dataset,
                            "task_id": task_id,
                            "step": step,
                            "action": "REPLAN_WITHOUT_ROLLBACK",
                            "reason": "The checked proposal had no state-changing HTTP method",
                        })
                        write_failure_event(
                            audit_path,
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            failure_type=(
                                "EXECUTION_NO_MUTATION"
                                if rollback_cause == "execution_error"
                                else "POSTCONDITION_FAILED"
                            ),
                            reason=failure_reason,
                            state_changed=state_changed,
                            rollback_required=False,
                            rollback_level=None,
                            rollback_verified=None,
                            decision_id=None,
                            action_id=None,
                            api_signature=action_api_signature(actions),
                            failure_signature=None,
                            rollback_attempts=rollback_attempts,
                            rollback_budget_exhausted=rollback_budget.exhausted,
                        )
                        agent.recover_observation(
                            output,
                            failure_reason,
                            failure_evidence_feedback=evidence_feedback,
                        )
                        continue

                    assert checkpoint_fingerprint is not None
                    assert effective_metadata is not None
                    assert effective_metadata.action_id is not None
                    signature = action_failure_signature(
                        decision_id=effective_metadata.decision_id or "unknown_decision",
                        actions=actions,
                        code=proposal.code,
                        reason=failure_reason,
                        cause=rollback_cause,
                    )
                    signature_digest = failure_signature_digest(signature)
                    rollback_target = decision_manager.failure_target(
                        action_checkpoint_id=checkpoint_id,
                        action_checkpoint_fingerprint=checkpoint_fingerprint,
                        action_id=effective_metadata.action_id,
                        failure_signature=signature,
                    )
                    if rollback_target.automatic_escalation:
                        repeated_decision_failures += 1
                    rollback_required = failure_needs_restore(
                        state_changed=state_changed,
                        current_fingerprint=post_execution_fingerprint,
                        target=rollback_target,
                    )
                    failure_type = (
                        "REPEATED_DECISION_FAILURE"
                        if rollback_target.automatic_escalation
                        else "EXECUTION_NO_MUTATION"
                        if rollback_cause == "execution_error" and state_changed is False
                        else "EXECUTION_WITH_MUTATION"
                        if rollback_cause == "execution_error"
                        else "POSTCONDITION_FAILED"
                    )
                    if not rollback_required:
                        no_mutation_failures += int(state_changed is False)
                        if rollback_target.rollback_level == "decision":
                            decision_manager.abandon_active_without_restore()
                        write_failure_event(
                            audit_path,
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            failure_type=failure_type,
                            reason=failure_reason,
                            state_changed=state_changed,
                            rollback_required=False,
                            rollback_level=None,
                            rollback_verified=None,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            api_signature=action_api_signature(actions),
                            failure_signature=signature_digest,
                            rollback_attempts=rollback_attempts,
                            rollback_budget_exhausted=rollback_budget.exhausted,
                        )
                        evidence_feedback = record_failure_evidence(
                            failure_ledger, audit_path,
                            dataset=dataset, task_id=task_id, step=step,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            operations=proposal_operations,
                            executed_call_count=len(new_calls),
                            reason=failure_reason, cause=rollback_cause,
                            state_changed=state_changed, rollback_required=False,
                            rollback_level=None, rollback_verified=None,
                        )
                        agent.recover(
                            failure_reason,
                            rollback_level="none",
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            failure_type=failure_type,
                            state_changed=state_changed,
                            rollback_required=False,
                            rollback_verified=None,
                            api_signature=action_api_signature(actions),
                            failure_evidence_feedback=evidence_feedback,
                        )
                        continue
                    if rollback_budget.exhausted:
                        # A repeated failure may request a broader rewind even though the
                        # failed action itself made no mutation. In that case the current
                        # verified state remains usable; disable rollback and let the agent
                        # choose a different strategy.
                        if state_changed is False:
                            write_failure_event(
                                audit_path,
                                dataset=dataset,
                                task_id=task_id,
                                step=step,
                                failure_type=failure_type,
                                reason=failure_reason,
                                state_changed=False,
                                rollback_required=False,
                                rollback_level=None,
                                rollback_verified=None,
                                decision_id=rollback_target.decision_id,
                                action_id=rollback_target.action_id,
                                api_signature=action_api_signature(actions),
                                failure_signature=signature_digest,
                                rollback_attempts=rollback_attempts,
                                rollback_budget_exhausted=True,
                                task_terminated_due_to_rollback_budget=False,
                                environment_safe_to_continue=True,
                            )
                            evidence_feedback = record_failure_evidence(
                                failure_ledger, audit_path,
                                dataset=dataset, task_id=task_id, step=step,
                                decision_id=rollback_target.decision_id,
                                action_id=rollback_target.action_id,
                                operations=proposal_operations,
                                executed_call_count=len(new_calls),
                                reason=failure_reason, cause=rollback_cause,
                                state_changed=False, rollback_required=False,
                                rollback_level=None, rollback_verified=None,
                            )
                            agent.recover(
                                failure_reason
                                + "; rollback budget is exhausted, so continue from the "
                                "current safe state with a different resource or strategy",
                                rollback_level="none",
                                decision_id=rollback_target.decision_id,
                                action_id=rollback_target.action_id,
                                failure_type=failure_type,
                                state_changed=False,
                                rollback_required=False,
                                rollback_verified=None,
                                api_signature=action_api_signature(actions),
                                failure_evidence_feedback=evidence_feedback,
                            )
                            continue
                        task_terminated_due_to_rollback_budget = True
                        error = (
                            "Environment cannot be safely continued: post-execution "
                            "verification failed after a state mutation and rollback is "
                            "unavailable"
                        )
                        stop_reason = "Unsafe mutated state with rollback unavailable"
                        write_failure_event(
                            audit_path,
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            failure_type=failure_type,
                            reason=failure_reason,
                            state_changed=state_changed,
                            rollback_required=True,
                            rollback_level=rollback_target.rollback_level,
                            rollback_verified=False,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            api_signature=action_api_signature(actions),
                            failure_signature=signature_digest,
                            rollback_attempts=rollback_attempts,
                            rollback_budget_exhausted=True,
                            task_terminated_due_to_rollback_budget=True,
                            environment_safe_to_continue=False,
                        )
                        record_failure_evidence(
                            failure_ledger, audit_path,
                            dataset=dataset, task_id=task_id, step=step,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            operations=proposal_operations,
                            executed_call_count=len(new_calls),
                            reason=failure_reason, cause=rollback_cause,
                            state_changed=state_changed, rollback_required=True,
                            rollback_level=rollback_target.rollback_level,
                            rollback_verified=False,
                        )
                        break
                    rollback_attempts += 1
                    restored, restored_fingerprint, rollback_error = restore_checkpoint(
                        world,
                        rollback_target.checkpoint_id,
                        rollback_target.checkpoint_fingerprint,
                    )
                    if restored:
                        rollback_successes += 1
                        if rollback_target.restored_abstract_state is not None:
                            state = rollback_target.restored_abstract_state
                    else:
                        rollback_failures += 1
                    within_decision_budget = decision_manager.record_restore(
                        rollback_target, restored=restored
                    )
                    write_rollback_event(
                        audit_path,
                        dataset=dataset,
                        task_id=task_id,
                        step=step,
                        checkpoint_id=rollback_target.checkpoint_id,
                        expected_fingerprint=rollback_target.checkpoint_fingerprint,
                        restored=restored,
                        restored_fingerprint=restored_fingerprint,
                        cause=rollback_cause,
                        reason=verification_error or "Environment verification failed",
                        error=rollback_error,
                        rollback_scope=rollback_target.rollback_level,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        consecutive_action_failures=(
                            rollback_target.consecutive_action_failures
                        ),
                        automatic_escalation=rollback_target.automatic_escalation,
                    )
                    if not restored:
                        error = f"Rollback verification failed: {rollback_error}"
                        stop_reason = "Rollback failure"
                        write_failure_event(
                            audit_path,
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            failure_type="ROLLBACK_FAILED",
                            reason=failure_reason,
                            state_changed=state_changed,
                            rollback_required=True,
                            rollback_level=rollback_target.rollback_level,
                            rollback_verified=False,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            api_signature=action_api_signature(actions),
                            failure_signature=signature_digest,
                            rollback_attempts=rollback_attempts,
                            rollback_budget_exhausted=rollback_budget.exhausted,
                            environment_safe_to_continue=False,
                        )
                        record_failure_evidence(
                            failure_ledger, audit_path,
                            dataset=dataset, task_id=task_id, step=step,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            operations=proposal_operations,
                            executed_call_count=len(new_calls),
                            reason=failure_reason, cause=rollback_cause,
                            state_changed=state_changed, rollback_required=True,
                            rollback_level=rollback_target.rollback_level,
                            rollback_verified=False,
                        )
                        break
                    rolled_back_codes.add(code_digest)
                    rollback_budget.record(rollback_cause)
                    write_failure_event(
                        audit_path,
                        dataset=dataset,
                        task_id=task_id,
                        step=step,
                        failure_type=failure_type,
                        reason=failure_reason,
                        state_changed=state_changed,
                        rollback_required=True,
                        rollback_level=rollback_target.rollback_level,
                        rollback_verified=True,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        api_signature=action_api_signature(actions),
                        failure_signature=signature_digest,
                        rollback_attempts=rollback_attempts,
                        rollback_budget_exhausted=rollback_budget.exhausted,
                        environment_safe_to_continue=True,
                        physical_restore_performed=True,
                    )
                    evidence_feedback = record_failure_evidence(
                        failure_ledger, audit_path,
                        dataset=dataset, task_id=task_id, step=step,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        operations=proposal_operations,
                        executed_call_count=len(new_calls),
                        reason=failure_reason, cause=rollback_cause,
                        state_changed=state_changed, rollback_required=True,
                        rollback_level=rollback_target.rollback_level,
                        rollback_verified=True,
                    )
                    _ = within_decision_budget
                    agent.recover(
                        failure_reason,
                        rollback_level=rollback_target.rollback_level,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        failure_type=failure_type,
                        state_changed=state_changed,
                        rollback_required=True,
                        rollback_verified=True,
                        api_signature=action_api_signature(actions),
                        failure_evidence_feedback=evidence_feedback,
                    )
                    continue
                post_step_decision: StepDecision | None = None
                next_state = state
                current_runtime_stage = "post_step_evaluation"
                for action_index, action in enumerate(actions):
                    post_step_checks += 1
                    safe_post_attributes = derive_safe_post_attributes(
                        output, action.category
                    )
                    action_decision = gate.evaluate_completed_step(
                        next_state,
                        action,
                        max_steps - step,
                        safe_post_attributes,
                    )
                    append_jsonl(audit_path, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "POST_STEP_PROBABILITY",
                        "dataset": dataset,
                        "task_id": task_id,
                        "step": step,
                        "action_index": action_index,
                        "action_count": len(actions),
                        "prior_state": next_state,
                        "completed_action": action.category,
                        "resulting_state": _state_after(action.category),
                        "safe_post_attributes": safe_post_attributes,
                        "decision": action_decision.model_dump(),
                    })
                    post_step_decision = action_decision
                    if not action_decision.allowed:
                        break
                    next_state = _state_after(action.category)
                if post_step_decision is not None and not post_step_decision.allowed:
                    postcondition_failures += 1
                    if checkpoint_id is None or checkpoint_fingerprint is None:
                        error = "Post-step gate rejected a step without a recoverable checkpoint"
                        stop_reason = "Rollback unavailable"
                        break
                    assert effective_metadata is not None
                    assert effective_metadata.action_id is not None
                    signature = action_failure_signature(
                        decision_id=effective_metadata.decision_id or "unknown_decision",
                        actions=actions,
                        code=proposal.code,
                        reason=post_step_decision.reason,
                        cause="probability_rejection",
                    )
                    signature_digest = failure_signature_digest(signature)
                    rollback_target = decision_manager.failure_target(
                        action_checkpoint_id=checkpoint_id,
                        action_checkpoint_fingerprint=checkpoint_fingerprint,
                        action_id=effective_metadata.action_id,
                        failure_signature=signature,
                    )
                    if rollback_target.automatic_escalation:
                        repeated_decision_failures += 1
                    failure_type = (
                        "REPEATED_DECISION_FAILURE"
                        if rollback_target.automatic_escalation
                        else "POSTCONDITION_FAILED"
                    )
                    if rollback_budget.exhausted:
                        task_terminated_due_to_rollback_budget = True
                        error = (
                            "Environment cannot be safely continued: probability gate "
                            "rejected a committed mutation and rollback is unavailable"
                        )
                        stop_reason = "Unsafe mutated state with rollback unavailable"
                        write_failure_event(
                            audit_path,
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            failure_type=failure_type,
                            reason=post_step_decision.reason,
                            state_changed=True,
                            rollback_required=True,
                            rollback_level=rollback_target.rollback_level,
                            rollback_verified=False,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            api_signature=action_api_signature(actions),
                            failure_signature=signature_digest,
                            rollback_attempts=rollback_attempts,
                            rollback_budget_exhausted=True,
                            task_terminated_due_to_rollback_budget=True,
                        )
                        break
                    rollback_attempts += 1
                    restored, restored_fingerprint, rollback_error = restore_checkpoint(
                        world,
                        rollback_target.checkpoint_id,
                        rollback_target.checkpoint_fingerprint,
                    )
                    if restored:
                        rollback_successes += 1
                        if rollback_target.restored_abstract_state is not None:
                            state = rollback_target.restored_abstract_state
                    else:
                        rollback_failures += 1
                    within_decision_budget = decision_manager.record_restore(
                        rollback_target, restored=restored
                    )
                    write_rollback_event(
                        audit_path,
                        dataset=dataset,
                        task_id=task_id,
                        step=step,
                        checkpoint_id=rollback_target.checkpoint_id,
                        expected_fingerprint=rollback_target.checkpoint_fingerprint,
                        restored=restored,
                        restored_fingerprint=restored_fingerprint,
                        cause="probability_rejection",
                        reason=post_step_decision.reason,
                        error=rollback_error,
                        rollback_scope=rollback_target.rollback_level,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        consecutive_action_failures=(
                            rollback_target.consecutive_action_failures
                        ),
                        automatic_escalation=rollback_target.automatic_escalation,
                    )
                    if not restored:
                        error = f"Post-step probability rejection; rollback failed: {rollback_error}"
                        stop_reason = "Rollback failure"
                        write_failure_event(
                            audit_path,
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            failure_type="ROLLBACK_FAILED",
                            reason=post_step_decision.reason,
                            state_changed=True,
                            rollback_required=True,
                            rollback_level=rollback_target.rollback_level,
                            rollback_verified=False,
                            decision_id=rollback_target.decision_id,
                            action_id=rollback_target.action_id,
                            api_signature=action_api_signature(actions),
                            failure_signature=signature_digest,
                            rollback_attempts=rollback_attempts,
                            rollback_budget_exhausted=rollback_budget.exhausted,
                        )
                        break
                    rolled_back_codes.add(code_digest)
                    rollback_budget.record("probability_rejection")
                    write_failure_event(
                        audit_path,
                        dataset=dataset,
                        task_id=task_id,
                        step=step,
                        failure_type=failure_type,
                        reason=post_step_decision.reason,
                        state_changed=True,
                        rollback_required=True,
                        rollback_level=rollback_target.rollback_level,
                        rollback_verified=True,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        api_signature=action_api_signature(actions),
                        failure_signature=signature_digest,
                        rollback_attempts=rollback_attempts,
                        rollback_budget_exhausted=rollback_budget.exhausted,
                    )
                    _ = within_decision_budget
                    agent.recover(
                        post_step_decision.reason,
                        rollback_level=rollback_target.rollback_level,
                        decision_id=rollback_target.decision_id,
                        action_id=rollback_target.action_id,
                        failure_type=failure_type,
                        state_changed=True,
                        rollback_required=True,
                        rollback_verified=True,
                        api_signature=action_api_signature(actions),
                    )
                    continue
                if actions:
                    # A verified read/authentication/write can all be meaningful
                    # progress after a failure; any such progress breaks the
                    # consecutive-equivalent-failure sequence.
                    if decision_manager.active is not None:
                        decision_manager.record_action_success()
                    state = next_state
                    verified_write_count += sum(
                        action.category == "WRITE_RESOURCE" for action in actions
                    )
                    if any(action.category == "WRITE_RESOURCE" for action in actions):
                        last_verified_write_step = step
                    if any(action.category == "READ_RESOURCE" for action in actions):
                        last_verified_read_step = step
                    if (
                        bulk_completion_requested(world.task.instruction)
                        and len(actions) == 1
                        and actions[0].category == "READ_RESOURCE"
                    ):
                        coverage = target_collection_evidence(
                            world.task.instruction,
                            actions,
                            output,
                        )
                        candidate_collection_count = coverage.candidate_collection_count
                        append_jsonl(audit_path, {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "kind": "COVERAGE_EVIDENCE",
                            "dataset": dataset,
                            "task_id": task_id,
                            "step": step,
                            "coverage_evidence_source": coverage.source,
                            "coverage_evidence_confidence": coverage.confidence,
                            "candidate_collection_count": coverage.candidate_collection_count,
                            "verified_target_count": coverage.verified_target_count,
                            "coverage_gate_enforced": coverage.gate_enforced,
                            "coverage_gate_reason": coverage.reason,
                        })
                        # Runtime collections stay candidates. Do not accumulate
                        # a maximum across unrelated or changing collections.
                        if bulk_target_count is None:
                            coverage_evidence_source = coverage.source
                            coverage_evidence_confidence = coverage.confidence
                            coverage_gate_enforced = False
                            coverage_gate_reason = coverage.reason
                    if graph_guidance is not None:
                        if len(tool_evidence.operation_contexts) == len(actions):
                            verified_tools.extend(context.operation_display
                                                  for context in tool_evidence.operation_contexts)
                        elif not control_flow:
                            for app_name, api_name in re.findall(
                                r"\bapis\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(", proposal.code):
                                verified_tools.append(f"{app_name}.{api_name}")
                    for operation, action in zip(proposal_operations, actions):
                        failure_ledger.record_success(
                            operation,
                            auth_refresh=action.category == "AUTHENTICATE",
                        )
                    rolled_back_codes.clear()
                rejection_budget.commit()
                captured_batch_accepted = False
                read_rechecks_performed = False
                if task_target_tracking and target_tracker is not None and tool_evidence.available:
                    captured_batch_accepted = (
                        len(tool_evidence.records) == len(actions)
                        and all(record.method == action.method
                            and _route_template(str(record.document.get("path", ""))) == _route_template(action.route)
                            for record, action in zip(tool_evidence.records, actions))
                    )
                    if captured_batch_accepted:
                        target_tracker.synchronize_state_version(post_execution_fingerprint)
                        # Only after the entire batch is verified may its returns become
                        # evidence. Replay pure READ batches in call order, rechecking each
                        # observation; never use an uncommitted intermediate WRITE state.
                        on_read = None
                        if (completion_verifier is not None and actions and all(a.method == "GET" for a in actions)
                                and (acquisition.pending or target_tracker._repair_episodes)):
                            def on_read(_index, _record):
                                verification, plan = check_completion(step, "evidence_or_repair_read", post_execution_fingerprint)
                                assess_acquisition(verification, plan, post_execution_fingerprint, step)
                            read_rechecks_performed = True
                        dispositions = target_tracker.observe_committed_batch(
                            tool_evidence.records, observation_prefix=f"verified_return_{step}",
                            state_version=post_execution_fingerprint,
                            mutation_attempted_before=target_mutation_before,
                            categories=[action.category for action in actions], on_read=on_read,
                        )
                    else:
                        dispositions = []
                        target_tracker.metrics["target_evidence_rejected_batches"] += 1
                    append_jsonl(audit_path, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "TASK_TARGET_TOOL_EVIDENCE", "dataset": dataset,
                        "task_id": task_id, "step": step,
                        "capture_boundary": tool_evidence.reason,
                        "captured_return_count": len(tool_evidence.records),
                        "batch_accepted": captured_batch_accepted,
                        "dispositions": dispositions,
                        "constraints": target_tracker.audit_constraints(),
                        "contract": target_tracker.audit_contract(),
                        **target_tracker.audit_metrics(),
                    })
                if task_target_tracking and target_tracker is not None and not tool_evidence.available:
                    append_jsonl(audit_path, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "TASK_TARGET_EVIDENCE_UNAVAILABLE", "dataset": dataset,
                        "task_id": task_id, "step": step, "reason": tool_evidence.reason,
                        "fallback": "single_direct_return_only" if len(actions) == 1 else "none",
                    })
                if (not captured_batch_accepted and not tool_evidence.available
                        and task_target_tracking and target_tracker is not None and len(actions) == 1
                        and actions[0].category == "READ_RESOURCE"):
                    namespace = getattr(getattr(world, "shell", None), "user_ns", None)
                    payload = verified_read_payload(
                        proposal.code, output,
                        namespace if isinstance(namespace, Mapping) else None,
                    )
                    if payload is not None:
                        operation, value, unfiltered = payload
                        app_name, api_name = operation.split(".", 1)
                        app_docs = world.task.api_docs.get(app_name, {})
                        doc = app_docs.get(api_name, {}) if isinstance(app_docs, Mapping) else None
                        parameters = doc.get("parameters", []) if isinstance(doc, Mapping) else None
                        # A default-paginated read cannot certify the whole collection.
                        complete = isinstance(parameters, (list, tuple)) and not any(
                            isinstance(parameter, Mapping) and any(
                                token in str(parameter.get("name", "")).lower()
                                for token in ("page", "offset", "limit")
                        ) for parameter in (parameters or [])
                        )
                        if isinstance(doc, Mapping) and re.search(
                            r"\b(?:paginated|pagination|first\s+\d+|up to\s+\d+)\b",
                            str(doc.get("description", "")), re.I,
                        ):
                            complete = False
                        if state_changed is True:
                            target_tracker.record_mutation_attempt()
                        target_tracker.synchronize_state_version(post_execution_fingerprint)
                        disposition = target_tracker.observe(
                            operation=operation, value=value,
                            observation_id=f"verified_read_{step}", verified=True,
                            complete=complete and post_execution_fingerprint is not None,
                            unfiltered=unfiltered,
                        )
                        append_jsonl(audit_path, {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "kind": "TASK_TARGET_OBSERVATION",
                            "dataset": dataset, "task_id": task_id, "step": step,
                            "disposition": disposition,
                            "constraints": target_tracker.audit_constraints(),
                            **target_tracker.audit_metrics(),
                        })
                restore_scope_history.extend(tool_evidence.attempts)
                active_tool_evidence = None
                if (completion_verifier is not None and target_tracker is not None
                        and not any(a.category == "COMPLETE_TASK" for a in actions)
                        and any(a.method == "GET" for a in actions)
                        and (acquisition.pending or target_tracker._repair_episodes) and not read_rechecks_performed):
                    verification, plan = check_completion(step, "evidence_or_repair_read", post_execution_fingerprint)
                    assess_acquisition(verification, plan, post_execution_fingerprint, step)
                current_runtime_stage = "agent_observation"
                agent.observe(output)
                if target_tracker is not None and task_target_tracking:
                    guidance = (target_tracker.qualification_guidance(MAX_API_CALLS_PER_STEP)
                                if not world.task_completed() else None)
                    if guidance:
                        agent.observe("[HARNESS_EVIDENCE_REQUIRED]\n" + json.dumps(guidance, default=str))
                    target_tracker.record_requirement_outcomes(completion_verifier)
                    flush_evidence_events(step)
                if world.task_completed():
                    committed = decision_manager.commit_active()
                    if committed is not None:
                        write_decision_event(
                            audit_path,
                            kind="DECISION_COMMITTED",
                            dataset=dataset,
                            task_id=task_id,
                            step=step,
                            decision_id=committed.decision_id,
                            action_id=None,
                            reason="AppWorld task completed",
                            metadata_explicit=committed.metadata_explicit,
                        )
                    completed = True
                    if completion_verifier is not None and target_tracker is not None:
                        target_tracker.metrics["completion_final_status"] = completion_candidate_status or "COMPLETED_WITH_UNCERTAINTY"
                        append_jsonl(audit_path, {"kind": "COMPLETION_FINAL_STATUS", "step": step,
                            "status": target_tracker.metrics["completion_final_status"],
                            "repair_episodes": target_tracker._repair_episodes})
                    stop_reason = "AppWorld task marked complete"
                    break
            committed = decision_manager.commit_active()
            if committed is not None:
                write_decision_event(
                    audit_path,
                    kind="DECISION_COMMITTED",
                    dataset=dataset,
                    task_id=task_id,
                    step=proposed,
                    decision_id=committed.decision_id,
                    action_id=None,
                    reason=f"Task execution stopped: {stop_reason}",
                    metadata_explicit=committed.metadata_explicit,
                )
    except Exception as exc:
        technical_failure_stage = current_runtime_stage
        technical_failure_location = technical_exception_location(exc)
        safe_message = safe_exception_text(exc)
        location_suffix = (
            f" at {technical_failure_location}" if technical_failure_location else ""
        )
        error = (
            f"{type(exc).__name__}: {safe_message} "
            f"[stage={technical_failure_stage}{location_suffix}]"
        )
        stop_reason = "Technical failure"
        try:
            append_jsonl(audit_path, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": "TECHNICAL_FAILURE",
                "dataset": dataset,
                "task_id": task_id,
                "step": current_step or None,
                "stage": technical_failure_stage,
                "error_type": type(exc).__name__,
                "error_message": safe_message,
                "location": technical_failure_location,
            })
        except Exception:
            # Preserve the original failure even if its audit destination is unavailable.
            pass

    success = False
    evaluation_pass_count = evaluation_fail_count = evaluation_total_count = 0
    evaluation_pass_rate = None
    official_failure_reasons: list[str] = []
    official_failure_details: list[dict[str, str]] = []
    if error is None:
        try:
            tracker = evaluate_task(task_id=task_id, experiment_name=experiment_name)
            success = bool(tracker.success)
            (evaluation_pass_count, evaluation_fail_count,
             evaluation_total_count, evaluation_pass_rate) = evaluation_test_counts(tracker)
            official_failure_reasons = evaluation_failure_reasons(tracker)
            official_failure_details = evaluation_failure_details(tracker)
        except Exception as exc:
            error = f"Evaluation {type(exc).__name__}: {exc}"
            stop_reason = "Evaluation failure"
    finished_at = datetime.now(timezone.utc).isoformat()
    audit_events: list[dict[str, Any]] = []
    try:
        audit_events = [
            json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        audit_events = []
    failure_classification_count = sum(
        event.get("kind") == "FAILURE_CLASSIFICATION" for event in audit_events
    )
    return TaskRunSummary(
        task_target_tracking_enabled=task_target_tracking,
        completion_verification_enabled=completion_verification,
        **(target_tracker.audit_metrics() if target_tracker is not None else {}),
        dataset=dataset,
        task_id=task_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=round(time.perf_counter() - started_clock, 3),
        success=success,
        completed=completed,
        evaluation_pass_count=evaluation_pass_count,
        evaluation_fail_count=evaluation_fail_count,
        evaluation_total_count=evaluation_total_count,
        evaluation_pass_rate=evaluation_pass_rate,
        evaluation_failure_reasons=official_failure_reasons,
        evaluation_failure_details=official_failure_details,
        steps_proposed=proposed,
        steps_executed=executed,
        api_calls_executed=api_calls_executed,
        rejected_steps=rejected,
        checkpointed_steps=checkpointed,
        precheck_rejections=precheck_rejections,
        execution_failures=execution_failures,
        no_mutation_failures=no_mutation_failures,
        postcondition_failures=postcondition_failures,
        repeated_decision_failures=repeated_decision_failures,
        failure_classification_count=failure_classification_count,
        failure_evidence_count=len(failure_ledger.evidence),
        active_failure_constraint_count=len(failure_ledger.active_constraints),
        rollback_attempts=rollback_attempts,
        rollback_successes=rollback_successes,
        rollback_failures=rollback_failures,
        rollback_budget_exhausted=rollback_budget.exhausted,
        decision_rollback_budget_exhausted=(
            not decision_manager.decision_rollback_available
        ),
        task_terminated_due_to_rollback_budget=(
            task_terminated_due_to_rollback_budget
        ),
        rollback_cause_counts=dict(rollback_budget.by_cause),
        decisions_opened=decision_manager.opened,
        explicit_decisions_opened=decision_manager.explicit_opened,
        implicit_decisions_opened=decision_manager.implicit_opened,
        decision_commits=decision_manager.commits,
        action_rollbacks=decision_manager.action_rollbacks,
        decision_rollbacks=decision_manager.decision_rollbacks,
        agent_requested_decision_rollbacks=(
            decision_manager.agent_requested_decision_rollbacks
        ),
        automatic_decision_rollbacks=(
            decision_manager.automatic_decision_rollbacks
        ),
        repeated_rolled_back_proposals=repeated_rolled_back_proposals,
        completion_coverage_checks=completion_coverage_checks,
        completion_coverage_rejections=completion_coverage_rejections,
        completion_coverage_advisories=completion_coverage_advisories,
        completion_format_checks=completion_format_checks,
        completion_format_rejections=completion_format_rejections,
        semantic_write_checks=semantic_write_checks,
        semantic_write_rejections=semantic_write_rejections,
        semantic_completion_checks=semantic_completion_checks,
        semantic_completion_rejections=semantic_completion_rejections,
        harness_metadata_advisories=harness_metadata_advisories,
        post_step_probability_checks=post_step_checks,
        guidance_queries=guidance_queries,
        guidance_hits=guidance_hits,
        guidance_errors=guidance_errors,
        probability_ranked_guidance_hits=probability_ranked_guidance_hits,
        mean_success_guidance_score=(
            round(sum(success_guidance_scores) / len(success_guidance_scores), 6)
            if success_guidance_scores else None
        ),
        stop_reason=stop_reason,
        appworld_output=str(output_dir),
        harness_audit=str(audit_path),
        error=error,
        technical_failure_stage=technical_failure_stage,
        technical_failure_location=technical_failure_location,
    )
