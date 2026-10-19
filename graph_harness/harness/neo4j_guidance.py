from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import os
from typing import Any, Mapping

from harness.graph_guidance import _tokens


TRAIN_PREFIX = "appworld:train:"
MAX_TRANSITIONS = 10000
TRAIN_TRANSITIONS_QUERY = """
MATCH (source:ContextEvent)-[edge:directlyPrecedes]->(target:ContextEvent)
WHERE source.task_uid STARTS WITH $train_prefix
  AND target.task_uid = source.task_uid
RETURN source.task_uid AS task_uid,
       source.node_uid AS source_uid,
       target.node_uid AS target_uid,
       source.context_signature_json AS source_signature,
       target.context_signature_json AS target_signature,
       edge.trace_support_count AS trace_support_count,
       edge.sample_size AS sample_size,
       edge.occurrence_count AS occurrence_count,
       edge.transition_probability AS transition_probability,
       edge.conditional_success_rate AS conditional_success_rate,
       edge.conditional_partial_rate AS conditional_partial_rate,
       edge.conditional_failure_rate AS conditional_failure_rate,
       edge.success_trace_coverage AS success_trace_coverage
LIMIT $limit
"""

def _signature(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or len(value) > 8000:
        return {}
    try:
        decoded = json.loads(value)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _api(signature: Mapping[str, Any]) -> str | None:
    app, action = signature.get("app"), signature.get("action")
    if not isinstance(app, str) or not isinstance(action, str):
        return None
    value = f"{app.lower()}.{action.lower()}"
    from harness.graph_guidance_provider import _API_NAME
    return value if _API_NAME.fullmatch(value) else None


def _goal_words(source: Mapping[str, Any], target: Mapping[str, Any]) -> frozenset[str]:
    fields = ("local_goal_role", "task_type", "domains", "object_roles")
    parts: list[str] = []
    for signature in (source, target):
        for field in fields:
            value = signature.get(field)
            if isinstance(value, (str, list, dict)):
                parts.append(json.dumps(value, ensure_ascii=False)[:1000])
    return frozenset(_tokens(" ".join(parts)[:4000]))


def _rate(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _edge_evidence(row: Mapping[str, Any]) -> dict[str, float | int | bool | None]:
    """Build a conservative success-driven score from schema-4 edge statistics."""

    success = _rate(row.get("conditional_success_rate"))
    partial = _rate(row.get("conditional_partial_rate"))
    failure = _rate(row.get("conditional_failure_rate"))
    transition = _rate(row.get("transition_probability"))
    coverage = _rate(row.get("success_trace_coverage"))
    sample_size = _count(row.get("sample_size"))
    occurrence_count = _count(row.get("occurrence_count"))
    evidence_available = success is not None
    if not evidence_available:
        return {
            "evidence_available": False,
            "guidance_score": None,
            "success_rate": success,
            "partial_rate": partial,
            "failure_rate": failure,
            "transition_probability": transition,
            "success_trace_coverage": coverage,
            "sample_size": sample_size or 0,
            "occurrence_count": occurrence_count or 0,
        }

    # A partial success receives half credit. The user requested a direct
    # probability score without confidence shrinkage or sample-size weighting.
    # Missing optional metrics are omitted and remaining weights are normalized.
    # This is a ranking score, not P(success | new task).
    utility = success + 0.5 * (partial or 0.0)
    utility = min(1.0, max(0.0, utility))
    weighted_score = 0.70 * utility
    included_weight = 0.70
    if transition is not None:
        weighted_score += 0.20 * transition
        included_weight += 0.20
    if coverage is not None:
        weighted_score += 0.10 * coverage
        included_weight += 0.10
    score = weighted_score / included_weight
    return {
        "evidence_available": True,
        "guidance_score": min(1.0, max(0.0, score)),
        "success_rate": success,
        "partial_rate": partial,
        "failure_rate": failure,
        "transition_probability": transition,
        "success_trace_coverage": coverage,
        "sample_size": sample_size,
        "occurrence_count": occurrence_count or 0,
    }


@dataclass(frozen=True)
class _Transition:
    task_uid: str
    source_uid: str
    target_uid: str
    source_api: str | None
    target_api: str | None
    goal_words: frozenset[str]
    trace_support: int
    starts_path: bool
    guidance_score: float | None
    success_rate: float | None
    partial_rate: float | None
    failure_rate: float | None
    transition_probability: float | None
    success_trace_coverage: float | None
    sample_size: int
    occurrence_count: int


class Neo4jGraphGuidanceProvider:
    """Snapshot train transitions from Neo4j, then serve bounded local hints."""

    def __init__(self, rows: list[Mapping[str, Any]], *, graph_version: str):
        self.graph_version = graph_version
        incoming = {
            (str(row.get("task_uid")), str(row.get("target_uid")))
            for row in rows if str(row.get("task_uid", "")).startswith(TRAIN_PREFIX)
        }
        transitions: list[_Transition] = []
        for row in rows:
            task_uid = row.get("task_uid")
            source_uid, target_uid = row.get("source_uid"), row.get("target_uid")
            if (not isinstance(task_uid, str) or not task_uid.startswith(TRAIN_PREFIX)
                    or not isinstance(source_uid, str) or not isinstance(target_uid, str)):
                continue
            source = _signature(row.get("source_signature"))
            target = _signature(row.get("target_signature"))
            source_api, target_api = _api(source), _api(target)
            words = _goal_words(source, target)
            support = row.get("trace_support_count")
            if (not words or not (source_api or target_api)
                    or type(support) is not int or support < 1):
                continue
            evidence = _edge_evidence(row)
            transitions.append(_Transition(
                task_uid, source_uid, target_uid, source_api, target_api,
                words, support, (task_uid, source_uid) not in incoming,
                evidence["guidance_score"], evidence["success_rate"],
                evidence["partial_rate"], evidence["failure_rate"],
                evidence["transition_probability"],
                evidence["success_trace_coverage"], int(evidence["sample_size"]),
                int(evidence["occurrence_count"]),
            ))
        self.transitions = transitions
        self.index_summary = {
            "train_tasks": len({row.task_uid for row in transitions}),
            "train_transitions_with_semantics": len(transitions),
            "train_transitions_with_probability_evidence": sum(
                transition.guidance_score is not None for transition in transitions
            ),
            "guidance_ranking": "direct_success_rate_transition_probability_coverage",
        }

    @classmethod
    def from_environment(cls) -> "Neo4jGraphGuidanceProvider":
        uri = os.environ.get("HARNESS_NEO4J_URI")
        user = os.environ.get("HARNESS_NEO4J_USER")
        password = os.environ.get("HARNESS_NEO4J_PASSWORD")
        database = os.environ.get("HARNESS_NEO4J_DATABASE", "neo4j")
        version = os.environ.get("HARNESS_NEO4J_GRAPH_VERSION", "neo4j-live-unversioned")
        if not uri or not user or not password:
            raise ValueError("Set HARNESS_NEO4J_URI, HARNESS_NEO4J_USER and HARNESS_NEO4J_PASSWORD")
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("The selected Python environment needs the neo4j driver") from exc
        with GraphDatabase.driver(uri, auth=(user, password),
                                  connection_timeout=10, max_transaction_retry_time=10) as driver:
            driver.verify_connectivity()
            records, _, _ = driver.execute_query(
                TRAIN_TRANSITIONS_QUERY,
                parameters_={"train_prefix": TRAIN_PREFIX, "limit": MAX_TRANSITIONS + 1},
                database_=database,
                routing_="r",
            )
            if len(records) > MAX_TRANSITIONS:
                raise ValueError("Train transition query exceeded the configured limit")
            rows = [record.data() for record in records]
        provider = cls(rows, graph_version=version)
        if not provider.transitions:
            raise ValueError("Neo4j returned no usable train transitions")
        return provider

    def retrieve(self, instruction: str, api_docs: Mapping[str, Any],
                 verified_tools: list[str], *, limit: int = 2) -> list[dict[str, Any]]:
        allowed = {f"{app}.{api}".lower() for app, apis in api_docs.items()
                   if isinstance(apis, Mapping) for api in apis}
        words = _tokens(instruction)
        if not words or not allowed or limit < 1:
            return []
        previous = verified_tools[-1].lower() if verified_tools else None
        candidates: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "tasks": set(), "best_by_task": {}, "overlap": 0,
        })
        for transition in self.transitions:
            overlap = len(words & transition.goal_words)
            if previous:
                if transition.source_api != previous or overlap < 1:
                    continue
                suggested = transition.target_api
            else:
                if not transition.starts_path or overlap < 2:
                    continue
                suggested = transition.source_api
            if not suggested or suggested not in allowed or suggested == previous:
                continue
            candidate = candidates[suggested]
            candidate["tasks"].add(transition.task_uid)
            best_by_task = candidate["best_by_task"]
            existing = best_by_task.get(transition.task_uid)
            transition_key = (
                transition.guidance_score is not None,
                transition.guidance_score or -1.0,
                transition.target_uid,
            )
            if existing is None or transition_key > existing[0]:
                best_by_task[transition.task_uid] = (transition_key, transition)
            candidate["overlap"] = max(candidate["overlap"], overlap)
        results: list[dict[str, Any]] = []
        for api, data in candidates.items():
            selected = [value[1] for value in data["best_by_task"].values()]
            scored = [edge for edge in selected if edge.guidance_score is not None]

            def average(field: str, rows: list[_Transition]) -> float | None:
                values = [getattr(row, field) for row in rows
                          if getattr(row, field) is not None]
                return sum(values) / len(values) if values else None

            results.append({
                "api": api,
                "source_split": "train",
                "source_train_tasks": len(data["tasks"]),
                "trace_support": sum(edge.trace_support for edge in selected),
                "probability_evidence_tasks": len(scored),
                "success_guidance_score": average("guidance_score", scored),
                "historical_success_rate": average("success_rate", scored),
                "historical_partial_rate": average("partial_rate", scored),
                "historical_failure_rate": average("failure_rate", scored),
                "mean_transition_probability": average(
                    "transition_probability", scored
                ),
                "mean_success_trace_coverage": average(
                    "success_trace_coverage", scored
                ),
                "probability_sample_size": sum(edge.sample_size for edge in scored),
                "matching_goal_terms": data["overlap"],
            })
        results.sort(key=lambda item: (
            (0, -(item["success_guidance_score"] or 0.0),
             -item["matching_goal_terms"], item["api"])
            if item["success_guidance_score"] is not None else
            (1, -item["source_train_tasks"], -item["matching_goal_terms"],
             -item["trace_support"], item["api"])
        ))
        return results[:limit]
