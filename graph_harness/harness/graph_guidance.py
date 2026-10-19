from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

from harness.graph_ready import GraphReadyBuild


_WORDS = re.compile(r"[a-z][a-z0-9]{2,}", re.IGNORECASE)
_API = re.compile(r"(?:apis\.)?([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)", re.IGNORECASE)
_GRAPH_TOOL = re.compile(r"\b([a-z][a-z0-9_]*)__([a-z][a-z0-9_]*)\b", re.IGNORECASE)
_STOP = {"the", "and", "for", "with", "from", "this", "that", "your", "user",
         "app", "task", "api", "please", "find", "get", "use", "need", "want",
         "goal", "event", "resource", "context", "action", "result"}


def _tokens(value: str) -> set[str]:
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", value).replace("_", " ")
    return {word.lower() for word in _WORDS.findall(words) if word.lower() not in _STOP}


def _api_names(value: object) -> set[str]:
    if not isinstance(value, str) or len(value) > 200:
        return set()
    pairs = _API.findall(value) + _GRAPH_TOOL.findall(value)
    return {f"{app.lower()}.{api.lower()}" for app, api in pairs}


def _node_text(row: Mapping[str, Any]) -> str:
    props = row.get("properties") or {}
    if not isinstance(props, dict):
        return ""
    return " ".join(str(props.get(key) or "") for key in
                    ("node_name", "name", "goal_name", "goal_label", "tool_name",
                     "api_name", "canonical_name", "description", "instruction"))[:2000]


def _decoded(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _goal_words(row: Mapping[str, Any]) -> set[str]:
    """Read train-side semantic fields for matching, never for prompt output."""
    props = row.get("properties") or {}
    if not isinstance(props, dict):
        return set()
    parts = [_node_text(row)]
    signature = _decoded(props.get("context_signature_json"))
    if isinstance(signature, dict):
        for key in ("goal", "task", "semantic_role", "semantic_type", "kind"):
            value = signature.get(key)
            if value is not None:
                parts.append(json.dumps(value, ensure_ascii=False)[:2000])
    variants = _decoded(props.get("source_name_variants"))
    if isinstance(variants, list):
        parts.extend(str(value)[:500] for value in variants[:5])
    return _tokens(" ".join(parts)[:6000])


@dataclass(frozen=True)
class _Path:
    task_id: str
    goal_words: frozenset[str]
    previous_tools: frozenset[str]
    next_tools: frozenset[str]
    trace_support: int
    starts_path: bool


class GraphReadyRetriever:
    """Index train-only transition paths; emit only allowlisted API identifiers."""

    def __init__(self, graph: GraphReadyBuild):
        self.graph_version = graph.graph_version
        train_tasks = {str(row.get("task_uid")): task_id for task_id, row in graph.tasks.items()
                       if row.get("dataset") == "train" and row.get("task_uid")}
        nodes: dict[str, dict[str, Any]] = {}
        for row in graph._rows("indexes/nodes.jsonl"):
            uid = str(row.get("node_uid") or "")
            labels = set(row.get("labels") or [])
            if uid and labels & {"Tool", "Goal", "Event", "Task", "ActionConcept"}:
                nodes[uid] = row

        event_tools: dict[str, set[str]] = defaultdict(set)
        concept_tools: dict[str, set[str]] = defaultdict(set)
        event_concepts: dict[str, set[str]] = defaultdict(set)
        goal_words: dict[str, set[str]] = defaultdict(set)
        for uid, row in nodes.items():
            if "Event" in row.get("labels", []):
                props = row.get("properties") or {}
                event_tools[uid].update(_api_names(_node_text(row)))
                event_tools[uid].update(_api_names(props.get("action_concept_signature_json")))
        for row in graph._rows("indexes/relationships.jsonl"):
            kind = row.get("relationship_type")
            source = str(row.get("source_uid") or "")
            target = str(row.get("target_uid") or "")
            if kind == "invokesTool":
                for owner_uid, tool_uid in ((source, target), (target, source)):
                    owner = nodes.get(owner_uid, {})
                    tool = nodes.get(tool_uid, {})
                    if "Tool" in tool.get("labels", []):
                        if "Event" in owner.get("labels", []):
                            event_tools[owner_uid].update(_api_names(_node_text(tool)))
                        elif "ActionConcept" in owner.get("labels", []):
                            concept_tools[owner_uid].update(_api_names(_node_text(tool)))
            elif kind == "hasActionConcept":
                for event_uid, concept_uid in ((source, target), (target, source)):
                    if ("Event" in nodes.get(event_uid, {}).get("labels", [])
                            and "ActionConcept" in nodes.get(concept_uid, {}).get("labels", [])):
                        event_concepts[event_uid].add(concept_uid)
            elif kind == "hasGoal":
                for task_uid, goal_uid in ((source, target), (target, source)):
                    goal = nodes.get(goal_uid, {})
                    if task_uid in train_tasks and "Goal" in goal.get("labels", []):
                        goal_words[task_uid].update(_goal_words(goal))

        # Graph builders may store task links on Goal nodes instead of a hasGoal edge.
        for row in nodes.values():
            if "Goal" not in row.get("labels", []):
                continue
            props = row.get("properties") or {}
            task_uids = props.get("task_uids") or []
            if isinstance(task_uids, list):
                for task_uid in task_uids:
                    if str(task_uid) in train_tasks:
                        goal_words[str(task_uid)].update(_goal_words(row))
        for event_uid, concepts in event_concepts.items():
            for concept_uid in concepts:
                event_tools[event_uid].update(concept_tools.get(concept_uid, set()))
                # Schema-4 transition rows can use ActionConcept/Event nodes,
                # while invokesTool is attached to their ContextEvent source.
                event_tools[concept_uid].update(event_tools.get(event_uid, set()))

        train_rows = [row for row in graph._rows("indexes/transition_statistics.jsonl")
                      if str(row.get("task_uid") or "") in train_tasks
                      and row.get("estimator_scope") == "exact_task"]
        incoming = {(str(row["task_uid"]), str(row.get("target_event_uid") or ""))
                    for row in train_rows}
        paths: list[_Path] = []
        for row in train_rows:
            task_uid = str(row["task_uid"])
            previous = event_tools.get(str(row.get("source_event_uid") or ""), set())
            following = event_tools.get(str(row.get("target_event_uid") or ""), set())
            if (previous or following) and goal_words.get(task_uid):
                paths.append(_Path(train_tasks[task_uid], frozenset(goal_words[task_uid]),
                                   frozenset(previous), frozenset(following),
                                   max(0, int(row.get("trace_support_count") or 0)),
                                   (task_uid, str(row.get("source_event_uid") or ""))
                                   not in incoming))
        self.paths = paths
        self.index_summary = {
            "train_tasks": len(train_tasks),
            "train_transition_rows": len(train_rows),
            "tool_nodes_with_api_name": sum(
                bool(_api_names(_node_text(row))) for row in nodes.values()
                if "Tool" in row.get("labels", [])
            ),
            "goal_linked_train_tasks": sum(bool(words) for words in goal_words.values()),
            "train_source_events_with_tools": len({
                str(row.get("source_event_uid")) for row in train_rows
                if event_tools.get(str(row.get("source_event_uid")))
            }),
            "train_paths_with_goal_and_tool": len(paths),
        }

    def retrieve(self, instruction: str, api_docs: Mapping[str, Any],
                 verified_tools: list[str], *, limit: int = 2) -> list[dict[str, Any]]:
        """Rank by shared goal words and last verified API; abstain on no match."""
        allowed = {f"{app}.{api}".lower() for app, apis in api_docs.items()
                   if isinstance(apis, Mapping) for api in apis}
        query = _tokens(instruction)
        if not query or not allowed or limit < 1:
            return []
        previous = verified_tools[-1].lower() if verified_tools else None
        candidates: dict[str, dict[str, Any]] = {}
        for path in self.paths:
            overlap = len(query & path.goal_words)
            path_match = bool(previous and previous in path.previous_tools)
            if overlap < (1 if path_match else 2):
                continue
            if previous and not path_match:
                continue
            if not previous and not path.starts_path:
                continue
            suggested = path.next_tools if previous else path.previous_tools
            for tool in suggested & allowed:
                if tool == previous:
                    continue
                candidate = candidates.setdefault(tool, {
                    "api": tool, "matching_goal_terms": 0, "source_train_tasks": set(),
                    "support_by_task": {}, "previous_api_matched": False,
                })
                candidate["matching_goal_terms"] = max(candidate["matching_goal_terms"], overlap)
                candidate["source_train_tasks"].add(path.task_id)
                support = candidate["support_by_task"]
                support[path.task_id] = max(support.get(path.task_id, 0), path.trace_support)
                candidate["previous_api_matched"] |= path_match
        result: list[dict[str, Any]] = []
        for candidate in candidates.values():
            result.append({
                "api": candidate["api"],
                "source_split": "train",
                "matching_goal_terms": candidate["matching_goal_terms"],
                "source_train_tasks": len(candidate["source_train_tasks"]),
                "trace_support": sum(candidate["support_by_task"].values()),
                "previous_api_matched": candidate["previous_api_matched"],
            })
        result.sort(key=lambda item: (-int(item["previous_api_matched"]),
                                      -item["source_train_tasks"],
                                      -item["matching_goal_terms"],
                                      -item["trace_support"], item["api"]))
        return result[:limit]


def format_guidance(hints: list[dict[str, Any]]) -> str:
    """Short advisory only; never interpolates graph-derived free text."""
    if not hints:
        return ""
    lines = ["Historical train path hints (read-only, uncertain), ranked by "
             "historical edge success evidence when available. Scores summarize "
             "historical train edges; they are not the current task's success "
             "probability and the hints are not required actions. Follow the current task and API docs; supply arguments "
             "from the live environment only:"]
    for hint in hints[:3]:
        evidence = ""
        if hint.get("success_guidance_score") is not None:
            evidence = (
                f"; historical score {hint['success_guidance_score']:.3f}"
                f" from {hint['probability_evidence_tasks']} train task(s)"
            )
        lines.append(
            f"- apis.{hint['api']} ({hint['source_train_tasks']} train task(s); "
            f"{hint['trace_support']} supporting trace(s){evidence})"
        )
    return "\n".join(lines)


def direct_api_name(code: str) -> str | None:
    """Get the direct API name from a statically checked, executed proposal."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute)
                and isinstance(fn.value.value, ast.Name)
                and fn.value.value.id == "apis"):
            names.append(f"{fn.value.attr}.{fn.attr}".lower())
    return names[0] if len(names) == 1 else None
