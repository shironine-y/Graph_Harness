from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Protocol


_API_NAME = re.compile(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*", re.ASCII)
MAX_HINTS = 2


class GraphGuidanceProvider(Protocol):
    """Read-only suggestion interface; no probability or execution authority."""

    graph_version: str

    def retrieve(self, instruction: str, api_docs: Mapping[str, Any],
                 verified_tools: list[str], *, limit: int = MAX_HINTS
                 ) -> list[dict[str, Any]]: ...


def _bounded_count(value: object) -> int | None:
    if type(value) is not int or value < 1:
        return None
    return min(value, 1_000_000)


def _bounded_rate(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return round(value, 6) if 0.0 <= value <= 1.0 else None


def query_graph_guidance(
    provider: GraphGuidanceProvider,
    instruction: str,
    api_docs: Mapping[str, Any],
    verified_tools: list[str],
) -> tuple[list[dict[str, Any]], str | None]:
    """Return prompt-safe hints and an error class; never expose provider text.

    Invalid individual hints are ignored. An unavailable or malformed provider
    abstains, so the agent continues with normal Harness checks.
    """
    allowed = {
        f"{app}.{api}".lower()
        for app, apis in api_docs.items() if isinstance(apis, Mapping)
        for api in apis
    }
    try:
        raw = provider.retrieve(instruction, api_docs, list(verified_tools), limit=MAX_HINTS)
    except Exception as exc:
        return [], type(exc).__name__[:80]
    if not isinstance(raw, list):
        return [], "InvalidResponse"
    safe: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw[:50]:
        if not isinstance(item, Mapping) or item.get("source_split") != "train":
            continue
        api = item.get("api")
        if not isinstance(api, str) or not _API_NAME.fullmatch(api):
            continue
        api = api.lower()
        if api not in allowed or api in seen:
            continue
        tasks = _bounded_count(item.get("source_train_tasks"))
        traces = _bounded_count(item.get("trace_support"))
        if tasks is None or traces is None:
            continue
        hint = {"api": api, "source_split": "train",
                "source_train_tasks": tasks, "trace_support": traces}
        score = _bounded_rate(item.get("success_guidance_score"))
        evidence_tasks = _bounded_count(item.get("probability_evidence_tasks"))
        sample_size = _bounded_count(item.get("probability_sample_size"))
        if score is not None and evidence_tasks is not None:
            hint.update({
                "success_guidance_score": score,
                "historical_success_rate": _bounded_rate(
                    item.get("historical_success_rate")
                ),
                "historical_partial_rate": _bounded_rate(
                    item.get("historical_partial_rate")
                ),
                "historical_failure_rate": _bounded_rate(
                    item.get("historical_failure_rate")
                ),
                "mean_transition_probability": _bounded_rate(
                    item.get("mean_transition_probability")
                ),
                "mean_success_trace_coverage": _bounded_rate(
                    item.get("mean_success_trace_coverage")
                ),
                "probability_evidence_tasks": evidence_tasks,
            })
            if sample_size is not None:
                hint["probability_sample_size"] = sample_size
        safe.append(hint)
        seen.add(api)
        if len(safe) == MAX_HINTS:
            break
    return safe, None
