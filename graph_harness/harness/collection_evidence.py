from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from collections.abc import Mapping

from harness.tool_evidence import read_is_complete


PAGING_NAMES = {"page_index", "page_number", "page", "offset", "page_limit", "limit",
                "page_size", "cursor", "after", "before"}
SORT_NAMES = {"sort_by", "sort_order", "order_by", "ascending", "descending"}


def success_schema(document):
    schemas = document.get("response_schemas")
    schema = schemas.get("success") if isinstance(schemas, Mapping) else None
    return schema if isinstance(schema, Mapping) else {}


def collection_rows(record):
    value = record.value
    if isinstance(value, list):
        return value
    if not isinstance(value, Mapping):
        return None
    schema = success_schema(record.document)
    keys = [key for key, example in schema.items()
            if isinstance(example, list) and isinstance(value.get(key), list)]
    if not keys and isinstance(value.get("items"), list):
        keys = ["items"]
    return value[keys[0]] if len(keys) == 1 else None


@dataclass
class CollectionRead:
    value: object
    complete: bool
    unfiltered: bool
    reason: str
    subject: str = ""
    pages: int = 1
    operation: str = ""
    observation_ids: tuple[str, ...] = ()


@dataclass
class _Pages:
    rows: dict = field(default_factory=dict)
    ends: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)
    total: int | None = None
    conflict: bool = False
    observations: dict = field(default_factory=dict)


class CollectionEvidence:
    def __init__(self):
        self._groups = {}

    def invalidate(self):
        self._groups.clear()

    def observe(self, record, state_version, observation_id=None):
        params = record.document.get("parameters", ())
        params = params if isinstance(params, (list, tuple)) else ()
        defaults = {p.get("name"): p.get("default") for p in params if isinstance(p, Mapping)}
        args = {k: v for k, v in defaults.items() if v is not None and k not in {
            "access_token", "token", "password", "client", "show", "track", "raise_on_failure"}}
        args.update(record.arguments)
        filters = {k: v for k, v in args.items() if k not in PAGING_NAMES}
        subject = getattr(record, "subject_key", "")
        unfiltered = not (set(filters) - SORT_NAMES)
        rows = collection_rows(record)
        if read_is_complete(record):
            return CollectionRead(record.value, True, unfiltered, "unpaginated_read", subject,
                                  operation=record.operation,
                                  observation_ids=(observation_id,) if observation_id else ())
        if rows is None or not all(isinstance(row, Mapping) for row in rows):
            return CollectionRead(record.value, False, unfiltered, "unsupported_page_shape", subject,
                                  operation=record.operation)
        mode = next((p for p in ("page_index", "page_number", "page", "offset", "cursor")
                     if p in defaults), None)
        base = defaults.get(mode, 0) if mode != "cursor" else defaults.get(mode)
        if mode is None or (mode != "cursor" and type(base) is not int):
            return CollectionRead(record.value, False, unfiltered, "unsupported_pagination", subject,
                                  operation=record.operation)
        # A documented default can be a later page; the full domain still starts
        # at zero (page_index/offset) or one (page_number/page).
        start = None if mode == "cursor" else 0 if mode in {"page_index", "offset"} else 1
        position = args.get(mode, base)
        if ((mode != "cursor" and (type(position) is not int or position < start))
                or (mode == "cursor" and position is not None and type(position) is not str)):
            return CollectionRead(record.value, False, unfiltered, "invalid_page_position", subject,
                                  operation=record.operation)
        limit_name = next((p for p in ("page_limit", "page_size", "limit") if p in defaults), None)
        limit = args.get(limit_name, defaults.get(limit_name)) if limit_name else None
        if mode == "offset" and (type(limit) is not int or limit <= 0):
            return CollectionRead(record.value, False, unfiltered, "offset_limit_unknown", subject,
                                  operation=record.operation)
        key = hashlib.sha256(json.dumps([record.operation, subject, filters, mode, limit,
                                        state_version], sort_keys=True, default=str).encode()).hexdigest()
        # Bound evidence memory without treating eviction as proof of completion.
        if len(self._groups) >= 128 and key not in self._groups:
            self._groups.clear()
        pages = self._groups.setdefault(key, _Pages())
        if len(pages.rows) >= 256 or sum(map(len, pages.rows.values())) + len(rows) > 20000:
            self._groups.pop(key, None)
            return CollectionRead(record.value, False, unfiltered, "evidence_capacity_reached",
                                  subject, operation=record.operation)
        if position in pages.rows and pages.rows[position] != rows:
            pages.conflict = True
        pages.rows[position] = list(rows)
        if observation_id:
            pages.observations[position] = observation_id
        schema = success_schema(record.document)
        value = record.value if isinstance(record.value, Mapping) else {}
        terminal = not rows
        if "has_more" in schema and type(value.get("has_more")) is bool:
            terminal = not value["has_more"]
        if "next_cursor" in schema and mode == "cursor":
            nxt = value.get("next_cursor")
            if nxt is not None and type(nxt) is not str:
                pages.conflict = True
            pages.links[position] = nxt
            terminal = nxt is None
        pages.ends[position] = terminal
        for total_key in ("total_count",):
            if total_key in schema and type(value.get(total_key)) is int:
                total = value[total_key]
                if total < 0 or (pages.total is not None and pages.total != total):
                    pages.conflict = True
                pages.total = total
        ordered, visited, positions, pos, complete = [], set(), [], start, False
        while pos in pages.rows and pos not in visited:
            visited.add(pos)
            positions.append(pos)
            ordered.extend(pages.rows[pos])
            if pages.ends[pos] or (pages.total is not None and len(ordered) == pages.total):
                complete = True
                break
            if mode == "cursor":
                if pos not in pages.links:
                    break
                pos = pages.links[pos]
            else:
                pos += limit if mode == "offset" else 1
        if pages.total is not None and len(ordered) > pages.total:
            pages.conflict = True
        if complete and pages.total is not None and len(ordered) != pages.total:
            pages.conflict = True
        complete = complete and not pages.conflict and state_version is not None
        reason = "complete_page_chain" if complete else (
            "conflicting_pages" if pages.conflict else "pagination_incomplete")
        return CollectionRead({"items": ordered}, complete, unfiltered, reason, subject,
                              len(visited), record.operation,
                              tuple(pages.observations[p] for p in positions if p in pages.observations))
