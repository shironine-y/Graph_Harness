from __future__ import annotations

import ast
import copy
from collections import Counter
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import secrets
from typing import Any, Mapping

from harness.task_contracts import CollectionBinding, parse_public_contract, resource_name
from harness.collection_evidence import CollectionEvidence


PREDICATES = frozenset({
    "ENTITY_SET_EQUALS", "ENTITY_SET_CONTAINS", "ENTITY_SET_EXCLUDES",
    "QUANTITY_EQUALS", "QUANTITY_AT_LEAST", "QUANTITY_AT_MOST", "FIELD_EQUALS",
    "ATTRIBUTE_GTE", "ATTRIBUTE_LTE", "TOTAL_LTE", "TOTAL_GTE", "MUST_EXIST",
    "MUST_NOT_EXIST", "DESTINATION_EQUALS", "STATE_EQUALS",
    "ATTRIBUTE_LT", "ATTRIBUTE_GT", "TOTAL_EQUALS", "QUANTITY_SUM_EQUALS",
    "FIELDS_PRESERVED", "ANSWER_FORMAT", "ANSWER_ORDER",
    "ENTITY_SET_INTERSECTS",
})
SOURCES = frozenset({"USER_INSTRUCTION", "VERIFIED_OBSERVATION", "VERIFIED_STATE"})
TARGET_METRICS = (
    "task_targets_created", "task_target_constraints",
    "target_provenance_user_instruction", "target_provenance_verified_observation",
    "target_provenance_verified_state", "candidate_sets_seen", "candidate_sets_promoted",
    "completion_verification_checks", "completion_satisfied", "completion_violated",
    "completion_unknown", "completion_blocks", "missing_target_count",
    "unexpected_target_count", "quantity_mismatch_count", "field_mismatch_count",
    "constraint_violation_count", "completion_verification_unknown",
    "task_contract_parsed_clauses", "task_contract_unresolved_clauses",
    "target_evidence_captured_reads", "target_evidence_rejected_batches",
    "completion_feedback_sent", "completion_feedback_deduplicated",
    "partial_collection_reads", "complete_paginated_reads", "subject_scope_conflicts",
    "completion_checks", "completion_evidence_pending_episodes", "completion_evidence_reads",
    "completion_evidence_budget_exhausted", "repair_started", "repair_resolved",
    "confirmed_violation_episodes", "rollback_scope_feedback_events",
    "final_preflight_checks", "final_preflight_violations", "final_preflight_unknown",
    "final_reconciliation_checks",
)


@dataclass(frozen=True)
class TargetProvenance:
    source: str
    operation: str | None = None
    observation_id: str | None = None

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError("Untrusted target provenance")
        if self.source != "USER_INSTRUCTION" and not self.observation_id:
            raise ValueError("Verified evidence requires an observation ID")


@dataclass(frozen=True)
class TargetConstraint:
    predicate: str
    scope: str
    expected: Any
    provenance: TargetProvenance
    field: str | None = None
    entity: Any = None
    identity_field: str | None = None

    def __post_init__(self):
        if self.predicate not in PREDICATES or not self.scope:
            raise ValueError("Invalid target constraint")


@dataclass
class TaskTarget:
    constraints: list[TargetConstraint] = field(default_factory=list)
    unresolved_clauses: list[str] = field(default_factory=list)
    pending_bindings: int = 0


@dataclass(frozen=True)
class VerifiedState:
    """An authoritative snapshot, independent of a model's completion claim.

    complete=False prevents absence/whole-set/aggregate inferences from partial reads.
    Scalar fields must be actually present in fields, never inferred from a missing key.
    """
    entities: tuple[Mapping[str, Any], ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)
    complete: bool = True
    reliable: bool = True
    identity_field: str = "id"
    observation_id: str | None = None
    collection_known: bool = True
    incomplete_reason: str | None = None
    operation: str | None = None
    evidence_ids: tuple[str, ...] = ()
    field_provenance: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    resource_fields: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    resource_entities: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)
    ambiguous_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    purpose: str
    scope: str
    required_fields: tuple[str, ...]
    app: str
    identity_field: str | None = None
    identity_value: Any = None
    constraint_refs: tuple[str, ...] = ()
    source_evidence_ids: tuple[str, ...] = ()
    subject_key: str = ""
    state_version: str | None = None


@dataclass(frozen=True)
class ReadCapability:
    operation: str
    route: str
    parameters: tuple[Mapping, ...]
    returned_field_paths: tuple[tuple[str, ...], ...]
    identity_argument: str | None


@dataclass(frozen=True)
class VerifiedIdentityRelation:
    source: tuple[str, str, Any]
    relation_field: str
    target: tuple[str, str, Any]
    evidence_id: str
    state_version: str | None
    subject_key: str


def _schema_field_paths(schema, prefix=()):
    """Walk AppWorld public example schemas; never guess paths or dereference refs."""
    if isinstance(schema, Mapping):
        if "$ref" in schema or ("properties" in schema and "type" in schema):
            return ()  # Formal schemas need a separate, explicit adapter.
        return tuple(path for key, value in schema.items() if type(key) is str
                     for path in _schema_field_paths(value, prefix + (key,)))
    if isinstance(schema, list):
        return tuple(dict.fromkeys(path for item in schema
                     for path in _schema_field_paths(item, prefix + ("[]",))))
    return (prefix,) if prefix else ()


def _field_names(field_name):
    # Reuse the existing selector/quantity vocabulary, not fuzzy aliases.
    return (field_name,) + {"type": ("product_type", "item_type"),
        "rating": ("product_rating",), "quantity": ("ordered_quantity",)}.get(field_name, ())


def _values_at_path(value, path):
    if not path:
        return [value]
    head, *tail = path
    if head == "[]":
        return [v for row in value for v in _values_at_path(row, tail)] if isinstance(value, list) else []
    return _values_at_path(value[head], tail) if isinstance(value, Mapping) and head in value else []


def _observational_document(doc):
    if not isinstance(doc, Mapping) or str(doc.get("method", "")).upper() != "GET":
        return False
    metadata = doc.get("execution_metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("read_only") is False:
        return False
    # GET is not a proof of no side effects. Public file-writing descriptions
    # and parameters exclude downloads from evidence acquisition.
    params = doc.get("parameters", ())
    return not (any(isinstance(p, Mapping) and p.get("name") in
        {"download_to_file_path", "overwrite"} for p in (params or ()) if isinstance(params, (list, tuple)))
        or re.search(r"\b(?:download|save|write|delete|create)\b.*\b(?:file|directory|disk)\b",
                     str(doc.get("description", "")), re.I))


def aggregate_verified_states(resources, *, entities=None, shared_fields=(), additive_fields=(), entity_additive_fields=()):
    """Preserve per-resource facts; cross-resource aggregation requires explicit semantics."""
    if not resources or any(not s.reliable for s in resources.values()):
        return VerifiedState(reliable=False, complete=False)
    states = list(resources.values())
    identities = {s.identity_field for s in states}
    if len(identities) != 1:
        return VerifiedState(reliable=False, complete=False, incomplete_reason="identity_ambiguous")
    scoped = {scope: copy.deepcopy(dict(s.fields)) for scope, s in resources.items()}
    evidence = tuple(dict.fromkeys(e for s in states for e in
        (s.evidence_ids or ((s.observation_id,) if s.observation_id else ()))))
    fields, provenance, ambiguous = {}, {}, []
    keys = set().union(*(s.fields for s in states))
    for key in keys:
        values = [s.fields.get(key, _MISSING) for s in states]
        available = all(v is not _MISSING for v in values)
        if len(states) == 1 and available:
            fields[key] = copy.deepcopy(values[0])
        elif key in additive_fields and available and all(_number(v) is not None for v in values):
            fields[key] = sum((_number(v) for v in values), Decimal(0))
        elif key in shared_fields and available and all(v == values[0] for v in values):
            fields[key] = copy.deepcopy(values[0])
        else:
            ambiguous.append(key)
            continue
        provenance[key] = tuple(dict.fromkeys(e for s in states for e in
            s.field_provenance.get(key, s.evidence_ids or ((s.observation_id,) if s.observation_id else ()))))
    raw_entities = tuple(r for s in states for r in s.entities)
    if entities is None and entity_additive_fields:
        grouped = {}
        identity = states[0].identity_field
        for row in raw_entities:
            grouped.setdefault(row.get(identity), []).append(row)
        combined = []
        for identifier, rows in grouped.items():
            row = {identity: identifier}
            for key in set().union(*(r for r in rows)) - {identity}:
                values = [r.get(key, _MISSING) for r in rows]
                available = all(v is not _MISSING for v in values)
                if key in entity_additive_fields and available and all(_number(v) is not None for v in values):
                    row[key] = sum((_number(v) for v in values), Decimal(0))
                elif available and all(v == values[0] for v in values):
                    row[key] = copy.deepcopy(values[0])
                else:
                    ambiguous.append("entity_field:" + key)
            combined.append(row)
        entities = tuple(combined)
    return VerifiedState(tuple(entities) if entities is not None else raw_entities,
        fields, complete=all(s.complete for s in states), identity_field=states[0].identity_field,
        observation_id=evidence[-1] if evidence else None, evidence_ids=evidence,
        field_provenance=provenance, resource_fields=scoped,
        resource_entities={scope: copy.deepcopy(s.entities) for scope, s in resources.items()},
        ambiguous_fields=tuple(sorted(set(ambiguous))),
        collection_known=all(s.collection_known for s in states), operation=states[0].operation)


@dataclass
class CompletionVerificationResult:
    status: str
    satisfied_count: int = 0
    unknown_count: int = 0
    missing_target_count: int = 0
    unexpected_target_count: int = 0
    quantity_mismatch_count: int = 0
    field_mismatch_count: int = 0
    preservation_violation_count: int = 0
    constraint_violation_count: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)
    unknown_reasons: list[dict[str, Any]] = field(default_factory=list)
    agent_details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def blocks_completion(self) -> bool:
        return self.status == "VIOLATED"

    def audit_dict(self) -> dict[str, Any]:
        return {key: value for key, value in vars(self).items()
                if key not in {"violations", "agent_details"}}

    def feedback(self) -> str:
        groups = {
            name: [violation for violation in self.violations if violation.get(counter, 0)]
            for name, counter in (
                ("missing_targets", "missing_target_count"),
                ("unexpected_targets", "unexpected_target_count"),
                ("quantity_mismatches", "quantity_mismatch_count"),
                ("field_mismatches", "field_mismatch_count"),
                ("preservation_violations", "preservation_violation_count"),
            )
        }
        return "[HARNESS_COMPLETION_UPDATE]\n" + json.dumps({
            **self.audit_dict(), **groups, "missing_evidence": self.unknown_reasons, "constraint_violations": self.agent_details or self.violations,
            "instruction": (
                "Known constraints are violated; repair the named objects and re-read affected state. "
                "No restore was performed." if self.blocks_completion else
                "Some requirements cannot be verified. Review unknown_reasons and obtain the "
                "missing evidence when possible. UNKNOWN is not a task failure or a verified pass; "
                "no rollback or rejection budget was consumed."
            ),
        }, sort_keys=True)


def _feedback_ids(values):
    """Only bounded non-secret entity identifiers from already observed records."""
    result = []
    for value in sorted(values, key=repr):
        if type(value) is int and abs(value) < 10**12:
            result.append(value)
        elif (type(value) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value)
              and not re.search(r"token|password|secret|cvv|card", value, re.I)):
            result.append(value)
        if len(result) >= 20:
            break
    return result


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


class CompletionVerifier:
    def __init__(self):
        self._salt = secrets.token_bytes(16)

    def _handle(self, value: Any) -> str:
        return hashlib.sha256(self._salt + repr(value).encode()).hexdigest()[:16]

    def verify(self, target: TaskTarget, current_verified_state: Mapping[str, VerifiedState]):
        result = CompletionVerificationResult("UNKNOWN")
        result.unknown_count = len(target.unresolved_clauses) + target.pending_bindings
        if target.unresolved_clauses:
            result.unknown_reasons.append({"reason": "instruction_semantics_unresolved",
                                          "count": len(target.unresolved_clauses)})
        if target.pending_bindings:
            result.unknown_reasons.append({"reason": "target_source_evidence_missing",
                                          "count": target.pending_bindings})
        if not target.constraints and not result.unknown_count:
            result.unknown_count = 1
            result.unknown_reasons.append({"reason": "no_supported_goal_constraints"})
        for index, constraint in enumerate(target.constraints, 1):
            state = current_verified_state.get(constraint.scope)
            outcome, details = self._check(constraint, state)
            if outcome is None:
                result.unknown_count += 1
                reason = "answer_uncheckable" if constraint.scope == "$answer" else "fresh_state_read_missing"
                if state is not None:
                    reason = ("state_evidence_unreliable" if not state.reliable else
                              (state.incomplete_reason or "partial_collection") if not state.complete else
                              "identity_ambiguous" if not state.collection_known else "field_missing")
                identity_needed = (constraint.predicate.startswith("ENTITY_SET_") or
                    constraint.predicate in {"MUST_EXIST", "MUST_NOT_EXIST"} or constraint.entity is not None)
                identity_ambiguous = False
                if state is not None and identity_needed:
                    identifiers = [r.get(state.identity_field) for r in state.entities]
                    identity_ambiguous = (not state.collection_known or
                        any(type(v) not in {str, int} for v in identifiers) or
                        len({repr(v) for v in identifiers}) != len(identifiers) or
                        (constraint.identity_field is not None and constraint.identity_field != state.identity_field))
                evidence_type = ("identity_ambiguous" if identity_ambiguous
                                 else "collection_incomplete_or_unproven" if state is not None and not state.complete
                                 else "field_missing" if state is not None and state.reliable and not identity_needed
                                 else reason)
                result.unknown_reasons.append({
                    "constraint_id": f"constraint_{index}", "reason": reason,
                    "evidence_source_operation": constraint.provenance.operation,
                    "missing_evidence_type": evidence_type,
                    "suggested_verification_scope": constraint.scope,
                    "field": constraint.field,
                    "predicate": constraint.predicate,
                    "operation": state.operation if state else None,
                })
            elif outcome:
                result.satisfied_count += 1
            else:
                result.constraint_violation_count += 1
                for name, count in details.items():
                    setattr(result, name, getattr(result, name) + count)
                result.violations.append({
                    "constraint_id": f"constraint_{index}",
                    "predicate": constraint.predicate,
                    "evidence_ids": list(state.evidence_ids or ((state.observation_id,) if state.observation_id else ())) if state else [],
                    "error_type": ("answer_format_error" if constraint.predicate in {"ANSWER_FORMAT", "ANSWER_ORDER"}
                                   else "task_constraint_violation"),
                    "field": constraint.field,
                    "target_handle": self._handle((constraint.scope, constraint.entity)),
                    "source": constraint.provenance.source,
                    "observation_id": constraint.provenance.observation_id,
                    **details,
                })
                repair = dict(result.violations[-1], scope=constraint.scope)
                if constraint.predicate.startswith("ENTITY_SET_") and state is not None:
                    actual = {row[state.identity_field] for row in state.entities}
                    expected = set(constraint.expected)
                    missing = expected - actual if state.complete and constraint.predicate != "ENTITY_SET_EXCLUDES" else set()
                    extra = (actual - expected if constraint.predicate == "ENTITY_SET_EQUALS" else
                             actual & expected if constraint.predicate == "ENTITY_SET_EXCLUDES" else set())
                    repair["missing_ids"] = _feedback_ids(missing)
                    repair["unexpected_ids"] = _feedback_ids(extra)
                elif constraint.entity is not None:
                    repair["entity_ids"] = _feedback_ids([constraint.entity])
                if state is not None and constraint.field and not re.search(
                        r"token|password|secret|cvv|card|email|address|recipient|destination", constraint.field, re.I):
                    selected = [r for r in state.entities if constraint.entity is None
                                or r.get(state.identity_field) == constraint.entity] if state.collection_known else [state.fields]
                    values = [_number(r.get(constraint.field)) for r in selected]
                    expected = _number(constraint.expected)
                    if expected is not None and values and all(v is not None for v in values):
                        repair["expected_numeric_value"] = str(expected)
                        repair["observed_numeric_values"] = [str(v) for v in values[:20]]
                result.agent_details.append(repair)
        if result.constraint_violation_count:
            result.status = "VIOLATED"
        elif target.constraints and not result.unknown_count:
            result.status = "SATISFIED"
        return result

    def _check(self, c: TargetConstraint, state: VerifiedState | None):
        if state is None or not state.reliable:
            return None, {}
        p = c.predicate
        if p in {"ANSWER_FORMAT", "ANSWER_ORDER"}:
            if "answer" not in state.fields:
                return None, {}
            value = state.fields["answer"]
            decoded_json = False
            if isinstance(value, str):
                try:
                    def invalid_constant(token):
                        raise ValueError("Nonfinite JSON constant")
                    decoded = json.loads(value, parse_constant=invalid_constant)
                    decoded_json = True
                except (ValueError, TypeError):
                    decoded = value
            else:
                decoded = value
            if p == "ANSWER_ORDER":
                if isinstance(decoded, Mapping):
                    actual = tuple(decoded)
                elif isinstance(decoded, (list, tuple)) and all(isinstance(s, str) for s in decoded):
                    actual = tuple(decoded)
                else:
                    return None, {}
                ok = actual == tuple(c.expected)
            else:
                def json_native(v):
                    if v is None or type(v) in {str, bool}:
                        return True
                    if type(v) in {int, float}:
                        return _number(v) is not None
                    if type(v) is list:
                        return all(json_native(s) for s in v)
                    if type(v) is dict:
                        return all(type(k) is str and json_native(s) for k, s in v.items())
                    return False
                ok = {"json": json_native(decoded) and (decoded_json or not isinstance(value, str)),
                      "object": isinstance(decoded, dict), "list": isinstance(decoded, list),
                      "number": _number(decoded) is not None and not isinstance(decoded, str),
                      "string": isinstance(value, str)}.get(c.expected)
                if ok is None:
                    return None, {}
            return ok, {"field_mismatch_count": int(not ok)}
        identity = state.identity_field
        if (not state.collection_known and (p.startswith("ENTITY_SET_")
                or p in {"MUST_EXIST", "MUST_NOT_EXIST"} or c.entity is not None)):
            return None, {}
        if c.identity_field is not None and c.identity_field != identity and state.entities:
            return None, {}
        rows = state.entities
        identities = [row.get(identity) for row in rows]
        if any(type(value) not in {str, int} for value in identities):
            return None, {}
        if len(set(identities)) != len(identities):
            return None, {}  # Ambiguous line items cannot be treated as an entity set.
        if p.startswith("ENTITY_SET_"):
            expected, actual = set(c.expected), set(identities)
            if p == "ENTITY_SET_INTERSECTS":
                if not state.complete and not expected & actual:
                    return None, {}
                ok = bool(expected & actual)
                return ok, {"missing_target_count": int(not ok)}
            missing = expected - actual if p != "ENTITY_SET_EXCLUDES" and state.complete else set()
            unexpected = (actual - expected if p == "ENTITY_SET_EQUALS"
                          else actual & expected if p == "ENTITY_SET_EXCLUDES" else set())
            if not state.complete and not unexpected:
                if p == "ENTITY_SET_CONTAINS" and expected <= actual:
                    return True, {}
                return None, {}
            return not (missing or unexpected), {
                "missing_target_count": len(missing), "unexpected_target_count": len(unexpected),
            }
        if p in {"MUST_EXIST", "MUST_NOT_EXIST"}:
            found = c.entity in identities if c.entity is not None else bool(rows)
            if not found and not state.complete:
                return None, {}
            ok = found if p == "MUST_EXIST" else not found
            return ok, {"missing_target_count" if p == "MUST_EXIST"
                        else "unexpected_target_count": int(not ok)}
        if p == "QUANTITY_SUM_EQUALS":
            if not state.complete or not state.collection_known:
                return None, {}
            numbers = [_number(row.get(c.field or "quantity")) for row in rows]
            if any(n is None for n in numbers) or _number(c.expected) is None:
                return None, {}
            ok = sum(numbers, Decimal(0)) == _number(c.expected)
            return ok, {"quantity_mismatch_count": int(not ok)}
        if c.entity is not None:
            selected = [row for row in rows if row[identity] == c.entity]
            if not selected:
                return (False, {"missing_target_count": 1}) if state.complete else (None, {})
        elif p.startswith(("QUANTITY_", "ATTRIBUTE_")) or p == "FIELD_EQUALS":
            selected = list(rows)
            if not selected:
                return None, {}
        else:
            selected = [state.fields]
        if p == "FIELDS_PRESERVED":
            expected = c.expected
            count = sum(row[key] != value for row in selected for key, value in expected.items() if key in row)
            missing = any(key not in row for row in selected for key in expected) or (
                not state.complete and c.entity is None)
            return (False if count else None if missing else True), {"field_mismatch_count": count, "preservation_violation_count": count}
        key = c.field or ({"DESTINATION_EQUALS": "destination", "STATE_EQUALS": "state"}
                          .get(p, "quantity" if p.startswith("QUANTITY_") else "total"))
        missing_fields = any(key not in row for row in selected) or (not state.complete and c.entity is None)
        if p.startswith("TOTAL_") and not state.complete:
            return None, {}
        values = [row[key] for row in selected if key in row]
        if not values:
            return None, {}
        if p in {"FIELD_EQUALS", "DESTINATION_EQUALS", "STATE_EQUALS"}:
            count = sum((_number(value) != _number(c.expected)) if type(c.expected) in {int, float}
                        else value != c.expected for value in values)
            return (False if count else None if missing_fields else True), {"field_mismatch_count": count}
        expected = _number(c.expected)
        numbers = [_number(value) for value in values]
        if expected is None:
            return None, {}
        missing_fields = missing_fields or any(number is None for number in numbers)
        numbers = [number for number in numbers if number is not None]
        if not numbers:
            return None, {}
        if p in {"QUANTITY_EQUALS", "TOTAL_EQUALS"}:
            count = sum(number != expected for number in numbers)
        elif p in {"QUANTITY_AT_LEAST", "ATTRIBUTE_GTE", "TOTAL_GTE"}:
            count = sum(number < expected for number in numbers)
        elif p == "ATTRIBUTE_LT":
            count = sum(number >= expected for number in numbers)
        elif p == "ATTRIBUTE_GT":
            count = sum(number <= expected for number in numbers)
        else:
            count = sum(number > expected for number in numbers)
        return (False if count else None if missing_fields else True), {"quantity_mismatch_count" if p.startswith("QUANTITY_")
                            else "field_mismatch_count": count}


def _normal(value: str) -> str:
    return value.lower().strip().replace(" ", "_").replace("-", "_")


def _collection(operation: str) -> str | None:
    name = operation.rsplit(".", 1)[-1]
    match = re.fullmatch(r"(?:get|show|list|read)_(?:my_)?([a-z][a-z0-9_]*)", name)
    return resource_name(match[1]) if match else None


def _success_schema(document):
    schemas = document.get("response_schemas", {}) if isinstance(document, Mapping) else {}
    return schemas.get("success", {}) if isinstance(schemas, Mapping) else {}


def _snapshot(value: Any, observation_id: str, document: Mapping | None = None,
              identity_hint: str | None = None) -> VerifiedState | None:
    """Accept unpaginated explicit collections or a single returned object.

    Count metadata, pagination envelopes and scalar results are not interpreted.
    No total is invented by summing guessed price/quantity fields.
    """
    value = copy.deepcopy(value)
    fields = value if isinstance(value, dict) else {}
    collection_known = True
    if isinstance(value, dict):
        if any(key in value for key in ("page", "page_index", "next", "next_page", "has_more")):
            return None
        schema = _success_schema(document)
        collection_keys = [key for key, example in schema.items()
                           if isinstance(example, list) and isinstance(value.get(key), list)] if isinstance(schema, Mapping) else []
        if "items" in value:
            rows = value["items"]
        elif len(collection_keys) == 1:
            rows = value[collection_keys[0]]
        elif "id" in value or (document and re.search(r"\{[a-z_]+_id\}", str(document.get("path", "")))):
            rows = [value]
        elif not any(isinstance(v, (list, dict)) for v in value.values()):
            rows = []
            collection_known = False
        else:
            return None
    else:
        rows = value
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return None
    if isinstance(value, dict) and any(
        key in value and (type(value[key]) is not int or value[key] != len(rows))
        for key in ("count", "total_count")
    ):
        return None
    candidates = {"id"} if not rows else set(rows[0])
    for row in rows:
        candidates &= set(row)
    identities = [key for key in candidates if key == "id" or key.endswith("_id")]
    # Multiple possible identities (e.g. line ID and underlying entity ID) require
    # explicit evidence; guessing the identity could manufacture a set mismatch.
    identity = identities[0] if len(identities) == 1 else None
    if identity is None and document:
        path_ids = re.findall(r"\{([a-z_]+_id)\}", str(document.get("path", "")))
        if len(path_ids) == 1 and path_ids[0] in identities:
            identity = path_ids[0]
        elif identity_hint in identities:
            identity = identity_hint
    if identity is None:
        return None
    if any(type(row.get(identity)) not in {str, int} for row in rows):
        return None
    if isinstance(fields, Mapping) and "total" not in fields and "total_cost" in fields:
        fields = dict(fields, total=fields["total_cost"])
    return VerifiedState(tuple(rows), fields, identity_field=identity,
                         observation_id=observation_id, collection_known=collection_known,
                         evidence_ids=(observation_id,), field_provenance={k: (observation_id,) for k in fields})


class TaskTargetTracker:
    def __init__(self, instruction: str, api_docs: Mapping | None = None):
        self.instruction = instruction
        self.task_target = TaskTarget()
        self.current_verified_state: dict[str, VerifiedState] = {}
        self.metrics = Counter({key: 0 for key in TARGET_METRICS})
        self.metrics["completion_verification_unknown"] = False
        self._promoted: set[tuple[str, str]] = set()
        self._bindings: list[tuple[str, str, str]] = []
        self._pending_destination: tuple[str, str] | None = None
        self._scope_apps: dict[str, set[str]] = {}
        self._state_version: str | None = None
        self._audit_salt = secrets.token_bytes(16)
        self._mutation_attempted = False
        self._api_docs = api_docs or {}
        self._contract_bindings: list[CollectionBinding] = []
        self._preservation: list[tuple[str, tuple[str, ...]]] = []
        self._initial_snapshots: dict[str, tuple[VerifiedState, TargetProvenance]] = {}
        self._details: dict[tuple[str, str, Any], Mapping] = {}
        self._purchase_receipts: dict[str, set[tuple[str, Any]]] = {}
        self._purchase_details: dict[tuple[str, str, Any], VerifiedState] = {}
        self._collection_evidence = CollectionEvidence()
        self._scope_subjects: dict[str, set[str]] = {}
        self._mutated_scopes: set[str] = set()
        self._qualified_states: dict[str, VerifiedState] = {}
        self._feedback_seen: set[str] = set()
        self._evidence_unknown: dict[str, str] = {}
        self.metrics["completion_final_status"] = None
        self.metrics["completion_unknown_reason_counts"] = {}
        self._repair_episodes = []
        self._repair_constraints = {}
        self._read_capabilities = None
        self._requirements = {}
        self._resolutions = {}
        self._evidence_events = []
        self._guidance_seen = set()
        self._detail_provenance = {}
        self._detail_conflicts = {}
        self._detail_subjects = {}
        self._detail_documents = {}
        self._field_projections = {}
        self._identity_relations = {}
        self._relation_conflicts = set()
        self._receipt_subjects = {}
        self._resolved_plans = {}
        self._derive_instruction()

    def add_constraint(self, constraint: TargetConstraint):
        if constraint in self.task_target.constraints:
            return
        self.task_target.constraints.append(constraint)
        self.metrics["task_targets_created"] = 1
        self.metrics["task_target_constraints"] += 1
        self.metrics["target_provenance_" + constraint.provenance.source.lower()] += 1

    def _derive_instruction(self):
        self._derive_legacy_instruction()
        self._bindings = [(resource_name(origin), resource_name(destination), predicate)
                          for origin, destination, predicate in self._bindings]
        contract = parse_public_contract(self.instruction, self._api_docs)
        self._contract_bindings = contract.bindings
        self._preservation = contract.preservation
        for binding in contract.bindings:
            # The new rule retains qualifiers and removal semantics; replace only
            # its equivalent legacy source/destination rule.
            self._bindings = [rule for rule in self._bindings
                              if rule != (binding.origin, binding.destination, binding.predicate)]
        provenance = TargetProvenance("USER_INSTRUCTION")
        for predicate, scope, expected, key in contract.constraints:
            if not any(c.scope == scope and c.field == key and c.expected == expected
                       and (c.predicate == predicate or {c.predicate, predicate} <= {"FIELD_EQUALS", "STATE_EQUALS", "DESTINATION_EQUALS"})
                       for c in self.task_target.constraints):
                self.add_constraint(TargetConstraint(predicate, scope, expected, provenance, key))
        self.task_target.unresolved_clauses = contract.unresolved_clauses
        self.task_target.pending_bindings = len(self._bindings) + len(self._contract_bindings) + len(self._preservation)
        self.metrics["task_contract_parsed_clauses"] = contract.parsed_clauses
        self.metrics["task_contract_unresolved_clauses"] = len(contract.unresolved_clauses)

    def _derive_legacy_instruction(self):
        text = " ".join(self.instruction.split())
        # Conditional, negative or alternative clauses need a richer parser. V1
        # leaves them untracked rather than reversing or guessing their meaning.
        if re.search(r"\b(?:not|never|unless|except|either|instead|if|or|approximately|about)\b"
                     r"|\bignore case\b|\bcase[ -]insensitive\b", text, re.I):
            return
        # Strict public phrases only. Qualified subsets (e.g. red items) are unknown.
        for match in re.finditer(
            r"\b(?:add|move|copy|transfer)\s+(?:all(?:\s+items)?|everything)\s+(?:currently\s+)?"
            r"(?:from|on|in)\s+(?:my|the)\s+([a-z_]+)\s+(?:to|into)\s+"
            r"(?:my|the)\s+([a-z_]+)(?=[.,;]|$)", text, re.I,
        ):
            self._bindings.append((_normal(match[1]), _normal(match[2]), "ENTITY_SET_CONTAINS"))
        for match in re.finditer(
            r"\b(?:make|set)\s+(?:my|the)\s+([a-z_]+)\s+(?:to\s+)?contain\s+"
            r"exactly\s+(?:all(?:\s+items)?|everything)\s+(?:currently\s+)?"
            r"(?:from|on|in)\s+(?:my|the)\s+([a-z_]+)(?=[.,;]|$)", text, re.I,
        ):
            self._bindings.append((_normal(match[2]), _normal(match[1]), "ENTITY_SET_EQUALS"))
        for match in re.finditer(
            r"\bbuy\s+(?:all(?:\s+items)?|everything)\s+(?:currently\s+)?"
            r"(?:from|on|in)\s+(?:my|the)\s+([a-z_]+)(?=[.,;]|$)", text, re.I,
        ):
            # Remember what must be purchased. Do not pretend cart contents prove
            # purchase completion; absence of a purchase-state binding stays UNKNOWN.
            self._bindings.append((_normal(match[1]), "completed_purchase_items", "ENTITY_SET_CONTAINS"))
        # A standalone final-set contract binds the observed collection to itself.
        scope_match = re.search(r"\b(?:in|on|into)\s+(?:my|the)\s+([a-z_]+)(?=[.,;]|$)", text, re.I)
        subject_match = re.search(r"\b(?:my|the)\s+([a-z_]+)\s+must\s+"
                                  r"(?:be empty|contain no items|exist|contain at least one item)\b",
                                  text, re.I)
        scope = (self._bindings[0][1] if len(self._bindings) == 1
                 else _normal(subject_match[1]) if subject_match and not self._bindings
                 else _normal(scope_match[1]) if scope_match and not self._bindings else None)
        if scope is None:
            return
        provenance = TargetProvenance("USER_INSTRUCTION")
        patterns = (
            (r"\beach item(?:'s)? quantity\s*(?:=|is|equals)\s*(\d+)\b", "QUANTITY_EQUALS", "quantity"),
            (r"(?<!seller )(?<!seller_)\b(?:rating|ratings)\s*(?:>=|at least)\s*(\d+(?:\.\d+)?)", "ATTRIBUTE_GTE", "rating"),
            (r"\bseller[ _]rating\s*(?:>=|at least)\s*(\d+(?:\.\d+)?)", "ATTRIBUTE_GTE", "seller_rating"),
            (r"\bbudget\s*(?:<=|at most)\s*\$?(\d+(?:\.\d+)?)", "TOTAL_LTE", "total"),
        )
        for pattern, predicate, key in patterns:
            matches = list(re.finditer(pattern, text, re.I))
            if len(matches) == 1:
                self.add_constraint(TargetConstraint(predicate, scope, matches[0][1], provenance, key))
        field_matches = list(re.finditer(
            r"\b(type|state|recipient)\s*=\s*(?:'([^']+)'|\"([^\"]+)\"|([\w-]+))", text, re.I,
        ))
        counts = Counter(match[1].lower() for match in field_matches)
        for match in field_matches:
            key = match[1].lower()
            if counts[key] != 1:
                continue
            expected = next(value for value in match.groups()[1:] if value is not None)
            predicate = {"type": "FIELD_EQUALS", "state": "STATE_EQUALS",
                         "recipient": "DESTINATION_EQUALS"}[key]
            self.add_constraint(TargetConstraint(predicate, scope, expected, provenance, key))
        if re.search(r"\bmust (?:be empty|contain no items)\b", text, re.I):
            self.add_constraint(TargetConstraint("MUST_NOT_EXIST", scope, None, provenance))
        if re.search(r"\bmust (?:exist|contain at least one item)\b", text, re.I):
            self.add_constraint(TargetConstraint("MUST_EXIST", scope, None, provenance))
        destination = re.search(r"\bdeliver to (?:my |the )?(work|home) address\b", text, re.I)
        if destination:
            self._pending_destination = (scope, destination[1].lower())

    def observe(self, *, operation: str, value: Any, observation_id: str,
                verified: bool, complete: bool = True, unfiltered: bool = True,
                state_read: bool = False, document: Mapping | None = None,
                evidence_ids: tuple[str, ...] = (), incomplete_reason: str | None = None, subject_key: str = "") -> str:
        if not verified:
            return "unverified"
        scope = _collection(operation)
        if scope is None:
            if re.fullmatch(r"(?:search|find|query)_[a-z0-9_]+", operation.rsplit(".", 1)[-1]):
                self.metrics["candidate_sets_seen"] += 1
                return "candidate_only"
            return "unrelated"
        self.metrics["candidate_sets_seen"] += 1
        identity_hint = self._documented_identity_hint(operation, document)
        snapshot = _snapshot(value, observation_id, document, identity_hint)
        if snapshot is None or not unfiltered:
            self.current_verified_state.pop(scope, None)
            self.current_verified_state.pop(operation.rsplit(".", 1)[0] + "." + scope, None)
            self._qualified_states.pop(operation.rsplit(".", 1)[0] + "." + scope, None)
            self._bind_state_aliases()
            return "candidate_only"
        app = operation.rsplit(".", 1)[0]
        qualified = app + "." + scope
        subjects = self._scope_subjects.setdefault(qualified, set())
        subjects.add(subject_key)
        if len(subjects) > 1:
            self.metrics["subject_scope_conflicts"] += 1
            snapshot = VerifiedState(reliable=False, complete=False,
                                     incomplete_reason="subject_scope_ambiguous", operation=operation)
            self._initial_snapshots.pop(qualified, None)
        else:
            snapshot = VerifiedState(snapshot.entities, snapshot.fields, complete=complete,
                identity_field=snapshot.identity_field, observation_id=observation_id,
                collection_known=snapshot.collection_known,
                incomplete_reason=incomplete_reason, operation=operation,
                evidence_ids=evidence_ids or snapshot.evidence_ids,
                field_provenance={k: evidence_ids or snapshot.field_provenance.get(k, ()) for k in snapshot.fields})
        self._qualified_states[qualified] = snapshot
        apps = self._scope_apps.setdefault(scope, set())
        apps.add(app)
        self.current_verified_state[app + "." + scope] = snapshot
        if len(apps) == 1:
            self.current_verified_state[scope] = snapshot
        else:
            self.current_verified_state.pop(scope, None)
        source = "VERIFIED_STATE" if state_read else "VERIFIED_OBSERVATION"
        provenance = TargetProvenance(source, operation, observation_id)
        if snapshot.complete and snapshot.reliable and self._initial_scope_available(qualified):
            self._initial_snapshots.setdefault(app + "." + scope, (snapshot, provenance))
        # A singular, documented detail response may enrich a source entity only
        # through the same explicit ID field and application.
        for row in snapshot.entities:
            self._record_identity_relations(app, snapshot.identity_field, row, document,
                                            observation_id, subject_key)
        if self._pending_destination and scope == "addresses":
            destination_scope, label = self._pending_destination
            matches = [row for row in snapshot.entities
                       if str(row.get("label", "")).lower() == label]
            if len(matches) == 1:
                self.add_constraint(TargetConstraint(
                    "DESTINATION_EQUALS", app + "." + destination_scope,
                    matches[0][snapshot.identity_field], provenance, "address_id"))
        promoted = False
        for origin, destination, predicate in self._bindings:
            if not snapshot.collection_known:
                continue
            if not complete or not snapshot.reliable or not self._initial_scope_available(qualified):
                # A first read after a write cannot establish the task's initial
                # "currently on" source set without dependency inference.
                continue
            if scope != origin or (origin, destination) in self._promoted:
                continue
            ids = [row.get(snapshot.identity_field) for row in snapshot.entities]
            if any(identity is None or isinstance(identity, (list, dict)) for identity in ids):
                continue
            if len(set(ids)) != len(ids):
                continue
            qualified_destination = app + "." + destination
            self.add_constraint(TargetConstraint(predicate, qualified_destination,
                frozenset(ids), provenance, identity_field=snapshot.identity_field))
            for row in snapshot.entities:
                if "quantity" in row and _number(row["quantity"]) is not None:
                    self.add_constraint(TargetConstraint("QUANTITY_EQUALS", qualified_destination,
                        row["quantity"], provenance, "quantity", row[snapshot.identity_field],
                        snapshot.identity_field))
            self._promoted.add((origin, destination))
            self.task_target.pending_bindings = max(0, self.task_target.pending_bindings - 1)
            self.metrics["candidate_sets_promoted"] += 1
            promoted = True
        promoted = self._promote_contracts() or promoted
        self._bind_state_aliases()
        return "target" if promoted else "candidate_only"

    @staticmethod
    def _resource_tokens(name, identity):
        def stem(word):
            word = re.sub(r"(?:ations?|ed)$", "", word)
            if word.endswith("ies"):
                return word[:-3] + "y"
            return word[:-1] if word.endswith("s") and not word.endswith("ss") else word
        tokens = tuple(stem(word) for word in name.split("_") if word)
        entity = stem(identity.removesuffix("_id"))
        return tuple(word for word in tokens if word != entity) if len(tokens) > 1 else tokens

    def _matching_scopes(self, scope, states):
        if scope in states:
            return [(scope, states[scope])]
        wanted_app, _, wanted_name = scope.rpartition(".")
        wanted_name = wanted_name or scope
        candidates = []
        for name, value in states.items():
            app, _, resource = name.rpartition(".")
            snapshot = value[0] if isinstance(value, tuple) else value
            if wanted_app and app != wanted_app:
                continue
            if self._resource_tokens(wanted_name, snapshot.identity_field) == self._resource_tokens(resource, snapshot.identity_field):
                candidates.append((name, value))
        return candidates

    def _initial_scope_available(self, qualified):
        app = qualified.split(".", 1)[0]
        return not ({"*", app + ".*", qualified} & self._mutated_scopes)

    def _bind_state_aliases(self):
        for constraint in self.task_target.constraints:
            if constraint.scope == "$answer":
                continue
            matching = self._matching_scopes(constraint.scope, self._qualified_states)
            if len(matching) == 1:
                self.current_verified_state[constraint.scope] = matching[0][1]
            else:
                self.current_verified_state.pop(constraint.scope, None)

    def _documented_identity_hint(self, operation, document):
        if not isinstance(document, Mapping):
            return None
        app_docs = self._api_docs.get(operation.split(".", 1)[0], {})
        if not isinstance(app_docs, Mapping):
            return None
        path = str(document.get("path", "")).rstrip("/")
        hints = set()
        for detail in app_docs.values():
            if not isinstance(detail, Mapping) or str(detail.get("method", "")).upper() != "GET":
                continue
            match = re.fullmatch(re.escape(path) + r"/\{([a-z_]+_id)\}", str(detail.get("path", "")))
            if match:
                hints.add(match[1])
        return next(iter(hints)) if len(hints) == 1 else None

    def _promote_contracts(self):
        promoted = False
        for binding in self._contract_bindings:
            key = (binding.origin, binding.destination, binding.selectors)
            if key in self._promoted:
                continue
            matching = self._matching_scopes(binding.origin, self._initial_snapshots)
            if len(matching) != 1:
                continue
            name, (snapshot, provenance) = matching[0]
            app = name.rsplit(".", 1)[0]
            if not snapshot.collection_known or not snapshot.complete or not snapshot.reliable:
                continue
            selected = []
            unknown = False
            for row in snapshot.entities:
                enriched = self._enriched_row(app, snapshot.identity_field, row)
                outcomes = [self._select_from_initial(app, snapshot, enriched, selector)
                            for selector in binding.selectors]
                if False in outcomes:
                    continue
                if None in outcomes:
                    unknown = True
                    break
                selected.append(row)
            if unknown:
                continue
            destination = binding.destination if "." in binding.destination else app + "." + binding.destination
            # Never compare IDs across applications without a verified mapping.
            if destination.rsplit(".", 1)[0] != app:
                continue
            ids = frozenset(row[snapshot.identity_field] for row in selected)
            if len(ids) != len(selected):
                continue
            self.add_constraint(TargetConstraint(binding.predicate, destination, ids, provenance,
                                                 identity_field=snapshot.identity_field))
            if binding.predicate == "ENTITY_SET_EXCLUDES" and binding.destination == binding.origin:
                retained = frozenset(row[snapshot.identity_field] for row in snapshot.entities) - ids
                self.add_constraint(TargetConstraint("ENTITY_SET_CONTAINS", destination, retained,
                    provenance, identity_field=snapshot.identity_field))
            if binding.remove_from_origin and destination != name:
                self.add_constraint(TargetConstraint("ENTITY_SET_EXCLUDES", name, ids, provenance,
                                                     identity_field=snapshot.identity_field))
            if binding.predicate not in {"ENTITY_SET_EXCLUDES", "ENTITY_SET_INTERSECTS"}:
                for row in selected:
                    if "quantity" in row and _number(row["quantity"]) is not None:
                        self.add_constraint(TargetConstraint("QUANTITY_EQUALS", destination, row["quantity"],
                            provenance, "quantity", row[snapshot.identity_field], snapshot.identity_field))
            self._promoted.add(key)
            self.metrics["candidate_sets_promoted"] += 1
            self.task_target.pending_bindings -= 1
            promoted = True
        for scope, allowed in self._preservation:
            key = (scope, "$preserved")
            if key in self._promoted:
                continue
            matching = self._matching_scopes(scope, self._initial_snapshots)
            if len(matching) != 1:
                continue
            name, (snapshot, provenance) = matching[0]
            if not snapshot.collection_known:
                continue
            for row in snapshot.entities:
                expected = {k: v for k, v in row.items() if k not in allowed and k != snapshot.identity_field}
                self.add_constraint(TargetConstraint("FIELDS_PRESERVED", name, expected, provenance,
                    entity=row[snapshot.identity_field], identity_field=snapshot.identity_field))
            self._promoted.add(key)
            self.task_target.pending_bindings -= 1
            promoted = True
        return promoted

    @staticmethod
    def _select(row, selector):
        keys = [selector.field] if selector.field in row else [key for key in
            {"type": ("product_type", "item_type"), "rating": ("product_rating",)}.get(selector.field, ()) if key in row]
        if len(keys) != 1:
            return None
        actual = row[keys[0]]
        if selector.comparison == "TYPE_EQ":
            if not isinstance(actual, str):
                return None
            def singular(text):
                text = " ".join(text.lower().split())
                if text.endswith("ies"):
                    return text[:-3] + "y"
                if text.endswith(("ches", "shes", "xes", "sses")):
                    return text[:-2]
                return text[:-1] if text.endswith("s") and not text.endswith("ss") else text
            return singular(actual) == singular(str(selector.expected))
        if selector.comparison == "EQ":
            return actual == selector.expected
        number, expected = _number(actual), _number(selector.expected)
        if number is None or expected is None:
            return None
        return {"LT": number < expected, "GT": number > expected,
                "GTE": number >= expected, "LTE": number <= expected}.get(selector.comparison)

    def _select_from_initial(self, app, snapshot, row, selector):
        if selector.comparison not in {"IN_COLLECTION", "NOT_IN_COLLECTION"}:
            return self._select(row, selector)
        reference = selector.expected if "." in selector.expected else app + "." + selector.expected
        matches = self._matching_scopes(reference, self._initial_snapshots)
        pair = matches[0][1] if len(matches) == 1 else None
        if pair is None or pair[0].identity_field != snapshot.identity_field:
            return None
        other = pair[0]
        if not other.complete or not other.collection_known:
            return None
        members = [r.get(other.identity_field) for r in other.entities]
        if any(type(value) not in {str, int} for value in members) or len(set(members)) != len(members):
            return None
        exists = row[snapshot.identity_field] in members
        return not exists if selector.comparison == "NOT_IN_COLLECTION" else exists

    def observe_committed_batch(self, records, *, observation_prefix, state_version,
                                mutation_attempted_before=False, categories=None, on_read=None):
        """Replay verified real returns in order; writes invalidate preceding reads."""
        from harness.tool_evidence import read_is_complete
        self._mutation_attempted = mutation_attempted_before
        dispositions = []
        for index, record in enumerate(records):
            if record.method != "GET":
                if categories is None or categories[index] == "WRITE_RESOURCE":
                    self.record_mutation_attempt(record)
                    self._record_receipt(record)
                    dispositions.append("mutation")
                else:
                    self.invalidate_state()
                    dispositions.append("non_read")
                continue
            self.metrics["target_evidence_captured_reads"] += 1
            if state_version is None:
                dispositions.append("state_version_unknown")
                if on_read is not None:
                    on_read(index, record)
                continue
            self._state_version = state_version
            detail = self._observe_documented_detail(record, f"{observation_prefix}_{index}")
            self._record_committed_read(record, f"{observation_prefix}_{index}", detail)
            if detail is not None:
                self._promote_contracts()
                dispositions.append(detail)
                self._refresh_completed_purchase_state()
                if on_read is not None:
                    on_read(index, record)
                continue
            evidence = self._collection_evidence.observe(record, state_version, f"{observation_prefix}_{index}")
            self.metrics["partial_collection_reads"] += int(not evidence.complete)
            self.metrics["complete_paginated_reads"] += int(evidence.reason == "complete_page_chain")
            if not evidence.complete:
                self._evidence_unknown[record.operation] = evidence.reason
            else:
                self._evidence_unknown.pop(record.operation, None)
            dispositions.append(self.observe(operation=record.operation, value=evidence.value,
                observation_id=f"{observation_prefix}_{index}", verified=True,
                complete=evidence.complete and state_version is not None,
                unfiltered=evidence.unfiltered, document=record.document,
                incomplete_reason=evidence.reason if not evidence.complete else None,
                subject_key=evidence.subject, evidence_ids=evidence.observation_ids))
            self._refresh_completed_purchase_state()
            if on_read is not None:
                on_read(index, record)
        self._state_version = state_version
        self._refresh_completed_purchase_state()
        return dispositions

    def _record_receipt(self, record):
        # A fresh receipt is necessary but not sufficient: require a subsequent
        # documented GET of that exact object before inspecting purchased items.
        if record.method != "POST" or not re.search(r"\b(?:place an? order|purchase|buy)\b",
                                                    str(record.document.get("description", "")), re.I):
            return
        value = record.value
        if not isinstance(value, Mapping):
            return
        schema = _success_schema(record.document)
        ids = [key for key in schema if key.endswith("_id") and key in value] if isinstance(schema, Mapping) else []
        if len(ids) != 1 or type(value[ids[0]]) not in {str, int}:
            return
        app = record.operation.split(".", 1)[0]
        self._purchase_receipts.setdefault(app, set()).add((ids[0], value[ids[0]]))
        self._receipt_subjects[(app, ids[0], value[ids[0]])] = record.subject_key

    def _record_identity_relations(self, app, key, row, document, evidence_id, subject):
        paths = _schema_field_paths(_success_schema(document))
        for field_name, value in row.items():
            if (field_name == key or not field_name.endswith("_id")
                    or type(value) not in {int, str} or not any(p[-1:] == (field_name,) for p in paths)):
                continue
            source, target = (app, key, row[key]), (app, field_name, value)
            edge_key = (source, field_name)
            old = self._identity_relations.get(edge_key)
            if old is not None and (old.target != target or old.subject_key != subject):
                self._identity_relations.pop(edge_key, None)
                self._relation_conflicts.add(edge_key)
            elif edge_key not in self._relation_conflicts:
                self._identity_relations[edge_key] = VerifiedIdentityRelation(
                    source, field_name, target, evidence_id, self._state_version, subject)

    def _merge_detail(self, identity, value, observation_id, subject):
        old_subject = self._detail_subjects.get(identity)
        if old_subject is not None and old_subject != subject:
            return False
        previous = self._details.setdefault(identity, {})
        conflicts = self._detail_conflicts.setdefault(identity, set())
        provenance = self._detail_provenance.setdefault(identity, {})
        for field_name, field_value in value.items():
            if field_name in previous and previous[field_name] != field_value:
                conflicts.add(field_name)
                previous.pop(field_name, None)
            elif field_name not in conflicts:
                previous[field_name] = copy.deepcopy(field_value)
                provenance[field_name] = tuple(dict.fromkeys((*provenance.get(field_name, ()), observation_id)))
        self._detail_subjects[identity] = subject
        return True

    def _observe_documented_detail(self, record, observation_id):
        path_ids = re.findall(r"\{([a-z_]+_id|id)\}", str(record.document.get("path", "")))
        if (len(path_ids) != 1 or not isinstance(record.value, Mapping)
                or not _observational_document(record.document)):
            return None
        key = path_ids[0]
        if key not in record.arguments or record.value.get(key) != record.arguments[key]:
            return "detail_identity_unknown"
        if type(record.value[key]) not in {str, int}:
            return "detail_identity_unknown"
        app = record.operation.split(".", 1)[0]
        identity = (app, key, record.value[key])
        # An object may be fully read even though its ID parameter restricts scope;
        # it must never stand in for a full collection read.
        if set(record.arguments) != {key}:
            return "filtered_detail_unknown"
        params = record.document.get("parameters", ())
        if not isinstance(params, (list, tuple)):
            return "filtered_detail_unknown"
        private = any(isinstance(p, Mapping) and p.get("required") and
                      p.get("name") in {"access_token", "token"} for p in params)
        if private:
            expected_subjects = set()
            for scope, (snapshot, _) in self._initial_snapshots.items():
                if scope.startswith(app + ".") and snapshot.identity_field == key and any(
                        row.get(key) == record.value[key] for row in snapshot.entities):
                    expected_subjects.update(self._scope_subjects.get(scope, ()))
            for receipt, snapshot in self._purchase_details.items():
                if receipt[0] == app and snapshot.identity_field == key and any(
                        row.get(key) == record.value[key] for row in snapshot.entities):
                    expected_subjects.add(self._receipt_subjects.get(receipt, ""))
            if expected_subjects and expected_subjects != {record.subject_key}:
                return "detail_subject_unknown"
        if (key, record.value[key]) in self._purchase_receipts.get(app, set()):
            if self._receipt_subjects.get(identity, record.subject_key) != record.subject_key:
                return "detail_subject_unknown"
        if not self._merge_detail(identity, record.value, observation_id, record.subject_key):
            return "detail_subject_unknown"
        self._detail_documents[identity] = record.document
        self._record_identity_relations(app, key, record.value, record.document, observation_id, record.subject_key)
        self._promote_contracts()
        self.qualification_requirements()
        if (key, record.value[key]) in self._purchase_receipts.get(app, set()):
            hints = {c.identity_field for c in self.task_target.constraints
                     if c.scope == app + ".completed_purchase_items" and c.identity_field is not None}
            if not hints:
                hints = {path[-1] for plan in self._matching_read_plans(record.operation, record.arguments)
                    for field, path in plan["field_projection"].items() if field == "$identity"}
            snapshot = _snapshot(self._details[identity], observation_id, record.document,
                                 next(iter(hints)) if len(hints) == 1 else None)
            if snapshot is not None and snapshot.collection_known:
                rows = []
                for row in snapshot.entities:
                    row = dict(row)
                    if "quantity" not in row and "ordered_quantity" in row:
                        row["quantity"] = row["ordered_quantity"]
                    rows.append(row)
                self._purchase_details[identity] = VerifiedState(tuple(rows), snapshot.fields,
                    identity_field=snapshot.identity_field, observation_id=observation_id,
                    operation=record.operation, evidence_ids=(observation_id,),
                    field_provenance=snapshot.field_provenance)
                self.current_verified_state[f"{app}.{key}:{record.value[key]}"] = self._purchase_details[identity]
        return "documented_detail"

    def _refresh_completed_purchase_state(self):
        for app, receipts in self._purchase_receipts.items():
            snapshots = [self._purchase_details.get((app, key, value)) for key, value in receipts]
            if not snapshots or any(s is None for s in snapshots):
                continue
            identity_fields = {s.identity_field for s in snapshots}
            if len(identity_fields) != 1:
                continue
            # Explicit quantities are additive for a shared entity; other conflicts
            # remain resource-scoped and cannot manufacture a verified field value.
            resources = {f"{app}.{key}:{value}": self._purchase_details[(app, key, value)]
                         for key, value in sorted(receipts, key=repr)}
            enriched_resources = {}
            for name, state in resources.items():
                rows, provenance = [], dict(state.field_provenance)
                evidence_ids = list(state.evidence_ids)
                for row in state.entities:
                    enriched = self._enriched_row(app, state.identity_field, row)
                    rows.append(enriched)
                    identity = (app, state.identity_field, row[state.identity_field])
                    for field_name in enriched:
                        sources = self._enrichment_sources(identity, field_name)
                        if sources:
                            provenance[field_name] = tuple(dict.fromkeys((*provenance.get(field_name, ()), *sources)))
                            evidence_ids.extend(sources)
                enriched_resources[name] = VerifiedState(tuple(rows), state.fields,
                    complete=state.complete, reliable=state.reliable, identity_field=state.identity_field,
                    observation_id=state.observation_id, operation=state.operation,
                    evidence_ids=tuple(dict.fromkeys(evidence_ids)), field_provenance=provenance)
            resources = enriched_resources
            state = aggregate_verified_states(resources, entity_additive_fields=("quantity",))
            state = replace(state, field_provenance={key: tuple(dict.fromkeys(
                source for resource in resources.values()
                for source in resource.field_provenance.get(key, ())))
                for key in set().union(*(resource.field_provenance for resource in resources.values()))})
            self.current_verified_state[app + ".completed_purchase_items"] = state
            if len(self._purchase_receipts) == 1:
                self.current_verified_state["completed_purchase_items"] = state

    @property
    def mutation_attempted(self):
        return self._mutation_attempted

    def observe_completion_answer(self, code, namespace):
        value = completion_answer_value(code, namespace)
        self.current_verified_state.pop("$answer", None)
        if value is not _MISSING:
            self.current_verified_state["$answer"] = VerifiedState(fields={"answer": value},
                                                                  collection_known=False)

    def invalidate_state(self):
        # Any attempted mutation or checkpoint restore can invalidate every cached read.
        # Keep the original required targets, but require a fresh read of actual state.
        self.current_verified_state.clear()
        # A violation proved on an abandoned version is not a final-state finding.
        self.metrics["completion_final_status"] = None
        self._details.clear()
        self._detail_provenance.clear()
        self._detail_conflicts.clear()
        self._detail_subjects.clear()
        self._detail_documents.clear()
        self._field_projections.clear()
        self._identity_relations.clear()
        self._relation_conflicts.clear()
        self._resolved_plans.clear()
        self._purchase_details.clear()
        self._qualified_states.clear()
        self._collection_evidence.invalidate()
        self._evidence_unknown.clear()
        self._state_version = None

    def synchronize_state_version(self, version: str | None):
        """Reuse the loop's existing fingerprint; never read additional private state."""
        if version is None or (self._state_version is not None and self._state_version != version):
            self.invalidate_state()
        self._state_version = version

    def record_mutation_attempt(self, record=None):
        self._mutation_attempted = True
        if record is None:
            self._mutated_scopes.add("*")
        else:
            app = record.operation.split(".", 1)[0]
            route = str(record.document.get("path", "")).split("/{", 1)[0].rstrip("/")
            affected = []
            closest = -1
            for api, doc in self._api_docs.get(app, {}).items():
                read_route = str(doc.get("path", "")).rstrip("/") if isinstance(doc, Mapping) else ""
                if (isinstance(doc, Mapping) and str(doc.get("method", "")).upper() == "GET"
                        and "{" not in read_route and read_route
                        and (route == read_route or route.startswith(read_route + "/"))):
                    scope = _collection(app + "." + api)
                    if scope:
                        if len(read_route) < closest:
                            continue
                        if len(read_route) > closest:
                            affected.clear()
                            closest = len(read_route)
                        affected.append(app + "." + scope)
            if not affected:
                name = route.rsplit("/", 1)[-1]
                affected = [app + "." + name] if re.fullmatch(r"[a-z_]+", name) else [app + ".*"]
            self._mutated_scopes.update(affected)
        self.invalidate_state()

    def feedback_once(self, result):
        """Deduplicate unresolved/mismatch feedback by actual state and evidence."""
        if result.status == "SATISFIED":
            return None
        # Constraint handles are per-verifier opaque labels, not feedback identity.
        key = hashlib.sha256(json.dumps([self._state_version, result.status,
            result.unknown_count, result.unknown_reasons,
            result.agent_details, self._evidence_unknown], sort_keys=True, default=str).encode()).hexdigest()
        if key in self._feedback_seen:
            self.metrics["completion_feedback_deduplicated"] += 1
            return None
        self._feedback_seen.add(key)
        self.metrics["completion_feedback_sent"] += 1
        feedback_result = copy.deepcopy(result)
        feedback_result.unknown_reasons.extend({"reason": reason, "operation": operation,
            "next_check": "Continue the same query through its remaining pages before whole-set comparisons"}
            for operation, reason in sorted(self._evidence_unknown.items()))
        return feedback_result.feedback()

    def _capabilities(self):
        if self._read_capabilities is None:
            capabilities = []
            for app, docs in self._api_docs.items():
                if not isinstance(docs, Mapping):
                    continue
                for api, doc in docs.items():
                    if not _observational_document(doc):
                        continue
                    params = doc.get("parameters")
                    if not isinstance(params, (list, tuple)) or any(
                            not isinstance(p, Mapping) or type(p.get("name")) is not str for p in params):
                        continue
                    ids = re.findall(r"\{([a-z_]+_id|id)\}", str(doc.get("path", "")))
                    if "{" in str(doc.get("path", "")) and len(ids) != 1:
                        continue
                    capabilities.append(ReadCapability(app + "." + api, str(doc.get("path", "")),
                        tuple(params), _schema_field_paths(_success_schema(doc)), ids[0] if ids else None))
            self._read_capabilities = tuple(capabilities)
        return self._read_capabilities

    def _register_requirement(self, purpose, scope, fields, app, *, identity_field=None,
                              identity_value=None, refs=(), evidence_ids=(), subject_key=""):
        identity = (purpose, scope, tuple(sorted(set(fields))), app, identity_field,
                    identity_value, tuple(refs), subject_key)
        rid = self._handle(identity)
        requirement = EvidenceRequirement(rid, purpose, scope, tuple(sorted(set(fields))), app,
            identity_field, identity_value, tuple(refs), tuple(evidence_ids), subject_key, self._state_version)
        previous = self._requirements.get(rid)
        self._requirements[rid] = requirement
        if previous is None:
            self._evidence_events.append({"kind": "EVIDENCE_REQUIREMENT", "requirement_id": rid,
                "stage": "created", "purpose": purpose, "scope": scope,
                "subject_handle": self._handle((app, subject_key, identity_field, identity_value)),
                "required_fields": list(requirement.required_fields), "constraint_refs": list(refs)})
        return requirement

    def _handle(self, value):
        return hashlib.sha256(self._audit_salt + repr(value).encode()).hexdigest()[:16]

    def qualification_requirements(self):
        requirements = []
        active = set()
        for index, binding in enumerate(self._contract_bindings):
            if (binding.origin, binding.destination, binding.selectors) in self._promoted:
                continue
            pairs = self._matching_scopes(binding.origin, self._initial_snapshots)
            if len(pairs) != 1:
                continue
            name, (snapshot, _) = pairs[0]
            if not snapshot.complete or not snapshot.reliable:
                continue
            app = name.split(".", 1)[0]
            subjects = self._scope_subjects.get(name, {""})
            if len(subjects) != 1:
                continue
            subject = next(iter(subjects))
            for row in snapshot.entities:
                enriched = self._enriched_row(app, snapshot.identity_field, row)
                outcomes = [self._select_from_initial(app, snapshot, enriched, sel) for sel in binding.selectors]
                if False in outcomes:
                    continue
                missing = [sel.field for sel, outcome in zip(binding.selectors, outcomes) if outcome is None]
                if missing:
                    req = self._register_requirement("TARGET_QUALIFICATION", binding.origin, missing, app,
                        identity_field=snapshot.identity_field, identity_value=row[snapshot.identity_field],
                        refs=(f"binding_{index}",), evidence_ids=snapshot.evidence_ids, subject_key=subject)
                    requirements.append(req)
                    active.add(req.requirement_id)
        for rid, req in list(self._requirements.items()):
            if req.purpose == "TARGET_QUALIFICATION" and rid not in active:
                refs = [r for r in req.constraint_refs if r.startswith("binding_")]
                if len(refs) != 1:
                    continue
                binding = self._contract_bindings[int(refs[0].split("_")[1])]
                pairs = self._matching_scopes(binding.origin, self._initial_snapshots)
                if len(pairs) != 1:
                    continue
                snapshot = pairs[0][1][0]
                rows = [row for row in snapshot.entities if row.get(req.identity_field) == req.identity_value]
                if len(rows) != 1 or any(self._select_from_initial(req.app, snapshot,
                        self._enriched_row(req.app, req.identity_field, rows[0]), selector) is None
                        for selector in binding.selectors):
                    continue
                if self._resolutions.get((rid, "resolved")) != self._state_version:
                    self._evidence_events.append({"kind": "EVIDENCE_REQUIREMENT", "requirement_id": rid,
                        "stage": "resolved", "purpose": req.purpose,
                        "target_promotion": (binding.origin, binding.destination, binding.selectors) in self._promoted})
                    self._resolutions[(rid, "resolved")] = self._state_version
        return requirements

    @staticmethod
    def _argument_compatible(parameter, value):
        kind = parameter.get("type")
        if kind in {"integer", "int"}:
            return type(value) is int
        if kind in {"string", "str"}:
            return type(value) is str
        if kind in {"number", "float"}:
            return type(value) in {int, float} and _number(value) is not None
        # Existing fixtures can omit type; the verified ID is still bounded.
        return kind is None and type(value) in {int, str}

    def resolve_read_requirement(self, req):
        candidates = []
        for cap in self._capabilities():
            app = cap.operation.split(".", 1)[0]
            if app != req.app:
                continue
            args, relation = {}, None
            if req.identity_field is not None:
                if cap.identity_argument != req.identity_field:
                    # A qualified semantic field can use a verified foreign ID edge.
                    if len(req.required_fields) != 1:
                        continue
                    qualifier, sep, field = req.required_fields[0].rpartition("_")
                    if not sep or cap.identity_argument != qualifier + "_id":
                        continue
                    source = (app, req.identity_field, req.identity_value)
                    edge = self._identity_relations.get((source, cap.identity_argument))
                    if (edge is None or edge.state_version != req.state_version
                            or edge.subject_key not in {"", req.subject_key}):
                        continue
                    param = next((p for p in cap.parameters if p["name"] == cap.identity_argument), {})
                    description = str(param.get("description", "")).lower()
                    if not re.search(r"\b" + re.escape(qualifier) + r"\b", description):
                        continue  # Names alone cannot prove an entity relation.
                    args[cap.identity_argument] = edge.target[2]
                    relation = {"source_field": edge.relation_field, "target_field": field,
                                "evidence_id": edge.evidence_id}
                else:
                    args[cap.identity_argument] = req.identity_value
                if (cap.identity_argument,) not in cap.returned_field_paths:
                    continue  # Require identity echo before binding detail results.
            else:
                if cap.identity_argument is not None:
                    continue
                scope = _collection(cap.operation)
                if not scope or not self._matching_scopes(req.scope,
                        {app + "." + scope: VerifiedState(identity_field="id")}):
                    continue
            if any(p.get("required") and p.get("default") is None
                    and p["name"] not in {*args, "access_token", "token"} for p in cap.parameters):
                continue
            if any(p["name"] in args and not self._argument_compatible(p, args[p["name"]])
                   for p in cap.parameters):
                continue
            projections = {}
            intermediate = False
            paths = cap.returned_field_paths
            if req.purpose == "FINAL_STATE" and req.identity_field is not None:
                # Receipt ID identifies the parent; target IDs identify one documented item array.
                constraints = [c for c in self.task_target.constraints if c.scope == req.scope]
                identities = {c.identity_field for c in constraints if c.identity_field is not None}
                if not identities:
                    identities = {p[-1] for p in paths if len(p) == 3 and p[1] == "[]"
                        and any(c.operation.split(".", 1)[0] == app and c.identity_argument == p[-1]
                            and (p[-1],) in c.returned_field_paths for c in self._capabilities())}
                if len(identities) != 1:
                    continue
                item_identity = next(iter(identities))
                roots = {p[:-1] for p in paths if p[-1] == item_identity and "[]" in p
                         and len(p) == 3 and p[1] == "[]"}
                if len(roots) != 1:
                    continue
                root = next(iter(roots))
                projections["$identity"] = root + (item_identity,)
                for field in req.required_fields:
                    matches = [p for p in paths if p[:-1] == root and p[-1] in _field_names(field)]
                    if len(matches) != 1:
                        if "_" in field and not matches:
                            intermediate = True
                            continue
                        break
                    projections[field] = matches[0]
                else:
                    plan = self._read_plan(cap, req, args, projections, relation)
                    plan["intermediate"] = intermediate
                    candidates.append(plan)
                continue
            for field in req.required_fields:
                names = (relation["target_field"],) if relation else _field_names(field)
                matches = [p for p in paths if p in [tuple(n.split(".")) for n in names]]
                if req.identity_field is None:
                    # Collection evidence must project fields from one item shape, not any nested key.
                    matches += [p for p in paths if p[-1] in names and "[]" in p
                                and (len(p) == 2 or (len(p) == 3 and p[1] == "[]"))]
                if len(set(matches)) != 1:
                    qualifier, sep, target_field = field.rpartition("_")
                    relation_key = qualifier + "_id"
                    source = (app, req.identity_field, req.identity_value)
                    if (req.identity_field is not None and relation is None and sep
                            and (relation_key,) in paths
                            and (source, relation_key) not in self._identity_relations
                            and (source, relation_key) not in self._relation_conflicts
                            and any(c.identity_argument == relation_key and
                                c.operation.split(".", 1)[0] == app and (target_field,) in c.returned_field_paths
                                and any(p["name"] == relation_key and re.search(r"\b" + re.escape(qualifier) + r"\b",
                                    str(p.get("description", "")), re.I) for p in c.parameters)
                                for c in self._capabilities())):
                        projections["$relation_identity"] = (relation_key,)
                        intermediate = True
                        continue
                    break
                projections[field] = matches[0]
            else:
                if req.identity_field is None and not req.required_fields and not any(
                        p[-1] == "id" or p[-1].endswith("_id") for p in paths):
                    continue
                plan = self._read_plan(cap, req, args, projections, relation)
                plan["intermediate"] = intermediate
                candidates.append(plan)
        status = "RELIABLE" if len(candidates) == 1 else "AMBIGUOUS" if candidates else "NO_PATH"
        result = {"status": status, "requirement_id": req.requirement_id,
                  "reason": "unique_schema_and_identity_match" if status == "RELIABLE" else
                            "multiple_proven_candidates" if candidates else "schema_scope_or_binding_unproven",
                  "candidates": [c["operation"] for c in candidates]}
        signature = json.dumps((req.state_version, result), sort_keys=True)
        if self._resolutions.get(req.requirement_id) != signature:
            self._evidence_events.append({"kind": "READ_RESOLUTION", **result})
            self._resolutions[req.requirement_id] = signature
        if len(candidates) == 1:
            result["plan"] = candidates[0]
            self._resolved_plans[req.requirement_id] = candidates[0]
        else:
            self._resolved_plans.pop(req.requirement_id, None)
        return result

    @staticmethod
    def _read_plan(cap, req, args, projections, relation):
        return {"operation": cap.operation, "scope": req.scope, "arguments": args,
                "purpose": req.purpose, "requirement_ids": [req.requirement_id],
                "required_fields": list(req.required_fields),
                "field_projection": {k: list(v) for k, v in projections.items()},
                "identity_field": req.identity_field, "identity_value": req.identity_value,
                "subject_key": req.subject_key, "state_version": req.state_version,
                "relation": relation, "required_authentication": [p["name"] for p in cap.parameters
                    if p.get("required") and p["name"] in {"access_token", "token"}]}

    def evidence_plan(self, result):
        """Completion acquisition excludes early qualification, which uses ordinary budgets."""
        requirements = []
        for reason in result.unknown_reasons:
            scope = reason.get("suggested_verification_scope")
            if not scope or scope == "$answer":
                continue
            field = reason.get("field")
            refs = (reason["constraint_id"],) if reason.get("constraint_id") else ()
            if scope.endswith("completed_purchase_items"):
                for app, receipts in self._purchase_receipts.items():
                    if scope not in {"completed_purchase_items", app + ".completed_purchase_items"}:
                        continue
                    fields = tuple(dict.fromkeys(c.field for c in self.task_target.constraints
                        if c.scope == scope and c.field))
                    for key, value in receipts:
                        subject = self._receipt_subjects.get((app, key, value), "")
                        state = self.current_verified_state.get(scope)
                        if state is not None and state.reliable and state.complete and state.entities:
                            for row in state.entities:
                                missing = tuple(f for f in fields if f not in row)
                                if missing:
                                    requirements.append(self._register_requirement("FINAL_STATE_FIELD", scope, missing, app,
                                        identity_field=state.identity_field, identity_value=row[state.identity_field],
                                        refs=refs, subject_key=subject, evidence_ids=state.evidence_ids))
                        else:
                            requirements.append(self._register_requirement("FINAL_STATE", scope, fields, app,
                                identity_field=key, identity_value=value, refs=refs, subject_key=subject))
            else:
                apps = (scope.split(".", 1)[0],) if "." in scope else tuple(self._api_docs)
                for app in apps:
                    requirements.append(self._register_requirement("FINAL_STATE", scope,
                        (field,) if field else (), app, refs=refs))
        for req in self.qualification_requirements():
            result.unknown_reasons.append({"reason": "missing_required_attribute",
                "missing_evidence_type": "missing_required_attribute", "fields": list(req.required_fields),
                "requirement_id": req.requirement_id, "suggested_verification_scope": req.scope})
        plans = {}
        for req in requirements:
            resolution = self.resolve_read_requirement(req)
            if resolution["status"] == "RELIABLE":
                plan = resolution["plan"]
                key = json.dumps((plan["operation"], plan["scope"], plan["arguments"]), sort_keys=True)
                if key in plans:
                    plans[key]["requirement_ids"].extend(plan["requirement_ids"])
                else:
                    plans[key] = plan
        return list(plans.values())

    def qualification_guidance(self, limit=5):
        requirements = self.qualification_requirements()
        reads = []
        for req in requirements:
            resolution = self.resolve_read_requirement(req)
            if resolution["status"] != "RELIABLE":
                continue
            signature = (req.requirement_id, req.state_version, self._resolutions[req.requirement_id])
            if signature in self._guidance_seen or len(reads) >= limit:
                continue
            self._guidance_seen.add(signature)
            plan = resolution["plan"]
            reads.append(plan)
            self._evidence_events.append({"kind": "EVIDENCE_READ", "stage": "recommended",
                "operation": plan["operation"], "requirement_ids": plan["requirement_ids"]})
        if not reads:
            return None
        return {"purpose": "TARGET_QUALIFICATION", "pending_object_field_gaps": len(requirements),
                "reads": reads, "budget": "ordinary_step_and_api_budget",
                "instruction": "Use normal proposals to obtain these public fields for the bound objects. "
                    "Candidates are not required targets. Only proven matches can be promoted; "
                    "do not infer the remaining objects from a few matches."}

    def drain_evidence_events(self):
        events, self._evidence_events = self._evidence_events, []
        return events

    def _matching_read_plans(self, operation, arguments):
        return [p for p in self._resolved_plans.values() if p["operation"] == operation
            and p["state_version"] == self._state_version
            and all(arguments.get(key, _MISSING) == value for key, value in p["arguments"].items())]

    def record_read_attempt(self, attempt):
        if attempt.get("method") != "GET":
            return
        plans = self._matching_read_plans(attempt["operation"], attempt.get("arguments", {}))
        if not plans:
            return
        refs = sorted({rid for plan in plans for rid in plan["requirement_ids"]})
        for stage, flag in (("dispatched", "dispatched"), ("returned", "returned")):
            if attempt.get(flag):
                self._evidence_events.append({"kind": "EVIDENCE_READ", "stage": stage,
                    "operation": attempt["operation"], "requirement_ids": refs})

    def _record_committed_read(self, record, observation_id, disposition):
        for plan in self._matching_read_plans(record.operation, record.arguments):
            private = bool(plan["required_authentication"])
            account_matches = not private or not plan["subject_key"] or plan["subject_key"] == record.subject_key
            bound_key = next(iter(plan["arguments"]), None)
            identity_matches = (bound_key is None or (isinstance(record.value, Mapping)
                and record.value.get(bound_key, _MISSING) == plan["arguments"][bound_key]))
            provided = all(_values_at_path(record.value, path) for path in plan["field_projection"].values())
            # Missing/conflicting/filtered details never become bound just because HTTP succeeded.
            accepted = disposition not in {"detail_identity_unknown", "detail_subject_unknown", "filtered_detail_unknown"}
            self._evidence_events.append({"kind": "EVIDENCE_READ", "stage": "committed",
                "operation": record.operation, "requirement_ids": plan["requirement_ids"],
                "evidence_id": observation_id})
            bound = account_matches and identity_matches and provided and accepted
            if bound and not plan.get("intermediate") and plan["purpose"] != "FINAL_STATE":
                identity = (record.operation.split(".", 1)[0], plan["identity_field"], plan["identity_value"])
                projections = self._field_projections.setdefault(identity, {})
                for name, path in plan["field_projection"].items():
                    values = _values_at_path(record.value, path)
                    if not name.startswith("$") and len(values) == 1:
                        previous = projections.get(name)
                        if previous is not None and previous[0] != values[0]:
                            projections[name] = (_MISSING, observation_id, plan["subject_key"])
                        elif previous is None or previous[0] is not _MISSING:
                            projections[name] = (copy.deepcopy(values[0]), observation_id, plan["subject_key"])
            self._evidence_events.append({"kind": "EVIDENCE_READ", "stage": "bound" if
                bound else "binding_rejected",
                "operation": record.operation, "requirement_ids": plan["requirement_ids"],
                "evidence_id": observation_id, "intermediate": plan.get("intermediate", False)})

    def record_requirement_outcomes(self, verifier):
        self.qualification_requirements()
        if verifier is None:
            return
        for rid, req in self._requirements.items():
            if req.purpose == "TARGET_QUALIFICATION":
                continue
            constraints = [c for index, c in enumerate(self.task_target.constraints, 1)
                if f"constraint_{index}" in req.constraint_refs]
            outcomes = [verifier._check(c, self.current_verified_state.get(c.scope))[0] for c in constraints]
            known = bool(outcomes) and all(outcome is not None for outcome in outcomes)
            signature = (self._state_version, known, tuple(outcomes))
            if self._resolutions.get((rid, "outcome")) == signature:
                continue
            self._resolutions[(rid, "outcome")] = signature
            self._evidence_events.append({"kind": "EVIDENCE_REQUIREMENT", "requirement_id": rid,
                "purpose": req.purpose, "stage": "resolved" if known else "still_unknown",
                "final_state_verified": known and all(outcomes)})

    def _enrichment_sources(self, identity, field_name):
        sources = list(self._detail_provenance.get(identity, {}).get(field_name, ()))
        projection = self._field_projections.get(identity, {}).get(field_name)
        if projection is not None and projection[0] is not _MISSING:
            sources.append(projection[1])
        for (source, relation_field), edge in self._identity_relations.items():
            prefix = relation_field.removesuffix("_id") + "_"
            if source == identity and edge.state_version == self._state_version and field_name.startswith(prefix):
                sources.append(edge.evidence_id)
                sources.extend(self._detail_provenance.get(edge.target, {}).get(field_name[len(prefix):], ()))
        return tuple(dict.fromkeys(sources))

    def _enriched_row(self, app, identity_field, row):
        identity = (app, identity_field, row[identity_field])
        details = self._details.get(identity, {})
        enriched = dict(row)
        conflicts = self._detail_conflicts.get(identity, set())
        for key, value in details.items():
            # Initial quantities remain task-source facts; details cannot rewrite them.
            if key in {identity_field, "quantity"}:
                continue
            if key in enriched and enriched[key] != value:
                enriched.pop(key, None)
                conflicts = conflicts | {key}
            elif key not in conflicts:
                enriched[key] = value
        for key in conflicts:
            enriched.pop(key, None)
        for canonical in ("type", "rating"):
            names = [name for name in _field_names(canonical) if name in enriched]
            if len(names) > 1 and any(enriched[name] != enriched[names[0]] for name in names[1:]):
                for name in names:
                    enriched.pop(name, None)
        for name, (value, evidence_id, subject) in self._field_projections.get(identity, {}).items():
            if value is _MISSING or (name in enriched and enriched[name] != value):
                enriched.pop(name, None)
            else:
                enriched[name] = value
        # Projection follows one verified relation, never merges foreign objects wholesale.
        for (source, relation_field), edge in self._identity_relations.items():
            if source != identity or edge.state_version != self._state_version:
                continue
            target = self._details.get(edge.target, {})
            document = self._detail_documents.get(edge.target, {})
            params = document.get("parameters", ())
            if not isinstance(params, (list, tuple)):
                continue
            private = any(p.get("required") and p.get("name") in {"access_token", "token"}
                          for p in params if isinstance(p, Mapping))
            if private and self._detail_subjects.get(edge.target) != edge.subject_key:
                continue
            qualifier = relation_field.removesuffix("_id")
            if not any(isinstance(p, Mapping) and p.get("name") == relation_field and
                       re.search(r"\b" + re.escape(qualifier) + r"\b", str(p.get("description", "")), re.I)
                       for p in params):
                continue
            declared = _schema_field_paths(_success_schema(document))
            for key, value in target.items():
                semantic = qualifier + "_" + key
                if ((key,) in declared and key not in self._detail_conflicts.get(edge.target, set())
                        and semantic not in enriched):
                    enriched[semantic] = value
        return enriched

    def checkpoint_state(self):
        keys = ("task_target", "_initial_snapshots", "_promoted", "_mutated_scopes",
                "_purchase_receipts", "_scope_subjects", "_scope_apps")
        return {key: copy.deepcopy(getattr(self, key)) for key in keys}

    def restore_checkpoint_state(self, state):
        for key, value in state.items():
            setattr(self, key, copy.deepcopy(value))
        self._mutation_attempted = bool(self._mutated_scopes)
        self.metrics["task_target_constraints"] = len(self.task_target.constraints)
        self.metrics["task_targets_created"] = int(bool(self.task_target.constraints))
        self.invalidate_state()

    def record_verification(self, result: CompletionVerificationResult, *, count_block=True):
        self.metrics["completion_verification_checks"] += 1
        self.metrics["completion_checks"] += 1
        counts = self.metrics["completion_unknown_reason_counts"]
        for reason in result.unknown_reasons:
            key = reason.get("missing_evidence_type", reason["reason"])
            counts[key] = counts.get(key, 0) + 1
        self.metrics["completion_" + result.status.lower()] += 1
        self.metrics["completion_blocks"] += int(count_block and result.blocks_completion)
        if result.status == "UNKNOWN" or result.unknown_count:
            self.metrics["completion_verification_unknown"] = True
        for name in ("missing_target_count", "unexpected_target_count", "quantity_mismatch_count",
                     "field_mismatch_count", "constraint_violation_count"):
            self.metrics[name] += getattr(result, name)

    def update_repair_episodes(self, result, verifier, *, state_version, constraints=None):
        if result.blocks_completion:
            ids = tuple(sorted(v["constraint_id"] for v in result.violations))
            comparable = [{k: v for k, v in detail.items() if k not in {"evidence_ids", "observation_id"}}
                          for detail in result.agent_details]
            fingerprint = hashlib.sha256(json.dumps(comparable, sort_keys=True, default=str).encode()).hexdigest()
            if not any(not e["repair_resolved"] and e["violation_fingerprint"] == fingerprint for e in self._repair_episodes):
                self._repair_episodes.append({"repair_episode_id": f"repair_{len(self._repair_episodes)+1}",
                    "repair_constraint_ids": ids, "violation_fingerprint": fingerprint,
                    "repair_started": True, "repair_resolved": False,
                    "blocked_state_version": state_version,
                    "blocked_evidence_ids": sorted({x for v in result.violations for x in v.get("evidence_ids", [])})})
                self._repair_constraints[self._repair_episodes[-1]["repair_episode_id"]] = tuple(
                    copy.deepcopy((constraints if constraints is not None else self.task_target.constraints)[int(label.rsplit("_", 1)[1])-1]) for label in ids)
                self.metrics["repair_started"] += 1
                self.metrics["confirmed_violation_episodes"] += 1
        for episode in self._repair_episodes:
            if episode["repair_resolved"]:
                continue
            checked, evidence, fresh = [], [], []
            for c in self._repair_constraints[episode["repair_episode_id"]]:
                state = self.current_verified_state.get(c.scope)
                checked.append(verifier._check(c, state)[0])
                ids = (state.evidence_ids or ((state.observation_id,) if state.observation_id else ())) if state else ()
                evidence.extend(ids)
                fresh.append(bool(set(ids) - set(episode["blocked_evidence_ids"])))
            if checked and all(v is True for v in checked) and all(fresh):
                episode["repair_resolved"] = True
                episode["resolved_evidence_ids"] = sorted(set(evidence))
                episode["resolved_state_version"] = state_version
                self.metrics["repair_resolved"] += 1
        return copy.deepcopy(self._repair_episodes)

    def audit_metrics(self):
        return dict(self.metrics, repair_episodes=copy.deepcopy(self._repair_episodes))

    def audit_constraints(self):
        """Retain provenance traceability without serializing expected/observed values."""
        return [{
            "constraint_id": f"constraint_{index}",
            "predicate": constraint.predicate,
            "scope_handle": hashlib.sha256(self._audit_salt + constraint.scope.encode()).hexdigest()[:16],
            "field": constraint.field,
            "provenance": {
                "source": constraint.provenance.source,
                "operation": constraint.provenance.operation,
                "observation_id": constraint.provenance.observation_id,
            },
        } for index, constraint in enumerate(self.task_target.constraints, 1)]

    def audit_contract(self):
        return {"parsed_clauses": self.metrics["task_contract_parsed_clauses"],
                "unresolved_clause_count": len(self.task_target.unresolved_clauses),
                "pending_bindings": self.task_target.pending_bindings,
                "unresolved_clause_handles": [hashlib.sha256(self._audit_salt + c.encode()).hexdigest()[:16]
                                               for c in self.task_target.unresolved_clauses]}


_MISSING = object()


def completion_answer_value(code, namespace):
    """Only literal values or inert names; never evaluate generated expressions."""
    try:
        calls = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "complete_task"]
        if len(calls) != 1:
            return _MISSING
        answers = [kw.value for kw in calls[0].keywords if kw.arg == "answer"]
        if len(answers) != 1:
            return _MISSING
        node = answers[0]
        # An assignment in this very proposal has not executed yet: namespace
        # values may be stale and cannot certify a forthcoming answer.
        assigned = {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if isinstance(node, ast.Name) and node.id not in assigned and isinstance(namespace, Mapping):
            value = namespace.get(node.id, _MISSING)
            return copy.deepcopy(value) if type(value) in {str, int, float, list, dict, type(None)} else _MISSING
        return ast.literal_eval(node)
    except (SyntaxError, ValueError, TypeError):
        return _MISSING


def verified_read_payload(code: str, output: str, namespace: Mapping[str, Any] | None):
    """Read direct API return values only; arbitrary model-rendered output is untrusted.

    Accepted code shapes are print(apis.x.read()) or assignment of that call followed
    by print of the unmodified variable. Loops, transforms and multiple calls are unknown.
    JSON text must match the direct returned value shape. No model expression is evaluated.
    """
    try:
        if namespace is not None and "print" in namespace:
            return None  # A rebound renderer is not direct-return evidence.
        statements = [node for node in ast.parse(code).body if not (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HARNESS_META"
                                                for t in node.targets))]
        def api_call(node):
            return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "apis")
        call = None
        value = None
        if len(statements) == 1 and isinstance(statements[0], ast.Expr):
            print_call = statements[0].value
            if (isinstance(print_call, ast.Call) and isinstance(print_call.func, ast.Name)
                and print_call.func.id == "print" and len(print_call.args) == 1
                and not print_call.keywords and api_call(print_call.args[0])):
                call = print_call.args[0]
                try:
                    value = json.loads(output)
                except (ValueError, TypeError):
                    value = ast.literal_eval(output)
        elif len(statements) in {1, 2} and isinstance(statements[0], ast.Assign):
            assignment = statements[0]
            if (len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name)
                and api_call(assignment.value) and namespace is not None):
                if len(statements) == 2:
                    printed = statements[1].value if isinstance(statements[1], ast.Expr) else None
                    if not (isinstance(printed, ast.Call) and isinstance(printed.func, ast.Name)
                            and printed.func.id == "print" and not printed.keywords
                            and len(printed.args) == 1 and isinstance(printed.args[0], ast.Name)
                            and printed.args[0].id == assignment.targets[0].id):
                        return None
                call = assignment.value
                value = namespace.get(assignment.targets[0].id)
        if call is None or not isinstance(value, (dict, list)):
            return None
        # Filter/pagination/subject arguments can make a collection incomplete or ambiguous.
        unfiltered = not call.args and all(keyword.arg in {"access_token", "token"}
                                          for keyword in call.keywords)
        operation = ast.unparse(call.func).removeprefix("apis.")
        return operation, value, unfiltered
    except (SyntaxError, ValueError, TypeError, KeyError):
        return None


class CompletionEvidencePending:
    """Small task-local acquisition state; never part of rollback checkpoint context."""
    def __init__(self, budget=2):
        if type(budget) is not int or budget < 0:
            raise ValueError("completion evidence read budget must be a nonnegative integer")
        self.budget = budget
        self.pending = None
        self.gaps = {}
        self.reads = 0
        self.episodes = 0
        self.exhausted = 0

    def assess(self, result, plan, state_version):
        if result.status != "UNKNOWN":
            self.pending = None
            return result.status
        if not plan or not self.budget:
            self.pending = None
            return "COMPLETED_WITH_UNCERTAINTY"
        # Version is deliberately NOT part of identity: writes/restores/repeated completion
        # cannot replenish acquisition allowance for the same constraint/scope gap.
        descriptors = sorted({(str(r.get("constraint_id") or ""), str(r.get("suggested_verification_scope") or ""),
                               str(r.get("field") or "")) for r in result.unknown_reasons})
        key = hashlib.sha256(json.dumps(descriptors, sort_keys=True).encode()).hexdigest()
        gap = self.gaps.get(key)
        if gap is None:
            # Also bound the entire task's acquisition calls, so changing labels cannot loop.
            gap = {"evidence_gap_key": key, "pending_reason": result.unknown_reasons,
                   "constraint_ids": [r["constraint_id"] for r in result.unknown_reasons if "constraint_id" in r],
                   "state_version": state_version, "read_allowance": self.budget,
                   "attempts_used": 0, "plan": plan, "exhaustion_recorded": False}
            self.gaps[key] = gap
            self.episodes += 1
        gap.update(plan=plan, state_version=state_version, pending_reason=result.unknown_reasons)
        if gap["attempts_used"] >= self.budget or self.reads >= self.budget:
            if not gap["exhaustion_recorded"]:
                gap["exhaustion_recorded"] = True
                self.exhausted += 1
            self.pending = None
            return "COMPLETED_WITH_UNCERTAINTY"
        self.pending = gap
        return "COMPLETION_EVIDENCE_PENDING"

    def related(self, operation, arguments=None):
        return bool(self.pending and any(p["operation"] == operation and
            (arguments is None or all(arguments.get(k, _MISSING) == v for k, v in p["arguments"].items()))
            for p in self.pending["plan"]))

    def consume(self, attempts):
        if not self.pending:
            return 0
        count = sum(a["method"] == "GET" and self.related(a["operation"], a.get("arguments", {}))
                    for a in attempts)
        self.pending["attempts_used"] += count
        self.reads += count
        return count

    def exit_for_write(self):
        self.pending = None


def final_semantic_preflight(tracker, verifier, actions):
    """Only an explicit public execution descriptor may bind a commit to a scope.

    Descriptor: execution_metadata.finalize = {input_scope, result_scope,
    identity_field, projected_fields}. It must come from API documentation, never
    HARNESS_META or an operation-name/HTTP-method guess. Unbound effects are advisory.
    """
    checks = []
    for index, action in enumerate(actions):
        # Actions carry routes, not always app names; require a unique documented match.
        matches = [(app, doc) for app, docs in tracker._api_docs.items() if isinstance(docs, Mapping) for doc in docs.values()
            if isinstance(doc, Mapping) and str(doc.get("method", "")).upper() == action.method
            and str(doc.get("path", "")) == action.route]
        if len(matches) != 1:
            continue
        app, doc = matches[0]
        metadata = doc.get("execution_metadata", {})
        descriptor = metadata.get("finalize") if isinstance(metadata, Mapping) else None
        if descriptor is None:
            # An explicitly described collection-consuming effect can be reported, but
            # natural prose alone is not a machine-verifiable scope descriptor.
            if re.search(r"\b(?:consume|submit|process|order)\b.*\ball (?:items|records|entries)\b", str(doc.get("description", "")), re.I):
                checks.append({"status": "UNKNOWN", "reason": "operation_scope_unbound", "action_index": index})
            continue
        required = {"input_scope", "result_scope", "identity_field", "projected_fields"}
        if not isinstance(descriptor, Mapping) or not required <= set(descriptor) or not isinstance(descriptor["projected_fields"], (list, tuple)):
            checks.append({"status": "UNKNOWN", "reason": "operation_scope_unbound", "action_index": index})
            continue
        origin, destination = descriptor["input_scope"], descriptor["result_scope"]
        if not all(type(v) is str and v for v in (origin, destination, descriptor["identity_field"])):
            checks.append({"status": "UNKNOWN", "reason": "operation_scope_unbound", "action_index": index})
            continue
        origin = origin if "." in origin else app + "." + origin
        destination = destination if "." in destination else app + "." + destination
        # This descriptor projects fields within one application. Cross-application
        # mappings need an explicit identity transform, which this interface lacks.
        if (origin.split(".", 1)[0] != app or destination.split(".", 1)[0] != app
                or any(type(v) is not str or not v for v in descriptor["projected_fields"])):
            checks.append({"status": "UNKNOWN", "reason": "operation_scope_unbound", "action_index": index})
            continue
        constraints = [c for c in tracker.task_target.constraints if c.scope == destination and
            (c.predicate.startswith("ENTITY_SET_") or c.field in descriptor["projected_fields"])]
        if not constraints:
            checks.append({"status": "UNKNOWN", "reason": "no_bound_constraints", "action_index": index})
            continue
        if any(a.category == "WRITE_RESOURCE" for a in actions[:index]):
            checks.append({"status": "EVIDENCE_PENDING", "reason": "split_mutation_and_finalize",
                           "action_index": index})
            continue
        state = tracker.current_verified_state.get(origin)
        if state is not None and state.identity_field != descriptor["identity_field"]:
            state = None
        target = TaskTarget([TargetConstraint(c.predicate, origin, c.expected, c.provenance,
            c.field, c.entity, c.identity_field) for c in constraints])
        result = verifier.verify(target, {origin: state} if state is not None else {})
        checks.append({"status": result.status, "action_index": index, "result": result,
                       "verification_scope": origin, "constraints": target.constraints})
    return checks
