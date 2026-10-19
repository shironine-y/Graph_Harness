from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

from harness.task_contracts import parse_public_contract
from harness.proposal_extraction import extract_output_requirements, extract_reorder_scopes, month_label


@dataclass(frozen=True)
class ContractSource:
    start: int
    end: int
    instruction_sha256: str


@dataclass(frozen=True)
class OutputContract:
    answer_mode: str = "UNKNOWN"
    answer_port: tuple[str, str, str] | None = None
    order: tuple[str, ...] | None = None
    sources: tuple[ContractSource, ...] = ()
    unresolved_reason: str | None = None
    order_normalizer: str = "EXACT"
    order_basis: str | None = None
    order_source: ContractSource | None = None
    request_unresolved_reason: str | None = None
    answer_basis: str | None = None


@dataclass(frozen=True)
class FieldContract:
    scope: str
    allowed_fields: tuple[str, ...]
    source: ContractSource


@dataclass(frozen=True)
class ProposalContracts:
    output: OutputContract
    writes: tuple[FieldContract, ...] = ()
    extraction_issues: tuple[tuple[str, str, int, int], ...] = ()
    instruction_sha256: str | None = None


@dataclass(frozen=True)
class Finding:
    check: str
    status: str
    reason: str
    scope: str | None = None
    fields: tuple[str, ...] = ()
    source: ContractSource | None = None

    def audit(self):
        # No proposal values, answer, expected labels, tokens, raw instruction or API errors.
        return {"check": self.check, "status": self.status, "reason": self.reason,
                "scope": self.scope, "fields": list(self.fields),
                "source": None if self.source is None else {
                    "start": self.source.start, "end": self.source.end,
                    "instruction_sha256": self.source.instruction_sha256}}


def finding_status(findings):
    statuses = {f.status for f in findings}
    return ("VIOLATED" if "VIOLATED" in statuses else "UNKNOWN" if "UNKNOWN" in statuses
            else "SATISFIED" if "SATISFIED" in statuses else "NOT_APPLICABLE")


def failures(findings):
    return ["Proposal contract violation: " + f.reason +
            ("; disallowed fields=" + ", ".join(f.fields) if f.fields else "")
            for f in findings if f.status == "VIOLATED"]


def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return [node.id, *reversed(parts)] if isinstance(node, ast.Name) else None


def _doc(docs, app, name):
    methods = docs.get(app, {})
    value = methods.get(name, {}) if isinstance(methods, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _conditional_calls(tree):
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    conditional = set()
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        parent = parents.get(call)
        while parent is not None:
            if isinstance(parent, (ast.If, ast.IfExp, ast.For, ast.While, ast.Try,
                                   ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                                   ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                conditional.add(call)
                break
            parent = parents.get(parent)
    return conditional


def _inert(value, budget=None, depth=0):
    budget = [10000] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or depth > 12:
        return False
    if value is None or type(value) in {str, bool, int}:
        return type(value) is not str or len(value) <= 100000
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in {tuple, list}:
        return all(_inert(v, budget, depth + 1) for v in value)
    if type(value) is dict:
        return all(type(k) in {str, int, float, bool} and _inert(k, budget, depth + 1)
                   and _inert(v, budget, depth + 1) for k, v in value.items())
    return False


def _value(node, namespace, assigned):
    if any(isinstance(n, ast.Name) and n.id in assigned for n in ast.walk(node)):
        return False, None  # Same-step assignments must not read stale namespace values.
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        value = namespace.get(node.id) if isinstance(node, ast.Name) and isinstance(namespace, Mapping) and node.id in namespace else None
        if not (isinstance(node, ast.Name) and isinstance(namespace, Mapping) and node.id in namespace):
            return False, None
    return (True, value) if _inert(value) else (False, None)


def _completion_port(docs):
    """The public AppWorld answer protocol adapter, independent of task wording.

    The adapter requires the documented default and both normative conditions.
    Missing/incompatible documentation is not replaced with a guessed protocol.
    """
    ports = []
    for app, methods in docs.items():
        if not isinstance(methods, Mapping):
            continue
        for name, doc in methods.items():
            if not isinstance(doc, Mapping) or str(doc.get("method", "")).upper() != "POST" or doc.get("path") != "/supervisor/message":
                continue
            for parameter in doc.get("parameters") or ():
                if not isinstance(parameter, Mapping) or parameter.get("name") != "answer":
                    continue
                description = " ".join(str(parameter.get("description", "")).split())
                if (parameter.get("default", object()) is None
                        and "Pass it if and only if the task requests an answer" in description
                        and "the answer must be left to the default value" in description):
                    ports.append((app, name, parameter["name"]))
    return ports[0] if len(ports) == 1 else None


def completion_protocol_guidance(docs):
    """Task-independent reminder derived from the public completion protocol."""
    port = _completion_port(docs if isinstance(docs, Mapping) else {})
    if port is None:
        return "Follow the documented completion API; its answer protocol is not bound by the Harness. "
    app, name, parameter = port
    return (f"The public completion API apis.{app}.{name} accepts {parameter} only when the user "
            "requests returned information. For an operation-only task, leave the answer at its "
            "documented default None. Do not add an unsolicited identifier, receipt, confirmation "
            "sentence or execution summary. Preserve explicitly requested answers, including tasks "
            "that combine operations and questions. Information saved to a file or sent through "
            "another API is not automatically a completion answer. If the Harness cannot classify "
            "the answer requirement, its UNKNOWN finding is not permission to invent an answer. ")


def compile_proposal_contracts(instruction, api_docs=None):
    """Adapt existing parser projections and grounded legacy extraction signals."""
    docs = api_docs if isinstance(api_docs, Mapping) else {}
    contract = parse_public_contract(instruction, docs)
    source = ContractSource(0, len(instruction), hashlib.sha256(instruction.encode()).hexdigest())
    orders = []
    answer_requested = False
    for predicate, scope, expected, _ in contract.constraints:
        if scope == "$answer" and predicate in {"ANSWER_FORMAT", "ANSWER_ORDER"}:
            answer_requested = True
        if scope == "$answer" and predicate == "ANSWER_ORDER":
            valid = (isinstance(expected, (tuple, list)) and 0 < len(expected) <= 100
                     and all(type(v) is str and v and len(v) <= 1000 for v in expected)
                     and len(set(expected)) == len(expected))
            orders.append((tuple(expected), "EXACT", "explicit_labels", source) if valid else None)
    extracted = extract_output_requirements(instruction, docs)
    sources = tuple(ContractSource(a, b, source.instruction_sha256) for a, b in extracted["requested"])
    for labels, normalizer, basis, a, b in extracted["orders"]:
        orders.append((labels, normalizer, basis, ContractSource(a, b, source.instruction_sha256)))
    same_orders = orders and orders[0] is not None and all(o is not None and o[:2] == orders[0][:2] for o in orders)
    chosen = orders[0] if same_orders else None
    reason = "conflicting_or_unrepresentable_answer_order" if orders and chosen is None else None
    issues = list(extracted["issues"])
    order_issue = next((x[1] for x in issues if x[0] == "ANSWER_ORDER"), None)
    if order_issue:
        reason, chosen = order_issue, None
    requested = answer_requested or bool(extracted["requested"])
    mode = "REQUESTED" if requested else "NONE" if extracted["none"] or extracted["action_only"] else "UNKNOWN"
    basis = "explicit_information_request" if requested else "explicit_no_answer_request" if extracted["none"] else "operation_only_discourse" if extracted["action_only"] else None
    request_reason = None
    if requested and extracted["none"]:
        mode, request_reason, basis = "UNKNOWN", "conflicting_answer_requirements", None
    elif not extracted["requested"] and any(x[0] == "ANSWER_PRESENCE" for x in issues):
        # A legacy format projection is not proof that a conditional or
        # role-unbound utterance requires an unconditional completion answer.
        mode, basis = "UNKNOWN", None
        request_reason = next(x[1] for x in issues if x[0] == "ANSWER_PRESENCE")
    if mode == "UNKNOWN" and chosen is not None:
        # Do not enforce a legacy answer-order projection whose delivery or
        # obligation scope is unresolved by the speech-act adapter.
        reason, chosen = request_reason or "answer_scope_unresolved", None
        issues.append(("ANSWER_ORDER", reason, 0, len(instruction)))
    if not sources and requested:
        sources = (source,)
    if mode == "NONE":
        sources = tuple(ContractSource(a, b, source.instruction_sha256)
                        for a, b in (extracted["none"] or extracted["action_only"]))
    if mode == "UNKNOWN" and not sources:
        sources = tuple(ContractSource(a, b, source.instruction_sha256)
                        for check, _, a, b in issues if check == "ANSWER_PRESENCE")
    writes = [FieldContract(scope, tuple(sorted(set(fields))), source)
              for scope, fields in contract.preservation if fields]
    drafts, field_issues = extract_reorder_scopes(instruction, docs)
    writes.extend(FieldContract(scope, fields, ContractSource(a, b, source.instruction_sha256))
                  for scope, fields, a, b in drafts)
    issues.extend(field_issues)
    order_source = chosen[3] if chosen else next((ContractSource(a, b, source.instruction_sha256)
        for check, _, a, b in issues if check == "ANSWER_ORDER"), None)
    output = OutputContract(mode, _completion_port(docs), chosen[0] if chosen else None,
        sources, reason, chosen[1] if chosen else "EXACT", chosen[2] if chosen else None,
        order_source, request_reason, basis)
    return ProposalContracts(output, tuple(writes), tuple(issues), source.instruction_sha256)


def check_output(code, contracts, api_docs=None, namespace=None):
    docs = api_docs if isinstance(api_docs, Mapping) else {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # Structural checker owns code syntax failures.
    output = contracts.output
    findings = []
    assigned = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    conditional = _conditional_calls(tree)
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        chain = _chain(call.func)
        if chain is None or len(chain) != 3 or chain[0] != "apis":
            continue
        doc = _doc(docs, chain[1], chain[2])
        if str(doc.get("method", "")).upper() != "POST" or doc.get("path") != "/supervisor/message":
            # Recognizing the canonical interface for diagnostics does not bind
            # its answer semantics or authorize rejection without public docs.
            if chain[1:] == ["supervisor", "complete_task"]:
                findings.append(Finding("ANSWER_PRESENCE", "UNKNOWN", "public_completion_protocol_unavailable",
                    source=output.sources[0] if output.sources else None))
            continue
        source = output.sources[0] if output.sources else None
        order_source = output.order_source or source
        if call in conditional:
            findings.append(Finding("ANSWER_PRESENCE", "UNKNOWN", "conditional_call_unproven", source=source))
            if output.order is not None:
                findings.append(Finding("ANSWER_ORDER", "UNKNOWN", "conditional_call_unproven", source=order_source))
            continue
        answer = next((k.value for k in call.keywords if k.arg == "answer"), None)
        if answer is None:
            answer = call.args[0] if call.args else ast.Constant(value=None)
        known, value = _value(answer, namespace, assigned)
        if any(k.arg is None for k in call.keywords):
            known = False  # Dynamic keyword expansion can override the default answer.
        if output.answer_mode == "UNKNOWN":
            findings.append(Finding("ANSWER_PRESENCE", "UNKNOWN", output.request_unresolved_reason or "answer_request_not_compiled", source=source))
        elif output.answer_port != tuple(chain[1:]) + ("answer",):
            findings.append(Finding("ANSWER_PRESENCE", "UNKNOWN", "public_completion_protocol_unavailable", source=source))
        elif not known:
            findings.append(Finding("ANSWER_PRESENCE", "UNKNOWN", "answer_uncheckable", source=source))
        else:
            ok = value is None if output.answer_mode == "NONE" else value is not None
            findings.append(Finding("ANSWER_PRESENCE", "SATISFIED" if ok else "VIOLATED",
                "answer_presence_matches_contract" if ok else "the compiled request requires no answer" if output.answer_mode == "NONE"
                else "the compiled request requires an answer", source=source))
        if output.unresolved_reason:
            findings.append(Finding("ANSWER_ORDER", "UNKNOWN", output.unresolved_reason, source=order_source))
        elif output.order is not None and output.answer_port != tuple(chain[1:]) + ("answer",):
            findings.append(Finding("ANSWER_ORDER", "UNKNOWN", "public_completion_protocol_unavailable", source=order_source))
        elif output.order is not None:
            actual = value if known else None
            if type(actual) is str:
                try:
                    def unique_pairs(pairs):
                        result = {}
                        for key, val in pairs:
                            if key in result:
                                raise ValueError("Duplicate JSON key")
                            result[key] = val
                        return result
                    actual = json.loads(actual, object_pairs_hook=unique_pairs)
                except (ValueError, TypeError, RecursionError):
                    actual = None
            keys = tuple(actual) if type(actual) is dict else tuple(actual) if type(actual) in {list, tuple} and all(type(v) is str for v in actual) else None
            if keys is not None and output.order_normalizer == "CALENDAR_MONTH":
                normalized = tuple(month_label(label) for label in keys)
                keys = normalized if all(normalized) else None
            status = "UNKNOWN" if keys is None else "SATISFIED" if keys == output.order else "VIOLATED"
            findings.append(Finding("ANSWER_ORDER", status, "answer_uncheckable" if keys is None else
                "answer_order_matches_contract" if status == "SATISFIED" else "answer labels or order differ from the compiled request", source=order_source))
    return findings


def _patch_scope(app, doc):
    path = str(doc.get("path", ""))
    parts = path.strip("/").split("/")
    if (str(doc.get("method", "")).upper() != "PATCH" or len(parts) != 3 or parts[0] != app
            or not re.fullmatch(r"\{[a-z_]+\}", parts[2])):
        return None
    return app + "." + parts[1], parts[2][1:-1]


def _documented_update_fields(app, doc, docs, identity):
    """Exact same-object GET fields plus explicit updated-field parameter semantics."""
    fields = set()
    for other in docs.get(app, {}).values():
        if not isinstance(other, Mapping) or str(other.get("method", "")).upper() != "GET" or other.get("path") != doc.get("path"):
            continue
        schemas = other.get("response_schemas")
        schema = schemas.get("success") if isinstance(schemas, Mapping) else None
        if isinstance(schema, Mapping) and identity in schema:
            fields.update(schema)
    updated = set()
    for p in doc.get("parameters") or ():
        if not isinstance(p, Mapping) or p.get("name") not in fields:
            continue
        name = p["name"]
        description = str(p.get("description", "")).strip().lower()
        # This is a public API adapter, not a natural-language instruction parser.
        label = name.replace("_", " ")
        if re.match(r"(?:the )?updated " + re.escape(label) + r"\b", description):
            updated.add(name)
    return updated


def check_write_fields(code, contracts, api_docs=None, namespace=None):
    docs = api_docs if isinstance(api_docs, Mapping) else {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    assigned = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    conditional = _conditional_calls(tree)
    findings = []
    for check, reason, start, end in contracts.extraction_issues:
        if check == "WRITE_FIELDS":
            findings.append(Finding(check, "UNKNOWN", reason,
                source=ContractSource(start, end, contracts.instruction_sha256) if contracts.instruction_sha256 else None))
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        chain = _chain(call.func)
        if chain is None or len(chain) != 3 or chain[0] != "apis":
            continue
        doc = _doc(docs, chain[1], chain[2])
        scope = _patch_scope(chain[1], doc)
        if scope is None:
            if contracts.writes and str(doc.get("method", "")).upper() in {"PATCH", "PUT", "POST", "DELETE"}:
                findings.append(Finding("WRITE_FIELDS", "UNKNOWN", "write_scope_or_method_not_supported"))
            continue
        matches = [c for c in contracts.writes if c.scope == scope[0]]
        if not matches:
            if contracts.writes:
                findings.append(Finding("WRITE_FIELDS", "UNKNOWN", "write_scope_not_bound", scope=scope[0]))
            continue
        allowed = {c.allowed_fields for c in matches}
        source = matches[0].source
        if call in conditional:
            findings.append(Finding("WRITE_FIELDS", "UNKNOWN", "conditional_call_unproven", scope[0], source=source))
            continue
        if len(allowed) != 1:
            findings.append(Finding("WRITE_FIELDS", "UNKNOWN", "conflicting_field_contracts", scope[0], source=source))
            continue
        allowed = set(next(iter(allowed)))
        documented = _documented_update_fields(chain[1], doc, docs, scope[1])
        if not allowed <= documented:
            findings.append(Finding("WRITE_FIELDS", "UNKNOWN", "allowed_field_mapping_unavailable", scope[0], source=source))
            continue
        forbidden = []
        unknown = bool(call.args)
        params = {p.get("name"): p for p in doc.get("parameters") or () if isinstance(p, Mapping)}
        for keyword in call.keywords:
            if keyword.arg == scope[1]:
                continue
            parameter = params.get(keyword.arg, {})
            if keyword.arg == "access_token" and str(parameter.get("description", "")).startswith("Access token"):
                continue
            if keyword.arg is None:
                unknown = True
                continue
            if keyword.arg in allowed:
                continue
            if keyword.arg not in documented:
                unknown = True
                continue
            known, value = _value(keyword.value, namespace, assigned)
            if not known or value is None:
                unknown = True  # Null is not presumed to be no-op without public semantics.
            else:
                forbidden.append(keyword.arg)
        status = "VIOLATED" if forbidden else "UNKNOWN" if unknown else "SATISFIED"
        findings.append(Finding("WRITE_FIELDS", status,
            "proposal changes fields outside the compiled request" if forbidden else
            "write_parameter_effect_uncheckable" if unknown else "proposal fields match the compiled request",
            scope[0], tuple(sorted(set(forbidden))), source))
    return findings
