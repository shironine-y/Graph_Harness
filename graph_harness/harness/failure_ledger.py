from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import secrets
from typing import Any, Iterable


FAILURE_CLASSES = {
    "AUTHENTICATION_FAILURE",
    "AUTHORIZATION_FAILURE",
    "INVALID_INPUT",
    "PRECONDITION_UNSATISFIED",
    "RESOURCE_UNAVAILABLE",
    "CAPACITY_OR_QUOTA",
    "NOT_FOUND",
    "CONFLICT",
    "TRANSIENT_FAILURE",
    "EXECUTION_EXCEPTION",
    "POSTCONDITION_MISMATCH",
    "HARNESS_VERIFICATION_FAILURE",
    "UNKNOWN_FAILURE",
}

RETRY_SEMANTICS = {
    "NEVER_SAME_TASK",
    "AFTER_PARAMETER_CHANGE",
    "AFTER_CONTEXT_CHANGE",
    "AFTER_AUTH_REFRESH",
    "AFTER_STATE_CHANGE",
    "TRANSIENT_RETRY",
    "MANUAL_REPLAN",
    "UNKNOWN",
}

_SECRET_FIELDS = {
    "access_token", "token", "password", "secret", "api_key", "cvv",
    "cvv_number", "card_number",
}


def _attribute_chain(node: ast.AST) -> list[str] | None:
    values: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        values.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    values.append(current.id)
    return list(reversed(values))


def _digest(value: Any, *, length: int = 20) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def _safe_expression(node: ast.AST) -> str:
    """Hash an expression immediately so literals never enter evidence or feedback."""
    return _digest(ast.dump(node, include_attributes=False))


def runtime_operation_context(app, api, document, arguments, context_signature):
    """Fingerprint a witnessed invocation, not an unexecuted branch or loop site."""
    from harness.appworld import _classify
    method = str(document.get("method", "UNKNOWN")).upper()
    route = str(document.get("path", "/unknown"))
    parts = [(str(k).lower(), _digest(v)) for k, v in sorted(arguments.items())
             if not str(k).startswith("_") and k not in {"client", "raise_on_failure", "show", "track"}]
    subjects = [(k, v) for k, v in parts if k not in _SECRET_FIELDS and not k.endswith("token")]
    parameters = _digest(parts)
    return OperationContext(
        operation_signature=_digest([app, api, method, route, _classify(method, route)]),
        operation_display=f"{app}.{api}", parameter_signature=parameters,
        subject_fingerprint=_digest(subjects or [("strategy", parameters)]),
        context_signature=context_signature,
        runtime_witnessed=True,
    )


@dataclass(frozen=True)
class OperationContext:
    operation_signature: str
    operation_display: str
    subject_fingerprint: str
    parameter_signature: str
    context_signature: str
    runtime_witnessed: bool = False


@dataclass
class FailureEvidence:
    evidence_id: str
    task_id: str
    step: int
    decision_id: str | None
    action_id: str | None
    operation_signature: str
    operation_display: str
    subject_handles: tuple[str, ...]
    parameter_signature: str
    context_signature: str
    failure_class: str
    reason_code: str
    retry_semantics: str
    state_changed: bool | None
    rollback_required: bool
    rollback_level: str | None
    rollback_verified: bool | None
    first_seen_step: int
    last_seen_step: int
    occurrence_count: int = 1
    resolved: bool = False
    superseded: bool = False

    def audit_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActiveFailureConstraint:
    evidence_id: str
    operation_signature: str
    operation_display: str
    subject_handles: tuple[str, ...]
    failure_class: str
    retry_semantics: str
    attempts: int
    required_change: str


def classify_failure(reason: str, cause: str) -> tuple[str, str, str]:
    """Return a generic class, reason code and retry semantics.

    The HTTP status and bounded structural cues jointly determine the result.  No
    application or task identifiers participate in classification.
    """
    text = " ".join(reason.lower().split())
    status_match = re.search(r"\bhttp\s+(\d{3})\b", text)
    status = int(status_match.group(1)) if status_match else None

    if cause in {"verification_mismatch", "harness_verification"}:
        return "HARNESS_VERIFICATION_FAILURE", "RUNTIME_VERIFICATION", "MANUAL_REPLAN"
    if cause in {"postcondition", "probability_rejection"}:
        return "POSTCONDITION_MISMATCH", "POSTCONDITION_REJECTED", "MANUAL_REPLAN"
    if status == 401:
        return "AUTHENTICATION_FAILURE", "HTTP_401", "AFTER_AUTH_REFRESH"
    if status == 403:
        return "AUTHORIZATION_FAILURE", "HTTP_403", "MANUAL_REPLAN"
    if status == 404:
        return "NOT_FOUND", "HTTP_404", "AFTER_CONTEXT_CHANGE"
    if status == 409:
        return "CONFLICT", "HTTP_409", "AFTER_STATE_CHANGE"
    if status == 429:
        return "CAPACITY_OR_QUOTA", "HTTP_429", "TRANSIENT_RETRY"
    if status in {408, 425} or (status is not None and status >= 500):
        return "TRANSIENT_FAILURE", f"HTTP_{status}", "TRANSIENT_RETRY"
    if status in {400, 422}:
        if any(word in text for word in ("expired", "revoked", "inactive")):
            return "PRECONDITION_UNSATISFIED", f"HTTP_{status}_EXPIRED", "NEVER_SAME_TASK"
        if any(word in text for word in (
            "insufficient", "does not have", "not have", "quota", "capacity",
        )):
            return (
                "PRECONDITION_UNSATISFIED",
                f"HTTP_{status}_PRECONDITION",
                "AFTER_STATE_CHANGE",
            )
        return "INVALID_INPUT", f"HTTP_{status}", "AFTER_PARAMETER_CHANGE"
    if any(word in text for word in ("timeout", "timed out", "temporarily unavailable")):
        return "TRANSIENT_FAILURE", "TRANSIENT_RUNTIME", "TRANSIENT_RETRY"
    if any(word in text for word in ("unavailable", "not available")):
        return "RESOURCE_UNAVAILABLE", "RESOURCE_UNAVAILABLE", "AFTER_CONTEXT_CHANGE"
    if cause == "execution_error":
        return "EXECUTION_EXCEPTION", "EXECUTION_EXCEPTION", "UNKNOWN"
    return "UNKNOWN_FAILURE", "UNKNOWN", "UNKNOWN"


def operation_contexts(
    code: str,
    actions: Iterable[Any],
    *,
    context_signature: str,
) -> list[OperationContext]:
    """Build value-free operation and argument identities from a proposal."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if chain and len(chain) == 3 and chain[0] == "apis":
                calls.append(node)
    calls.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
    action_list = list(actions)
    contexts: list[OperationContext] = []
    for index, action in enumerate(action_list):
        call = calls[min(index, len(calls) - 1)] if calls else None
        chain = _attribute_chain(call.func) if call is not None else None
        app = chain[1] if chain and len(chain) == 3 else "unknown_app"
        api = chain[2] if chain and len(chain) == 3 else "unknown_api"
        method = str(getattr(action, "method", "UNKNOWN")).upper()
        route = str(getattr(action, "route", "/unknown"))
        category = str(getattr(action, "category", "UNKNOWN"))
        operation_display = f"{app}.{api}"
        operation_signature = _digest([app, api, method, route, category])

        parameter_parts: list[tuple[str, str]] = []
        subject_parts: list[tuple[str, str]] = []
        if call is not None:
            for position, value in enumerate(call.args):
                hashed = _safe_expression(value)
                parameter_parts.append((f"arg_{position}", hashed))
                subject_parts.append((f"arg_{position}", hashed))
            for keyword in call.keywords:
                name = (keyword.arg or "kwargs").lower()
                hashed = _safe_expression(keyword.value)
                parameter_parts.append((name, hashed))
                if name not in _SECRET_FIELDS and not name.endswith("token"):
                    subject_parts.append((name, hashed))
        parameter_signature = _digest(parameter_parts)
        subject_fingerprint = _digest(subject_parts or [("strategy", parameter_signature)])
        contexts.append(OperationContext(
            operation_signature=operation_signature,
            operation_display=operation_display,
            subject_fingerprint=subject_fingerprint,
            parameter_signature=parameter_signature,
            context_signature=context_signature,
        ))
    return contexts


class TaskFailureLedger:
    """Task-scoped evidence and active constraints; never shared across tasks."""

    MAX_ACTIVE_FEEDBACK = 4
    MAX_FEEDBACK_CHARS = 2400

    def __init__(self, task_id: str, *, nonce: bytes | None = None):
        self.task_id = task_id
        self._nonce = nonce or secrets.token_bytes(16)
        self._handles: dict[str, str] = {}
        self._evidence: list[FailureEvidence] = []
        self._last_update: FailureEvidence | None = None

    @property
    def evidence(self) -> tuple[FailureEvidence, ...]:
        return tuple(self._evidence)

    @property
    def active_constraints(self) -> tuple[ActiveFailureConstraint, ...]:
        return tuple(self._constraint(item) for item in self._evidence
                     if not item.resolved and not item.superseded)

    def _handle(self, fingerprint: str, prefix: str = "entity") -> str:
        private_key = hashlib.sha256(self._nonce + fingerprint.encode()).hexdigest()
        if private_key not in self._handles:
            self._handles[private_key] = f"{prefix}_{len(self._handles) + 1}"
        return self._handles[private_key]

    @staticmethod
    def _required_change(retry_semantics: str) -> str:
        return {
            "NEVER_SAME_TASK": "choose another entity or strategy",
            "AFTER_PARAMETER_CHANGE": "revise arguments before retrying",
            "AFTER_CONTEXT_CHANGE": "refresh context or choose another entity",
            "AFTER_AUTH_REFRESH": "refresh authentication before retrying",
            "AFTER_STATE_CHANGE": "change the relevant state or choose another entity",
            "TRANSIENT_RETRY": "retry later with a bounded attempt",
            "MANUAL_REPLAN": "inspect diagnostics and replan",
            "UNKNOWN": "change the strategy before retrying",
        }[retry_semantics]

    def _constraint(self, evidence: FailureEvidence) -> ActiveFailureConstraint:
        return ActiveFailureConstraint(
            evidence_id=evidence.evidence_id,
            operation_signature=evidence.operation_signature,
            operation_display=evidence.operation_display,
            subject_handles=evidence.subject_handles,
            failure_class=evidence.failure_class,
            retry_semantics=evidence.retry_semantics,
            attempts=evidence.occurrence_count,
            required_change=self._required_change(evidence.retry_semantics),
        )

    def record_failure(
        self,
        *,
        step: int,
        decision_id: str | None,
        action_id: str | None,
        operation: OperationContext | None,
        reason: str,
        cause: str,
        api_executed: bool,
        state_changed: bool | None,
        rollback_required: bool,
        rollback_level: str | None,
        rollback_verified: bool | None,
    ) -> FailureEvidence | None:
        failure_class, reason_code, retry_semantics = classify_failure(reason, cause)
        if (
            not api_executed
            or operation is None
            or failure_class in {"HARNESS_VERIFICATION_FAILURE", "POSTCONDITION_MISMATCH"}
        ):
            return None
        handle = self._handle(operation.subject_fingerprint)
        for item in self._evidence:
            if (
                not item.resolved and not item.superseded
                and item.operation_signature == operation.operation_signature
                and item.subject_handles == (handle,)
                and item.parameter_signature == operation.parameter_signature
                and item.context_signature == operation.context_signature
                and item.failure_class == failure_class
                and item.retry_semantics == retry_semantics
            ):
                item.last_seen_step = step
                item.occurrence_count += 1
                item.state_changed = state_changed
                item.rollback_required = rollback_required
                item.rollback_level = rollback_level
                item.rollback_verified = rollback_verified
                self._last_update = item
                return item
        evidence = FailureEvidence(
            evidence_id=f"failure_{len(self._evidence) + 1}",
            task_id=self.task_id,
            step=step,
            decision_id=decision_id,
            action_id=action_id,
            operation_signature=operation.operation_signature,
            operation_display=operation.operation_display,
            subject_handles=(handle,),
            parameter_signature=operation.parameter_signature,
            context_signature=operation.context_signature,
            failure_class=failure_class,
            reason_code=reason_code,
            retry_semantics=retry_semantics,
            state_changed=state_changed,
            rollback_required=rollback_required,
            rollback_level=rollback_level,
            rollback_verified=rollback_verified,
            first_seen_step=step,
            last_seen_step=step,
        )
        self._evidence.append(evidence)
        self._last_update = evidence
        return evidence

    def record_success(self, operation: OperationContext | None, *, auth_refresh: bool = False) -> None:
        if auth_refresh:
            for item in self._evidence:
                if item.failure_class == "AUTHENTICATION_FAILURE" and not item.resolved:
                    item.superseded = True
        if operation is None:
            return
        handle = self._handle(operation.subject_fingerprint)
        for item in self._evidence:
            if item.resolved or item.superseded:
                continue
            if (
                item.operation_signature == operation.operation_signature
                and item.subject_handles == (handle,)
                and item.parameter_signature == operation.parameter_signature
                and item.context_signature == operation.context_signature
            ):
                item.resolved = True

    def equivalent_retry_rejection(self, operations: Iterable[OperationContext]) -> str | None:
        for operation in operations:
            handle = self._handle(operation.subject_fingerprint)
            for item in self._evidence:
                if item.resolved or item.superseded:
                    continue
                if (
                    item.operation_signature != operation.operation_signature
                    or item.subject_handles != (handle,)
                    or item.parameter_signature != operation.parameter_signature
                    or item.context_signature != operation.context_signature
                ):
                    continue
                if item.retry_semantics in {
                    "NEVER_SAME_TASK", "AFTER_PARAMETER_CHANGE", "AFTER_CONTEXT_CHANGE",
                    "AFTER_AUTH_REFRESH", "AFTER_STATE_CHANGE",
                }:
                    return (
                        "An equivalent deterministic failure is still active for "
                        f"{operation.operation_display} on {handle}; "
                        f"{self._required_change(item.retry_semantics)}."
                    )
        return None

    def format_feedback(self, *, relevant_operations: Iterable[OperationContext] = ()) -> str:
        update = self._last_update
        parts: list[str] = []
        if update is not None:
            parts.extend([
                "[HARNESS_FAILURE_UPDATE]",
                f"evidence_id: {update.evidence_id}",
                f"operation: {update.operation_display}",
                f"subject_handle: {update.subject_handles[0]}",
                f"failure_class: {update.failure_class}",
                f"reason_code: {update.reason_code}",
                f"retry_semantics: {update.retry_semantics}",
                "state_changed: " + (
                    "unknown" if update.state_changed is None
                    else str(update.state_changed).lower()
                ),
                "rollback: " + (
                    update.rollback_level.upper() if update.rollback_required
                    and update.rollback_level else "NOT_REQUIRED"
                ),
                "",
            ])
        relevant = {operation.operation_signature for operation in relevant_operations}
        active = list(self.active_constraints)
        active.sort(key=lambda item: (item.operation_signature not in relevant, -item.attempts))
        shown = active[:self.MAX_ACTIVE_FEEDBACK]
        parts.append("[HARNESS_ACTIVE_CONSTRAINTS]")
        if not shown:
            parts.append("- none")
        for item in shown:
            parts.extend([
                f"- handle: {item.subject_handles[0]}",
                f"  operation: {item.operation_display}",
                f"  failure_class: {item.failure_class}",
                f"  attempts: {item.attempts}",
                f"  retry_semantics: {item.retry_semantics}",
                f"  required_change: {item.required_change}",
            ])
        omitted = len(active) - len(shown)
        if omitted:
            parts.append(f"- omitted_active_constraints: {omitted}")
        return "\n".join(parts)[:self.MAX_FEEDBACK_CHARS]
