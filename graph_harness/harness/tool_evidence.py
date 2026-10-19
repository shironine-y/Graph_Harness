from __future__ import annotations

import copy
import hashlib
import secrets
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ToolReturn:
    operation: str
    value: object
    arguments: Mapping
    document: Mapping
    method: str
    subject_key: str = ""
    attempt_index: int = -1


class RuntimeApiBudgetError(RuntimeError):
    pass


class ToolEvidenceCapture:
    def __init__(self, world, enabled=True, max_calls=None, context_signature="state_unknown"):
        self.world = world
        self.enabled = enabled
        self.records: list[ToolReturn] = []
        self.available = False
        self.reason = "disabled" if not enabled else "local_requester_unavailable"
        self._requester = None
        self._original = None
        self._wrapper = None
        self._had_own_method = False
        self.max_calls = max_calls
        self.call_count = 0
        self.context_signature = context_signature
        self.operation_contexts = []
        self.attempts = []
        self.on_attempt = None
        self.on_dispatch = None

    def __enter__(self):
        if not self.enabled and self.max_calls is None:
            return self
        requester = getattr(self.world, "requester", None)
        original = getattr(requester, "request", None)
        if not callable(original) or not hasattr(requester, "__dict__"):
            return self
        self._requester, self._original = requester, original
        self._had_own_method = "request" in requester.__dict__
        salt = getattr(self.world, "_harness_subject_salt", None)
        if salt is None:
            salt = secrets.token_bytes(16)
            self.world._harness_subject_salt = salt

        def capture(*args, **kwargs):
            app = kwargs.get("_app_name", args[0] if args else None)
            api = kwargs.get("_api_name", args[1] if len(args) > 1 else None)
            if self.max_calls is not None and self.call_count >= self.max_calls:
                raise RuntimeApiBudgetError("Runtime API call budget exceeded; split the batch")
            self.call_count += 1
            docs = getattr(getattr(self.world, "task", None), "api_docs", {})
            app_docs = docs.get(app, {}) if isinstance(docs, Mapping) else {}
            doc = app_docs.get(api, {}) if isinstance(app_docs, Mapping) else {}
            if (isinstance(app, str) and isinstance(api, str) and isinstance(doc, Mapping)
                    and str(doc.get("method", "")).upper() in {"GET", "POST", "PUT", "PATCH", "DELETE"}):
                from harness.failure_ledger import runtime_operation_context
                self.operation_contexts.append(runtime_operation_context(
                    app, api, doc, kwargs, self.context_signature))
            if not isinstance(doc, Mapping):
                doc = {}
            safe_arguments = {k: v for k, v in kwargs.items() if not k.startswith("_") and k not in {
                "access_token", "token", "password", "client", "raise_on_failure", "show", "track"}}
            from harness.appworld import _classify
            attempt = {"category": _classify(str(doc.get("method", "")).upper(), str(doc.get("path", ""))), "operation": f"{app}.{api}", "method": str(doc.get("method", "")).upper(),
                       "arguments": safe_arguments, "dispatched": False, "returned": False}
            self.attempts.append(attempt)
            if self.on_attempt is not None:
                self.on_attempt(attempt)
            if self.on_dispatch is not None:
                self.on_dispatch(attempt)
            attempt["dispatched"] = True
            value = original(*args, **kwargs)  # Preserve errors and return semantics.
            attempt["returned"] = True
            if not self.enabled:
                return value
            if not isinstance(app, str) or not isinstance(api, str):
                self.reason = "ambiguous_operation"
                return value
            if not isinstance(doc, Mapping) or str(doc.get("method", "")).upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                self.reason = "public_document_unavailable"
                return value
            # Credentials and framework options are never retained as target data.
            arguments = {k: v for k, v in kwargs.items() if not k.startswith("_") and k not in {
                "access_token", "token", "password", "client", "raise_on_failure", "show", "track"}}
            subject = kwargs.get("access_token", kwargs.get("token"))
            token_key = hashlib.sha256(salt + repr(subject).encode()).hexdigest() if subject is not None else ""
            token_subjects = getattr(self.world, "_harness_token_subjects", None)
            if token_subjects is None:
                token_subjects = self.world._harness_token_subjects = {}
            # A successful login can prove that refreshed tokens identify the same
            # account. Keep only salted identifiers in this task-local mapping.
            owner = next((kwargs[k] for k in ("email", "username", "phone_number") if k in kwargs), None)
            if api == "login" and owner is not None and isinstance(value, Mapping):
                refreshed = value.get("access_token")
                if type(refreshed) is str:
                    refreshed_key = hashlib.sha256(salt + repr(refreshed).encode()).hexdigest()
                    token_subjects[refreshed_key] = hashlib.sha256(salt + repr((app, owner)).encode()).hexdigest()
            subject_key = token_subjects.get(token_key, token_key)
            try:
                self.records.append(ToolReturn(f"{app}.{api}", copy.deepcopy(value),
                    copy.deepcopy(arguments), doc, str(doc["method"]).upper(), subject_key,
                    len(self.attempts) - 1))
            except Exception:
                self.reason = "response_not_copyable"
            return value

        self._wrapper = capture
        requester.request = capture
        self.available, self.reason = self.enabled, "request_boundary" if self.enabled else "disabled"
        return self

    def __exit__(self, *exc):
        if self._requester is not None:
            if self._requester.request is self._wrapper:
                if self._had_own_method:
                    self._requester.request = self._original
                else:
                    del self._requester.__dict__["request"]
            else:
                # A replaced boundary is not trustworthy even if execution succeeds.
                self.records.clear()
                self.operation_contexts.clear()
                self.attempts.clear()
                self.available, self.reason = False, "request_boundary_replaced"
        return False


def read_is_complete(record: ToolReturn) -> bool:
    """Defaults matter: an unrequested page limit still makes a read partial."""
    parameters = record.document.get("parameters")
    if not isinstance(parameters, (list, tuple)):
        return False
    if any(isinstance(p, Mapping) and any(t in str(p.get("name", "")).lower()
            for t in ("page", "offset", "limit", "cursor")) for p in parameters):
        return False
    import re
    return not re.search(r"\b(?:paginated|pagination|first\s+\d+|up to\s+\d+)\b",
                         str(record.document.get("description", "")), re.I)
