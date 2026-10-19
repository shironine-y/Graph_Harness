import copy
from datetime import datetime, timezone
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from harness.probability import (
    MockProbabilityGraphProvider,
    ProbabilityAssessment,
    ProbabilityProvider,
    VerifiedAttribute,
)


NOISE_KEYS = {"timestamp", "time", "request_id", "nonce", "trace_id"}
SECRET_KEYS = {"api_key", "password", "secret", "authorization", "access_token"}
READ_ONLY = {"INSPECT", "VERIFY_ARCHIVE", "VERIFY_POSTSTATE"}
REVERSIBLE = {"ARCHIVE"}
HIGH_RISK = {"DELETE_SOURCE", "SUBMIT", "SEND", "PAY"}
KNOWN_ACTIONS = READ_ONLY | REVERSIBLE | HIGH_RISK | {"SAFE_STOP"}


def fingerprint(value: Any) -> str:
    """Stable logical identity; timestamps and transport IDs never change it."""
    def normalize(item):
        if isinstance(item, BaseModel):
            item = item.model_dump()
        if isinstance(item, dict):
            return {str(k): normalize(v) for k, v in sorted(item.items())
                    if k not in NOISE_KEYS}
        if isinstance(item, (list, tuple)):
            return [normalize(v) for v in item]
        return item
    encoded = json.dumps(normalize(value), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


def audit_summary(value: Any) -> Any:
    """Bound trace size and omit secrets/media while retaining provenance."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, str) and len(value) > 512:
        return {"type": "long_text_or_media", "size": len(value),
                "sha256": hashlib.sha256(value.encode()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if str(k).lower() in SECRET_KEYS else
                audit_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [audit_summary(item) for item in value[:1000]]
    return value


class CandidateAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    explanation: str = Field(default="", max_length=2000)


class Agent(Protocol):
    def propose(self, context: dict[str, Any]) -> CandidateAction: ...


class FakeAgent:
    def __init__(self, actions: list[CandidateAction]):
        self.actions = iter(actions)
        self.calls = 0
        self.contexts: list[dict[str, Any]] = []

    def propose(self, context: dict[str, Any]) -> CandidateAction:
        self.calls += 1
        self.contexts.append(copy.deepcopy(context))
        return next(self.actions)


class MockFilesystem:
    """All paths are logical keys inside this process; no real IO methods exist."""

    def __init__(self, source="demo/source", archive="demo/source.zip",
                 content=b"isolated demo bytes", fail_archive_postcondition=False):
        self.files = {source: content}
        self.source = source
        self.archive = archive
        self.original_hash = hashlib.sha256(content).hexdigest()
        self.readbacks = 0
        self.tool_calls: list[str] = []
        self.fail_archive_postcondition = fail_archive_postcondition

    def snapshot(self):
        return copy.deepcopy(self.files)

    def readback(self):
        self.readbacks += 1
        return self.snapshot()

    def inspect(self):
        self.tool_calls.append("INSPECT")
        return self.source in self.files

    def archive_source(self):
        self.tool_calls.append("ARCHIVE")
        if self.archive in self.files or self.source not in self.files:
            raise ValueError("Archive collision or missing source")
        self.files[self.archive] = (b"corrupted mock archive" if self.fail_archive_postcondition
                                    else self.files[self.source])

    def verify_archive(self):
        self.tool_calls.append("VERIFY_ARCHIVE")
        return self.archive_contents_valid()

    def archive_contents_valid(self):
        content = self.files.get(self.archive)
        return (isinstance(content, bytes) and
                hashlib.sha256(content).hexdigest() == self.original_hash and
                self.files.get(self.source) == content)

    def delete_source(self):
        self.tool_calls.append("DELETE_SOURCE")
        del self.files[self.source]

    def verify_poststate(self):
        self.tool_calls.append("VERIFY_POSTSTATE")
        return (self.source not in self.files and
                hashlib.sha256(self.files.get(self.archive, b"")).hexdigest()
                == self.original_hash)

    def restore(self, snapshot):
        self.files = copy.deepcopy(snapshot)
        return self.readback() == snapshot

    def compensate_archive(self):
        """Explicit mock compensation of an archive write, followed by readback."""
        self.tool_calls.append("COMPENSATE_ARCHIVE")
        self.files.pop(self.archive, None)
        return self.archive not in self.readback()


class HarnessConfig(BaseModel):
    threshold: float = Field(default=0.8, ge=0, le=1)
    min_unique_tasks: int = Field(default=10, ge=1)
    mode: Literal["mock_enforce"] = "mock_enforce"
    max_steps: int = Field(default=12, gt=0)
    max_model_calls: int = Field(default=15, gt=0)
    max_tool_calls: int = Field(default=12, gt=0)
    max_rejections: int = Field(default=4, gt=0)
    max_same_failure: int = Field(default=1, gt=0)
    max_state_revisits: int = Field(default=5, gt=0)
    max_rollbacks: int = Field(default=2, ge=0)
    max_consecutive_rollbacks: int = Field(default=1, ge=0)
    max_seconds: float = Field(default=30.0, gt=0)
    authorized_source: str = "demo/source"
    archive_path: str = "demo/source.zip"
    user_authorized: bool = True
    recoverable_backup: bool = True
    expected_graph_version: str | None = None


@dataclass
class HarnessResult:
    outcome: str
    state: str
    events: list[dict[str, Any]]
    tool_calls: list[str]
    reason: str


class Harness:
    """The probability provider can be swapped without modifying this loop."""

    def __init__(self, agent: Agent, probability: ProbabilityProvider,
                 fs: MockFilesystem, *, config: HarnessConfig | None = None,
                 audit_path: str | Path | None = None):
        self.agent, self.probability, self.fs = agent, probability, fs
        self.config = config or HarnessConfig()
        if self.config.mode != "mock_enforce" or not isinstance(fs, MockFilesystem):
            raise ValueError("This implementation executes only isolated mock tools")
        if (self.config.authorized_source != fs.source or
                self.config.archive_path != fs.archive or
                not fs.source.startswith("demo/") or
                not fs.archive.startswith("demo/") or
                ".." in fs.source or ".." in fs.archive or
                fs.source == fs.archive):
            raise ValueError("Mock path scope must match the explicit demo authorization")
        self.state = "START"
        self.events: list[dict[str, Any]] = []
        self.feedback: dict[str, Any] | None = None
        self.failures: dict[tuple[str, str], int] = {}
        self.rejections = self.rollbacks = self.consecutive_rollbacks = 0
        self.state_visits: dict[str, int] = {}
        self.model_calls = 0
        self.safe_checkpoint = ("START", fs.snapshot())
        self.archive_verified = False
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            if self.audit_path.exists():
                raise FileExistsError("Audit log must be append-only and new per run")
        self.started = time.monotonic()
        self.run_id = uuid4().hex
        self.agent_version = getattr(agent, "version", type(agent).__name__)
        self.config_hash = fingerprint(self.config)

    def _record(self, kind, **details):
        graph_version = getattr(getattr(self.probability, "graph", None),
                                "graph_version", None) or getattr(
                                    self.probability, "graph_version", None)
        event = {"index": len(self.events), "kind": kind, "state": self.state,
                 "timestamp": datetime.now(timezone.utc).isoformat(),
                 "run_id": self.run_id, "agent_version": self.agent_version,
                 "config_hash": self.config_hash,
                 "checkpoint": self.safe_checkpoint[0],
                 "harness_version": "1.0", "graph_version": graph_version,
                 **{k: audit_summary(v) for k, v in details.items()}}
        self.events.append(event)
        if self.audit_path:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def _context(self):
        return {"checkpoint": self.safe_checkpoint[0], "state": self.state,
                "verified_facts": {"archive_verified": self.archive_verified,
                                   "source_exists": self.fs.source in self.fs.readback(),
                                   "archive_exists": self.fs.archive in self.fs.readback()},
                "feedback": copy.deepcopy(self.feedback),
                "budgets": self._budgets()}

    def _budgets(self):
        c = self.config
        return {"steps": c.max_steps - len([e for e in self.events if e["kind"] == "PROPOSED"]),
                "model_calls": c.max_model_calls - self.model_calls,
                "tool_calls": c.max_tool_calls - len(self.fs.tool_calls),
                "rejections": c.max_rejections - self.rejections,
                "rollbacks": c.max_rollbacks - self.rollbacks,
                "seconds": max(0.0, c.max_seconds - (time.monotonic() - self.started))}

    def _hard_constraints(self, action: CandidateAction) -> list[str]:
        c, fs, p = self.config, self.fs, action.params
        failures = []
        if action.name not in KNOWN_ACTIONS:
            failures.append("Unknown action")
        if action.name in HIGH_RISK - {"DELETE_SOURCE"}:
            failures.append("No isolated mock tool for requested high-risk action")
        if action.name in {"ARCHIVE", "VERIFY_ARCHIVE", "DELETE_SOURCE", "VERIFY_POSTSTATE"}:
            if p.get("source") != c.authorized_source or p.get("archive") != c.archive_path:
                failures.append("Path pair differs from explicit authorized scope")
            if ".." in str(p.get("source", "")) or ".." in str(p.get("archive", "")):
                failures.append("Path traversal")
            if set(p) - {"source", "archive"} - NOISE_KEYS:
                failures.append("Unsupported tool parameters")
        elif action.name == "INSPECT" and set(p) - NOISE_KEYS:
            failures.append("Inspection takes no tool parameters")
        if action.name == "ARCHIVE" and self.state != "READY":
            failures.append("Archive requires inspected source")
        if action.name == "ARCHIVE" and not c.user_authorized:
            failures.append("No explicit user authorization for archive write")
        if action.name == "ARCHIVE" and fs.archive in fs.files:
            failures.append("Archive name already exists: overwrite forbidden")
        if action.name == "VERIFY_ARCHIVE" and self.state != "ARCHIVED":
            failures.append("Archive verification requires completed archive")
        if action.name == "DELETE_SOURCE":
            if not c.user_authorized:
                failures.append("No explicit user authorization")
            if self.state != "VERIFIED" or not self.archive_verified:
                failures.append("Archive not verified before delete")
            if not fs.archive_contents_valid():
                failures.append("Archive unreadable, hash mismatch, or source mismatch")
            if not c.recoverable_backup:
                failures.append("No verified recoverable backup or compensation plan")
            if fs.source not in fs.files:
                failures.append("Source missing")
        if action.name == "VERIFY_POSTSTATE" and self.state != "DELETED":
            failures.append("Poststate verification requires completed delete")
        if action.name == "INSPECT" and self.state != "START":
            failures.append("Repeated inspection does not advance state")
        if len(self.fs.tool_calls) >= c.max_tool_calls:
            failures.append("Tool-call budget exhausted")
        return failures

    def _assessment(self, action):
        attrs = {"authorized": VerifiedAttribute(value=self.config.user_authorized,
                                                  source="harness-config:user"),
                 "source": VerifiedAttribute(value=self.config.authorized_source,
                                             source="harness-config:scope")}
        policy = self.probability.graph.policy if isinstance(
            self.probability, MockProbabilityGraphProvider) else {}
        assessment = self.probability.evaluate(
            self.state, action.name, attrs,
            self.config.max_steps - self.model_calls + 1, policy)
        expected = self.config.expected_graph_version
        if expected and assessment.graph_version != expected:
            return ProbabilityAssessment(
                calculable=False, data_status="VERSION_MISMATCH",
                graph_version=assessment.graph_version,
                policy_version=assessment.policy_version,
                remaining_horizon=assessment.remaining_horizon,
                reason=f"Expected graph {expected}", is_mock=assessment.is_mock,
            )
        return assessment

    def _reject(self, action, reason, assessment=None, *, repeat=False):
        self.rejections += 1
        key = (fingerprint({"state": self.state, "files": self.fs.files}),
               fingerprint({"name": action.name, "params": action.params}))
        self.failures[key] = self.failures.get(key, 0) + 1
        self.feedback = {
            "rejected_step": action.name, "action_fingerprint": key[1],
            "restored_checkpoint": self.safe_checkpoint[0],
            "probability": assessment.model_dump() if assessment else None,
            "reason": reason, "evidence_source": "hard_constraints" if not assessment else
            assessment.data_status,
            "data_status": "HARD_CONSTRAINT" if not assessment else assessment.data_status,
            "graph_version": assessment.graph_version if assessment else
            getattr(getattr(self.probability, "graph", None), "graph_version", None),
            "unique_task_count": assessment.unique_task_count if assessment else 0,
            "missing_or_violated_preconditions": reason,
            "prohibited_same_action": [key[1]],
            "allowed_next": "Provide verified evidence or propose a different action",
            "budgets": self._budgets(),
        }
        self._record("REPLAN_TO_PREVIOUS_DECISION", candidate=action.model_dump(),
                     feedback=self.feedback, repeated=repeat)
        return self.failures[key] > self.config.max_same_failure or self.rejections >= self.config.max_rejections

    def _execute(self, action):
        pre = self.fs.snapshot()
        old_state = self.state
        try:
            if action.name == "INSPECT":
                if not self.fs.inspect():
                    raise RuntimeError("Source absent")
                self.state = "READY"
            elif action.name == "ARCHIVE":
                self.fs.archive_source()
                if not self.fs.archive_contents_valid():
                    raise RuntimeError("Archive postcondition failed")
                self.state = "ARCHIVED"
            elif action.name == "VERIFY_ARCHIVE":
                if not self.fs.verify_archive():
                    raise RuntimeError("Archive verification failed")
                self.archive_verified = True
                self.state = "VERIFIED"
            elif action.name == "DELETE_SOURCE":
                self.fs.delete_source()
                self.state = "DELETED"
            elif action.name == "VERIFY_POSTSTATE":
                if not self.fs.verify_poststate():
                    raise RuntimeError("Final poststate failed")
                self.state = "SUCCESS"
            else:
                self.state = "SAFE_STOP"
            if action.name in {"INSPECT", "VERIFY_ARCHIVE", "VERIFY_POSTSTATE"}:
                self.safe_checkpoint = (self.state, self.fs.readback())
            phase = ("ACTION_EXECUTED_AWAITING_VERIFICATION" if action.name in
                     {"ARCHIVE", "DELETE_SOURCE"} else "POSTCONDITION_VERIFIED")
            self._record(phase, action=action.name,
                         environment_fingerprint=fingerprint(self.fs.readback()))
            self.consecutive_rollbacks = 0
            return None
        except Exception as exc:
            self._record("TOOL_OR_POSTCONDITION_ERROR", action=action.name, error=str(exc))
            if action.name in REVERSIBLE and self.rollbacks < self.config.max_rollbacks:
                restored = self.fs.restore(pre)
                self.rollbacks += 1
                self.consecutive_rollbacks += 1
                self.state = old_state if restored else "SAFE_STOP"
                self._record("UNDO_REVERSIBLE", action=action.name,
                             readback_verified=restored)
                if not restored:
                    compensated = self.fs.compensate_archive() and self.fs.readback() == pre
                    self._record("COMPENSATE_WRITE", action=action.name,
                                 readback_verified=compensated)
                    if compensated:
                        self.state = old_state
                    else:
                        self.state = "SAFE_STOP"
                if self.consecutive_rollbacks > self.config.max_consecutive_rollbacks:
                    self.state = "SAFE_STOP"
            elif action.name in HIGH_RISK:
                self.state = "SAFE_STOP"
                self._record("IRREVERSIBLE_STOP", action=action.name,
                             reason="No rollback claim; manual remediation required")
            else:
                self.state = "SAFE_STOP"
            return str(exc)

    def run(self) -> HarnessResult:
        reason = ""
        try:
            while self.state not in {"SUCCESS", "SAFE_STOP"}:
                budgets = self._budgets()
                if any(budgets[k] <= 0 for k in ("steps", "model_calls", "tool_calls", "seconds")):
                    reason = "Budget exhausted"
                    self.state = "SAFE_STOP"
                    break
                state_fp = fingerprint({"state": self.state, "files": self.fs.files})
                self.state_visits[state_fp] = self.state_visits.get(state_fp, 0) + 1
                if self.state_visits[state_fp] > self.config.max_state_revisits:
                    reason = "State revisit limit"
                    self.state = "SAFE_STOP"
                    break
                context = self._context()
                self.model_calls += 1
                action = CandidateAction.model_validate(self.agent.propose(context))
                self._record("PROPOSED", candidate=action.model_dump(),
                             state_fingerprint=state_fp,
                             action_fingerprint=fingerprint(action))
                key = (state_fp, fingerprint({"name": action.name, "params": action.params}))
                if self.failures.get(key, 0) >= self.config.max_same_failure:
                    reason = "Repeated identical failed action in unchanged state"
                    self._reject(action, reason, repeat=True)
                    self.state = "SAFE_STOP"
                    break
                if action.name == "SAFE_STOP":
                    reason = "Agent requested safe stop"
                    self.state = "SAFE_STOP"
                    break
                failures = self._hard_constraints(action)
                if failures:
                    reason = "; ".join(failures)
                    if self._reject(action, reason):
                        self.state = "SAFE_STOP"
                    continue
                assessment = self._assessment(action)
                self._record("PROBABILITY_ASSESSMENT", action=action.name,
                             assessment=assessment.model_dump())
                if action.name in HIGH_RISK:
                    mean = (assessment.probabilities or {}).get("SUCCESS")
                    lower = assessment.success_interval[0] if assessment.success_interval else None
                    if not assessment.calculable or not assessment.gate_eligible:
                        reason = "No gate-grade probability assessment"
                        if self._reject(action, reason, assessment):
                            self.state = "SAFE_STOP"
                        continue
                    if assessment.unique_task_count < self.config.min_unique_tasks:
                        reason = "Insufficient unique-task evidence for probability gate"
                        if self._reject(action, reason, assessment):
                            self.state = "SAFE_STOP"
                        continue
                    if mean is None or mean < self.config.threshold:
                        reason = "Mean success probability below threshold"
                        if self._reject(action, reason, assessment):
                            self.state = "SAFE_STOP"
                        continue
                    if lower is None or lower < self.config.threshold:
                        reason = "VERIFY_MORE: conservative lower bound below threshold"
                        self._record("VERIFY_MORE", action=action.name, reason=reason)
                        if self._reject(action, reason, assessment):
                            self.state = "SAFE_STOP"
                        continue
                else:
                    self._record("PREPARATION_ALLOWED", action=action.name,
                                 reason="Read-only or reversible progress within budget")
                self._record("TOOL_EXECUTION", action=action.name)
                error = self._execute(action)
                if error:
                    reason = error
                    if self.state != "SAFE_STOP":
                        if self._reject(action, f"Postcondition failed: {error}"):
                            self.state = "SAFE_STOP"
            if not reason:
                reason = "Verified goal" if self.state == "SUCCESS" else "Safe stop"
        except Exception as exc:
            reason = f"Technical failure: {type(exc).__name__}: {exc}"
            self.state = "SAFE_STOP"
            self._record("TECHNICAL_FAILURE", reason=reason)
        self._record("TERMINAL", outcome=self.state, reason=reason,
                     poststate_verified=self.state == "SUCCESS")
        return HarnessResult(self.state, self.state, self.events,
                             list(self.fs.tool_calls), reason)
