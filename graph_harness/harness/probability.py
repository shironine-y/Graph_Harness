import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


OUTCOMES = ("SUCCESS", "PARTIAL", "FAILURE")


class Transition(BaseModel):
    next_state: str
    count: int = Field(ge=0)
    unique_task_count: int = Field(ge=0)
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_counts(self):
        if self.count != self.unique_task_count:
            raise ValueError("Mock counts require one observation per unique task")
        if len(self.evidence_refs) != self.unique_task_count:
            raise ValueError("Every mock task must have one evidence reference")
        return self


class TransitionCell(BaseModel):
    state: str
    action: str
    transitions: list[Transition]
    exhaustive: bool = True
    verified_attributes: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    side_effect_level: str = "READ_ONLY"
    compensation_plan: str = "NONE"

    @model_validator(mode="after")
    def check_cell(self):
        if not self.exhaustive or not self.transitions:
            raise ValueError("Each mock cell must enumerate all next-state outcomes")
        targets = [t.next_state for t in self.transitions]
        if len(targets) != len(set(targets)):
            raise ValueError("Duplicate next state in mock cell")
        refs = [r for t in self.transitions for r in t.evidence_refs]
        if len(refs) != len(set(refs)):
            raise ValueError("A task is counted twice in one state/action cell")
        return self


class MockGraphSpec(BaseModel):
    schema_version: str = "1.0"
    graph_version: str
    data_status: Literal["MOCK_DEMO_ONLY"] = "MOCK_DEMO_ONLY"
    domain: str = "filesystem_demo"
    benchmark: str = "mock_harness"
    task_family: str = "archive_then_delete"
    terminal_outcomes: dict[str, Literal["SUCCESS", "PARTIAL", "FAILURE"]]
    goal_predicates: list[str] = Field(default_factory=list)
    state_predicates: dict[str, list[str]] = Field(default_factory=dict)
    policy: dict[str, str]
    cells: list[TransitionCell]
    max_horizon: int = Field(gt=0)
    model_version: str = "mock-counts-v1"
    prompt_version: str = "fake-agent-v1"
    tool_version: str = "in-memory-fs-v1"
    harness_version: str = "1.0"

    @model_validator(mode="after")
    def check_graph(self):
        names = [(c.state, c.action) for c in self.cells]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate state/action cell")
        known = set(self.terminal_outcomes) | {c.state for c in self.cells}
        if any(t.next_state not in known for c in self.cells for t in c.transitions):
            raise ValueError("Unknown next state")
        if any((s, a) not in names for s, a in self.policy.items()):
            raise ValueError("Policy action lacks transition cell")
        if any(s not in self.policy for s in known - set(self.terminal_outcomes)):
            raise ValueError("Nonterminal state lacks fixed policy")
        return self


class VerifiedAttribute(BaseModel):
    value: Any
    source: str


class ProbabilityAssessment(BaseModel):
    calculable: bool
    probability_kind: str = "P_FINAL_OUTCOME_FINITE_HORIZON"
    probabilities: dict[str, float] | None = None
    success_interval: tuple[float, float] | None = None
    sample_count: int = 0
    unique_task_count: int = 0
    denominator: int = 0
    prior: dict[str, float] = Field(default_factory=dict)
    posterior: dict[str, Any] = Field(default_factory=dict)
    fallback_level: str = "none"
    evidence_refs: list[str] = Field(default_factory=list)
    data_status: str
    graph_version: str
    policy_version: str
    remaining_horizon: int
    reason: str | None = None
    gate_eligible: bool = False
    is_mock: bool = False


class ProbabilityProvider(ABC):
    @abstractmethod
    def evaluate(self, task_state, candidate_action, verified_attributes,
                 remaining_horizon, policy_context) -> ProbabilityAssessment:
        """Evaluate one candidate under a declared continuation policy."""


class KnowledgeGraphProbabilityProvider(ProbabilityProvider):
    """Explicit placeholder: reference graph lacks gate-grade observations."""

    def __init__(self, graph_version: str):
        self.graph_version = graph_version

    def evaluate(self, task_state, candidate_action, verified_attributes,
                 remaining_horizon, policy_context) -> ProbabilityAssessment:
        return ProbabilityAssessment(
            calculable=False, data_status="INSUFFICIENT_REFERENCE_EVIDENCE",
            graph_version=self.graph_version, policy_version="undefined",
            remaining_horizon=remaining_horizon,
            reason="Missing exhaustive conditional outcomes, denominator, and fixed policy",
        )


class MockProbabilityGraphProvider(ProbabilityProvider):
    def __init__(self, graph: MockGraphSpec, *, prior_alpha: float = 0.5,
                 draws: int = 2000, seed: int = 11):
        if prior_alpha <= 0 or draws < 100:
            raise ValueError("Positive Dirichlet prior and at least 100 draws required")
        self.graph = graph
        self.prior_alpha = prior_alpha
        self.draws = draws
        self.seed = seed
        self.cells = {(c.state, c.action): c for c in graph.cells}

    @classmethod
    def from_json(cls, path: str | Path, **kwargs):
        return cls(MockGraphSpec.model_validate_json(Path(path).read_text(encoding="utf-8")),
                   **kwargs)

    def _unknown(self, reason, horizon):
        return ProbabilityAssessment(
            calculable=False, data_status="INSUFFICIENT_MOCK_EVIDENCE",
            graph_version=self.graph.graph_version,
            policy_version=f"{self.graph.graph_version}:fixed-policy",
            remaining_horizon=horizon, reason=reason, is_mock=True,
        )

    def evaluate(self, task_state, candidate_action, verified_attributes,
                 remaining_horizon, policy_context) -> ProbabilityAssessment:
        if not isinstance(remaining_horizon, int) or not 1 <= remaining_horizon <= self.graph.max_horizon:
            return self._unknown("Invalid or exhausted horizon", remaining_horizon)
        if policy_context != self.graph.policy:
            return self._unknown("Continuation policy/version mismatch", remaining_horizon)
        cell = self.cells.get((task_state, candidate_action))
        if cell is None:
            return self._unknown("No exhaustive state/action cell", remaining_horizon)
        for key, value in cell.verified_attributes.items():
            attr = verified_attributes.get(key)
            if not isinstance(attr, VerifiedAttribute) or not attr.source or attr.value != value:
                return self._unknown(f"Missing verified attribute: {key}", remaining_horizon)
        if any(t.count == 0 for t in cell.transitions):
            return self._unknown("A possible transition has no observed tasks", remaining_horizon)
        reachable = {(cell.state, cell.action)}
        frontier = {t.next_state for t in cell.transitions}
        for _ in range(remaining_horizon - 1):
            pairs = [(s, self.graph.policy.get(s)) for s in frontier
                     if s not in self.graph.terminal_outcomes]
            if any((s, a) not in self.cells for s, a in pairs):
                return self._unknown("Incomplete continuation coverage", remaining_horizon)
            reachable.update(pairs)
            frontier = {t.next_state for s, a in pairs
                        for t in self.cells[(s, a)].transitions}
        rng = random.Random(self.seed)
        estimates = []
        for _ in range(self.draws):
            sampled = {}
            for key in reachable:
                transitions = self.cells[key].transitions
                weights = [rng.gammavariate(t.count + self.prior_alpha, 1)
                           for t in transitions]
                sampled[key] = [w / sum(weights) for w in weights]

            memo = {}

            def value(state, steps):
                if state in self.graph.terminal_outcomes:
                    outcome = self.graph.terminal_outcomes[state]
                    return tuple(float(name == outcome) for name in OUTCOMES)
                if steps == 0:
                    return (0.0, 0.0, 1.0)  # horizon exhausted
                action = self.graph.policy[state]
                return q(state, action, steps)

            def q(state, action, steps):
                key = (state, action, steps)
                if key not in memo:
                    cell_now = self.cells[(state, action)]
                    components = [value(t.next_state, steps - 1)
                                  for t in cell_now.transitions]
                    memo[key] = tuple(sum(p * component[i] for p, component in
                                          zip(sampled[(state, action)], components))
                                      for i in range(3))
                return memo[key]

            estimates.append(q(task_state, candidate_action, remaining_horizon))

        means = [sum(row[i] for row in estimates) / len(estimates) for i in range(3)]
        ordered = sorted(row[0] for row in estimates)
        n = sum(t.unique_task_count for t in cell.transitions)
        refs = [r for t in cell.transitions for r in t.evidence_refs]
        return ProbabilityAssessment(
            calculable=True,
            probabilities=dict(zip(OUTCOMES, means)),
            success_interval=(ordered[int(.025 * self.draws)],
                              ordered[min(self.draws - 1, int(.975 * self.draws))]),
            sample_count=n, unique_task_count=n, denominator=n,
            prior={"dirichlet_alpha_per_outcome": self.prior_alpha},
            posterior={"method": "seeded_dirichlet_monte_carlo", "draws": self.draws},
            evidence_refs=refs, fallback_level="exact_state_action_mock",
            data_status="MOCK_DEMO_ONLY", graph_version=self.graph.graph_version,
            policy_version=f"{self.graph.graph_version}:fixed-policy",
            remaining_horizon=remaining_horizon, gate_eligible=True, is_mock=True,
        )


def write_mock_graph(path: str | Path) -> MockGraphSpec:
    """Build a fully enumerated synthetic graph with disjoint synthetic task IDs."""
    definition = [
        ("START", "INSPECT", [("READY", 65), ("START", 25), ("FAILURE", 10)]),
        ("READY", "ARCHIVE", [("ARCHIVED", 95), ("PARTIAL", 3), ("ERROR", 2)]),
        ("ARCHIVED", "VERIFY_ARCHIVE", [("VERIFIED", 97), ("READY", 2), ("FAILURE", 1)]),
        ("VERIFIED", "DELETE_SOURCE", [("DELETED", 98), ("PARTIAL", 1), ("FAILURE", 1)]),
        ("DELETED", "VERIFY_POSTSTATE", [("SUCCESS", 99), ("PARTIAL", 1)]),
        ("ERROR", "REPAIR_PARAMS", [("READY", 80), ("FAILURE", 20)]),
    ]
    for state in ("START", "READY", "ARCHIVED", "VERIFIED", "DELETED", "ERROR"):
        target = "PARTIAL" if state in {"ARCHIVED", "VERIFIED", "DELETED"} else "FAILURE"
        definition.append((state, "SAFE_STOP", [(target, 100)]))
    cells = []
    for state, action, outcomes in definition:
        transitions = [Transition(next_state=target, count=count,
                                  unique_task_count=count,
                                  evidence_refs=[f"mock:{state}:{action}:{i:03d}"
                                                 for i in range(offset, offset + count)])
                       for offset, (target, count) in
                       zip((sum(n for _, n in outcomes[:j]) for j in range(len(outcomes))),
                           outcomes)]
        level = ("IRREVERSIBLE" if action == "DELETE_SOURCE" else
                 "REVERSIBLE" if action in {"ARCHIVE", "REPAIR_PARAMS"} else "READ_ONLY")
        cells.append(TransitionCell(
            state=state, action=action, transitions=transitions,
            preconditions={"DELETE_SOURCE": ["explicit_authorization", "archive_verified",
                                                 "backup_available", "path_scope_valid"],
                           "ARCHIVE": ["source_inspected", "no_archive_collision"]}.get(action, []),
            postconditions={"ARCHIVE": ["archive_exists"],
                            "VERIFY_ARCHIVE": ["archive_checksum_valid"],
                            "DELETE_SOURCE": ["source_absent"],
                            "VERIFY_POSTSTATE": ["source_absent", "archive_checksum_valid"]
                            }.get(action, []),
            side_effect_level=level,
            compensation_plan="UNDO_ARCHIVE" if action == "ARCHIVE" else "NONE",
        ))
    graph = MockGraphSpec(
        graph_version="mock-archive-v1", max_horizon=12,
        terminal_outcomes={name: name for name in OUTCOMES},
        goal_predicates=["source_absent", "archive_checksum_valid",
                         "poststate_verified"],
        state_predicates={"VERIFIED": ["archive_checksum_valid"],
                          "DELETED": ["source_absent"],
                          "SUCCESS": ["source_absent", "archive_checksum_valid",
                                      "poststate_verified"]},
        policy={"START": "INSPECT", "READY": "ARCHIVE",
                "ARCHIVED": "VERIFY_ARCHIVE", "VERIFIED": "DELETE_SOURCE",
                "DELETED": "VERIFY_POSTSTATE", "ERROR": "REPAIR_PARAMS"}, cells=cells,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    return graph
