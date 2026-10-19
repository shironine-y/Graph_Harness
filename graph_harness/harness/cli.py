import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

from harness import __version__
from harness.engine import CandidateAction, FakeAgent, Harness, HarnessConfig, MockFilesystem
from harness.graph import DrawioGraphAdapter
from harness.probability import MockProbabilityGraphProvider, write_mock_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = Path(__file__).resolve().parent / "resources" / "reference.drawio"
DEFAULT_ENV_FILE = PROJECT_ROOT.parent / "baseline" / "tau3-bench" / ".env"


def _action(name: str, params: dict | None = None) -> CandidateAction:
    return CandidateAction(name=name, params=params or {})


def _new_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return PROJECT_ROOT / "outputs" / f"run_{stamp}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_trace(path: Path, result) -> None:
    lines = [f"outcome={result.outcome}", f"reason={result.reason}",
             "tools=" + ", ".join(result.tool_calls), ""]
    for event in result.events:
        assessment = event.get("assessment") or (event.get("feedback") or {}).get(
            "probability"
        )
        detail = event.get("reason") or (event.get("feedback") or {}).get("reason", "")
        probability = ""
        if assessment and assessment.get("probabilities"):
            mean = assessment["probabilities"]["SUCCESS"]
            lower, upper = assessment["success_interval"]
            probability = (
                f" success={mean:.3f} interval=[{lower:.3f},{upper:.3f}] "
                f"status={assessment['data_status']}"
            )
        action = event.get("action") or (event.get("candidate") or {}).get("name", "")
        lines.append(
            f"{event['index']:02d} {event['kind']} state={event['state']} "
            f"action={action} {detail}{probability}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_local_demo(
    *,
    reference: Path,
    output: Path,
    config: HarnessConfig,
    draws: int = 2000,
    seed: int = 11,
    remote_smoke: bool = False,
    env_file: Path = DEFAULT_ENV_FILE,
) -> dict:
    """Run the isolated success/blocked demonstrations and return a summary."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    event_graph, execution_graph = DrawioGraphAdapter().import_file(reference)
    (output / "event_graph_v1.json").write_text(
        event_graph.model_dump_json(indent=2), encoding="utf-8"
    )
    (output / "execution_graph_v1.json").write_text(
        execution_graph.model_dump_json(indent=2), encoding="utf-8"
    )
    import_report = {
        "source_file": str(reference.resolve()),
        "source_sha256": event_graph.source_sha256,
        "graph_version": event_graph.graph_version,
        "element_count": len(event_graph.elements),
        "event_count": len(event_graph.events),
        "relation_count": len(event_graph.relations),
        "warnings": event_graph.warnings,
        "execution_graph_executable": execution_graph.executable,
        "gate_eligibility": "NO: reference has incomplete outcomes/denominators",
    }
    (output / "drawio_import_report.json").write_text(
        json.dumps(import_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    graph_path = output / "mock_graph_v1.json"
    graph = write_mock_graph(graph_path)
    provider = MockProbabilityGraphProvider(graph, draws=draws, seed=seed)
    paths = {"source": "demo/source", "archive": "demo/source.zip"}
    success_actions = [
        _action("INSPECT"),
        _action("ARCHIVE", paths),
        _action("VERIFY_ARCHIVE", paths),
        _action("DELETE_SOURCE", paths),
        _action("VERIFY_POSTSTATE", paths),
    ]
    if remote_smoke:
        from harness.remote_agent import RemoteProposalAgent

        agent = RemoteProposalAgent(env_file, enabled=True)
        name = "remote_mock_smoke"
    else:
        agent = FakeAgent(success_actions)
        name = "demo_success"
    success = Harness(
        agent,
        provider,
        MockFilesystem(),
        config=config,
        audit_path=output / f"{name}_audit.jsonl",
    ).run()
    _write_trace(output / f"{name}_trace.txt", success)

    blocked = None
    if not remote_smoke:
        invalid = {**paths, "source": "demo/unauthorized", "request_id": "first"}
        invalid_again = {**paths, "source": "demo/unauthorized", "request_id": "second"}
        blocked_agent = FakeAgent(
            [_action("DELETE_SOURCE", invalid), _action("DELETE_SOURCE", invalid_again)]
        )
        blocked = Harness(
            blocked_agent,
            provider,
            MockFilesystem(),
            config=config,
            audit_path=output / "demo_blocked_audit.jsonl",
        ).run()
        _write_trace(output / "demo_blocked_trace.txt", blocked)

    summary = {
        "harness_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "mode": config.mode,
        "remote_smoke": remote_smoke,
        "config": config.model_dump(),
        "config_sha256": hashlib.sha256(
            config.model_dump_json().encode("utf-8")
        ).hexdigest(),
        "reference_sha256": _sha256(reference),
        "mock_graph_sha256": _sha256(graph_path),
        "success_demo": {"outcome": success.outcome, "reason": success.reason},
        "blocked_demo": (
            {"outcome": blocked.outcome, "reason": blocked.reason} if blocked else None
        ),
        "safety_boundary": "isolated MockFilesystem only; no real path operations",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run probability-graph Harness commands.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min-unique-tasks", type=int, default=10)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--remote-smoke", action="store_true")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    output = args.output_dir or _new_output_dir()
    config = HarnessConfig(
        threshold=args.threshold,
        min_unique_tasks=args.min_unique_tasks,
    )
    summary = run_local_demo(
        reference=args.reference,
        output=output,
        config=config,
        draws=args.draws,
        seed=args.seed,
        remote_smoke=args.remote_smoke,
        env_file=args.env_file,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
