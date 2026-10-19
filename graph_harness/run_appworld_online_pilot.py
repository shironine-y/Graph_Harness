from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from dataset_selection import dataset_path, normalize_dataset_name
import time
from uuid import uuid4

from harness.appworld import write_appworld_graph
from harness.appworld_online import (
    AdvisoryOnlineGate,
    MAX_API_CALLS_PER_STEP,
    GraphReadyOnlineGate,
    OnlineProbabilityGate,
    build_empirical_graph,
    load_empirical_trace_export,
    load_dataset_ids,
    load_model_settings,
    run_online_task,
    select_online_tasks,
)
from harness.graph_ready import GraphReadyBuild
from harness.graph_guidance import GraphReadyRetriever
from harness.neo4j_guidance import Neo4jGraphGuidanceProvider
from harness.appworld_official import run_official_evaluation
from harness.task_targets import TARGET_METRICS


ROOT = Path(__file__).resolve().parent
DEFAULT_APPWORLD = ROOT.parent / "baseline" / "appworld"
DEFAULT_ENV = ROOT.parent / "baseline" / "tau3-bench" / ".env"
DEFAULT_TRACE_EXPORT = ROOT / "data" / "appworld_train_20x5_sanitized.json"


def audit_report_consistency(summaries) -> list[str]:
    """Check that task summaries agree with their append-only audit logs."""
    errors: list[str] = []
    required_rollback_fields = {
        "cause", "rollback_scope", "rollback_verified", "physical_restore_performed",
        "environment_safe_to_continue", "agent_requested",
    }
    required_failure_fields = {
        "failure_type", "state_changed", "rollback_required", "rollback_level",
        "rollback_requested_by_agent", "rollback_request_accepted",
        "rollback_request_rejection_reason", "rollback_budget_exhausted",
        "task_terminated_due_to_rollback_budget", "rollback_verified",
    }
    for row in summaries:
        audit = Path(row.harness_audit)
        if not audit.is_file():
            errors.append(f"{row.task_id}: missing harness audit {audit}")
            continue
        try:
            events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{row.task_id}: unreadable harness audit ({type(exc).__name__})")
            continue
        rollbacks = [event for event in events if event.get("kind") == "ROLLBACK"]
        if len(rollbacks) != row.rollback_attempts:
            errors.append(
                f"{row.task_id}: rollback attempts summary={row.rollback_attempts} "
                f"audit={len(rollbacks)}"
            )
        restored = sum(bool(event.get("restored")) for event in rollbacks)
        if restored != row.rollback_successes:
            errors.append(
                f"{row.task_id}: rollback successes summary={row.rollback_successes} "
                f"audit={restored}"
            )
        cause_counts = Counter(
            str(event.get("cause")) for event in rollbacks if event.get("restored")
        )
        for cause in set(cause_counts) | set(row.rollback_cause_counts):
            if cause_counts.get(cause, 0) != row.rollback_cause_counts.get(cause, 0):
                errors.append(
                    f"{row.task_id}: rollback cause {cause} summary="
                    f"{row.rollback_cause_counts.get(cause, 0)} audit={cause_counts.get(cause, 0)}"
                )
        classifications = sum(
            event.get("kind") == "FAILURE_CLASSIFICATION" for event in events
        )
        if classifications != row.failure_classification_count:
            errors.append(
                f"{row.task_id}: failure classifications summary="
                f"{row.failure_classification_count} audit={classifications}"
            )
        for event in events:
            if event.get("kind") != "FAILURE_CLASSIFICATION":
                continue
            missing = sorted(required_failure_fields - set(event))
            if missing:
                errors.append(
                    f"{row.task_id}: failure step {event.get('step')} missing "
                    + ", ".join(missing)
                )
        for event in rollbacks:
            missing = sorted(required_rollback_fields - set(event))
            if missing:
                errors.append(
                    f"{row.task_id}: rollback step {event.get('step')} missing "
                    + ", ".join(missing)
                )
        technical_events = [
            event for event in events if event.get("kind") == "TECHNICAL_FAILURE"
        ]
        expected_technical_events = int(row.stop_reason == "Technical failure")
        if len(technical_events) != expected_technical_events:
            errors.append(
                f"{row.task_id}: technical failure events summary="
                f"{expected_technical_events} audit={len(technical_events)}"
            )
        if technical_events:
            event = technical_events[-1]
            if event.get("stage") != row.technical_failure_stage:
                errors.append(
                    f"{row.task_id}: technical failure stage summary="
                    f"{row.technical_failure_stage!r} audit={event.get('stage')!r}"
                )
            if event.get("location") != row.technical_failure_location:
                errors.append(
                    f"{row.task_id}: technical failure location summary="
                    f"{row.technical_failure_location!r} audit={event.get('location')!r}"
                )
        completion_events = [e for e in events if e.get("kind") == "COMPLETION_VERIFICATION"]
        completion_counts = Counter(e.get("status") for e in completion_events)
        checks = {
            "completion_verification_checks": len(completion_events),
            "completion_satisfied": completion_counts["SATISFIED"],
            "completion_violated": completion_counts["VIOLATED"],
            "completion_unknown": completion_counts["UNKNOWN"],
            "completion_blocks": sum(e.get("completion_attempt_blocked", e.get("status") == "VIOLATED") for e in completion_events),
            "completion_checks": len(completion_events),
            "completion_evidence_reads": sum(e.get("kind") == "COMPLETION_EVIDENCE_READ" for e in events),
            "rollback_scope_feedback_events": sum(e.get("kind") == "ROLLBACK_SCOPE_FEEDBACK" for e in events),
            "final_preflight_checks": sum(e.get("kind") == "FINAL_SEMANTIC_PREFLIGHT" for e in events),
            "final_preflight_violations": sum(e.get("kind") == "FINAL_SEMANTIC_PREFLIGHT" and e.get("status") == "VIOLATED" for e in events),
            "final_preflight_unknown": sum(e.get("kind") == "FINAL_SEMANTIC_PREFLIGHT" and e.get("status") in {"UNKNOWN", "EVIDENCE_PENDING"} for e in events),
            "final_reconciliation_checks": sum(e.get("phase") == "final_reconciliation" for e in completion_events),
        }
        for name, count in checks.items():
            if getattr(row, name, 0) != count:
                errors.append(f"{row.task_id}: {name} summary={getattr(row, name, 0)} audit={count}")
        for event in completion_events:
            if (event.get("rollback_required") is not False
                    or event.get("rollback_budget_consumed") is not False
                    or event.get("ordinary_rejection_budget_consumed") is not False):
                errors.append(f"{row.task_id}: completion check incorrectly entered recovery budget")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AppWorld tasks with graph guidance and execution safeguards.")
    parser.add_argument("--appworld-root", type=Path, default=DEFAULT_APPWORLD)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--trace-export", type=Path)
    parser.add_argument("--graph-ready-build", type=Path)
    parser.add_argument("--graph-guidance", action="store_true",
                        help="Give the agent bounded train-graph API hints; no probability gating")
    parser.add_argument("--neo4j-guidance", action="store_true",
                        help="Query train transitions from Neo4j; hints are advisory only")
    parser.add_argument("--advisory-control", action="store_true",
                        help="Run the same Harness without graph lookup or probability gating")
    parser.add_argument("--train-tasks", type=int)
    parser.add_argument("--test-tasks", type=int, default=3)
    parser.add_argument("--dataset-name", default="test_normal",
                        help="Name or .txt basename under AppWorld data/datasets; no fixed dataset whitelist")
    parser.add_argument(
        "--test-task-id",
        dest="test_task_ids",
        action="append",
        metavar="TASK_ID",
        help=("Run this exact task from --dataset-name instead of randomly selecting test tasks; "
              "repeat the option to run multiple explicit tasks"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--generation-seed", type=int, default=None,
                        help="Model sampling seed, independent of task-selection seed")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min-unique-tasks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-rejections", type=int, default=5)
    parser.add_argument("--max-rollbacks", type=int, default=4)
    parser.add_argument("--max-action-retries", type=int, default=1)
    parser.add_argument("--max-decision-rollbacks", type=int, default=2)
    parser.add_argument("--task-target-tracking", action="store_true",
                        help="Track targets from public instruction and verified reads")
    parser.add_argument("--completion-verification", action="store_true",
                        help="Enable confirmed-target-mismatch completion checks")
    parser.add_argument("--completion-evidence-read-budget", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--stop-on-technical-failure",
        action="store_true",
        help=(
            "Stop the remaining batch after an internal/model technical failure; "
            "completed task results and audits are still saved"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--confirm-online", action="store_true")
    parser.add_argument(
        "--skip-official-evaluation", action="store_true",
        help="Skip AppWorld's official batch evaluator after tasks finish",
    )
    args = parser.parse_args()

    if args.completion_evidence_read_budget < 0:
        raise SystemExit("--completion-evidence-read-budget cannot be negative")
    if args.completion_verification:
        args.task_target_tracking = True
    if args.max_action_retries < 0:
        raise SystemExit("--max-action-retries cannot be negative")
    if args.max_decision_rollbacks < 1:
        raise SystemExit("--max-decision-rollbacks must be positive")

    appworld_root = args.appworld_root.resolve()
    if args.graph_ready_build is not None and args.trace_export is not None:
        raise SystemExit("Select either --graph-ready-build or --trace-export, not both")
    if args.neo4j_guidance and (args.graph_ready_build or args.trace_export
                                or args.graph_guidance):
        raise SystemExit("--neo4j-guidance cannot be combined with copied graph modes")
    if args.advisory_control and (args.neo4j_guidance or args.graph_ready_build
                                  or args.trace_export or args.graph_guidance):
        raise SystemExit("--advisory-control cannot be combined with a graph source")
    if args.graph_guidance and args.graph_ready_build is None:
        raise SystemExit("--graph-guidance requires --graph-ready-build")
    if (args.neo4j_guidance or args.advisory_control) and args.train_tasks not in (None, 0):
        raise SystemExit("Advisory A/B modes are evaluated on held-out test tasks only")
    graph_ready = GraphReadyBuild(args.graph_ready_build) if args.graph_ready_build else None
    retriever = GraphReadyRetriever(graph_ready) if args.graph_guidance else None
    if retriever is not None and not retriever.paths:
        raise SystemExit("Graph guidance has no train paths with linked goals and tools: "
                         + json.dumps(retriever.index_summary, ensure_ascii=False))
    train_ids = set(load_dataset_ids(appworld_root, "train"))
    try:
        dataset_name = normalize_dataset_name(args.dataset_name)
        test_ids = set(load_dataset_ids(appworld_root, dataset_name))
        selected_dataset_path = dataset_path(appworld_root, dataset_name)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    advisory_mode = args.neo4j_guidance or args.advisory_control
    if advisory_mode:
        trace_export = None
        records = None
        graph = None
        graph_ids: set[str] = set()
    elif graph_ready is None:
        trace_export = (args.trace_export or DEFAULT_TRACE_EXPORT).resolve()
        if not trace_export.is_file():
            raise SystemExit(
                f"Stopped: empirical trace export not found: {trace_export}. "
                "The online pilot never falls back to ground-truth/reference traces."
            )
        records = load_empirical_trace_export(trace_export)
        graph_ids = {record.task_id for record in records}
        non_train = sorted(graph_ids - train_ids)
        if non_train:
            raise SystemExit(
                "Stopped: trace export contains task IDs outside AppWorld train: "
                + ", ".join(non_train[:5])
            )
        graph = build_empirical_graph(records, min_unique_tasks=args.min_unique_tasks)
    else:
        trace_export = None
        records = None
        graph = graph_ready
        graph_ids = set(graph_ready.tasks)
        unexpected = graph_ids - train_ids - set(load_dataset_ids(appworld_root, "dev"))
        if unexpected or graph_ids & test_ids:
            raise SystemExit("Graph-ready build contains tasks outside AppWorld train/dev")
    train_count = args.train_tasks if args.train_tasks is not None else (
        0 if graph_ready or advisory_mode else 3
    )
    selected = select_online_tasks(
        appworld_root,
        graph_ids,
        train_count=train_count,
        test_count=0 if args.test_task_ids else args.test_tasks,
        seed=args.seed,
        dataset_name=dataset_name,
    )
    if args.test_task_ids:
        explicit_test_ids = list(dict.fromkeys(args.test_task_ids))
        unknown_test_ids = [task_id for task_id in explicit_test_ids
                            if task_id not in test_ids]
        if unknown_test_ids:
            raise SystemExit(
                f"Explicit test task IDs are not in AppWorld {dataset_name}: "
                + ", ".join(unknown_test_ids)
            )
        selected[dataset_name] = explicit_test_ids
    settings = load_model_settings(args.env_file)
    if args.generation_seed is not None:
        settings.generation_seed = args.generation_seed
    if args.temperature is not None:
        if not 0 <= args.temperature <= 2:
            parser.error("--temperature must be between 0 and 2")
        settings.temperature = args.temperature
    process_id = os.getpid()
    run_id = os.environ.get("HARNESS_RUN_ID") or uuid4().hex
    run_id = "".join(ch for ch in run_id if ch.isalnum() or ch in "-_")[:80] or uuid4().hex
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_name = f"probability_harness_{stamp}_{run_id[-8:]}"
    output = (args.output_dir or ROOT / "outputs" / f"appworld_online_{stamp}_{run_id[-8:]}").resolve()
    launch_log = os.environ.get("HARNESS_LAUNCH_LOG")
    plan = {
        "mode": "PLAN" if args.plan else "ONLINE",
        "graph_mode": ("neo4j_train_advisory" if args.neo4j_guidance else
                       "harness_advisory_control" if args.advisory_control else
                       "graph_ready_schema4" if graph_ready else "empirical_trace_pilot"),
        "model": settings.model,
        "model_sampling": {"temperature": settings.temperature, "top_p": settings.top_p,
                           "generation_seed": settings.generation_seed,
                           "task_selection_seed": args.seed},
        "run_id": run_id,
        "process_id": process_id,
        "experiment_name": experiment_name,
        "harness_output": str(output),
        "appworld_output": str(appworld_root / "experiments" / "outputs" / experiment_name),
        "launch_log": launch_log,
        "base_url": settings.base_url,
        "api_key_configured": bool(settings.api_key),
        "empirical_trace_count": (None if advisory_mode else
                                  graph_ready.summary["trace_count"] if graph_ready else len(records)),
        "graph_task_count": None if advisory_mode else len(graph_ids),
        "graph_cells": (None if advisory_mode else
                        graph_ready.summary["transition_count"] if graph_ready else len(graph.cells)),
        "graph_cells_with_safe_post_attributes": None if graph_ready or advisory_mode else sum(
            bool(cell.post_attributes) for cell in graph.cells
        ),
        "runs_per_graph_task": 5 if graph_ready is None and not advisory_mode else None,
        "graph_task_ids": None if advisory_mode else sorted(graph_ids),
        "trace_export": str(trace_export) if trace_export else None,
        "trace_export_sha256": hashlib.sha256(trace_export.read_bytes()).hexdigest() if trace_export else None,
        "empirical_successes": None if graph_ready or advisory_mode else sum(record.success for record in records),
        "empirical_failures": None if graph_ready or advisory_mode else sum(not record.success for record in records),
        "graph_source": ("Neo4j train ContextEvent transitions" if args.neo4j_guidance else
                         "none (Harness-only advisory control)" if args.advisory_control else
                         str(graph_ready.root) if graph_ready else
                         "evaluated AppWorld train model runs (20 tasks x 5 runs)"),
        "graph_split_counts": (None if advisory_mode else
                               graph_ready.split_counts if graph_ready else {"train": len(graph_ids)}),
        "online_tasks": selected,
        "dataset": dataset_name,
        "dataset_file": str(selected_dataset_path),
        "dataset_sha256": hashlib.sha256(selected_dataset_path.read_bytes()).hexdigest(),
        "dataset_task_count": len(test_ids),
        "test_data_used_for_graph": False,
        "threshold": args.threshold,
        "probability_timing": "after_verified_step_before_commit",
        "probability_kind": (
            None if args.advisory_control else
            "EXACT_TASK_OBSERVED_TRANSITION_STATISTICS" if graph_ready or args.neo4j_guidance
            else graph.probability_kind
        ),
        "probability_gate_enabled": graph_ready is None and not advisory_mode,
        "graph_guidance_enabled": retriever is not None or args.neo4j_guidance,
        "graph_guidance_index": retriever.index_summary if retriever else None,
        "graph_guidance_ranking": (
            "direct_edge_success_rate_transition_probability_coverage"
            if args.neo4j_guidance else
            "semantic_path_frequency" if retriever is not None else None
        ),
        "gate_limitation": (
            "Schema-4 statistics are exact-task observations, not calibrated cross-task "
            "success estimates. Static checks and rollback remain active; the 0.8 "
            "probability threshold is advisory in graph-ready mode."
            if graph_ready or args.neo4j_guidance else
            "No graph evidence is queried in the Harness-only control."
            if args.advisory_control else None
        ),
        "min_unique_tasks": args.min_unique_tasks,
        "max_steps_per_task": args.max_steps,
        "max_rejections_per_task": args.max_rejections,
        "rejection_budget_mode": "consecutive_reset_after_verified_commit",
        "max_rollbacks_per_task": args.max_rollbacks,
        "rollback_budget_mode": (
            "verified_physical_restores_only"
        ),
        "rollback_budget_exhaustion_policy": (
            "disable_future_restore_continue_if_safe_stop_only_if_mutated_state_is_unsafe"
        ),
        "execution_failure_policy": "fingerprint_before_rollback_no_mutation_replan_in_place",
        "explicit_decision_points_enabled": True,
        "cross_step_dependency_inference_enabled": False,
        "decision_marker": "literal HARNESS_META assignment in model code",
        "rollback_policy": "mutation_aware_action_then_validated_equivalent_failure_decision",
        "agent_requested_decision_rollback_policy": "request_only_harness_validated",
        "max_action_retries_per_decision": args.max_action_retries,
        "max_decision_rollbacks_per_task": args.max_decision_rollbacks,
        "max_api_calls_per_step": MAX_API_CALLS_PER_STEP,
        "runtime_api_verification": "bounded_structural_paths_exact_method_route_order",
        "runtime_api_budget": "request_boundary_before_extra_call",
        "checkpoint_runtime_restore": "inert_python_values_and_target_context_fresh_reads_required",
        "sequential_stop_policy": "one_advance_before_inspecting_stop_condition",
        "completion_coverage_guard": "advisory_counts_only_confirmed_target_predicates_gate",
        "completion_format_guard": "instruction_only_explicit_answer_order",
        "semantic_write_guard": (
            "instruction_fields_before_write_and_post_write_read_before_completion"
        ),
        "official_failure_reasons_in_task_results": True,
        "task_target_tracking_enabled": args.task_target_tracking,
        "completion_verification_enabled": args.completion_verification,
        "completion_evidence_read_budget": args.completion_evidence_read_budget,
        "task_contract_extraction": "atomic_public_clauses_with_explicit_unresolved_parts",
        "target_evidence_binding": "verified_request_boundary_returns_in_call_order",
        "collection_evidence": "query_subject_state_scoped_contiguous_pages",
        "graph_version": (
            os.environ.get("HARNESS_NEO4J_GRAPH_VERSION", "neo4j-live-unversioned")
            if args.neo4j_guidance else "harness-advisory-control-v1"
            if args.advisory_control else graph.graph_version
        ),
        "warning": (
            "Probabilities are empirical, model/service/settings-specific estimates. "
            "Online execution is limited to AppWorld sandboxes."
        ),
        "official_evaluator_enabled": not args.skip_official_evaluation,
        "stop_on_technical_failure": args.stop_on_technical_failure,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.plan:
        print("Preview only. No model requests or AppWorld task executions were made.")
        return
    if not args.confirm_online:
        raise SystemExit("Stopped: add --confirm-online after reviewing --plan output.")

    if args.neo4j_guidance:
        try:
            retriever = Neo4jGraphGuidanceProvider.from_environment()
        except Exception as exc:
            raise SystemExit(
                f"Stopped: Neo4j guidance unavailable ({type(exc).__name__}); "
                "check driver, connection settings, and train graph schema"
            ) from exc
        plan["graph_guidance_index"] = retriever.index_summary

    output.mkdir(parents=True, exist_ok=False)
    if advisory_mode:
        pass
    elif graph_ready is None:
        write_appworld_graph(graph, output / "train_empirical_graph.json")
    else:
        (output / "graph_reference.json").write_text(
            json.dumps(graph_ready.summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gate = (AdvisoryOnlineGate(plan["graph_version"]) if advisory_mode else
            GraphReadyOnlineGate(graph_ready) if graph_ready else OnlineProbabilityGate(
                graph, threshold=args.threshold,
                min_unique_tasks=args.min_unique_tasks, seed=args.seed,
            ))
    summaries = []
    batch_stopped_early = False
    batch_stop_reason = None
    experiment_started_at = datetime.now(timezone.utc).isoformat()
    experiment_started_clock = time.perf_counter()
    for dataset, task_ids in selected.items():
        for task_id in task_ids:
            print(f"Running {dataset} {task_id} with {settings.model} ...", flush=True)
            summary = run_online_task(
                appworld_root=appworld_root,
                dataset=dataset,
                task_id=task_id,
                experiment_name=experiment_name,
                settings=settings,
                gate=gate,
                max_steps=args.max_steps,
                max_rejections=args.max_rejections,
                max_rollbacks=args.max_rollbacks,
                max_output_tokens=args.max_output_tokens,
                max_action_retries=args.max_action_retries,
                max_decision_rollbacks=args.max_decision_rollbacks,
                graph_guidance=retriever,
                task_target_tracking=args.task_target_tracking,
                completion_verification=args.completion_verification,
                completion_evidence_read_budget=args.completion_evidence_read_budget,
            )
            summaries.append(summary)
            with (output / "task_results.jsonl").open("a", encoding="utf-8") as handle:
                task_result = summary.model_dump()
                task_result.update({
                    "run_id": run_id,
                    "process_id": process_id,
                    "experiment_name": experiment_name,
                })
                handle.write(json.dumps(task_result, ensure_ascii=False) + "\n")
            print(summary.model_dump_json(indent=2), flush=True)
            if (
                args.stop_on_technical_failure
                and summary.stop_reason == "Technical failure"
            ):
                batch_stopped_early = True
                batch_stop_reason = (
                    f"Stopped after technical failure in {dataset}/{task_id}"
                )
                print(batch_stop_reason, flush=True)
                break
        if batch_stopped_early:
            break

    by_dataset = {}
    for dataset in selected:
        rows = [row for row in summaries if row.dataset == dataset]
        evaluation_pass_count = sum(row.evaluation_pass_count for row in rows)
        evaluation_fail_count = sum(row.evaluation_fail_count for row in rows)
        evaluation_total_count = evaluation_pass_count + evaluation_fail_count
        by_dataset[dataset] = {
            "tasks": len(rows),
            "successes": sum(row.success for row in rows),
            "completed": sum(row.completed for row in rows),
            "task_goal_completion_percent": round(
                100 * sum(row.success for row in rows) / len(rows), 3
            ) if rows else None,
            "evaluation_pass_count": evaluation_pass_count,
            "evaluation_fail_count": evaluation_fail_count,
            "evaluation_total_count": evaluation_total_count,
            "evaluation_pass_rate": round(
                evaluation_pass_count / evaluation_total_count, 6
            ) if evaluation_total_count else None,
            "evaluation_failure_reasons_by_task": {
                row.task_id: row.evaluation_failure_reasons
                for row in rows if row.evaluation_failure_reasons
            },
            "evaluation_failure_details_by_task": {
                row.task_id: row.evaluation_failure_details
                for row in rows if row.evaluation_failure_details
            },
            "steps_proposed": sum(row.steps_proposed for row in rows),
            "steps_executed": sum(row.steps_executed for row in rows),
            "api_calls_executed": sum(row.api_calls_executed for row in rows),
            "duration_seconds": round(sum(row.duration_seconds for row in rows), 3),
            "rejected_steps": sum(row.rejected_steps for row in rows),
            "checkpointed_steps": sum(row.checkpointed_steps for row in rows),
            "precheck_rejections": sum(row.precheck_rejections for row in rows),
            "execution_failures": sum(row.execution_failures for row in rows),
            "no_mutation_failures": sum(row.no_mutation_failures for row in rows),
            "postcondition_failures": sum(row.postcondition_failures for row in rows),
            "repeated_decision_failures": sum(
                row.repeated_decision_failures for row in rows
            ),
            "failure_classification_count": sum(
                row.failure_classification_count for row in rows
            ),
            "failure_evidence_count": sum(
                row.failure_evidence_count for row in rows
            ),
            "active_failure_constraint_count": sum(
                row.active_failure_constraint_count for row in rows
            ),
            "rollback_attempts": sum(row.rollback_attempts for row in rows),
            "rollback_successes": sum(row.rollback_successes for row in rows),
            "rollback_failures": sum(row.rollback_failures for row in rows),
            "rollback_budget_exhausted_tasks": sum(
                row.rollback_budget_exhausted for row in rows
            ),
            "decision_rollback_budget_exhausted_tasks": sum(
                row.decision_rollback_budget_exhausted for row in rows
            ),
            "tasks_terminated_due_to_rollback_budget": sum(
                row.task_terminated_due_to_rollback_budget for row in rows
            ),
            "rollback_cause_counts": {
                cause: sum(row.rollback_cause_counts.get(cause, 0) for row in rows)
                for cause in ("execution_error", "verification_mismatch",
                              "probability_rejection", "agent_replan")
            },
            "decisions_opened": sum(row.decisions_opened for row in rows),
            "explicit_decisions_opened": sum(
                row.explicit_decisions_opened for row in rows
            ),
            "implicit_decisions_opened": sum(
                row.implicit_decisions_opened for row in rows
            ),
            "decision_commits": sum(row.decision_commits for row in rows),
            "action_rollbacks": sum(row.action_rollbacks for row in rows),
            "decision_rollbacks": sum(row.decision_rollbacks for row in rows),
            "agent_requested_decision_rollbacks": sum(
                row.agent_requested_decision_rollbacks for row in rows
            ),
            "automatic_decision_rollbacks": sum(
                row.automatic_decision_rollbacks for row in rows
            ),
            "repeated_rolled_back_proposals": sum(
                row.repeated_rolled_back_proposals for row in rows
            ),
            "completion_coverage_checks": sum(
                row.completion_coverage_checks for row in rows
            ),
            "completion_coverage_rejections": sum(
                row.completion_coverage_rejections for row in rows
            ),
            "completion_coverage_advisories": sum(
                row.completion_coverage_advisories for row in rows
            ),
            "completion_format_checks": sum(
                row.completion_format_checks for row in rows
            ),
            "completion_format_rejections": sum(
                row.completion_format_rejections for row in rows
            ),
            "semantic_write_checks": sum(
                row.semantic_write_checks for row in rows
            ),
            "semantic_write_rejections": sum(
                row.semantic_write_rejections for row in rows
            ),
            "semantic_completion_checks": sum(
                row.semantic_completion_checks for row in rows
            ),
            "semantic_completion_rejections": sum(
                row.semantic_completion_rejections for row in rows
            ),
            "harness_metadata_advisories": sum(
                row.harness_metadata_advisories for row in rows
            ),
            "post_step_probability_checks": sum(
                row.post_step_probability_checks for row in rows
            ),
            "guidance_queries": sum(row.guidance_queries for row in rows),
            "guidance_hits": sum(row.guidance_hits for row in rows),
            "guidance_errors": sum(row.guidance_errors for row in rows),
            "probability_ranked_guidance_hits": sum(
                row.probability_ranked_guidance_hits for row in rows
            ),
            "mean_success_guidance_score": (
                round(sum(
                    row.mean_success_guidance_score * row.probability_ranked_guidance_hits
                    for row in rows if row.mean_success_guidance_score is not None
                ) / sum(
                    row.probability_ranked_guidance_hits for row in rows
                ), 6)
                if sum(row.probability_ranked_guidance_hits for row in rows) else None
            ),
            "technical_failures": sum(row.error is not None for row in rows),
            "technical_failure_details": {
                row.task_id: {
                    "stage": row.technical_failure_stage,
                    "location": row.technical_failure_location,
                    "error": row.error,
                }
                for row in rows if row.stop_reason == "Technical failure"
            },
        }
        by_dataset[dataset].update({
            name: (any(getattr(row, name) for row in rows)
                   if name == "completion_verification_unknown"
                   else sum(getattr(row, name) for row in rows))
            for name in TARGET_METRICS
        })
        reason_counts = Counter()
        for row in rows:
            reason_counts.update(row.completion_unknown_reason_counts)
        by_dataset[dataset]["completion_unknown_reason_counts"] = dict(reason_counts)
        by_dataset[dataset]["completion_final_status_counts"] = dict(Counter(
            row.completion_final_status or "NOT_COMPLETED" for row in rows))
        checks = sum(row.completion_checks for row in rows)
        completed_checks = [row for row in rows if row.completed and row.completion_verification_enabled]
        by_dataset[dataset]["completion_check_unknown_rate"] = (
            sum(row.completion_unknown for row in rows) / checks if checks else None)
        by_dataset[dataset]["completion_final_uncertainty_rate"] = (
            sum(row.completion_final_status == "COMPLETED_WITH_UNCERTAINTY" for row in completed_checks)
            / len(completed_checks) if completed_checks else None)
    report_consistency_errors = audit_report_consistency(summaries)
    official_evaluation = {}
    if not args.skip_official_evaluation:
        for dataset in selected:
            task_ids = [
                row.task_id for row in summaries if row.dataset == dataset
            ]
            if not task_ids:
                continue
            try:
                official_evaluation[dataset] = run_official_evaluation(
                    appworld_root=appworld_root,
                    experiment_name=experiment_name,
                    dataset=dataset,
                    task_ids=task_ids,
                )
            except Exception as exc:
                official_evaluation[dataset] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    technical_failure_details = {
        row.task_id: {
            "dataset": row.dataset,
            "stage": row.technical_failure_stage,
            "location": row.technical_failure_location,
            "error": row.error,
        }
        for row in summaries if row.stop_reason == "Technical failure"
    }
    experiment_finished_at = datetime.now(timezone.utc).isoformat()
    report = {
        "started_at": experiment_started_at,
        "finished_at": experiment_finished_at,
        "duration_seconds": round(time.perf_counter() - experiment_started_clock, 3),
        "experiment_name": experiment_name,
        "run_id": run_id,
        "process_id": process_id,
        "model": settings.model,
        "model_sampling": plan["model_sampling"],
        "graph_version": plan["graph_version"],
        "graph_mode": plan["graph_mode"],
        "probability_gate_enabled": plan["probability_gate_enabled"],
        "graph_guidance_enabled": plan["graph_guidance_enabled"],
        "task_target_tracking_enabled": args.task_target_tracking,
        "completion_verification_enabled": args.completion_verification,
        "completion_evidence_read_budget": args.completion_evidence_read_budget,
        "task_contract_extraction": "atomic_public_clauses_with_explicit_unresolved_parts",
        "target_evidence_binding": "verified_request_boundary_returns_in_call_order",
        "empirical_trace_count": plan["empirical_trace_count"],
        "graph_task_count": plan["graph_task_count"],
        "test_data_used_for_graph": False,
        "results": by_dataset,
        "official_evaluation": official_evaluation,
        "report_consistency_errors": report_consistency_errors,
        "technical_failure_details": technical_failure_details,
        "batch_stopped_early": batch_stopped_early,
        "batch_stop_reason": batch_stop_reason,
        "artifacts": {
            "harness_output": str(output),
            "appworld_output": str(appworld_root / "experiments" / "outputs" / experiment_name),
            "launch_log": launch_log,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
