from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable


def _unique_task_ids(task_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(task_id) for task_id in task_ids))


def subset_report_stem(dataset: str, task_ids: Iterable[str]) -> str:
    """Return a stable file name without claiming a subset is the full dataset."""

    ids = sorted(_unique_task_ids(task_ids))
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:12]
    return f"harness_subset_{dataset}_{digest}"


def scenario_coverage(
    selected_task_ids: Iterable[str],
    dataset_task_ids: Iterable[str],
    scenario_id_for_task,
) -> dict[str, Any]:
    """Describe whether every selected scenario contains all of its task variants."""

    selected = set(_unique_task_ids(selected_task_ids))
    dataset = set(_unique_task_ids(dataset_task_ids))
    all_by_scenario: dict[str, set[str]] = defaultdict(set)
    selected_by_scenario: dict[str, set[str]] = defaultdict(set)
    for task_id in dataset:
        all_by_scenario[str(scenario_id_for_task(task_id))].add(task_id)
    for task_id in selected:
        selected_by_scenario[str(scenario_id_for_task(task_id))].add(task_id)

    complete = 0
    partial = 0
    missing_variants = 0
    for scenario_id, selected_ids in selected_by_scenario.items():
        expected_ids = all_by_scenario.get(scenario_id, selected_ids)
        missing = expected_ids - selected_ids
        if missing:
            partial += 1
            missing_variants += len(missing)
        else:
            complete += 1
    return {
        "selected_tasks": len(selected),
        "selected_scenarios": len(selected_by_scenario),
        "complete_scenarios": complete,
        "partial_scenarios": partial,
        "missing_task_variants": missing_variants,
        "scenario_goal_completion_interpretable": partial == 0,
    }


def run_official_evaluation(
    *,
    appworld_root: Path,
    experiment_name: str,
    dataset: str,
    task_ids: Iterable[str],
) -> dict[str, Any]:
    """Evaluate saved task databases with AppWorld's official evaluator.

    This function never invokes the model or executes the agent. It calls
    ``appworld.evaluator.evaluate_tasks`` over the already saved task outputs.
    """

    appworld_root = Path(appworld_root).resolve()
    ids = _unique_task_ids(task_ids)
    if not ids:
        raise ValueError("Official evaluation requires at least one task ID")

    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    from appworld import AppWorld, update_root
    from appworld.common.printer import table_data_to_string
    from appworld.evaluator import Metric, evaluate_tasks
    from appworld.task import load_task_ids, task_id_to_generator_id

    update_root(str(appworld_root))
    AppWorld.close_all()
    dataset_ids = list(load_task_ids(dataset_name=dataset))
    unknown = sorted(set(ids) - set(dataset_ids))
    if unknown:
        raise ValueError(
            f"Task IDs are outside AppWorld {dataset}: {', '.join(unknown[:5])}"
        )

    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    metrics = evaluate_tasks(
        task_ids=ids,
        experiment_name=experiment_name,
        suppress_errors=True,
        include_details=True,
        save_reports=True,
    )
    table = Metric.build_report(metrics)
    text_report = "Text Evaluation Report:\n" + table_data_to_string(table)
    coverage = scenario_coverage(ids, dataset_ids, task_id_to_generator_id)

    evaluation_dir = (
        appworld_root / "experiments" / "outputs" / experiment_name / "evaluations"
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    full_dataset = set(ids) == set(dataset_ids)
    stem = dataset if full_dataset else subset_report_stem(dataset, ids)
    json_path = evaluation_dir / f"{stem}.json"
    text_path = evaluation_dir / f"{stem}.txt"
    metadata_path = evaluation_dir / f"{stem}.metadata.json"
    json_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text_path.write_text(text_report + "\n", encoding="utf-8")

    summary = {
        "status": "complete",
        "evaluator": "appworld.evaluator.evaluate_tasks",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.perf_counter() - started_clock, 3),
        "dataset": dataset,
        "full_dataset": full_dataset,
        "task_ids": ids,
        "task_count": len(ids),
        "aggregate": metrics.get("aggregate", {}),
        "difficulty_report": table,
        "scenario_coverage": coverage,
        "json_report": str(json_path),
        "text_report": str(text_path),
        "per_task_reports_saved": True,
        "metadata_report": str(metadata_path),
    }
    metadata_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
