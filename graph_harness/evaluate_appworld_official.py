from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.appworld_official import run_official_evaluation


ROOT = Path(__file__).resolve().parent
DEFAULT_APPWORLD = ROOT.parent / "baseline" / "appworld"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an AppWorld experiment with the official evaluator.")
    parser.add_argument("--appworld-root", type=Path, default=DEFAULT_APPWORLD)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--dataset", default="test_normal")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--task-id", dest="task_ids", action="append",
        help="Saved task output to evaluate; repeat for multiple tasks",
    )
    source.add_argument(
        "--task-results", type=Path,
        help="Harness task_results.jsonl whose task IDs should be evaluated",
    )
    args = parser.parse_args()
    task_ids = args.task_ids
    if args.task_results is not None:
        task_ids = []
        for line in args.task_results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("dataset") == args.dataset:
                    task_ids.append(str(row["task_id"]))
    result = run_official_evaluation(
        appworld_root=args.appworld_root,
        experiment_name=args.experiment_name,
        dataset=args.dataset,
        task_ids=task_ids,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
