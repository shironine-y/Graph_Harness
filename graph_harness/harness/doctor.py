import json
from pathlib import Path
from tempfile import TemporaryDirectory

from harness import __version__
from harness.cli import DEFAULT_REFERENCE, run_local_demo
from harness.engine import HarnessConfig


def run_check() -> dict:
    checks = {"reference_exists": DEFAULT_REFERENCE.is_file()}
    if not checks["reference_exists"]:
        raise FileNotFoundError(DEFAULT_REFERENCE)
    with TemporaryDirectory(prefix="probability_harness_doctor_") as directory:
        summary = run_local_demo(
            reference=DEFAULT_REFERENCE,
            output=Path(directory) / "run",
            config=HarnessConfig(),
            draws=300,
            seed=7,
        )
        checks["success_path"] = summary["success_demo"]["outcome"] == "SUCCESS"
        checks["unsafe_path_blocked"] = summary["blocked_demo"]["outcome"] == "SAFE_STOP"
        checks["mock_only_boundary"] = "MockFilesystem" in summary["safety_boundary"]
    return {
        "harness_version": __version__,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> None:
    report = run_check()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
