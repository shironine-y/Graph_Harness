from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import uuid
from dataset_selection import dataset_path, load_dataset_ids, normalize_dataset_name

ROOT = Path(__file__).resolve().parent


class LaunchError(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LaunchError(f"Duplicate config key: {key}")
        result[key] = value
    return result


def _keys(value, allowed, label):
    if not isinstance(value, dict):
        raise LaunchError(f"{label} must be a JSON object")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise LaunchError(f"Unsupported fields in {label}: {', '.join(unknown)}. "
                          "Use this launcher's compact config, not the StepAgent config.")


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise LaunchError(f"{label} must be a non-empty string")
    if any(c in value for c in "\r\n\0"):
        raise LaunchError(f"{label} cannot contain a newline or NUL")
    return value


def _int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise LaunchError(f"{label} must be an integer >= {minimum}")


def _number(value, label, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise LaunchError(f"{label} must be a finite number")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise LaunchError(f"{label} is outside the supported range")


def _bool(value, label):
    if type(value) is not bool:
        raise LaunchError(f"{label} must be true or false")


def _path(value, base):
    # Do not dereference a virtualenv's bin/python symlink: its original path is
    # how Python discovers pyvenv.cfg and selects the correct site-packages.
    return Path(os.path.abspath(base / Path(os.path.expandvars(_string(value, "path"))).expanduser()))


def resolve_secret(section, field, env_field, label):
    reference = section.get(env_field)
    if reference is not None:
        _string(reference, label + " environment variable")
        if not reference.isidentifier() or not reference.isascii():
            raise LaunchError(label + " environment variable name is invalid")
        value = os.environ.get(reference)
        if value is None:
            raise LaunchError("Required environment variable is not set: " + reference)
        section[field] = value
    return _string(section.get(field), label)


def python_path(value, base):
    interpreter = _string(value, "runtime.python_executable")
    if not any(c in interpreter for c in ("/", "\\")):
        found = shutil.which(interpreter)
        if found is None:
            raise LaunchError("Configured Python command was not found on PATH")
        return Path(found)
    return _path(interpreter, base)


def load_config(path):
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchError("config.json is not valid UTF-8 JSON") from exc
    _keys(config, {"schema_version", "runtime", "run", "model", "neo4j"}, "config")
    if config.get("schema_version") != "1.0":
        raise LaunchError("schema_version must be '1.0'")
    runtime, run, model, neo = (config.setdefault(k, {}) for k in ("runtime", "run", "model", "neo4j"))
    _keys(runtime, {"python_executable", "appworld_root", "output_root", "entry_script",
                    "background", "fail_on_insecure_config_permissions"}, "runtime")
    _keys(run, {"mode", "dataset", "train_tasks", "test_tasks", "test_task_ids", "task_selection_seed",
                "max_steps", "max_rejections", "max_rollbacks", "max_action_retries", "max_decision_rollbacks",
                "task_target_tracking", "completion_verification", "completion_evidence_read_budget",
                "stop_on_technical_failure", "run_official_evaluation"}, "run")
    _keys(model, {"api_key", "api_key_env", "base_url", "name", "temperature", "generation_seed", "max_output_tokens",
                  "enable_thinking", "sampling"}, "model")
    _keys(neo, {"enabled", "uri", "username", "password", "password_env", "database", "graph_version"}, "neo4j")
    sampling = model.setdefault("sampling", {})
    _keys(sampling, {"top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"}, "model.sampling")
    runtime.setdefault("output_root", "./outputs")
    runtime.setdefault("background", True)
    runtime.setdefault("fail_on_insecure_config_permissions", False)
    for k in ("background", "fail_on_insecure_config_permissions"):
        _bool(runtime[k], f"runtime.{k}")
    defaults = {"mode": "online", "dataset": "test_normal", "train_tasks": 0, "test_tasks": "all",
                "test_task_ids": [], "task_selection_seed": 17, "max_steps": 100, "max_rejections": 5,
                "max_rollbacks": 4, "max_action_retries": 1, "max_decision_rollbacks": 2,
                "task_target_tracking": False, "completion_verification": False,
                "completion_evidence_read_budget": 2,
                "stop_on_technical_failure": True, "run_official_evaluation": True}
    for k, v in defaults.items():
        run.setdefault(k, v)
    if run["mode"] not in {"online", "plan"}:
        raise LaunchError("run.mode must be online or plan")
    try:
        run["dataset"] = normalize_dataset_name(run["dataset"])
    except ValueError as exc:
        raise LaunchError(f"run.dataset: {exc}") from exc
    for k in ("train_tasks", "max_action_retries", "completion_evidence_read_budget"):
        _int(run[k], f"run.{k}")
    if run["train_tasks"] != 0:
        raise LaunchError("Neo4j/advisory modes require run.train_tasks=0")
    _int(run["task_selection_seed"], "run.task_selection_seed", -(2**63))
    for k in ("max_steps", "max_rejections", "max_rollbacks", "max_decision_rollbacks"):
        _int(run[k], f"run.{k}", 1)
    for k in ("stop_on_technical_failure", "run_official_evaluation", "task_target_tracking", "completion_verification"):
        _bool(run[k], f"run.{k}")
    if run["completion_verification"]:
        run["task_target_tracking"] = True
    if not isinstance(run["test_task_ids"], list):
        raise LaunchError("run.test_task_ids must be an array")
    for value in run["test_task_ids"]:
        _string(value, "run.test_task_ids item")
    run["test_task_ids"] = list(dict.fromkeys(run["test_task_ids"]))
    if run["test_tasks"] != "all":
        _int(run["test_tasks"], "run.test_tasks")
    resolve_secret(model, "api_key", "api_key_env", "model.api_key")
    for k in ("api_key", "base_url", "name"):
        value = _string(model.get(k), f"model.{k}")
        if value != value.strip().strip("\"'"):
            raise LaunchError(f"model.{k} cannot have outer quotes or whitespace")
    model.setdefault("temperature", 1.)
    _number(model["temperature"], "model.temperature", 0, 2)
    if model.get("generation_seed") is not None:
        _int(model["generation_seed"], "model.generation_seed")
    model.setdefault("max_output_tokens", 8192)
    _int(model["max_output_tokens"], "model.max_output_tokens", 1)
    model.setdefault("enable_thinking", True)
    _bool(model["enable_thinking"], "model.enable_thinking")
    for k, v in {"top_p": .95, "top_k": 20, "min_p": 0., "presence_penalty": 0., "repetition_penalty": 1.}.items():
        sampling.setdefault(k, v)
    _number(sampling["top_p"], "model.sampling.top_p", 0, 1)
    if sampling["top_p"] == 0:
        raise LaunchError("The backend does not preserve top_p=0; use a positive value")
    _number(sampling["presence_penalty"], "model.sampling.presence_penalty", -2, 2)
    if sampling["top_k"] is not None:
        _int(sampling["top_k"], "model.sampling.top_k", -1)
    if sampling["min_p"] is not None:
        _number(sampling["min_p"], "model.sampling.min_p", 0, 1)
    if sampling["repetition_penalty"] is not None:
        _number(sampling["repetition_penalty"], "model.sampling.repetition_penalty", 0)
        if sampling["repetition_penalty"] == 0:
            raise LaunchError("repetition_penalty must be positive")
    neo.setdefault("enabled", True)
    _bool(neo["enabled"], "neo4j.enabled")
    if neo["enabled"]:
        resolve_secret(neo, "password", "password_env", "neo4j.password")
        for k in ("uri", "username", "password"):
            _string(neo.get(k), f"neo4j.{k}")
        neo.setdefault("database", "neo4j")
        neo.setdefault("graph_version", "neo4j-live-unversioned")
        for k in ("database", "graph_version"):
            _string(neo[k], f"neo4j.{k}")
    return config, hashlib.sha256(raw).hexdigest()


def redact(config):
    safe = json.loads(json.dumps(config))
    safe["model"]["api_key"] = "***REDACTED***"
    if "password" in safe["neo4j"]:
        safe["neo4j"]["password"] = "***REDACTED***"
    return safe


def safe_error(exc, config):
    message = f"{type(exc).__name__}: {exc}"
    for secret in (config.get("model", {}).get("api_key"), config.get("neo4j", {}).get("password")):
        if secret:
            message = message.replace(secret, "***REDACTED***")
    return message


def prepare(config, config_path):
    base = config_path.parent
    runtime, run = config["runtime"], config["run"]
    python = python_path(runtime.get("python_executable"), base)
    appworld = _path(runtime.get("appworld_root"), base)
    output_root = _path(runtime["output_root"], base)
    entry = _path(runtime.get("entry_script", str(ROOT / "run_appworld_online_pilot.py")), base)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise LaunchError(f"Configured Python is missing or not executable: {python}")
    if not (appworld / "src" / "appworld").is_dir():
        raise LaunchError(f"AppWorld source directory is missing: {appworld / 'src' / 'appworld'}")
    if not entry.is_file():
        raise LaunchError(f"Backend entry is missing: {entry}. Keep the existing server entry; "
                          "this launcher does not replace Agent code.")
    try:
        ids = load_dataset_ids(appworld, run["dataset"])
    except (ValueError, OSError) as exc:
        raise LaunchError(str(exc)) from exc
    unknown = set(run["test_task_ids"]) - set(ids)
    if unknown:
        raise LaunchError(f"Specified IDs are not in {run['dataset']}: " + ", ".join(sorted(unknown)))
    count = len(ids) if run["test_tasks"] == "all" else run["test_tasks"]
    if run["test_task_ids"]:
        count = 0
    elif count > len(ids):
        raise LaunchError(f"Requested {count} tasks but {run['dataset']} contains only {len(ids)}")
    elif count == 0:
        raise LaunchError("No tasks selected: set test_tasks or test_task_ids")
    return python, appworld, output_root, entry, count


def child_environment(config, appworld, entry):
    env = os.environ.copy()
    env["APPWORLD_ROOT"] = str(appworld)
    env["PYTHONPATH"] = os.pathsep.join([str(entry.parent), str(appworld / "src")]
                                     + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUNBUFFERED"] = "1"
    if config["neo4j"]["enabled"]:
        for field, suffix in {"uri": "URI", "username": "USER", "password": "PASSWORD",
                              "database": "DATABASE", "graph_version": "GRAPH_VERSION"}.items():
            env["HARNESS_NEO4J_" + suffix] = config["neo4j"][field]
    return env


def backend_args(config, appworld, output, env_file, count):
    run = config["run"]
    args = ["--appworld-root", str(appworld), "--env-file", str(env_file),
            "--neo4j-guidance" if config["neo4j"]["enabled"] else "--advisory-control",
            "--train-tasks", "0", "--test-tasks", str(count), "--dataset-name", run["dataset"]]
    for task_id in run["test_task_ids"]:
        args.extend(["--test-task-id", task_id])
    for k, flag in {"task_selection_seed": "--seed", "max_steps": "--max-steps",
                    "max_rejections": "--max-rejections", "max_rollbacks": "--max-rollbacks",
                    "max_action_retries": "--max-action-retries", "max_decision_rollbacks": "--max-decision-rollbacks"}.items():
        args.extend([flag, str(run[k])])
    if run.get("task_target_tracking"):
        args.append("--task-target-tracking")
    if run.get("completion_verification"):
        args.append("--completion-verification")
        args.extend(["--completion-evidence-read-budget", str(run["completion_evidence_read_budget"])])
    args.extend(["--max-output-tokens", str(config["model"]["max_output_tokens"]), "--output-dir", str(output)])
    if run["stop_on_technical_failure"]:
        args.append("--stop-on-technical-failure")
    if not run["run_official_evaluation"]:
        args.append("--skip-official-evaluation")
    args.append("--plan" if run["mode"] == "plan" else "--confirm-online")
    return args


def check_supported_options(entry, args):
    tree = ast.parse(entry.read_text(encoding="utf-8-sig"))
    supported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            supported.update(a.value for a in node.args if isinstance(a, ast.Constant)
                             and isinstance(a.value, str) and a.value.startswith("--"))
    missing = {a for a in args if a.startswith("--")} - supported
    if missing:
        hint = (". Migrate the updated run_appworld_online_pilot.py as well as the launcher"
                if "--dataset-name" in missing else "")
        raise LaunchError("Backend does not define required options: " + ", ".join(sorted(missing)) + hint)


def model_env_text(config):
    model = config["model"]
    values = {"TAU3_API_KEY": model["api_key"], "TAU3_BASE_URL": model["base_url"],
              "TAU3_MODEL": model["name"], "TAU3_TEMPERATURE": model["temperature"],
              "TAU3_ENABLE_THINKING": str(model["enable_thinking"]).lower()}
    if model.get("generation_seed") is not None:
        values["TAU3_GENERATION_SEED"] = model["generation_seed"]
    for k, v in model["sampling"].items():
        values["TAU3_" + k.upper()] = "auto" if v is None else v
    return "".join(f"{k}={v}\n" for k, v in values.items())


@contextmanager
def private_model_env(config):
    fd, name = tempfile.mkstemp(prefix="appworld-launch-model-", suffix=".env")
    try:
        with os.fdopen(fd, "w+", encoding="utf-8") as stream:
            stream.write(model_env_text(config))
            stream.flush()
            proc_path = Path(f"/proc/self/fd/{stream.fileno()}")
            if sys.platform.startswith("linux") and proc_path.exists():
                # Unlink immediately; even SIGKILL cannot leave a named secret file.
                os.unlink(name)
                yield proc_path
            else:
                yield Path(name)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".launch-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def check_runtime(config, entry, appworld):
    sys.path[:0] = [str(entry.parent), str(appworld / "src")]
    importlib.import_module("pydantic")
    importlib.import_module("appworld")
    if config["neo4j"]["enabled"]:
        provider_type = importlib.import_module("harness.neo4j_guidance").Neo4jGraphGuidanceProvider
        provider = provider_type.from_environment()
        print("Neo4j preflight passed:", json.dumps(provider.index_summary, ensure_ascii=False), flush=True)
    print("Preflight passed: interpreter, dependencies, dataset and backend options. "
          "No model calls or task executions were made.", flush=True)
    ids = load_dataset_ids(appworld, config["run"]["dataset"])
    selected = len(config["run"]["test_task_ids"]) or (
        len(ids) if config["run"]["test_tasks"] == "all" else config["run"]["test_tasks"])
    print(f"Dataset: {config['run']['dataset']} | File: {dataset_path(appworld, config['run']['dataset'])} "
          f"| Available: {len(ids)} | Selected: {selected}", flush=True)


def run_worker(config, sha, args, prepared):
    _, appworld, _, entry, count = prepared
    if args._expected_sha and sha != args._expected_sha:
        raise LaunchError("config.json changed after preparation; refusing mixed configuration")
    os.environ.update(child_environment(config, appworld, entry))
    if args.check:
        check_runtime(config, entry, appworld)
        return 0
    sys.path[:0] = [str(entry.parent), str(appworld / "src")]
    os.environ["HARNESS_RUN_ID"] = args._run_id
    os.environ["HARNESS_LAUNCH_LOG"] = args._launch_log
    if hasattr(signal, "SIGTERM"):
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
    status = 0
    try:
        with private_model_env(config) as env_file:
            sys.argv = [str(entry), *backend_args(config, appworld, args._output, env_file, count)]
            runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as exc:
        status = exc.code if type(exc.code) is int else (0 if exc.code is None else 1)
        if isinstance(exc.code, str):
            print(safe_error(exc, config), file=sys.stderr, flush=True)
    except Exception as exc:
        status = 1
        print("Backend failed: " + safe_error(exc, config), file=sys.stderr, flush=True)
    finally:
        output = Path(args._output)
        if output.is_dir():
            write_json(output / "config_snapshot.json", {"config": redact(config), "source_sha256": sha})
        write_json(Path(args._launch_log).with_suffix(".result.json"), {
            "run_id": args._run_id, "pid": os.getpid(), "exit_code": status,
            "finished_at": datetime.now().astimezone().isoformat(),
            "harness_output": str(output), "config_sha256": sha,
        })
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description="Launch the AppWorld Harness from JSON configuration.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--check", action="store_true", help="Read-only environment and Neo4j preflight")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    for flag in ("--_run-id", "--_output", "--_launch-log", "--_expected-sha"):
        parser.add_argument(flag, default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = {}
    try:
        config_path = args.config.expanduser().resolve()
        config, sha = load_config(config_path)
        prepared = prepare(config, config_path)
        python, appworld, output_root, entry, count = prepared
        check_supported_options(entry, backend_args(config, appworld, output_root / "placeholder", "placeholder.env", count))
        if args._worker:
            return run_worker(config, sha, args, prepared)
        if (os.name == "posix" and stat.S_IMODE(config_path.stat().st_mode) & 0o077
                and (not config["model"].get("api_key_env") or
                     (config["neo4j"]["enabled"] and not config["neo4j"].get("password_env")))):
            if config["runtime"]["fail_on_insecure_config_permissions"]:
                raise LaunchError("Config permissions are insecure; use chmod 600 on config.json")
            print("Warning: config.json contains plaintext secrets; run chmod 600 on it.", file=sys.stderr)
        env = child_environment(config, appworld, entry)
        worker = [str(python), "-u", "-X", "utf8", str(Path(__file__).resolve()),
                  "--config", str(config_path), "--_worker", "--_expected-sha", sha]
        checked = subprocess.run([*worker, "--check"], env=env, cwd=entry.parent)
        if checked.returncode:
            return checked.returncode
        if args.check:
            return 0
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        output = output_root / f"{config['run']['dataset']}_{stamp}_{run_id[-8:]}"
        launches = output_root / "appworld_launches"
        launches.mkdir(parents=True, exist_ok=True)
        log = launches / f"launch_{run_id}.log"
        metadata = {"run_id": run_id, "config_path": str(config_path), "config_sha256": sha,
                    "harness_output": str(output), "log": str(log),
                    "task_count": len(config["run"]["test_task_ids"]) or count,
                    "dataset": config["run"]["dataset"],
                    "dataset_file": str(dataset_path(appworld, config["run"]["dataset"])),
                    "config": redact(config)}
        manifest = launches / f"run_{run_id}.json"
        write_json(manifest, metadata)
        command = [*worker, "--_run-id", run_id, "--_output", str(output), "--_launch-log", str(log)]
        if not config["runtime"]["background"] or config["run"]["mode"] == "plan":
            print(f"Running in foreground; report directory: {output}", flush=True)
            return subprocess.run(command, env=env, cwd=entry.parent).returncode
        kwargs = {"start_new_session": True} if os.name == "posix" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        with log.open("wb") as stream:
            process = subprocess.Popen(command, env=env, cwd=entry.parent, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, close_fds=True, **kwargs)
        metadata["pid"] = process.pid
        write_json(manifest, metadata)
        (launches / "latest_log.txt").write_text(str(log) + "\n", encoding="utf-8")
        (launches / "latest_pid.txt").write_text(str(process.pid) + "\n", encoding="utf-8")
        print(f"Started AppWorld online pilot: {process.pid}\nLog: {log}\nHarness output: {output}\n"
              f"Tasks: {metadata['task_count']}\nStarted means spawned, not completed; check the log and launch result file.")
        return 0
    except Exception as exc:
        print("Stopped: " + safe_error(exc, config), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
