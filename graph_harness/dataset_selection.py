from pathlib import Path
import random
import re


def normalize_dataset_name(value: str) -> str:
    """Accept a dataset basename with or without .txt, never an external path."""
    if not isinstance(value, str):
        raise ValueError("Dataset must be a filename or dataset name")
    name = value[:-4] if value.endswith(".txt") else value
    # AppWorld discovers datasets using the part before the first dot, so
    # dots in the stem cannot be evaluated reliably without changing AppWorld.
    if not re.fullmatch(r"\w[\w-]*", name):
        raise ValueError("Dataset must be a safe basename, not a path: use e.g. test_challenge_25")
    return name


def dataset_path(appworld_root: str | Path, dataset: str) -> Path:
    name = normalize_dataset_name(dataset)
    directory = (Path(appworld_root) / "data" / "datasets").resolve()
    path = (directory / f"{name}.txt").resolve()
    if path.parent != directory:
        raise ValueError("Dataset file must remain inside AppWorld data/datasets")
    if not path.is_file():
        raise FileNotFoundError(f"AppWorld dataset list not found: {path}")
    return path


def load_dataset_ids(appworld_root: str | Path, dataset: str) -> list[str]:
    path = dataset_path(appworld_root, dataset)
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
           if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{path.name} must contain non-empty, unique task IDs")
    return ids


def select_online_tasks(
    appworld_root: str | Path,
    reference_ids: set[str],
    *,
    train_count: int = 3,
    test_count: int = 3,
    seed: int = 17,
    dataset_name: str = "test_normal",
) -> dict[str, list[str]]:
    name = normalize_dataset_name(dataset_name)
    for label, value in (("train_count", train_count), ("test_count", test_count)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    if name == "train" and train_count:
        raise ValueError("Cannot select train twice; use train_count=0 when dataset_name=train")
    rng = random.Random(seed + 1)
    train = ([task_id for task_id in load_dataset_ids(appworld_root, "train")
              if task_id not in reference_ids] if train_count else [])
    test = load_dataset_ids(appworld_root, name)
    if len(train) < train_count or len(test) < test_count:
        raise ValueError(f"Not enough disjoint train/{name} tasks for the requested pilot")
    return {"train": rng.sample(train, train_count), name: rng.sample(test, test_count)}
