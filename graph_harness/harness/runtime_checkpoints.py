from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping


def _plain(value, remaining, depth=0):
    remaining[0] -= 1
    if remaining[0] < 0 or depth > 16:
        raise ValueError("checkpoint value too large")
    if type(value) in {type(None), bool, int, float, str, bytes}:
        if isinstance(value, (str, bytes)) and len(value) > 1000000:
            raise ValueError("checkpoint value too large")
        return value
    if type(value) in {list, tuple, set, frozenset}:
        return type(value)(_plain(v, remaining, depth + 1) for v in value)
    if type(value) is dict:
        return {_plain(k, remaining, depth + 1): _plain(v, remaining, depth + 1)
                for k, v in value.items()}
    raise ValueError("non-inert runtime value")


class RuntimeCheckpoints:
    def __init__(self, world, snapshot_context=lambda: None, restore_context=lambda _: None):
        self.world = world
        namespace = getattr(getattr(world, "shell", None), "user_ns", {})
        self.framework_names = set(namespace) if isinstance(namespace, Mapping) else set()
        self.snapshot_context = snapshot_context
        self.restore_context = restore_context
        self.slots = {}
        self.last_restore = {}

    def save(self, checkpoint_id, level):
        confirmed = self.world.save_state(checkpoint_id)
        if confirmed != checkpoint_id:
            return confirmed
        namespace = getattr(getattr(self.world, "shell", None), "user_ns", {})
        values, omitted = {}, []
        remaining = [100000]
        for name, value in namespace.items() if isinstance(namespace, Mapping) else ():
            if name in self.framework_names or name.startswith("_"):
                continue
            try:
                values[name] = _plain(value, remaining)
            except (ValueError, TypeError, RecursionError):
                omitted.append(name)
        self.slots[level] = (checkpoint_id, values, omitted, copy.deepcopy(self.snapshot_context()))
        return confirmed

    def restore(self, checkpoint_id):
        saved = next((v for v in self.slots.values() if v[0] == checkpoint_id), None)
        if saved is None:
            raise RuntimeError("Runtime checkpoint is unavailable")
        _, values, omitted, context = saved
        namespace = getattr(getattr(self.world, "shell", None), "user_ns", None)
        removed = 0
        if isinstance(namespace, MutableMapping):
            for name in list(namespace):
                if name not in self.framework_names and not name.startswith("_"):
                    del namespace[name]
                    removed += 1
            # Drop notebook output caches which can retain abandoned objects.
            for name in ("_", "__", "___"):
                namespace.pop(name, None)
            output_cache = namespace.get("_oh")
            if isinstance(output_cache, dict):
                output_cache.clear()
            namespace.update(copy.deepcopy(values))
        self.last_restore = {"restore_level": next(level for level, v in self.slots.items() if v[0] == checkpoint_id),
                             "python_variables_removed": removed,
                             "python_variables_restored": len(values),
                             "unsupported_variables_discarded": len(omitted),
                             "evidence_cache_invalidated": True}
        self.restore_context(copy.deepcopy(context))
