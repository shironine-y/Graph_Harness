from __future__ import annotations

import ast


class CheckedActions(list):
    """Keep the longest trace for preflight, and legal traces for runtime matching."""
    def __init__(self, paths):
        unique = []
        for path in paths:
            if tuple(path) not in unique:
                unique.append(tuple(path))
        self.paths = tuple(unique)
        preflight = list(max(self.paths, key=len, default=()))
        # Preflight must cover a write in a shorter alternative branch too.
        for path in self.paths:
            for action in path:
                if action not in preflight:
                    preflight.append(action)
        super().__init__(preflight)

    def resolve(self, calls, matches):
        return next((list(path) for path in self.paths if len(path) == len(calls)
                     and all(matches(a, c) for a, c in zip(path, calls))), None)


def compile_paths(tree, actions_by_site, iteration_count, *, max_calls=5, max_paths=512):
    """Enumerate structural branches, bounded iterations and break/continue paths.

    Conditions are not evaluated. Required straight-line calls cannot disappear.
    Deferred/unsupported control flow is rejected by the caller's static checks.
    Paths contain site IDs, not payloads or expected output values.
    """
    def checked(flows):
        flows = list(dict.fromkeys(flows))
        if len(flows) > max_paths:
            raise ValueError("API control flow has too many paths; split the batch")
        if any(len(path) > max_calls for path, _ in flows):
            raise ValueError(f"At most {max_calls} AppWorld API calls are allowed per step")
        return flows

    def expr(node):
        if node is None:
            return ()
        result = ()
        for child in ast.iter_child_nodes(node):
            result += expr(child)
        if isinstance(node, ast.Call) and (node.lineno, node.col_offset) in actions_by_site:
            result += ((node.lineno, node.col_offset),)
        return result

    def sequence(statements, flows=None):
        flows = [((), None)] if flows is None else flows
        for statement in statements:
            next_flows = []
            for prefix, signal in flows:
                if signal:
                    next_flows.append((prefix, signal))
                else:
                    next_flows.extend((prefix + suffix, sig) for suffix, sig in stmt(statement))
            flows = checked(next_flows)
        return flows

    def stmt(node):
        if isinstance(node, ast.If):
            head = expr(node.test)
            return checked([(head + path, sig) for path, sig in
                            sequence(node.body) + sequence(node.orelse)])
        if isinstance(node, ast.For):
            if not any(isinstance(n, ast.Call) and (n.lineno, n.col_offset) in actions_by_site
                       for n in ast.walk(node)):
                return [((), None)]
            count = iteration_count(node.iter)
            if count is not None and count > max_calls:
                raise ValueError(f"At most {max_calls} AppWorld API calls are allowed per step")
            if count is None or count < 0:
                raise ValueError("AppWorld API loops require a literal collection or constant range, "
                                 "or a statically bounded collection of at most five items")
            live, stopped = [((), None)], []
            body = sequence(node.body)
            for _ in range(count):
                following = []
                for prefix, _ in live:
                    for suffix, signal in body:
                        if signal == "break":
                            stopped.append((prefix + suffix, None))
                        else:
                            following.append((prefix + suffix, None))
                live = checked(following)
                stopped = checked(stopped)
            tail = sequence(node.orelse)
            return checked(stopped + [(p + s, sig) for p, _ in live for s, sig in tail])
        if isinstance(node, ast.Break):
            return [((), "break")]
        if isinstance(node, ast.Continue):
            return [((), "continue")]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return [((), None)]
        return [(expr(node), None)]

    return [path for path, _ in sequence(tree.body)]
