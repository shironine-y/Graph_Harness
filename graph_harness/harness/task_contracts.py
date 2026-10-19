from __future__ import annotations

from dataclasses import dataclass, field
import re


def resource_name(text: str, apps=()) -> str:
    text = re.sub(r"\b(?:my|the|currently)\b", "", text.lower()).strip()
    app = next((name for name in apps if text.startswith(name + " ")), None)
    located = re.fullmatch(r"(.+?) (?:on|in) ([a-z_]+)", text)
    if located and located[2] in apps:
        app, text = located[2], located[1]
        name = re.sub(r"[\s-]+", "_", text)
        return f"{app}.{name}"
    if app:
        text = text[len(app):].strip()
    name = re.sub(r"[\s-]+", "_", text)
    if name in {"wish_list", "wishlist"}:
        name = "wishlist"
    return f"{app}.{name}" if app else name


@dataclass(frozen=True)
class Selector:
    field: str
    comparison: str
    expected: object


@dataclass(frozen=True)
class CollectionBinding:
    origin: str
    destination: str
    predicate: str
    selectors: tuple[Selector, ...] = ()
    remove_from_origin: bool = False


@dataclass
class PublicContract:
    constraints: list[tuple[str, str, object, str | None]] = field(default_factory=list)
    bindings: list[CollectionBinding] = field(default_factory=list)
    preservation: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    unresolved_clauses: list[str] = field(default_factory=list)
    parsed_clauses: int = 0


def sequential_stop_conditions(instruction: str, apps=()):
    """Public, explicit until-clauses; unsupported wording is not guessed."""
    text = " ".join(instruction.split())
    pattern = (r"keep going to (?:the )?(previous|next) ([a-z_]+) on ([a-z_]+) "
               r"until you reach a \2 I have already ([a-z_]+)")
    return [(m[1].lower(), m[2].lower(), m[3].lower())
            for m in re.finditer(pattern, text, re.I) if m[3].lower() in apps]


def parse_public_contract(instruction: str, apps=()) -> PublicContract:
    contract = PublicContract()
    text = " ".join(instruction.split())
    # Split sentences without splitting numeric decimal thresholds.
    clauses = [c.strip(" .;") for c in re.split(
        r";|\.(?=\s+[A-Z]|\s*$)|,?\s+and then\s+|,\s+(?:and\s+)?(?=(?:have|shuffle|play|start|buy|return|remove|move)\b)", text, flags=re.I)
               if c.strip(" .;")]
    for clause in clauses:
        before = len(contract.constraints) + len(contract.bindings) + len(contract.preservation)
        covered = False
        # Conditional and alternative interpretations cannot be safely flattened.
        conditional = bool(re.search(r"\b(?:if|unless|either|instead|except)\b", clause, re.I))
        if not conditional:
            # Relational subsets need two complete, verified source collections;
            # the adjective does not turn an arbitrary returned list into targets.
            match = re.fullmatch(r"remove all (?:the )?([a-z_]+) that I have (not )?([a-z_]+) from (.+)", clause, re.I)
            if match:
                origin = resource_name(match[4], apps)
                app = origin.split(".", 1)[0] + "." if "." in origin else ""
                reference = app + resource_name(match[3] + " " + match[1])
                contract.bindings.append(CollectionBinding(origin, origin, "ENTITY_SET_EXCLUDES",
                    (Selector("$identity", "NOT_IN_COLLECTION" if match[2] else "IN_COLLECTION", reference),)))
                covered = True
            match = re.fullmatch(r"reset (.+?) (?:to be (?:the )?same as|with all of) (.+)", clause, re.I)
            if match:
                def located(value):
                    m = re.fullmatch(r"(.+?) (?:on|in) (?:my )?([a-z_]+)", value, re.I)
                    if m and m[2].lower() in apps:
                        return m[2].lower() + "." + resource_name(m[1])
                    return resource_name(value, apps)
                destination = located(match[1])
                origin = match[2]
                if origin.lower().startswith("its ") and "." in destination:
                    origin = destination.split(".", 1)[0] + "." + resource_name(origin[4:])
                else:
                    origin = located(origin)
                contract.bindings.append(CollectionBinding(origin, destination, "ENTITY_SET_EQUALS"))
                covered = True
            match = re.fullmatch(r"keep going to (?:the )?(?:previous|next) ([a-z_]+) on ([a-z_]+) "
                                 r"until you reach a \1 I have already ([a-z_]+)", clause, re.I)
            if match and match[2].lower() in apps:
                app = match[2].lower()
                contract.bindings.append(CollectionBinding(app + "." + resource_name(match[3] + " " + match[1] + "s"),
                    app + ".current_" + resource_name(match[1]), "ENTITY_SET_INTERSECTS"))
                covered = True
            match = re.fullmatch(
                r"(?:add|copy|transfer|move) (?:all(?: items)?|everything) "
                r"(?:currently )?(?:from|on|in) (.+?) (?:to|into) (.+)", clause, re.I)
            if match and not re.search(r"\b(?:not|never|or|and)\b", clause, re.I):
                contract.bindings.append(CollectionBinding(
                    resource_name(match[1], apps), resource_name(match[2], apps),
                    "ENTITY_SET_CONTAINS", remove_from_origin=clause.lower().startswith("move ")))
                covered = True
            match = re.fullmatch(
                r"(?:make|set) (.+?) (?:to )?contain exactly (?:all(?: items)?|everything) "
                r"(?:currently )?(?:from|on|in) (.+)", clause, re.I)
            if match:
                contract.bindings.append(CollectionBinding(resource_name(match[2], apps),
                    resource_name(match[1], apps), "ENTITY_SET_EQUALS"))
                covered = True
            match = re.fullmatch(
                r"(?:move|copy|add|remove|delete) all (?:items|products|records|songs|files|tasks|notes) "
                r"with ([a-z_ ]+?) (under|over|at least|at most|=|>=|<=|<|>) "
                r"([0-9]+(?:\.[0-9]+)?) (?:from|in) (.+?)(?: (?:to|into) (.+))?", clause, re.I)
            if match:
                comparison = {"under": "LT", "over": "GT", "at least": "GTE", "at most": "LTE",
                              "=": "EQ", ">=": "GTE", "<=": "LTE", "<": "LT", ">": "GT"}[match[2].lower()]
                origin = resource_name(match[4], apps)
                deleting = clause.lower().startswith(("remove ", "delete "))
                if deleting or match[5]:
                    contract.bindings.append(CollectionBinding(origin,
                        origin if deleting else resource_name(match[5], apps),
                        "ENTITY_SET_EXCLUDES" if deleting else "ENTITY_SET_CONTAINS",
                        (Selector(resource_name(match[1]), comparison, match[3]),),
                        remove_from_origin=clause.lower().startswith("move ")))
                    covered = True
            # Natural word order: "products with under 4.2 rating".
            match = re.fullmatch(
                r"(?:move|copy|add|remove|delete) all (?:items|products|records|songs|files|tasks|notes) "
                r"with (under|over|at least|at most) ([0-9]+(?:\.[0-9]+)?) ([a-z_ ]+?) "
                r"(?:from|in) (.+?)(?: (?:to|into) (.+))?", clause, re.I)
            if match:
                origin = resource_name(match[4], apps)
                deleting = clause.lower().startswith(("remove ", "delete "))
                if deleting or match[5]:
                    contract.bindings.append(CollectionBinding(origin,
                        origin if deleting else resource_name(match[5], apps),
                        "ENTITY_SET_EXCLUDES" if deleting else "ENTITY_SET_CONTAINS",
                        (Selector(resource_name(match[3]),
                                  {"under": "LT", "over": "GT", "at least": "GTE", "at most": "LTE"}[match[1].lower()], match[2]),),
                        remove_from_origin=clause.lower().startswith("move ")))
                    covered = True
            match = re.fullmatch(r"(?:remove|delete) (?:all(?: items)?|everything) (?:from|in) (.+)", clause, re.I)
            if match:
                contract.constraints.append(("MUST_NOT_EXIST", resource_name(match[1], apps), None, None))
                covered = True
            match = re.fullmatch(
                r"(move|copy|add|buy|place an order for|make an order for) all ([a-z_ ]+?) "
                r"(?:from|in|on) (.+?)(?: (?:to|into) (.+))?", clause, re.I)
            if match and not covered and not re.search(r"\b(?:with|that|not|or|except|if|and|liked|downloaded)\b", match[2], re.I):
                buying = match[1].lower() in {"buy", "place an order for", "make an order for"}
                if buying or match[4]:
                    contract.bindings.append(CollectionBinding(resource_name(match[3], apps),
                        "completed_purchase_items" if buying else resource_name(match[4], apps),
                        "ENTITY_SET_CONTAINS", (Selector("type", "TYPE_EQ", match[2]),),
                        remove_from_origin=match[1].lower() == "move"))
                    covered = True
            match = re.fullmatch(r"(.+?) must (be empty|contain no items|exist|contain at least one item)", clause, re.I)
            if match:
                predicate = "MUST_NOT_EXIST" if match[2].lower() in {"be empty", "contain no items"} else "MUST_EXIST"
                contract.constraints.append((predicate, resource_name(match[1], apps), None, None))
                covered = True
            match = re.fullmatch(r"(?:update|change) only ([a-z_ ,]+?) (?:in|on|of) (.+?)(?: without changing other fields)?", clause, re.I)
            if match:
                fields = tuple(resource_name(s) for s in re.split(r",|\band\b", match[1]) if s.strip())
                contract.preservation.append((resource_name(match[2], apps), fields))
                # The allowed update itself has no specified expected value.
                covered = False
            match = re.fullmatch(r"(?:return|answer|output)(?: the answer)? (?:as|in) (JSON|a JSON object|a JSON list|a number|a string)", clause, re.I)
            if match:
                kind = {"json": "json", "a json object": "object", "a json list": "list",
                        "a number": "number", "a string": "string"}[match[1].lower()]
                contract.constraints.append(("ANSWER_FORMAT", "$answer", kind, "answer"))
                covered = True
            match = re.fullmatch(r"(?:return|answer|output)(?: the answer)? (?:with )?(?:keys|months|items) in (?:the )?order\s*[:=]\s*(.+)", clause, re.I)
            if match:
                order = tuple(s.strip(" '\"") for s in match[1].split(","))
                if order and all(order):
                    contract.constraints.append(("ANSWER_ORDER", "$answer", order, "answer"))
                    covered = True
            # Declarative field constraints share an explicitly named scope.
            match = re.search(r"\b(?:in|on|of) (?:my|the) ([a-z_ ]+)$", clause, re.I)
            if match and not covered:
                scope = resource_name(match[1], apps)
                body = clause[:match.start()].strip(" ,")
                pieces = [s.strip() for s in re.split(r",|\band\b", body) if s.strip()]
                piece_results = []
                for piece in pieces:
                    m = re.fullmatch(r"(?:each item(?:'s)? )?([a-z_ ]+?)\s*(>=|<=|=|>|<|is|equals|at least|at most)\s*(.+)", piece, re.I)
                    if not m:
                        piece_results.append(False)
                        continue
                    key = resource_name(m[1])
                    numeric = re.fullmatch(r"\$?([0-9]+(?:\.[0-9]+)?)", m[3])
                    op = m[2].lower()
                    if op in {">", "<", ">=", "<=", "at least", "at most"} and not numeric:
                        piece_results.append(False)
                        continue
                    expected = numeric[1] if numeric else m[3].strip(" '\"")
                    if key == "budget":
                        key = "total"
                    pred = {">=": "ATTRIBUTE_GTE", "at least": "ATTRIBUTE_GTE", "<=": "ATTRIBUTE_LTE",
                            "at most": "ATTRIBUTE_LTE", ">": "ATTRIBUTE_GT", "<": "ATTRIBUTE_LT"}.get(op, "FIELD_EQUALS")
                    if key == "quantity":
                        pred = {"FIELD_EQUALS": "QUANTITY_EQUALS", "ATTRIBUTE_GTE": "QUANTITY_AT_LEAST",
                                "ATTRIBUTE_LTE": "QUANTITY_AT_MOST"}.get(pred, pred)
                    elif key == "total":
                        pred = {"FIELD_EQUALS": "TOTAL_EQUALS", "ATTRIBUTE_GTE": "TOTAL_GTE", "ATTRIBUTE_LTE": "TOTAL_LTE"}.get(pred, pred)
                    if pred == "FIELD_EQUALS" and numeric and key not in {"type", "state", "recipient"}:
                        expected = float(numeric[1]) if "." in numeric[1] else int(numeric[1])
                    contract.constraints.append((pred, scope, expected, key))
                    piece_results.append(True)
                covered = bool(piece_results) and all(piece_results)
        # Extract useful purchase constraints even when selection/binding is unknown.
        if not conditional and re.match(r"(?:buy|place an order|make an order)\b", clause, re.I):
            scope = "completed_purchase_items"
            m = re.match(r"(?:buy|place an order for|make an order for) (?:all(?: items)?|everything) "
                         r"(?:currently )?(?:from|on|in) (.+?)(?=,| and |$)", clause, re.I)
            if m:
                contract.bindings.append(CollectionBinding(resource_name(m[1], apps), scope, "ENTITY_SET_CONTAINS"))
                covered = m.end() == len(clause)
            for m in re.finditer(r"at least ([0-9]+(?:\.[0-9]+)?) (product|seller) rating", clause, re.I):
                key = "rating" if m[2].lower() == "product" else "seller_rating"
                contract.constraints.append(("ATTRIBUTE_GTE", scope, m[1], key))
            m = re.match(r"buy (?:me )?(\d+)\s", clause, re.I)
            if m:
                contract.constraints.append(("QUANTITY_SUM_EQUALS", scope, m[1], "quantity"))
            m = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", clause)
            if m:
                contract.constraints.append(("TOTAL_LTE", scope, m[1], "total"))
            # Until the full item selector/optimization is resolved, don't claim
            # the entire task is satisfied just because these fields pass.
        after = len(contract.constraints) + len(contract.bindings) + len(contract.preservation)
        if after > before:
            contract.parsed_clauses += 1
        if not covered:
            contract.unresolved_clauses.append(clause)
    return contract
