from dataclasses import dataclass
from collections.abc import Mapping
import calendar
import re
from harness.task_contracts import parse_public_contract


MONTHS = {name.lower(): number for names in (calendar.month_name, calendar.month_abbr)
          for number, name in enumerate(names) if name}
MONTH_PATTERN = re.compile(r"\b(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b", re.I)
ORDER_PATTERN = re.compile(
    r"\b(?:reverse[ -]chronological(?:ly)?|chronological(?:ly)?|oldest\s+to\s+newest|"
    r"newest\s+to\s+oldest|in\s+(?:the\s+)?(?:following|given|listed|specified|requested|same|this|that)\s+order|"
    r"in\s+(?:ascending|descending)\s+order|order\s+(?:shown|listed|given|specified))\b", re.I)
UNCERTAIN = re.compile(r"\b(?:if|unless|either|instead|except|maybe)\b|\byou may\b", re.I)
INFO = re.compile(r"(?:please\s+)?(?:what|which|who|where|when|how\s+(?:many|much|long|often))\b|"
    r"(?:please\s+)?(?:tell me|let me know|show me|give me|provide me|return|report|list)\b", re.I)
INFO_BOUNDARY = re.compile(r"(?:^|\band(?:\s+then)?\s+|\bthen\s+|,\s*)(?P<request>"
    + INFO.pattern + r")", re.I)
NO_ANSWER = re.compile(r"(?:please\s+)?(?:do not|don't) (?:return|provide|give)(?: me)? (?:an?|any|the) (?:answer|reply)|"
    r"no (?:answer|reply) (?:is )?(?:needed|required)", re.I)

# Speech-act recognition only: these verbs establish an operation request, not
# its targets, effects, correctness or completion. No application/benchmark ID.
ACTION_HEAD = re.compile(
    r"(?:please\s+)?(?:(?:I want you to|I need you to)\s+)?(?:"
    r"accept|add|arrange|book|buy|cancel|capitalize|change|clear|compress|convert|"
    r"correct|create|decline|delete|disable|download|edit|enable|export|follow|"
    r"format|invite|keep|like|mark|modify|move|normalize|order|organize|pay|play|"
    r"post|purchase|rate|remove|rename|reserve|reset|schedule|send|share|sort|"
    r"start|stop|subscribe|terminate|transfer|unfollow|unlike|update|upload|"
    r"give\s+(?:an?\s+)?(?:\d+(?:\.\d+)?[- ]star\s+)?rating\s+to|"
    r"make\s+(?:an?\s+)?(?:payment requests?|requests?|reservation|booking|appointment)"
    r")\b", re.I)
OUTPUT_SENSITIVE = re.compile(
    r"\b(?:tell|explain|describe|summarize|report|return|provide|show|answer|reply|"
    r"respond|response|confirmation|confirm|calculate|compute|identify|name|recommend|"
    r"retrieve|fetch|get|estimate|evaluate|analyze|analyse|suggest|predict|count|output)\b|"
    r"\b(?:let me know|find out|want to know|need to know|would like to know)\b", re.I)
CONTEXT_HEAD = re.compile(
    r"(?:I|we|he|she|they|it)\s+(?:am|is|are|was|were|have|has|had|paid|went|"
    r"received|sent|need to|want to)\b|I'(?:ve|m)\b|"
    r"(?:the|a|an|some|all|its|these|those)\s+.+?\s+(?:is|are|was|were|has|have|"
    r"had|should|must|correspond|contain|contains)\b|here\s*,", re.I)


@dataclass(frozen=True)
class Clause:
    start: int
    end: int
    text: str
    masked: str


def mask_literals(text):
    chars = list(text)
    # An apostrophe inside a word is not a quote delimiter.
    pattern = r'"(?:\\.|[^"\\])*"|(?<!\w)\'(?:\\.|[^\'\\])*\''
    for match in re.finditer(pattern, text):
        chars[match.start():match.end()] = " " * (match.end() - match.start())
    return "".join(chars)


def clauses(instruction):
    masked = mask_literals(instruction)
    start = 0
    for boundary in list(re.finditer(r";|(?<!\d)[.?!](?=\s|$)", masked)) + [None]:
        end = boundary.start() if boundary else len(instruction)
        left, right = start, end
        while left < right and instruction[left].isspace():
            left += 1
        while right > left and instruction[right - 1].isspace():
            right -= 1
        if left < right:
            yield Clause(left, right, instruction[left:right], masked[left:right])
        start = boundary.end() if boundary else len(instruction)


def complete_action_sources(instruction, api_docs):
    """Bounded operation-only discourse; never infer NONE from query absence alone.

    Every sentence must have a supported imperative or a declarative context
    role; at least one positive operation is required. Unbound output-producing
    language vetoes inference even if it appears after an operation verb.
    Actual no-answer enforcement additionally requires the public API protocol.
    """
    sources = []
    resource_words = {"note", "notes", "file", "files", "folder", "folders", "directory", "directories"}
    for methods in api_docs.values():
        if not isinstance(methods, Mapping):
            continue
        for doc in methods.values():
            if not isinstance(doc, Mapping) or str(doc.get("method", "")).upper() not in {"POST", "PATCH", "DELETE"}:
                continue
            for part in str(doc.get("path", "")).strip("/").split("/"):
                if re.fullmatch(r"[a-z_]+", part):
                    for word in part.split("_"):
                        resource_words.add(word)
                        if word.endswith("s"):
                            resource_words.add(word[:-1])
    if "?" in mask_literals(instruction):
        return []
    for clause in clauses(instruction):
        text = clause.masked.strip()
        if OUTPUT_SENSITIVE.search(text):
            return []
        action = ACTION_HEAD.match(text)
        if action:
            if not clause.text[action.end():].strip():
                return []
            # "Create a poem" can request returned content. Positive container
            # or public resource vocabulary is required for an operation role;
            # listing some known answer artifacts would miss novel artifacts.
            created = re.match(r"(?:please\s+)?create\s+(?:(?:a|an|the|my|new)\s+)*(?P<object>[a-z_]+)\b", text, re.I)
            if created and created["object"].lower() not in resource_words:
                return []
            if re.match(r"(?:please\s+)?(?:send|share)\s+(?:me|us)\b", text, re.I):
                return []
            if re.match(r"(?:please\s+)?start\b", text, re.I) and not re.match(
                    r"(?:please\s+)?start\s+(?:playing|recording|downloading|uploading|"
                    r"(?:(?:a|an|the|my)\s+)?(?:timer|recording|download|upload|playback|service|session|job))\b", text, re.I):
                return []
            sources.append((clause.start, clause.end))
        else:
            if INFO.search(text) or not CONTEXT_HEAD.match(text):
                return []
    return sources


def extract_output_requirements(instruction, api_docs=None):
    requested, none, orders, issues = [], [], [], []
    masked_instruction = mask_literals(instruction)
    # An information-producing action may deliver into storage/a messaging
    # channel instead of the completion answer. Do not collapse these roles.
    alternate_delivery = bool(re.search(
        r"\b(?:into|to|in)\s+(?:(?:a|the|my)\s+)?(?:new\s+)?(?:file|note|document|spreadsheet)\b|"
        r"\b(?:by|via)\s+(?:email|sms|message|notification)\b", masked_instruction, re.I))
    for clause in clauses(instruction):
        text = clause.masked
        if NO_ANSWER.fullmatch(text.strip()):
            none.append((clause.start, clause.end))
            continue
        request = INFO_BOUNDARY.search(text)
        if not request:
            if re.search(r"\b(?:you may|do not|don't|never)\s+(?:return|report|list|tell me|show me)\b", text, re.I):
                issues.append(("ANSWER_PRESENCE", "conditional_or_optional_answer_request", clause.start, clause.end))
                if ORDER_PATTERN.search(text):
                    issues.append(("ANSWER_ORDER", "conditional_or_optional_order", clause.start, clause.end))
            continue
        # Negated/conditional requests are not unconditional answer obligations.
        if UNCERTAIN.search(text) or re.search(r"\b(?:not|never|don't)\b", text[:request.end()], re.I):
            issues.append(("ANSWER_PRESENCE", "conditional_or_optional_answer_request", clause.start, clause.end))
            if ORDER_PATTERN.search(text):
                issues.append(("ANSWER_ORDER", "conditional_or_optional_order", clause.start, clause.end))
            continue
        begin = clause.start + request.start("request")
        request_text = text[request.start("request"):]
        request_verb = request["request"].lower().removeprefix("please ")
        if alternate_delivery:
            issues.append(("ANSWER_PRESENCE", "output_delivery_scope_unbound", begin, clause.end))
            continue
        if request_verb in {"return", "give me", "provide me", "show me"}:
            value_request = bool(re.match(
                r"(?:please\s+)?(?:return|give me|provide me|show me)\s+(?:(?:a|an|the|my)\s+)?"
                r"(?:answer|title|name|count|number|total|summary|details|information|list|JSON|keys|months)\b", request_text, re.I))
            value_request = value_request or bool(re.fullmatch(
                r"(?:please\s+)?(?:return|give me|provide me|show me)\s+(?:(?:a|an|the|my)\s+)?"
                r"(?:[a-z][a-z_-]*\s+){1,3}(?:count|total|title|name)", request_text.strip(), re.I))
            calendar_request = bool(ORDER_PATTERN.search(request_text) and MONTH_PATTERN.search(request_text)
                and re.search(r"\b(?:months?|monthly|expenses?|spending|costs?|amounts?|totals?)\b", request_text, re.I))
            if not value_request and not calendar_request:
                issues.append(("ANSWER_PRESENCE", "information_or_resource_action_role_unbound", begin, clause.end))
                continue
        requested.append((begin, clause.end))
        if not ORDER_PATTERN.search(text):
            continue
        # Order must belong to the output request, not a preceding mutation.
        requested_text = text[request.start("request"):]
        if not ORDER_PATTERN.search(requested_text):
            continue
        matches = [m for m in MONTH_PATTERN.finditer(requested_text)
                   if not (m.group().lower() == "may" and m.group() != "May")]
        if len(matches) < 2:
            continue
        before_months = requested_text[:matches[0].start()]
        after_months = requested_text[matches[-1].end():]
        period_scope = bool(re.search(r"\b(?:months?\s*[:=]?|(?:expenses?|spending|costs?|amounts?|totals?)\s+(?:for|in|during))\s*$", before_months, re.I)
            or re.match(r"\s*(?:expenses?|spending|costs?|amounts?|totals?)\b", after_months, re.I))
        middle = MONTH_PATTERN.sub("", requested_text[matches[0].start():matches[-1].end()])
        named_list = not re.sub(r"\band\b|[,\s]", "", middle, flags=re.I)
        if not period_scope or not named_list:
            issues.append(("ANSWER_ORDER", "calendar_output_role_unbound", begin, clause.end))
            continue
        numbers = [MONTHS[m.group().lower()] for m in matches]
        reverse = bool(re.search(r"reverse[ -]chronological|newest\s+to\s+oldest|descending", requested_text, re.I))
        forward = bool(re.search(r"(?<!reverse )(?<!reverse-)\bchronological|oldest\s+to\s+newest|ascending", requested_text, re.I))
        if len(set(numbers)) != len(numbers) or re.search(r"\b(?:19|20)\d{2}\b", requested_text) or (reverse and forward):
            issues.append(("ANSWER_ORDER", "ambiguous_calendar_scope_or_direction", begin, clause.end))
            continue
        expected = sorted(numbers, reverse=reverse) if reverse or forward else numbers
        orders.append((tuple(calendar.month_name[n] for n in expected), "CALENDAR_MONTH",
                       "calendar_reverse" if reverse else "calendar_ascending" if forward else "instruction_listed",
                       begin, clause.end))
    action_only = complete_action_sources(instruction, api_docs or {}) if not issues and not requested and not none else []
    if not requested and not none and not action_only and not any(x[0] == "ANSWER_PRESENCE" for x in issues):
        issues.append(("ANSWER_PRESENCE", "instruction_answer_intent_unresolved", 0, len(instruction)))
    return {"requested": requested, "none": none, "action_only": action_only, "orders": orders, "issues": issues}


def extract_reorder_scopes(instruction, api_docs):
    """Recognize a pure reorder directive; field names come from public API docs.

    Mixed operations, conditions, negation and missing/multiple resource bindings
    stay unresolved. A sort-key mention is never treated as a field to update.
    """
    drafts, issues = [], []
    parts = list(clauses(instruction))
    for clause in parts:
        text = clause.masked.strip()
        if not re.match(r"(?:please\s+)?(?:only\s+)?(?:reorder|sort)\b", text, re.I):
            continue
        if (len(parts) != 1 or UNCERTAIN.search(text) or re.search(
                r"\b(?:not|never|don't|and|also|then|rename|change|update|edit|delete|add|create|return|report|tell|send|pay|follow)\b", text, re.I)):
            issues.append(("WRITE_FIELDS", "mixed_or_conditional_reorder_scope", clause.start, clause.end))
            continue
        scope_matches = []
        for app, methods in api_docs.items():
            if not isinstance(methods, Mapping):
                continue
            for doc in methods.values():
                if not isinstance(doc, Mapping) or str(doc.get("method", "")).upper() != "PATCH":
                    continue
                pieces = str(doc.get("path", "")).strip("/").split("/")
                if len(pieces) != 3 or pieces[0] != app or not re.fullmatch(r"\{[a-z_]+\}", pieces[2]):
                    continue
                resource = pieces[1].replace("_", " ")
                target = rf"(?:my|the)\s+{re.escape(app)}\s+{re.escape(resource)}|{re.escape(app)}\s+{re.escape(resource)}"
                match = re.fullmatch(r"(?:please\s+)?(?:only\s+)?(?:reorder|sort)\s+(?:" + target +
                    r")(?P<tail>(?:\s+by\s+.+|\s+in\s+(?:ascending|descending)\s+order)?)", text, re.I)
                if match:
                    scope_matches.append((app + "." + pieces[1], doc, match["tail"].strip()))
        scopes = {scope for scope, _, _ in scope_matches}
        if len(scopes) != 1:
            issues.append(("WRITE_FIELDS", "reorder_resource_scope_unbound", clause.start, clause.end))
            continue
        scope = next(iter(scopes))
        tail = scope_matches[0][2]
        if tail:
            source_app = scope.split(".")[0]
            fields_in_source = set()
            for detail in api_docs[source_app].values():
                if isinstance(detail, Mapping) and detail.get("method") == "GET" and detail.get("path") == scope_matches[0][1].get("path"):
                    schemas = detail.get("response_schemas")
                    schema = schemas.get("success") if isinstance(schemas, Mapping) else None
                    if isinstance(schema, Mapping):
                        fields_in_source.update(str(k).replace("_", " ") for k in schema)
            criterion = "|".join(re.escape(k) for k in sorted(fields_in_source, key=len, reverse=True))
            pure_tail = bool(re.fullmatch(r"in (?:ascending|descending) order", tail, re.I)
                or re.fullmatch(r"by moving .+ (?:before|after) .+", tail, re.I)
                or (criterion and re.fullmatch(r"by (?:" + criterion + r")(?: in (?:ascending|descending) order)?", tail, re.I)))
            if not pure_tail:
                issues.append(("WRITE_FIELDS", "reorder_operation_effect_unresolved", clause.start, clause.end))
                continue
        # This is a semantic API adapter: require an explicit documented updated
        # ordering meaning; a familiar-looking parameter name is insufficient.
        fields = set()
        for _, doc, _ in scope_matches:
            for p in doc.get("parameters") or ():
                if not isinstance(p, Mapping):
                    continue
                description = str(p.get("description", "")).lower().replace("_", " ")
                if re.search(r"\bupdated (?:order index|ordering index|sort order|display order)\b", description):
                    fields.add(p.get("name"))
        if len(fields) != 1 or not all(isinstance(f, str) for f in fields):
            issues.append(("WRITE_FIELDS", "reorder_field_meaning_unbound", clause.start, clause.end))
            continue
        drafts.append((scope, tuple(fields), clause.start, clause.end))
    return drafts, issues


def month_label(label):
    """Normalize whole labels only; never search arbitrary answer text/values."""
    if type(label) is not str:
        return None
    value = MONTHS.get(label.strip().lower())
    return calendar.month_name[value] if value is not None else None
