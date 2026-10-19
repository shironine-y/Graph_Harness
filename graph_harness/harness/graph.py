import base64
from collections import defaultdict
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Literal
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import zlib

from pydantic import BaseModel, Field


SCHEMA_VERSION = "1.0"
EXPECTED_FIELDS = (
    "taskType", "authenticated", "sourceDirs", "actor", "targetNum",
    "archives", "necessity", "pre_condition", "traceIDs", "count",
    "p_transition", "p_success", "p_coverage", "recoverable",
)


class FieldValue(BaseModel):
    raw: str | None = None
    status: Literal["KNOWN", "EXPLICIT_EMPTY", "PLACEHOLDER", "UNOBSERVED", "UNKNOWN"]
    source_element_id: str | None = None


class GraphElement(BaseModel):
    element_id: str
    kind: Literal["EVENT", "TASK_GOAL", "EVENT_MARKER", "ATTRIBUTE_SNAPSHOT",
                  "RELATION_ANNOTATION", "TEXT", "EDGE"]
    label: str
    raw_value: str
    position: dict[str, float] = Field(default_factory=dict)
    source_cell: str | None = None
    target_cell: str | None = None
    event_id: str | None = None
    fields: dict[str, FieldValue] = Field(default_factory=dict)


class EventRelation(BaseModel):
    element_id: str
    source_event: str
    target_event: str
    relation_type: str = "precedes"
    annotation_ids: list[str] = Field(default_factory=list)
    observed_statistics: list[dict[str, str | int | None]] = Field(default_factory=list)
    status: Literal["BOUND", "UNRESOLVED"] = "BOUND"


class EventGraphSpec(BaseModel):
    schema_version: str = SCHEMA_VERSION
    graph_version: str
    domain: str = "filesystem_demo"
    benchmark: str = "drawio_reference"
    task_family: str = "archive_then_delete"
    source_sha256: str
    diagram_id: str
    elements: list[GraphElement]
    events: dict[str, str]
    relations: list[EventRelation]
    warnings: list[str]
    data_status: str = "reference_only"
    model_version: str = "not_recorded"
    prompt_version: str = "not_recorded"
    tool_version: str = "not_recorded"
    harness_version: str = "1.0"


class ExecutionTransition(BaseModel):
    state_id: str
    action_id: str
    next_state_id: str
    source_event: str | None = None
    target_event: str | None = None
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    side_effect_level: str
    compensation_plan: str
    terminal_outcome: str | None = None
    data_status: str = "NORMATIVE_MOCK_MAPPING_NOT_REFERENCE_PROBABILITY"


class ExecutionGraphSpec(BaseModel):
    schema_version: str = SCHEMA_VERSION
    graph_version: str
    source_graph_version: str
    domain: str = "filesystem_demo"
    benchmark: str = "drawio_reference"
    task_family: str = "archive_then_delete"
    state_predicates: dict[str, FieldValue]
    goal_predicates: list[str]
    required_action_sequence: list[str]
    side_effect_levels: dict[str, str]
    compensation_plan: dict[str, str]
    terminal_outcomes: dict[str, str]
    unresolved_relations: list[str]
    action_ids: dict[str, str] = Field(default_factory=dict)
    preconditions: dict[str, list[str]] = Field(default_factory=dict)
    postconditions: dict[str, list[str]] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    transitions: list[ExecutionTransition] = Field(default_factory=list)
    executable: bool = False
    data_status: str = "UNRESOLVED"


class _MarkupText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ("br", "div", "p"):
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def _plain(value: str) -> str:
    parser = _MarkupText()
    parser.feed(value)
    return "\n".join(line.strip() for line in "".join(parser.parts).splitlines()
                     if line.strip())


def _field(raw: str | None, element_id: str | None) -> FieldValue:
    if raw is None:
        return FieldValue(status="UNOBSERVED")
    stripped = raw.strip()
    if not stripped:
        status = "EXPLICIT_EMPTY"
    elif stripped.lower() in {"xxxx", "xxx", "n", "none", "null", "...", "......"}:
        status = "PLACEHOLDER"
    elif "\ufffd" in stripped:
        status = "UNKNOWN"
    else:
        status = "KNOWN"
    return FieldValue(raw=raw, status=status, source_element_id=element_id)


def _fields(label: str, element_id: str) -> dict[str, FieldValue]:
    values = {}
    for line in label.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = _field(value, element_id)
    return {key: values.get(key, _field(None, None)) for key in
            (*EXPECTED_FIELDS, *sorted(set(values) - set(EXPECTED_FIELDS)))}


def _diagram_root(data: bytes) -> tuple[str, ET.Element]:
    root = ET.fromstring(data)
    diagram = root.find("diagram") if root.tag == "mxfile" else None
    if diagram is None:
        raise ValueError("Expected a draw.io mxfile with a diagram.")
    if diagram.find("mxGraphModel") is not None:
        return diagram.get("id", "unknown"), diagram.find("mxGraphModel")
    if not diagram.text:
        raise ValueError("Empty draw.io diagram.")
    compressed = base64.b64decode(diagram.text, validate=True)
    decoded = zlib.decompress(compressed, -15).decode("utf-8")
    return diagram.get("id", "unknown"), ET.fromstring(unquote(decoded))


def _position(cell: ET.Element) -> dict[str, float]:
    geometry = cell.find("mxGeometry")
    if geometry is None:
        return {}
    return {key: float(geometry.get(key)) for key in ("x", "y", "width", "height")
            if geometry.get(key) is not None}


def _marker_inside(marker: GraphElement, body: GraphElement) -> bool:
    m, b = marker.position, body.position
    if not all(key in m and key in b for key in ("x", "y", "width", "height")):
        return False
    x, y = m["x"] + m["width"] / 2, m["y"] + m["height"] / 2
    return b["x"] <= x <= b["x"] + b["width"] and b["y"] <= y <= b["y"] + b["height"]


class DrawioGraphAdapter:
    """Keep every source element and bind annotations only by explicit event IDs."""

    def import_file(self, path: str | Path) -> tuple[EventGraphSpec, ExecutionGraphSpec]:
        data = Path(path).read_bytes()
        diagram_id, graph = _diagram_root(data)
        cells = [cell for cell in graph.iter("mxCell") if cell.get("id")]
        elements: list[GraphElement] = []
        warnings: list[str] = []
        for cell in cells:
            value = cell.get("value", "")
            label = _plain(value)
            fields = _fields(label, cell.get("id")) if "=" in label else {}
            if cell.get("edge") == "1":
                kind = "EDGE"
            elif re.fullmatch(r"E\d+", label):
                kind = "EVENT_MARKER"
            elif fields.get("sourceEvent") and fields.get("targetEvent"):
                kind = "RELATION_ANNOTATION"
            elif fields:
                kind = "ATTRIBUTE_SNAPSHOT"
            elif "rounded=1" in cell.get("style", ""):
                kind = "EVENT"
            else:
                kind = "TEXT"
            elements.append(GraphElement(
                element_id=cell.get("id"), kind=kind, label=label,
                raw_value=value, position=_position(cell),
                source_cell=cell.get("source"), target_cell=cell.get("target"),
                fields=fields,
            ))
            if "\ufffd" in label:
                warnings.append(f"Damaged source label at {cell.get('id')}; raw value retained.")

        markers = [item for item in elements if item.kind == "EVENT_MARKER"]
        bodies = [item for item in elements if item.kind == "EVENT"]
        events: dict[str, str] = {}
        for marker in markers:
            matches = [body for body in bodies if _marker_inside(marker, body)]
            if len(matches) != 1:
                warnings.append(f"UNRESOLVED event marker {marker.label}: {len(matches)} bodies.")
                continue
            body = matches[0]
            if marker.label in events:
                warnings.append(f"UNRESOLVED duplicate event marker {marker.label}.")
                continue
            events[marker.label] = body.element_id
            body.event_id = marker.label
            marker.event_id = marker.label
            if marker.label == "E1":
                body.kind = "TASK_GOAL"

        by_cell = {cell: event for event, cell in events.items()}
        relations = []
        for element in elements:
            if element.kind != "EDGE":
                continue
            source = by_cell.get(element.source_cell)
            target = by_cell.get(element.target_cell)
            if source and target:
                relations.append(EventRelation(
                    element_id=element.element_id, source_event=source,
                    target_event=target,
                ))
            elif source or target:
                warnings.append(f"UNRESOLVED endpoint on edge {element.element_id}.")

        by_pair: dict[tuple[str, str], list[EventRelation]] = defaultdict(list)
        for relation in relations:
            by_pair[(relation.source_event, relation.target_event)].append(relation)
        for element in elements:
            if element.kind != "RELATION_ANNOTATION":
                continue
            source_field = element.fields["sourceEvent"]
            target_field = element.fields["targetEvent"]
            relation = element.fields.get("relationType")
            if (source_field.status != "KNOWN" or target_field.status != "KNOWN" or
                    relation is None or relation.status != "KNOWN"):
                warnings.append(f"UNRESOLVED annotation {element.element_id}: "
                                "missing source/target/relation type.")
                continue
            source = source_field.raw
            target = target_field.raw
            matches = by_pair.get((source, target), [])
            if len(matches) != 1:
                warnings.append(f"UNRESOLVED annotation {element.element_id}: "
                                f"{source}->{target} matches {len(matches)} edges.")
                continue
            if matches[0].annotation_ids:
                matches[0].status = "UNRESOLVED"
                warnings.append(f"UNRESOLVED duplicate annotation for edge "
                                f"{matches[0].element_id}.")
                continue
            matches[0].annotation_ids.append(element.element_id)
            if matches[0].relation_type != "precedes" and matches[0].relation_type != relation.raw:
                warnings.append(f"UNRESOLVED conflicting relation type for {element.element_id}.")
                matches[0].status = "UNRESOLVED"
            else:
                matches[0].relation_type = relation.raw
            for raw_name, kind in (
                ("p_transition", "OBSERVED_BRANCH_FREQUENCY"),
                ("p_success", "DIRECT_HISTORICAL_FINAL_OUTCOME_UNCALIBRATED"),
                ("p_coverage", "SUCCESS_PATH_COVERAGE"),
                ("recoverable", "HISTORICAL_RECOVERABILITY_UNCALIBRATED"),
            ):
                statistic = element.fields.get(raw_name)
                if statistic and statistic.status == "KNOWN":
                    matches[0].observed_statistics.append({
                        "probability_kind": kind,
                        "probability_value": statistic.raw,
                        "denominator": None,
                        "unique_task_count": None,
                        "source_element_id": element.element_id,
                        "data_status": "demo_only_not_gate_eligible",
                    })

        if any(r.source_event == "E8" and r.target_event == "E10" for r in relations):
            warnings.append("E8->E10 bypasses E9 poststate verification; never executable as success.")
        for element in elements:
            if element.kind == "RELATION_ANNOTATION":
                for key in ("p_transition", "p_success", "recoverable"):
                    field = element.fields.get(key)
                    if field and field.status == "KNOWN":
                        warnings.append(f"{key} at {element.element_id} is demo_only; "
                                        "not calibrated or gate-eligible.")
        sha = hashlib.sha256(data).hexdigest()
        event_graph = EventGraphSpec(
            graph_version=f"drawio-{sha[:12]}", source_sha256=sha,
            diagram_id=diagram_id, elements=elements, events=events,
            relations=relations, warnings=list(dict.fromkeys(warnings)),
        )
        execution_graph = ExecutionGraphSpec(
            graph_version=f"reference-execution-{sha[:12]}",
            source_graph_version=event_graph.graph_version,
            state_predicates={name: _field(None, None) for name in
                              ("authorized", "archive_verified", "poststate_verified")},
            goal_predicates=["archive_exists", "source_absent", "poststate_verified"],
            required_action_sequence=["ARCHIVE", "VERIFY_ARCHIVE", "DELETE_SOURCE",
                                      "VERIFY_POSTSTATE"],
            side_effect_levels={"ARCHIVE": "REVERSIBLE", "VERIFY_ARCHIVE": "READ_ONLY",
                                "DELETE_SOURCE": "IRREVERSIBLE",
                                "VERIFY_POSTSTATE": "READ_ONLY"},
            compensation_plan={"ARCHIVE": "UNDO_ARCHIVE", "DELETE_SOURCE": "NONE"},
            terminal_outcomes={"E10": "SUCCESS_UNVERIFIED_IN_REFERENCE"},
            unresolved_relations=[warning for warning in warnings if "UNRESOLVED" in warning],
            action_ids={action: f"mock-only:{action}" for action in
                        ("ARCHIVE", "VERIFY_ARCHIVE", "DELETE_SOURCE", "VERIFY_POSTSTATE")},
            preconditions={"DELETE_SOURCE": ["explicit_authorization", "path_pair_matched",
                                                 "archive_readable", "archive_checksum_valid",
                                                 "backup_or_compensation_available"]},
            postconditions={"ARCHIVE": ["archive_exists"],
                            "VERIFY_ARCHIVE": ["archive_checksum_valid"],
                            "DELETE_SOURCE": ["source_absent"],
                            "VERIFY_POSTSTATE": ["source_absent", "archive_checksum_valid"]},
            evidence_refs=[sha],
            transitions=[
                ExecutionTransition(state_id=state, action_id=action,
                                    next_state_id=target, source_event=source_event,
                                    target_event=target_event,
                                    preconditions=pre, postconditions=post,
                                    side_effect_level=side, compensation_plan=comp,
                                    terminal_outcome=terminal)
                for state, action, target, source_event, target_event, pre, post, side,
                    comp, terminal in (
                        ("START", "INSPECT", "READY", "E4", "E5", [],
                         ["source_inspected"], "READ_ONLY", "NONE", None),
                        ("READY", "ARCHIVE", "ARCHIVED", "E5", "E8",
                         ["explicit_authorization", "path_scope_valid"],
                         ["archive_exists"], "REVERSIBLE", "UNDO_ARCHIVE", None),
                        ("ARCHIVED", "VERIFY_ARCHIVE", "VERIFIED", "E8", "E9",
                         ["archive_exists"], ["archive_checksum_valid"],
                         "READ_ONLY", "NONE", None),
                        ("VERIFIED", "DELETE_SOURCE", "DELETED", "E8", "E9",
                         ["archive_checksum_valid", "backup_available"],
                         ["source_absent"], "IRREVERSIBLE", "NONE", None),
                        ("DELETED", "VERIFY_POSTSTATE", "SUCCESS", "E9", "E10",
                         ["source_absent"], ["poststate_verified"],
                         "READ_ONLY", "NONE", "SUCCESS"),
                    )
            ],
        )
        return event_graph, execution_graph
