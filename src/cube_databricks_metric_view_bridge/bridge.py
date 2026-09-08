# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Stable Cube-owned composition boundary for Metric View conversion."""

from __future__ import annotations

import dataclasses
import importlib
import re
from contextvars import ContextVar
from typing import Literal

import yaml
from ossie import OSIDocument
from ossie_cube import ConversionError
from ossie_databricks._common import dump_yaml as dump_databricks_yaml
from ossie_databricks._common import load_yaml as load_databricks_yaml
from sqlglot import Dialect, exp, parse_one
from sqlglot.errors import SqlglotError
from sqlglot.tokens import TokenType

from .view_projection import _convert_cube_view_to_ossie_with_provenance


@dataclasses.dataclass(frozen=True)
class BridgeIssue:
    """One converter diagnostic with origin preserved for caller policy."""

    origin: Literal["ossie_cube", "ossie_databricks"]
    code: str
    message: str
    element: str | None = None


@dataclasses.dataclass(frozen=True)
class ConversionResult:
    """Deterministic intermediate and final artifacts for one Cube view."""

    ossie_yaml: str
    metric_view_yaml: str
    source: str
    issues: tuple[BridgeIssue, ...]


_JOINED_COMPLEX_WARNING = re.compile(
    r"^\[field '([^']+)'\] complex expression on a joined table; "
    r"emitted as-is, verify qualification$"
)

# The pinned converter exposes diagnostics only through its private ``_warn``
# function, which normally delegates to Python's process-global warnings state.
# Route only those converter diagnostics into a call-local sink. Direct uses of
# the upstream converter, and every unrelated warning, continue through the
# original function unchanged.
_databricks_converter = importlib.import_module("ossie_databricks.ossie_to_metric_view")
convert_ossie_to_metric_view = _databricks_converter.convert_ossie_to_metric_view
_original_databricks_warn = _databricks_converter._warn
_converter_warning_sink: ContextVar[list[str] | None] = ContextVar(
    "cube_bridge_databricks_warning_sink",
    default=None,
)


def _route_databricks_warning(scope, message):
    sink = _converter_warning_sink.get()
    if sink is None:
        return _original_databricks_warn(scope, message)
    sink.append(f"[{scope}] {message}")
    return None


_databricks_converter._warn = _route_databricks_warning


def convert_cube_view_to_databricks_metric_view(
    files: dict[str, str],
    view: str,
    *,
    source: str | None = None,
    strict_fanout: bool = True,
) -> ConversionResult:
    """Convert one static Cube YAML view into Metric View YAML 1.1.

    The bridge deliberately exposes its normalized Ossie artifact as part of the
    result. Embedders can validate member preservation without importing private
    implementation details from either Apache Ossie converter.
    """

    projected, resolved_source, cube_issues, provenance = (
        _convert_cube_view_to_ossie_with_provenance(
            files,
            view,
            source=source,
            strict_fanout=strict_fanout,
        )
    )
    document = yaml.safe_load(projected)
    normalized = OSIDocument.model_validate(document).to_osi_yaml()

    caught: list[str] = []
    warning_token = _converter_warning_sink.set(caught)
    try:
        metric_view_yaml = convert_ossie_to_metric_view(
            normalized,
            source=resolved_source,
        )
    finally:
        _converter_warning_sink.reset(warning_token)

    metric_view_yaml, qualified_fields = _qualify_joined_computed_dimensions(
        normalized,
        metric_view_yaml,
        resolved_source,
    )
    metric_view_yaml = _apply_metric_expression_templates(
        normalized,
        metric_view_yaml,
        resolved_source,
        provenance.metric_templates,
        provenance.dataset_markers,
    )
    _validate_public_surface(normalized, metric_view_yaml)
    unhandled_warnings = [
        message
        for message in caught
        if not _is_handled_joined_expression_warning(message, qualified_fields)
    ]
    unsafe_dimensions = [
        message for message in unhandled_warnings if _JOINED_COMPLEX_WARNING.fullmatch(message)
    ]
    if unsafe_dimensions:
        raise ConversionError(
            f"{unsafe_dimensions[0]}; refusing to return a publishable Metric View artifact"
        )
    issues = tuple(
        BridgeIssue(
            origin="ossie_cube",
            code=item.issue_type.value,
            message=item.detail or str(item),
            element=item.element_name,
        )
        for item in cube_issues
    ) + tuple(
        BridgeIssue(
            origin="ossie_databricks",
            code="DATABRICKS_CONVERTER_WARNING",
            message=message,
        )
        for message in unhandled_warnings
    )
    return ConversionResult(
        ossie_yaml=normalized,
        metric_view_yaml=metric_view_yaml,
        source=resolved_source,
        issues=issues,
    )


def _qualify_joined_computed_dimensions(
    ossie_yaml: str,
    metric_view_yaml: str,
    source: str,
) -> tuple[str, frozenset[str]]:
    """Finish a known gap in the pinned Ossie Databricks converter.

    Apache Ossie field expressions are dataset-scoped, while Metric View
    dimensions share one namespace. The pinned converter qualifies a single
    joined column but only warns for a computed expression. This adapter safely
    qualifies computed joined fields with the converter's deterministic join path
    and source fields with Databricks' reserved ``source`` alias.
    """

    document = yaml.safe_load(ossie_yaml)
    model = document["semantic_model"][0]
    metric_view = load_databricks_yaml(metric_view_yaml)
    if not isinstance(metric_view, dict):
        return metric_view_yaml, frozenset()

    qualifiers = _dataset_qualifiers(model, source)
    dimensions = {
        str(item.get("name", "")).casefold(): item
        for item in metric_view.get("dimensions") or []
        if isinstance(item, dict)
    }
    qualified: set[str] = set()
    changed = False
    for dataset in model.get("datasets") or []:
        dataset_name = dataset.get("name")
        is_source = dataset_name == source
        qualifier = "source" if is_source else qualifiers.get(dataset_name)
        if not qualifier:
            continue
        for field in dataset.get("fields") or []:
            field_name = field.get("name")
            if not isinstance(field_name, str):
                continue
            dimension = dimensions.get(field_name.casefold())
            if dimension is None or not isinstance(dimension.get("expr"), str):
                continue
            original_expression = _field_expression(field)
            if not is_source and _is_simple_identifier(original_expression):
                continue
            replacement = _qualify_bare_columns(dimension["expr"], qualifier)
            if replacement is None:
                raise ConversionError(
                    f"[field '{field_name}'] expression on dataset '{dataset_name}' "
                    "cannot be safely qualified; refusing to return a publishable "
                    "Metric View artifact"
                )
            dimension["expr"] = replacement
            changed = True
            if not is_source:
                qualified.add(field_name.casefold())

    if not changed:
        return metric_view_yaml, frozenset()
    return dump_databricks_yaml(metric_view), frozenset(qualified)


def _field_expression(field: dict) -> str:
    expression = field.get("expression")
    if not isinstance(expression, dict):
        return ""
    for dialect in expression.get("dialects") or []:
        if isinstance(dialect, dict) and isinstance(dialect.get("expression"), str):
            return dialect["expression"]
    return ""


def _dataset_qualifiers(model: dict, source: str) -> dict[str, str]:
    datasets = [
        item.get("name")
        for item in model.get("datasets") or []
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    adjacency: dict[str, list[str]] = {name: [] for name in datasets}
    for relationship in model.get("relationships") or []:
        left = relationship.get("from")
        right = relationship.get("to")
        if left not in adjacency or right not in adjacency:
            continue
        adjacency[left].append(right)
        adjacency[right].append(left)

    if source not in adjacency:
        return {}
    paths: dict[str, tuple[str, ...]] = {source: ()}
    used_aliases = {"source"}
    used_normalized_aliases = {"source"}

    def visit(parent):
        for child in adjacency[parent]:
            if child in paths:
                continue
            base = child
            alias = base
            suffix = 2
            while alias in used_aliases:
                alias = f"{base}_{suffix}"
                suffix += 1
            normalized_alias = alias.casefold()
            if normalized_alias in used_normalized_aliases:
                raise ConversionError(
                    f"Databricks join alias '{alias}' collides case-insensitively with "
                    "another join alias or the reserved 'source' alias"
                )
            used_aliases.add(alias)
            used_normalized_aliases.add(normalized_alias)
            paths[child] = paths[parent] + (alias,)
            visit(child)

    visit(source)
    return {name: ".".join(path) for name, path in paths.items() if path}


def _apply_metric_expression_templates(
    ossie_yaml: str,
    metric_view_yaml: str,
    source: str,
    templates: dict[str, str],
    dataset_markers: dict[str, str],
) -> str:
    """Render provenance-bearing metric templates against emitted join aliases."""

    document = yaml.safe_load(ossie_yaml)
    model = document["semantic_model"][0]
    qualifiers = _dataset_qualifiers(model, source)
    metric_view = load_databricks_yaml(metric_view_yaml)
    if not isinstance(metric_view, dict):
        raise ConversionError("Databricks converter returned a non-mapping Metric View")
    measures = {
        str(item.get("name", "")).casefold(): item
        for item in metric_view.get("measures") or []
        if isinstance(item, dict)
    }
    marker_paths = {}
    for dataset, marker in dataset_markers.items():
        if dataset == source:
            marker_paths[marker.casefold()] = ("source",)
        elif dataset in qualifiers:
            marker_paths[marker.casefold()] = tuple(qualifiers[dataset].split("."))
        else:
            raise ConversionError(
                f"dataset '{dataset}' has no emitted Metric View join path from source '{source}'"
            )

    for name, template in templates.items():
        measure = measures.get(name.casefold())
        if measure is None or not isinstance(measure.get("expr"), str):
            raise ConversionError(
                f"Databricks converter dropped selected metric '{name}'; refusing publication"
            )
        measure["expr"] = _render_metric_template(template, marker_paths, name)
    return dump_databricks_yaml(metric_view)


def _render_metric_template(expression, marker_paths, metric_name):
    """Replace only provenance markers while preserving the original SQL text."""

    try:
        tokens = Dialect.get_or_raise("databricks").tokenize(expression)
        tree = parse_one(expression, read="databricks")
        if (
            tree is None
            or tree.find(exp.Query) is not None
            or any(token.token_type is TokenType.ARROW for token in tokens)
        ):
            raise ConversionError(
                f"metric '{metric_name}' contains nested SQL bindings whose dataset "
                "provenance cannot be safely rendered"
            )
        edits = []
        for column in tree.find_all(exp.Column):
            target, parts = _complete_column_path(column)
            if target is None or len(parts) < 2:
                raise ConversionError(
                    f"metric '{metric_name}' contains a column path that cannot be safely rendered"
                )
            path = marker_paths.get(parts[0].name.casefold())
            if path is None:
                raise ConversionError(
                    f"metric '{metric_name}' contains an untracked physical column reference "
                    f"'{target.sql(dialect='databricks')}'; refusing publication"
                )
            start = parts[0].meta.get("start")
            next_start = parts[1].meta.get("start")
            if not isinstance(start, int) or not isinstance(next_start, int):
                raise ConversionError(
                    f"metric '{metric_name}' contains a column whose source position "
                    "cannot be determined"
                )
            rendered_path = _render_identifier_path(path)
            edits.append((start, next_start, f"{rendered_path}." if rendered_path else ""))
        return _apply_text_edits(expression, edits)
    except SqlglotError as error:
        raise ConversionError(
            f"metric '{metric_name}' cannot be safely parsed for publication: {error}"
        ) from error


def _validate_public_surface(ossie_yaml, metric_view_yaml):
    model = yaml.safe_load(ossie_yaml)["semantic_model"][0]
    metric_view = load_databricks_yaml(metric_view_yaml)
    if not isinstance(metric_view, dict):
        raise ConversionError("Databricks converter returned a non-mapping Metric View")

    expected_dimensions = [
        field["name"]
        for dataset in model.get("datasets") or []
        for field in dataset.get("fields") or []
    ]
    expected_measures = [metric["name"] for metric in model.get("metrics") or []]
    actual_dimensions = [item.get("name") for item in metric_view.get("dimensions") or []]
    actual_measures = [item.get("name") for item in metric_view.get("measures") or []]

    for kind, expected, actual in (
        ("dimension", expected_dimensions, actual_dimensions),
        ("measure", expected_measures, actual_measures),
    ):
        expected_normalized = {str(name).casefold() for name in expected}
        actual_normalized = {str(name).casefold() for name in actual}
        if expected_normalized != actual_normalized or len(expected) != len(actual):
            missing = sorted(expected_normalized - actual_normalized)
            unexpected = sorted(actual_normalized - expected_normalized)
            raise ConversionError(
                f"Databricks conversion changed the selected public {kind} surface "
                f"(missing={missing}, unexpected={unexpected}); refusing publication"
            )


def _qualify_bare_columns(
    expression: str,
    qualifier: str,
) -> str | None:
    """Prefix dataset-scoped columns without crossing a SQL binding scope.

    Subqueries and lambdas introduce names that do not belong to the field's
    dataset. Until the bridge performs full scope analysis, refusing those forms
    is safer than rewriting them and suppressing the upstream warning.
    """

    try:
        tokens = Dialect.get_or_raise("databricks").tokenize(expression)
        if any(token.token_type is TokenType.ARROW for token in tokens):
            return None
        tree = parse_one(expression, read="databricks")
        if tree is None or tree.find(exp.Query) is not None:
            return None
        prefix = tuple(part for part in qualifier.split(".") if part)
        if not prefix:
            return expression
        edits = []
        rendered_prefix = _render_identifier_path(prefix)
        for column in tree.find_all(exp.Column):
            _, parts = _complete_column_path(column)
            if not parts:
                return None
            start = parts[0].meta.get("start")
            if not isinstance(start, int):
                return None
            edits.append((start, start, f"{rendered_prefix}."))
        return _apply_text_edits(expression, edits)
    except SqlglotError:
        return None


def _complete_column_path(
    column: exp.Column,
) -> tuple[exp.Expression | None, tuple[exp.Identifier, ...]]:
    """Return the replaceable node and every identifier in a dotted path."""

    if not all(isinstance(part, exp.Identifier) for part in column.parts):
        return None, ()
    target: exp.Expression = column
    parts = list(column.parts)
    while isinstance(target.parent, exp.Dot) and target.parent.this is target:
        outer = target.parent.expression
        if not isinstance(outer, exp.Identifier):
            return None, ()
        parts.append(outer)
        target = target.parent
    return target, tuple(parts)


def _render_identifier_path(parts: tuple[str, ...]) -> str:
    return ".".join(exp.to_identifier(part).sql(dialect="databricks") for part in parts)


def _apply_text_edits(text: str, edits: list[tuple[int, int, str]]) -> str:
    """Apply non-overlapping source edits from right to left."""

    result = text
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result


def _is_simple_identifier(expression: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression.strip()))


def _is_handled_joined_expression_warning(
    message: str,
    qualified_fields: frozenset[str],
) -> bool:
    match = _JOINED_COMPLEX_WARNING.fullmatch(message)
    return bool(match and match.group(1).casefold() in qualified_fields)
