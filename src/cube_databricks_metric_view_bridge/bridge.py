# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Stable Cube-owned composition boundary for Metric View conversion."""

from __future__ import annotations

import dataclasses
import re
import warnings
from collections import deque
from typing import Literal

import yaml
from ossie import OSIDocument
from ossie_databricks import convert_ossie_to_metric_view
from ossie_databricks._common import dump_yaml as dump_databricks_yaml
from ossie_databricks._common import load_yaml as load_databricks_yaml
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from .view_projection import convert_cube_view_to_ossie


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

    projected, resolved_source, cube_issues = convert_cube_view_to_ossie(
        files,
        view,
        source=source,
        strict_fanout=strict_fanout,
    )
    document = yaml.safe_load(projected)
    normalized = OSIDocument.model_validate(document).to_osi_yaml()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metric_view_yaml = convert_ossie_to_metric_view(
            normalized,
            source=resolved_source,
        )

    metric_view_yaml, qualified_fields = _qualify_joined_computed_dimensions(
        normalized,
        metric_view_yaml,
        resolved_source,
    )
    unhandled_warnings = [
        str(item.message)
        for item in caught
        if not _is_handled_joined_expression_warning(str(item.message), qualified_fields)
    ]
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
    qualifies every bare SQL column with the converter's deterministic join path.
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
    for dataset in model.get("datasets") or []:
        dataset_name = dataset.get("name")
        qualifier = qualifiers.get(dataset_name)
        if not qualifier:
            continue
        for field in dataset.get("fields") or []:
            field_name = field.get("name")
            if not isinstance(field_name, str):
                continue
            dimension = dimensions.get(field_name.casefold())
            if dimension is None or not isinstance(dimension.get("expr"), str):
                continue
            expression = dimension["expr"]
            if _is_simple_identifier(expression):
                continue
            replacement = _qualify_bare_columns(expression, qualifier)
            if replacement is None:
                continue
            dimension["expr"] = replacement
            qualified.add(field_name.casefold())

    if not qualified:
        return metric_view_yaml, frozenset()
    return dump_databricks_yaml(metric_view), frozenset(qualified)


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
    queue: deque[str] = deque([source])
    while queue:
        parent = queue.popleft()
        for child in adjacency[parent]:
            if child in paths:
                continue
            base = child
            alias = base
            suffix = 2
            while alias in used_aliases:
                alias = f"{base}_{suffix}"
                suffix += 1
            used_aliases.add(alias)
            paths[child] = paths[parent] + (alias,)
            queue.append(child)
    return {name: ".".join(path) for name, path in paths.items() if path}


def _qualify_bare_columns(expression: str, qualifier: str) -> str | None:
    try:
        tree = parse_one(expression, read="databricks")
        if tree is None:
            return None
        prefix = [part for part in qualifier.split(".") if part]
        if not prefix:
            return expression
        for column in list(tree.find_all(exp.Column)):
            if column.table:
                continue
            replacement = parse_one(
                ".".join(prefix + [column.name]),
                read="databricks",
            )
            column.replace(replacement)
        return tree.sql(dialect="databricks")
    except ParseError:
        return None


def _is_simple_identifier(expression: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression.strip()))


def _is_handled_joined_expression_warning(
    message: str,
    qualified_fields: frozenset[str],
) -> bool:
    match = _JOINED_COMPLEX_WARNING.fullmatch(message)
    return bool(match and match.group(1).casefold() in qualified_fields)
