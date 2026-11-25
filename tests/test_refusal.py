# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Every refusal names its family and the model change that fixes it."""

import ast
from pathlib import Path

import pytest
from ossie_cube import ConversionError

import cube_databricks_metric_view_bridge.refusal as refusal_module
from cube_databricks_metric_view_bridge import (
    convert_cube_view_to_databricks_metric_view,
    convert_cube_view_to_ossie,
)
from cube_databricks_metric_view_bridge.refusal import Refusal, RefusalCode

_PACKAGE = Path(refusal_module.__file__).parent


def _base_names(node):
    return {getattr(base, "id", None) for base in node.bases}


def _refusal_call_problem(call):
    if len(call.args) != 3 or call.keywords:
        return "takes (RefusalCode.<family>, reason, fix) positionally"
    code, _, fix = call.args
    if not (
        isinstance(code, ast.Attribute)
        and getattr(code.value, "id", None) == "RefusalCode"
        and code.attr in RefusalCode.__members__
    ):
        return "has no RefusalCode family"
    if isinstance(fix, ast.Constant) and not str(fix.value).strip():
        return "has an empty fix"
    return None


def test_every_bridge_refusal_has_a_family_code_and_a_fix():
    problems = []
    for path in sorted(_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        refusal_classes = {"Refusal"} | {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and "Refusal" in _base_names(node)
        }
        for node in ast.walk(tree):
            where = f"{path.name}:{getattr(node, 'lineno', '?')}"
            if (
                isinstance(node, ast.ClassDef)
                and node.name != "Refusal"
                and "ConversionError" in _base_names(node)
            ):
                problems.append(f"{where} {node.name} subclasses ConversionError, not Refusal")
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None)
            if name == "ConversionError":
                problems.append(f"{where} raises ConversionError with no family or fix")
            elif name == "Refusal":
                problem = _refusal_call_problem(node)
                if problem:
                    problems.append(f"{where} Refusal {problem}")
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef) or cls.name not in refusal_classes - {"Refusal"}:
                continue
            for node in ast.walk(cls):
                if (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "__init__"
                    and isinstance(node.func.value, ast.Call)
                    and getattr(node.func.value.func, "id", None) == "super"
                ):
                    problem = _refusal_call_problem(node)
                    if problem:
                        problems.append(f"{path.name}:{node.lineno} {cls.name} {problem}")
    assert problems == []


def test_a_refusal_without_a_fix_sentence_is_rejected():
    with pytest.raises(ValueError, match="fix must be one sentence"):
        Refusal(RefusalCode.MEMBER_UNSUPPORTED, "view 'v': it fails", "")
    with pytest.raises(ValueError, match="fix must be one sentence"):
        Refusal(RefusalCode.MEMBER_UNSUPPORTED, "view 'v': it fails", "exclude it")


def test_a_refusal_keeps_only_the_first_line_of_a_quoted_parser_error():
    refusal = Refusal(
        RefusalCode.MEMBER_SQL_UNSUPPORTED,
        "measure 'm' cannot be parsed: Expecting ). Line 1, Col: 10.\n  SUM(\x1b[4mamount\x1b[0m",
        "Rewrite it.",
    )

    assert str(refusal) == "measure 'm' cannot be parsed: Expecting ). Line 1, Col: 10. Rewrite it."


def test_a_refusal_message_ends_with_its_fix():
    refusal = Refusal(RefusalCode.MEMBER_UNSUPPORTED, "view 'v': it fails.", "Leave it out.")

    assert isinstance(refusal, ConversionError)
    assert str(refusal) == "view 'v': it fails. Leave it out."
    assert (refusal.code, refusal.reason, refusal.fix) == (
        "CUBE_MEMBER_UNSUPPORTED",
        "view 'v': it fails",
        "Leave it out.",
    )


_CUBES = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - name: users
        sql: "{CUBE}.user_id = {users}.id"
        relationship: many_to_one
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: status, sql: status, type: string}
      - {name: country, sql: country, type: string}
      - {name: city, sql: city, type: string}
      - name: created_at
        sql: created_at
        type: time
        granularities:
          - {name: fiscal_year, interval: 1 year, offset: 3 months}
      - {name: top_item, sql: "(SELECT max(sku) FROM main.sales.items)", type: string}
    measures:
      - {name: count, type: count}
      - {name: broken, sql: "SUM(amount", type: number}
    segments:
      - {name: paid_only, sql: "{CUBE}.status = 'paid'"}
    hierarchies:
      - {name: geo, levels: [country, city]}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: name, sql: name, type: string}
"""


@pytest.mark.parametrize(
    ("cubes", "code", "names"),
    [
        (
            "[{join_path: orders, includes: [count]}, {join_path: users, includes: [name]}]",
            "CUBE_VIEW_JOIN_PATH_UNSUPPORTED",
            "view 'probe'",
        ),
        ("[{join_path: orders, includes: [count, geo]}]", "CUBE_MEMBER_UNSUPPORTED", "orders.geo"),
        (
            "[{join_path: orders, includes: [count, paid_only]}]",
            "CUBE_MEMBER_UNSUPPORTED",
            "orders.paid_only",
        ),
        (
            "[{join_path: orders, includes: [count, created_at]}]",
            "CUBE_MEMBER_UNSUPPORTED",
            "orders.created_at",
        ),
        (
            "[{join_path: orders, includes: [count, top_item]}]",
            "CUBE_MEMBER_SQL_UNSUPPORTED",
            "top_item",
        ),
        ("[{join_path: orders, includes: [count, broken]}]", "CUBE_MEMBER_SQL_UNSUPPORTED", "broken"),
    ],
    ids=["multiple roots", "hierarchy", "segment", "custom granularity", "subquery", "unparseable"],
)
def test_refusals_report_their_family_and_a_fix_naming_the_member(cubes, code, names):
    model = _CUBES + f"views:\n  - name: probe\n    cubes: {cubes}\n"

    with pytest.raises(Refusal) as refused:
        convert_cube_view_to_databricks_metric_view({"model.yml": model}, "probe")

    assert refused.value.code == code
    assert "\n" not in str(refused.value)
    assert str(refused.value).endswith(refused.value.fix)
    assert names in refused.value.fix


def test_the_ossie_projection_keeps_a_custom_granularity_the_metric_view_refuses():
    model = _CUBES + (
        "views:\n  - name: probe\n    cubes: [{join_path: orders, includes: [count, created_at]}]\n"
    )

    ossie_yaml = convert_cube_view_to_ossie({"model.yml": model}, "probe")[0]

    assert "fiscal_year" in ossie_yaml
