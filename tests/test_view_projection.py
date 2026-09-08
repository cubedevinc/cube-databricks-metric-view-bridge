# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Publication projection of a Cube view into Apache Ossie."""

import pytest
from _util import by_name, expr_of, model_of, parse
from ossie_cube import ConversionError, IssueType, convert_cube_to_ossie

from cube_databricks_metric_view_bridge import (
    convert_cube_view_to_databricks_metric_view,
    convert_cube_view_to_ossie,
)

_MODEL = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - name: users
        sql: "{CUBE}.user_id = {users}.id"
        relationship: many_to_one
    dimensions:
      - name: id
        sql: id
        type: number
        primary_key: true
      - name: user_id
        sql: user_id
        type: number
      - name: status
        sql: status
        type: string
        title: Source status
      - name: gross
        sql: gross
        type: number
      - name: discount
        sql: discount
        type: number
      - name: net_amount
        sql: "{gross} - {discount}"
        type: number
    measures:
      - name: count
        type: count
      - name: revenue
        sql: "{net_amount}"
        type: sum
      - name: average_value
        sql: "{revenue} / {count}"
        type: number
  - name: users
    sql_table: main.sales.users
    dimensions:
      - name: id
        sql: id
        type: number
        primary_key: true
      - name: first_name
        sql: first_name
        type: string
      - name: last_name
        sql: last_name
        type: string
      - name: full_name
        sql: "CONCAT({first_name}, ' ', {last_name})"
        type: string
      - name: city
        sql: city
        type: string
    measures:
      - name: lifetime_value
        sql: ltv
        type: sum
views:
  - name: sales
    description: Curated sales surface
    meta:
      ai_context: Prefer this view for sales questions.
    cubes:
      - join_path: orders
        includes:
          - status
          - average_value
      - join_path: orders.users
        alias: customer
        prefix: true
        includes:
          - name: full_name
            alias: name
            title: Customer name
            description: Full customer name
            meta:
              ai_context: Use this instead of separate name parts.
"""


def _project(text=_MODEL, **kwargs):
    return convert_cube_view_to_ossie({"model.yml": text}, "sales", **kwargs)


def test_projects_exact_view_surface_and_resolves_source():
    out, source, issues = _project()
    model = model_of(out)

    assert source == "orders"
    assert not list(issues)
    assert model["name"] == "sales"
    assert model["description"] == "Curated sales surface"
    assert model["ai_context"]["instructions"].startswith("Prefer this view")
    datasets = by_name(model["datasets"])
    assert set(by_name(datasets["orders"].get("fields"))) == {"status"}
    assert set(by_name(datasets["users"].get("fields"))) == {"customer_name"}
    assert set(by_name(model["metrics"])) == {"average_value"}


def test_lossless_import_remains_unprojected():
    out, _ = convert_cube_to_ossie({"model.yml": _MODEL}, view="sales")
    model = model_of(out)
    datasets = by_name(model["datasets"])
    assert {"id", "user_id", "status", "gross", "discount", "net_amount"} <= set(
        by_name(datasets["orders"]["fields"])
    )
    assert {"count", "revenue", "average_value", "lifetime_value"} <= set(by_name(model["metrics"]))


def test_member_alias_prefix_and_metadata_override_are_effective():
    out, _, _ = _project()
    users = by_name(model_of(out)["datasets"])["users"]
    field = by_name(users["fields"])["customer_name"]

    assert field["label"] == "Customer name"
    assert field["description"] == "Full customer name"
    assert field["ai_context"]["instructions"].startswith("Use this instead")


def test_hidden_calculated_measure_dependencies_are_inlined():
    out, _, _ = _project()
    metrics = by_name(model_of(out)["metrics"])
    expr = expr_of(metrics["average_value"])

    assert "SUM(" in expr
    assert "COUNT(DISTINCT orders.id)" in expr
    assert "revenue" not in expr
    assert "count" not in set(by_name(model_of(out)["metrics"]))


def test_hidden_computed_dimensions_are_inlined_in_fields_and_metrics():
    out, _, _ = _project()
    model = model_of(out)
    customer_name = by_name(by_name(model["datasets"])["users"]["fields"])["customer_name"]
    metric = by_name(model["metrics"])["average_value"]

    assert expr_of(customer_name) == "CONCAT(first_name, ' ', last_name)"
    assert "orders.gross" in expr_of(metric)
    assert "orders.discount" in expr_of(metric)
    assert "orders.net_amount" not in expr_of(metric)


def test_wildcard_and_excludes_follow_cube_view_semantics():
    text = _MODEL.replace(
        "includes:\n          - status\n          - average_value",
        "includes: '*'\n        excludes:\n          - id\n          - user_id\n          - gross\n          - discount\n          - net_amount\n          - count\n          - revenue",
    ).replace(
        "includes:\n          - name: full_name\n            alias: name\n            title: Customer name\n            description: Full customer name\n            meta:\n              ai_context: Use this instead of separate name parts.",
        "includes: '*'\n        excludes:\n          - id\n          - first_name\n          - last_name\n          - lifetime_value",
    )
    out, _, _ = _project(text)
    model = model_of(out)

    assert set(by_name(by_name(model["datasets"])["orders"]["fields"])) == {"status"}
    assert set(by_name(by_name(model["datasets"])["users"]["fields"])) == {
        "customer_full_name",
        "customer_city",
    }
    assert set(by_name(model["metrics"])) == {"average_value"}


def test_source_override_is_returned_and_validated():
    _, source, _ = _project(source="users")
    assert source == "users"
    with pytest.raises(ConversionError, match="not in the projected datasets"):
        _project(source="payments")


@pytest.mark.parametrize(
    "edit, message",
    [
        (("- status\n          - average_value", "- missing"), "does not exist"),
        (("- status\n          - average_value", "- id"), "collides"),
    ],
)
def test_invalid_or_colliding_members_are_rejected(edit, message):
    old, new = edit
    text = _MODEL.replace(old, new)
    if message == "collides":
        text = text.replace(
            "- name: full_name\n            alias: name", "- name: full_name\n            alias: id"
        )
        text = text.replace("prefix: true", "prefix: false")
    with pytest.raises(ConversionError, match=message):
        _project(text)


def test_case_insensitive_output_collision_is_rejected():
    text = _MODEL.replace(
        "- name: full_name\n            alias: name",
        "- name: full_name\n            alias: STATUS",
    ).replace("prefix: true", "prefix: false")
    with pytest.raises(ConversionError, match="case-insensitive"):
        _project(text)


def test_missing_join_in_join_path_is_rejected():
    text = _MODEL.replace("joins:\n      - name: users", "joins_disabled:\n      - name: users")
    with pytest.raises(ConversionError, match="declares no join"):
        _project(text)


def test_multi_root_view_is_rejected():
    text = _MODEL.replace("join_path: orders.users", "join_path: users")
    with pytest.raises(ConversionError, match="multiple join roots"):
        _project(text)


def test_selected_geo_segment_and_split_projection_are_rejected():
    geo = _MODEL.replace(
        "- name: status\n        sql: status\n        type: string",
        "- name: status\n        type: geo\n        latitude: { sql: lat }\n        longitude: { sql: lon }",
    )
    with pytest.raises(ConversionError, match="uses geo"):
        _project(geo)

    segment = _MODEL.replace(
        "measures:\n      - name: count",
        "segments:\n      - name: active\n        sql: active = true\n    measures:\n      - name: count",
    ).replace("- status\n          - average_value", "- active")
    with pytest.raises(ConversionError, match="selected segment"):
        _project(segment)

    split = _MODEL.replace("- join_path: orders\n", "- join_path: orders\n        split: true\n")
    with pytest.raises(ConversionError, match="split view projections"):
        _project(split)


def test_unselected_fanout_metric_does_not_block_publication():
    out, _, issues = _project()
    assert "lifetime_value" not in set(by_name(model_of(out).get("metrics")))
    assert not issues.of_type(IssueType.FANOUT_UNSAFE_METRIC)


def test_selected_fanout_metric_is_strict_by_default_and_reported_when_relaxed():
    text = _MODEL.replace(
        "- name: full_name\n            alias: name\n            title: Customer name\n            description: Full customer name\n            meta:\n              ai_context: Use this instead of separate name parts.",
        "- lifetime_value",
    )
    with pytest.raises(ConversionError, match="FANOUT_UNSAFE_METRIC"):
        _project(text)

    out, _, issues = _project(text, strict_fanout=False)
    assert "customer_lifetime_value" in by_name(model_of(out)["metrics"])
    assert len(issues.of_type(IssueType.FANOUT_UNSAFE_METRIC)) == 1


def test_projection_converts_to_databricks_metric_view_without_member_collisions():
    result = convert_cube_view_to_databricks_metric_view(
        {"model.yml": _MODEL},
        "sales",
    )
    metric_view = parse(result.metric_view_yaml)

    assert metric_view["version"] == "1.1"
    assert metric_view["source"] == "main.sales.orders"
    assert [dimension["name"] for dimension in metric_view["dimensions"]] == [
        "status",
        "customer_name",
    ]
    assert metric_view["dimensions"][1]["expr"] == (
        "CONCAT(users.first_name, ' ', users.last_name)"
    )
    assert [measure["name"] for measure in metric_view["measures"]] == ["average_value"]
    assert "revenue" not in metric_view["measures"][0]["expr"]
