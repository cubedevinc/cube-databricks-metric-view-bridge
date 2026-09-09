# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Publication projection of a Cube view into Apache Ossie."""

import pytest
from _util import by_name, expr_of, model_of, parse
from ossie_cube import ConversionError, IssueType, convert_cube_to_ossie

import cube_databricks_metric_view_bridge.view_projection as view_projection
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


def test_bare_count_uses_primary_key_physical_sql():
    text = """
cubes:
  - name: orders
    sql_table: samples.tpch.orders
    dimensions:
      - {name: id, sql: o_orderkey, type: number, primary_key: true}
    measures:
      - {name: count, type: count}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [count]}
"""

    out, _, _ = _project(text)
    assert expr_of(by_name(model_of(out)["metrics"])["count"]) == (
        "COUNT(DISTINCT orders.o_orderkey)"
    )

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert by_name(metric_view["measures"])["count"]["expr"] == (
        "COUNT(DISTINCT source.o_orderkey)"
    )


def test_bare_count_resolves_every_composite_primary_key_dimension():
    text = """
cubes:
  - name: lines
    sql_table: samples.tpch.lineitem
    dimensions:
      - {name: order_id, sql: l_orderkey, type: number, primary_key: true}
      - {name: line_id, sql: l_linenumber, type: number, primary_key: true}
    measures:
      - {name: count, type: count}
views:
  - name: sales
    cubes:
      - {join_path: lines, includes: [count]}
"""

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert by_name(metric_view["measures"])["count"]["expr"] == (
        "COUNT(DISTINCT source.l_orderkey, source.l_linenumber)"
    )


def test_filtered_composite_primary_key_count_filters_each_tuple_operand():
    text = """
cubes:
  - name: lines
    sql_table: samples.tpch.lineitem
    dimensions:
      - {name: order_id, sql: l_orderkey, type: number, primary_key: true}
      - {name: line_id, sql: l_linenumber, type: number, primary_key: true}
      - {name: active, sql: is_active, type: boolean}
    measures:
      - name: active_count
        type: count
        filters:
          - sql: "{active} = true"
views:
  - name: sales
    cubes:
      - {join_path: lines, includes: [active_count]}
"""

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert by_name(metric_view["measures"])["active_count"]["expr"] == (
        "COUNT(DISTINCT CASE WHEN (source.is_active = true) THEN source.l_orderkey END, "
        "CASE WHEN (source.is_active = true) THEN source.l_linenumber END)"
    )


@pytest.mark.parametrize("hidden_dependency", [False, True])
def test_bare_count_primary_key_dependencies_add_required_joins(hidden_dependency):
    selected = "published" if hidden_dependency else "count"
    calculated = (
        '      - {name: published, sql: "{count}", type: number}\n'
        if hidden_dependency
        else ""
    )
    text = f"""
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {{name: users, sql: "{{CUBE}}.user_id = {{users}}.id", relationship: many_to_one}}
    dimensions:
      - name: external_id
        sql: "{{users}}.external_id"
        type: number
        primary_key: true
    measures:
      - {{name: count, type: count}}
{calculated}  - name: users
    sql_table: main.sales.users
    dimensions:
      - {{name: id, sql: id, type: number, primary_key: true}}
views:
  - name: sales
    cubes:
      - {{join_path: orders, includes: [{selected}]}}
"""

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)

    assert [join["name"] for join in metric_view["joins"]] == ["users"]
    assert by_name(metric_view["measures"])[selected]["expr"] == (
        "COUNT(DISTINCT users.external_id)"
    )


@pytest.mark.parametrize("hidden_dependency", [False, True])
def test_bare_count_primary_key_dependencies_are_validated(hidden_dependency):
    selected = "published" if hidden_dependency else "count"
    calculated = (
        '      - {name: published, sql: "{count}", type: number}\n'
        if hidden_dependency
        else ""
    )
    text = f"""
cubes:
  - name: orders
    sql_table: main.sales.orders
    dimensions:
      - name: id
        sql: "{{missing_id}}"
        type: number
        primary_key: true
    measures:
      - {{name: count, type: count}}
{calculated}views:
  - name: sales
    cubes:
      - {{join_path: orders, includes: [{selected}]}}
"""

    with pytest.raises(ConversionError, match="missing_id.*does not match"):
        _project(text)


def test_bare_count_preserves_a_recorded_physical_primary_key():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    meta:
      ossie:
        primary_key: [id]
    dimensions:
      - {name: id, sql: surrogate_id, type: number}
    measures:
      - {name: count, type: count}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [count]}
"""

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert by_name(metric_view["measures"])["count"]["expr"] == (
        "COUNT(DISTINCT source.id)"
    )


def test_hidden_computed_dimensions_are_inlined_in_fields_and_metrics():
    out, _, _ = _project()
    model = model_of(out)
    customer_name = by_name(by_name(model["datasets"])["users"]["fields"])["customer_name"]
    metric = by_name(model["metrics"])["average_value"]

    assert expr_of(customer_name) == "CONCAT(first_name, ' ', last_name)"
    assert "orders.gross" in expr_of(metric)
    assert "orders.discount" in expr_of(metric)
    assert "orders.net_amount" not in expr_of(metric)


def test_case_dimensions_render_sql_labels_with_source_and_join_provenance():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: user_id, sql: user_id, type: number}
      - {name: status, sql: status_code, type: string}
      - {name: preferred_label, sql: preferred_label, type: string}
      - {name: default_label, sql: default_label, type: string}
      - name: source_segment
        type: string
        case:
          when:
            - sql: "{status} = 'vip'"
              label: {sql: "{preferred_label}"}
          else:
            label: {sql: "{default_label}"}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: tier, sql: tier_code, type: string}
      - {name: preferred_label, sql: preferred_label, type: string}
      - {name: default_label, sql: default_label, type: string}
      - name: joined_segment
        type: string
        case:
          when:
            - sql: "{tier} = 'vip'"
              label: {sql: "{preferred_label}"}
          else:
            label: {sql: "{default_label}"}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [source_segment]}
      - {join_path: orders.users, includes: [joined_segment]}
"""

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    dimensions = by_name(metric_view["dimensions"])

    assert dimensions["source_segment"]["expr"] == (
        "CASE WHEN source.status_code = 'vip' THEN source.preferred_label "
        "ELSE source.default_label END"
    )
    assert dimensions["joined_segment"]["expr"] == (
        "CASE WHEN users.tier_code = 'vip' THEN users.preferred_label "
        "ELSE users.default_label END"
    )


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


@pytest.mark.parametrize("yaml_value", ["''", "0", "false", "null"])
@pytest.mark.parametrize("alias_kind", ["cube", "member"])
def test_falsy_or_non_string_aliases_are_rejected(yaml_value, alias_kind):
    target = "alias: customer" if alias_kind == "cube" else "alias: name"
    text = _MODEL.replace(target, f"alias: {yaml_value}")

    with pytest.raises(ConversionError, match="alias.*non-empty string"):
        _project(text)


def test_unknown_exclusion_is_rejected_before_wildcard_publication():
    text = _MODEL.replace(
        "includes:\n          - status\n          - average_value",
        "includes: '*'\n        excludes: [average_vlaue]",
    )

    with pytest.raises(ConversionError, match="excluded member.*does not exist"):
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


@pytest.mark.parametrize("hidden_dependency", [False, True])
def test_rolling_window_measures_are_rejected(hidden_dependency):
    selected = "published" if hidden_dependency else "rolling_revenue"
    calculated = (
        '      - {name: published, sql: "{rolling_revenue}", type: number}\n'
        if hidden_dependency
        else ""
    )
    text = f"""
cubes:
  - name: orders
    sql_table: main.sales.orders
    measures:
      - name: rolling_revenue
        sql: amount
        type: sum
        rolling_window:
          trailing: 7 day
{calculated}views:
  - name: sales
    cubes:
      - {{join_path: orders, includes: [{selected}]}}
"""

    with pytest.raises(ConversionError, match="rolling_window"):
        _project(text)


@pytest.mark.parametrize("hidden_dependency", [False, True])
@pytest.mark.parametrize(
    ("property_name", "property_value"),
    [
        ("multi_stage", "true"),
        ("group_by", "[status]"),
        ("reduce_by", "[status]"),
        ("add_group_by", "[status]"),
        (
            "time_shift",
            "[{time_dimension: created_at, interval: 1 day, type: prior}]",
        ),
        ("grain", "{include: [status]}"),
    ],
)
def test_multi_stage_measure_semantics_are_rejected(
    hidden_dependency,
    property_name,
    property_value,
):
    selected = "published" if hidden_dependency else "staged_revenue"
    calculated = (
        '      - {name: published, sql: "{staged_revenue}", type: number}\n'
        if hidden_dependency
        else ""
    )
    text = f"""
cubes:
  - name: orders
    sql_table: main.sales.orders
    measures:
      - name: staged_revenue
        sql: amount
        type: sum
        {property_name}: {property_value}
{calculated}views:
  - name: sales
    cubes:
      - {{join_path: orders, includes: [{selected}]}}
"""

    with pytest.raises(ConversionError, match=property_name):
        _project(text)


@pytest.mark.parametrize(
    ("property_name", "property_value"),
    [
        ("multi_stage", "false"),
        ("group_by", "[]"),
        ("reduce_by", "[]"),
        ("add_group_by", "[]"),
        ("time_shift", "[]"),
        ("grain", "{}"),
    ],
)
def test_noop_multi_stage_defaults_remain_static(property_name, property_value):
    text = f"""
cubes:
  - name: orders
    sql_table: main.sales.orders
    measures:
      - name: revenue
        sql: amount
        type: sum
        {property_name}: {property_value}
views:
  - name: sales
    cubes:
      - {{join_path: orders, includes: [revenue]}}
"""

    out, _, _ = _project(text)
    assert set(by_name(model_of(out)["metrics"])) == {"revenue"}


@pytest.mark.parametrize(
    ("property_name", "property_value"),
    [
        ("multi_stage", "'false'"),
        ("group_by", "{}"),
        ("reduce_by", "false"),
        ("add_group_by", "status"),
        ("time_shift", "{}"),
        ("grain", "[]"),
    ],
)
def test_malformed_multi_stage_defaults_are_rejected(property_name, property_value):
    text = f"""
cubes:
  - name: orders
    sql_table: main.sales.orders
    measures:
      - name: revenue
        sql: amount
        type: sum
        {property_name}: {property_value}
views:
  - name: sales
    cubes:
      - {{join_path: orders, includes: [revenue]}}
"""

    with pytest.raises(ConversionError, match=f"malformed {property_name}"):
        _project(text)


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


def test_hidden_unsafe_measure_dependency_is_retained_and_blocks_strict_publication():
    text = _MODEL.replace(
        '      - name: average_value\n        sql: "{revenue} / {count}"\n        type: number',
        "      - name: average_value\n"
        '        sql: "{revenue} / {count}"\n'
        "        type: number\n"
        "      - name: risky_value\n"
        '        sql: "{users.lifetime_value} / {count}"\n'
        "        type: number",
    ).replace(
        "          - status\n          - average_value",
        "          - status\n          - risky_value",
    )

    with pytest.raises(ConversionError, match="FANOUT_UNSAFE_METRIC"):
        _project(text)

    _, _, issues = _project(text, strict_fanout=False)
    unsafe_elements = {item.element_name for item in issues.of_type(IssueType.FANOUT_UNSAFE_METRIC)}
    assert unsafe_elements == {"orders.risky_value", "users.lifetime_value"}


def test_hidden_transitive_cross_cube_dependencies_add_their_cubes_and_joins():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - name: users
        sql: "{CUBE}.user_id = {users}.id"
        relationship: many_to_one
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: user_id, sql: user_id, type: number}
    measures:
      - name: customer_value
        sql: "{users.account_value}"
        type: number
  - name: users
    sql_table: main.sales.users
    joins:
      - name: accounts
        sql: "{CUBE}.account_id = {accounts}.id"
        relationship: many_to_one
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: account_id, sql: account_id, type: number}
    measures:
      - name: account_value
        sql: "{accounts.max_lifetime_value}"
        type: number
  - name: accounts
    sql_table: main.sales.accounts
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
    measures:
      - name: max_lifetime_value
        sql: ltv
        type: max
views:
  - name: sales
    cubes:
      - join_path: orders
        includes: [customer_value]
"""

    out, source, issues = _project(text)
    model = model_of(out)

    assert source == "orders"
    assert not list(issues)
    assert set(by_name(model["datasets"])) == {"orders", "users", "accounts"}
    assert [(item["from"], item["to"]) for item in model["relationships"]] == [
        ("orders", "users"),
        ("users", "accounts"),
    ]
    assert set(by_name(model["metrics"])) == {"customer_value"}
    assert expr_of(by_name(model["metrics"])["customer_value"]) == "MAX(accounts.ltv)"

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert metric_view["source"] == "main.sales.orders"
    assert metric_view["joins"][0]["name"] == "users"
    assert metric_view["joins"][0]["joins"][0]["name"] == "accounts"
    assert by_name(metric_view["measures"])["customer_value"]["expr"] == ("MAX(users.accounts.ltv)")


def test_measure_dimension_dependencies_add_transitive_cubes_and_joins():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - name: users
        sql: "{CUBE}.user_id = {users}.id"
        relationship: many_to_one
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: user_id, sql: user_id, type: number}
    measures:
      - name: max_adjusted_score
        sql: "{users.adjusted_score}"
        type: max
  - name: users
    sql_table: main.sales.users
    joins:
      - name: profiles
        sql: "{CUBE}.profile_id = {profiles}.id"
        relationship: many_to_one
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: profile_id, sql: profile_id, type: number}
      - {name: multiplier, sql: multiplier, type: number}
      - name: adjusted_score
        sql: "{profiles.base_score} * {multiplier}"
        type: number
  - name: profiles
    sql_table: main.sales.profiles
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: base_score, sql: base_score, type: number}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [max_adjusted_score]}
"""

    out, source, issues = _project(text)
    model = model_of(out)

    assert source == "orders"
    assert not list(issues)
    assert set(by_name(model["datasets"])) == {"orders", "users", "profiles"}
    assert [(item["from"], item["to"]) for item in model["relationships"]] == [
        ("orders", "users"),
        ("users", "profiles"),
    ]
    assert expr_of(by_name(model["metrics"])["max_adjusted_score"]) == (
        "MAX(profiles.base_score * users.multiplier)"
    )

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert metric_view["joins"][0]["name"] == "users"
    assert metric_view["joins"][0]["joins"][0]["name"] == "profiles"
    assert by_name(metric_view["measures"])["max_adjusted_score"]["expr"] == (
        "MAX(users.profiles.base_score * users.multiplier)"
    )


def test_raw_source_column_is_not_inlined_as_same_named_computed_dimension():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: gross, sql: gross, type: number}
      - {name: discount, sql: discount, type: number}
      - {name: amount, sql: "{gross} - {discount}", type: number}
    measures:
      - {name: raw_amount, sql: "{CUBE}.amount", type: sum}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [raw_amount]}
"""

    out, _, _ = _project(text)
    assert expr_of(by_name(model_of(out)["metrics"])["raw_amount"]) == "SUM(orders.amount)"

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert by_name(metric_view["measures"])["raw_amount"]["expr"] == "SUM(source.amount)"


def test_provenance_preserves_exact_aggregate_sql():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
    measures:
      - name: median_amount
        sql: PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)
        type: number
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [median_amount]}
"""

    out, _, _ = _project(text)
    assert expr_of(by_name(model_of(out)["metrics"])["median_amount"]) == (
        "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY orders.amount)"
    )

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert by_name(metric_view["measures"])["median_amount"]["expr"] == (
        "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY source.amount)"
    )


def test_lone_measure_reference_is_not_shadowed_by_same_named_cube():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
    measures:
      - {name: users, sql: amount, type: sum}
      - {name: total, sql: "{users}", type: number}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [total]}
"""

    out, _, _ = _project(text)
    assert expr_of(by_name(model_of(out)["metrics"])["total"]) == "SUM(orders.amount)"

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert by_name(metric_view["measures"])["total"]["expr"] == "SUM(source.amount)"


def test_local_dimension_reference_is_not_shadowed_by_same_named_cube():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
    dimensions:
      - {name: users, sql: customer_name, type: string}
      - {name: display_name, sql: "UPPER({users})", type: string}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [display_name]}
      - {join_path: orders.users, includes: []}
"""

    out, _, _ = _project(text)
    orders = by_name(model_of(out)["datasets"])["orders"]

    assert expr_of(by_name(orders["fields"])["display_name"]) == "UPPER(customer_name)"


def test_raw_physical_path_is_not_rewritten_as_a_nested_dataset_reference():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
    measures:
      - {name: struct_total, sql: accounts.balance, type: max}
      - {name: joined_total, sql: "{accounts}.balance", type: max}
  - name: users
    sql_table: main.sales.users
    joins:
      - {name: accounts, sql: "{CUBE}.account_id = {accounts}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: account_id, sql: account_id, type: number}
  - name: accounts
    sql_table: main.sales.accounts
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [struct_total, joined_total]}
      - {join_path: orders.users.accounts, includes: []}
"""

    out, _, _ = _project(text)
    assert expr_of(by_name(model_of(out)["metrics"])["struct_total"]) == (
        "MAX(orders.accounts.balance)"
    )
    assert expr_of(by_name(model_of(out)["metrics"])["joined_total"]) == ("MAX(accounts.balance)")

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert by_name(metric_view["measures"])["struct_total"]["expr"] == (
        "MAX(source.accounts.balance)"
    )
    assert by_name(metric_view["measures"])["joined_total"]["expr"] == (
        "MAX(users.accounts.balance)"
    )


def test_physical_path_cannot_impersonate_a_provenance_marker_by_case():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    measures:
      - name: max_external_value
        sql: __CUBE_DMV_DATASET_0__.value
        type: max
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [max_external_value]}
"""

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)

    assert by_name(metric_view["measures"])["max_external_value"]["expr"] == (
        "MAX(source.__CUBE_DMV_DATASET_0__.value)"
    )


def test_raw_joined_column_adds_its_hidden_dependency_join():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
    measures:
      - {name: max_ltv, sql: "{users}.ltv", type: max}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [max_ltv]}
"""

    out, _, _ = _project(text)
    model = model_of(out)
    assert set(by_name(model["datasets"])) == {"orders", "users"}
    assert [(item["from"], item["to"]) for item in model["relationships"]] == [("orders", "users")]

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert metric_view["joins"][0]["name"] == "users"
    assert by_name(metric_view["measures"])["max_ltv"]["expr"] == "MAX(users.ltv)"


def test_qualified_reference_to_missing_known_cube_member_is_rejected():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
    measures:
      - {name: misspelled_name, sql: "{users.nmae}", type: max}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: name, sql: name, type: string}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [misspelled_name]}
"""

    with pytest.raises(ConversionError, match="does not match a dimension or measure"):
        _project(text)


def test_qualified_reference_to_unknown_cube_is_rejected():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    measures:
      - {name: misspelled_cube, sql: "{usres.name}", type: max}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [misspelled_cube]}
"""

    with pytest.raises(ConversionError, match="unknown cube qualifier 'usres'"):
        _project(text)


def test_bare_reference_to_missing_local_member_is_rejected():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    measures:
      - {name: revenue, sql: amount, type: sum}
      - {name: misspelled_revenue, sql: "{revene}", type: number}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [misspelled_revenue]}
"""

    with pytest.raises(ConversionError, match=r"reference '\{revene\}'.*does not match"):
        _project(text)


def test_raw_joined_column_form_remains_allowed_without_member_metadata():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
    measures:
      - {name: max_external_score, sql: "{users}.external_score", type: max}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [max_external_score]}
"""

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)

    assert by_name(metric_view["measures"])["max_external_score"]["expr"] == (
        "MAX(users.external_score)"
    )


def test_hidden_dependency_dataset_can_be_selected_as_source():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
    measures:
      - {name: max_ltv, sql: "{users}.ltv", type: max}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [max_ltv]}
"""

    out, source, _ = _project(text, source="users")
    assert source == "users"
    assert set(by_name(model_of(out)["datasets"])) == {"orders", "users"}

    result = convert_cube_view_to_databricks_metric_view(
        {"model.yml": text},
        "sales",
        source="users",
    )
    metric_view = parse(result.metric_view_yaml)
    assert metric_view["source"] == "main.sales.users"
    assert metric_view["joins"][0]["name"] == "orders"
    assert by_name(metric_view["measures"])["max_ltv"]["expr"] == "MAX(source.ltv)"


@pytest.mark.parametrize("expression", ["{users.name}", "{users}.name"])
def test_selected_dimension_cannot_read_an_implicit_joined_dataset(expression):
    text = f"""
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {{name: users, sql: "{{CUBE}}.user_id = {{users}}.id", relationship: many_to_one}}
    dimensions:
      - name: user_name
        sql: "{expression}"
        type: string
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {{name: id, sql: id, type: number, primary_key: true}}
      - {{name: name, sql: name, type: string}}
views:
  - name: sales
    cubes:
      - {{join_path: orders, includes: [user_name]}}
"""

    with pytest.raises(ConversionError, match="dataset-scoped Ossie field"):
        _project(text)


def test_selected_dimension_transitively_validates_joined_dependencies():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
    dimensions:
      - {name: user_name, sql: "UPPER({inner_name})", type: string}
      - {name: inner_name, sql: "{users.name}", type: string}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: name, sql: name, type: string}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [user_name]}
"""

    with pytest.raises(ConversionError, match="dataset-scoped Ossie field"):
        _project(text)


def test_join_alias_qualification_matches_databricks_depth_first_assignment():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: a, sql: "{CUBE}.a_id = {a}.id", relationship: many_to_one}
      - {name: source_2, sql: "{CUBE}.s2_id = {source_2}.id", relationship: many_to_one}
  - name: a
    sql_table: main.sales.a
    joins:
      - {name: source, sql: "{CUBE}.s_id = {source}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
  - name: source
    sql_table: main.sales.source_table
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: display, sql: "UPPER(name)", type: string}
  - name: source_2
    sql_table: main.sales.source_2_table
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders.a.source, includes: [display]}
      - {join_path: orders.source_2, includes: []}
"""

    result = convert_cube_view_to_databricks_metric_view({"model.yml": text}, "sales")
    metric_view = parse(result.metric_view_yaml)
    assert metric_view["joins"][0]["name"] == "a"
    assert metric_view["joins"][0]["joins"][0]["name"] == "source_2"
    assert metric_view["joins"][1]["name"] == "source_2_2"
    assert by_name(metric_view["dimensions"])["display"]["expr"] == ("UPPER(a.source_2.name)")


def test_implicit_dimension_dependency_rejects_ambiguous_join_paths():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
      - {name: accounts, sql: "{CUBE}.account_id = {accounts}.id", relationship: many_to_one}
    measures:
      - {name: value, sql: "{accounts.adjusted_value}", type: max}
  - name: users
    sql_table: main.sales.users
    joins:
      - {name: accounts, sql: "{CUBE}.account_id = {accounts}.id", relationship: many_to_one}
  - name: accounts
    sql_table: main.sales.accounts
    dimensions:
      - {name: adjusted_value, sql: value * 2, type: number}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [value]}
"""

    with pytest.raises(ConversionError, match="multiple declared join paths"):
        _project(text)


def test_implicit_measure_dependency_rejects_ambiguous_join_paths():
    text = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
      - {name: accounts, sql: "{CUBE}.account_id = {accounts}.id", relationship: many_to_one}
    measures:
      - {name: value, sql: "{accounts.max_value}", type: number}
  - name: users
    sql_table: main.sales.users
    joins:
      - {name: accounts, sql: "{CUBE}.account_id = {accounts}.id", relationship: many_to_one}
  - name: accounts
    sql_table: main.sales.accounts
    measures:
      - {name: max_value, sql: value, type: max}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [value]}
"""

    with pytest.raises(ConversionError, match="multiple declared join paths"):
        _project(text)


def test_unreachable_dependency_path_search_is_bounded(monkeypatch):
    connected = [f"cube_{index}" for index in range(9)]
    cubes = {
        name: {
            "joins": [{"name": other} for other in connected if other != name],
        }
        for name in connected
    }
    cubes["unreachable"] = {"joins": []}
    calls = 0
    original = view_projection._as_named_list

    def bounded_named_list(value, label):
        nonlocal calls
        calls += 1
        if calls > len(cubes):
            raise AssertionError("path search revisited the dense unreachable component")
        return original(value, label)

    monkeypatch.setattr(view_projection, "_as_named_list", bounded_named_list)

    assert view_projection._declared_join_paths(cubes, connected[0], "unreachable") == []
    assert calls == len(cubes)


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
