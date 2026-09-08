# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end contract tests for the Cube-owned bridge boundary."""

import dataclasses

import pytest
import yaml
from ossie_cube import ConversionError

from cube_databricks_metric_view_bridge import (
    ConversionResult,
    convert_cube_view_to_databricks_metric_view,
)

_NESTED_MODEL = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - name: customers
        sql: "{CUBE}.customer_id = {customers}.id"
        relationship: many_to_one
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: customer_id, sql: customer_id, type: number}
  - name: customers
    sql_table: main.sales.customers
    joins:
      - name: countries
        sql: "{CUBE}.country_id = {countries}.id"
        relationship: many_to_one
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: country_id, sql: country_id, type: number}
  - name: countries
    sql_table: main.sales.countries
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: name, sql: name, type: string}
      - {name: code, sql: code, type: string}
      - name: display_name
        sql: "CONCAT({name}, ' (', {code}, ')')"
        type: string
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [id]}
      - {join_path: orders.customers, includes: []}
      - join_path: orders.customers.countries
        includes: [display_name]
"""


def _convert(model=_NESTED_MODEL):
    return convert_cube_view_to_databricks_metric_view(
        {"model/views/sales.yml": model},
        "sales",
    )


def test_returns_stable_artifacts_and_source():
    result = _convert()

    assert isinstance(result, ConversionResult)
    assert result.source == "orders"
    assert yaml.safe_load(result.ossie_yaml)["semantic_model"][0]["name"] == "sales"
    metric_view = yaml.safe_load(result.metric_view_yaml)
    assert metric_view["version"] == "1.1"
    assert metric_view["source"] == "main.sales.orders"


def test_nested_computed_dimension_uses_full_metric_view_join_path():
    result = _convert()
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["display_name"] == (
        "CONCAT(customers.countries.name, ' (', customers.countries.code, ')')"
    )
    assert not any("complex expression on a joined table" in item.message for item in result.issues)


def test_joined_expression_preserves_quoted_physical_column_identifiers():
    model = _NESTED_MODEL.replace(
        "      - name: display_name\n"
        "        sql: \"CONCAT({name}, ' (', {code}, ')')\"\n"
        "        type: string",
        "      - name: display_name\n"
        "        sql: \"CASE WHEN `order` > 1 THEN CONCAT(`first name`, ' ', `last name`) "
        'ELSE `x-y` END"\n'
        "        type: string",
    )

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["display_name"] == (
        "CASE WHEN customers.countries.`order` > 1 "
        "THEN CONCAT(customers.countries.`first name`, ' ', "
        "customers.countries.`last name`) ELSE customers.countries.`x-y` END"
    )
    assert not any("complex expression on a joined table" in item.message for item in result.issues)


def test_joined_expression_prefixes_the_complete_physical_column_path():
    model = _NESTED_MODEL.replace(
        "sql: \"CONCAT({name}, ' (', {code}, ')')\"",
        "sql: \"CONCAT(address.city, ' ', address.zip)\"",
    )

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["display_name"] == (
        "CONCAT(customers.countries.address.city, ' ', customers.countries.address.zip)"
    )
    assert not any("complex expression on a joined table" in item.message for item in result.issues)


def test_joined_expression_prefixes_a_root_multipart_physical_column():
    model = _NESTED_MODEL.replace(
        "sql: \"CONCAT({name}, ' (', {code}, ')')\"",
        "sql: address.city",
    )

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["display_name"] == "customers.countries.address.city"
    assert not any("complex expression on a joined table" in item.message for item in result.issues)


def test_already_join_qualified_expression_is_not_prefixed_twice():
    model = _NESTED_MODEL.replace(
        "sql: \"CONCAT({name}, ' (', {code}, ')')\"",
        "sql: \"CONCAT(customers.countries.name, ' (', customers.countries.code, ')')\"",
    )

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["display_name"] == (
        "CONCAT(customers.countries.name, ' (', customers.countries.code, ')')"
    )
    assert not any("complex expression on a joined table" in item.message for item in result.issues)


@pytest.mark.parametrize(
    "expression",
    [
        "(SELECT max(value) FROM items)",
        "transform(values, x -> x + 1)",
        "source.id + value",
    ],
)
def test_unsafe_joined_dimension_expressions_fail_closed(expression):
    model = _NESTED_MODEL.replace(
        "sql: \"CONCAT({name}, ' (', {code}, ')')\"",
        f'sql: "{expression}"',
    )

    with pytest.raises(ConversionError, match="refusing to return a publishable"):
        _convert(model)


def test_non_root_source_override_updates_source_join_direction_and_qualification():
    model = _NESTED_MODEL.replace(
        "- {join_path: orders, includes: [id]}",
        "- {join_path: orders, includes: []}",
    )
    result = convert_cube_view_to_databricks_metric_view(
        {"model/views/sales.yml": model},
        "sales",
        source="customers",
    )
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert result.source == "customers"
    assert metric_view["source"] == "main.sales.customers"
    assert [item["name"] for item in metric_view["joins"]] == ["orders", "countries"]
    assert expressions["display_name"] == "CONCAT(countries.name, ' (', countries.code, ')')"
    assert not any("complex expression on a joined table" in item.message for item in result.issues)


def test_source_override_that_drops_a_selected_member_fails_closed():
    with pytest.raises(ConversionError, match=r"missing=\['id'\]"):
        convert_cube_view_to_databricks_metric_view(
            {"model/views/sales.yml": _NESTED_MODEL},
            "sales",
            source="customers",
        )


def test_behavior_neutral_ossie_warnings_remain_visible_to_caller_policy():
    result = _convert()

    warnings = [item for item in result.issues if item.origin == "ossie_databricks"]
    assert warnings
    assert any("primary_key/unique_keys" in item.message for item in warnings)


def test_unparseable_joined_expression_is_not_silently_claimed_as_fixed():
    model = _NESTED_MODEL.replace(
        "sql: \"CONCAT({name}, ' (', {code}, ')')\"",
        'sql: "value @@ not valid sql"',
    )
    with pytest.raises(ConversionError, match="refusing to return a publishable"):
        _convert(model)


def test_metric_with_nested_sql_bindings_returns_no_publishable_artifact():
    model = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
    measures:
      - name: unsafe_metric
        sql: "(SELECT max(value) FROM items)"
        type: max
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [unsafe_metric]}
"""

    with pytest.raises(ConversionError, match="nested SQL bindings"):
        _convert(model)


def test_conversion_is_deterministic_and_result_is_immutable():
    first = _convert()
    second = _convert()

    assert first == second
    try:
        first.source = "countries"
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ConversionResult must remain immutable")
