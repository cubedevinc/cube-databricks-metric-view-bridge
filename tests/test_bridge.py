# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end contract tests for the Cube-owned bridge boundary."""

import dataclasses
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml
from ossie_cube import ConversionError

import cube_databricks_metric_view_bridge.bridge as bridge_module
from cube_databricks_metric_view_bridge import (
    ConversionResult,
    DatasetSourceResolution,
    convert_cube_view_to_databricks_metric_view,
)
from cube_databricks_metric_view_bridge.bridge import _dataset_qualifiers, _qualify_table_source

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


def _convert(model=_NESTED_MODEL, **kwargs):
    return convert_cube_view_to_databricks_metric_view(
        {"model/views/sales.yml": model},
        "sales",
        **kwargs,
    )


def test_returns_stable_artifacts_and_source():
    result = _convert()

    assert isinstance(result, ConversionResult)
    assert result.source == "orders"
    assert yaml.safe_load(result.ossie_yaml)["semantic_model"][0]["name"] == "sales"
    metric_view = yaml.safe_load(result.metric_view_yaml)
    assert metric_view["version"] == "1.1"
    assert metric_view["source"] == "main.sales.orders"
    assert result.dataset_sources == (
        DatasetSourceResolution("orders", "default", "main.sales.orders", "main.sales.orders"),
        DatasetSourceResolution(
            "customers", "default", "main.sales.customers", "main.sales.customers"
        ),
        DatasetSourceResolution(
            "countries", "default", "main.sales.countries", "main.sales.countries"
        ),
    )


def test_catalog_qualifies_every_two_part_source_and_clears_handled_warnings():
    model = _NESTED_MODEL.replace("main.sales.", "sales.")

    result = _convert(
        model,
        expected_data_source="default",
        default_catalog="analytics",
    )

    ossie = yaml.safe_load(result.ossie_yaml)["semantic_model"][0]
    assert [item["source"] for item in ossie["datasets"]] == [
        "analytics.sales.orders",
        "analytics.sales.customers",
        "analytics.sales.countries",
    ]
    metric_view = yaml.safe_load(result.metric_view_yaml)
    assert metric_view["source"] == "analytics.sales.orders"
    assert metric_view["joins"][0]["source"] == "analytics.sales.customers"
    assert metric_view["joins"][0]["joins"][0]["source"] == "analytics.sales.countries"
    assert [item.resolved_source for item in result.dataset_sources] == [
        "analytics.sales.orders",
        "analytics.sales.customers",
        "analytics.sales.countries",
    ]
    assert not any(item.code == "SOURCE_NOT_FULLY_QUALIFIED" for item in result.issues)


def test_catalog_and_schema_qualify_one_part_source():
    model = _NESTED_MODEL.replace("main.sales.", "")

    result = _convert(model, default_catalog="analytics", default_schema="sales")

    assert yaml.safe_load(result.metric_view_yaml)["source"] == "analytics.sales.orders"
    assert result.dataset_sources[0] == DatasetSourceResolution(
        "orders",
        "default",
        "orders",
        "analytics.sales.orders",
    )


@pytest.mark.parametrize(
    "model, kwargs, message",
    [
        (
            _NESTED_MODEL.replace("main.sales.", "sales."),
            {},
            "provide default_catalog",
        ),
        (
            _NESTED_MODEL.replace("main.sales.", ""),
            {"default_catalog": "analytics"},
            "provide default_schema",
        ),
    ],
)
def test_unqualified_sources_fail_without_explicit_namespace_context(model, kwargs, message):
    with pytest.raises(ConversionError, match=message):
        _convert(model, **kwargs)


def test_expected_data_source_rejects_a_joined_cube_from_another_connection():
    model = _NESTED_MODEL.replace(
        "\n  - name: customers\n    sql_table:",
        "\n  - name: customers\n    data_source: secondary\n    sql_table:",
    )

    with pytest.raises(
        ConversionError,
        match="dataset 'customers'.*'secondary'.*requested data source 'default'",
    ):
        _convert(model, expected_data_source="default")


def test_expected_non_default_data_source_requires_explicit_cube_ownership():
    with pytest.raises(
        ConversionError,
        match="dataset 'orders'.*'default'.*requested data source 'warehouse'",
    ):
        _convert(expected_data_source="warehouse")


def test_query_sources_are_not_qualified():
    model = _NESTED_MODEL.replace(
        "sql_table: main.sales.orders",
        "sql: SELECT * FROM main.sales.orders",
    )

    result = _convert(model, default_catalog="ignored", default_schema="ignored")

    source = result.dataset_sources[0]
    assert source.original_source == source.resolved_source
    assert source.resolved_source == "SELECT * FROM main.sales.orders"


def test_identifier_qualification_is_quote_aware():
    resolved = _qualify_table_source(
        "`default.schema`.`line.items`",
        "line_items",
        default_catalog="`analytics.catalog`",
        default_schema=None,
    )

    assert resolved == "`analytics.catalog`.`default.schema`.`line.items`"


def test_quoted_dotted_identifiers_survive_the_pinned_converter():
    model = _NESTED_MODEL
    for table in ("orders", "customers", "countries"):
        model = model.replace(
            f"sql_table: main.sales.{table}",
            f"sql_table: '`sales.schema`.{table}'",
        )

    result = _convert(model, default_catalog="`analytics.catalog`")

    metric_view = yaml.safe_load(result.metric_view_yaml)
    assert metric_view["source"] == "`analytics.catalog`.`sales.schema`.orders"
    assert metric_view["joins"][0]["source"] == ("`analytics.catalog`.`sales.schema`.customers")
    assert metric_view["joins"][0]["joins"][0]["source"] == (
        "`analytics.catalog`.`sales.schema`.countries"
    )


@pytest.mark.parametrize(
    "source",
    [
        "catalog..orders",
        ".schema.orders",
        "..orders",
        "``.schema.orders",
        "` `.schema.orders",
    ],
)
def test_empty_source_identifier_parts_are_rejected(source):
    with pytest.raises(ConversionError, match="empty identifier part"):
        _qualify_table_source(
            source,
            "orders",
            default_catalog="analytics",
            default_schema="sales",
        )


@pytest.mark.parametrize("value", ["analytics.reporting", "", "   ", "``", "` `"])
def test_default_catalog_must_be_one_non_empty_identifier(value):
    with pytest.raises(ConversionError, match="default_catalog"):
        _qualify_table_source(
            "default.orders",
            "orders",
            default_catalog=value,
            default_schema=None,
        )


def test_nested_computed_dimension_uses_full_metric_view_join_path():
    result = _convert()
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["display_name"] == (
        "CONCAT(customers.countries.name, ' (', customers.countries.code, ')')"
    )
    assert not any("complex expression on a joined table" in item.message for item in result.issues)


def test_simple_joined_dimension_is_not_qualified_twice():
    model = _NESTED_MODEL.replace("includes: [display_name]", "includes: [name]")

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["name"] == "customers.countries.name"


@pytest.mark.parametrize("expression", ["CURRENT_DATE", "CURRENT_USER", "TRUE"])
def test_joined_keyword_expression_is_not_rewritten_as_a_column(expression):
    model = _NESTED_MODEL.replace(
        "sql: \"CONCAT({name}, ' (', {code}, ')')\"",
        f'sql: "{expression}"',
    )

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["display_name"] == expression


def test_source_dimensions_use_explicit_source_provenance():
    model = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - name: users
        sql: "{CUBE}.user_id = {users}.id"
        relationship: many_to_one
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: user_profile_name, sql: users.name, type: string}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [id, user_profile_name]}
      - {join_path: orders.users, includes: []}
"""

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["id"] == "source.id"
    assert expressions["user_profile_name"] == "source.users.name"


def test_source_metric_path_is_not_mistaken_for_a_join_alias():
    model = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - name: users
        sql: "{CUBE}.user_id = {users}.id"
        relationship: many_to_one
    measures:
      - {name: max_profile_balance, sql: users.balance, type: max}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [max_profile_balance]}
      - {join_path: orders.users, includes: []}
"""

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    measures = {item["name"]: item["expr"] for item in metric_view["measures"]}

    assert measures["max_profile_balance"] == "MAX(source.users.balance)"


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


def test_prefix_shaped_physical_path_retains_owning_dataset_provenance():
    model = _NESTED_MODEL.replace(
        "sql: \"CONCAT({name}, ' (', {code}, ')')\"",
        "sql: \"CONCAT(customers.countries.name, ' (', customers.countries.code, ')')\"",
    )

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["display_name"] == (
        "CONCAT(customers.countries.customers.countries.name, ' (', "
        "customers.countries.customers.countries.code, ')')"
    )
    assert not any("complex expression on a joined table" in item.message for item in result.issues)


def test_source_shaped_physical_path_is_not_treated_as_metric_view_alias():
    model = _NESTED_MODEL.replace(
        "sql: \"CONCAT({name}, ' (', {code}, ')')\"",
        'sql: "source.id + value"',
    )

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}

    assert expressions["display_name"] == (
        "customers.countries.source.id + customers.countries.value"
    )
    assert not any("complex expression on a joined table" in item.message for item in result.issues)


@pytest.mark.parametrize(
    "expression",
    [
        "(SELECT max(value) FROM items)",
        "transform(values, x -> x + 1)",
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


@pytest.mark.parametrize(
    "datasets, relationships",
    [
        (
            ["orders", "users", "Users"],
            [
                {"from": "orders", "to": "users"},
                {"from": "orders", "to": "Users"},
            ],
        ),
        (
            ["orders", "Source"],
            [{"from": "orders", "to": "Source"}],
        ),
    ],
)
def test_case_insensitive_join_alias_collisions_fail_closed(datasets, relationships):
    model = {
        "datasets": [{"name": name} for name in datasets],
        "relationships": relationships,
    }

    with pytest.raises(ConversionError, match="join alias.*case-insensitively"):
        _dataset_qualifiers(model, "orders")


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


def test_concurrent_conversions_keep_upstream_diagnostics_call_local(monkeypatch):
    original = bridge_module.convert_ossie_to_metric_view
    start = threading.Barrier(4)

    def synchronized_converter(*args, **kwargs):
        start.wait()
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge_module, "convert_ossie_to_metric_view", synchronized_converter)

    def convert(index):
        dataset = f"orders_{index}"
        model = f"""
cubes:
  - name: {dataset}
    sql_table: main.sales.{dataset}
    dimensions:
      - {{name: id, sql: id, type: number, primary_key: true}}
views:
  - name: sales
    cubes:
      - {{join_path: {dataset}, includes: [id]}}
"""
        return convert_cube_view_to_databricks_metric_view({"model.yml": model}, "sales")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(convert, range(4)))

    for index, result in enumerate(results):
        messages = [issue.message for issue in result.issues if issue.origin == "ossie_databricks"]
        assert any(f"dataset 'orders_{index}'" in message for message in messages)
        assert not any(
            f"dataset 'orders_{other}'" in message
            for other in range(4)
            if other != index
            for message in messages
        )


def test_unrelated_warnings_are_not_swallowed_or_mislabeled(monkeypatch):
    original = bridge_module.convert_ossie_to_metric_view

    def noisy_converter(*args, **kwargs):
        thread = threading.Thread(
            target=warnings.warn,
            args=("unrelated application warning",),
            kwargs={"stacklevel": 2},
        )
        thread.start()
        thread.join()
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge_module, "convert_ossie_to_metric_view", noisy_converter)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = _convert()

    assert [str(item.message) for item in emitted] == ["unrelated application warning"]
    assert not any("unrelated application warning" in issue.message for issue in result.issues)
