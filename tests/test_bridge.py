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


def test_multi_stage_ratio_without_a_stage_modifier_publishes_its_aggregates():
    model = """
cubes:
  - name: adoption
    sql_table: main.product.adoption
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: account_id, sql: account_id, type: number}
    measures:
      - {name: eligible, sql: "{CUBE.account_id}", type: count_distinct}
      - name: adopted
        sql: "case when {CUBE}.score >= 0.5 then {CUBE}.account_id end"
        type: count_distinct
      - name: adopted_share
        multi_stage: true
        sql: "{CUBE.adopted} / NULLIF({CUBE.eligible}, 0)"
        type: number
views:
  - name: sales
    cubes:
      - {join_path: adoption, includes: [adopted_share]}
"""

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    measures = {item["name"]: item["expr"] for item in metric_view["measures"]}

    assert measures == {
        "adopted_share": (
            "COUNT(DISTINCT case when source.score >= 0.5 then source.account_id end)"
            " / NULLIF(COUNT(DISTINCT source.account_id), 0)"
        )
    }


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


_CONDITION_MODEL = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - name: customers
        relationship: many_to_one
        sql: >-
          {CUBE}.customer_id = {customers}.id
          AND {CUBE}.ordered_at >= {customers}.valid_from
          AND {CUBE}.ordered_at < {customers}.valid_to
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: order_month, sql: "date_trunc('month', {CUBE}.ordered_at)", type: time}
    measures:
      - {name: count, type: count}
  - name: customers
    sql_table: main.sales.customers_history
    joins:
      - name: regions
        relationship: many_to_one
        sql: >-
          {CUBE}.region_id = {regions}.id
          AND {CUBE}.valid_from >= {regions}.valid_from
          AND {CUBE}.valid_from < {regions}.valid_to
    dimensions:
      - {name: version_id, sql: version_id, type: number, primary_key: true}
      - {name: tier, sql: tier, type: string}
    measures:
      - {name: credit_limit, sql: credit_limit, type: sum}
  - name: regions
    sql_table: main.sales.regions_history
    dimensions:
      - {name: version_id, sql: version_id, type: number, primary_key: true}
      - {name: name, sql: name, type: string}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [id, count]}
      - {join_path: orders.customers, includes: [tier]}
      - {join_path: orders.customers.regions, prefix: true, includes: [name]}
"""


def _joins_by_name(metric_view):
    found = {}

    def visit(joins):
        for join in joins or []:
            found[join["name"]] = join
            visit(join.get("joins"))

    visit(metric_view.get("joins"))
    return found


def test_range_join_condition_is_published_as_written():
    result = _convert(_CONDITION_MODEL)
    metric_view = yaml.safe_load(result.metric_view_yaml)
    joins = _joins_by_name(metric_view)

    assert joins["customers"]["on"] == (
        "source.customer_id = customers.id AND source.ordered_at >= customers.valid_from "
        "AND source.ordered_at < customers.valid_to"
    )
    # A nested join reads its parent by the parent join's name, as an equality join does.
    assert joins["regions"]["on"] == (
        "customers.region_id = regions.id AND customers.valid_from >= regions.valid_from "
        "AND customers.valid_from < regions.valid_to"
    )
    assert not any("using" in join or "rely" in join for join in joins.values())
    expressions = {item["name"]: item["expr"] for item in metric_view["dimensions"]}
    assert expressions["regions_name"] == "customers.regions.name"


def test_expression_join_condition_resolves_member_references_and_keeps_literals():
    model = _CONDITION_MODEL.replace(
        "{CUBE}.customer_id = {customers}.id\n"
        "          AND {CUBE}.ordered_at >= {customers}.valid_from\n"
        "          AND {CUBE}.ordered_at < {customers}.valid_to",
        "CAST({CUBE}.customer_id AS INT) = {customers}.id\n"
        "          AND {CUBE.order_month} = dateadd(DAY, 1, {customers.tier})\n"
        "          AND {customers}.is_current = true AND {customers}.kind IN ('a', 'b')",
    )

    result = _convert(model)
    joins = _joins_by_name(yaml.safe_load(result.metric_view_yaml))

    assert joins["customers"]["on"] == (
        "CAST(source.customer_id AS INT) = customers.id "
        "AND date_trunc('month', source.ordered_at) = dateadd(DAY, 1, customers.tier) "
        "AND customers.is_current = true AND customers.kind IN ('a', 'b')"
    )


@pytest.mark.parametrize(
    ("condition", "published"),
    [
        ("{customers}.is_current = true", "customers.is_current = true"),
        ("{customers}.snapshot_date = current_date", "customers.snapshot_date = current_date"),
    ],
)
def test_keyword_in_an_equality_join_is_not_read_as_a_column(condition, published):
    model = _NESTED_MODEL.replace(
        '"{CUBE}.customer_id = {customers}.id"',
        f'"{{CUBE}}.customer_id = {{customers}}.id AND {condition}"',
    )

    joins = _joins_by_name(yaml.safe_load(_convert(model).metric_view_yaml))

    assert joins["customers"]["on"] == f"source.customer_id = customers.id AND {published}"


def test_keyword_in_a_one_to_many_join_reached_by_a_measure_is_not_read_as_a_column():
    model = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - name: line_items
        relationship: one_to_many
        sql: "{CUBE}.id = {line_items}.order_id AND {line_items}.is_valid = true"
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
    measures:
      - {name: max_qty, sql: "{line_items}.qty", type: max}
  - name: line_items
    sql_table: main.sales.line_items
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [id, max_qty]}
"""

    joins = _joins_by_name(yaml.safe_load(_convert(model).metric_view_yaml))

    assert joins["line_items"]["on"] == (
        "source.id = line_items.order_id AND line_items.is_valid = true"
    )
    assert joins["line_items"]["cardinality"] == "one_to_many"


def test_equality_joins_keep_their_column_form():
    joins = _joins_by_name(yaml.safe_load(_convert().metric_view_yaml))

    assert joins["customers"]["on"] == "source.customer_id = customers.id"
    assert joins["customers"]["rely"] == {"at_most_one_match": True}


def test_condition_join_keeps_fanout_checks():
    model = _CONDITION_MODEL.replace("includes: [tier]", "includes: [tier, credit_limit]")

    with pytest.raises(ConversionError, match="credit_limit"):
        _convert(model)
    relaxed = _convert(model, strict_fanout=False)
    assert any(item.code == "FANOUT_UNSAFE_METRIC" for item in relaxed.issues)


def test_source_override_reorients_a_condition_join():
    model = _CONDITION_MODEL.replace(
        "- {join_path: orders, includes: [id, count]}",
        "- {join_path: orders, includes: []}",
    )

    result = _convert(model, source="customers")
    joins = _joins_by_name(yaml.safe_load(result.metric_view_yaml))

    assert joins["orders"]["on"] == (
        "orders.customer_id = source.id AND orders.ordered_at >= source.valid_from "
        "AND orders.ordered_at < source.valid_to"
    )
    assert joins["regions"]["on"].startswith("source.region_id = regions.id")


def test_one_to_one_condition_join_is_published_and_its_note_left_to_the_caller():
    model = _CONDITION_MODEL.replace(
        "- name: customers\n        relationship: many_to_one",
        "- name: customers\n        relationship: one_to_one",
    )

    result = _convert(model)
    joins = _joins_by_name(yaml.safe_load(result.metric_view_yaml))

    assert joins["customers"]["on"].startswith("source.customer_id = customers.id AND")
    assert any(
        item.code == "PARKED_IN_META" and item.element == "join 'orders' -> 'customers'"
        for item in result.issues
    )


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ("{CUBE}.customer_id = {customers}.id AND {regions}.name = 'EU'", "reads cube 'regions'"),
        ("{CUBE}.customer_id = {customers}.id AND {CUBE.count} > 0", "references measure"),
    ],
)
def test_condition_join_reading_beyond_its_two_cubes_is_refused(condition, message):
    model = _CONDITION_MODEL.replace(
        "{CUBE}.customer_id = {customers}.id\n"
        "          AND {CUBE}.ordered_at >= {customers}.valid_from\n"
        "          AND {CUBE}.ordered_at < {customers}.valid_to",
        condition,
    )

    with pytest.raises(ConversionError, match=message):
        _convert(model)


def test_public_ossie_projection_does_not_carry_condition_joins():
    from cube_databricks_metric_view_bridge import convert_cube_view_to_ossie

    ossie_yaml, _, _ = convert_cube_view_to_ossie(
        {"model/views/sales.yml": _CONDITION_MODEL}, "sales", strict_fanout=False
    )

    assert "__cube_dmv_join_" not in ossie_yaml


def test_condition_member_reference_keeps_date_units_as_units():
    model = _CONDITION_MODEL.replace(
        "{name: order_month, sql: \"date_trunc('month', {CUBE}.ordered_at)\", type: time}",
        '{name: order_month, sql: "dateadd(DAY, 1, ordered_at)", type: time}',
    ).replace(
        "{CUBE}.customer_id = {customers}.id\n"
        "          AND {CUBE}.ordered_at >= {customers}.valid_from\n"
        "          AND {CUBE}.ordered_at < {customers}.valid_to",
        "{CUBE}.customer_id = {customers}.id AND {order_month} < {customers}.valid_to",
    )

    joins = _joins_by_name(yaml.safe_load(_convert(model).metric_view_yaml))

    assert joins["customers"]["on"] == (
        "source.customer_id = customers.id AND dateadd(DAY, 1, source.ordered_at) < customers.valid_to"
    )


@pytest.mark.parametrize(
    "condition",
    [
        "orders.customer_id = customers.id AND orders.ordered_at >= Customers.valid_from",
        "{CUBE}.customer_id = customers.id AND {CUBE}.ordered_at >= customers.valid_from",
    ],
)
def test_condition_column_qualified_by_a_joined_cube_name_reads_that_cube(condition):
    model = _CONDITION_MODEL.replace(
        "{CUBE}.customer_id = {customers}.id\n"
        "          AND {CUBE}.ordered_at >= {customers}.valid_from\n"
        "          AND {CUBE}.ordered_at < {customers}.valid_to",
        condition,
    )

    joins = _joins_by_name(yaml.safe_load(_convert(model).metric_view_yaml))

    assert joins["customers"]["on"] == (
        "source.customer_id = customers.id AND source.ordered_at >= customers.valid_from"
    )


def test_condition_column_qualified_by_another_table_name_is_refused():
    model = _CONDITION_MODEL.replace(
        "{CUBE}.customer_id = {customers}.id\n"
        "          AND {CUBE}.ordered_at >= {customers}.valid_from\n"
        "          AND {CUBE}.ordered_at < {customers}.valid_to",
        "{CUBE}.customer_id = {customers}.id AND regions.name = 'EU'",
    )

    with pytest.raises(ConversionError, match="reads 'regions.name'"):
        _convert(model)


def test_condition_member_reading_a_cube_outside_the_view_is_refused():
    model = (
        _CONDITION_MODEL.replace(
            "{CUBE}.customer_id = {customers}.id\n"
            "          AND {CUBE}.ordered_at >= {customers}.valid_from\n"
            "          AND {CUBE}.ordered_at < {customers}.valid_to",
            "{CUBE}.customer_id = {customers}.id AND {CUBE.segment_code} = {customers}.tier",
        )
        .replace(
            "      - {name: id, sql: id, type: number, primary_key: true}\n"
            "      - {name: order_month,",
            "      - {name: id, sql: id, type: number, primary_key: true}\n"
            '      - {name: segment_code, sql: "{segments}.code", type: string}\n'
            "      - {name: order_month,",
        )
        .replace(
            "views:",
            "  - name: segments\n"
            "    sql_table: main.sales.segments\n"
            "    dimensions:\n"
            "      - {name: code, sql: code, type: string, primary_key: true}\n"
            "views:",
        )
    )

    with pytest.raises(ConversionError, match="reads 'segments.code'"):
        _convert(model)
