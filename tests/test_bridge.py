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


def test_metric_with_a_lambda_returns_no_publishable_artifact():
    model = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
    measures:
      - name: unsafe_metric
        sql: "aggregate(items, 0, (acc, x) -> acc + x)"
        type: max
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [unsafe_metric]}
"""

    with pytest.raises(ConversionError, match="nested SQL bindings"):
        _convert(model)


_SUBQUERY_MODEL = """
cubes:
  - name: usage
    sql: >
      SELECT metric_ts, metric_type, metric_value, account_id, bot_id
      FROM main.product.usage_metrics
    joins:
      - name: accounts
        sql: "{CUBE}.account_id = {accounts}.id"
        relationship: many_to_one
    dimensions:
      - {name: metric_ts, sql: "{CUBE}.metric_ts", type: time, primary_key: true}
      - {name: account_id, sql: "{CUBE}.account_id", type: string, primary_key: true}
      - {name: bot_id, sql: "coalesce({CUBE}.bot_id, '')", type: string}
    measures:
      - name: licensed_usage
        type: sum
        sql: >
          case when {CUBE}.metric_type = 'usage' and {CUBE}.account_id IN (
            SELECT account_id FROM main.product.usage_metrics
            WHERE metric_ts = {CUBE}.metric_ts -- the same month
              AND coalesce(properties.is_trial, 0) <> 1
          ) then {CUBE}.metric_value else 0 end
      - name: busy_accounts
        type: count_distinct
        sql: >
          case when {CUBE}.account_id IN (
            SELECT p.account_id FROM (
              SELECT u.account_id, SUM(u.metric_value) AS total
              FROM main.product.usage_metrics u
              WHERE u.metric_ts = {CUBE}.metric_ts AND u.properties['bot_id'] = {bot_id}
              GROUP BY u.account_id
            ) p
            WHERE p.total > 5
          ) then {CUBE}.account_id end
      - name: usage_per_busy_account
        type: number
        sql: "{licensed_usage} / nullif({busy_accounts}, 0)"
  - name: accounts
    sql_table: main.product.accounts
    dimensions:
      - {name: id, sql: id, type: string, primary_key: true}
      - {name: region, sql: region, type: string}
views:
  - name: adoption
    cubes:
      - join_path: usage
        includes: [metric_ts, licensed_usage, busy_accounts, usage_per_busy_account]
      - {join_path: usage.accounts, includes: [region]}
"""

_LICENSED_USAGE = (
    "SUM(case when source.metric_type = 'usage' and source.account_id IN (\n"
    "  SELECT account_id FROM main.product.usage_metrics\n"
    "  WHERE metric_ts = __cube_outer_0__ -- the same month\n"
    "    AND coalesce(properties.is_trial, 0) <> 1\n"
    ") then source.metric_value else 0 end)"
)
_BUSY_ACCOUNTS = (
    "COUNT(DISTINCT case when source.account_id IN (\n"
    "  SELECT p.account_id FROM (\n"
    "    SELECT u.account_id, SUM(u.metric_value) AS total\n"
    "    FROM main.product.usage_metrics u\n"
    "    WHERE u.metric_ts = __cube_outer_0__ AND u.properties['bot_id'] = coalesce(__cube_outer_1__, '')\n"
    "    GROUP BY u.account_id\n"
    "  ) p\n"
    "  WHERE p.total > 5\n"
    ") then source.account_id end)"
)


def test_metric_subquery_reads_the_outer_row_through_uniquely_named_source_columns():
    result = convert_cube_view_to_databricks_metric_view(
        {"model/usage.yml": _SUBQUERY_MODEL}, "adoption"
    )
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expressions = {item["name"]: item["expr"] for item in metric_view["measures"]}

    assert metric_view["source"] == (
        "SELECT *, metric_ts AS __cube_outer_0__, bot_id AS __cube_outer_1__ FROM "
        "(SELECT metric_ts, metric_type, metric_value, account_id, bot_id "
        "FROM main.product.usage_metrics) AS __cube_outer_source__"
    )

    assert expressions == {
        "licensed_usage": _LICENSED_USAGE,
        "busy_accounts": _BUSY_ACCOUNTS,
        "usage_per_busy_account": f"{_LICENSED_USAGE} / nullif({_BUSY_ACCOUNTS}, 0)",
    }
    assert [item["name"] for item in metric_view["dimensions"]] == ["metric_ts", "region"]


@pytest.mark.parametrize("reference", ["{accounts.region}", "{accounts}.region"])
def test_metric_subquery_reading_a_joined_cube_fails_closed(reference):
    model = _SUBQUERY_MODEL.replace(
        "WHERE metric_ts = {CUBE}.metric_ts",
        f"WHERE metric_ts = {{CUBE}}.metric_ts AND region = {reference}",
    )

    with pytest.raises(ConversionError, match="reads join 'accounts' inside a subquery"):
        convert_cube_view_to_databricks_metric_view(
            {"model/usage.yml": model}, "adoption", strict_fanout=False
        )


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


_JOINED_DIMENSION_MODEL = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: status, sql: status, type: string}
      - {name: user_name, sql: "{users.name}", type: string}
      - {name: user_tier, sql: "{users}.tier", type: string}
      - name: label
        sql: "CONCAT({CUBE}.status, ' / ', COALESCE({users.name}, 'n/a'))"
        type: string
      - {name: shouted_name, sql: "UPPER({user_name})", type: string}
      - {name: country, sql: "{users.country_name}", type: string}
    measures:
      - {name: count, type: count}
  - name: users
    sql_table: main.sales.users
    joins:
      - {name: countries, sql: "{CUBE}.country_id = {countries}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: name, sql: name, type: string}
      - {name: tier, sql: tier, type: string}
      - {name: country_name, sql: "{countries.name}", type: string}
  - name: countries
    sql_table: main.sales.countries
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: name, sql: name, type: string}
views:
  - name: sales
    cubes:
      - join_path: orders
        includes: [status, user_name, user_tier, label, shouted_name, country, count]
"""


def _dimension_expressions(result):
    metric_view = yaml.safe_load(result.metric_view_yaml)
    return {item["name"]: item["expr"] for item in metric_view["dimensions"]}


def test_dimension_reading_a_joined_cube_is_published_through_its_join():
    result = _convert(_JOINED_DIMENSION_MODEL)
    expressions = _dimension_expressions(result)

    assert expressions == {
        "status": "source.status",
        "user_name": "users.name",
        "user_tier": "users.tier",
        "label": "CONCAT(source.status, ' / ', COALESCE(users.name, 'n/a'))",
        "shouted_name": "UPPER(users.name)",
        "country": "users.countries.name",
    }
    joins = _joins_by_name(yaml.safe_load(result.metric_view_yaml))
    assert set(joins) == {"users", "countries"}
    # The Cube converter's stash of the unconverted SQL is gone, so no field warns.
    assert not [issue for issue in result.issues if "[field " in issue.message]
    # The intermediate Ossie field keeps each column's dataset.
    fields = {
        field["name"]: field["expression"]["dialects"][0]["expression"]
        for dataset in yaml.safe_load(result.ossie_yaml)["semantic_model"][0]["datasets"]
        for field in dataset.get("fields") or []
    }
    assert fields["label"] == "CONCAT(orders.status, ' / ', COALESCE(users.name, 'n/a'))"


def test_joined_dataset_dimension_reading_a_deeper_cube_uses_both_join_paths():
    model = _JOINED_DIMENSION_MODEL.replace(
        '      - {name: country_name, sql: "{countries.name}", type: string}',
        '      - {name: country_name, sql: "{countries.name}", type: string}\n'
        "      - name: tier_country\n"
        "        sql: \"CONCAT({CUBE}.tier, '@', {countries.name})\"\n"
        "        type: string",
    ).replace(
        "includes: [status, user_name, user_tier, label, shouted_name, country, count]",
        "includes: [status, count]\n      - {join_path: orders.users, includes: [tier_country]}",
    )

    expressions = _dimension_expressions(_convert(model))

    assert expressions["tier_country"] == "CONCAT(users.tier, '@', users.countries.name)"


def test_dimension_reading_a_one_to_many_join_is_refused():
    model = (
        _JOINED_DIMENSION_MODEL.replace(
            "relationship: many_to_one}\n    dimensions:\n      - {name: id, sql: id, type: number, primary_key: true}\n      - {name: status",
            "relationship: many_to_one}\n"
            '      - {name: items, sql: "{CUBE}.id = {items}.order_id", relationship: one_to_many}\n'
            "    dimensions:\n      - {name: id, sql: id, type: number, primary_key: true}\n"
            '      - {name: item_sku, sql: "{items.sku}", type: string}\n      - {name: status',
        )
        .replace(
            "views:",
            "  - name: items\n"
            "    sql_table: main.sales.items\n"
            "    dimensions:\n"
            "      - {name: id, sql: id, type: number, primary_key: true}\n"
            "      - {name: sku, sql: sku, type: string}\n"
            "views:",
        )
        .replace("includes: [status, user_name,", "includes: [item_sku, status, user_name,")
    )

    with pytest.raises(ConversionError, match="reads join 'items', which is one-to-many"):
        _convert(model)


def test_dimension_reading_a_joined_cube_keeps_its_presentation_metadata():
    model = _JOINED_DIMENSION_MODEL.replace(
        '      - {name: user_tier, sql: "{users}.tier", type: string}\n',
        '      - {name: user_tier, sql: "{users}.tier", type: string, format: link}\n',
    )

    result = _convert(model)
    fields = {
        field["name"]: field
        for dataset in yaml.safe_load(result.ossie_yaml)["semantic_model"][0]["datasets"]
        for field in dataset.get("fields") or []
    }

    assert [extension["data"] for extension in fields["user_tier"]["custom_extensions"]] == [
        '{"_v": 1, "format": "link"}'
    ]


def test_date_units_stay_units_in_dimensions_and_measures_reading_joined_cubes():
    model = (
        _JOINED_DIMENSION_MODEL.replace(
            "      - {name: status, sql: status, type: string}\n",
            "      - {name: status, sql: status, type: string}\n"
            "      - name: days_since_signup\n"
            '        sql: "DATEDIFF(DAY, {users.created_at}, {CUBE}.created_at)"\n'
            "        type: number\n",
        )
        .replace(
            "      - {name: count, type: count}\n",
            "      - {name: count, type: count}\n"
            "      - name: signup_hours\n"
            '        sql: "TIMESTAMPDIFF(HOUR, {CUBE}.created_at, {CUBE}.shipped_at)"\n'
            "        type: sum\n",
        )
        .replace(
            "      - {name: tier, sql: tier, type: string}\n",
            "      - {name: tier, sql: tier, type: string}\n"
            "      - {name: created_at, sql: created_at, type: time}\n",
        )
        .replace("includes: [status,", "includes: [days_since_signup, signup_hours, status,")
    )

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)

    assert _dimension_expressions(result)["days_since_signup"] == (
        "DATEDIFF(DAY, users.created_at, source.created_at)"
    )
    assert {item["name"]: item["expr"] for item in metric_view["measures"]}["signup_hours"] == (
        "SUM(TIMESTAMPDIFF(HOUR, source.created_at, source.shipped_at))"
    )


def test_join_path_member_reference_publishes_as_the_joined_member():
    # Cube reads `{CUBE.users.tier}` as `users.tier` joined from `orders`.
    model = (
        _JOINED_DIMENSION_MODEL.replace(
            "      - {name: count, type: count}\n",
            "      - {name: count, type: count}\n"
            "      - name: gold_users\n"
            "        type: count_distinct\n"
            "        sql: \"CASE WHEN {CUBE.users.tier} = 'gold' THEN {CUBE.users.id} END\"\n"
            "      - name: gold_users_by_cube\n"
            "        type: count_distinct\n"
            "        sql: \"CASE WHEN {users.tier} = 'gold' THEN {users.id} END\"\n",
        )
        .replace(
            "      - {name: status, sql: status, type: string}\n",
            "      - {name: status, sql: status, type: string}\n"
            '      - {name: path_country, sql: "{orders.users.countries.name}", type: string}\n',
        )
        .replace(
            "includes: [status,", "includes: [gold_users, gold_users_by_cube, path_country, status,"
        )
    )

    result = _convert(model)
    measures = {
        item["name"]: item["expr"] for item in yaml.safe_load(result.metric_view_yaml)["measures"]
    }

    assert (
        measures["gold_users"] == "COUNT(DISTINCT CASE WHEN users.tier = 'gold' THEN users.id END)"
    )
    assert measures["gold_users"] == measures["gold_users_by_cube"]
    assert _dimension_expressions(result)["path_country"] == "users.countries.name"


def test_join_path_member_reference_off_the_published_joins_is_refused():
    model = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
      - {name: countries, sql: "{CUBE}.ship_country_id = {countries}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: user_country, sql: "{CUBE.users.countries.name}", type: string}
    measures:
      - {name: count, type: count}
  - name: users
    sql_table: main.sales.users
    joins:
      - {name: countries, sql: "{CUBE}.country_id = {countries}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
  - name: countries
    sql_table: main.sales.countries
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: name, sql: name, type: string}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [user_country, count]}
      - {join_path: orders.users, includes: []}
      - {join_path: orders.countries, includes: []}
"""

    with pytest.raises(ConversionError, match="through join path 'orders.users.countries'"):
        _convert(model)


def test_join_path_member_reference_follows_its_own_join_among_several():
    # `users` is reachable directly and through `accounts`; Cube joins along the path named.
    model = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
      - {name: accounts, sql: "{CUBE}.account_id = {accounts}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
    measures:
      - {name: gold_users, type: count_distinct, sql: "CASE WHEN {CUBE.users.tier} = 'gold' THEN {CUBE}.user_id END"}
  - name: accounts
    sql_table: main.sales.accounts
    joins:
      - {name: users, sql: "{CUBE}.owner_id = {users}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: tier, sql: tier, type: string}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [gold_users]}
"""

    result = _convert(model)
    metric_view = yaml.safe_load(result.metric_view_yaml)

    assert set(_joins_by_name(metric_view)) == {"users"}
    assert metric_view["measures"][0]["expr"] == (
        "COUNT(DISTINCT CASE WHEN users.tier = 'gold' THEN source.user_id END)"
    )
    with pytest.raises(ConversionError, match="multiple declared join paths"):
        _convert(model.replace("{CUBE.users.tier}", "{users.tier}"))


@pytest.mark.parametrize(
    "reference",
    ["{CUBE.users.tier_rank}", "{users.tier_rank}", "{order_max}", "{order_max_calc}"],
)
def test_aggregate_measure_reading_a_measure_is_refused(reference):
    # SQL cannot nest one aggregate in another; Cube itself plans this as a subquery.
    model = (
        _JOINED_DIMENSION_MODEL.replace(
            "      - {name: count, type: count}\n",
            "      - {name: count, type: count}\n"
            "      - {name: order_max, type: max, sql: id}\n"
            '      - {name: order_max_calc, type: number, sql: "MAX({CUBE}.id) + 0"}\n'
            "      - name: ranked_users\n"
            "        type: count_distinct\n"
            f'        sql: "CASE WHEN {reference} > 1 THEN {{users.id}} END"\n',
        )
        .replace(
            '      - {name: country_name, sql: "{countries.name}", type: string}\n',
            '      - {name: country_name, sql: "{countries.name}", type: string}\n'
            "    measures:\n"
            "      - {name: tier_rank, type: max, sql: rank}\n",
        )
        .replace("includes: [status,", "includes: [ranked_users, status,")
    )

    with pytest.raises(ConversionError, match="'orders.ranked_users' aggregates measure"):
        _convert(model)


def test_plain_reference_does_not_follow_a_path_another_member_named():
    # Cube joins `{users.tier}` alone along the shortest path, orders -> users, so
    # a plain reference stays ambiguous here whatever path another member names.
    model = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    joins:
      - {name: users, sql: "{CUBE}.user_id = {users}.id", relationship: many_to_one}
      - {name: accounts, sql: "{CUBE}.account_id = {accounts}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
    measures:
      - name: owner_gold
        type: count_distinct
        sql: "CASE WHEN {CUBE.accounts.users.tier} = 'gold' THEN {CUBE}.id END"
      - {name: buyer_gold, type: count_distinct, sql: "CASE WHEN {users.tier} = 'gold' THEN {CUBE}.id END"}
  - name: accounts
    sql_table: main.sales.accounts
    joins:
      - {name: users, sql: "{CUBE}.owner_id = {users}.id", relationship: many_to_one}
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
  - name: users
    sql_table: main.sales.users
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: tier, sql: tier, type: string}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [owner_gold, buyer_gold]}
"""

    with pytest.raises(ConversionError, match="'users' through multiple declared join paths"):
        _convert(model)
    owner_only = yaml.safe_load(
        _convert(model.replace("[owner_gold, buyer_gold]", "[owner_gold]")).metric_view_yaml
    )
    assert owner_only["measures"][0]["expr"] == (
        "COUNT(DISTINCT CASE WHEN accounts.users.tier = 'gold' THEN source.id END)"
    )


def test_aggregate_over_a_row_level_calculated_measure_is_published():
    # Cube renders a calculated measure's SQL as written, so this is SUM(price - discount).
    model = _JOINED_DIMENSION_MODEL.replace(
        "      - {name: count, type: count}\n",
        "      - {name: count, type: count}\n"
        '      - {name: net_price, type: number, sql: "{CUBE}.price - {CUBE}.discount"}\n'
        '      - {name: net_revenue, type: sum, sql: "{net_price}"}\n',
    ).replace("includes: [status,", "includes: [net_revenue, status,")

    measures = {
        item["name"]: item["expr"]
        for item in yaml.safe_load(_convert(model).metric_view_yaml)["measures"]
    }

    assert measures["net_revenue"] == "SUM((source.price - source.discount))"


@pytest.mark.parametrize(
    "subquery",
    [
        "(SELECT account_id FROM main.product.trials) UNION SELECT account_id "
        "FROM main.product.usage_metrics WHERE metric_ts = {CUBE}.metric_ts",
        "((SELECT account_id FROM main.product.trials)) except all SELECT account_id "
        "FROM main.product.usage_metrics WHERE metric_ts = {CUBE}.metric_ts",
        "FROM main.product.usage_metrics SELECT account_id WHERE metric_ts = {CUBE}.metric_ts",
    ],
)
def test_metric_subquery_columns_stay_unqualified_in_every_query_form(subquery):
    model = f"""
cubes:
  - name: usage
    sql_table: main.product.usage_metrics
    dimensions:
      - {{name: account_id, sql: account_id, type: string, primary_key: true}}
    measures:
      - name: licensed_usage
        type: sum
        sql: "case when account_id IN ({subquery}) then metric_value end"
views:
  - name: adoption
    cubes:
      - {{join_path: usage, includes: [licensed_usage]}}
"""
    result = convert_cube_view_to_databricks_metric_view({"model/usage.yml": model}, "adoption")
    metric_view = yaml.safe_load(result.metric_view_yaml)
    expected = subquery.replace("{CUBE}.metric_ts", "__cube_outer_0__")

    assert metric_view["source"] == (
        "SELECT *, metric_ts AS __cube_outer_0__ FROM main.product.usage_metrics"
    )
    assert metric_view["measures"][0]["expr"] == (
        f"SUM(case when source.account_id IN ({expected}) then source.metric_value end)"
    )


def test_aggregate_inside_a_subquery_does_not_make_a_referenced_measure_aggregating():
    model = """
cubes:
  - name: usage
    sql_table: main.product.usage_metrics
    dimensions:
      - {name: account_id, sql: account_id, type: string, primary_key: true}
    measures:
      - name: busy_value
        type: number
        sql: >
          case when {CUBE}.account_id IN (
            SELECT account_id FROM main.product.usage_metrics GROUP BY account_id
            HAVING SUM(metric_value) > 5
          ) then {CUBE}.metric_value else 0 end
      - {name: busy_total, type: sum, sql: "{busy_value}"}
views:
  - name: adoption
    cubes:
      - {join_path: usage, includes: [busy_total]}
"""
    result = convert_cube_view_to_databricks_metric_view({"model/usage.yml": model}, "adoption")

    assert yaml.safe_load(result.metric_view_yaml)["measures"][0]["expr"] == (
        "SUM((case when source.account_id IN (\n"
        "  SELECT account_id FROM main.product.usage_metrics GROUP BY account_id\n"
        "  HAVING SUM(metric_value) > 5\n"
        ") then source.metric_value else 0 end\n))"
    )


_DIMENSION_IN_SUBQUERY_MODEL = """
cubes:
  - name: usage
    sql_table: main.product.usage_metrics
    dimensions:
      - {name: account_id, sql: account_id, type: string, primary_key: true}
      - {name: bot_key, sql: "%s", type: string}
      - {name: bot_label, sql: "{bot_key}", type: string}
    measures:
      - {name: total, type: sum, sql: metric_value}
      - name: bot_usage
        type: sum
        sql: >
          case when {CUBE}.account_id IN (
            SELECT u.account_id FROM main.product.usage_metrics u WHERE u.bot = %s
          ) then {CUBE}.metric_value end
views:
  - name: adoption
    cubes:
      - {join_path: usage, includes: [bot_usage]}
"""


@pytest.mark.parametrize("reference", ["{bot_key}", "{bot_label}"])
def test_dimension_with_bare_columns_inside_a_measure_subquery_fails_closed(reference):
    # Cube pastes the dimension's SQL into the subquery, where `bot_id` binds to `u`.
    model = _DIMENSION_IN_SUBQUERY_MODEL % ("coalesce(bot_id, '')", reference)

    with pytest.raises(ConversionError, match="bare column"):
        convert_cube_view_to_databricks_metric_view({"model/usage.yml": model}, "adoption")


def test_single_identifier_dimension_inside_a_measure_subquery_reads_the_outer_row():
    # Cube prefixes a dimension whose SQL is one identifier with the cube's alias.
    model = _DIMENSION_IN_SUBQUERY_MODEL % ("bot_id", "{bot_label}")
    result = convert_cube_view_to_databricks_metric_view({"model/usage.yml": model}, "adoption")
    metric_view = yaml.safe_load(result.metric_view_yaml)

    assert metric_view["source"] == (
        "SELECT *, bot_id AS __cube_outer_0__ FROM main.product.usage_metrics"
    )
    assert "WHERE u.bot = __cube_outer_0__" in metric_view["measures"][0]["expr"]


@pytest.mark.parametrize(
    "reference, error",
    [("hash({CUBE}.*)", "cannot be safely rendered"), ("{total}", "reads measure")],
)
def test_star_or_measure_reference_inside_a_measure_subquery_fails_closed(reference, error):
    model = _DIMENSION_IN_SUBQUERY_MODEL % ("bot_id", reference)

    with pytest.raises(ConversionError, match=error):
        convert_cube_view_to_databricks_metric_view({"model/usage.yml": model}, "adoption")


_OPERATOR_DIMENSION_MODEL = """
cubes:
  - name: orders
    sql_table: main.sales.orders
    dimensions:
      - {name: id, sql: id, type: number, primary_key: true}
      - {name: gross, sql: "{CUBE}.price + {CUBE}.tax", type: number}
      - {name: gross_label, sql: "{gross}", type: number}
      - {name: is_open, sql: "{CUBE}.status = 'open' OR {CUBE}.status = 'new'", type: boolean}
    measures:
      - {name: doubled_gross, type: sum, sql: "{gross} * 2"}
      - {name: doubled_label, type: sum, sql: "{gross_label} * 2"}
      - {name: open_tax, type: sum, sql: "case when {is_open} AND {CUBE}.tax > 0 then {CUBE}.tax end"}
      - name: open_count
        type: count
        filters:
          - sql: "{is_open} AND {CUBE}.tax > 0"
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [doubled_gross, doubled_label, open_tax, open_count]}
"""


def test_dimension_with_a_top_level_operator_keeps_its_precedence_inside_a_measure():
    result = _convert(_OPERATOR_DIMENSION_MODEL)
    expressions = {
        item["name"]: item["expr"] for item in yaml.safe_load(result.metric_view_yaml)["measures"]
    }

    assert expressions["doubled_gross"] == "SUM((source.price + source.tax) * 2)"
    assert expressions["doubled_label"] == "SUM((source.price + source.tax) * 2)"
    assert expressions["open_tax"] == (
        "SUM(case when (source.status = 'open' OR source.status = 'new') "
        "AND source.tax > 0 then source.tax end)"
    )
    assert (
        "(source.status = 'open' OR source.status = 'new') AND source.tax > 0"
        in (expressions["open_count"])
    )
