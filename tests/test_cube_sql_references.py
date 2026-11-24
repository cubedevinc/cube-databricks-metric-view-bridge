# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""A cube's SQL that reads another cube's SQL through ``{other_cube.sql()}``."""

import pytest
import yaml
from ossie_cube import ConversionError

from cube_databricks_metric_view_bridge import (
    convert_cube_view_to_databricks_metric_view,
    convert_cube_view_to_ossie,
)

_MODEL = """
cubes:
  - name: orders
    sql: "SELECT * FROM {orders_base.sql()} WHERE o_orderstatus <> 'X'"
    joins:
      - name: order_totals
        sql: "{CUBE}.id = {order_totals}.order_id"
        relationship: many_to_one
    dimensions:
      - {name: id, sql: o_orderkey, type: number, primary_key: true}
      - {name: status, sql: o_orderstatus, type: string}
    measures:
      - {name: count, type: count}
  - name: orders_base
    sql_table: main.tpch.orders
    dimensions:
      - {name: id, sql: o_orderkey, type: number, primary_key: true}
  - name: order_totals
    sql: "SELECT t.order_id, t.total FROM ({line_totals.sql()}) AS t"
    dimensions:
      - {name: order_id, sql: order_id, type: number, primary_key: true}
      - {name: total, sql: total, type: number}
  - name: line_totals
    sql: >-
      SELECT l_orderkey AS order_id, SUM(l_extendedprice) AS total
      FROM main.tpch.lineitem GROUP BY l_orderkey
    dimensions:
      - {name: order_id, sql: order_id, type: number, primary_key: true}
views:
  - name: sales
    cubes:
      - {join_path: orders, includes: [status, count]}
      - {join_path: orders.order_totals, includes: [total]}
"""

_LINE_TOTALS_SQL = (
    "SELECT l_orderkey AS order_id, SUM(l_extendedprice) AS total "
    "FROM main.tpch.lineitem GROUP BY l_orderkey"
)


def _convert(model=_MODEL, **kwargs):
    return convert_cube_view_to_databricks_metric_view({"model.yml": model}, "sales", **kwargs)


def _sources(result):
    metric_view = yaml.safe_load(result.metric_view_yaml)
    return metric_view["source"], {join["name"]: join["source"] for join in metric_view["joins"]}


def test_referenced_cube_sql_is_inlined_as_cube_compiles_it():
    result = _convert()

    root, joins = _sources(result)
    # A table reads as written; SQL reads in parentheses, inside the caller's own.
    assert root == "SELECT * FROM main.tpch.orders WHERE o_orderstatus <> 'X'"
    assert joins == {
        "order_totals": f"SELECT t.order_id, t.total FROM (({_LINE_TOTALS_SQL})) AS t",
    }
    assert "{" not in result.metric_view_yaml
    resolved = {item.dataset: item.resolved_source for item in result.dataset_sources}
    assert resolved == {"orders": root, "order_totals": joins["order_totals"]}


def test_the_ossie_projection_carries_the_inlined_sql_too():
    ossie_yaml, _, _ = convert_cube_view_to_ossie({"model.yml": _MODEL}, "sales")

    datasets = yaml.safe_load(ossie_yaml)["semantic_model"][0]["datasets"]
    assert {dataset["name"]: dataset["source"] for dataset in datasets} == {
        "orders": "SELECT * FROM main.tpch.orders WHERE o_orderstatus <> 'X'",
        "order_totals": f"SELECT t.order_id, t.total FROM (({_LINE_TOTALS_SQL})) AS t",
    }


@pytest.mark.parametrize(
    ("base", "inlined"),
    [
        # A two-part table stays as Cube sends it; the publish session's catalog resolves it.
        ("sql_table: tpch.orders", "tpch.orders"),
        # Cube reads a bare 'select * from <table>' as the table itself.
        ('sql: "select * from main.tpch.orders"', "main.tpch.orders"),
        # Nested: the referenced cube's own reference expands first, then the rule applies.
        ('sql: "SELECT * FROM {deeper.sql()}"', "main.tpch.orders"),
        (
            'sql: "SELECT * FROM {deeper.sql()} WHERE o_totalprice > 0"',
            "(SELECT * FROM main.tpch.orders WHERE o_totalprice > 0)",
        ),
        ('sql_table: "{deeper.sql()}"', "main.tpch.orders"),
        ("sqlTable: main.tpch.orders\n    dataSource: default", "main.tpch.orders"),
        # Cube YAML's escaped literal braces are not templates.
        (
            r"sql: 'SELECT * FROM main.tpch.orders WHERE o_comment RLIKE ''[0-9]\{3\}'''",
            r"(SELECT * FROM main.tpch.orders WHERE o_comment RLIKE '[0-9]\{3\}')",
        ),
        # Cube inlines the text whatever data source the referenced cube names.
        ("sql_table: main.tpch.orders\n    data_source: other", "main.tpch.orders"),
    ],
)
def test_referenced_sql_follows_cube_expansion_rules(base, inlined):
    model = _with_cube(
        _MODEL.replace("    sql_table: main.tpch.orders\n", f"    {base}\n", 1),
        "deeper",
        "sql_table: main.tpch.orders",
    )

    root, _ = _sources(_convert(model))

    assert root == f"SELECT * FROM {inlined} WHERE o_orderstatus <> 'X'"


def _with_cube(model, name, source):
    return model.replace(
        "views:\n",
        f"  - name: {name}\n"
        f"    {source}\n"
        "    dimensions:\n"
        "      - {name: id, sql: o_orderkey, type: number, primary_key: true}\n"
        "views:\n",
        1,
    )


def test_reference_written_with_spaces_is_inlined():
    model = _MODEL.replace("{orders_base.sql()}", "{ orders_base . sql( ) }")

    root, _ = _sources(_convert(model))

    assert root == "SELECT * FROM main.tpch.orders WHERE o_orderstatus <> 'X'"


@pytest.mark.parametrize(
    ("base", "source"),
    [
        (
            'sql: "SELECT o_orderkey FROM main.tpch.orders"',
            "SELECT o_orderkey FROM main.tpch.orders",
        ),
        # A table source is qualified like any sql_table.
        ("sql_table: tpch.orders", "main.tpch.orders"),
    ],
)
def test_a_cube_whose_whole_sql_is_one_reference_takes_that_source(base, source):
    model = _MODEL.replace(
        "\"SELECT * FROM {orders_base.sql()} WHERE o_orderstatus <> 'X'\"",
        '"{orders_base.sql()}"',
    ).replace("    sql_table: main.tpch.orders\n", f"    {base}\n", 1)

    root, _ = _sources(_convert(model, default_catalog="main"))

    assert root == source


def test_a_cube_whose_sql_table_is_a_reference_takes_that_source():
    model = _MODEL.replace(
        "    sql: \"SELECT * FROM {orders_base.sql()} WHERE o_orderstatus <> 'X'\"\n",
        '    sql_table: "{orders_base.sql()}"\n',
    )

    root, _ = _sources(_convert(model))

    assert root == "main.tpch.orders"


def test_a_hidden_dependency_cube_gets_its_reference_expanded():
    model = (
        _MODEL.replace(
            "      - {name: count, type: count}\n",
            "      - {name: count, type: count}\n"
            '      - {name: max_total, sql: "{order_totals}.total", type: max}\n',
        )
        .replace("includes: [status, count]", "includes: [status, max_total]")
        .replace("      - {join_path: orders.order_totals, includes: [total]}\n", "")
    )

    _, joins = _sources(_convert(model))

    assert joins == {
        "order_totals": f"SELECT t.order_id, t.total FROM (({_LINE_TOTALS_SQL})) AS t",
    }


def test_a_cube_own_sql_keeps_other_templates_for_the_caller():
    model = _MODEL.replace(
        "WHERE o_orderstatus <> 'X'",
        "WHERE {FILTER_PARAMS.orders.status.filter('o_orderstatus')}",
    )

    root, _ = _sources(_convert(model))

    assert root == (
        "SELECT * FROM main.tpch.orders WHERE {FILTER_PARAMS.orders.status.filter('o_orderstatus')}"
    )


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            ("{orders_base.sql()}", "{nowhere.sql()}"),
            r"cube 'orders' reads \{nowhere.sql\(\)\}, but cube 'nowhere' is not in the model",
        ),
        (
            ("{orders_base.sql()}", "{orders.sql()}"),
            r"cube 'orders' reads \{orders.sql\(\)\} in a cycle: orders -> orders",
        ),
        (
            ("sql_table: main.tpch.orders", 'sql: "SELECT * FROM {nowhere.sql()}"'),
            r"cube 'orders_base' reads \{nowhere.sql\(\)\}, but cube 'nowhere' is not in the model",
        ),
        (
            ("sql_table: main.tpch.orders", 'sql: "SELECT * FROM t WHERE {CUBE}.x = 1"'),
            r"cube 'orders' reads \{orders_base.sql\(\)\}, whose SQL uses '\{CUBE\}'",
        ),
        (
            (
                "sql_table: main.tpch.orders",
                "sql: \"SELECT * FROM t WHERE {FILTER_PARAMS.orders_base.id.filter('x')}\"",
            ),
            r"whose SQL uses '\{FILTER_PARAMS.orders_base.id.filter\('x'\)\}'",
        ),
        (
            ("{orders_base.sql()}", "{orders.order_totals.sql()}"),
            r"cube 'orders' reads '\{orders.order_totals.sql\(\)\}', which names a join path",
        ),
        (
            ("{orders_base.sql()}", "{CUBE.sql()}"),
            r"cube 'orders' reads \{CUBE.sql\(\)\} in a cycle: orders -> orders",
        ),
    ],
)
def test_references_cube_cannot_expand_faithfully_are_refused(edit, message):
    model = _MODEL.replace(*edit, 1)

    with pytest.raises(ConversionError, match=message):
        _convert(model)


def test_a_cycle_through_two_cubes_is_refused():
    model = _MODEL.replace("sql_table: main.tpch.orders", 'sql: "SELECT * FROM {orders.sql()}"', 1)

    with pytest.raises(ConversionError, match=r"in a cycle: orders -> orders_base -> orders"):
        _convert(model)
