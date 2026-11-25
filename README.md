# Cube Databricks Metric View Bridge

Cube-owned, offline conversion bridge for publishing a selected static Cube YAML
view as Databricks Unity Catalog Metric View YAML 1.1 through Apache Ossie:

```text
Cube YAML view → projected Ossie model → Databricks Metric View YAML
```

The bridge is intentionally separate from Apache Ossie. It provides Cube with a
stable integration boundary while upstream converter work is reviewed, without
redistributing modified packages under Apache product names.

## Status

This repository is public and under initial development. It is not yet a supported
release.

## Supported input

The bridge accepts a mapping of safe relative file names to static Cube YAML and an
exact Cube view name. The projected public surface observes view `includes`,
`excludes`, cube/member aliases, prefixes, titles, descriptions, and AI context.
Hidden measure and computed-dimension dependencies are inlined without becoming
public Metric View members.

Publication is deliberately conservative. It rejects ambiguous/multi-root view
graphs, split views, name collisions, and selected members without a faithful
static form. Fan-out-unsafe metrics are rejected by default. Rolling-window and
multi-stage measure properties are rejected until they can be lowered without
changing grain or time-shift semantics. The exception is a `multi_stage: true`
measure of `type: number` with no `group_by`, `reduce_by`, `add_group_by`,
`time_shift` or `grain`: Cube computes it from the measures it references at the
query's grain, so it publishes as the same calculated measure without `multi_stage`
would. With `strict_multi_stage=False`, a selected measure that uses one of these
properties or `rolling_window`, or references such a measure through other
measures, is left out of the view and reported as a `MULTI_STAGE_MEASURE_PARKED`
issue naming it, and the rest of the view converts; a view left with no member is
still refused. Metric expressions must be statically
parseable without lambda binding scopes so the bridge can prove the source dataset
of every physical column. A measure may contain a subquery, such as a correlated
`{CUBE}.id IN (SELECT id FROM t WHERE t.ts = {CUBE}.ts)`. The subquery's own
columns are published as written, so they bind to its own `FROM` as in Cube.
Databricks does not resolve `source.col` inside a subquery, and a bare `col` there
would bind to the subquery's own table, so each source column a subquery reads
from the outer row is added to the Metric View `source` under a unique name
(`SELECT *, ts AS __cube_outer_0__ FROM ...`) and referenced by that name. A
subquery that reads a joined cube or a measure is rejected, and so is one that reads
a dimension whose SQL has a bare column: Cube pastes that SQL into the subquery,
where the column binds to the subquery's own table. Dimension and join SQL still
reject subqueries. Joined-dataset references retain their
Cube reference provenance; a physical multipart column is never inferred to be a
join merely because its first component matches a cube name.
Source-owned fields and metric columns use Databricks' explicit `source` qualifier,
so a source struct path cannot be mistaken for a same-named emitted join.

The final conversion is accepted only when every selected dimension and measure is
still present. A `source` override may select an explicitly projected or hidden
dependency dataset, but it is rejected if reorienting joins would make Databricks
drop a selected dimension through a one-to-many path. Joined computed dimensions
are published only when the bridge can qualify every column safely; ambiguous SQL
fails closed rather than returning a potentially publishable artifact.

A dimension that reads another cube, such as `sql: "{users.name}"` on `orders`, is
published with each column qualified by its join path (`users.name`), as metrics
are. A Metric View dimension must have one value per source row, so one that reads a
table behind a one-to-many join is rejected. `convert_cube_view_to_ossie` still
rejects such a dimension, since an Ossie field reads only its own dataset.

A Metric View dimension can read the dimensions before it by name, and Databricks binds
`source.id` in a later dimension to an earlier dimension named `source`, not to the
source; a dimension named like a join does the same. Such a dimension is emitted after
the others, and a view where two of them read each other's names is rejected.

A reference through a join path from the member's own cube, such as
`{CUBE.users.tier}` on `orders` or `{orders.users.countries.name}`, reads the last
cube's member joined along that path, as Cube does, so it publishes as `users.tier`.
The bridge joins that cube along the named path; a view whose own `join_path` entries
reach it another way is rejected, since a Metric View joins each cube once. A plain
reference such as `{users.tier}` names no path, so it still needs exactly one declared
join path from the view root, or a view `join_path` entry that picks one.

An aggregate measure such as `count_distinct` or `max` whose SQL or filters read a
measure that aggregates is rejected: SQL cannot nest one aggregate in another.
Reference a dimension instead. A calculated measure (`type: number`) can still combine
measures, and an aggregate over a row-level calculated measure is published.

A measure that reads a column outside any aggregate, such as
`CASE WHEN {users.tier} = 'gold' THEN {count} END`, is rejected. Cube computes it only
when that column is grouped by, and a Metric View measure must aggregate every column
it reads. Put the condition inside the aggregate, as in
`COUNT(CASE WHEN {users.tier} = 'gold' THEN {CUBE}.id END)`.

A join whose condition is not a plain column equality, such as a range on effective
dates (SCD2), an expression like `CAST(...)` or `date_trunc(...)`, or a comparison
with a literal or keyword like `true` or `current_date`, is published with its Cube
condition as the Metric View join's `on`. Member references in the condition expand
to their SQL. Such a join gets the same fan-out checks as an equality join and no
`rely.at_most_one_match`. A join `R -> X` whose condition reads one other cube `Y`
and no column of `R`, such as `{Y}.plan = {X}.plan_name`, is published nested under
`Y` when `R -> X` and `R -> Y` are both `many_to_one` or `one_to_one`: Cube joins `Y`
first, and a Metric View join's `on` reads only its parent and itself. The `R -> Y`
join is published even when the view selects nothing of `Y`. Any other condition that
reads a cube other than the two it joins, a condition that reads a measure, and a
`table.column` written without a cube reference are rejected, and so is a view that
also reaches `X` through another join path. So is a
condition that reads no column of the joined cube, such as `1 = 1`: a Metric View join
keeps one match per row, where Cube keeps every row the condition matches. The
intermediate Ossie artifact of
`convert_cube_view_to_databricks_metric_view` carries such a join as a relationship
on a carrier column, since an Ossie relationship holds only column equalities;
`convert_cube_view_to_ossie` leaves the join out, as the Cube converter does.

A Metric View's rows are its source's rows. When every join in a view's projection is
`many_to_one` or `one_to_one`, no source row repeats, so a measure of the source cube
aggregates each row once, as Cube does, whatever joined columns it reads. A `sum` of
`CASE WHEN {customers.tier} = 'gold' THEN {CUBE}.amount END` on `orders` publishes as
`SUM(CASE WHEN customers.tier = 'gold' THEN source.amount END)`, across an equality,
range or expression join. The Cube converter reports such a measure as fan-out unsafe,
since it judges the model without a source, and `convert_cube_view_to_ossie` keeps that
verdict. A measure of a joined cube repeats once per source row and is still refused, as
is a measure that references another cube's measure, any measure of a projection with a
`one_to_many` join, and any measure when `source` selects a dataset other than the root.

A cube `sql` or `sql_table` that reads another cube's SQL through
`{other_cube.sql()}` is expanded the way Cube compiles it, so the dataset source is
plain SQL: the referenced cube's `sql_table` as written, or its `sql` in parentheses
(just the table when that SQL is a bare `select * from <table>`), with references
inside it expanded first. A cube whose whole `sql` is one reference takes that
source. A reference is rejected when it is part of a cycle, names a cube missing
from the input, goes through a join path (`{orders.users.sql()}`), or leads to SQL
that uses any other Cube template (`{CUBE}`, a member, query context), which Cube
resolves as the referenced cube. Escaped braces (`\{`, `\}`) are literal text.

Cube data source ownership and physical relation completion are explicit inputs.
When `expected_data_source` is supplied, every projected and hidden dependency
cube must use that Cube data source (`default` when `data_source` is omitted).
Two-part `schema.table` sources can be completed with `default_catalog`; one-part
table sources additionally require `default_schema`. Already-qualified sources
and `SELECT`/`WITH` query sources are preserved. The result records the original
and resolved source for every projected dataset so callers can show exactly what
will be published.

Product policy remains outside this package: choosing all/specific/pattern views,
credentials, catalog/schema settings, writes, ownership, deletion, scheduling,
and feature flags belong to Cube Cloud.

## Python API

```python
from cube_databricks_metric_view_bridge import (
    convert_cube_view_to_databricks_metric_view,
)

result = convert_cube_view_to_databricks_metric_view(
    {
        "model.yml": """
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
      - {join_path: orders, includes: [id, count]}
""",
    },
    "sales",
    expected_data_source="default",
    default_catalog="samples",
)

print(result.metric_view_yaml)
print(result.ossie_yaml)
print(result.source)
for dataset in result.dataset_sources:
    print(dataset.dataset, dataset.data_source, dataset.original_source, dataset.resolved_source)
for issue in result.issues:
    print(issue.origin, issue.code, issue.message)
```

The result preserves both the normalized Ossie artifact and final Metric View YAML
so callers can verify that the selected public-member contract did not change.

## Refusals

A view the bridge cannot publish faithfully raises `Refusal`, a subclass of the
Ossie `ConversionError`. `Refusal.code` names its family, `Refusal.fix` is one
sentence saying what to change in the Cube model, and the message ends with that
sentence. Each family shares one remedy:

| Code | Refused when | Fix |
| --- | --- | --- |
| `CUBE_VIEW_NOT_FOUND` | The requested view is not in the model. | Publish a view the model defines. |
| `CUBE_VIEW_INVALID` | The view definition is malformed. | Correct the named view entry. |
| `CUBE_MODEL_INVALID` | A cube definition is malformed or cyclic. | Correct the named cube or member. |
| `CUBE_VIEW_JOIN_PATH_UNSUPPORTED` | The view has several roots, or reaches a cube along more than one path. | One root, one `join_path` per cube. |
| `CUBE_VIEW_JOIN_NOT_REPRESENTABLE` | A join or fan-out a Metric View join cannot hold. | Change the join condition, or publish a view rooted at the joined cube. |
| `CUBE_MEMBER_UNSUPPORTED` | A selected segment, hierarchy, geo, switch or sub_query dimension, custom granularity, split view entry, or measure with no Metric View form. | Leave the member out of the view. |
| `CUBE_MEMBER_SQL_UNSUPPORTED` | A member's SQL cannot be qualified or rendered for one Metric View namespace. | Rewrite the named member's SQL, or leave it out. |
| `CUBE_MEMBER_NAME_CONFLICT` | Published names or join aliases collide ignoring case, or dimensions read each other's names. | Rename one in the view. |
| `CUBE_PRIMARY_KEY_REQUIRED` | A bare `count` measure's cube has no primary key. | Mark the key dimension `primary_key: true`. |
| `CUBE_SOURCE_UNSUPPORTED` | A cube's `sql_table` or `sql` is not a static table or SELECT. | Use `catalog.schema.table` or a plain SELECT. |
| `CUBE_DATA_SOURCE_MISMATCH` | A cube belongs to another data source than the one requested. | Align the cube's `data_source`. |
| `CUBE_VIEW_FULLY_WITHHELD` | No selected member has a Metric View form. | Include at least one publishable member. |
| `CUBE_VIEW_PROJECTION_FAILED` | A caller error, or converter output the bridge cannot trust. | Report the view; nothing in the model is known to cause it. |

`ConversionError`s raised inside the Ossie converters carry no family yet.

## Dependency and provenance policy

The distributable package metadata and development lockfile pin Apache Ossie core,
Cube, and Databricks converters to commit
`71222da768cf792e8820a08f77e5b37d649842d9`, the reviewed head of
`apache/ossie#289`. Apache packages remain unmodified and are installed from the
Apache repository. Cube-specific projection and compatibility behavior lives under
the separately branded `cube_databricks_metric_view_bridge` namespace.

The bridge currently uses explicitly isolated private converter primitives because
the upstream Cube converter does not yet expose a view-projection API. That coupling
is contained within this package and covered by integration tests. Once a suitable
public API is accepted upstream, the bridge can change internally without changing
its public result contract. The pinned Databricks converter also exposes diagnostics
only through a private warning hook; the bridge routes that hook into a context-local
sink so it does not capture or suppress unrelated process warnings.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run ruff check .
uv run pytest
uv build
```

## License and trademarks

Licensed under the Apache License 2.0. See `LICENSE` and `NOTICE`.

Apache, Apache Ossie, and Ossie are trademarks of The Apache Software Foundation.
This project is not an Apache Software Foundation release.
