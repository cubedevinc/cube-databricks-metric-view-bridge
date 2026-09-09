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
changing grain or time-shift semantics. Metric expressions must be statically
parseable without nested query or lambda binding scopes so the bridge can prove the
source dataset of every physical column. Joined-dataset references retain their
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
