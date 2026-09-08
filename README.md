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

This repository is under initial development. It is not yet a supported release.

The repository is currently internal during development. A sandbox that installs
the package from GitHub will require the repository to become public or the package
to be published in a service-accessible package registry before production use.

## Supported input

The bridge accepts a mapping of safe relative file names to static Cube YAML and an
exact Cube view name. The projected public surface observes view `includes`,
`excludes`, cube/member aliases, prefixes, titles, descriptions, and AI context.
Hidden measure and computed-dimension dependencies are inlined without becoming
public Metric View members.

Publication is deliberately conservative. It rejects ambiguous/multi-root view
graphs, split views, name collisions, and selected members without a faithful
static form. Fan-out-unsafe metrics are rejected by default.

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
)

print(result.metric_view_yaml)
print(result.ossie_yaml)
print(result.source)
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
its public result contract.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```

## License and trademarks

Licensed under the Apache License 2.0. See `LICENSE` and `NOTICE`.

Apache, Apache Ossie, and Ossie are trademarks of The Apache Software Foundation.
This project is not an Apache Software Foundation release.
