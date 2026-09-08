# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Distribution metadata must retain the reviewed Apache source provenance."""

from importlib.metadata import requires

_OSSIE_COMMIT = "71222da768cf792e8820a08f77e5b37d649842d9"


def test_apache_dependencies_are_commit_pinned_in_installed_metadata():
    package_requirements = requires("cube-databricks-metric-view-bridge") or []
    apache_requirements = [item for item in package_requirements if item.startswith("apache-ossie")]

    assert len(apache_requirements) == 3
    assert all(
        f"git+https://github.com/apache/ossie.git@{_OSSIE_COMMIT}" in item
        for item in apache_requirements
    )
    assert any("#subdirectory=python" in item for item in apache_requirements)
    assert any("#subdirectory=converters/cube" in item for item in apache_requirements)
    assert any("#subdirectory=converters/databricks" in item for item in apache_requirements)
