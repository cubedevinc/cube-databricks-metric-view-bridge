# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Distribution metadata must retain the reviewed Apache source provenance."""

import tomllib
from importlib.metadata import requires
from pathlib import Path

_OSSIE_COMMIT = "71222da768cf792e8820a08f77e5b37d649842d9"
_HATCHLING_VERSION = "1.27.0"


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


def test_source_build_backend_is_exactly_pinned_and_locked():
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())

    assert project["build-system"]["requires"] == [f"hatchling=={_HATCHLING_VERSION}"]
    assert any(
        package["name"] == "hatchling" and package["version"] == _HATCHLING_VERSION
        for package in lock["package"]
    )
