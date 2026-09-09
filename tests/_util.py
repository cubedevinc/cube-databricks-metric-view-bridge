# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Small structural helpers for conversion tests."""

import yaml


def parse(yaml_str):
    return yaml.safe_load(yaml_str)


def model_of(ossie_yaml):
    document = parse(ossie_yaml)
    assert len(document["semantic_model"]) == 1
    return document["semantic_model"][0]


def by_name(items):
    return {item["name"]: item for item in items or []}


def expr_of(item, dialect="ANSI_SQL"):
    for entry in item["expression"]["dialects"]:
        if entry["dialect"] == dialect:
            return entry["expression"]
    raise AssertionError(f"{item['name']} has no {dialect} expression")
