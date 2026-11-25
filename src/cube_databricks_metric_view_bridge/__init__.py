# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Cube view to Databricks Metric View conversion through Apache Ossie."""

from .bridge import (
    BridgeIssue,
    ConversionResult,
    DatasetSourceResolution,
    convert_cube_view_to_databricks_metric_view,
)
from .refusal import Refusal, RefusalCode
from .view_projection import convert_cube_view_to_ossie

__version__ = "0.15.0"

__all__ = [
    "BridgeIssue",
    "ConversionResult",
    "DatasetSourceResolution",
    "Refusal",
    "RefusalCode",
    "convert_cube_view_to_databricks_metric_view",
    "convert_cube_view_to_ossie",
]
