# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Why the bridge refuses a view, grouped into families that share one remedy."""

from __future__ import annotations

import re
from enum import StrEnum

from ossie_cube import ConversionError


class RefusalCode(StrEnum):
    """Stable family codes; callers link each one to its documentation."""

    VIEW_NOT_FOUND = "CUBE_VIEW_NOT_FOUND"
    VIEW_INVALID = "CUBE_VIEW_INVALID"
    MODEL_INVALID = "CUBE_MODEL_INVALID"
    VIEW_JOIN_PATH_UNSUPPORTED = "CUBE_VIEW_JOIN_PATH_UNSUPPORTED"
    VIEW_JOIN_NOT_REPRESENTABLE = "CUBE_VIEW_JOIN_NOT_REPRESENTABLE"
    MEMBER_UNSUPPORTED = "CUBE_MEMBER_UNSUPPORTED"
    MEMBER_SQL_UNSUPPORTED = "CUBE_MEMBER_SQL_UNSUPPORTED"
    MEMBER_NAME_CONFLICT = "CUBE_MEMBER_NAME_CONFLICT"
    PRIMARY_KEY_REQUIRED = "CUBE_PRIMARY_KEY_REQUIRED"
    SOURCE_UNSUPPORTED = "CUBE_SOURCE_UNSUPPORTED"
    DATA_SOURCE_MISMATCH = "CUBE_DATA_SOURCE_MISMATCH"
    VIEW_FULLY_WITHHELD = "CUBE_VIEW_FULLY_WITHHELD"
    # Not a model problem: a caller error or a converter result the bridge cannot trust.
    VIEW_PROJECTION_FAILED = "CUBE_VIEW_PROJECTION_FAILED"


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

REPORT_FIX = "Report the view to Cube support: this is a converter limitation, not a model error."


def leave_out(member: str) -> str:
    """The fix for a member with no Metric View form: publish the view without it."""

    return f"Leave {member} out of the view: drop it from includes, or list it in excludes."


class Refusal(ConversionError):
    """A refused view: its family ``code``, the ``reason``, and the model change as ``fix``."""

    def __init__(self, code: RefusalCode, reason: str, fix: str) -> None:
        fix = fix.strip()
        if not fix.endswith(".") or "\n" in fix:
            raise ValueError(f"refusal fix must be one sentence ending with a period: {fix!r}")
        # Callers show only a message's first line, so a reason quoting a multi-line
        # parser error keeps its first line and the fix stays visible.
        lines = [line for line in _ANSI_ESCAPE.sub("", reason).splitlines() if line.strip()]
        reason = (lines[0] if lines else "").rstrip(". ")
        super().__init__(f"{reason}. {fix}")
        self.code = RefusalCode(code).value
        self.reason = reason
        self.fix = fix
