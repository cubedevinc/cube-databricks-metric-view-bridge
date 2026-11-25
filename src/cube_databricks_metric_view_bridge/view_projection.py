# Copyright 2026 Cube Dev, Inc.
#
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0

"""Project a Cube view into a publication-ready Apache Ossie model.

The ordinary Cube import is deliberately lossless: it converts every cube member
and parks view curation in ``custom_extensions``.  A publication target needs the
opposite boundary.  It must expose exactly the members the selected Cube view
exposes, under their view aliases, so a spoke such as Databricks Metric Views does
not accidentally publish private implementation members.
"""

import copy
import dataclasses
import functools
import re
from collections import deque

from ossie_cube._common import (
    _CUBE_REF_RE,
    CALCULATED_MEASURE_TYPES,
    OSSIE_VERSION,
    VENDOR,
    ConversionError,
    cube_sql_to_ossie,
    dump_yaml,
    filtered_operand,
    load_yaml,
    normalize_identifier,
    read_stash,
    snake,
    snake_keys,
    sub_outside_quotes,
    write_stash,
)
from ossie_cube.converter_issues import IssueLog, IssueType
from ossie_cube.cube_to_osi import (
    _RELATIONSHIP_ALIASES,
    _ai_context_from_meta,
    _as_named_list,
    _collect,
    _decompose_join_sql,
    _is_generated_part,
    _MeasureResolver,
    _primary_key_of,
    convert_cube_to_ossie,
    parked_of,
)
from ossie_cube.expressions import _REF_SENTINEL, _mask_references, has_top_level_operator
from sqlglot import Dialect, exp, parse_one
from sqlglot.errors import SqlglotError
from sqlglot.tokens import TokenType


@dataclasses.dataclass(frozen=True)
class _SelectedMember:
    cube: str
    source_name: str
    output_name: str
    kind: str
    override: dict


@dataclasses.dataclass(frozen=True)
class _ProjectionProvenance:
    metric_templates: dict[str, str]
    # Selected dimensions that read another cube, by output name.
    dimension_templates: dict[str, str]
    dataset_markers: dict[str, str]
    dataset_data_sources: dict[str, str]
    # Carrier column -> (declaring cube, joined cube, condition template).
    join_conditions: dict[str, tuple[str, str, str]]


def convert_cube_view_to_ossie(
    files, view, source=None, strict_fanout=True, strict_multi_stage=True
):
    """Convert one Cube view's public projection to Apache Ossie YAML.

    Returns ``(ossie_yaml_str, resolved_source, IssueLog)``. ``resolved_source``
    is the common root of the view's join paths unless ``source`` explicitly
    selects another dataset in the projection.  It can be passed directly to a
    downstream converter whose format needs an explicit fact/source dataset.

    Unlike :func:`convert_cube_to_ossie`, this is a publication transform rather
    than a round-trip transform: only included view dimensions and measures are
    emitted, and Cube view aliases/prefixes become their Ossie names.  Unsafe
    fan-out is strict by default because publishing a metric that can over-count
    is worse than refusing it with a precise error.

    A selected measure with no faithful Metric View form (a stage modifier, a
    non-``number`` multi-stage type or ``rolling_window``), or one that reaches
    such a measure through its references, refuses the view by default. With
    ``strict_multi_stage=False`` each such measure is left out and reported as a
    ``MULTI_STAGE_MEASURE_PARKED`` issue naming it, and the rest converts.
    """
    projected, resolved_source, issues, _ = _convert_cube_view_to_ossie_with_provenance(
        files,
        view,
        source=source,
        strict_fanout=strict_fanout,
        strict_multi_stage=strict_multi_stage,
    )
    return projected, resolved_source, issues


def _convert_cube_view_to_ossie_with_provenance(
    files, view, source=None, strict_fanout=True, metric_view=False, strict_multi_stage=True
):
    if not isinstance(files, dict) or not files:
        raise ConversionError("expected a non-empty mapping of {filename: YAML}")
    if not isinstance(view, str) or not view.strip():
        raise ConversionError("view projection requires a non-empty view name")

    collection_issues = IssueLog()
    cubes, _, views, _, _ = _collect(files, collection_issues)
    cubes, path_references, plain_references = _inline_join_path_references(cubes)
    if view not in views:
        raise ConversionError(
            f"requested view '{view}' not found; views present: {sorted(views) or 'none'}"
        )
    if not cubes:
        raise ConversionError("view projection requires the cubes referenced by the view")

    selected_view = views[view]
    selected, paths, required_cubes, joins = _resolve_view(selected_view, view, cubes)
    parked = {} if strict_multi_stage else _unpublishable_measures(cubes, selected)
    if parked:
        if all((member.cube, member.source_name) in parked for member in selected):
            raise ConversionError(next(iter(parked.values())))
        selected_view = _excluding(selected_view, parked)
        selected, paths, required_cubes, joins = _resolve_view(selected_view, view, cubes)

    needed_measures = _measure_closure(
        cubes,
        {(member.cube, member.source_name) for member in selected if member.kind == "measure"},
    )
    selected_dimensions = {
        (member.cube, member.source_name) for member in selected if member.kind == "dimension"
    }
    needed_dimensions, raw_dependency_cubes = _dimension_dependency_closure(
        cubes,
        needed_measures,
        selected_dimensions,
    )
    dependency_cubes = {
        cube_name for cube_name, _ in needed_measures | needed_dimensions
    } | raw_dependency_cubes
    needed_members = needed_measures | needed_dimensions
    required_cubes, joins = _expand_dependency_graph(
        view,
        cubes,
        paths[0][0],
        required_cubes,
        joins,
        dependency_cubes,
        {path for member in needed_members for path in path_references.get(member, ())},
        {cube for member in needed_members for cube in plain_references.get(member, ())},
    )
    resolved_source = _resolve_source(view, source, paths[0][0], required_cubes)
    projected_cubes = _project_cubes(cubes, required_cubes, joins, needed_measures)
    carriers = _condition_join_carriers(projected_cubes) if metric_view else {}
    synthetic = dump_yaml(
        {
            "cubes": [
                _with_carrier_joins(cube_name, cube, carriers)
                for cube_name, cube in projected_cubes.items()
            ],
            "views": [selected_view],
        }
    )
    ossie_yaml, conversion_issues = convert_cube_to_ossie(
        {"model.yml": synthetic}, view=view, strict_fanout=False
    )
    model = load_yaml(ossie_yaml, "projected Ossie model")["semantic_model"][0]

    provenance = _apply_projection(model, projected_cubes, selected, carriers, metric_view)
    source_grain = (
        _source_grain_measures(cubes, projected_cubes, resolved_source, needed_measures)
        if metric_view and resolved_source == paths[0][0]
        else frozenset()
    )
    issues = _publication_issues(
        collection_issues, conversion_issues, selected, required_cubes, source_grain
    )
    for (cube_name, measure_name), message in parked.items():
        issues.add(IssueType.MULTI_STAGE_MEASURE_PARKED, f"{cube_name}.{measure_name}", message)
    if strict_fanout:
        unsafe = issues.of_type(IssueType.FANOUT_UNSAFE_METRIC)
        if unsafe:
            raise ConversionError(f"{unsafe[0]} (refused under strict mode)")

    return (
        dump_yaml({"version": OSSIE_VERSION, "semantic_model": [model]}),
        resolved_source,
        issues,
        provenance,
    )


def _resolve_view(view, view_name, cubes):
    entries = view.get("cubes") or []
    if not isinstance(entries, list) or not entries:
        raise ConversionError(f"view '{view_name}' has no cube projections")

    selected = []
    paths = []
    required = []
    joins = {}
    parent_of = {}
    path_of = {}
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ConversionError(f"view '{view_name}': cubes[{index}] must be a mapping")
        entry = snake_keys(raw_entry)
        if entry.get("split"):
            raise ConversionError(
                f"view '{view_name}': split view projections are not supported for "
                "publication; publish each split view explicitly"
            )
        join_path = entry.get("join_path")
        if not isinstance(join_path, str) or not join_path.strip():
            raise ConversionError(f"view '{view_name}': cubes[{index}] has no non-empty join_path")
        parts = join_path.split(".")
        if any(not part for part in parts):
            raise ConversionError(f"view '{view_name}': invalid join_path '{join_path}'")
        if len(set(parts)) != len(parts):
            raise ConversionError(
                f"view '{view_name}': cyclic join_path '{join_path}' is not publishable"
            )
        missing = [part for part in parts if part not in cubes]
        if missing:
            raise ConversionError(
                f"view '{view_name}': join_path '{join_path}' references unknown "
                f"cube '{missing[0]}'"
            )
        paths.append(parts)
        for part in parts:
            if part not in required:
                required.append(part)
        leaf = parts[-1]
        prior_path = path_of.setdefault(leaf, tuple(parts))
        if prior_path != tuple(parts):
            raise ConversionError(
                f"view '{view_name}': cube '{leaf}' is reached through multiple "
                "join paths; publication would make its member expressions ambiguous"
            )
        for left, right in zip(parts, parts[1:], strict=False):
            prior_parent = parent_of.setdefault(right, left)
            if prior_parent != left:
                raise ConversionError(
                    f"view '{view_name}': cube '{right}' has multiple parents; "
                    "diamond join projections are not supported"
                )
            join = _find_join(cubes[left], left, right)
            joins.setdefault(left, {})[right] = join

        selected.extend(_members_from_entry(view_name, entry, leaf, cubes[leaf]))

    roots = {path[0] for path in paths}
    if len(roots) != 1:
        raise ConversionError(
            f"view '{view_name}' has multiple join roots {sorted(roots)}; select a "
            "single-root view for publication"
        )
    if not selected:
        raise ConversionError(f"view '{view_name}' exposes no dimensions or measures")

    seen = {}
    deduped = []
    exact = set()
    for member in selected:
        key = normalize_identifier(member.output_name)
        identity = (member.cube, member.source_name, member.output_name, member.kind)
        if identity in exact:
            continue
        exact.add(identity)
        if key in seen:
            other = seen[key]
            raise ConversionError(
                f"view '{view_name}': published name '{member.output_name}' from "
                f"'{member.cube}.{member.source_name}' collides with "
                f"'{other.cube}.{other.source_name}' (identifiers are "
                "case-insensitive); use an alias or prefix"
            )
        seen[key] = member
        deduped.append(member)
    return deduped, paths, required, joins


def _find_join(cube, own_name, target):
    matches = [
        join
        for join in _as_named_list(cube.get("joins"), f"cube '{own_name}' joins")
        if join.get("name") == target
    ]
    if not matches:
        raise ConversionError(
            f"view join_path requires '{own_name}.{target}', but cube "
            f"'{own_name}' declares no join to '{target}'"
        )
    if len(matches) > 1:
        raise ConversionError(f"cube '{own_name}' declares more than one join to '{target}'")
    return matches[0]


def _members_from_entry(view_name, entry, cube_name, cube):
    prefix = entry.get("prefix", False)
    if not isinstance(prefix, bool):
        raise ConversionError(f"view '{view_name}': prefix for cube '{cube_name}' must be boolean")
    alias = entry["alias"] if "alias" in entry else cube_name
    if not isinstance(alias, str) or not alias:
        raise ConversionError(
            f"view '{view_name}': alias for cube '{cube_name}' must be a non-empty string"
        )

    collections = {
        kind: {
            member.get("name"): member
            for member in _as_named_list(cube.get(key), f"cube '{cube_name}' {key}")
            if member.get("name")
        }
        for kind, key in (
            ("dimension", "dimensions"),
            ("measure", "measures"),
            ("segment", "segments"),
            ("hierarchy", "hierarchies"),
        )
    }
    includes = entry.get("includes", [])
    if includes == "*":
        requested = [
            (name, name, {})
            for kind, members in collections.items()
            for name, definition in members.items()
            if definition.get("public") is not False
            and (kind != "measure" or not _is_generated_part(definition))
        ]
    elif isinstance(includes, list):
        requested = []
        for include in includes:
            if isinstance(include, str):
                source_name, output_name, override = include, include, {}
            elif isinstance(include, dict):
                source_name = include.get("name")
                output_name = include["alias"] if "alias" in include else source_name
                if not isinstance(source_name, str) or not source_name:
                    raise ConversionError(
                        f"view '{view_name}': include for cube '{cube_name}' must name a member"
                    )
                if not isinstance(output_name, str) or not output_name:
                    raise ConversionError(
                        f"view '{view_name}': alias for member '{source_name}' must be a "
                        "non-empty string"
                    )
                override = include
            else:
                raise ConversionError(
                    f"view '{view_name}': includes for cube '{cube_name}' must "
                    "contain strings or mappings"
                )
            if "." in output_name or "." in source_name:
                raise ConversionError(
                    f"view '{view_name}': paths are not allowed in includes ('{source_name}')"
                )
            requested.append((source_name, output_name, override))
    else:
        raise ConversionError(
            f"view '{view_name}': includes for cube '{cube_name}' must be '*' or a list"
        )

    excludes = entry.get("excludes")
    if excludes is None:
        excludes = []
    if not isinstance(excludes, list) or not all(isinstance(v, str) for v in excludes):
        raise ConversionError(
            f"view '{view_name}': excludes for cube '{cube_name}' must be a list of member names"
        )
    known_members = {name for members in collections.values() for name in members}
    unknown_excludes = [name for name in excludes if name not in known_members]
    if unknown_excludes:
        raise ConversionError(
            f"view '{view_name}': excluded member '{cube_name}.{unknown_excludes[0]}' "
            "does not exist"
        )
    out = []
    for source_name, output_name, override in requested:
        found = [
            (kind, members[source_name])
            for kind, members in collections.items()
            if source_name in members
        ]
        if not found:
            raise ConversionError(
                f"view '{view_name}': member '{cube_name}.{source_name}' does not exist"
            )
        if source_name in excludes:
            continue
        kind, definition = found[0]
        if kind == "measure" and _is_generated_part(definition):
            raise ConversionError(
                f"view '{view_name}': measure '{cube_name}.{source_name}' is an internal "
                "Ossie-generated aggregate helper and cannot be published directly"
            )
        if kind in ("segment", "hierarchy"):
            raise ConversionError(
                f"view '{view_name}': selected {kind} "
                f"'{cube_name}.{source_name}' has no Databricks Metric View "
                "equivalent; exclude it from the published view"
            )
        dtype = str(definition.get("type") or "").lower()
        if kind == "dimension" and (dtype in ("geo", "switch") or definition.get("sub_query")):
            feature = "sub_query" if definition.get("sub_query") else dtype
            raise ConversionError(
                f"view '{view_name}': selected dimension "
                f"'{cube_name}.{source_name}' uses {feature}, which has no safe "
                "publication form"
            )
        final_name = f"{alias}_{output_name}" if prefix else output_name
        out.append(_SelectedMember(cube_name, source_name, final_name, kind, override))
    return out


def _resolve_source(view_name, requested, default, required):
    if requested is None:
        return default
    if not isinstance(requested, str) or not requested:
        raise ConversionError(f"view '{view_name}': source must be a dataset name")
    if requested not in required:
        raise ConversionError(
            f"view '{view_name}': source '{requested}' is not in the projected datasets {required}"
        )
    return requested


def _project_cubes(cubes, required, joins, needed_measures):
    projected = {}
    for cube_name in required:
        cube = copy.deepcopy(cubes[cube_name])
        cube["joins"] = list((joins.get(cube_name) or {}).values())
        cube["measures"] = [
            measure
            for measure in _as_named_list(cube.get("measures"), f"cube '{cube_name}' measures")
            if (cube_name, measure.get("name")) in needed_measures
        ]
        for measure in cube["measures"]:
            # _measure_closure admits multi_stage only on a calculated measure with no
            # stage modifier; the Ossie converter would otherwise park it.
            measure.pop("multi_stage", None)
        for key in ("sql", "sql_table"):
            if isinstance(cube.get(key), str):
                cube[key] = _inline_cube_sql(cube[key], cubes, (cube_name,))
        projected[cube_name] = cube
    return projected


_CUBE_SQL_CALL_RE = re.compile(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*sql\s*\(\s*\)\s*\}")
# A '.sql()' call left after expansion, such as one on a join path ('{a.b.sql()}').
_CUBE_SQL_ANY_CALL_RE = re.compile(r"\{[^{}]*\.\s*sql\s*\(\s*\)\s*\}")
# Cube's BaseQuery.cubeSql reads a cube whose SQL is a bare 'select * from <table>' as the table.
_SELECT_STAR_FROM_RE = re.compile(r"^\s*select\s+\*\s+from\s+([a-zA-Z0-9_\-`\".*]+)\s*$", re.I)
# Cube YAML writes a literal brace as '\{' or '\}'.
_ESCAPED_BRACE_RE = re.compile(r"\\[{}]")


def _inline_cube_sql(sql, cubes, stack):
    """Expand each ``{other_cube.sql()}`` in a cube's SQL as Cube does when it compiles a query.

    A dataset source must be plain SQL. ``stack`` is the chain of cubes whose SQL is
    being expanded, the cube that holds ``sql`` last.
    """

    whole = _CUBE_SQL_CALL_RE.fullmatch(sql.strip())
    if whole:
        result = _referenced_cube_sql(whole.group(1), cubes, stack)[0]
    else:

        def expand(match):
            text, is_table = _referenced_cube_sql(match.group(1), cubes, stack)
            if is_table:
                return text
            select_star = _SELECT_STAR_FROM_RE.match(text)
            return select_star.group(1) if select_star else f"({text})"

        result = _CUBE_SQL_CALL_RE.sub(expand, sql)
    if left := _CUBE_SQL_ANY_CALL_RE.search(result):
        raise ConversionError(
            f"cube '{stack[-1]}' reads '{left.group(0)}', which names a join path; "
            "reference the cube itself, as in '{other_cube.sql()}'"
        )
    return result


def _referenced_cube_sql(target, cubes, stack):
    """A cube's table or expanded SQL, and whether it is a table."""

    referrer = stack[-1]
    reference = f"{{{target}.sql()}}"
    if target in ("CUBE", "TABLE"):
        target = referrer
    if target in stack:
        chain = " -> ".join((*stack, target))
        raise ConversionError(f"cube '{referrer}' reads {reference} in a cycle: {chain}")
    cube = cubes.get(target)
    if cube is None:
        raise ConversionError(
            f"cube '{referrer}' reads {reference}, but cube '{target}' is not in the model"
        )

    if isinstance(cube.get("sql_table"), str):
        text = _inline_cube_sql(cube["sql_table"], cubes, (*stack, target))
        is_table = True
    elif isinstance(cube.get("sql"), str):
        text = _inline_cube_sql(cube["sql"], cubes, (*stack, target))
        is_table = False
    else:
        raise ConversionError(f"cube '{referrer}' reads {reference}, but '{target}' has no SQL")
    # Cube resolves the rest of the referenced SQL ({CUBE}, members, context) as that
    # cube; inlined into another cube's source, the same text would mean something else.
    if leftover := _CUBE_REF_RE.search(_ESCAPED_BRACE_RE.sub("", text)):
        raise ConversionError(
            f"cube '{referrer}' reads {reference}, whose SQL uses '{leftover.group(0)}', "
            "which only Cube can resolve"
        )
    return text, is_table


def _is_plain_column(name):
    """Whether SQL reads ``name`` as a column; the pinned Cube converter takes any
    bare word in a join equality for one, ``true`` and ``current_date`` included."""

    try:
        return isinstance(parse_one(name, read="databricks"), exp.Column)
    except SqlglotError:
        return False


def _condition_join_carriers(cubes):
    """Name a carrier column for each join whose condition has no Ossie relationship form.

    An Ossie relationship is an equi-join on column pairs, so the Cube converter
    parks a range or expression condition and drops the join with every cube
    behind it. Such a join travels instead as an equality on a carrier column no
    table has: the converters keep it in the join graph, with the same fan-out
    checks as any other join, and the bridge writes the Cube condition into the
    Metric View join. Returns {(cube, joined cube): carrier}.
    """

    base = "__cube_dmv_join_"
    corpus = repr(cubes).casefold()
    while base in corpus:
        base = "_" + base
    carriers = {}
    for cube_name, cube in cubes.items():
        for join in _as_named_list(cube.get("joins"), f"cube '{cube_name}' joins"):
            relationship = _RELATIONSHIP_ALIASES.get(snake(str(join.get("relationship") or "")))
            sql = join.get("sql")
            if relationship is None or not isinstance(sql, str):
                continue
            pairs = _decompose_join_sql(sql, cube_name, join.get("name"), "", cubes, IssueLog())
            if pairs is None or not all(
                _is_plain_column(column) for pair in pairs for column in pair
            ):
                carriers[(cube_name, join.get("name"))] = f"{base}{len(carriers)}__"
    return carriers


def _with_carrier_joins(cube_name, cube, carriers):
    joins = []
    for join in _as_named_list(cube.get("joins"), f"cube '{cube_name}' joins"):
        carrier = carriers.get((cube_name, join.get("name")))
        if carrier:
            join = {**join, "sql": f"{{CUBE}}.{carrier} = {{{join['name']}}}.{carrier}"}
        joins.append(join)
    return {**cube, "joins": joins}


def _expand_dependency_graph(
    view_name, cubes, root, required, joins, dependency_cubes, member_paths, plain_reads
):
    """Add the cubes that hidden member dependencies need.

    A join-path reference such as `{CUBE.users.name}` names the path Cube joins
    along, which is followed from its first cube. A cube read without a path
    (`plain_reads`) must be reachable from the root along one declared path, even
    when a join-path reference already placed it: Cube joins a plain reference
    along the shortest path, not along another member's.
    """

    expanded_required = list(required)
    expanded_joins = {cube_name: dict(targets) for cube_name, targets in joins.items()}
    parent_of = {
        target: cube_name for cube_name, targets in expanded_joins.items() for target in targets
    }

    def follows(path):
        return all(
            right != root and parent_of.get(right, left) == left
            for left, right in zip(path, path[1:], strict=False)
        )

    pending = set(member_paths)
    while ready := sorted(path for path in pending if path[0] in expanded_required):
        pending.difference_update(ready)
        for path in ready:
            if not follows(path):
                continue
            for left, right in zip(path, path[1:], strict=False):
                parent_of[right] = left
                expanded_joins.setdefault(left, {})[right] = _find_join(cubes[left], left, right)
                if right not in expanded_required:
                    expanded_required.append(right)

    placed_by_paths = set(expanded_required) - set(required)
    for target in sorted(dependency_cubes):
        if target in expanded_required and not (
            target in placed_by_paths and target in plain_reads
        ):
            continue
        paths = _declared_join_paths(cubes, root, target)
        if not paths:
            raise ConversionError(
                f"view '{view_name}': selected members depend on cube '{target}', "
                f"but it has no declared join path from view root '{root}'"
            )
        if len(paths) > 1:
            raise ConversionError(
                f"view '{view_name}': selected members depend on cube '{target}' through "
                "multiple declared join paths; add an explicit view join_path to disambiguate it"
            )
        for left, right in zip(paths[0], paths[0][1:], strict=False):
            prior_parent = parent_of.setdefault(right, left)
            if prior_parent != left:
                raise ConversionError(
                    f"view '{view_name}': dependency cube '{right}' would have multiple "
                    "parents; add an explicit view join_path to disambiguate it"
                )
            expanded_joins.setdefault(left, {})[right] = _find_join(cubes[left], left, right)
            if right not in expanded_required:
                expanded_required.append(right)

    for path in sorted(member_paths):
        if any(
            right not in expanded_joins.get(left, {})
            for left, right in zip(path, path[1:], strict=False)
        ):
            raise ConversionError(
                f"view '{view_name}': selected members read cube '{path[-1]}' through join "
                f"path '{'.'.join(path)}', which the view's joins do not follow; a Metric "
                "View joins each cube along one path"
            )
    return expanded_required, expanded_joins


def _declared_join_paths(cubes, start, target):
    """Return one path, plus an alternative when the route is ambiguous.

    Enumerating every simple path is factorial for a dense cyclic component,
    especially when ``target`` is unreachable. A breadth-first reachability
    pass finds one path in linear time. Any distinct path must omit at least one
    edge from that first path, so repeating the bounded search with each of
    those edges removed is sufficient to detect ambiguity in polynomial time.
    """

    adjacency = {}
    for cube_name, cube in cubes.items():
        adjacency[cube_name] = [
            join.get("name")
            for join in _as_named_list(cube.get("joins"), f"cube '{cube_name}' joins")
            if join.get("name") in cubes
        ]

    def find_path(blocked_edge=None):
        parents = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, ()):
                if (current, neighbor) == blocked_edge or neighbor in parents:
                    continue
                parents[neighbor] = current
                if neighbor == target:
                    path = [target]
                    while parents[path[-1]] is not None:
                        path.append(parents[path[-1]])
                    return list(reversed(path))
                queue.append(neighbor)
        return None

    first = find_path()
    if first is None:
        return []
    for edge in zip(first, first[1:], strict=False):
        alternative = find_path(edge)
        if alternative is not None:
            return [first, alternative]
    return [first]


class _NoPublicationForm(ConversionError):
    """The first measure a closure reaches that no Metric View can compute."""

    def __init__(self, measure, feature):
        super().__init__(
            f"measure '{measure[0]}.{measure[1]}' {feature} with no faithful Databricks Metric "
            "View publication form"
        )
        self.measure = measure
        self.feature = feature


def _unpublishable_measures(cubes, selected):
    """Each selected measure that reaches a measure with no Metric View form, with why."""
    measures = [(item.cube, item.source_name) for item in selected if item.kind == "measure"]
    try:
        _measure_closure(cubes, set(measures))
        return {}
    except _NoPublicationForm:
        pass
    parked = {}
    for key in dict.fromkeys(measures):
        try:
            _measure_closure(cubes, {key})
        except _NoPublicationForm as refusal:
            root = refusal.measure
            parked[key] = (
                str(refusal)
                if root == key
                else f"measure '{key[0]}.{key[1]}' depends on measure '{root[0]}.{root[1]}', "
                f"which {refusal.feature} with no faithful Databricks Metric View publication form"
            )
    return parked


def _excluding(view, measures):
    """The view with these (cube, measure) pairs added to its entries' own excludes."""
    result = copy.deepcopy(view)
    for entry in result.get("cubes") or []:
        leaf = str(snake_keys(entry).get("join_path") or "").split(".")[-1]
        excludes = list(entry.get("excludes") or [])
        entry["excludes"] = excludes + [
            name for cube_name, name in measures if cube_name == leaf and name not in excludes
        ]
    return result


def _measure_closure(cubes, initial):
    measures = {}
    for cube_name, cube in cubes.items():
        for measure in _as_named_list(cube.get("measures"), f"cube '{cube_name}' measures"):
            measures[(cube_name, measure.get("name"))] = measure
    needed = set(initial)
    queue = list(initial)
    while queue:
        cube_name, measure_name = queue.pop()
        measure = measures.get((cube_name, measure_name))
        if measure is None:
            continue
        if measure.get("rolling_window") is not None:
            raise _NoPublicationForm((cube_name, measure_name), "uses rolling_window")
        unsupported_multi_stage = _active_multi_stage_properties(
            measure,
            cube_name,
            measure_name,
        )
        if unsupported_multi_stage:
            properties = ", ".join(unsupported_multi_stage)
            raise _NoPublicationForm(
                (cube_name, measure_name),
                f"uses unsupported Cube multi-stage properties ({properties})",
            )
        for text in _measure_expression_texts(measure):
            for target in _member_references(text, cube_name, cubes):
                if target in measures and target not in needed:
                    needed.add(target)
                    queue.append(target)
    return needed


def _active_multi_stage_properties(measure, cube_name, measure_name):
    active = []
    if "multi_stage" in measure:
        value = measure["multi_stage"]
        if not isinstance(value, bool):
            raise ConversionError(
                f"measure '{cube_name}.{measure_name}' has malformed multi_stage; "
                "expected a boolean"
            )
        if value:
            active.append("multi_stage")

    for property_name in ("group_by", "reduce_by", "add_group_by", "time_shift"):
        if property_name not in measure:
            continue
        value = measure[property_name]
        if not isinstance(value, list):
            raise ConversionError(
                f"measure '{cube_name}.{measure_name}' has malformed {property_name}; "
                "expected a list"
            )
        if value:
            active.append(property_name)

    if "grain" in measure:
        value = measure["grain"]
        if not isinstance(value, dict):
            raise ConversionError(
                f"measure '{cube_name}.{measure_name}' has malformed grain; expected a mapping"
            )
        if value:
            active.append("grain")

    measure_type = str(measure.get("type") or "").lower()
    if active == ["multi_stage"] and measure_type == "number":
        # With no stage modifier, Cube computes a number measure from the measures it
        # references at the query's grain: the same result as a calculated measure.
        return []
    return active


def _dimension_dependency_closure(cubes, needed_measures, initial_dimensions):
    """Find dimensions and raw joined datasets needed by published members."""

    measures = {}
    dimensions = {}
    for cube_name, cube in cubes.items():
        for measure in _as_named_list(cube.get("measures"), f"cube '{cube_name}' measures"):
            measures[(cube_name, measure.get("name"))] = measure
        for dimension in _as_named_list(cube.get("dimensions"), f"cube '{cube_name}' dimensions"):
            dimensions[(cube_name, dimension.get("name"))] = dimension

    needed = set(initial_dimensions)
    queue = list(initial_dimensions)
    raw_dependency_cubes = set()
    for cube_name, measure_name in needed_measures:
        measure = measures.get((cube_name, measure_name))
        if measure is None:
            continue
        measure_type = str(measure.get("type") or "").lower().replace("-", "_")
        if measure_type == "count" and measure.get("sql") is None:
            for primary_key, is_dimension in _publication_primary_keys(
                cubes[cube_name],
                cube_name,
            ):
                if not is_dimension:
                    continue
                target = (cube_name, primary_key)
                if target not in needed:
                    needed.add(target)
                    queue.append(target)
        for text in _measure_expression_texts(measure):
            for kind, target in _cube_references(text, cube_name, cubes):
                if kind == "dataset":
                    raw_dependency_cubes.add(target)
                elif target in dimensions and target not in needed:
                    needed.add(target)
                    queue.append(target)

    while queue:
        cube_name, dimension_name = queue.pop()
        dimension = dimensions[(cube_name, dimension_name)]
        for text in _dimension_expression_texts(dimension):
            for kind, target in _cube_references(text, cube_name, cubes):
                if kind == "dataset":
                    raw_dependency_cubes.add(target)
                elif target in measures:
                    raise ConversionError(
                        f"dimension '{cube_name}.{dimension_name}' references measure "
                        f"'{target[0]}.{target[1]}'; correlated/sub-query dimensions "
                        "cannot be published safely"
                    )
                elif target in dimensions and target not in needed:
                    needed.add(target)
                    queue.append(target)
    return needed, raw_dependency_cubes


def _publication_primary_keys(cube, cube_name):
    """Return primary-key names paired with whether each names a Cube dimension."""

    primary_keys = _primary_key_of(cube, cube_name)
    recorded_columns = parked_of(cube.get("meta")).get("primary_key")
    return [(primary_key, not bool(recorded_columns)) for primary_key in primary_keys]


def _measure_expression_texts(measure):
    texts = [measure.get("sql")]
    texts.extend(f.get("sql") for f in measure.get("filters") or [] if isinstance(f, dict))
    return (text for text in texts if isinstance(text, str))


def _parse_masked(sql):
    """Cube SQL parsed with each `{...}` reference as a sentinel column, and {sentinel: body}."""

    masked, references = _mask_references(str(sql))
    sentinels = {
        _REF_SENTINEL.format(index): reference[1:-1].strip()
        for index, reference in enumerate(references)
    }
    return parse_one(masked, read="databricks"), sentinels


def _has_bare_column(sql):
    """Whether Cube pastes ``sql`` with a column that binds to whatever scope it lands in.

    Cube prefixes only SQL that is a single identifier with the cube's alias.
    """

    if re.fullmatch(r"\s*[_a-zA-Z][_a-zA-Z0-9]*\s*", str(sql)):
        return False
    try:
        tree, sentinels = _parse_masked(sql)
    except SqlglotError:
        return True
    return any(
        not column.table and column.name not in sentinels for column in tree.find_all(exp.Column)
    )


def _references_inside_subqueries(sql):
    """The `{...}` reference bodies that Cube SQL places inside a subquery."""

    try:
        tree, sentinels = _parse_masked(sql)
    except SqlglotError:
        return []
    return [
        sentinels[column.name]
        for column in tree.find_all(exp.Column)
        if column.name in sentinels and column.find_ancestor(exp.Query) is not None
    ]


def _dimension_expression_texts(dimension):
    sql = dimension.get("sql")
    if isinstance(sql, str):
        yield sql
    case = dimension.get("case")
    if not isinstance(case, dict):
        return
    holders = [item for item in case.get("when") or [] if isinstance(item, dict)]
    otherwise = case.get("else")
    if isinstance(otherwise, dict):
        holders.append(otherwise)
    for holder in holders:
        condition = holder.get("sql")
        if isinstance(condition, str):
            yield condition
        label = holder.get("label")
        if isinstance(label, dict) and isinstance(label.get("sql"), str):
            yield label["sql"]


def _inline_join_path_references(cubes):
    """Rewrite `{CUBE.users.name}` as `{users.name}`, keeping the join path it names.

    Cube reads a member reference through a chain of joined cubes as that cube's
    member, joined along the chain. The bridge joins each cube along one path, so
    each member's chains are kept to follow, and the other cubes it reads without
    a path are kept to resolve as before.
    """

    paths = {}
    plain = {}
    result = {}
    for cube_name, cube in cubes.items():
        result[cube_name] = cube
        for collection in ("dimensions", "measures"):
            members = _as_named_list(cube.get(collection), f"cube '{cube_name}' {collection}")
            rewritten = []
            for member in members:
                key = (cube_name, member.get("name"))
                paths[key], plain[key] = [], set()
                rewrite = functools.partial(
                    _inline_paths,
                    own_cube=cube_name,
                    cubes=cubes,
                    paths=paths[key],
                    plain=plain[key],
                )
                rewritten.append(_rewrite_sql_values(member, rewrite))
            if rewritten != members:
                result[cube_name] = {**result[cube_name], collection: rewritten}
    return result, paths, plain


def _rewrite_sql_values(value, rewrite):
    if isinstance(value, list):
        return [_rewrite_sql_values(item, rewrite) for item in value]
    if not isinstance(value, dict):
        return value
    rewritten = {}
    for key, item in value.items():
        if key == "sql" and isinstance(item, str):
            rewritten[key] = rewrite(item)
        elif key == "meta":
            rewritten[key] = item
        else:
            rewritten[key] = _rewrite_sql_values(item, rewrite)
    return rewritten


def _inline_paths(text, *, own_cube, cubes, paths, plain):
    def replace(match):
        if match.start() and text[match.start() - 1] == "\\":
            return match.group(0)
        parts = [part.strip() for part in match.group(1).split(".")]
        head = own_cube if parts[0] in ("CUBE", "TABLE") else parts[0]
        path = [head]
        # Only a path from the member's own cube is followed. Cube reads a member of
        # the current cube before a joined cube of that name.
        for segment in parts[1:-1] if head == own_cube else ():
            if (
                segment not in cubes
                or segment in path
                or _cube_has_member(cubes[path[-1]], path[-1], segment)
            ):
                break
            path.append(segment)
        if len(path) < 2 or len(path) != len(parts) - 1:
            if head != own_cube and head in cubes:
                plain.add(head)
            return match.group(0)
        paths.append(tuple(path))
        return (
            text[match.start() : match.start(1)]
            + f"{path[-1]}.{parts[-1]}"
            + text[match.end(1) : match.end()]
        )

    return _CUBE_REF_RE.sub(replace, text)


def _has_aggregate(sql):
    """Whether ``sql`` aggregates its own rows; an aggregate inside a subquery does not."""

    try:
        tree = parse_one(sql, read="databricks")
    except SqlglotError:
        return True
    return any(node.find_ancestor(exp.Query) is None for node in tree.find_all(exp.AggFunc))


def _cube_references(text, own_cube, cubes):
    for match in _CUBE_REF_RE.finditer(text):
        if match.start() and text[match.start() - 1] == "\\":
            continue
        body = match.group(1).strip()
        head, dot, rest = body.partition(".")
        if (
            not dot
            and body in cubes
            and body != own_cube
            and re.match(r"\s*\.", text[match.end() :])
        ):
            yield "dataset", body
            continue
        if not dot and body in ("CUBE", "TABLE", own_cube):
            continue
        target = (
            (own_cube, body)
            if not dot
            else (own_cube if head in ("CUBE", "TABLE", own_cube) else head, rest)
        )
        if dot and target[0] not in cubes:
            raise ConversionError(
                f"reference '{{{body}}}' uses unknown cube qualifier '{head}'; "
                "use '{cube}.column' for a raw joined-cube column"
            )
        if not _cube_has_member(cubes[target[0]], target[0], target[1]):
            raise ConversionError(
                f"reference '{{{body}}}' does not match a dimension or measure in "
                f"cube '{target[0]}'; use '{{{target[0]}}}.{target[1]}' for a raw column"
            )
        yield "member", target


def _cube_has_member(cube, cube_name, member_name):
    return any(
        member.get("name") == member_name
        for collection in ("dimensions", "measures")
        for member in _as_named_list(cube.get(collection), f"cube '{cube_name}' {collection}")
    )


def _member_references(text, own_cube, cubes):
    for kind, target in _cube_references(text, own_cube, cubes):
        if kind == "member":
            yield target


def _dataset_markers(cubes):
    base = "__cube_dmv_dataset_"
    corpus = repr(cubes).casefold()
    while base in corpus:
        base = "_" + base
    return {cube_name: f"{base}{index}__" for index, cube_name in enumerate(cubes)}


def _replace_raw_dataset_aliases(sql, own_cube, markers, *, reject_cross, scope):
    text = str(sql)

    def replace(match):
        if match.start() and text[match.start() - 1] == "\\":
            return match.group(0)
        dataset = match.group(1).strip()
        if (
            dataset not in markers
            or dataset == own_cube
            or re.match(r"\s*\.", text[match.end() :]) is None
        ):
            return match.group(0)
        if reject_cross:
            raise ConversionError(
                f"dimension '{scope}' reads raw joined-cube columns from '{dataset}'; "
                "a dataset-scoped Ossie field cannot preserve that join"
            )
        return markers[dataset]

    return _CUBE_REF_RE.sub(replace, text)


def _qualify_physical_columns(
    expression, qualifier, known_qualifiers, scope, bare_only=False, subqueries=False
):
    """Attach provenance without reserializing or otherwise changing user SQL.

    With ``bare_only``, a multipart column outside ``known_qualifiers`` is refused
    rather than read as a struct path of ``qualifier``. With ``subqueries``, a
    subquery is allowed and its columns are left as written: they bind to its own
    FROM, and its Cube references already carry a dataset marker.
    """

    try:
        tokens = Dialect.get_or_raise("databricks").tokenize(expression)
        tree = parse_one(expression, read="databricks")
        if (
            tree is None
            or (not subqueries and tree.find(exp.Query) is not None)
            or any(token.token_type is TokenType.ARROW for token in tokens)
        ):
            raise ConversionError(
                f"{scope} contains nested SQL bindings that cannot be safely "
                "attributed to Cube datasets"
            )
        edits = []
        for column in tree.find_all(exp.Column):
            if subqueries and column.find_ancestor(exp.Query) is not None:
                continue
            _, parts = _complete_column_path(column)
            if not parts:
                raise ConversionError(
                    f"{scope} contains a column path whose Cube dataset provenance "
                    "cannot be determined"
                )
            if parts[0].name.casefold() in known_qualifiers:
                continue
            if bare_only and len(parts) > 1:
                path = ".".join(part.name for part in parts)
                raise ConversionError(
                    f"{scope} reads '{path}', which is not a column of a cube it can "
                    "reach; reference columns as '{cube}.column'"
                )
            start = parts[0].meta.get("start")
            if not isinstance(start, int):
                raise ConversionError(
                    f"{scope} contains a column whose source position cannot be determined"
                )
            edits.append((start, start, f"{qualifier}."))
        return _apply_text_edits(expression, edits)
    except SqlglotError as error:
        raise ConversionError(
            f"{scope} is not safely parseable for Cube dataset provenance: {error}"
        ) from error


def _rename_column_qualifiers(expression, renames):
    """Replace the first part of each multipart column found in ``renames``.

    SQL that cannot be parsed is returned unchanged for the caller's own checks.
    """

    try:
        tree = parse_one(expression, read="databricks")
    except SqlglotError:
        return expression
    edits = []
    for column in tree.find_all(exp.Column) if tree is not None else ():
        _, parts = _complete_column_path(column)
        if len(parts) < 2 or parts[0].name.casefold() not in renames:
            continue
        start, end = parts[0].meta.get("start"), parts[1].meta.get("start")
        if isinstance(start, int) and isinstance(end, int):
            edits.append((start, end, f"{renames[parts[0].name.casefold()]}."))
    return _apply_text_edits(expression, edits)


def _complete_column_path(column):
    if not all(isinstance(part, exp.Identifier) for part in column.parts):
        return None, ()
    target = column
    parts = list(column.parts)
    while isinstance(target.parent, exp.Dot) and target.parent.this is target:
        outer = target.parent.expression
        if not isinstance(outer, exp.Identifier):
            return None, ()
        parts.append(outer)
        target = target.parent
    return target, tuple(parts)


def _apply_text_edits(text, edits):
    """Apply non-overlapping source edits from right to left."""

    result = text
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result


def _render_dataset_markers(expression, replacements):
    def render(text):
        for marker, replacement in replacements.items():
            text = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(marker)}(?![A-Za-z0-9_])",
                replacement,
                text,
            )
        return text

    return sub_outside_quotes(expression, render)


class _DimensionResolver:
    def __init__(self, cubes, dataset_markers):
        self._cubes = cubes
        self._dataset_markers = dataset_markers
        self._known_markers = frozenset(marker.casefold() for marker in dataset_markers.values())
        self._dimensions = {}
        self._measures = set()
        self._cache = {}
        for cube_name, cube in cubes.items():
            for dim in _as_named_list(cube.get("dimensions"), f"cube '{cube_name}' dimensions"):
                self._dimensions[(cube_name, dim.get("name"))] = dim
            for measure in _as_named_list(cube.get("measures"), f"cube '{cube_name}' measures"):
                self._measures.add((cube_name, measure.get("name")))

    def has_dimension(self, cube_name, member_name):
        return (cube_name, member_name) in self._dimensions

    def has_bare_column(self, cube_name, member_name, stack=()):
        """Whether a dimension's SQL, or a dimension it reads, has a bare column."""

        marker = (cube_name, member_name)
        if marker in stack:
            return False
        for text in _dimension_expression_texts(self._dimensions[marker]):
            if _has_bare_column(text):
                return True
            for kind, target in _cube_references(text, cube_name, self._cubes):
                if (
                    kind == "member"
                    and target in self._dimensions
                    and self.has_bare_column(*target, stack + (marker,))
                ):
                    return True
        return False

    def reads_other_cube(self, cube_name, member_name, stack=()):
        """Whether a dimension reads another cube, directly or through its own dimensions."""

        marker = (cube_name, member_name)
        if marker in stack:
            return False
        for text in _dimension_expression_texts(self._dimensions[marker]):
            for kind, target in _cube_references(text, cube_name, self._cubes):
                if kind == "dataset" or target[0] != cube_name:
                    return True
                if target in self._dimensions and self.reads_other_cube(*target, stack + (marker,)):
                    return True
        return False

    def join_condition(self, own_cube, target, sql):
        """A join's Cube SQL with every column qualified by its cube's dataset marker."""

        scope = f"join '{own_cube}' -> '{target}'"
        for match in _CUBE_REF_RE.finditer(sql):
            if match.start() and sql[match.start() - 1] == "\\":
                continue
            head, dot, _ = match.group(1).strip().partition(".")
            if head in ("CUBE", "TABLE", own_cube, target) or (
                not dot and _cube_has_member(self._cubes[own_cube], own_cube, head)
            ):
                continue
            raise ConversionError(
                f"{scope} reads cube '{head}'; a Databricks Metric View join condition "
                "can read only the two cubes it joins"
            )
        for kind, reference in _cube_references(sql, own_cube, self._cubes):
            if kind == "member" and reference in self._measures:
                raise ConversionError(
                    f"{scope} references measure '{reference[0]}.{reference[1]}'; a join "
                    "condition can read only columns and dimensions"
                )

        # Columns are qualified with sqlglot rather than qualify_bare_columns, which
        # reads a date unit such as DAY as a column. A bare column belongs to the
        # declaring cube, as the Cube converter reads it in an equality join; a
        # `table.column` with no cube reference could name any table, so it is refused.
        def qualified(expression, cube_name):
            return _qualify_physical_columns(
                expression.strip(),
                self._dataset_markers[cube_name],
                self._known_markers,
                scope,
                bare_only=True,
            )

        def resolve(body):
            head, dot, rest = body.partition(".")
            cube_name = own_cube if head in ("CUBE", "TABLE", own_cube) or not dot else head
            member = rest if dot else body
            if not self.has_dimension(cube_name, member):
                return None
            inner = qualified(self.expression(cube_name, member), cube_name)
            return f"({inner})" if has_top_level_operator(inner) else inner

        prepared = _replace_raw_dataset_aliases(
            sql, own_cube, self._dataset_markers, reject_cross=False, scope=scope
        )
        out, _ = cube_sql_to_ossie(
            prepared,
            own_cube,
            resolve_ref=resolve,
            self_prefix=self._dataset_markers[own_cube],
            cube_names=self._cubes,
        )
        # Cube aliases each cube's table by the cube's name, so in its join SQL
        # `orders.id` is a column of the joined cube `orders`.
        named = {
            cube_name.casefold(): self._dataset_markers[cube_name]
            for cube_name in (own_cube, target)
        }
        condition = qualified(_rename_column_qualifiers(out, named), own_cube)
        # A Metric View join keeps one match per row, so a condition such as
        # '1 = 1' would keep one row where Cube keeps every row it matches.
        if self._dataset_markers[target] not in condition:
            raise ConversionError(
                f"{scope} reads no column of '{target}'; a Databricks Metric View join keeps "
                "one match per row, where Cube keeps every row this condition matches"
            )
        return condition

    def expression(self, cube_name, member_name, qualified=False, stack=()):
        key = (cube_name, member_name, qualified)
        if key in self._cache:
            return self._cache[key]
        marker = (cube_name, member_name)
        if marker in stack:
            chain = " -> ".join(f"{c}.{m}" for c, m in stack + (marker,))
            raise ConversionError(f"dimension reference cycle: {chain}")
        dim = self._dimensions[marker]
        dtype = str(dim.get("type") or "string").lower()
        if dtype in ("geo", "switch") or dim.get("sub_query"):
            raise ConversionError(
                f"dimension '{cube_name}.{member_name}' has no safe static publication expression"
            )
        if dim.get("case") is not None:
            expr = self._case_expression(
                cube_name, member_name, dim["case"], qualified, stack + (marker,)
            )
        elif dim.get("sql") is None:
            expr = f"{self._dataset_markers[cube_name]}.{member_name}" if qualified else member_name
        else:
            expr = self._translate(dim["sql"], cube_name, qualified, stack + (marker,))
        self._cache[key] = expr
        return expr

    def _translate(self, sql, cube_name, qualified, stack):
        scope = f"dimension '{stack[0][0]}.{stack[0][1]}'"
        prepared = _replace_raw_dataset_aliases(
            sql,
            cube_name,
            self._dataset_markers,
            reject_cross=not qualified,
            scope=f"{stack[0][0]}.{stack[0][1]}",
        )
        out, _ = cube_sql_to_ossie(
            prepared,
            cube_name,
            resolve_ref=lambda body: self._resolve(body, cube_name, qualified, stack),
            self_prefix=self._dataset_markers[cube_name] if qualified else None,
            cube_names=self._cubes,
        )
        if qualified:
            out = _qualify_physical_columns(
                out,
                self._dataset_markers[cube_name],
                self._known_markers,
                scope,
            )
        return out

    def _resolve(self, body, own_cube, qualified, stack):
        head, dot, rest = body.partition(".")
        if not dot:
            if body in ("CUBE", "TABLE", own_cube):
                return None
            target = (own_cube, body)
        else:
            target = (
                own_cube if head in ("CUBE", "TABLE", own_cube) else head,
                rest,
            )
        if target in self._measures:
            raise ConversionError(
                f"dimension '{stack[0][0]}.{stack[0][1]}' references measure "
                f"'{target[0]}.{target[1]}'; correlated/sub-query dimensions "
                "cannot be published safely"
            )
        if target not in self._dimensions:
            return None
        if not qualified and target[0] != own_cube:
            raise ConversionError(
                f"dimension '{stack[0][0]}.{stack[0][1]}' reads joined-cube "
                f"dimension '{target[0]}.{target[1]}'; a dataset-scoped Ossie "
                "field cannot preserve that join"
            )
        inner = self.expression(target[0], target[1], qualified, stack)
        return f"({inner})" if has_top_level_operator(inner) else inner

    def _case_expression(self, cube_name, member_name, case, qualified, stack):
        if not isinstance(case, dict):
            raise ConversionError(f"dimension '{cube_name}.{member_name}' has a non-mapping case")
        parts = []
        for branch in case.get("when") or []:
            if not isinstance(branch, dict) or branch.get("sql") is None:
                raise ConversionError(
                    f"dimension '{cube_name}.{member_name}' has an invalid case.when entry"
                )
            condition = self._translate(branch["sql"], cube_name, qualified, stack)
            parts.append(
                f"WHEN {condition} THEN {self._case_label(branch, cube_name, qualified, stack)}"
            )
        if not parts:
            raise ConversionError(
                f"dimension '{cube_name}.{member_name}' has no case.when branches"
            )
        otherwise = case.get("else")
        if isinstance(otherwise, dict) and "label" in otherwise:
            parts.append(f"ELSE {self._case_label(otherwise, cube_name, qualified, stack)}")
        return "CASE " + " ".join(parts) + " END"

    def _case_label(self, holder, cube_name, qualified, stack):
        label = holder.get("label")
        if isinstance(label, dict):
            if label.get("sql") is None:
                raise ConversionError("case label object has no sql")
            return self._translate(label["sql"], cube_name, qualified, stack)
        return "'" + str(label if label is not None else "").replace("'", "''") + "'"


class _PublicationMeasureResolver(_MeasureResolver):
    """Inline measures while preserving the dataset origin of every column."""

    def __init__(self, cubes, pk_by_cube, issues, dimensions, dataset_markers):
        super().__init__(cubes, pk_by_cube, issues)
        self._dimensions = dimensions
        self._dataset_markers = dataset_markers
        self._known_markers = frozenset(marker.casefold() for marker in dataset_markers.values())
        self._publication_primary_keys = {
            cube_name: _publication_primary_keys(cube, cube_name)
            for cube_name, cube in cubes.items()
        }

    def _expression(self, cname, mname, stack, inline_refs):
        key = (cname, mname)
        measure = self._raw[key]
        measure_type = str(measure.get("type") or "").lower().replace("-", "_")
        if measure_type == "count" and measure.get("sql") is None:
            cache = self._caches[inline_refs]
            if key in cache:
                return cache[key]
            if key in stack:
                chain = " -> ".join(f"{c}.{m}" for c, m in stack + (key,))
                raise ConversionError(f"measure reference cycle: {chain}")
            filters = [
                self._translate(item["sql"], cname, stack + (key,), inline_refs)
                for item in measure.get("filters") or []
                if isinstance(item, dict) and item.get("sql")
            ]
            expression = self._primary_key_count_expression(cname, filters)
            return self._remember(cache, key, expression)
        return super()._expression(cname, mname, stack, inline_refs)

    def _primary_key_count_expression(self, cube_name, filters):
        primary_keys = self._publication_primary_keys.get(cube_name) or []
        if not primary_keys:
            raise ConversionError(
                f"Cube '{cube_name}': a bare `type: count` measure needs the cube's "
                "primary key to convert safely, but no dimension declares "
                "`primary_key: true`"
            )
        resolved = [
            self._dimensions.expression(cube_name, primary_key, qualified=True)
            if is_dimension
            else f"{self._dataset_markers[cube_name]}.{primary_key}"
            for primary_key, is_dimension in primary_keys
        ]
        operands = [filtered_operand(expression, filters) for expression in resolved]
        return f"COUNT(DISTINCT {', '.join(operands)})"

    def _translate(self, sql, cname, stack, inline_refs):
        prepared = _replace_raw_dataset_aliases(
            sql,
            cname,
            self._dataset_markers,
            reject_cross=False,
            scope=f"{cname}.{stack[0][1] if stack else '<unknown>'}",
        )

        def target_of(body):
            head, dot, rest = body.partition(".")
            if not dot:
                return None if body in ("CUBE", "TABLE", cname) else (cname, body)
            return (cname if head in ("CUBE", "TABLE", cname) else head, rest)

        scope = f"measure '{cname}.{stack[0][1] if stack else '<unknown>'}'"
        for target in map(target_of, _references_inside_subqueries(sql)):
            if target is not None and self.is_measure(*target):
                raise ConversionError(
                    f"{scope} reads measure '{target[0]}.{target[1]}' inside a subquery, "
                    "which has no Metric View form"
                )
            # Cube pastes a dimension's SQL into the subquery as written, so its bare
            # columns bind to the subquery's own table, which the Metric View cannot match.
            if (
                target is not None
                and self._dimensions.has_dimension(*target)
                and self._dimensions.has_bare_column(*target)
            ):
                raise ConversionError(
                    f"{scope} reads dimension '{target[0]}.{target[1]}' inside a subquery, "
                    "and its SQL has a bare column that Cube binds to the subquery's own "
                    "table; write the column as {CUBE}.column in that dimension"
                )

        def resolve(body):
            target = target_of(body)
            if target is not None and self._dimensions.has_dimension(*target):
                inner = self._dimensions.expression(*target, qualified=True)
                return f"({inner})" if has_top_level_operator(inner) else inner
            if (
                target is not None
                and stack
                and self.is_measure(*target)
                and self.aggregate_of(*stack[-1]) not in CALCULATED_MEASURE_TYPES
                and _has_aggregate(
                    super(_PublicationMeasureResolver, self)._resolve(body, cname, stack, True)
                )
            ):
                # SQL cannot nest aggregates; Cube plans this as a subquery instead.
                raise ConversionError(
                    f"measure '{stack[-1][0]}.{stack[-1][1]}' aggregates measure "
                    f"'{target[0]}.{target[1]}'; an aggregate of an aggregate has no "
                    "Metric View form, so reference a dimension instead"
                )
            return super(_PublicationMeasureResolver, self)._resolve(
                body,
                cname,
                stack,
                inline_refs,
            )

        out, _ = cube_sql_to_ossie(
            prepared,
            cname,
            resolve_ref=resolve,
            self_prefix=self._dataset_markers[cname],
            cube_names=self._cube_names,
        )
        return _qualify_physical_columns(
            out,
            self._dataset_markers[cname],
            self._known_markers,
            f"measure '{cname}.{stack[0][1] if stack else '<unknown>'}'",
            subqueries=True,
        )


def _apply_projection(model, cubes, selected, carriers, metric_view):
    datasets = {dataset["name"]: dataset for dataset in model.get("datasets") or []}
    metrics = {metric["name"]: metric for metric in model.get("metrics") or []}

    pk_by_cube = {name: _primary_key_of(cube, name) for name, cube in cubes.items()}
    resolver_issues = IssueLog()
    dataset_markers = _dataset_markers(cubes)
    dataset_names = {marker: cube_name for cube_name, marker in dataset_markers.items()}
    dim_resolver = _DimensionResolver(cubes, dataset_markers)
    join_conditions = {
        carrier: (
            own,
            target,
            dim_resolver.join_condition(own, target, _find_join(cubes[own], own, target)["sql"]),
        )
        for (own, target), carrier in carriers.items()
    }
    measure_resolver = _PublicationMeasureResolver(
        cubes,
        pk_by_cube,
        resolver_issues,
        dim_resolver,
        dataset_markers,
    )
    converts = {
        key: measure_resolver.inlined(*key) is not None for key in measure_resolver.measures()
    }
    counts = {}
    for key, measure in measure_resolver.measures().items():
        if converts[key] and not _is_generated_part(measure):
            norm = normalize_identifier(key[1])
            counts[norm] = counts.get(norm, 0) + 1

    projected_fields = {name: [] for name in datasets}
    projected_metrics = []
    metric_templates = {}
    dimension_templates = {}
    for member in selected:
        if member.kind == "dimension":
            fields = {field["name"]: field for field in datasets[member.cube].get("fields") or []}
            if member.source_name not in fields:
                raise ConversionError(
                    f"selected dimension '{member.cube}.{member.source_name}' "
                    "has no Ossie field representation"
                )
            item = copy.deepcopy(fields[member.source_name])
            item["name"] = member.output_name
            if metric_view and dim_resolver.reads_other_cube(member.cube, member.source_name):
                # A Metric View dimension can read any joined table, so this one keeps
                # each column's dataset and bridge.py renders it against join aliases.
                template = dim_resolver.expression(member.cube, member.source_name, qualified=True)
                dimension_templates[member.output_name] = template
                expression = _render_dataset_markers(template, dataset_names)
                # The Cube converter kept the SQL it could not convert; this replaces it.
                stash = read_stash(item)
                stash.pop("sql", None)
                _drop_cube_extension(item)
                write_stash(item, stash)
            else:
                expression = dim_resolver.expression(member.cube, member.source_name)
            item["expression"]["dialects"][0]["expression"] = expression
            _apply_override(item, member)
            projected_fields[member.cube].append(item)
        else:
            key = (member.cube, member.source_name)
            if not converts.get(key):
                raise ConversionError(
                    f"selected measure '{member.cube}.{member.source_name}' has "
                    "no static Ossie expression"
                )
            source_metric_name = (
                member.source_name
                if counts[normalize_identifier(member.source_name)] == 1
                else f"{member.cube}__{member.source_name}"
            )
            if source_metric_name not in metrics:
                raise ConversionError(
                    f"selected measure '{member.cube}.{member.source_name}' "
                    "was not converted to an Ossie metric"
                )
            item = copy.deepcopy(metrics[source_metric_name])
            item["name"] = member.output_name
            template = measure_resolver.inlined(*key)
            metric_templates[member.output_name] = template
            item["expression"]["dialects"][0]["expression"] = _render_dataset_markers(
                template, dataset_names
            )
            _apply_override(item, member)
            projected_metrics.append(item)

    for dataset in model.get("datasets") or []:
        fields = projected_fields.get(dataset["name"]) or []
        if fields:
            dataset["fields"] = fields
        else:
            dataset.pop("fields", None)
    if projected_metrics:
        model["metrics"] = projected_metrics
    else:
        model.pop("metrics", None)

    # The ordinary import's CUBE extensions exist to rebuild the original Cube
    # files. This model is a one-way publication artifact, so model/dataset/join
    # regeneration data is irrelevant and would surface as a misleading
    # "foreign-vendor extension dropped" warning in every downstream spoke.
    _drop_cube_extension(model)
    for dataset in model.get("datasets") or []:
        _drop_cube_extension(dataset)
    for relationship in model.get("relationships") or []:
        _drop_cube_extension(relationship)

    dataset_data_sources = {}
    for cube_name, cube in cubes.items():
        data_source = cube.get("data_source", "default")
        if not isinstance(data_source, str) or not data_source.strip():
            raise ConversionError(
                f"cube '{cube_name}': data_source must be a non-empty string when provided"
            )
        dataset_data_sources[cube_name] = data_source

    return _ProjectionProvenance(
        metric_templates,
        dimension_templates,
        dataset_markers,
        dataset_data_sources,
        join_conditions,
    )


def _apply_override(item, member):
    override = member.override
    if override.get("title"):
        item["label"] = override["title"]
    if override.get("description"):
        item["description"] = override["description"]
    if "meta" in override and override.get("meta") is not None:
        item.pop("ai_context", None)
        ai = _ai_context_from_meta(override.get("meta"))
        if ai:
            item["ai_context"] = ai

    # Ossie has no native presentation-format field. Keep the effective Cube
    # format/currency in its vendor extension so a composition layer can either
    # map it for the target or report the limitation without consulting Cube again.
    stash = read_stash(item)
    for key in ("format", "currency"):
        if override.get(key):
            stash[key] = override[key]
    if stash:
        write_stash(item, stash)


def _drop_cube_extension(item):
    extensions = [
        extension
        for extension in item.get("custom_extensions") or []
        if extension.get("vendor_name") != VENDOR
    ]
    if extensions:
        item["custom_extensions"] = extensions
    else:
        item.pop("custom_extensions", None)


def _source_grain_measures(cubes, projected, source, measures):
    """The measures a Metric View aggregates once per source row, as Cube does.

    The Cube converter refuses any non-idempotent aggregate that reads a cube on the
    one side of a many-to-one join, since it judges the model with no source. A Metric
    View's rows are its source's rows, and with only many-to-one and one-to-one joins
    none repeats, so a measure of the source cube is safe whatever joined columns it
    reads, provided every measure it references is also on the source cube.
    """

    to_one = ("many_to_one", "one_to_one")
    for cube_name, cube in projected.items():
        for join in _as_named_list(cube.get("joins"), f"cube '{cube_name}' joins"):
            if _RELATIONSHIP_ALIASES.get(snake(str(join.get("relationship") or ""))) not in to_one:
                return frozenset()
    return frozenset(
        f"{source}.{measure_name}"
        for cube_name, measure_name in measures
        if cube_name == source
        and all(owner == source for owner, _ in _measure_closure(cubes, {(source, measure_name)}))
    )


def _publication_issues(collection, conversion, selected, required, source_grain):
    wanted = {f"{member.cube}.{member.source_name}" for member in selected}
    out = IssueLog()
    for issues, is_conversion in ((collection, False), (conversion, True)):
        for issue in issues:
            element = issue.element_name
            if issue.issue_type == IssueType.FANOUT_UNSAFE_METRIC and element in source_grain:
                continue
            member_issue = any(element == name for name in wanted)
            cube_issue = any(element == f"cube '{name}'" for name in required)
            join_issue = element.startswith("join '")
            # The synthetic conversion contains only the selected measures and
            # their transitive dependencies. A dependency-level fan-out issue is
            # therefore part of a selected measure's safety contract even though
            # the dependency itself is not exposed as a public metric.
            dependency_fanout = is_conversion and issue.issue_type == IssueType.FANOUT_UNSAFE_METRIC
            if member_issue or cube_issue or join_issue or dependency_fanout:
                out.issues.append(issue)
    return out
