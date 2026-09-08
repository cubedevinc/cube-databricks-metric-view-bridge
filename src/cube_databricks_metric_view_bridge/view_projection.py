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

from ossie_cube._common import (
    _CUBE_REF_RE,
    DOTTED_REF_RE,
    OSSIE_VERSION,
    VENDOR,
    ConversionError,
    cube_sql_to_ossie,
    dump_yaml,
    load_yaml,
    normalize_identifier,
    read_stash,
    snake_keys,
    split_dotted_ref,
    sub_outside_quotes,
    write_stash,
)
from ossie_cube.converter_issues import IssueLog, IssueType
from ossie_cube.cube_to_osi import (
    _ai_context_from_meta,
    _as_named_list,
    _collect,
    _is_generated_part,
    _MeasureResolver,
    _primary_key_of,
    convert_cube_to_ossie,
)
from ossie_cube.expressions import has_top_level_operator, qualify_bare_columns


@dataclasses.dataclass(frozen=True)
class _SelectedMember:
    cube: str
    source_name: str
    output_name: str
    kind: str
    override: dict


def convert_cube_view_to_ossie(files, view, source=None, strict_fanout=True):
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
    """
    if not isinstance(files, dict) or not files:
        raise ConversionError("expected a non-empty mapping of {filename: YAML}")
    if not isinstance(view, str) or not view.strip():
        raise ConversionError("view projection requires a non-empty view name")

    collection_issues = IssueLog()
    cubes, _, views, _, _ = _collect(files, collection_issues)
    if view not in views:
        raise ConversionError(
            f"requested view '{view}' not found; views present: {sorted(views) or 'none'}"
        )
    if not cubes:
        raise ConversionError("view projection requires the cubes referenced by the view")

    selected_view = views[view]
    selected, paths, required_cubes, joins = _resolve_view(selected_view, view, cubes)
    resolved_source = _resolve_source(view, source, paths, required_cubes)

    needed_measures = _measure_closure(
        cubes,
        {
            (member.cube, member.source_name)
            for member in selected
            if member.kind == "measure"
        },
    )
    required_cubes, joins = _expand_measure_dependency_graph(
        view,
        cubes,
        paths[0][0],
        required_cubes,
        joins,
        needed_measures,
    )
    projected_cubes = _project_cubes(cubes, required_cubes, joins, needed_measures)
    synthetic = dump_yaml(
        {
            "cubes": list(projected_cubes.values()),
            "views": [selected_view],
        }
    )
    ossie_yaml, conversion_issues = convert_cube_to_ossie(
        {"model.yml": synthetic}, view=view, strict_fanout=False
    )
    model = load_yaml(ossie_yaml, "projected Ossie model")["semantic_model"][0]

    dim_resolver = _DimensionResolver(projected_cubes)
    _apply_projection(model, projected_cubes, selected, dim_resolver)
    issues = _publication_issues(collection_issues, conversion_issues, selected, required_cubes)
    if strict_fanout:
        unsafe = issues.of_type(IssueType.FANOUT_UNSAFE_METRIC)
        if unsafe:
            raise ConversionError(f"{unsafe[0]} (refused under strict mode)")

    return (
        dump_yaml({"version": OSSIE_VERSION, "semantic_model": [model]}),
        resolved_source,
        issues,
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
    alias = entry.get("alias") or cube_name
    if not isinstance(alias, str):
        raise ConversionError(f"view '{view_name}': alias for cube '{cube_name}' must be a string")

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
        requested = [(name, name, {}) for members in collections.values() for name in members]
    elif isinstance(includes, list):
        requested = []
        for include in includes:
            if isinstance(include, str):
                source_name, output_name, override = include, include, {}
            elif isinstance(include, dict):
                source_name = include.get("name")
                output_name = include.get("alias") or source_name
                if not isinstance(source_name, str) or not source_name:
                    raise ConversionError(
                        f"view '{view_name}': include for cube '{cube_name}' must name a member"
                    )
                if not isinstance(output_name, str):
                    raise ConversionError(
                        f"view '{view_name}': alias for member '{source_name}' must be a string"
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

    excludes = entry.get("excludes") or []
    if not isinstance(excludes, list) or not all(isinstance(v, str) for v in excludes):
        raise ConversionError(
            f"view '{view_name}': excludes for cube '{cube_name}' must be a list of member names"
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


def _resolve_source(view_name, requested, paths, required):
    default = paths[0][0]
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
        projected[cube_name] = cube
    return projected


def _expand_measure_dependency_graph(
    view_name, cubes, root, required, joins, needed_measures
):
    """Add uniquely reachable cubes needed only by hidden measure dependencies."""

    expanded_required = list(required)
    expanded_joins = {cube_name: dict(targets) for cube_name, targets in joins.items()}
    parent_of = {
        target: cube_name
        for cube_name, targets in expanded_joins.items()
        for target in targets
    }
    dependency_cubes = sorted({cube_name for cube_name, _ in needed_measures})
    for target in dependency_cubes:
        if target in expanded_required:
            continue
        paths = _declared_join_paths(cubes, root, target, limit=2)
        if not paths:
            raise ConversionError(
                f"view '{view_name}': selected measures depend on cube '{target}', "
                f"but it has no declared join path from view root '{root}'"
            )
        if len(paths) > 1:
            raise ConversionError(
                f"view '{view_name}': selected measures depend on cube '{target}' through "
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
    return expanded_required, expanded_joins


def _declared_join_paths(cubes, start, target, limit):
    """Return up to ``limit`` simple paths following declared Cube joins."""

    found = []

    def visit(current, path):
        if len(found) >= limit:
            return
        joins = _as_named_list(cubes[current].get("joins"), f"cube '{current}' joins")
        for join in joins:
            neighbor = join.get("name")
            if neighbor not in cubes or neighbor in path:
                continue
            next_path = path + [neighbor]
            if neighbor == target:
                found.append(next_path)
            else:
                visit(neighbor, next_path)
            if len(found) >= limit:
                return

    visit(start, [start])
    return found


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
        texts = [measure.get("sql")]
        texts.extend(f.get("sql") for f in measure.get("filters") or [] if isinstance(f, dict))
        for text in texts:
            if not isinstance(text, str):
                continue
            for match in _CUBE_REF_RE.finditer(text):
                body = match.group(1).strip()
                head, dot, rest = body.partition(".")
                target = (
                    (cube_name, body)
                    if not dot
                    else (cube_name if head in ("CUBE", "TABLE") else head, rest)
                )
                if target in measures and target not in needed:
                    needed.add(target)
                    queue.append(target)
    return needed


class _DimensionResolver:
    def __init__(self, cubes):
        self._cubes = cubes
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
            expr = f"{cube_name}.{member_name}" if qualified else member_name
        else:
            expr = self._translate(dim["sql"], cube_name, qualified, stack + (marker,))
        self._cache[key] = expr
        return expr

    def _translate(self, sql, cube_name, qualified, stack):
        prepared = qualify_bare_columns(sql) if qualified else sql
        out, _ = cube_sql_to_ossie(
            prepared,
            cube_name,
            resolve_ref=lambda body: self._resolve(body, cube_name, qualified, stack),
            self_prefix=cube_name if qualified else None,
            cube_names=self._cubes,
        )
        return out

    def _resolve(self, body, own_cube, qualified, stack):
        head, dot, rest = body.partition(".")
        if not dot:
            if body in ("CUBE", "TABLE") or body in self._cubes:
                return None
            target = (own_cube, body)
        else:
            target = (own_cube if head in ("CUBE", "TABLE") else head, rest)
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


def _apply_projection(model, cubes, selected, dim_resolver):
    datasets = {dataset["name"]: dataset for dataset in model.get("datasets") or []}
    metrics = {metric["name"]: metric for metric in model.get("metrics") or []}

    pk_by_cube = {name: _primary_key_of(cube, name) for name, cube in cubes.items()}
    resolver_issues = IssueLog()
    measure_resolver = _MeasureResolver(cubes, pk_by_cube, resolver_issues)
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
            item["expression"]["dialects"][0]["expression"] = dim_resolver.expression(
                member.cube, member.source_name
            )
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
            expression = measure_resolver.inlined(*key)
            item["expression"]["dialects"][0]["expression"] = _inline_metric_dimensions(
                expression, dim_resolver
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


def _inline_metric_dimensions(expression, resolver):
    def replace(text):
        def one(match):
            cube_name, member_name = split_dotted_ref(match.group(0))
            key = (cube_name.strip('"'), member_name.strip('"'))
            if not resolver.has_dimension(*key):
                return match.group(0)
            inner = resolver.expression(*key, qualified=True)
            return f"({inner})" if has_top_level_operator(inner) else inner

        return DOTTED_REF_RE.sub(one, text)

    return sub_outside_quotes(expression, replace)


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


def _publication_issues(collection, conversion, selected, required):
    wanted = {f"{member.cube}.{member.source_name}" for member in selected}
    out = IssueLog()
    for issues, is_conversion in ((collection, False), (conversion, True)):
        for issue in issues:
            element = issue.element_name
            member_issue = any(element == name for name in wanted)
            cube_issue = any(element == f"cube '{name}'" for name in required)
            join_issue = element.startswith("join '")
            # The synthetic conversion contains only the selected measures and
            # their transitive dependencies. A dependency-level fan-out issue is
            # therefore part of a selected measure's safety contract even though
            # the dependency itself is not exposed as a public metric.
            dependency_fanout = (
                is_conversion and issue.issue_type == IssueType.FANOUT_UNSAFE_METRIC
            )
            if member_issue or cube_issue or join_issue or dependency_fanout:
                out.issues.append(issue)
    return out
