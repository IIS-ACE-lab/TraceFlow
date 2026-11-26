# services/visualization/vscode_monitor/make_blueprint.py
"""
Keep the project blueprint (services/side_channel_core/config/blueprint/) in step with the keys the code
reads, and build one from DLSCA_ASCAD's config (how the first blueprint was built, 2026-10-11).

The blueprint is what a new project starts from (the panel and `make scaffold` copy its files), and what the
panel draws in every project's config forms: a key a project's file lacks is drawn from the blueprint, with
its menu. So the blueprint must hold every key of each file's schema (utils/config_schema.py) that a new
project is offered (every key but config_schema.RETIRED), empty, and nothing else:

    python -m services.visualization.vscode_monitor.make_blueprint check     # the keys missing / not offered, per file
    python -m services.visualization.vscode_monitor.make_blueprint sync      # add every missing key, empty, in place

A key added to a schema (a new setting the code reads) fails test_config_schema until `sync` put it in the
blueprint (its comment is then written by hand). The lists a project fills (data entries, models,
optimizers) are checked in templates.yaml, where their one item is.

`from-project` builds the blueprint files from projects/DLSCA_ASCAD/config with every value emptied
(a single value -> nothing, a list of values -> [], a list of blocks -> its first block emptied, a mapping
keyed by names -> {}, experiment_queue.yaml -> one key ``example: []``); it writes only with --write.

Author: Shraddha Mishra
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from services.visualization.vscode_monitor import yamlspan

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "projects" / "DLSCA_ASCAD" / "config"
BLUEPRINT = REPO / "services" / "side_channel_core" / "config" / "blueprint"
FILES = (
    "params.yaml", "experiment_queue.yaml", "graph_config.yaml", "model_filter.yaml",
    "model_judge.yaml", "test_config.yaml", "dev.yaml", "staging.yaml", "prod.yaml",
)
NAMED_KEY_MAPPINGS = {"labels"}            # keys are model names, not settings
NAMED_KEY_FILES = {"experiment_queue.yaml"}  # top-level keys are figure names


def _empty_value(text: str, node: Node, edits: list) -> None:
    """Edits that empty one value node (recursing into blocks)."""
    if isinstance(node, ScalarNode):
        start, end = yamlspan.span(node)
        if end > start and node.tag != "tag:yaml.org,2002:null":
            edits.append((start, end, ""))
        return
    if isinstance(node, SequenceNode):
        if not node.value:
            return
        if yamlspan.is_scalar_list(node):
            edits.append((node.start_mark.index, yamlspan.content_end(node), "[]"))
            return
        first, rest = node.value[0], node.value[1:]
        _empty_value(text, first, edits)
        if rest:
            if node.flow_style:
                edits.append((yamlspan.content_end(first), yamlspan.content_end(rest[-1]), ""))
            else:
                edits.append((
                    yamlspan.line_end(text, yamlspan.content_end(first) - 1),
                    yamlspan.line_end(text, yamlspan.content_end(rest[-1]) - 1),
                    "",
                ))
        return
    if isinstance(node, MappingNode):
        for key_node, value_node in node.value:
            if str(key_node.value) in NAMED_KEY_MAPPINGS and isinstance(value_node, MappingNode) and value_node.value:
                edits.append((value_node.start_mark.index, yamlspan.content_end(value_node), "{}"))
            else:
                _empty_value(text, value_node, edits)


def _named_keys_to_example(text: str, root: MappingNode, edits: list) -> None:
    """Keep the first top-level key, renamed ``example``, with an emptied value."""
    (first_key, first_value), rest = root.value[0], root.value[1:]
    # The comment block right above the first key describes that key only.
    head = yamlspan.line_start(text, first_key.start_mark.index)
    top = head
    while top > 0:
        previous = yamlspan.line_start(text, top - 1)
        if not text[previous:top].lstrip().startswith("#"):
            break
        top = previous
    edits.append((top, head, ""))
    edits.append((first_key.start_mark.index, first_key.end_mark.index, "example"))
    _empty_value(text, first_value, edits)
    if rest:
        edits.append((
            yamlspan.line_end(text, yamlspan.content_end(first_value) - 1),
            yamlspan.line_end(text, yamlspan.content_end(rest[-1][1]) - 1),
            "",
        ))


def empty_text(text: str, *, named_keys: bool = False) -> str:
    """The text of one config file with every value emptied (see module docstring)."""
    root = yamlspan.compose(text)
    if root is None:
        return text
    edits: list[tuple[int, int, str]] = []
    if named_keys and isinstance(root, MappingNode) and root.value:
        _named_keys_to_example(text, root, edits)
    else:
        _empty_value(text, root, edits)
    out = yamlspan.apply(text, edits)
    yaml.safe_load(out)                    # must still be YAML
    return out


# ── check / sync: the blueprint against the schemas ──────────────────────────
# Each blueprint file and the schema (or the part of it) its keys follow.
FILE_SCHEMAS = {
    "params.yaml": lambda cs: cs.PARAMS,
    "model_judge.yaml": lambda cs: cs.JUDGE,
    "auto_search.yaml": lambda cs: {"search": cs.AUTO_SEARCH["search"], "tracking": cs.AUTO_SEARCH["tracking"]},
    "master_project.yaml": lambda cs: cs.MASTER_PROJECT,
}
# A list a project fills itself (empty in the blueprint): its one item is in templates.yaml under this key.
TEMPLATE_LISTS = {"data": "data", "model": "model", "trainer.optimizer": "optimizer"}


def _paths_of(data: Any, path: str = "", lists: dict | None = None) -> set[str]:
    """Every key path a YAML document holds (a list's items as `*`); an empty list under a template key
    continues into templates.yaml's item."""
    out: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            where = f"{path}.{key}" if path else str(key)
            out.add(where)
            out |= _paths_of(value, where, lists)
    elif isinstance(data, list):
        items = data
        if not items and lists and path in lists and lists[path] is not None:
            items = [lists[path]]
        for item in items:
            if isinstance(item, (dict, list)):
                out |= _paths_of(item, f"{path}.*" if path else "*", lists)
    return out


def check() -> dict[str, dict[str, list[str]]]:
    """Per blueprint file: the offered keys it lacks (`missing`) and the retired keys it holds (`not offered`)."""
    from services.side_channel_core.utils import config_schema as cs

    templates = yaml.safe_load((BLUEPRINT / "templates.yaml").read_text(encoding="utf-8")) or {}
    lists = {path: templates.get(name) for path, name in TEMPLATE_LISTS.items()}
    out = {}
    for name, schema_of in FILE_SCHEMAS.items():
        data = yaml.safe_load((BLUEPRINT / name).read_text(encoding="utf-8")) or {}
        offered = cs.offered_keys(schema_of(cs)) - cs.RETIRED.get(name, set())
        have = _paths_of(data, lists=lists)
        # a key under a list whose item is nowhere (no template): not checkable
        missing = sorted(k for k in offered - have if not any(k.startswith(f"{p}.*") for p in lists if lists[p] is None))
        out[name] = {"missing": missing, "not offered": sorted(have & cs.RETIRED.get(name, set()))}
    return out


def sync() -> dict[str, list[str]]:
    """Add every missing key to its blueprint file, empty (a block: {} first, then its keys), parents first;
    a key under a template list goes into templates.yaml's item. Returns what was added, per file."""
    from services.visualization.vscode_monitor import yamledit

    added: dict[str, list[str]] = {}
    for name, report in check().items():
        missing = sorted(report["missing"], key=lambda k: (k.count("."), k))
        for dotted in missing:
            file, path = name, dotted.split(".")
            if "*" in path:                                  # under a list: templates.yaml's item
                star = path.index("*")
                file, path = "templates.yaml", [TEMPLATE_LISTS[".".join(path[:star])], *path[star + 1:]]
            target = BLUEPRINT / file
            text = target.read_text(encoding="utf-8")
            parent, key = path[:-1], path[-1]
            data = yaml.safe_load(text) or {}
            node = data
            for seg in parent:
                node = node.get(seg) if isinstance(node, dict) else None
            if isinstance(node, dict) and key in node:
                continue
            # the key with every missing key under it, as one block (its leaves null), then each leaf emptied
            tree: dict = {}
            for m in [k for k in missing if k == dotted or k.startswith(f"{dotted}.")]:
                cursor = tree
                for seg in m[len(dotted):].strip(".").split(".") if m != dotted else []:
                    cursor = cursor.setdefault(seg, {})
            leaves = [p for p in _leaves(tree) if p]
            value = json.dumps(_nulled(tree)) if tree else ""
            if node is None and parent:                     # an empty parent (`key:`): the block is its value
                text = yamledit.edit_text(text, "set", parent, json.dumps({key: _nulled(tree)}))
            else:
                text = yamledit.edit_text(text, "put", parent, value, key)
            for leaf in leaves:
                text = yamledit.edit_text(text, "set", [*path, *leaf], "")
            target.write_text(text, encoding="utf-8")
            added.setdefault(file, []).extend([".".join(path), *(".".join([*path, *leaf]) for leaf in leaves)])
    return added


# ── complete: a project's files against the blueprint ───────────────────────
# The blueprint files whose keys a project's file of the same name must hold (templates.yaml's lists aside).
COMPLETED_FILES = ("params.yaml", "model_judge.yaml", "test_config.yaml", "model_filter.yaml")


def complete_project(config: Path, hash_safe: bool = True) -> dict[str, dict[str, list[str]]]:
    """Add to each config file of a project every key of the blueprint file it lacks, with the blueprint's
    value: a project made from an older template, or the blueprint grown since (make_blueprint sync), gets
    the keys the panel draws for it. Lists a project fills (data, model, optimizers) and their items are
    left alone. hash_safe (the scaffold's way, every project, runs or not): only keys whose value is empty
    or an empty list are added, which the config loader drops or hashes like an absent key, so no run made
    before changes its hash; the others (a path, metrics_cfg's constants, an empty block) are listed as
    `left`, for Load Blueprint in the panel. Returns per file: added, left."""
    from services.side_channel_core.utils import config_schema as cs
    from services.visualization.vscode_monitor import yamledit

    out: dict[str, dict[str, list[str]]] = {}
    for name in COMPLETED_FILES:
        target = config / name
        if not target.is_file():
            continue
        retired = cs.RETIRED.get(name, set())          # never written into a project, whatever the blueprint holds
        want = yaml.safe_load((BLUEPRINT / name).read_text(encoding="utf-8")) or {}
        text = target.read_text(encoding="utf-8")
        have = yaml.safe_load(text) or {}
        if not isinstance(have, dict) or not isinstance(want, dict):
            continue
        added: list[str] = []
        left: list[str] = []

        def walk(w: Any, h: Any, path: list[str]) -> None:
            nonlocal text
            if not isinstance(w, dict) or not isinstance(h, dict):
                return
            for key, value in w.items():
                here = [*path, key]
                if ".".join(here) in retired:
                    continue
                if key in h:
                    if isinstance(value, dict) and isinstance(h[key], dict):
                        walk(value, h[key], here)
                    continue
                if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                    continue                             # a list of blocks: the project's own (Add a ...)
                safe = value is None or value == [] or (isinstance(value, dict) and value and _all_empty(value))
                if hash_safe and not safe:
                    left.append(".".join(here))
                    continue
                if isinstance(value, dict) and value:     # a block, as the blueprint holds it (retired keys out); its null leaves written empty
                    value = _without_retired(value, ".".join(here), retired)
                    text = yamledit.edit_text(text, "put", path, json.dumps(value), key)
                    for leaf in _null_leaves(value, (key,)):
                        text = yamledit.edit_text(text, "set", [*path, *leaf], "")
                else:
                    text = yamledit.edit_text(text, "put", path, "" if value is None else json.dumps(value), key)
                added.append(".".join(here))

        walk(want, have, [])
        if added:
            target.write_text(text, encoding="utf-8")
        if added or left:
            out[name] = {"added": added, "left": left}
    return out


def _all_empty(block: dict) -> bool:
    """Every leaf empty (null) or an empty list: what the loader drops or hashes like an absent key."""
    return all(v is None or v == [] or (isinstance(v, dict) and v and _all_empty(v)) for v in block.values())


def _without_retired(block: dict, prefix: str, retired: set) -> dict:
    """The block without the keys config_schema.RETIRED names under it (at any depth)."""
    return {k: (_without_retired(v, f"{prefix}.{k}", retired) if isinstance(v, dict) else v)
            for k, v in block.items() if f"{prefix}.{k}" not in retired}


def _null_leaves(block: dict, path: tuple = ()) -> list:
    """The paths of a block's null leaves: written `null` by a put, they are set empty right after."""
    return [leaf for k, v in block.items()
            for leaf in (_null_leaves(v, (*path, k)) if isinstance(v, dict) and v else [(*path, k)] if v is None else [])]


def _nulled(tree: dict) -> Any:
    """A tree of keys as data: a block for a key with keys under it, null for a leaf."""
    return {k: (_nulled(v) if v else None) for k, v in tree.items()} if tree else None


def _leaves(tree: dict, path: tuple = ()) -> list:
    return [path] if not tree else [leaf for k, v in tree.items() for leaf in _leaves(v, (*path, k))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check", help="the keys each blueprint file lacks or should not offer")
    sub.add_parser("sync", help="add every missing key, empty, to the blueprint files")
    cp = sub.add_parser("complete", help="add to a project's config files the blueprint keys they lack")
    cp.add_argument("--project", required=True, help="the project folder under projects/")
    cp.add_argument("--all", action="store_true", help="every key, also those a run hashes (a project without runs)")
    fp = sub.add_parser("from-project", help="build the blueprint from DLSCA_ASCAD's config, every value emptied")
    fp.add_argument("--write", action="store_true", help="replace the blueprint files (without it: build in memory)")
    args = parser.parse_args(argv)
    if args.command == "complete":
        report = complete_project(REPO / "projects" / args.project / "config", hash_safe=not getattr(args, "all", False))
        print(json.dumps(report, indent=1))
        return 0
    if args.command in ("check", "sync"):
        report = check() if args.command == "check" else sync()
        print(json.dumps(report, indent=1))
        return 0 if args.command == "sync" or not any(v for r in report.values() for v in r.values()) else 1
    for name in FILES:
        source = SOURCE / name
        text = empty_text(source.read_text(encoding="utf-8"), named_keys=name in NAMED_KEY_FILES)
        header = f"# services/side_channel_core/config/blueprint/{name}\n"
        body = text.split("\n", 1)[1] if text.startswith("#") and "config/" in text.split("\n", 1)[0] else text
        if args.write:
            (BLUEPRINT / name).write_text(header + body, encoding="utf-8")
        print(f"{name}: {len(source.read_text().splitlines())} -> {len((header + body).splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
