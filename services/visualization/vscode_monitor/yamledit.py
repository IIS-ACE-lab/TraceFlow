# services/visualization/vscode_monitor/yamledit.py
"""
The TraceFlow Monitor's config editor: a YAML file as a form tree, and one
edit at a time written back into the text.

    python -m services.visualization.vscode_monitor.yamledit tree FILE
    python -m services.visualization.vscode_monitor.yamledit edit FILE --sha SHA256 \\
        --op set|add|remove|duplicate|delete --path '["trainer", "epochs"]' [--value TEXT]

``tree`` prints ``{"sha256", "tree"}``: entries only (no comments), each with
its path, a list of mapping keys and sequence indices.

``edit`` applies one operation and prints ``{"ok": true, "sha256"}``,
``{"stale": true, "sha256"}`` (the file is no longer the text the caller
rendered: nothing written, the file wins) or ``{"error"}``:

    set PATH VALUE        a single value, or a whole list by a list value ([0, 10000]; a block
                          list under a key becomes that flow list, the key line's comment kept)
    put PATH KEY VALUE    the entry KEY of the mapping at PATH: set, or added when
                          missing (an empty `labels:` becomes a block mapping)
    add PATH VALUE        append VALUE to the list at PATH (an empty `models:` becomes a block
                          list); a mapping VALUE ({target: title, text: ""}) is appended as a
                          block item (a text rule, a facet panel)
    remove PATH           drop one item of a value list
    duplicate PATH [KEY]  copy a list item after itself; a mapping entry
                          under the new key KEY
    delete PATH           drop a list item or a mapping entry

Only the text of that entry changes; every other byte, comment and blank
line stays. The new text is parsed again and written (atomically) only when
its data equals the old data with exactly that change. VALUE follows YAML:
written plain when it reads back as typed, else quoted; a number such as
``1e-5`` (a string to YAML 1.1) given where a number stood is written as a
float (``1.0e-05``); an empty VALUE is null. A value that was quoted stays a
quoted string.

The blueprint (services/side_channel_core/config/blueprint) is never edited.

Author: Shraddha Mishra
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from services.visualization.vscode_monitor import yamlspan

BLUEPRINT = Path("services/side_channel_core/config/blueprint")
NULL_TAG = "tag:yaml.org,2002:null"
_TYPES = {"null": "null", "bool": "bool", "int": "int", "float": "float", "str": "str"}
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


class EditError(ValueError):
    """An edit the editor refuses; the message is shown in the panel."""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── tree ─────────────────────────────────────────────────────────────────────

def _scalar_type(node: ScalarNode) -> str:
    return _TYPES.get(node.tag.rsplit(":", 1)[-1], "str")


def _tree(text: str, node: Node, path: list) -> dict[str, Any]:
    if isinstance(node, MappingNode):
        return {"k": "map", "path": path, "flow": bool(node.flow_style),
                "entries": [{"key": str(k.value), "node": _tree(text, v, path + [str(k.value)])} for k, v in node.value]}
    if isinstance(node, SequenceNode):
        return {"k": "list", "path": path, "flow": bool(node.flow_style), "scalars": yamlspan.is_scalar_list(node),
                "items": [_tree(text, item, path + [i]) for i, item in enumerate(node.value)]}
    shown = node.value if node.style else text[node.start_mark.index:node.end_mark.index]
    # a null keeps its written text: `seed: null` shows null (an unseeded run), `seed:` shows nothing (no setting)
    return {"k": "scalar", "path": path, "text": shown,
            "type": _scalar_type(node), "quoted": node.style in ("'", '"'), "multiline": node.style in ("|", ">")}


def tree(text: str) -> dict[str, Any]:
    root = yamlspan.compose(text)
    return {"sha256": sha256(text), "tree": None if root is None else _tree(text, root, [])}


def comments(text: str) -> dict[str, str]:
    """Each key's comment, by its dotted path (a list item: *): the comment on the key's own line and the comment
    lines right above it, at its indentation (choices: the blueprint's explanation of a key the file lacks)."""

    lines = text.split("\n")
    out: dict[str, str] = {}

    def comment_of(key_node: Any) -> str:
        row = key_node.start_mark.line
        line = lines[row]
        after = line[line.index(":", key_node.start_mark.column) + 1:] if ":" in line[key_node.start_mark.column:] else ""
        hit = after.find("#")
        inline = after[hit + 1:].strip() if hit >= 0 and after[:hit].count('"') % 2 == 0 and after[:hit].count("'") % 2 == 0 else ""
        above, k = [], row - 1
        indent = key_node.start_mark.column
        while k >= 0 and lines[k].strip().startswith("#") and len(lines[k]) - len(lines[k].lstrip()) == indent and len(above) < 12:
            above.insert(0, lines[k].strip()[1:].strip())
            k -= 1
        return " ".join(x for x in [*above, inline] if x and not x.startswith("──") and not set(x) <= set("─ "))

    def walk(node: Any, path: list[str]) -> None:
        for item, child, key_node in yamlspan.children(node):
            sub = [*path, "*" if isinstance(item, int) else str(item)]
            if key_node is not None:
                hint = comment_of(key_node)
                if hint:
                    out.setdefault(".".join(sub), hint)
            if hasattr(child, "value") and isinstance(child.value, list):
                walk(child, sub)

    root = yamlspan.compose(text)
    if root is not None:
        walk(root, [])
    return out


# ── values ───────────────────────────────────────────────────────────────────

def _load_scalar(text: str) -> Any:
    return yamlspan.load(f"k: {text}\n")["k"]


def render(value: str, old: ScalarNode | None, quote: bool = False) -> tuple[str, Any]:
    """(YAML text, the value it stands for) of what the user typed; quote: a string is always quoted."""
    if old is not None and old.style in ("'", '"'):
        if old.style == "'":
            return "'" + value.replace("'", "''") + "'", value
        return json.dumps(value, ensure_ascii=False), value
    if value.strip() == "":
        return "", None
    old_type = _scalar_type(old) if old is not None else None
    stripped = value.strip()
    # What the panel sends as JSON (a drafted text, a list, a block) is read as JSON: a quoted string is that
    # string, and 1e-07 in a list is the number it is (YAML 1.1 reads 1e-07 as text); a list or block is
    # written in YAML's own flow form (1.0e-07), which reads back as the same data.
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
        try:
            text = json.loads(stripped)
        except ValueError:
            text = None
        if isinstance(text, str):
            plain = not quote and "\n" not in text and _plain_reads_as(text)
            return (text if plain else json.dumps(text, ensure_ascii=False)), text
    if stripped[:1] in ("[", "{"):
        try:
            data = json.loads(stripped)
        except ValueError:
            data = None
        if isinstance(data, (list, dict)):
            return yaml.safe_dump(data, default_flow_style=True, allow_unicode=True, sort_keys=False, width=10**6).strip(), data
    if "\n" not in value:
        try:
            loaded = _load_scalar(value)
        except yaml.YAMLError:
            loaded = _QUOTE
        if loaded is None and value.strip() not in _NULL_WORDS:
            loaded = _QUOTE              # text YAML reads as nothing (#4472C4 is a comment): a string, quoted
        if old_type == "str" and (loaded is None or isinstance(loaded, bool)):
            loaded = _QUOTE              # a key holding text keeps text: a title typed yes / off / null stays that word
        if isinstance(loaded, str) and _NUMBER.match(value.strip()) and old_type in (None, "int", "float", "null"):
            number = float(value)
            return yaml.safe_dump(number).split("\n", 1)[0], number
        if loaded is not _QUOTE and not isinstance(loaded, str):
            return value, loaded
        if loaded == value and not quote:
            return value, loaded
    return json.dumps(value, ensure_ascii=False), value


_QUOTE = object()


def _plain_reads_as(text: str) -> bool:
    """`text` written plain reads back as that very string."""
    try:
        return _load_scalar(text) == text
    except yaml.YAMLError:
        return False
_NULL_WORDS = {"null", "Null", "NULL", "~"}


# ── data mirror of an edit (what the file must read as afterwards) ──────────

def _data_key(container: Any, item: Any) -> Any:
    if isinstance(container, list):
        return item
    for key in container:
        if str(key) == str(item) or (isinstance(key, bool) and str(key).lower() == str(item).lower()):
            return key
    raise EditError(f"no entry {item!r}")


def _data_parent(data: Any, path: list) -> tuple[Any, Any]:
    node = data
    for item in path[:-1]:
        node = node[_data_key(node, item)]
    return node, _data_key(node, path[-1])


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


# ── text edits ───────────────────────────────────────────────────────────────

def _item_lines(text: str, item: Node) -> tuple[int, int]:
    """[start, end) of the whole lines a block list item or mapping entry occupies."""
    return yamlspan.line_start(text, item.start_mark.index), yamlspan.line_end(text, max(yamlspan.content_end(item) - 1, item.start_mark.index))


def _newline(text: str) -> str:
    """The file's line ending: CRLF when it has any, else LF."""
    return "\r\n" if "\r\n" in text else "\n"


def _insert_after(text: str, at: int, chunk: str) -> str:
    nl = _newline(text)
    if not chunk.endswith("\n"):
        chunk += nl
    if at == len(text) and text and not text.endswith("\n"):
        chunk = nl + chunk
    return text[:at] + chunk + text[at:]


def _set(text: str, root: Node, path: list, value: str, data: Any, quote: bool = False) -> str:
    node = yamlspan.find(root, path)
    if isinstance(node, SequenceNode) and node.flow_style:
        # A whole flow list written at once (a [lower, upper] range): only by a value that is a list.
        new, expected = render(value, None, quote)
        if not isinstance(expected, list):
            raise EditError("set: a flow list is replaced only by a list, e.g. [0, 10000]")
        parent, key = _data_parent(data, path)
        parent[key] = expected
        return yamlspan.apply(text, [(node.start_mark.index, yamlspan.content_end(node), new)])
    if isinstance(node, SequenceNode) and path:
        # A block list (one item per line) replaced as a whole by a flow list: the value goes on the
        # key's line, before a comment written there (a facet's arrangement.order from the panel).
        new, expected = render(value, None, quote)
        if not isinstance(expected, list):
            raise EditError("set: a list is replaced only by a list, e.g. [[1, 2], [3]]")
        holder = yamlspan.find(root, path[:-1])
        key_node = next((k for k, v in getattr(holder, "value", []) if v is node and isinstance(holder, MappingNode)), None)
        if key_node is None:
            raise EditError("set: only a list under a key can be replaced as a whole")
        colon = text.index(":", key_node.end_mark.index)
        line_end = text.find("\n", colon)
        rest = text[colon + 1:line_end if line_end >= 0 else len(text)].strip()
        comment = f"   {rest}" if rest.startswith("#") else ""
        parent, key = _data_parent(data, path)
        parent[key] = expected
        return yamlspan.apply(text, [(colon + 1, yamlspan.content_end(node), f" {new}{comment}")])
    if not isinstance(node, ScalarNode):
        raise EditError("set: not a single value; use add/remove/duplicate/delete")
    if node.style in ("|", ">"):
        raise EditError("set: a multi-line text block; edit it in the file")
    new, expected = render(value, node, quote)
    start, end = yamlspan.span(node)
    if start == end and new and text[start - 1:start] in (":", "-"):
        new = " " + new                                    # `key:` -> `key: value`
    parent, key = _data_parent(data, path)
    parent[key] = expected
    return yamlspan.apply(text, [(start, end, new)])


def _key_column(root: Node, path: list) -> int:
    """Column of the mapping key that holds the entry at path (where a child block is indented from)."""
    parent = yamlspan.find(root, path[:-1])
    for key_node, _value in parent.value if isinstance(parent, MappingNode) else []:
        if str(key_node.value) == str(path[-1]):
            return key_node.start_mark.column
    raise EditError(f"no mapping entry {path[-1]!r} at {path[:-1]}")


def _is_null(node: Node) -> bool:
    return isinstance(node, ScalarNode) and node.tag == NULL_TAG


class _BlockDumper(yaml.SafeDumper):
    """A new block as a hand-written file has it: lists indented under their key, a list of
    plain values on one line ([png, pdf], ["line", 1.5])."""

    def increase_indent(self, flow: bool = False, indentless: bool = False):  # noqa: D102
        return super().increase_indent(flow, False)


_BlockDumper.add_representer(list, lambda d, v: d.represent_sequence(
    "tag:yaml.org,2002:seq", v, flow_style=all(not isinstance(x, (dict, list)) for x in v)))


def _dump_block(value: Any) -> list[str]:
    return yaml.dump(value, Dumper=_BlockDumper, sort_keys=False, default_flow_style=False, allow_unicode=True, width=1000).splitlines()


def _block_item(entry: dict, indent: str, nl: str) -> str:
    """A mapping as one block list item: `- key: value` then its other keys under it."""
    body = _dump_block(entry)
    lines = [f"{indent}- {body[0]}"] + [f"{indent}  {line}" for line in body[1:]]
    return nl.join(lines)


def _add_mapping(text: str, root: Node, path: list, entry: dict, data: Any) -> str:
    """Append a mapping item (a text rule, a facet panel) to the list at path, in block style."""
    seq = yamlspan.find(root, path)
    nl = _newline(text)
    parent, key = _data_parent(data, path) if path else (None, None)
    empty = _is_null(seq) or (isinstance(seq, SequenceNode) and seq.flow_style and not seq.value)
    if path and empty:                                         # `text_rules:` or `text_rules: []`: start a block list
        parent[key] = [entry]
        indent = " " * (_key_column(root, path) + 2)
        start = seq.start_mark.index
        end = yamlspan.content_end(seq) if isinstance(seq, SequenceNode) else seq.end_mark.index
        while start > 0 and text[start - 1] == " ":            # `key: []` -> `key:` then the block
            start -= 1
        lead = "" if text[start - 1:start] == "\n" else nl
        return yamlspan.apply(text, [(start, end, f"{lead}{_block_item(entry, indent, nl)}")])
    if not isinstance(seq, SequenceNode):
        raise EditError("add: not a list")
    (parent[key] if path else data).append(entry)
    if seq.flow_style:
        close = yamlspan.content_end(seq) - 1
        flow = yaml.safe_dump(entry, default_flow_style=True, sort_keys=False, width=1000).strip()
        return yamlspan.apply(text, [(close, close, (", " if seq.value else "") + flow)])
    last = seq.value[-1]
    indent = " " * seq.start_mark.column
    end = yamlspan.line_end(text, yamlspan.content_end(last) - 1)
    # the comment lines right under the last item, indented as its keys, are its own: the new item goes after them
    while end < len(text):
        nxt = text[end:].split("\n", 1)[0]
        if not (nxt.lstrip().startswith("#") and len(nxt) - len(nxt.lstrip()) > len(indent)):
            break
        end += len(nxt) + 1
    return _insert_after(text, end, f"{_block_item(entry, indent, nl)}{nl}")


def _mapping_value(value: str) -> Any:
    """A whole block the panel sends as JSON ({"learning_rate": [1e-05]}): read as JSON, so 1e-05 is the
    number it is (YAML 1.1 reads 1e-05 as text); a hand-typed {...} as YAML."""
    if not value or not value.lstrip().startswith("{"):
        return None
    try:
        return json.loads(value)
    except ValueError:
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError:
            return None


def _add(text: str, root: Node, path: list, value: str, data: Any, quote: bool = False) -> str:
    parsed = _mapping_value(value)
    if isinstance(parsed, dict):
        return _add_mapping(text, root, path, parsed, data)
    seq = yamlspan.find(root, path)
    if path and _is_null(seq):                                   # `models:` (empty): start a block list
        new, expected = render(value, None, quote)
        if not new:
            raise EditError("add: empty value")
        parent, key = _data_parent(data, path)
        parent[key] = [expected]
        indent = " " * (_key_column(root, path) + 2)
        at = seq.start_mark.index
        return yamlspan.apply(text, [(at, seq.end_mark.index, f"{_newline(text)}{indent}- {new}")])
    if path and isinstance(seq, ScalarNode) and seq.style not in ("|", ">"):   # `seed: 42`: one value becomes [42, 43]
        new, expected = render(value, seq, quote)
        if not new:
            raise EditError("add: empty value")
        parent, key = _data_parent(data, path)
        parent[key] = [parent[key], expected]
        start, end = yamlspan.span(seq)
        return yamlspan.apply(text, [(start, end, f"[{text[start:end]}, {new}]")])
    if not yamlspan.is_scalar_list(seq):
        raise EditError("add: not a list of values")
    new, expected = render(value, seq.value[-1] if seq.value else None, quote)
    if not new:
        raise EditError("add: empty value")
    parent, key = _data_parent(data, path) if path else (None, None)
    (parent[key] if path else data).append(expected)
    if seq.flow_style:
        close = yamlspan.content_end(seq) - 1                       # the `]`
        return yamlspan.apply(text, [(close, close, (", " if seq.value else "") + new)])
    last = seq.value[-1]
    indent = " " * seq.start_mark.column
    return _insert_after(text, yamlspan.line_end(text, last.end_mark.index - 1), f"{indent}- {new}{_newline(text)}")


def _put(text: str, root: Node, path: list, key: str, value: str, data: Any, quote: bool = False) -> str:
    """Set the entry `key` of the mapping at path, adding it when missing."""
    key = str(key or "").strip()
    if not key:
        raise EditError("put: no key")
    node = yamlspan.find(root, path)
    if isinstance(node, MappingNode) and any(str(k.value) == key for k, _ in node.value):
        return _set(text, root, path + [key], value, data, quote)
    if not (isinstance(node, MappingNode) or (path and _is_null(node))):
        raise EditError("put: not a mapping")
    parsed = _mapping_value(value)
    if isinstance(parsed, dict) and parsed and isinstance(node, MappingNode) and not node.flow_style and node.value:
        # A whole block added under a block mapping (export.svg with its defaults): written in block style.
        parent, pkey = _data_parent(data, path) if path else (None, None)
        (parent[pkey] if path else data)[key] = parsed
        nl = _newline(text)
        indent = " " * node.value[0][0].start_mark.column
        body = _dump_block(parsed)
        block = nl.join([f"{indent}{key}:"] + [f"{indent}  {line}" for line in body])
        last_key, last_value = node.value[-1]
        end = yamlspan.line_end(text, max(yamlspan.content_end(last_value) - 1, last_key.end_mark.index))
        return _insert_after(text, end, f"{block}{nl}")
    new, expected = render(value, None, quote)
    key_text = key if isinstance(_load_scalar(key), str) and _load_scalar(key) == key else json.dumps(key, ensure_ascii=False)
    parent, pkey = _data_parent(data, path) if path else (None, None)
    target = parent[pkey] if path else data
    if target is None:
        target = parent[pkey] = {}
    target[key] = expected
    entry = f"{key_text}: {new}" if new else f"{key_text}:"
    nl = _newline(text)
    if _is_null(node):                                           # `labels:` (empty): start a block mapping
        indent = " " * (_key_column(root, path) + 2)
        return yamlspan.apply(text, [(node.start_mark.index, node.end_mark.index, f"{nl}{indent}{entry}")])
    if node.flow_style:
        close = yamlspan.content_end(node) - 1                   # the `}`
        return yamlspan.apply(text, [(close, close, (", " if node.value else "") + entry)])
    if not path:                                                 # a new top-level key: the file's last entry, after
        body = text if text.endswith(nl) else text + nl          # the comments closing the one before (they are its)
        return f"{body}{nl}{entry}{nl}"
    last_key, last_value = node.value[-1]
    indent = " " * node.value[0][0].start_mark.column
    end = yamlspan.line_end(text, max(yamlspan.content_end(last_value) - 1, last_key.end_mark.index))
    return _insert_after(text, end, f"{indent}{entry}{nl}")


def _drop_item(text: str, seq: SequenceNode, index: int) -> str:
    item = seq.value[index]
    if seq.flow_style:
        if len(seq.value) == 1:
            return yamlspan.apply(text, [(item.start_mark.index, yamlspan.content_end(item), "")])
        if index < len(seq.value) - 1:
            return yamlspan.apply(text, [(item.start_mark.index, seq.value[index + 1].start_mark.index, "")])
        return yamlspan.apply(text, [(yamlspan.content_end(seq.value[index - 1]), yamlspan.content_end(item), "")])
    if len(seq.value) == 1:                                # an emptied block list is `[]`, not null
        return yamlspan.apply(text, [(seq.start_mark.index, yamlspan.content_end(seq), "[]")])
    start, end = _item_lines(text, item)
    return yamlspan.apply(text, [(start, end, "")])


def _remove(text: str, root: Node, path: list, data: Any) -> str:
    if not path or not isinstance(path[-1], int):
        raise EditError("remove: not a list item")
    seq = yamlspan.find(root, path[:-1])
    if not yamlspan.is_scalar_list(seq):
        raise EditError("remove: not an item of a list of values; use delete")
    return _delete(text, root, path, data)


def _move(text: str, root: Node, path: list, value: str, data: Any) -> str:
    """A block list item swapped with its neighbour (value -1: the one before, 1: the one after): the
    lines of both items trade places; the comments between them stay where they are."""
    seq = yamlspan.find(root, path[:-1]) if path else None
    if not isinstance(seq, SequenceNode) or seq.flow_style:
        raise EditError("move: only items of a block list")
    i = int(path[-1])
    j = i + (1 if str(value).strip() in ("1", "+1", "down") else -1)
    if not (0 <= i < len(seq.value) and 0 <= j < len(seq.value)):
        raise EditError(f"move: item {i} has no neighbour there")
    parent, key = _data_parent(data, path)
    parent[i], parent[j] = parent[j], parent[i]
    (a0, a1), (b0, b1) = sorted([_item_lines(text, seq.value[i]), _item_lines(text, seq.value[j])])
    return text[:a0] + text[b0:b1] + text[a1:b0] + text[a0:a1] + text[b1:]


def _delete(text: str, root: Node, path: list, data: Any) -> str:
    if not path:
        raise EditError("delete: the whole file")
    parent_node = yamlspan.find(root, path[:-1])
    parent, key = _data_parent(data, path)
    if isinstance(parent_node, SequenceNode):
        index = int(path[-1])
        if not 0 <= index < len(parent_node.value):
            raise EditError(f"delete: no item {index}")
        del parent[key]
        return _drop_item(text, parent_node, index)
    if not isinstance(parent_node, MappingNode) or parent_node.flow_style:
        raise EditError("delete: only entries of block mappings and lists; edit flow mappings in the file")
    pairs = [(k, v) for k, v in parent_node.value if str(k.value) == str(path[-1])]
    if not pairs:
        raise EditError(f"delete: no entry {path[-1]!r}")
    key_node, value_node = pairs[0]
    del parent[key]
    if len(parent_node.value) == 1:
        if len(path) == 1:
            raise EditError("delete: a config file keeps at least one entry")
        # the mapping's only entry: the mapping becomes `{}` (a block mapping with no entry would be null)
        return yamlspan.apply(text, [(parent_node.start_mark.index, yamlspan.content_end(parent_node), "{}")])
    start = yamlspan.line_start(text, key_node.start_mark.index)
    end = yamlspan.line_end(text, max(yamlspan.content_end(value_node) - 1, key_node.end_mark.index))
    return yamlspan.apply(text, [(start, end, "")])


def _duplicate(text: str, root: Node, path: list, value: str | None, data: Any) -> str:
    if not path:
        raise EditError("duplicate: the whole file")
    parent_node = yamlspan.find(root, path[:-1])
    parent, key = _data_parent(data, path)
    if isinstance(parent_node, SequenceNode):
        index = int(path[-1])
        item = parent_node.value[index]
        parent.insert(index + 1, copy.deepcopy(parent[key]))
        if parent_node.flow_style:
            end = yamlspan.content_end(item)
            return yamlspan.apply(text, [(end, end, ", " + text[item.start_mark.index:end])])
        start, end = _item_lines(text, item)
        return _insert_after(text, end, text[start:end])
    if not isinstance(parent_node, MappingNode) or parent_node.flow_style:
        raise EditError("duplicate: only items of lists and entries of block mappings")
    new_key = (value or "").strip()
    if not new_key:
        raise EditError("duplicate: a mapping entry needs a new key")
    if any(str(k.value) == new_key for k, _ in parent_node.value):
        raise EditError(f"duplicate: {new_key!r} exists")
    key_text, _ = render(new_key, None)
    if not isinstance(_load_scalar(key_text), str):
        key_text = json.dumps(new_key, ensure_ascii=False)          # a key is a name, never a number
    key_node, value_node = next((k, v) for k, v in parent_node.value if str(k.value) == str(path[-1]))
    start = yamlspan.line_start(text, key_node.start_mark.index)
    end = yamlspan.line_end(text, max(yamlspan.content_end(value_node) - 1, key_node.end_mark.index))
    chunk = text[start:end]
    k0, k1 = key_node.start_mark.index - start, key_node.end_mark.index - start
    parent[new_key] = copy.deepcopy(parent[key])
    return _insert_after(text, end, chunk[:k0] + key_text + chunk[k1:])


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _layer_chunks(text: str, seq: SequenceNode) -> list[tuple[int, int, int, int]]:
    """Per item of a block list: (chunk_start, item_start, item_end, chunk_end). The chunk is the item's lines
    with the comment lines right above it (no blank line between) and the comment lines under it indented
    deeper than its dash (its own); what lies between two chunks (blank lines, comments set apart) is a gap."""
    lines = text.splitlines(keepends=True)
    starts, at = [], 0
    for line in lines:
        starts.append(at)
        at += len(line)
    line_of = {s: i for i, s in enumerate(starts)}
    dash = seq.start_mark.column
    out, floor = [], 0                      # the first item takes the comment lines right above it, under its key
    for item in seq.value:
        item_start, item_end = _item_lines(text, item)
        first = line_of[item_start]
        while first > 0 and starts[first - 1] >= floor and _is_comment(lines[first - 1]):
            first -= 1
        last = line_of.get(item_end, len(lines))
        while last < len(lines) and _is_comment(lines[last]) and len(lines[last]) - len(lines[last].lstrip()) > dash:
            last += 1
        chunk_end = starts[last] if last < len(lines) else len(text)
        out.append((starts[first], item_start, item_end, chunk_end))
        floor = chunk_end
    return out


def _layer_text(entry: Any, indent: str, nl: str, flow: bool) -> str:
    """One list item written fresh: `- Name: {args}` (flow, as the list's items are) or as a block."""
    if flow and isinstance(entry, dict) and len(entry) == 1:
        name, args = next(iter(entry.items()))
        body = yaml.safe_dump(args if args is not None else {}, default_flow_style=True, sort_keys=False, width=10000).strip()
        return f"{indent}- {name}: {body}{nl}"
    return _block_item(entry, indent, nl) + nl


def _value_text(v: Any) -> str:
    """A value as the set / put ops take it (YAML text)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return str(v)


def _arrange_entries(value: str) -> list:
    try:
        entries = json.loads(value)
    except ValueError as exc:
        raise EditError(f"arrange: VALUE is not JSON ({exc})") from None
    if not isinstance(entries, list) or not all(isinstance(e, dict) and ("from" in e or "value" in e) for e in entries):
        raise EditError('arrange: VALUE must be a list of {"from": k} / {"from": k, "value": v} / {"value": v}')
    return entries


def _flow_item(item: Node) -> bool:
    return isinstance(item, MappingNode) and len(item.value) == 1 and bool(getattr(item.value[0][1], "flow_style", False))


def _arrange(text: str, root: Node, path: list, value: str, data: Any) -> str:
    """The items of the list at path in a new order, each kept with its own comments: VALUE is a JSON list of
    {"from": k} (item k as written), {"from": k, "value": v} (item k with its arguments changed: a block item
    through set / put / delete of each changed key, so the comments inside it stay; a flow item's line rewritten
    with its trailing comment kept) and {"value": v} (a new item in the list's style). A gap (blank lines,
    comments set apart) stays before the item it preceded; an emptied list's gaps follow the last item. An
    empty list (`[]` or nothing) is filled with the new items."""
    entries = _arrange_entries(value)
    seq = yamlspan.find(root, path)
    parent, key = _data_parent(data, path)
    if _is_null(seq) or (isinstance(seq, SequenceNode) and not seq.value):          # an empty list: append each new item
        if any("from" in e for e in entries):
            raise EditError("arrange: the list is empty; only new items can go into it")
        for e in entries:
            text = edit_text(text, "add", path, json.dumps(e["value"]))
        parent[key] = [e["value"] for e in entries]
        return text
    if not isinstance(seq, SequenceNode) or seq.flow_style:
        raise EditError("arrange: only a block list")
    old = list(parent[key])
    for e in entries:
        k = e.get("from")
        if "from" in e and (not isinstance(k, int) or not 0 <= k < len(seq.value)):
            raise EditError(f"arrange: no layer {k} (the list has {len(seq.value)})")
    # 1. a changed block item: its keys set, added or dropped one by one (its comments stay); indices unchanged
    for e in entries:
        if "from" not in e or "value" not in e or _same(e["value"], old[e["from"]]):
            continue
        item, was = seq.value[e["from"]], old[e["from"]]
        new = e["value"]
        if _flow_item(item) or not (isinstance(was, dict) and isinstance(new, dict) and len(was) == len(new) == 1
                                    and next(iter(was)) == next(iter(new)) and isinstance(was[next(iter(was))], dict)
                                    and isinstance(new[next(iter(new))], dict)):
            continue                                       # a flow line or another layer: rewritten in step 2
        name = next(iter(new))
        at = path + [e["from"], name]
        before, after = was[name], new[name]
        for arg in after:
            if arg not in before:
                text = edit_text(text, "put", at, _value_text(after[arg]), arg)
            elif not _same(before[arg], after[arg]):
                text = edit_text(text, "set", at + [arg], _value_text(after[arg]))
        for arg in before:
            if arg not in after:
                text = edit_text(text, "delete", at + [arg])
        e["done"] = True
    root = yamlspan.compose(text)
    seq = yamlspan.find(root, path)
    nl = _newline(text)
    chunks = _layer_chunks(text, seq)
    indent = " " * seq.start_mark.column
    flow = sum(_flow_item(i) for i in seq.value) * 2 >= len(seq.value)
    gap_before = [""] + [text[chunks[i - 1][3]:chunks[i][0]] for i in range(1, len(chunks))]
    used, held = set(), ""
    body, new_data = "", []
    for pos, e in enumerate(entries):
        if "from" in e:
            k = e["from"]
            c0, i0, i1, c1 = chunks[k]
            used.add(k)
            if "value" in e and not e.get("done") and not _same(e["value"], old[k]):
                rest = text[yamlspan.content_end(seq.value[k].value[0][1]):i1] if _flow_item(seq.value[k]) else nl
                piece = text[c0:i0] + _layer_text(e["value"], indent, nl, _flow_item(seq.value[k])).rstrip("\r\n") + \
                    (rest if rest.strip() else nl) + text[i1:c1]
            else:
                piece = text[c0:c1]
            if pos == 0:                                   # moved to the top: its gap follows it
                body += piece
                held = gap_before[k]
            else:
                body += held + gap_before[k] + piece
                held = ""
            new_data.append(e["value"] if "value" in e else old[k])
        else:
            body += held + _layer_text(e["value"], indent, nl, flow)
            held = ""
            new_data.append(e["value"])
    body += held + "".join(gap_before[k] for k in range(len(chunks)) if k not in used)
    parent[key] = new_data
    return text[:chunks[0][0]] + body + text[chunks[-1][3]:]


def edit_text(text: str, op: str, path: list, value: str | None = None, key: str | None = None) -> str:
    """The text after one edit; EditError when refused or when the result is not exactly that edit.

    A value written plain that does not read back (``a, b`` inside a flow
    mapping) is tried once more, quoted.
    """
    root = yamlspan.compose(text)
    if root is None:
        raise EditError("the file has no entries")
    data = yamlspan.load(text)
    problem = ""
    for quote in (False, True) if op in ("set", "add", "put") else (False,):
        expected = copy.deepcopy(data)
        try:
            if op == "set":
                new = _set(text, root, path, value or "", expected, quote)
            elif op == "add":
                new = _add(text, root, path, value or "", expected, quote)
            elif op == "put":
                new = _put(text, root, path, key, value or "", expected, quote)
            elif op == "remove":
                new = _remove(text, root, path, expected)
            elif op == "delete":
                new = _delete(text, root, path, expected)
            elif op == "move":
                new = _move(text, root, path, value or "", expected)
            elif op == "duplicate":
                new = _duplicate(text, root, path, value, expected)
            elif op == "arrange":
                new = _arrange(text, root, path, value or "", expected)
            else:
                raise EditError(f"unknown op {op!r}")
        except (KeyError, IndexError) as exc:
            raise EditError(f"{op}: no entry at {path} ({exc})") from None
        try:
            got = yamlspan.load(new)
        except yaml.YAMLError as exc:
            problem = f"{op}: the result would not be valid YAML ({exc.__class__.__name__})"
            continue
        if _same(got, expected):
            return new
        problem = f"{op}: the result does not read back as that one change; nothing written"
    raise EditError(problem)


# ── file ─────────────────────────────────────────────────────────────────────

def _guard(path: Path) -> None:
    blueprint = (Path(__file__).resolve().parents[3] / BLUEPRINT).resolve()
    if path.resolve().is_relative_to(blueprint):
        raise EditError("the blueprint is never edited")


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.chmod(tmp, path.stat().st_mode & 0o7777)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read(path: Path) -> str:
    """The file's text with its line endings as they are (a CRLF file stays CRLF when written back)."""
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def edit_file(path: Path, sha: str, op: str, item_path: list, value: str | None = None,
              key: str | None = None) -> dict[str, Any]:
    _guard(path)
    text = read(path)
    if sha256(text) != sha:
        return {"stale": True, "sha256": sha256(text)}
    new = edit_text(text, op, item_path, value, key)
    _atomic_write(path, new)
    return {"ok": True, "sha256": sha256(new)}


OPS = ("set", "put", "add", "remove", "duplicate", "delete", "move", "arrange")


def preview(text: str, op: str, item_path: list, value: str | None = None, key: str | None = None) -> dict[str, Any]:
    """What one edit would write (the planner's Preview Code): the new text and its added and removed lines."""
    import difflib

    new = edit_text(text, op, item_path, value, key)
    diff = [d for d in difflib.ndiff(text.splitlines(), new.splitlines()) if d[:2] in ("+ ", "- ")]
    return {"text": new, "added": [d[2:] for d in diff if d.startswith("+ ")],
            "removed": [d[2:] for d in diff if d.startswith("- ")]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TraceFlow Monitor config editor")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tree")
    t.add_argument("file")
    e = sub.add_parser("edit")
    e.add_argument("file")
    e.add_argument("--sha", required=True)
    e.add_argument("--op", required=True, choices=OPS)
    e.add_argument("--path", required=True, help="JSON list of keys and indices")
    e.add_argument("--value")
    e.add_argument("--key", help="put: the mapping entry to set or add")
    a = sub.add_parser("value", help="the data at a path, as JSON (the planner reads a model's layers)")
    a.add_argument("file")
    a.add_argument("--path", required=True)
    v = sub.add_parser("preview", help="the text one edit would write, and its added / removed lines; nothing written")
    v.add_argument("file")
    v.add_argument("--op", required=True, choices=OPS)
    v.add_argument("--path", required=True)
    v.add_argument("--value")
    v.add_argument("--key")
    args = parser.parse_args(argv)
    path = Path(args.file)
    try:
        if args.command == "tree":
            out = tree(read(path))
        else:
            item_path = json.loads(args.path)
            if not isinstance(item_path, list) or not all(isinstance(i, (str, int)) for i in item_path):
                raise EditError("--path: a JSON list of keys and indices")
            if args.command == "value":
                found = yamlspan.load(read(path))
                for step in item_path:
                    found = found[step]
                out = {"value": found}
            elif args.command == "preview":
                out = preview(read(path), args.op, item_path, args.value, args.key)
            else:
                out = edit_file(path, args.sha, args.op, item_path, args.value, args.key)
    except (EditError, OSError, yaml.YAMLError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        out = {"error": f"{exc.__class__.__name__ if not isinstance(exc, EditError) else 'refused'}: {exc}"}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
