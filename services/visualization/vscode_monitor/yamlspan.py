# services/visualization/vscode_monitor/yamlspan.py
"""
Where each entry of a YAML file sits in its text.

PyYAML's composer keeps, for every node, the character offsets where it
starts and ends (``start_mark.index`` / ``end_mark.index``). Replacing exactly
that span changes one entry and leaves every comment, blank line and quote
elsewhere in the file as it was. The blueprint builder and the panel's editor
(``yamledit.py``) both edit YAML this way.

Paths address entries the way the panel shows them: a list of mapping keys
(strings) and sequence indices (ints), e.g. ``["trainer", "optimizer", 0,
"learning_rate"]``.

Author: Shraddha Mishra
"""
from __future__ import annotations

from typing import Iterator, Union

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

PathItem = Union[str, int]


# libyaml when PyYAML was built with it: the same marks (character offsets), ten times faster
# on a 10 000-line graph_config.yaml. Its plain scalars have style '' where PyYAML's have None.
LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def compose(text: str) -> Node | None:
    """The node tree of one YAML document (None for an empty document)."""
    return yaml.compose(text, Loader=LOADER)


def load(text: str):
    """The data of one YAML document, as yaml.safe_load gives it."""
    return yaml.load(text, Loader=LOADER)


def span(node: Node) -> tuple[int, int]:
    """Character offsets [start, end) of a node in the text it was composed from."""
    return node.start_mark.index, node.end_mark.index


def children(node: Node) -> Iterator[tuple[PathItem, Node, Node | None]]:
    """(path item, value node, key node or None) for every child of a collection."""
    if isinstance(node, MappingNode):
        for key_node, value_node in node.value:
            yield str(key_node.value), value_node, key_node
    elif isinstance(node, SequenceNode):
        for index, item in enumerate(node.value):
            yield index, item, None


def content_end(node: Node) -> int:
    """Offset just after the node's last real character.

    A block collection's ``end_mark`` runs on to the next token, through the
    comments and blank lines after it; those belong to whatever follows, so
    an edit that replaces or deletes a block stops here instead.
    """
    if isinstance(node, ScalarNode) or getattr(node, "flow_style", False):
        return node.end_mark.index
    items = list(children(node))
    if not items:
        return node.end_mark.index
    return content_end(items[-1][1])


def line_end(text: str, index: int) -> int:
    """Offset just after the newline that ends the line holding ``index`` (or len(text))."""
    newline = text.find("\n", index)
    return len(text) if newline < 0 else newline + 1


def find(root: Node, path: list[PathItem]) -> Node:
    """The node at ``path``; KeyError when an item of the path does not exist."""
    node = root
    for item in path:
        for name, child, _key in children(node):
            if name == item:
                node = child
                break
        else:
            raise KeyError(f"no entry {item!r} under {path}")
    return node


def is_scalar_list(node: Node) -> bool:
    """A sequence whose items are all scalars (a value list such as a sweep)."""
    return isinstance(node, SequenceNode) and all(isinstance(i, ScalarNode) for i in node.value)


def line_start(text: str, index: int) -> int:
    """Offset of the first character of the line holding ``index``."""
    return text.rfind("\n", 0, index) + 1


def apply(text: str, edits: list[tuple[int, int, str]]) -> str:
    """Replace each [start, end) span by its new text; spans must not overlap."""
    out = text
    for start, end, new in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + new + out[end:]
    return out
