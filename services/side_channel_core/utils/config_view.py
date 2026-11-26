# services/side_channel_core/utils/config_view.py
"""
The resolved view of a run configuration: what the run actually uses.

Applied to the params_out snapshot and to the CONFIG dump in the run logs; the
in-memory configuration the run trains with is never changed.

Two rules:

1. Inactive alternatives are dropped, at any depth. A mapping holds a
   discriminated (tagged) union when one of its string values names a sibling
   key whose value is a mapping: the string is the discriminator, the named
   sibling the selected variant, and every other sibling mapping that shares
   at least half of its keys with the selected one is an alternative. Only the
   selected variant is kept:

       deep_learning: {framework: keras, pytorch: {...}, keras: {...}}
         -> deep_learning: {framework: keras, keras: {...}}

   Sibling mappings unrelated to the variant (fewer shared keys) stay. Lists of
   alternatives (model blocks by model_id, data entries by name) are already
   reduced to the selected entry by the sweep before a run starts.

2. Sections other tools own are dropped: `graph` (graph_config.yaml), `test`
   (test_config.yaml) and `import` (import_config.yaml), merged by master.yaml
   extra_configs. The graph manager loads graph_config.yaml itself, make test
   reads the model names from test_config.yaml itself (sca/test_request.py),
   and make import reads import_config.yaml itself (utils/model_import.py: an
   imported run carries its own `model_import` record instead); the test sweep
   creates the run's own `test` block (utils/sweep_test_runner.py). No
   training, test or retry step reads any of them from a snapshot, and a
   snapshot holds what the run was trained with, not what is asked of it later.

Neither rule touches anything the model hash reads: the hash takes the selected
deep_learning variant only (framework_switch.FrameworkSpec.hash_payload), and
of these sections only the file paths in project.extra_configs, which stay.

Author: Shraddha Mishra
"""

from __future__ import annotations

import copy
from typing import Any

# Top-level sections left out of the snapshot and the logged config.
EXCLUDED_SECTIONS = ("graph", "test", "import")
# Also left out of the logged CONFIG only (the snapshot keeps them): none.
LOG_ONLY_EXCLUDED_SECTIONS: tuple[str, ...] = ()


def _shares_structure(selected: dict, other: dict) -> bool:
    """True when `other` is an alternative to `selected`: at least half of the smaller key set is shared."""
    a, b = set(selected), set(other)
    if not a or not b:
        return False
    return 2 * len(a & b) >= min(len(a), len(b))


def drop_unselected_variants(node: Any) -> Any:
    """A copy of `node` with every discriminated union reduced to its selected variant (rule 1)."""
    if isinstance(node, dict):
        selected = {
            value for key, value in node.items()
            if isinstance(value, str) and value != key and isinstance(node.get(value), dict)
        }
        alternatives = {
            key for key, value in node.items()
            if key not in selected and isinstance(value, dict)
            and any(_shares_structure(node[s], value) for s in selected)
        }
        return {key: drop_unselected_variants(value) for key, value in node.items() if key not in alternatives}
    if isinstance(node, list):
        return [drop_unselected_variants(item) for item in node]
    return copy.deepcopy(node)


def resolved_view(cfg: dict, *, for_log: bool = False) -> dict:
    """The configuration as the run uses it: excluded sections dropped, unions reduced (rules 1 and 2).

    for_log: the CONFIG block of the run logs, which also leaves out LOG_ONLY_EXCLUDED_SECTIONS.
    """
    if not isinstance(cfg, dict):
        return cfg
    drop = EXCLUDED_SECTIONS + (LOG_ONLY_EXCLUDED_SECTIONS if for_log else ())
    kept = {key: value for key, value in cfg.items() if key not in drop}
    return drop_unselected_variants(kept)
