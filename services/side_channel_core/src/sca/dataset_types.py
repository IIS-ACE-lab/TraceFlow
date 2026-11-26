# services/side_channel_core/src/sca/dataset_types.py
"""
Database types: what a `data:` entry's `type` switches on.

    data:
      - name: ascad          # free
        type: ASCADf         # one of TYPES
        database: ...        # exactly the keys the type allows

Each type states its key mode (sca.variant: fixed_key | variable_key) and the
keys its data entries require and allow. A data entry with an unknown type, an
unknown key or a missing required key is a configuration error
(DatasetTypeError) when the configuration is loaded (utils/config_loader.py).

Backward compatibility: an entry without `type` (every run and snapshot made
before types existed) whose database lies in the ASCAD v1 fixed-key folder is
ASCADf; any other entry without `type` is refused. sca.variant is no longer a
setting: the type gives it (dataset_variant), and a sca.variant still present,
as in an old snapshot, must agree with the type.

Author: Shraddha Mishra
"""

from __future__ import annotations

from services.side_channel_core.utils.config_errors import ConfigError
import math
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping, Any, Mapping, Optional


class DatasetTypeError(ConfigError):     # a configuration error (utils/config_errors.py), a ValueError as before
    """A data entry does not agree with its database type."""


@dataclass(frozen=True)
class SourceFile:
    """One database file of a type, where it lives and where it comes from (utils/dataset_fetch.py)."""
    path: str                       # under services/side_channel_core/data/
    sha256: str                     # of the file itself, as its authors publish it
    url: str                        # the official download
    member: Optional[str] = None    # the file inside the zip archive at url
    archive_sha256: Optional[str] = None
    # The key bytes whose processing the stored trace window was cut around (ANSSI's ASCAD.h5 files:
    # byte 2); a run attacking another byte is refused (require_attack_byte). None: any byte (raw
    # traces, whose window the data entry chooses and verifies, and every non-ASCAD file).
    window_bytes: Optional[frozenset[int]] = None


@dataclass(frozen=True)
class Conversion:
    """A database made once from its official downloads, in the ASCAD layout the runtime reads
    (src/sca/converters/<module>.py convert(inputs, output))."""
    output: str                          # under services/side_channel_core/data/
    inputs: tuple[SourceFile, ...]       # the downloads, each checked against its SHA-256
    converter: str                       # module name under sca.converters
    version: int = 1                     # recorded in the output; a new version rebuilds it


@dataclass(frozen=True)
class DatasetType:
    name: str
    variant: str                    # fixed_key | variable_key (sca.variant)
    required: frozenset[str]        # data-entry keys the type needs
    optional: frozenset[str]        # data-entry keys the type also accepts
    description: str
    sources: tuple[SourceFile, ...] = ()
    conversions: tuple[Conversion, ...] = ()
    shared_key: bool = True         # fixed_key: profiling and attack use the same key
    random_profiling_keys: bool = False   # a random key per profiling trace; the attack set's is fixed
    target: str = "sbox"            # the label intermediate (sca/leakage_models.py TARGETS)
    attack_bytes: Optional[frozenset[int]] = None   # the key bytes the target is defined for; None: any

    @property
    def allowed(self) -> frozenset[str]:
        return self.required | self.optional | {"name", "type", "alias", *TEST_TIME_KEYS}


# The keys of every ASCAD v1 entry of DLSCA_ASCAD (ASCAD.h5, its desync files and the raw
# ATMega8515 traces) and of the blueprint params.yaml.
_ASCAD_V1_REQUIRED = frozenset({"database", "profiling_range", "attack_range", "test_range", "target_range"})
_ASCAD_V1_OPTIONAL = frozenset({
    "generate_data_out", "debug_h5_inspect", "show_contract", "require_contract", "multilabel",
    "profiling_desync", "attack_desync", "validation_range", "trace_normalization",
    "trace_window_verified_by_attack_byte",
    # per-byte windows of a raw trace file (ascad_data_loader._target_points_for_attack_byte); joins the hash
    "target_range_by_attack_byte",
    # the profiling rows trained on, drawn from profiling_range (ascad_data_loader.profiling_indices); joins the hash
    "profiling_split",
    # operator switches (data_split, the contract check): rebuild the run's files, their chunking and how many rows the
    # contract verifies; none changes the experiment, so utils/model_naming leaves them out of the hash
    "force_regenerate_dataset", "force_regenerate_splits", "split_chunk_size", "contract_verification_samples",
})
# The entry keys read only when a trained run is tested (owner 2026-10-27), left out of the run hash: attack_trace_budget,
# the traces of one test attack for every run on the entry (utils/sweep_test_runner.apply_data_budget).
TEST_TIME_KEYS = frozenset({"attack_trace_budget"})
# The entry keys that never change the experiment: left out of the run hash (utils/model_naming.make_experiment_hash)
OPERATOR_KEYS = frozenset({"generate_data_out", "force_regenerate_dataset", "force_regenerate_splits", "split_chunk_size",
                           "contract_verification_samples"})

# ANSSI-FR/ASCAD ATMEGA_AES_v1/ATM_AES_v1_fixed_key/Readme.md: one zip, SHA-256 per member.
_ASCADF = "ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/"
_ASCADF_ZIP = "https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e"
_ASCADF_ZIP_SHA256 = "a6884faf97133f9397aeb1af247dc71ab7616f3c181190f127ea4c474a0ad72c"
# ANSSI-FR/ASCAD ATMEGA_AES_v1/ATM_AES_v1_variable_key/Readme.md: one download per file.
_ASCADR = "ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases/"
_GOUV = "https://www.data.gouv.fr/api/1/datasets/r/"
# ANSSI cut the window of every ASCAD.h5 file (and its desync copies) around the first-round S-box of
# the third key byte, index 2 (ATM_AES_v1_fixed_key and ATM_AES_v1_variable_key Readme.md;
# Benadjila et al. Sect. 2.5.2): fixed key 700 samples [45400, 46100) of the raw traces, variable key 1,400.
_ASCAD_WINDOW_BYTES = frozenset({2})
# Zaid et al. 2020 repository (gabzai/Methodology-for-efficient-CNN-architectures-in-SCA) at commit ece9760;
# the official AES_HD repository (AESHD/AES_HD_Dataset) at commit 8baa486.
_ZAID = "https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/ece9760d9af0d871b1e4f682fd4cfcd18ce8c246/"
_AESHD = "https://raw.githubusercontent.com/AESHD/AES_HD_Dataset/8baa486eeb38a043ed4dafadf55bcd03e8846def/"

TYPES: dict[str, DatasetType] = {
    t.name: t for t in (
        DatasetType(
            name="ASCADf",
            variant="fixed_key",
            required=_ASCAD_V1_REQUIRED,
            optional=_ASCAD_V1_OPTIONAL,
            description="ASCAD v1 fixed key (ATMega8515, Boolean masking), Benadjila et al.",
            sources=tuple(
                SourceFile(_ASCADF + name, sha, _ASCADF_ZIP, member="ASCAD_data/ASCAD_databases/" + name,
                           archive_sha256=_ASCADF_ZIP_SHA256,
                           window_bytes=None if name.startswith("ATMega8515_raw") else _ASCAD_WINDOW_BYTES)
                for name, sha in (
                    ("ASCAD.h5", "f56625977fb6db8075ab620b1f3ef49a2a349ae75511097505855376e9684f91"),
                    ("ASCAD_desync50.h5", "8716a01d4aea2df0650636504803af57dd597623854facfa75beae5a563c0937"),
                    ("ASCAD_desync100.h5", "f6b9e967af287e82f0a152320e58f8f0ded35cd74d499b5f7b1505a5ce338b8e"),
                    ("ATMega8515_raw_traces.h5", "51e722f6c63a590ce2c4633c9a9534e8e1b22a9cde8e4532e32c11ac089d4625"),
                )
            ),
        ),
        DatasetType(
            name="ASCADr",
            # A random key per profiling trace (two acquisitions in three), one fixed key in the attack
            # set (ATM_AES_v1_variable_key/Readme.md): labels take each trace's own key, and the attack
            # is a fixed-key attack.
            variant="fixed_key",
            random_profiling_keys=True,
            shared_key=False,
            required=_ASCAD_V1_REQUIRED,
            optional=_ASCAD_V1_OPTIONAL,
            description="ASCAD v1 variable key (ATMega8515, Boolean masking), 200k profiling / 100k attack x 1,400",
            sources=(
                SourceFile(_ASCADR + "ASCAD.h5", "d834da6ca5a288c4ba5add8e336845270a055d6eaf854dcf2f325a2eb6d7de06",
                           _GOUV + "b4ace767-c2a4-4db4-8e01-4527b5b91f00", window_bytes=_ASCAD_WINDOW_BYTES),
                SourceFile(_ASCADR + "ASCAD_desync50.h5", "0fa048bf42b9d8bbf9514770072c070175637e1b6fb6da370e2b020b1ecca673",
                           _GOUV + "4ad6d44a-f6de-483f-807f-d0ccab76d2a9", window_bytes=_ASCAD_WINDOW_BYTES),
                SourceFile(_ASCADR + "ASCAD_desync100.h5", "cb82c553b84c29454ea23ec043730ed845e17c2a0e261853afc11e028fdf2710",
                           _GOUV + "f1936388-71be-408f-b8ec-472bb3398e39", window_bytes=_ASCAD_WINDOW_BYTES),
                SourceFile(_ASCADR + "ATMega8515_raw_traces.h5", "6f13d7c380c937892c09b439910c4313d551adf011d2f4d76ad8b9b3de27b852",
                           _GOUV + "3217dcc0-184f-402b-8914-e31cc120c51c"),
            ),
        ),
        DatasetType(
            name="CHES_CTF_2018",
            # One fixed key in the profiling set, another fixed key in the attack set; float32 traces
            # z-scored by the redistributor (AISyLab/EnsembleSCA README). Licence of the redistributed
            # file not stated; the original CTF data (Zenodo 3733418) is CC-BY-4.0.
            variant="fixed_key",
            required=_ASCAD_V1_REQUIRED,
            optional=_ASCAD_V1_OPTIONAL,
            description="CHES CTF 2018 (STM32F4, Boolean masking), AISyLab copy: 45k profiling / 5k attack x 2,200",
            shared_key=False,       # profiling key 175cf299..., attack key 2eee5e79...
            conversions=(
                Conversion(
                    output="CHES_CTF_2018/CHES_CTF_2018_data/CHES_CTF_2018.h5",
                    inputs=(SourceFile("CHES_CTF_2018/CHES_CTF_2018_data/source/ches_ctf.h5",
                                       "af47caa04894be0fa68a78147826ce07d09d8bfd270c355bc5ea63e342884218",
                                       "https://www.dropbox.com/s/lpw1k3so99krmmq/ches_ctf.h5?dl=1"),),
                    converter="ches_ctf_2018",
                ),
            ),
        ),
        DatasetType(
            name="AES_RD",
            # Random delays (Coron and Kizhvatov), one fixed key given in the README only; traces
            # CC BY-NC 4.0 (ikizhvatov/randomdelays-traces). Split as Zaid et al. 2020: first
            # 25,000 traces profiling, last 25,000 attack. The byte: Zaid et al. 2020's target is
            # Z = Sbox[P1 xor k*], "P1 denotes the first byte of the plaintext" (Sect. 3.2, p. 10), as their
            # AES_RD scripts attack it; a byte the study does not define is not choosable (owner 2026-10-23).
            variant="fixed_key",
            required=_ASCAD_V1_REQUIRED,
            optional=_ASCAD_V1_OPTIONAL,
            attack_bytes=frozenset({0}),
            description="AES_RD (ATmega16, random delays): 25k profiling / 25k attack x 3,500",
            conversions=(
                Conversion(
                    output="AES_RD/AES_RD_data/AES_RD.h5",
                    inputs=(SourceFile("AES_RD/AES_RD_data/source/ctraces_fm16x4_2.mat",
                                       "f23271e3a938ec44feda5d9b4d82373cca92af226a7edd78fe3a8cad624d4b65",
                                       "https://media.githubusercontent.com/media/ikizhvatov/randomdelays-traces/master/ctraces_fm16x4_2.mat"),),
                    converter="aes_rd",
                ),
            ),
        ),
        DatasetType(
            name="DPAv4",
            # DPA contest v4 (ATmega163, AES-256 RSM masking) as Zaid et al. 2020 distribute it; the
            # mask is known, the target is the masked S-box output of byte 0. Profiling masks are
            # recovered from Zaid's labels (sca/converters/dpav4.py). Licence: not stated.
            variant="fixed_key",
            required=_ASCAD_V1_REQUIRED,
            optional=_ASCAD_V1_OPTIONAL,
            description="DPA contest v4, known mask (Zaid et al. 2020): 4,500 profiling / 500 attack x 4,000",
            target="sbox_masked",
            attack_bytes=frozenset({0}),
            conversions=(
                Conversion(
                    output="DPAv4/DPAv4_data/DPAv4.h5",
                    inputs=(SourceFile("DPAv4/DPAv4_data/source/DPAv4_dataset.zip",
                                       "c42e0626793848ad38634f1765354fbecd9df3fa606ceb593a94febe6ebeda1f",
                                       _ZAID + "DPA-contest%20v4/DPAv4_dataset.zip"),),
                    converter="dpav4",
                ),
            ),
        ),
        DatasetType(
            name="AES_HD",
            # AES-128 on a Virtex-5 (SASEBO-GII), unprotected, last round. Official traces and labels;
            # c[7] and c[11] from Zaid et al. 2020's copy; key byte 0x00 by their convention
            # (owner ruling 2026-10-16, sca/converters/aes_hd.py). Licence: not stated.
            variant="fixed_key",
            required=_ASCAD_V1_REQUIRED,
            optional=_ASCAD_V1_OPTIONAL,
            description="AES_HD (FPGA, unprotected, last round): 50k profiling / 25k attack x 1,250",
            target="inv_sbox_hd",
            attack_bytes=frozenset({11}),
            conversions=(
                Conversion(
                    output="AES_HD/AES_HD_data/AES_HD.h5",
                    inputs=tuple(
                        SourceFile(f"AES_HD/AES_HD_data/source/{name}", sha, _AESHD + name)
                        for name, sha in (
                            ("traces_1.csv", "0bec17f8ffe4eb4b3d060eb70136e46e6317583905ee9df33fec7dba848f543a"),
                            ("traces_2.csv", "d5704b21e77312a33fa196426944d523dc2e4cc6db46a9e0f5ff489e542d305e"),
                            ("traces_3.csv", "181f700a9c7393018c2f5655944aa2b1ca479bddc58a11d82b09386de56d85cb"),
                            ("traces_4.csv", "10fc903d16150b738d4d15b2e7281a8ac0a10d56ad0ea056bf43155fe82f55ec"),
                            ("traces_5.csv", "2cd6c89b13963bc17e21a30405edbefafb06ccf835883435158af9d04462bd1a"),
                            ("labels.csv", "9e4ce4c7a0fe9eb319341e8f58d667870c6df9757426add5f0a92a66b0e50962"),
                        )
                    ) + (SourceFile("AES_HD/AES_HD_data/source/AES_HD_dataset.zip",
                                    "00a3d02f01bae8c4fcefda33e3d1adb57bed0509ded3cdcf586e213b3d87e41b",
                                    _ZAID + "AES_HD/AES_HD_dataset.zip"),),
                    converter="aes_hd",
                ),
            ),
        ),
    )
}


def _matches(text: str, relative: str) -> bool:
    return text.endswith("data/" + relative) or text == relative


def source_file(path: str | Any) -> Optional[SourceFile]:
    """The registered download of a database path (absolute or relative), matched on its tail."""
    text = str(path).replace("\\", "/")
    for dtype in TYPES.values():
        for source in dtype.sources + tuple(i for c in dtype.conversions for i in c.inputs):
            if _matches(text, source.path):
                return source
    return None


def conversion_for(path: str | Any) -> Optional[Conversion]:
    """The registered conversion whose output a database path is, if any."""
    text = str(path).replace("\\", "/")
    for dtype in TYPES.values():
        for conversion in dtype.conversions:
            if _matches(text, conversion.output):
                return conversion
    return None


def registered(path: str | Any) -> bool:
    """True when the path is a database TraceFlow can fetch or make (utils/dataset_fetch.py)."""
    return source_file(path) is not None or conversion_for(path) is not None


def registered_owner(path: str | Any) -> Optional[tuple[str, Optional[str]]]:
    """(type, None) for a database file a type registers (a download or a conversion's output);
    (type, output) for a download a conversion turns into that output; None for a file no type
    registers (the entry's own traces)."""
    text = str(path).replace("\\", "/")
    for dtype in TYPES.values():
        if any(_matches(text, s.path) for s in dtype.sources):
            return dtype.name, None
        for conversion in dtype.conversions:
            if _matches(text, conversion.output):
                return dtype.name, None
            if any(_matches(text, i.path) for i in conversion.inputs):
                return dtype.name, conversion.output
    return None


def database_problems(entry: Mapping[str, Any], dtype: DatasetType) -> list[str]:
    """A registered file is the database of its own type only: the type fixes the label target, the
    shared and per-trace keys and the attack bytes, so a file of another type would be labelled with
    the wrong intermediate. A conversion's download is not a database: the run reads its output."""
    if not entry.get("database"):
        return []
    owner = registered_owner(entry["database"])
    if owner is None:
        return []
    name = Path(str(entry["database"])).name
    owner_type, output = owner
    if owner_type != dtype.name:
        return [f"database {name} is a registered {owner_type} file, not {dtype.name}: declare type: {owner_type}, "
                f"or point database at a {dtype.name} file ({', '.join(database_files(dtype))})"]
    if output is not None:
        return [f"database {name} is a download {dtype.name} is made from, not its database: point database at "
                f"data/{output}"]
    return []


def database_files(dtype: DatasetType) -> list[str]:
    """The database files a type registers, as data entries name them (under data/)."""
    return [f"data/{s.path}" for s in dtype.sources] + [f"data/{c.output}" for c in dtype.conversions]

LEGACY_TYPE = "ASCADf"
# A database under this folder, in an entry without `type`, is ASCADf (runs made before types).
LEGACY_FOLDER = "data/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/"


def _entry_name(entry: Mapping[str, Any]) -> str:
    return str(entry.get("name") or entry.get("alias") or "<unnamed>")


def resolve_dataset_type(entry: Mapping[str, Any]) -> DatasetType:
    """The type of one data entry: its `type`, or ASCADf for a legacy ASCAD fixed-key entry."""
    if not isinstance(entry, Mapping):
        raise DatasetTypeError(f"a data entry must be a mapping, got {type(entry).__name__}")
    declared = entry.get("type")
    if declared is not None:
        if declared not in TYPES:
            raise DatasetTypeError(
                f"data entry {_entry_name(entry)!r}: unknown type {declared!r}; known types: {sorted(TYPES)}"
            )
        return TYPES[str(declared)]
    database = str(entry.get("database") or "").replace("\\", "/")
    if LEGACY_FOLDER in database or database.startswith(LEGACY_FOLDER.split("/", 1)[1]):
        return TYPES[LEGACY_TYPE]
    raise DatasetTypeError(
        f"data entry {_entry_name(entry)!r} has no type; declare `type:` (one of {sorted(TYPES)}). "
        f"Only a database under {LEGACY_FOLDER} is taken as {LEGACY_TYPE} without one."
    )


@dataclass(frozen=True)
class TraceCounts:
    """The traces of one registered database file, read from the file (2026-10-16)."""
    profiling: int                  # Profiling_traces rows; for a raw file, all rows
    attack: int                     # Attack_traces rows; for a raw file, all rows
    samples: int                    # samples per trace
    raw: bool = False               # one flat 'traces' array: profiling_range and attack_range index it


# Files whose counts are not listed (ASCADr's desync and raw files) are checked when the run reads them.
TRACE_COUNTS: dict[str, TraceCounts] = {
    _ASCADF + "ASCAD.h5": TraceCounts(50_000, 10_000, 700),
    _ASCADF + "ASCAD_desync50.h5": TraceCounts(50_000, 10_000, 700),
    _ASCADF + "ASCAD_desync100.h5": TraceCounts(50_000, 10_000, 700),
    _ASCADF + "ATMega8515_raw_traces.h5": TraceCounts(60_000, 60_000, 100_000, raw=True),
    _ASCADR + "ASCAD.h5": TraceCounts(200_000, 100_000, 1_400),
    "CHES_CTF_2018/CHES_CTF_2018_data/CHES_CTF_2018.h5": TraceCounts(45_000, 5_000, 2_200),
    "AES_RD/AES_RD_data/AES_RD.h5": TraceCounts(25_000, 25_000, 3_500),
    "DPAv4/DPAv4_data/DPAv4.h5": TraceCounts(4_500, 500, 4_000),
    "AES_HD/AES_HD_data/AES_HD.h5": TraceCounts(50_000, 25_000, 1_250),
}


def trace_counts(path: str | Any) -> Optional[TraceCounts]:
    text = str(path).replace("\\", "/")
    for relative, counts in TRACE_COUNTS.items():
        if _matches(text, relative):
            return counts
    return None


def _pairs(value: Any) -> list:
    """A range or a sweep list of ranges, as a list of [start, end] pairs (malformed ones skipped)."""
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return [list(value)]
    if isinstance(value, (list, tuple)):
        return [list(v) for v in value if isinstance(v, (list, tuple)) and len(v) == 2
                and all(isinstance(x, int) for x in v)]
    return []


def _largest(value: Any) -> int:
    values = value if isinstance(value, (list, tuple)) else [value]
    return max([int(v) for v in values if isinstance(v, int)] or [0])


def range_problems(entry: Mapping[str, Any]) -> list[str]:
    """The entry's ranges that do not fit its registered database file (TRACE_COUNTS)."""
    counts = trace_counts(entry.get("database", ""))
    if counts is None:
        return []
    name = Path(str(entry["database"])).name
    groups = ("rows", "rows") if counts.raw else ("profiling traces", "attack traces")
    problems = []
    for key, limit, what in (("profiling_range", counts.profiling, groups[0]),
                             ("attack_range", counts.attack, groups[1])):
        for start, end in _pairs(entry.get(key)):
            if not 0 <= start < end <= limit:
                problems.append(f"{key} [{start}, {end}] is outside the {limit:,} {what} of {name} "
                                f"(use 0 <= start < end <= {limit})")
    attack_sizes = [end - start for start, end in _pairs(entry.get("attack_range"))]
    for key in ("validation_range", "test_range"):
        for start, end in _pairs(entry.get(key)):
            for size in attack_sizes:
                if not 0 <= start < end <= size:
                    problems.append(f"{key} [{start}, {end}] is outside the {size:,} traces attack_range "
                                    f"selects (it indexes them: use 0 <= start < end <= {size})")
    shift = max(_largest(entry.get("profiling_desync", 0)), _largest(entry.get("attack_desync", 0)))
    for start, end in _pairs(entry.get("target_range")):
        if not 0 <= start < end or end + shift > counts.samples:
            extra = f" plus the desync shift of up to {shift}" if shift else ""
            problems.append(f"target_range [{start}, {end}]{extra} is outside the {counts.samples:,} samples "
                            f"per trace of {name}")
    return problems


# data[].profiling_split: `enabled` (true | false), and with enabled: true the arguments of scikit-learn's
# train_test_split, every one written (none is assumed)
PROFILING_SPLIT_KEYS = ("test_size", "shuffle", "random_state")
# What the panel writes when enabled is switched to true (still written, so nothing is assumed at run time; owner
# 2026-10-24): a 90/10 split, scikit-learn's default shuffle, and random_state 0 (its default, None, draws a different
# split every call, which a reproducible run cannot take).
PROFILING_SPLIT_DEFAULTS = {"test_size": 0.1, "shuffle": True, "random_state": 0}


def profiling_split_enabled(entry: Mapping[str, Any]) -> bool:
    """Whether the entry trains on a profiling_split part of profiling_range (enabled: true), not all of it."""
    split = entry.get("profiling_split") if isinstance(entry, Mapping) else None
    return isinstance(split, Mapping) and split.get("enabled") is True


def profiling_split_reproducible(entry: Mapping[str, Any]) -> bool:
    """Whether the rows the entry trains on come out the same on every run: no split, a split without shuffling, or a
    shuffle with a random_state. random_state null (absent) shuffles unseeded, a new cut every run, as Zaid et al.'s
    scripts (random.shuffle without a seed)."""
    if not profiling_split_enabled(entry):
        return True
    split = entry["profiling_split"]
    states = split.get("random_state") if isinstance(split.get("random_state"), list) else [split.get("random_state")]
    return split.get("shuffle") is False or all(isinstance(s, int) and not isinstance(s, bool) for s in states)


def profiling_split_problems(entry: Mapping[str, Any]) -> list[str]:
    """What is wrong with the entry's profiling_split (absent: nothing). `enabled` is always written; with
    enabled: false the other keys are not read. test_size as scikit-learn reads it: a fraction in (0, 1) of
    profiling_range, rounded up, or a row count; the rest of the range is trained on."""
    split = entry.get("profiling_split")
    if split is None:
        return []
    if not isinstance(split, Mapping):
        return [f"profiling_split must be a mapping (enabled, then {list(PROFILING_SPLIT_KEYS)}), not {split!r}"]
    unknown = sorted(set(split) - {"enabled", *PROFILING_SPLIT_KEYS})
    if unknown:
        return [f"profiling_split does not take {unknown}"]
    if not isinstance(split.get("enabled"), bool):
        return [f"profiling_split.enabled must be true or false, not {split.get('enabled')!r}"]
    if not split["enabled"]:
        return []
    problems = []
    missing = [key for key in PROFILING_SPLIT_KEYS if key != "random_state" and key not in split]
    if missing:
        problems.append(f"profiling_split (enabled) is missing {missing} (train_test_split's arguments, each written)")
    test_size = split.get("test_size")
    if "test_size" in split:
        if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
            problems.append(f"profiling_split.test_size must be a fraction in (0, 1) or a row count, not {test_size!r}")
        elif isinstance(test_size, float) and not 0 < test_size < 1:
            problems.append(f"profiling_split.test_size {test_size} is not a fraction in (0, 1)")
        elif isinstance(test_size, int) and test_size < 1:
            problems.append(f"profiling_split.test_size {test_size} holds out no row")
        else:
            for start, end in _pairs(entry.get("profiling_range")):
                rows = end - start
                held_out = math.ceil(test_size * rows) if isinstance(test_size, float) else test_size
                if held_out >= rows:
                    problems.append(f"profiling_split.test_size {test_size} holds out {held_out} of the {rows} rows "
                                    f"of profiling_range [{start}, {end}]: none is left to train on")
    if "shuffle" in split and not isinstance(split["shuffle"], bool):
        problems.append(f"profiling_split.shuffle must be true or false, not {split['shuffle']!r}")
    random_state = split.get("random_state")
    # a list: one run per split (utils/sweep_engine: the one data list that sweeps), e.g. Wouters et al.'s ten
    states = random_state if isinstance(random_state, list) else [random_state]
    if isinstance(random_state, list) and (not random_state or len(set(map(repr, random_state))) != len(random_state)):
        problems.append(f"profiling_split.random_state as a list sweeps one run per split: give distinct values, not {random_state!r}")
    for state in states:
        if state is not None and (isinstance(state, bool) or not isinstance(state, int) or state < 0):
            problems.append(f"profiling_split.random_state must be a non-negative integer, null (unseeded) or a list of "
                            f"them (one run each), not {random_state!r}")
            break
    return problems


def validate_entry(entry: Mapping[str, Any]) -> DatasetType:
    """Check one data entry against its type: unknown keys and missing required keys fail, a
    registered database file of another type (or a conversion's download), and ranges that do not
    fit its registered database file."""
    dtype = resolve_dataset_type(entry)
    unknown = sorted(set(entry) - dtype.allowed)
    missing = sorted(dtype.required - set(entry))
    problems = []
    if unknown:
        problems.append(f"keys a {dtype.name} entry does not take: {unknown}")
    if missing:
        problems.append(f"required keys missing: {missing}")
    if problems:
        raise DatasetTypeError(
            f"data entry {_entry_name(entry)!r} ({dtype.name}): " + "; ".join(problems)
            + f". A {dtype.name} entry takes: {sorted(dtype.allowed)}"
        )
    ranges = database_problems(entry, dtype) + range_problems(entry) + profiling_split_problems(entry)
    if ranges:
        raise DatasetTypeError(f"data entry {_entry_name(entry)!r} ({dtype.name}): " + "; ".join(ranges))
    return dtype


def _active_entry(cfg: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The run's data entry: cfg['data'] itself once the sweep selected it, else the active one."""
    data = cfg.get("data") if isinstance(cfg, Mapping) else None
    if isinstance(data, Mapping) and ("database" in data or "type" in data):
        return data
    if data is None:
        return None
    from sca.ascad_data_loader import resolve_active_data_config

    return resolve_active_data_config(cfg)


def dataset_variant(cfg: Mapping[str, Any]) -> str:
    """sca.variant as the run's database type gives it; a configured sca.variant must agree."""
    entry = _active_entry(cfg)
    if entry is None:
        raise DatasetTypeError("cfg['data'] is required to know the database type")
    dtype = resolve_dataset_type(entry)
    configured = ((cfg.get("sca") or {}) if isinstance(cfg, Mapping) else {}).get("variant")
    if configured is not None and str(configured).strip().lower() != dtype.variant:
        raise DatasetTypeError(
            f"sca.variant={configured!r} disagrees with data entry {_entry_name(entry)!r} of type "
            f"{dtype.name} ({dtype.variant}); sca.variant is given by the type: remove it"
        )
    return dtype.variant


def _active_type(cfg: Mapping[str, Any]) -> Optional[DatasetType]:
    """The run's database type. None only when the configuration holds no data (none at all, or a stub entry
    naming neither a database nor a type: a hand-built configuration of a unit or a tool). A data entry that
    names a database (every entry a run reads: config_loader requires one) and whose type cannot be resolved stops here
    (DatasetTypeError, naming why): the readers below shape the labels, the key and the attack (label
    target, shared key, random profiling keys, sca.variant), and each used to fall back to ASCADf's on
    any error, so a DPAv4 or AES_HD run would have been ranked against Sbox[p ^ k] without a word
    (audit 2026-10-25)."""
    if not isinstance(cfg, Mapping) or cfg.get("data") is None:
        return None
    try:
        entry = _active_entry(cfg)
    except (KeyError, ValueError, TypeError) as exc:
        raise DatasetTypeError(f"the run's data entry cannot be selected ({exc}): its database type, and with it the "
                               "label target and the key mode, would be guessed") from exc
    if entry is None or ("type" not in entry and "database" not in entry):
        return None                     # a stub naming no database and no type (a unit's configuration): no data to read
    return resolve_dataset_type(entry)


def resolved_variant(cfg: Mapping[str, Any]) -> Optional[str]:
    """sca.variant for the readers (data loader, splits, attack, LDD, metrics): the run's database
    type gives it, and a configured sca.variant must agree (dataset_variant). A configuration
    without data (a hand-built configuration of a unit or a tool) keeps its configured sca.variant;
    a data entry whose type cannot be resolved stops (_active_type)."""
    configured = ((cfg.get("sca") or {}) if isinstance(cfg, Mapping) else {}).get("variant")
    if _active_type(cfg) is None:
        return configured
    return dataset_variant(cfg)


def random_profiling_keys(cfg: Mapping[str, Any]) -> bool:
    """Whether the run's database has a random key per profiling trace (ASCADr). False without data;
    a data entry whose type cannot be resolved stops (_active_type)."""
    dtype = _active_type(cfg)
    return dtype.random_profiling_keys if dtype is not None else False


def label_target(cfg: Mapping[str, Any]) -> str:
    """The run's label target (sca/leakage_models.py TARGETS). sbox only without data (a hand-built
    configuration); a data entry whose type cannot be resolved stops (_active_type), never ranked against
    a guessed intermediate."""
    dtype = _active_type(cfg)
    return dtype.target if dtype is not None else "sbox"


# What each label target computes, for the messages that refuse a setting it does not support.
TARGET_TEXT = {
    "sbox": "Sbox[p[b] ^ k[b]], any byte b",
    "sbox_masked": "Sbox[p[0] ^ k[0]] ^ mask: the stored mask is byte 0's",
    "inv_sbox_hd": "InvSbox[c[11] ^ k[11]] ^ c[7]: the last-round register of byte 11",
}


def attack_bytes_allowed(entry: Mapping[str, Any]) -> list[int]:
    """The sca.attack_byte values require_attack_byte accepts for this data entry: the bytes its type
    defines a target for, within the trace window its database file was cut for (all 16 when neither
    restricts). The panel offers the union over a project's entries."""
    try:
        dtype = resolve_dataset_type(entry)
    except DatasetTypeError:
        return list(range(16))
    allowed = set(dtype.attack_bytes) if dtype.attack_bytes is not None else set(range(16))
    source = source_file(entry.get("database", "")) if isinstance(entry, Mapping) else None
    if source is not None and source.window_bytes is not None:
        allowed &= set(source.window_bytes)
    # a window the entry declares verified for some bytes (trace_window_verified_by_attack_byte, the raw traces cut
    # to a target_range): those are its bytes; the loader refuses the others on such a window anyway
    verified = entry.get("trace_window_verified_by_attack_byte") if isinstance(entry, Mapping) else None
    if isinstance(verified, Mapping):
        declared = set()
        for key, value in verified.items():
            try:
                byte = int(str(key), 16) if str(key).lower().startswith("0x") else int(key)
            except (TypeError, ValueError):
                continue
            if value is True or str(value).strip().lower() == "true":
                declared.add(byte)
        if declared:
            allowed &= declared
    return sorted(allowed)


def derived_attack_byte(entry: Mapping[str, Any]) -> Optional[int]:
    """The one byte a data entry defines a target for (attack_bytes_allowed), or None when it allows several."""
    allowed = attack_bytes_allowed(entry)
    return allowed[0] if len(allowed) == 1 else None


def project_attack_byte(cfg: Mapping[str, Any]) -> Optional[int]:
    """The one byte every data entry of a project defines (derived_attack_byte), or None when they differ, one
    allows several, or there is none: for a reader that compiles the file before any entry is selected
    (AutoSearch's search space)."""
    data = cfg.get("data") if isinstance(cfg, Mapping) else None
    entries = (list(data.values()) if isinstance(data, Mapping) and "database" not in data
               else data if isinstance(data, list) else [data] if isinstance(data, Mapping) else [])
    bytes_ = {derived_attack_byte(e) for e in entries if isinstance(e, Mapping)}
    return bytes_.pop() if len(bytes_) == 1 and None not in bytes_ else None


def resolve_attack_byte(cfg: Any) -> None:
    """sca.attack_byte left empty follows the run's data entry (owner 2026-10-23: the source of the byte is the
    choice of the dataset): the entry's type, within its file's window, defines one byte (ASCADf 2, DPAv4 0,
    AES_HD 11, AES_RD 0) and the run takes it, written into cfg before the sweep and the hash, as if set. An
    entry that allows several bytes (CHES_CTF_2018, the raw ASCAD traces) needs it set. A value set is kept and
    checked (require_attack_byte)."""
    sca = cfg.get("sca") if isinstance(cfg, Mapping) else None
    raw = sca.get("attack_byte") if isinstance(sca, Mapping) else None
    if raw not in (None, [], ()):
        return
    entry = _active_entry(cfg)
    if entry is None:
        return
    byte = derived_attack_byte(entry)
    if byte is None:
        allowed = attack_bytes_allowed(entry)
        raise DatasetTypeError(
            f"sca.attack_byte is empty and data entry {_entry_name(entry)!r} (type {_active_type(cfg).name}) defines "
            f"a target for bytes {allowed}: set the byte to attack (one, or a list to sweep)"
        )
    if not isinstance(cfg, MutableMapping):
        return
    if not isinstance(cfg.get("sca"), MutableMapping):
        cfg["sca"] = dict(cfg.get("sca") or {})
    cfg["sca"]["attack_byte"] = [byte]


def require_attack_byte(cfg: Mapping[str, Any]) -> None:
    """Refuse a sca.attack_byte the run's database type does not define its target for, or that lies
    outside the trace window its database file was cut for (ASCAD.h5: byte 2)."""
    dtype = _active_type(cfg)
    if dtype is None:
        return
    raw = ((cfg.get("sca") or {}) if isinstance(cfg, Mapping) else {}).get("attack_byte")
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    entry = _active_entry(cfg)
    if dtype.attack_bytes is not None:
        wrong = [v for v in values if v not in dtype.attack_bytes]
        if wrong:
            allowed = sorted(dtype.attack_bytes)
            raise DatasetTypeError(
                f"sca.attack_byte {wrong} is not defined for data entry {_entry_name(entry)!r} "
                f"(type {dtype.name}): its label is {TARGET_TEXT[dtype.target]}. "
                f"Set sca.attack_byte: {allowed}"
            )
    database = entry.get("database", "") if isinstance(entry, Mapping) else ""
    source = source_file(database)
    if source is None or source.window_bytes is None:
        return
    wrong = [v for v in values if v not in source.window_bytes]
    if wrong:
        allowed = sorted(source.window_bytes)
        counts = trace_counts(database)
        samples = f"{counts.samples:,} samples" if counts is not None else "samples"
        raise DatasetTypeError(
            f"sca.attack_byte {wrong} lies outside the trace window of data entry {_entry_name(entry)!r} "
            f"(type {dtype.name}): {Path(source.path).name} holds the {samples} ANSSI cut around the "
            f"first-round S-box of byte {', '.join(map(str, allowed))}, so another byte's leakage is not "
            f"in its traces. Set sca.attack_byte: {allowed}, or attack byte {wrong[0]} on the raw traces "
            f"(ATMega8515_raw_traces.h5) with a target_range verified for it "
            f"(trace_window_verified_by_attack_byte)"
        )


def require_cross_validation_settings(cfg: Mapping[str, Any]) -> None:
    """Refuse cross-validation of an entry with a profiling_split, and cross-validation that rank-attacks folds of
    random profiling keys (ASCADr)."""
    enabled = ((cfg.get("cross_validation") or {}) if isinstance(cfg, Mapping) else {}).get("enabled", False)
    if not (enabled is True or (isinstance(enabled, (list, tuple)) and True in enabled)):
        return
    entry = _active_entry(cfg)
    if entry is not None and profiling_split_enabled(entry):
        # The folds partition the whole profiling_range themselves; a split before them would leave rows unused.
        raise DatasetTypeError(
            f"data entry {_entry_name(entry)!r} has profiling_split.enabled: true, which selects the rows one training uses; "
            "cross-validation partitions profiling_range itself: set profiling_split.enabled: false or "
            "cross_validation.enabled: false"
        )
    if not random_profiling_keys(cfg):
        return
    # A random key per profiling trace: a fold's held-out traces have no single key to rank. The folds
    # report key-independent metrics only (owner ruling 2026-10-16); rank metrics and GE come from the
    # test on the fixed-key attack set.
    attack_times = ((cfg.get("perform_attack") or {}).get("validation") or {}).get("attack_times")
    if attack_times not in (0, [0]):
        dtype = _active_type(cfg)
        raise DatasetTypeError(
            f"cross_validation with data entry {_entry_name(_active_entry(cfg))!r} (type {dtype.name}: "
            "a random key per profiling trace) needs perform_attack.validation.attack_times: 0 "
            f"(it is {attack_times!r}): its folds have no single key to rank, so they report accuracy "
            "and loss only, and rank metrics and GE come from the test on the fixed-key attack set"
        )


def _values(value: Any) -> list:
    """A setting's values: the list a sweep declares, or the one value."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def require_target_settings(cfg: Mapping[str, Any]) -> None:
    """Refuse the leakage models and attack orders the type's ciphertext cannot support (AES_HD).

    AES_HD's c[11] and c[7] are not measured ciphertext but Zaid et al.'s re-encoding of the label
    (sca/converters/aes_hd.py): c[11] = Sbox[row // 256] is constant over each block of 256 rows.
    Inside a block every wrong key's hypothesis is the true label XOR one constant, so MSB leaves
    128 keys tied, and an unshuffled attack's first traces see one or two c[11] values."""
    dtype = _active_type(cfg)
    if dtype is None or dtype.target != "inv_sbox_hd":
        return
    name = _entry_name(_active_entry(cfg))
    models = [str(m).strip().upper() for m in _values((cfg.get("sca") or {}).get("leakage_model"))]
    if "MSB" in models:
        raise DatasetTypeError(
            f"sca.leakage_model MSB cannot rank keys on data entry {name!r} (type {dtype.name}): its "
            "c[11] is Sbox[row // 256], constant over each block of 256 traces, so every wrong key's "
            "MSB hypothesis there is the true one or its complement and 128 keys stay tied. "
            "Set sca.leakage_model: ID or HW"
        )
    attack = cfg.get("perform_attack") or {}
    for process in ("validation", "test"):
        block = attack.get(process) or {}
        attacked = any(times not in (0, None) for times in _values(block.get("attack_times")))
        if attacked and False in _values(block.get("shuffle")):
            raise DatasetTypeError(
                f"perform_attack.{process}.shuffle: false on data entry {name!r} (type {dtype.name}) "
                "accumulates traces in file order, where c[11] = Sbox[row // 256] changes once every "
                "256 traces, so the first traces of the rank curve see one or two c[11] values. "
                f"Set perform_attack.{process}.shuffle: true"
            )


def check_run_settings(cfg: Mapping[str, Any]) -> None:
    """The settings outside the data entry that its database type constrains, checked once the
    command has selected the entry (core/cli.py, core/orchestrator.py), before any run is planned.
    An entry without a resolvable type was already refused when the configuration loaded
    (validate_entry); a hand-built configuration without one is left to its own settings."""
    if _active_type(cfg) is None:
        return
    dataset_variant(cfg)                        # a leftover sca.variant must agree with the type
    resolve_attack_byte(cfg)                   # empty: the entry's byte, written in before the sweep and the hash
    require_attack_byte(cfg)
    require_cross_validation_settings(cfg)
    require_target_settings(cfg)


def shared_key(cfg: Mapping[str, Any]) -> bool:
    """Whether the run's database uses one key for profiling and attack (ASCADf: yes; CHES_CTF_2018:
    a different fixed key in the attack set). True without data; a data entry whose type cannot be
    resolved stops (_active_type)."""
    dtype = _active_type(cfg)
    return dtype.shared_key if dtype is not None else True


def hash_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The data entry as the model hash reads it: `type` left out when it is ASCADf, so a run
    keeps the hash it had before types existed (utils/model_naming.py); `multilabel` read as 0 when
    the entry leaves it out (2026-10-19: no longer a setting, single-label is the only mode the loader
    has), so a run keeps the hash every run made with `multilabel: 0` has. `profiling_split` joins only
    when enabled, with the keys a run reads (as cross_validation in make_experiment_hash): enabled: false
    trains on all of profiling_range, the run every entry without the block made, so it keeps their hash."""
    out = dict(entry)
    if out.get("type") == LEGACY_TYPE:
        out.pop("type")
    out.setdefault("multilabel", 0)
    split = out.pop("profiling_split", None)
    if profiling_split_enabled(entry):
        out["profiling_split"] = {key: split.get(key) for key in ("enabled", *PROFILING_SPLIT_KEYS)}   # random_state null too
    return out


def hash_variant(cfg: Mapping[str, Any]) -> Optional[str]:
    """sca_training_contract.variant of the model hash: the configured sca.variant (old runs and
    snapshots), else the type's key mode, else None (a configuration without a data type)."""
    configured = ((cfg.get("sca") or {}) if isinstance(cfg, Mapping) else {}).get("variant")
    if configured is not None:
        return configured
    try:
        return dataset_variant(cfg)
    except (DatasetTypeError, KeyError, ValueError, TypeError):
        return None


__all__ = [
    "Conversion", "DatasetType", "DatasetTypeError", "SourceFile", "TYPES", "source_file", "conversion_for",
    "attack_bytes_allowed", "registered", "registered_owner", "database_problems", "database_files", "shared_key", "random_profiling_keys", "label_target", "require_attack_byte",
    "require_cross_validation_settings", "check_run_settings", "TARGET_TEXT", "LEGACY_TYPE", "LEGACY_FOLDER",
    "derived_attack_byte", "project_attack_byte", "resolve_attack_byte", "resolve_dataset_type", "validate_entry", "PROFILING_SPLIT_KEYS", "PROFILING_SPLIT_DEFAULTS", "profiling_split_enabled", "profiling_split_reproducible", "profiling_split_problems", "dataset_variant", "resolved_variant", "hash_entry", "hash_variant",
]
