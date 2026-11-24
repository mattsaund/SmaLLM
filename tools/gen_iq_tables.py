#!/usr/bin/env python3
"""Generates src/tensor/iq_tables.rs from ggml's lookup tables.

The IQ weight formats store groups of four or eight weights as an index
into a grid of allowed patterns. ggml keeps those grids in
ggml/src/ggml-common.h, one 64-bit or 32-bit word per pattern, a byte per
weight. That is 33 KB of tables. Every byte in a grid is one of a few
levels, though: three for the IQ2 and IQ1 grids, eight for the IQ3 grids.
So this writes each pattern as a 16-bit code instead, with 2 or 3 bits per
weight that select a level. The tables shrink to 9 KB.

The script checks the facts the decoders in src/tensor/quant.rs rely on
instead of tables: the sign bytes of the IQ2 and IQ3 formats and the bit
masks that test them. It fails if a newer ggml changes them.

For each grid it also writes a hash of ggml's original table. A test
unpacks every code the way the decoders do and checks that it gets the
same bytes back.

Usage:
    python3 tools/gen_iq_tables.py path/to/llama.cpp > src/tensor/iq_tables.rs
"""

import re
import sys
from pathlib import Path

# The grids, with the name ggml gives each one and the bytes in one entry.
GRIDS = [
    ("IQ2XXS", "iq2xxs_grid", 8),
    ("IQ2XS", "iq2xs_grid", 8),
    ("IQ2S", "iq2s_grid", 8),
    ("IQ3XXS", "iq3xxs_grid", 4),
    ("IQ3S", "iq3s_grid", 4),
    ("IQ1S", "iq1s_grid", 8),
]

# Which grids share one list of levels, and the name of that list.
LEVELS = {
    "IQ2XXS": "IQ2",
    "IQ2XS": "IQ2",
    "IQ2S": "IQ2",
    "IQ3XXS": "IQ3XXS",
    "IQ3S": "IQ3S",
    "IQ1S": "IQ1S",
}

# What each list of levels is for, for its comment.
LEVEL_NOTES = {
    "IQ2": "The magnitudes in the IQ2_XXS, IQ2_XS and IQ2_S grids.",
    "IQ3XXS": "The magnitudes in the IQ3_XXS grid.",
    "IQ3S": "The magnitudes in the IQ3_S grid.",
    "IQ1S": "The values in the IQ1_S and IQ1_M grid.",
}

# What each grid is for, for its comment.
GRID_NOTES = {
    "IQ2XXS": "The IQ2_XXS grid: 256 patterns of eight weights.",
    "IQ2XS": "The IQ2_XS grid: 512 patterns of eight weights.",
    "IQ2S": "The IQ2_S grid: 1024 patterns of eight weights.",
    "IQ3XXS": "The IQ3_XXS grid: 256 patterns of four weights.",
    "IQ3S": "The IQ3_S grid: 512 patterns of four weights.",
    "IQ1S": "The IQ1_S and IQ1_M grid: 2048 patterns of eight weights.",
}


def table(source, *names):
    """Returns the numbers in the first of ggml's tables `names` that exists,
    in order. Some tables were renamed, so a caller may give both names."""
    for name in names:
        pattern = r"GGML_TABLE_BEGIN\(\w+,\s*" + name + r",\s*\w+\)(.*?)GGML_TABLE_END\(\)"
        match = re.search(pattern, source, re.S)
        if match:
            numbers = re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", match.group(1))
            return [int(value, 0) for value in numbers]
    sys.exit(f"table {' or '.join(names)} is not in ggml-common.h")


def signed_byte(value):
    """Reads a byte as a two's complement number."""
    return value - 256 if value >= 128 else value


def entry_values(entry, width):
    """The weights of one grid entry: its bytes, first byte first."""
    return [signed_byte((entry >> (8 * j)) & 0xFF) for j in range(width)]


def fnv1a(data):
    """The 64-bit FNV-1a hash of a sequence of bytes."""
    hash_value = 0xCBF29CE484222325
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def check_signs(source):
    """Checks that ggml's sign tables follow the rules the decoders use."""
    signs = table(source, "ksigns_iq2xs")
    for index, value in enumerate(signs):
        parity = bin(index).count("1") & 1
        if value != index | (parity << 7):
            sys.exit(f"ksigns_iq2xs[{index}] is {value}, not the parity rule")
    if table(source, "kmask_iq2xs") != [1 << j for j in range(8)]:
        sys.exit("kmask_iq2xs is no longer one bit per weight")


def pack(name, entries, width):
    """Packs a grid into codes. Returns the levels, bits per weight and codes."""
    values = [entry_values(entry, width) for entry in entries]
    levels = sorted({value for row in values for value in row})
    bits = 2 if len(levels) <= 4 else 3
    if len(levels) > 8 or width * bits > 16:
        sys.exit(f"{name}: {len(levels)} levels do not fit a 16-bit code")
    codes = []
    for row in values:
        code = 0
        for j, value in enumerate(row):
            code |= levels.index(value) << (bits * j)
        codes.append(code)
    # Unpack every code again, as the decoders do, before trusting it.
    for code, row in zip(codes, values):
        unpacked = [levels[(code >> (bits * j)) & ((1 << bits) - 1)] for j in range(width)]
        if unpacked != row:
            sys.exit(f"{name}: a code does not unpack to its entry")
    padded = levels + [0] * ((1 << bits) - len(levels))
    return padded, bits, codes


def rust_list(numbers, per_line, form):
    """Formats numbers as the lines of a Rust array."""
    items = [form(number) for number in numbers]
    lines = []
    for i in range(0, len(items), per_line):
        lines.append("    " + ", ".join(items[i : i + per_line]) + ",")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    header = Path(sys.argv[1]) / "ggml" / "src" / "ggml-common.h"
    source = header.read_text()
    check_signs(source)

    print("//! The lookup grids of the IQ weight formats, and the FP4 levels.")
    print("//!")
    print("//! Generated by tools/gen_iq_tables.py from ggml/src/ggml-common.h in")
    print("//! llama.cpp. Do not edit by hand.")
    print("//!")
    print("//! ggml stores a grid entry as one byte per weight. Here an entry is a")
    print("//! 16-bit code instead: 2 or 3 bits per weight, first weight lowest,")
    print("//! each selecting one of the grid's levels. Entry `k` of a grid here")
    print("//! unpacks to the bytes of entry `k` in ggml's grid.")
    print()

    written_levels = set()
    hashes = []
    for name, ggml_name, width in GRIDS:
        entries = table(source, ggml_name)
        levels, bits, codes = pack(name, entries, width)
        group = LEVELS[name]
        if group not in written_levels:
            written_levels.add(group)
            print(f"/// {LEVEL_NOTES[group]} {bits} bits select one.")
            items = ", ".join(str(level) for level in levels)
            print(f"pub(crate) const {group}_LEVELS: [i8; {len(levels)}] = [{items}];")
            print()
        print(f"/// {GRID_NOTES[name]}")
        print(f"/// Each code holds {width} indices into [`{group}_LEVELS`], {bits} bits each.")
        print("#[rustfmt::skip]")
        print(f"pub(crate) static {name}_GRID: [u16; {len(codes)}] = [")
        print(rust_list(codes, 12, lambda code: f"0x{code:04x}"))
        print("];")
        print()
        original = [byte for entry in entries for byte in entry.to_bytes(width, "little")]
        hashes.append((name, fnv1a(original)))

    # Newer ggml shares this table with NVFP4 under the name kvalues_fp4.
    fp4 = table(source, "kvalues_mxfp4", "kvalues_fp4")
    print("/// The values of the 4-bit floats in MXFP4 blocks, doubled so that each")
    print("/// is an integer. ggml calls this table `kvalues_mxfp4`.")
    print("#[rustfmt::skip]")
    print(f"pub(crate) const MXFP4_LEVELS: [i8; {len(fp4)}] = [")
    print(rust_list(fp4, 16, str))
    print("];")
    print()

    print("/// The 64-bit FNV-1a hash of each grid as ggml stores it, entry after")
    print("/// entry, its bytes in memory order. A test checks the codes against it.")
    print("#[cfg(test)]")
    print(f"pub(crate) const GRID_HASHES: [(&str, u64); {len(hashes)}] = [")
    for name, hash_value in hashes:
        print(f'    ("{name}", 0x{hash_value:016x}),')
    print("];")


if __name__ == "__main__":
    main()
