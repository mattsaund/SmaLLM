#!/usr/bin/env python3
"""Generates src/tensor/iq_tables.rs from ggml's lookup tables.

The IQ weight formats store groups of four or eight weights as an index
into a grid of allowed patterns. ggml keeps those grids in
ggml/src/ggml-common.h, one 64-bit or 32-bit word per pattern, a byte per
weight.

The grids of eight weights are 30 KB as ggml stores them. Every byte in
them is one of three levels, though, so this writes each pattern as a
16-bit code instead: 2 bits per weight that select a level. Those grids
shrink to 7.5 KB. The decoders in src/tensor/quant.rs expand a code four
weights at a time, through a table they build from the levels.

The grids of four weights stay as ggml stores them. They are 3 KB, and a
packed form would save only half of that.

The script also checks the facts the decoders rely on instead of tables:
the sign bytes of the IQ2 and IQ3 formats and the bit masks that test
them. It fails if a newer ggml changes them.

For each grid it writes a hash of ggml's original table. A test expands
every entry the way the decoders do and checks that it gets the same
bytes back.

Usage:
    python3 tools/gen_iq_tables.py path/to/llama.cpp > src/tensor/iq_tables.rs
"""

import re
import sys
from pathlib import Path

# The grids: the name here, the name ggml gives it, the weights in one
# entry, and what it is for.
GRIDS = [
    ("IQ2XXS", "iq2xxs_grid", 8, "The IQ2_XXS grid: 256 patterns of eight weights."),
    ("IQ2XS", "iq2xs_grid", 8, "The IQ2_XS grid: 512 patterns of eight weights."),
    ("IQ2S", "iq2s_grid", 8, "The IQ2_S grid: 1024 patterns of eight weights."),
    ("IQ3XXS", "iq3xxs_grid", 4, "The IQ3_XXS grid: 256 patterns of four weights."),
    ("IQ3S", "iq3s_grid", 4, "The IQ3_S grid: 512 patterns of four weights."),
    ("IQ1S", "iq1s_grid", 8, "The IQ1_S and IQ1_M grid: 2048 patterns of eight weights."),
]

# The levels of the packed grids. Grids that share levels share a list.
LEVELS = {
    "IQ2XXS": "IQ2",
    "IQ2XS": "IQ2",
    "IQ2S": "IQ2",
    "IQ1S": "IQ1S",
}

# What each list of levels is for, for its comment.
LEVEL_NOTES = {
    "IQ2": "The magnitudes in the IQ2_XXS, IQ2_XS and IQ2_S grids.",
    "IQ1S": "The values in the IQ1_S and IQ1_M grid.",
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


def pack(name, entries):
    """Packs a grid of eight weights into 16-bit codes, 2 bits per weight.
    Returns the levels, padded to four, and the codes."""
    values = [entry_values(entry, 8) for entry in entries]
    levels = sorted({value for row in values for value in row})
    if len(levels) > 4:
        sys.exit(f"{name}: {len(levels)} levels do not fit in 2 bits")
    codes = []
    for row in values:
        code = 0
        for j, value in enumerate(row):
            code |= levels.index(value) << (2 * j)
        codes.append(code)
    # Unpack every code again, as the decoders do, before trusting it.
    for code, row in zip(codes, values):
        if [levels[(code >> (2 * j)) & 3] for j in range(8)] != row:
            sys.exit(f"{name}: a code does not unpack to its entry")
    return levels + [0] * (4 - len(levels)), codes


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
    print("//! ggml stores a grid entry as one byte per weight. The grids of eight")
    print("//! weights are packed here: an entry is a 16-bit code, 2 bits per weight,")
    print("//! first weight lowest, each selecting one of the grid's levels. Entry")
    print("//! `k` of a grid here unpacks to the bytes of entry `k` in ggml's grid.")
    print("//! The grids of four weights keep ggml's form.")
    print()

    written_levels = set()
    hashes = []
    for name, ggml_name, width, note in GRIDS:
        entries = table(source, ggml_name)
        if width == 8:
            levels, codes = pack(name, entries)
            group = LEVELS[name]
            if group not in written_levels:
                written_levels.add(group)
                print(f"/// {LEVEL_NOTES[group]} 2 bits select one.")
                items = ", ".join(str(level) for level in levels)
                print(f"pub(crate) const {group}_LEVELS: [i8; 4] = [{items}];")
                print()
            print(f"/// {note}")
            print(f"/// Each code holds 8 indices into [`{group}_LEVELS`], 2 bits each.")
            print("#[rustfmt::skip]")
            print(f"pub(crate) static {name}_GRID: [u16; {len(codes)}] = [")
            print(rust_list(codes, 12, lambda code: f"0x{code:04x}"))
        else:
            print(f"/// {note}")
            print("/// Each word holds the four magnitudes, first weight lowest.")
            print("#[rustfmt::skip]")
            print(f"pub(crate) static {name}_GRID: [u32; {len(entries)}] = [")
            print(rust_list(entries, 8, lambda entry: f"0x{entry:08x}"))
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
    print("/// entry, its bytes in memory order. A test checks the grids against it.")
    print("#[cfg(test)]")
    print(f"pub(crate) const GRID_HASHES: [(&str, u64); {len(hashes)}] = [")
    for name, hash_value in hashes:
        print(f'    ("{name}", 0x{hash_value:016x}),')
    print("];")


if __name__ == "__main__":
    main()
