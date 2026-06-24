#!/usr/bin/env python3
"""Remove corrupt per-glyph gvar entries from a variable font.

When one glyph's gvar tuple data is malformed, fontTools fails to dump or
recompile the whole table. This script finds those glyphs, clears only their
variation records, and saves a repaired copy. All other gvar data is kept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fontTools.ttLib import TTFont


def find_corrupt_gvar_glyphs(font: TTFont) -> list[str]:
    gvar = font["gvar"]
    bad: list[str] = []
    for name in font.getGlyphOrder():
        try:
            gvar.variations.get(name)
        except Exception:
            bad.append(name)
    return bad


def repair_gvar(font: TTFont, glyph_names: list[str] | None = None) -> list[str]:
    if "gvar" not in font:
        raise ValueError("Font has no gvar table.")

    gvar = font["gvar"]
    if glyph_names is None:
        glyph_names = find_corrupt_gvar_glyphs(font)

    for name in font.getGlyphOrder():
        if name in glyph_names:
            continue
        gvar.variations.get(name)

    for name in glyph_names:
        gvar.variations[name] = []

    return glyph_names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clear corrupt gvar entries without removing the gvar table."
    )
    parser.add_argument("input", type=Path, help="Input variable .ttf/.otf")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: <stem>-gvar-fixed<suffix>)",
    )
    parser.add_argument(
        "--glyph",
        action="append",
        dest="glyphs",
        metavar="NAME",
        help="Clear gvar for a specific glyph (repeatable). Default: auto-detect corrupt entries.",
    )
    args = parser.parse_args(argv)

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        print(f"Error: not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}-gvar-fixed{input_path.suffix}")
    else:
        output_path = output_path.expanduser().resolve()

    font = TTFont(input_path, lazy=True)
    targets = args.glyphs
    if targets is None:
        targets = find_corrupt_gvar_glyphs(font)
        if not targets:
            print("No corrupt gvar entries found.")
            return 0
        print(f"Found {len(targets)} corrupt gvar entr{'y' if len(targets) == 1 else 'ies'}:")
        for name in targets:
            print(f"  - {name}")
    else:
        print(f"Clearing gvar for {len(targets)} glyph(s):")
        for name in targets:
            print(f"  - {name}")

    repair_gvar(font, targets)
    font.save(output_path)
    print(f"Saved: {output_path}")

    verify = TTFont(output_path, lazy=True)
    remaining = find_corrupt_gvar_glyphs(verify)
    if remaining:
        print("Warning: corrupt entries remain:", ", ".join(remaining), file=sys.stderr)
        return 1

    print("Verified: all gvar entries decode cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
