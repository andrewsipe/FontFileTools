#!/usr/bin/env python3
"""Repair fonts that fail ttx/fontTools due to oversized glyf instructions.

TrueType stores per-glyph instructionLength as a 16-bit field. fontTools packs
and unpacks it as a signed short (``>h``), so any glyph with more than 32767
bytes of hint bytecode crashes on compile — including during WOFF2 glyf
reconstruction. Some webfonts ship pathological instruction blobs (tens of KB
per glyph) that trip this.

This script:
  1. Opens the font (WOFF2 reconstruct allowed to drop illegal-sized programs)
  2. Strips TrueType hinting from every glyph
  3. Removes ``fpgm`` / ``prep`` / ``cvt `` tables
  4. Clears maxp hinting limits
  5. Saves a plain TTF that ttx can dump
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import Glyph

# fontTools uses signed int16 for instructionLength; OT allows uint16, but
# values above this cannot round-trip through fontTools.
_MAX_SAFE_INSTRUCTIONS = 32767


def _patch_woff2_reconstruct(strip_all_hints: bool):
    """Allow WOFF2 glyf rebuild when instruction streams exceed signed int16."""
    orig_compile = Glyph.compileCoordinates
    orig_pack = struct.pack

    def compile_coordinates(self, *, optimizeSize=True):
        if hasattr(self, "program") and self.program is not None:
            try:
                bytecode = self.program.getBytecode()
            except Exception:
                bytecode = b""
            if strip_all_hints or len(bytecode) > _MAX_SAFE_INSTRUCTIONS:
                self.program = ttProgram.Program()
                self.program.fromBytecode(b"")
        return orig_compile(self, optimizeSize=optimizeSize)

    def pack(fmt, *args):
        # Spec is uint16; keep a safety net if anything still overflows >h.
        if (
            fmt == ">h"
            and len(args) == 1
            and isinstance(args[0], int)
            and not (-32768 <= args[0] <= 32767)
            and 0 <= args[0] <= 65535
        ):
            return orig_pack(">H", args[0])
        return orig_pack(fmt, *args)

    Glyph.compileCoordinates = compile_coordinates
    struct.pack = pack
    return orig_compile, orig_pack


def _restore_patches(orig_compile, orig_pack) -> None:
    Glyph.compileCoordinates = orig_compile
    struct.pack = orig_pack


def _clear_glyph_program(glyph: Glyph) -> None:
    if hasattr(glyph, "program"):
        glyph.program = ttProgram.Program()
        glyph.program.fromBytecode(b"")


def _empty_glyph() -> Glyph:
    glyph = Glyph()
    glyph.numberOfContours = 0
    glyph.xMin = glyph.yMin = glyph.xMax = glyph.yMax = 0
    return glyph


def strip_truetype_hints(font: TTFont) -> dict[str, int]:
    """Remove per-glyph programs and global hinting tables. Returns stats."""
    stats = {
        "glyphs_cleared": 0,
        "glyphs_replaced_empty": 0,
        "tables_removed": 0,
    }

    if "glyf" not in font:
        raise ValueError("Font has no glyf table (not a TrueType outline font).")

    glyf = font["glyf"]
    for name in list(glyf.keys()):
        try:
            glyph = glyf[name]
            _clear_glyph_program(glyph)
            if hasattr(glyph, "removeHinting"):
                try:
                    glyph.removeHinting()
                except Exception:
                    pass
            stats["glyphs_cleared"] += 1
        except Exception:
            glyf[name] = _empty_glyph()
            stats["glyphs_replaced_empty"] += 1

    for tag in ("fpgm", "prep", "cvt "):
        if tag in font:
            del font[tag]
            stats["tables_removed"] += 1

    maxp = font["maxp"]
    for attr, value in (
        ("maxZones", 1),
        ("maxTwilightPoints", 0),
        ("maxStorage", 0),
        ("maxFunctionDefs", 0),
        ("maxInstructionDefs", 0),
        ("maxStackElements", 0),
        ("maxSizeOfInstructions", 0),
    ):
        if hasattr(maxp, attr):
            setattr(maxp, attr, value)

    return stats


def repair_font(input_path: Path, output_path: Path, *, strip_all_hints: bool = True) -> dict[str, int]:
    orig_compile, orig_pack = _patch_woff2_reconstruct(strip_all_hints=strip_all_hints)
    try:
        font = TTFont(str(input_path), recalcBBoxes=False, recalcTimestamp=False)
        # Force table loads (triggers WOFF2 glyf/loca reconstruct).
        for tag in list(font.keys()):
            _ = font[tag]
        stats = strip_truetype_hints(font)
        font.flavor = None
        font.save(str(output_path))
        return stats
    finally:
        _restore_patches(orig_compile, orig_pack)


def verify_font(path: Path) -> None:
    font = TTFont(str(path), recalcBBoxes=False, recalcTimestamp=False)
    glyf = font["glyf"]
    for name in glyf.keys():
        glyph = glyf[name]
        _ = glyph.numberOfContours
        if glyph.numberOfContours > 0 and not glyph.isComposite():
            _ = len(glyph.coordinates)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strip oversized/broken TrueType hints so WOFF2/TTF fonts open and "
            "ttx-dump cleanly under fontTools."
        )
    )
    parser.add_argument("input", type=Path, help="Input .woff2 / .ttf / .otf")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: <stem>-hints-fixed.ttf)",
    )
    parser.add_argument(
        "--keep-small-hints",
        action="store_true",
        help=(
            "Only drop instruction streams larger than 32767 bytes during "
            "WOFF2 reconstruct; still strip all hints before save (default)."
        ),
    )
    args = parser.parse_args(argv)

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        print(f"Error: not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}-hints-fixed.ttf")
    else:
        output_path = output_path.expanduser().resolve()

    print(f"Repairing: {input_path}")
    stats = repair_font(
        input_path,
        output_path,
        strip_all_hints=not args.keep_small_hints,
    )
    print(
        f"Cleared hints on {stats['glyphs_cleared']} glyph(s); "
        f"replaced {stats['glyphs_replaced_empty']} corrupt glyph(s); "
        f"removed {stats['tables_removed']} hinting table(s)."
    )
    print(f"Saved: {output_path}")

    try:
        verify_font(output_path)
    except Exception as exc:
        print(f"Warning: verification failed: {exc}", file=sys.stderr)
        return 1

    print("Verified: glyf decodes cleanly (safe for ttx).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
