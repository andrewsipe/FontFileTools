#!/usr/bin/env python3
"""
Merge each ``{name}`` + ``{name}-Space`` font pair and write desktop SFNT files
into ``DIRECTORY/combined/``, using the main file’s base name (no ``-Space`` suffix).

Example:
  python HoeflerSpaceCombiner.py ~/Fonts/Uncombined
  # creates ~/Fonts/Uncombined/combined/MyFont.ttf from MyFont.woff + MyFont-Space.woff
"""
import argparse
from copy import deepcopy
from pathlib import Path

from fontTools.cffLib.transforms import desubroutinizeCharString
from fontTools.ttLib import TTFont

VALID_EXTS = (".woff", ".woff2", ".ttf", ".otf")


def next_free_path(desired_path: Path) -> Path:
    """If *desired_path* exists, return ``stem#N`` + suffix (same pattern as TTX_converter)."""
    if not desired_path.exists():
        return desired_path
    parent = desired_path.parent
    stem = desired_path.stem
    suffix = desired_path.suffix
    n = 1
    while True:
        candidate = parent / f"{stem}#{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def find_font_pairs(directory: Path):
    """Find pairs of main and companion ``{stem}-Space`` font files in *directory*."""
    dir_path = directory.resolve()
    font_pairs = []
    entries = [p for p in dir_path.iterdir() if p.is_file()]

    for path in entries:
        if path.suffix.lower() not in VALID_EXTS:
            continue
        stem = path.stem
        if stem.endswith("-Space"):
            continue
        space_path = next(
            (
                p
                for p in entries
                if p.suffix.lower() in VALID_EXTS and p.stem == f"{stem}-Space"
            ),
            None,
        )
        if space_path:
            font_pairs.append((path, space_path))
    return font_pairs


def _copy_glyf_glyph(main_font, space_font, glyph_name):
    """Copy a ``glyf`` glyph, pulling in composite components from *space_font* if needed."""
    space_glyf = space_font["glyf"]
    main_glyf = main_font["glyf"]
    if glyph_name not in space_glyf or glyph_name not in main_glyf:
        return

    def ensure_component(name, visiting):
        if name in main_glyf:
            return
        if name not in space_glyf:
            raise ValueError(
                f"Cannot copy '{glyph_name}': component '{name}' is not present "
                "in the space font's glyf table."
            )
        if name in visiting:
            raise ValueError(f"Circular composite glyph dependency involving '{name}'.")
        visiting.add(name)
        comp_glyph = space_glyf[name]
        if comp_glyph.isComposite():
            for comp in comp_glyph.components:
                ensure_component(comp.glyphName, visiting)
        main_glyf[name] = deepcopy(space_glyf[name])
        visiting.remove(name)

    src = space_glyf[glyph_name]
    if src.isComposite():
        for comp in src.components:
            ensure_component(comp.glyphName, set())
    main_glyf[glyph_name] = deepcopy(src)


def _copy_cff_glyphs(main_font, space_font, glyph_names):
    """Copy PostScript CFF (``CFF `` table) outlines; desubroutinize so subroutine indices need not match."""
    main_cs = main_font["CFF "].cff.topDictIndex[0].CharStrings
    space_cs = space_font["CFF "].cff.topDictIndex[0].CharStrings
    for glyph in glyph_names:
        if glyph not in space_cs or glyph not in main_cs:
            continue
        dest_cs, _fd = main_cs.getItemAndSelector(glyph)
        src = deepcopy(space_cs[glyph])
        src.decompile()
        try:
            desubroutinizeCharString(src)
        except Exception:
            src = deepcopy(space_cs[glyph])
            src.decompile()
        src.private = dest_cs.private
        src.globalSubrs = dest_cs.globalSubrs
        main_cs[glyph] = src


def merge_and_convert(main_path: Path, space_path: Path, output_dir: Path):
    """Merge metrics and outlines; strip WOFF flavor; save under *output_dir* using *main_path* stem."""
    main_path = Path(main_path)
    space_path = Path(space_path)
    output_dir = Path(output_dir)

    main_font = TTFont(main_path)
    space_font = TTFont(space_path)
    try:
        is_cff = main_font.sfntVersion == "OTTO"
        extension = "otf" if is_cff else "ttf"
        main_font.flavor = None

        target_glyphs = ["space", "uni00A0", ".notdef"]

        for glyph in target_glyphs:
            if glyph in space_font.getGlyphOrder() and glyph in main_font.getGlyphOrder():
                width, lsb = space_font["hmtx"].metrics[glyph]
                main_font["hmtx"].metrics[glyph] = (width, lsb)

        if "glyf" in main_font and "glyf" in space_font:
            for glyph in target_glyphs:
                if glyph in space_font.getGlyphOrder():
                    _copy_glyf_glyph(main_font, space_font, glyph)
        elif "CFF " in main_font and "CFF " in space_font:
            _copy_cff_glyphs(main_font, space_font, target_glyphs)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = next_free_path(output_dir / f"{main_path.stem}.{extension}")
        main_font.save(output_path)
        return str(output_path)
    finally:
        main_font.close()
        space_font.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge each main font with its {stem}-Space companion; write results under "
            "DIRECTORY/combined/ using the main filename (no -Space suffix)."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="Folder containing paired fonts (optional if -d is used)",
    )
    parser.add_argument(
        "-d",
        "--directory",
        dest="directory_flag",
        default=None,
        help="Same as positional directory (either may be used)",
    )
    parser.add_argument(
        "--output-subdir",
        default="combined",
        metavar="NAME",
        help="Subdirectory inside DIRECTORY for merged fonts (default: combined)",
    )
    args = parser.parse_args()

    dir_arg = args.directory or args.directory_flag or "."
    source_dir = Path(dir_arg).expanduser().resolve()
    if not source_dir.is_dir():
        print(f"Not a directory: {source_dir}")
        return

    combined_dir = (source_dir / args.output_subdir).resolve()

    pairs = find_font_pairs(source_dir)

    if not pairs:
        print("No matching '-Space' font pairs found.")
        return

    print(f"Found {len(pairs)} pair(s). Writing to {combined_dir}/")
    for main_file, space_file in pairs:
        try:
            result = merge_and_convert(main_file, space_file, combined_dir)
            print(f"DONE: {main_file.name} + {space_file.name} -> {result}")
        except Exception as e:
            print(f"ERROR processing {main_file.name}: {e}")


if __name__ == "__main__":
    main()
