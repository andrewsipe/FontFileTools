#!/usr/bin/env python3
"""
Merge Jung Lee-style split fonts (A+B, or A+B+C) into one complete font.

Supports:
  - Static CFF OpenType (.otf)
  - Static TrueType (.ttf)
  - TrueType variable fonts (.ttf / .otf with glyf + gvar + fvar)

Does NOT read .woff / .woff2. Convert those to .otf or .ttf first, then run
this tool on the converted folder.

Naming (required)
  The part letter is the LAST character of the filename stem — not a hyphen
  suffix and not a separate word:

    YES  MyFontA.otf  + MyFontB.otf  (+ MyFontC.otf)  →  merged/MyFont.otf
    YES  MyFontA.ttf  + MyFontB.ttf  (+ MyFontC.ttf)  →  merged/MyFont.ttf
    NO   MyFont-A.otf     (looks for MyFont-B.otf)
    NO   MyFont_A.otf     (looks for MyFont_B.otf)
    NO   MyFont A.otf     (space breaks the stem)

A / B / C for one family must share the same extension (.otf with .otf, or
.ttf with .ttf).

Usage:
    pip install fonttools
    python3 JungLeeMergeTool.py [FONT_FOLDER]

FONT_FOLDER defaults to the current directory. Merged fonts go to
FONT_FOLDER/merged/.

OpenType layout (GDEF/GSUB/GPOS) from each part is combined: *A* is the
structural base (names, OS/2, fvar/STAT when present); *B* and *C* donate
glyphs, cmap entries, and extra lookups/features.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from fontTools import subset as ftsubset
from fontTools.merge import layout as merge_layout
from fontTools.merge.base import mergeObjects
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
from pathlib import Path

SUPPORTED_EXTS = (".otf", ".ttf")

HELP_EPILOG = """
naming (required):
  The part letter is the LAST character of the filename stem:

    YES  MyFontA.otf + MyFontB.otf (+ MyFontC.otf)  →  merged/MyFont.otf
    YES  MyFontA.ttf + MyFontB.ttf (+ MyFontC.ttf)  →  merged/MyFont.ttf
    NO   MyFont-A.otf   (script looks for MyFont-B.otf)
    NO   MyFont_A.otf   (script looks for MyFont_B.otf)

  A/B/C for one family must share the same extension.

before you run:
  1. Convert .woff / .woff2 → .otf or .ttf (this script does not open WOFF).
  2. Name parts as above (stem ends with A, B, or C).
  3. Put matching A+B(+C) files in one folder, then pass that folder.

supported:
  Static CFF .otf, static TrueType .ttf, and TrueType variable fonts
  (glyf + gvar + fvar). CFF2 variable fonts are not supported.

examples:
  python3 JungLeeMergeTool.py
  python3 JungLeeMergeTool.py /path/to/converted/fonts

output:
  FONT_FOLDER/merged/<stem>.otf or .ttf  (same extension as the A file)
"""


def _replace_named_xml_children(dst_parent, src_parent, names):
    """Replace elements with matching ``name`` attribute; order of siblings preserved."""
    for gname in names:
        src = next((c for c in src_parent if c.get("name") == gname), None)
        if src is None:
            continue
        new_el = copy.deepcopy(src)
        for i, child in enumerate(list(dst_parent)):
            if child.get("name") == gname:
                dst_parent.remove(child)
                dst_parent.insert(i, new_el)
                break


def _merge_cmap_entries_from_b(merged: TTFont, font_a: TTFont, font_b: TTFont) -> None:
    """
    Add Unicode → glyph mappings from B that A did not have.
    BMP updates go to format 4 when no format 12 exists; supplementary
    codepoints require a format-12 subtable (created if missing).
    """
    cmap_a = font_a.getBestCmap() or {}
    cmap_b = font_b.getBestCmap() or {}
    if not cmap_b:
        return
    merged_order = set(merged.getGlyphOrder())
    additions = {u: g for u, g in cmap_b.items() if u not in cmap_a and g in merged_order}
    if not additions:
        return

    cmap = merged["cmap"]
    st12 = cmap.getcmap(3, 10) or cmap.getcmap(0, 4) or cmap.getcmap(0, 6)
    if st12 is not None:
        m = dict(st12.cmap)
        m.update(additions)
        st12.cmap = m
        return

    st4 = cmap.getcmap(3, 1) or cmap.getcmap(0, 3)
    bmp_add = {u: g for u, g in additions.items() if u <= 0xFFFF}
    has_smp = any(u > 0xFFFF for u in additions)

    if st4 is not None and bmp_add:
        m = dict(st4.cmap)
        m.update(bmp_add)
        st4.cmap = m

    if has_smp:
        new12 = CmapSubtable.newSubtable(12)
        new12.platformID = 3
        new12.platEncID = 10
        new12.language = 0
        base = merged.getBestCmap() or {}
        new12.cmap = {**dict(base), **additions}
        cmap.tables.append(new12)


def desubroutinize(path_in, path_out):
    """Flatten CFF subroutines so charstrings are self-contained."""
    font = TTFont(str(path_in))
    options = ftsubset.Options()
    options.desubroutinize = True
    # Default subsetter layout list omits ss01/aalt/dlig/… and prunes GSUB/GPOS.
    options.layout_features = ["*"]
    options.layout_closure = False
    subsetter = ftsubset.Subsetter(options=options)
    subsetter.populate(glyphs=font.getGlyphOrder())
    subsetter.subset(font)
    font.save(str(path_out))


def _strip_gdef_varstore(font: TTFont) -> None:
    """
    fontTools' layout merger cannot combine GDEF ItemVariationStore (VarStore).
    Drop it so GDEF/GSUB/GPOS can merge; caret/ligature GDEF deltas are lost.
    """
    gdef = font.get("GDEF")
    if gdef is None:
        return
    gdef.ensureDecompiled()
    table = gdef.table
    if getattr(table, "VarStore", None) is None:
        return
    table.VarStore = None
    if table.Version >= 0x00010003:
        if getattr(table, "MarkGlyphSetsDef", None) is not None:
            table.Version = 0x00010002
        else:
            table.Version = 0x00010000


def _merge_otl_donor_into_merged(merged: TTFont, donor: TTFont) -> None:
    """
    Combine GDEF/GSUB/GPOS from donor into merged (base already has OTL).
    Uses fontTools merge layout pre/post passes so lookup and feature indices stay valid.
    """
    otl_tags = ("GDEF", "GSUB", "GPOS")
    if not any(donor.get(tag) is not None for tag in otl_tags):
        return

    _strip_gdef_varstore(merged)
    _strip_gdef_varstore(donor)

    merge_layout.layoutPreMerge(merged)
    merge_layout.layoutPreMerge(donor)
    for tag in otl_tags:
        dtab = donor.get(tag)
        if dtab is None:
            continue
        mtab = merged.get(tag)
        if mtab is None:
            merged[tag] = copy.deepcopy(dtab)
        else:
            mtab.table = mergeObjects((mtab.table, dtab.table))
    merge_layout.layoutPostMerge(merged)


def _outline_kind(font: TTFont) -> str:
    if "CFF2" in font:
        raise RuntimeError(
            "CFF2 variable fonts are not supported (only CFF static .otf, "
            "or TrueType glyf / gvar)."
        )
    if "CFF " in font:
        return "cff"
    if "glyf" in font:
        return "glyf"
    raise RuntimeError("Font has neither CFF nor glyf outlines — cannot merge.")


def _glyphs_to_take(font_a: TTFont, font_b: TTFont) -> set[str]:
    existing = set(font_a.getGlyphOrder())
    cmap_a = font_a.getBestCmap() or {}
    cmap_b = font_b.getBestCmap() or {}
    missing_by_name = {g for g in font_b.getGlyphOrder() if g not in existing}
    from_b_by_unicode = {cmap_b[u] for u in cmap_b if u not in cmap_a}
    return missing_by_name | from_b_by_unicode


def _expand_glyf_components(glyf, names: set[str], already_in_base: set[str]) -> set[str]:
    """Include donor-only composite components required by glyphs in ``names``."""
    result = set(names)
    stack = list(names)
    while stack:
        name = stack.pop()
        if name not in glyf:
            continue
        glyph = glyf[name]
        if not glyph.isComposite():
            continue
        for component in glyph.components:
            comp = component.glyphName
            if comp in already_in_base or comp in result:
                continue
            if comp not in glyf:
                raise RuntimeError(
                    f"Composite '{name}' references missing component '{comp}'"
                )
            result.add(comp)
            stack.append(comp)
    return result


def _fvar_axis_signature(font: TTFont) -> tuple | None:
    if "fvar" not in font:
        return None
    return tuple(
        (a.axisTag, a.minValue, a.defaultValue, a.maxValue) for a in font["fvar"].axes
    )


def _copy_gvar_glyphs(merged: TTFont, donor: TTFont, glyph_names: set[str]) -> None:
    if "gvar" not in donor:
        return
    donor_gvar = donor["gvar"]
    donor_gvar.ensureDecompiled()
    if "gvar" not in merged:
        # Base was static; adopt donor gvar shell and only keep merged glyphs.
        merged["gvar"] = copy.deepcopy(donor_gvar)
        merged_gvar = merged["gvar"]
        keep = set(merged.getGlyphOrder())
        merged_gvar.variations = {
            name: copy.deepcopy(list(vars_))
            for name, vars_ in donor_gvar.variations.items()
            if name in keep
        }
        return

    merged_gvar = merged["gvar"]
    merged_gvar.ensureDecompiled()
    for name in glyph_names:
        if name in donor_gvar.variations:
            merged_gvar.variations[name] = [
                copy.deepcopy(tv) for tv in donor_gvar.variations[name]
            ]


def _merge_two_cff(path_a: Path, path_b: Path, path_out: Path) -> tuple[int, int]:
    """Merge donor CFF glyphs into base via TTX CharStrings / hmtx."""
    suffix = path_a.suffix or ".otf"
    tmp_a = path_out.parent / (path_a.stem + "_desub_tmp" + suffix)
    tmp_b = path_out.parent / (path_b.stem + "_desub_tmp" + suffix)
    try:
        desubroutinize(path_a, tmp_a)
        desubroutinize(path_b, tmp_b)

        a = TTFont(str(tmp_a))
        b = TTFont(str(tmp_b))

        glyphs_to_take = _glyphs_to_take(a, b)
        if not glyphs_to_take:
            shutil.copy(str(path_a), str(path_out))
            return len(a.getGlyphOrder()), 0

        existing = set(a.getGlyphOrder())
        to_add = glyphs_to_take - existing
        to_replace = glyphs_to_take & existing

        ttx_a = tempfile.NamedTemporaryFile(suffix=".ttx", delete=False).name
        ttx_b = tempfile.NamedTemporaryFile(suffix=".ttx", delete=False).name
        ttx_out = tempfile.NamedTemporaryFile(suffix=".ttx", delete=False).name

        try:
            a.saveXML(ttx_a)
            b.saveXML(ttx_b)

            root_a = ET.parse(ttx_a).getroot()
            root_b = ET.parse(ttx_b).getroot()
            tree_a = ET.ElementTree(root_a)

            go_a = root_a.find("GlyphOrder")
            go_b = root_b.find("GlyphOrder")
            next_id = max(int(el.get("id")) for el in go_a.findall("GlyphID")) + 1
            for el in go_b.findall("GlyphID"):
                if el.get("name") in to_add:
                    new_el = copy.deepcopy(el)
                    new_el.set("id", str(next_id))
                    go_a.append(new_el)
                    next_id += 1

            cs_a = root_a.find(".//CharStrings")
            cs_b = root_b.find(".//CharStrings")
            if cs_a is None or cs_b is None:
                raise RuntimeError("Missing CFF CharStrings in TTX export.")
            _replace_named_xml_children(cs_a, cs_b, to_replace)
            for child in cs_b:
                if child.get("name") in to_add:
                    cs_a.append(copy.deepcopy(child))

            hmtx_a = root_a.find("hmtx")
            hmtx_b = root_b.find("hmtx")
            if hmtx_a is None or hmtx_b is None:
                raise RuntimeError("Missing hmtx table in TTX export.")
            _replace_named_xml_children(hmtx_a, hmtx_b, to_replace)
            for child in hmtx_b:
                if child.get("name") in to_add:
                    hmtx_a.append(copy.deepcopy(child))

            tree_a.write(ttx_out, encoding="unicode", xml_declaration=True)
            merged = TTFont()
            merged.importXML(ttx_out)
            _merge_cmap_entries_from_b(merged, a, b)
            _merge_otl_donor_into_merged(merged, b)
            merged.save(str(path_out))

            return len(merged.getGlyphOrder()), len(glyphs_to_take)

        finally:
            for f in [ttx_a, ttx_b, ttx_out]:
                try:
                    os.unlink(f)
                except OSError:
                    pass
    finally:
        for tmp in [tmp_a, tmp_b]:
            if tmp.exists():
                tmp.unlink()


def _merge_two_glyf(path_a: Path, path_b: Path, path_out: Path) -> tuple[int, int]:
    """Merge donor TrueType glyphs (and gvar, if present) into base."""
    a = TTFont(str(path_a))
    b = TTFont(str(path_b))

    sig_a, sig_b = _fvar_axis_signature(a), _fvar_axis_signature(b)
    if sig_a is not None and sig_b is not None and sig_a != sig_b:
        raise RuntimeError(
            f"fvar axes differ between {path_a.name} and {path_b.name}: "
            f"{sig_a} vs {sig_b}"
        )

    glyphs_to_take = _glyphs_to_take(a, b)
    if not glyphs_to_take:
        n = len(a.getGlyphOrder())
        a.close()
        b.close()
        shutil.copy(str(path_a), str(path_out))
        return n, 0

    existing = set(a.getGlyphOrder())
    glyf_b = b["glyf"]
    glyphs_to_take = _expand_glyf_components(glyf_b, glyphs_to_take, existing)

    to_add = glyphs_to_take - existing
    to_replace = glyphs_to_take & existing

    # Decompile gvar while glyphOrder still matches the table header count.
    if "gvar" in a:
        a["gvar"].ensureDecompiled()
    if "gvar" in b:
        b["gvar"].ensureDecompiled()

    glyf_a = a["glyf"]
    hmtx_a = a["hmtx"].metrics
    hmtx_b = b["hmtx"].metrics

    order = list(a.getGlyphOrder())
    for name in b.getGlyphOrder():
        if name in to_add and name not in existing:
            order.append(name)
            existing.add(name)
    a.setGlyphOrder(order)

    for name in to_add | to_replace:
        if name not in glyf_b:
            raise RuntimeError(f"Donor is missing glyph '{name}'")
        glyf_a[name] = copy.deepcopy(glyf_b[name])
        if name not in hmtx_b:
            raise RuntimeError(f"Donor is missing hmtx for '{name}'")
        hmtx_a[name] = hmtx_b[name]

    a["maxp"].numGlyphs = len(order)
    _copy_gvar_glyphs(a, b, to_add | to_replace)
    if "gvar" in a:
        a["gvar"].glyphCount = len(order)

    # Cmap compare against the pre-merge base (donor B still has its own cmap).
    base_for_cmap = TTFont(str(path_a))
    try:
        _merge_cmap_entries_from_b(a, base_for_cmap, b)
    finally:
        base_for_cmap.close()

    _merge_otl_donor_into_merged(a, b)
    a.save(str(path_out))
    total = len(order)
    taken = len(glyphs_to_take)
    a.close()
    b.close()
    return total, taken


def merge_two(path_a: Path, path_b: Path, path_out: Path) -> tuple[int, int]:
    """
    Merge glyphs from path_b (donor) into path_a (base). Argument order is fixed:
    the first font supplies cmap/name/OS/2 and OTL structure; donor glyphs
    and donor OTL are folded in afterward.
    Saves result to path_out. Returns (total_glyphs, glyphs_taken_from_donor).
    """
    peek = TTFont(str(path_a))
    try:
        kind = _outline_kind(peek)
    finally:
        peek.close()

    donor_peek = TTFont(str(path_b))
    try:
        donor_kind = _outline_kind(donor_peek)
    finally:
        donor_peek.close()

    if kind != donor_kind:
        raise RuntimeError(
            f"Outline mismatch: {path_a.name} is {kind}, {path_b.name} is {donor_kind}"
        )

    if kind == "cff":
        return _merge_two_cff(path_a, path_b, path_out)
    return _merge_two_glyf(path_a, path_b, path_out)


def _find_a_fonts(folder: Path) -> list[Path]:
    found: list[Path] = []
    for ext in SUPPORTED_EXTS:
        found.extend(folder.glob(f"*A{ext}"))
    # Prefer stable order; if both FooA.otf and FooA.ttf exist, keep both and
    # let the user see two merges (unusual).
    return sorted(found, key=lambda p: (p.stem.lower(), p.suffix.lower()))


def _partner(path_a: Path, letter: str) -> Path:
    return path_a.parent / f"{path_a.stem[:-1]}{letter}{path_a.suffix}"


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Merge Jung Lee-style A+B (+C) font parts into single fonts. "
            "Supports static CFF .otf and TrueType .ttf (including variable glyf/gvar). "
            "Does not open .woff/.woff2 — convert those first."
        ),
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "directory",
        nargs="?",
        default=".",
        type=Path,
        help=(
            "Folder of converted fonts named <stem>A.otf|.ttf with matching "
            "B (and optional C). Default: current directory"
        ),
    )
    args = ap.parse_args()
    folder = args.directory.expanduser().resolve()
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        return

    woffs = list(folder.glob("*.woff")) + list(folder.glob("*.woff2"))
    if woffs and not _find_a_fonts(folder):
        print(f"Found {len(woffs)} .woff/.woff2 file(s) but no *A.otf / *A.ttf.")
        print("Convert WOFF → OTF or TTF first, then re-run on the converted folder.")
        print("This script does not read WOFF directly.")
        return

    output = folder / "merged"
    output.mkdir(exist_ok=True)

    a_files = _find_a_fonts(folder)
    if not a_files:
        print(f"No *A.otf or *A.ttf files in {folder}")
        print(
            "Name parts so the stem ends with A/B/C, e.g. MyFontA.otf + MyFontB.otf "
            "(not MyFont-A.otf)."
        )
        return

    print(f"Scanning: {folder}")
    print(f"Found {len(a_files)} A-font(s). Merging...\n")
    success, skipped, errors = 0, 0, 0

    for path_a in a_files:
        stem = path_a.stem[:-1]  # Strip trailing "A"
        path_b = _partner(path_a, "B")
        path_c = _partner(path_a, "C")
        out_path = output / f"{stem}{path_a.suffix}"

        if not path_b.exists():
            print(f"  SKIP  {path_a.name} — no matching B file ({path_b.name})")
            skipped += 1
            continue

        try:
            print(f"  Merging {path_a.name} + {path_b.name}", end="", flush=True)

            if path_c.exists():
                print(f" + {path_c.name}", end="", flush=True)
                tmp_ab = output / f"{stem}_AB_tmp{path_a.suffix}"
                _, added_b = merge_two(path_a, path_b, tmp_ab)
                total, added_c = merge_two(tmp_ab, path_c, out_path)
                added = added_b + added_c
                tmp_ab.unlink()
            else:
                total, added = merge_two(path_a, path_b, out_path)

            print(f" → {out_path.name}  ({total} glyphs, +{added} merged in)")
            success += 1

        except Exception as e:
            print(f"\n  ERROR processing {path_a.name}: {e}")
            errors += 1

    print(f"\nDone. {success} merged, {skipped} skipped, {errors} errors.")
    print(f"Output folder: {output.resolve()}")


if __name__ == "__main__":
    main()
