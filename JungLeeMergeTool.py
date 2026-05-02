#!/usr/bin/env python3
"""
Merge A+B (and optionally A+B+C) font pairs into complete fonts.

Expects files named <stem>A.otf, <stem>B.otf, and optionally <stem>C.otf
(e.g. MyFamilyA.otf + MyFamilyB.otf → merged/MyFamily.otf).

Usage:
    pip install fonttools
    python3 JungLeeMergeTool.py [FONT_FOLDER]

FONT_FOLDER defaults to the current directory. Merged fonts go to FONT_FOLDER/merged/.

OpenType layout (GDEF/GSUB/GPOS) from each part is combined: the *A* file is the
structural base; *B* and *C* contribute extra lookups/features on top (e.g. kern
from B, figure features from C).
"""

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


def _merge_otl_donor_into_merged(merged: TTFont, donor: TTFont) -> None:
    """
    Combine GDEF/GSUB/GPOS from donor into merged (base already has OTL from TTX).
    Uses fontTools merge layout pre/post passes so lookup and feature indices stay valid.
    """
    otl_tags = ("GDEF", "GSUB", "GPOS")
    if not any(donor.get(tag) is not None for tag in otl_tags):
        return

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


def merge_two(path_a, path_b, path_out):
    """
    Merge glyphs from path_b (donor) into path_a (base). Argument order is fixed:
    the first font supplies cmap/name/OS/2 and OTL structure in TTX; donor glyphs
    and donor OTL are folded in afterward.
    Saves result to path_out. Returns (total_glyphs, glyphs_taken_from_donor).
    """
    tmp_a = path_out.parent / (path_a.stem + "_desub_tmp.otf")
    tmp_b = path_out.parent / (path_b.stem + "_desub_tmp.otf")
    try:
        desubroutinize(path_a, tmp_a)
        desubroutinize(path_b, tmp_b)

        a = TTFont(str(tmp_a))
        b = TTFont(str(tmp_b))

        existing = set(a.getGlyphOrder())
        cmap_a = a.getBestCmap()
        cmap_b = b.getBestCmap()
        if cmap_a is None:
            cmap_a = {}
        if cmap_b is None:
            cmap_b = {}

        # Names B has that A does not (unencoded extras, etc.)
        missing_by_name = {g for g in b.getGlyphOrder() if g not in existing}
        # Glyphs that back codepoints present in B's cmap but not A's (split-font case:
        # same glyph order/names in both fonts; B fills outlines A left blank)
        from_b_by_unicode = {cmap_b[u] for u in cmap_b if u not in cmap_a}
        glyphs_to_take = missing_by_name | from_b_by_unicode

        if not glyphs_to_take:
            shutil.copy(str(path_a), str(path_out))
            return len(a.getGlyphOrder()), 0

        to_add = glyphs_to_take - existing
        to_replace = glyphs_to_take & existing

        # Export to TTX for XML-level merge
        ttx_a = tempfile.NamedTemporaryFile(suffix=".ttx", delete=False).name
        ttx_b = tempfile.NamedTemporaryFile(suffix=".ttx", delete=False).name
        ttx_out = tempfile.NamedTemporaryFile(suffix=".ttx", delete=False).name

        try:
            a.saveXML(ttx_a)
            b.saveXML(ttx_b)

            root_a = ET.parse(ttx_a).getroot()
            root_b = ET.parse(ttx_b).getroot()
            tree_a = ET.ElementTree(root_a)

            # 1. GlyphOrder (append only new glyph names)
            go_a = root_a.find("GlyphOrder")
            go_b = root_b.find("GlyphOrder")
            next_id = max(int(el.get("id")) for el in go_a.findall("GlyphID")) + 1
            for el in go_b.findall("GlyphID"):
                if el.get("name") in to_add:
                    new_el = copy.deepcopy(el)
                    new_el.set("id", str(next_id))
                    go_a.append(new_el)
                    next_id += 1

            # 2. CFF CharStrings — replace overlapping glyphs, then append new
            cs_a = root_a.find(".//CharStrings")
            cs_b = root_b.find(".//CharStrings")
            if cs_a is None or cs_b is None:
                raise RuntimeError(
                    "Missing CFF CharStrings in TTX — this tool expects CFF/.otf sources, not TrueType glyf."
                )
            _replace_named_xml_children(cs_a, cs_b, to_replace)
            for child in cs_b:
                if child.get("name") in to_add:
                    cs_a.append(copy.deepcopy(child))

            # 3. hmtx
            hmtx_a = root_a.find("hmtx")
            hmtx_b = root_b.find("hmtx")
            if hmtx_a is None or hmtx_b is None:
                raise RuntimeError("Missing hmtx table in TTX export.")
            _replace_named_xml_children(hmtx_a, hmtx_b, to_replace)
            for child in hmtx_b:
                if child.get("name") in to_add:
                    hmtx_a.append(copy.deepcopy(child))

            # cmap: filled after importXML via fontTools (format 4 + 12, supplementary)

            # Compile merged TTX → OTF
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


def main():
    ap = argparse.ArgumentParser(
        description="Merge Jung Lee-style A+B (+C) OTF pairs into single fonts."
    )
    ap.add_argument(
        "directory",
        nargs="?",
        default=".",
        type=Path,
        help="Folder containing *A.otf / *B.otf / optional *C.otf (default: current directory)",
    )
    args = ap.parse_args()
    folder = args.directory.expanduser().resolve()
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        return

    output = folder / "merged"
    output.mkdir(exist_ok=True)

    a_files = sorted(folder.glob("*A.otf"))
    if not a_files:
        print(f"No *A.otf files in {folder}")
        print("Files must be named <name>A.otf with matching <name>B.otf (e.g. FooA.otf, FooB.otf).")
        return

    print(f"Scanning: {folder}")
    print(f"Found {len(a_files)} A-font(s). Merging...\n")
    success, skipped, errors = 0, 0, 0

    for path_a in a_files:
        stem = path_a.stem[:-1]  # Strip trailing "A"
        path_b = path_a.parent / (stem + "B.otf")
        path_c = path_a.parent / (stem + "C.otf")
        out_path = output / (stem + ".otf")

        if not path_b.exists():
            print(f"  SKIP  {path_a.name} — no matching B file found")
            skipped += 1
            continue

        try:
            print(f"  Merging {path_a.name} + {path_b.name}", end="", flush=True)

            if path_c.exists():
                # Three-way merge: A+B → temp, then temp+C → output
                print(f" + {path_c.name}", end="", flush=True)
                tmp_ab = output / (stem + "_AB_tmp.otf")
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