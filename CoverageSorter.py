#!/usr/bin/env python3
"""
Script to manually sort Coverage tables in font files by glyph ID order.
This fixes the warning: "GSUB/GPOS Coverage is not sorted by glyph ids"
"""

from fontTools import ttLib
import sys


def get_glyph_id(font, glyph_name):
    """Get the glyph ID for a glyph name."""
    try:
        return font.getGlyphID(glyph_name)
    except (KeyError, ValueError, AttributeError):
        return float("inf")  # Put unknown glyphs at the end


def sort_coverage(font, coverage):
    """Sort a Coverage table by glyph IDs."""
    if not hasattr(coverage, "glyphs") or not coverage.glyphs:
        return

    # Get glyph IDs and sort
    glyph_data = [(get_glyph_id(font, g), g) for g in coverage.glyphs]
    glyph_data.sort(key=lambda x: x[0])
    coverage.glyphs = [g for _, g in glyph_data]


def sort_class_def(font, class_def):
    """Sort a ClassDef table by glyph IDs."""
    if not hasattr(class_def, "classDefs") or not class_def.classDefs:
        return

    # ClassDef is a dict, just ensure it's ordered by glyph ID
    sorted_items = sorted(
        class_def.classDefs.items(), key=lambda x: get_glyph_id(font, x[0])
    )
    class_def.classDefs = dict(sorted_items)


def process_lookup(font, lookup):
    """Process a single lookup and sort all its Coverage tables."""
    if not hasattr(lookup, "SubTable"):
        return

    for subtable in lookup.SubTable:
        # Sort main Coverage
        if hasattr(subtable, "Coverage"):
            sort_coverage(font, subtable.Coverage)

        # Handle different subtable types
        if hasattr(subtable, "ClassDef"):
            sort_class_def(font, subtable.ClassDef)

        if hasattr(subtable, "BacktrackCoverage"):
            for cov in subtable.BacktrackCoverage:
                sort_coverage(font, cov)

        if hasattr(subtable, "InputCoverage"):
            for cov in subtable.InputCoverage:
                sort_coverage(font, cov)

        if hasattr(subtable, "LookAheadCoverage"):
            for cov in subtable.LookAheadCoverage:
                sort_coverage(font, cov)

        # PairPos specific
        if hasattr(subtable, "PairSet"):
            try:
                if hasattr(subtable.Coverage, "glyphs") and subtable.Coverage.glyphs:
                    # Need to reorder PairSet to match sorted Coverage
                    old_glyphs = list(subtable.Coverage.glyphs)
                    sort_coverage(font, subtable.Coverage)
                    new_glyphs = subtable.Coverage.glyphs

                    # Create mapping from old position to new position
                    # Build mapping by iterating with indices to handle duplicates correctly
                    old_to_new = {}
                    for old_idx, glyph in enumerate(old_glyphs):
                        if glyph in new_glyphs:
                            new_idx = new_glyphs.index(glyph)
                            old_to_new[old_idx] = new_idx

                    # Reorder PairSet array
                    if subtable.PairSet and len(old_to_new) == len(old_glyphs):
                        old_pairset = subtable.PairSet[:]
                        new_pairset = [None] * len(old_pairset)
                        for old_idx, new_idx in old_to_new.items():
                            if old_idx < len(old_pairset):
                                new_pairset[new_idx] = old_pairset[old_idx]

                        # Validate no None values remain
                        if None in new_pairset:
                            # Fallback: keep original order if mapping incomplete
                            print(
                                "  Warning: Incomplete PairSet mapping, keeping original order"
                            )
                        else:
                            subtable.PairSet = new_pairset
            except (AttributeError, TypeError, ValueError) as e:
                print(f"  Warning: Failed to reorder PairSet: {e}")
                pass

        # LigatureSubst specific - reorder ligature sets
        if hasattr(subtable, "ligatures"):
            try:
                if hasattr(subtable.Coverage, "glyphs") and subtable.Coverage.glyphs:
                    old_glyphs = list(subtable.Coverage.glyphs)
                    sort_coverage(font, subtable.Coverage)
                    new_glyphs = subtable.Coverage.glyphs

                    old_ligatures = subtable.ligatures.copy()
                    new_ligatures = {}
                    for glyph in new_glyphs:
                        if glyph in old_ligatures:
                            new_ligatures[glyph] = old_ligatures[glyph]
                    subtable.ligatures = new_ligatures
            except (AttributeError, TypeError):
                pass


def process_table(font, table_tag):
    """Process GSUB or GPOS table."""
    if table_tag not in font:
        return

    table = font[table_tag]
    print(f"  Processing {table_tag} table...")

    if hasattr(table, "table"):
        table = table.table

    # Process all lookup lists
    if hasattr(table, "LookupList") and table.LookupList:
        for i, lookup in enumerate(table.LookupList.Lookup):
            process_lookup(font, lookup)

    # Process FeatureList
    if hasattr(table, "FeatureList") and table.FeatureList:
        for feature_record in table.FeatureList.FeatureRecord:
            if hasattr(feature_record, "Feature"):
                # Features reference lookups, no Coverage to sort here
                pass


def process_gdef(font):
    """Process GDEF table."""
    if "GDEF" not in font:
        return

    print("  Processing GDEF table...")
    gdef = font["GDEF"].table

    # Sort LigCaretList Coverage
    if hasattr(gdef, "LigCaretList") and gdef.LigCaretList:
        lig_caret = gdef.LigCaretList
        if hasattr(lig_caret, "Coverage"):
            old_glyphs = (
                list(lig_caret.Coverage.glyphs) if lig_caret.Coverage.glyphs else []
            )
            sort_coverage(font, lig_caret.Coverage)
            new_glyphs = lig_caret.Coverage.glyphs

            # Reorder LigGlyph array to match sorted Coverage
            if hasattr(lig_caret, "LigGlyph") and lig_caret.LigGlyph and old_glyphs:
                old_lig_glyphs = lig_caret.LigGlyph[:]
                new_lig_glyphs = [None] * len(old_lig_glyphs)

                for i, old_glyph in enumerate(old_glyphs):
                    if old_glyph in new_glyphs and i < len(old_lig_glyphs):
                        new_idx = new_glyphs.index(old_glyph)
                        new_lig_glyphs[new_idx] = old_lig_glyphs[i]

                # Validate no None values remain before assigning
                if None in new_lig_glyphs:
                    print(
                        "  Warning: Incomplete LigGlyph mapping, filtering None values"
                    )
                    lig_caret.LigGlyph = [lg for lg in new_lig_glyphs if lg is not None]
                else:
                    lig_caret.LigGlyph = new_lig_glyphs

    # Sort AttachList Coverage
    if hasattr(gdef, "AttachList") and gdef.AttachList:
        if hasattr(gdef.AttachList, "Coverage"):
            sort_coverage(font, gdef.AttachList.Coverage)

    # Sort MarkAttachClassDef
    if hasattr(gdef, "MarkAttachClassDef") and gdef.MarkAttachClassDef:
        sort_class_def(font, gdef.MarkAttachClassDef)

    # Sort GlyphClassDef
    if hasattr(gdef, "GlyphClassDef") and gdef.GlyphClassDef:
        sort_class_def(font, gdef.GlyphClassDef)


def sort_font_coverage(input_path, output_path=None):
    """
    Load a font file, sort all Coverage tables, and save.

    Args:
        input_path: Path to input font file (.ttf, .otf, or .ttx)
        output_path: Path for output file (optional, defaults to input_path with '_sorted' suffix)
    """
    # Determine output path
    if output_path is None:
        if input_path.endswith(".ttx"):
            output_path = input_path.replace(".ttx", "_sorted.ttx")
        else:
            parts = input_path.rsplit(".", 1)
            output_path = f"{parts[0]}_sorted.{parts[1]}"

    print(f"Loading font: {input_path}")

    # Load the font
    if input_path.endswith(".ttx"):
        font = ttLib.TTFont()
        font.importXML(input_path)
    else:
        font = ttLib.TTFont(input_path)

    print("Sorting Coverage tables...")

    # Process tables
    process_table(font, "GSUB")
    process_table(font, "GPOS")
    process_gdef(font)

    print(f"Saving sorted font: {output_path}")

    # Save the font
    if output_path.endswith(".ttx"):
        font.saveXML(output_path)
    else:
        font.save(output_path)

    print("Done! Coverage tables have been sorted.")
    print(f"\nVerify by running: ttx {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python TTXCoverageSorter.py <input_font> [output_font]")
        print("\nExamples:")
        print("  python TTXCoverageSorter.py myfont.ttf")
        print("  python TTXCoverageSorter.py myfont.ttx myfont_fixed.ttx")
        print("  python TTXCoverageSorter.py myfont.ttf myfont_fixed.ttf")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        sort_font_coverage(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
