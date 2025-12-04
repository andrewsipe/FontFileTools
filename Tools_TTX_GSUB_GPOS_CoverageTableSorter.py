#!/usr/bin/env python3
"""
Sort Coverage table glyph lists by GlyphOrder (glyph ID) in TTX files.

This tool processes TTX (XML) font files and sorts all <Glyph value="..."/>
entries within Coverage tables according to the font's GlyphOrder table,
which defines glyph IDs.

IMPORTANT: Coverage tables must be sorted by glyph ID (position in GlyphOrder),
NOT alphabetically!
"""

import argparse
import sys
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import FontCore.core_console_styles as cs


def extract_glyph_order(ttx_path):
    """
    Extract the GlyphOrder from a TTX file.
    Returns a dict mapping glyph names to their IDs.
    """
    try:
        tree = ET.parse(ttx_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid TTX XML: {e}")

    glyph_order = root.find("GlyphOrder")
    if glyph_order is None:
        raise ValueError("No GlyphOrder table found in TTX file")

    glyph_to_id = {}
    for glyph_id_elem in glyph_order.findall("GlyphID"):
        glyph_name = glyph_id_elem.get("name")
        glyph_id = int(glyph_id_elem.get("id"))
        if glyph_name:
            glyph_to_id[glyph_name] = glyph_id

    return glyph_to_id


def process_ttx_text(ttx_path, output_path=None, backup=False, verbose=False):
    """
    Text-based processing that preserves formatting.
    Sorts Coverage blocks by GlyphOrder (glyph ID).
    """
    ttx_path = Path(ttx_path)

    if output_path is None:
        output_path = ttx_path
    else:
        output_path = Path(output_path)

    # First, extract the glyph order
    if verbose:
        cs.StatusIndicator("info").add_message("Extracting GlyphOrder...").emit()

    glyph_to_id = extract_glyph_order(ttx_path)

    if verbose:
        cs.StatusIndicator("info").add_message(
            f"Found {len(glyph_to_id)} glyphs in GlyphOrder"
        ).emit()

    # Read the entire TTX file
    with open(ttx_path, "r", encoding="utf-8") as f:
        content = f.read()

    total_coverage = 0
    sorted_coverage = 0

    # Pattern to match Coverage blocks with their Glyph entries
    def process_coverage_block(match):
        nonlocal total_coverage, sorted_coverage

        total_coverage += 1
        full_block = match.group(0)
        indent = match.group(1)
        coverage_attrs = match.group(2)
        inner_content = match.group(3)

        # Find all Glyph value lines
        glyph_pattern = re.compile(r'(\s*)<Glyph value="([^"]+)"/>')
        glyph_matches = list(glyph_pattern.finditer(inner_content))

        if len(glyph_matches) <= 1:
            return full_block

        # Extract glyph values
        glyph_values = [m.group(2) for m in glyph_matches]

        # Sort by glyph ID (position in GlyphOrder)
        # Glyphs not in GlyphOrder get a high number to sort last
        sorted_values = sorted(glyph_values, key=lambda g: glyph_to_id.get(g, 999999))

        # Check if sorting is needed
        if glyph_values == sorted_values:
            return full_block

        sorted_coverage += 1

        if verbose:
            # Show what changed
            unsorted_ids = [glyph_to_id.get(g, -1) for g in glyph_values[:5]]
            sorted_ids = [glyph_to_id.get(g, -1) for g in sorted_values[:5]]
            cs.StatusIndicator("info").add_message(
                f"Sorted Coverage (was IDs {unsorted_ids}, now {sorted_ids})"
            ).emit()

        # Rebuild the Coverage block with sorted glyphs
        # Detect the indentation of the first Glyph element
        first_glyph_indent = glyph_matches[0].group(1) if glyph_matches else "        "

        # Build sorted glyph lines
        sorted_glyph_lines = [
            f'{first_glyph_indent}<Glyph value="{value}"/>' for value in sorted_values
        ]

        # Reconstruct the Coverage block
        result = f"{indent}<Coverage{coverage_attrs}>\n"
        result += "\n".join(sorted_glyph_lines)
        result += f"\n{indent}</Coverage>"

        return result

    # Pattern to match Coverage blocks
    coverage_pattern = re.compile(
        r"(\s*)<Coverage([^>]*)>\s*\n((?:\s*<Glyph[^>]+/>\s*\n)+)\s*\1</Coverage>",
        re.MULTILINE,
    )

    # Replace all Coverage blocks with sorted versions
    new_content = coverage_pattern.sub(process_coverage_block, content)

    # Create backup if requested
    if backup and output_path == ttx_path:
        backup_path = ttx_path.with_suffix(ttx_path.suffix + ".bak")
        import shutil

        shutil.copy2(ttx_path, backup_path)
        if verbose:
            cs.StatusIndicator("info").add_file(str(backup_path)).add_message(
                "Created backup"
            ).emit()

    # Write the modified content
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return total_coverage, sorted_coverage


def main():
    parser = argparse.ArgumentParser(
        description="""
        Sort Coverage table glyph lists by GlyphOrder (glyph ID) in TTX files.
        
        This tool processes TTX (XML) font files and sorts all <Glyph value="..."/> 
        entries within Coverage tables according to the font's GlyphOrder table.
        
        IMPORTANT: Coverage tables MUST be sorted by glyph ID (the order glyphs 
        appear in the GlyphOrder table), NOT alphabetically!
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert font to TTX, sort Coverage tables, and compile back
  ttx font.otf
  %(prog)s font.ttx
  ttx -o font-fixed.otf font.ttx
  
  # With backup and verbose output
  %(prog)s font.ttx --backup -v
  
  # Save to different file
  %(prog)s font.ttx -o font-sorted.ttx
        """,
    )

    parser.add_argument("ttx_file", help="TTX file to process")
    parser.add_argument(
        "-o", "--output", help="Output TTX file (default: overwrite input)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup file (.bak) before modifying",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    ttx_path = Path(args.ttx_file)

    if not ttx_path.exists():
        cs.StatusIndicator("error").with_explanation(
            f"TTX file not found: {ttx_path}"
        ).emit()
        sys.exit(1)

    if not ttx_path.suffix.lower() == ".ttx":
        cs.StatusIndicator("warning").with_explanation(
            f"File doesn't have .ttx extension: {ttx_path.suffix}"
        ).emit()

    try:
        if args.verbose:
            cs.StatusIndicator("info").add_file(str(ttx_path)).add_message(
                "Processing"
            ).emit()

        total, sorted_count = process_ttx_text(
            ttx_path, args.output, args.backup, args.verbose
        )

        output_path = Path(args.output) if args.output else ttx_path

        if sorted_count == 0:
            cs.StatusIndicator("info").add_message(
                f"All {total} Coverage tables already sorted by glyph ID"
            ).emit()
        else:
            cs.StatusIndicator("updated").add_file(ttx_path.name).add_message(
                f"Sorted {sorted_count} of {total} Coverage tables by glyph ID"
            ).emit()

            if args.verbose:
                cs.StatusIndicator("saved").add_file(str(output_path)).emit()

            # Suggest next steps
            if output_path.suffix.lower() == ".ttx":
                font_name = output_path.stem
                cs.StatusIndicator("info").add_message(
                    f"Next: ttx -o {font_name}.otf {output_path.name}"
                ).emit()

    except Exception as e:
        cs.StatusIndicator("error").with_explanation(f"Error processing: {e}").emit()
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
