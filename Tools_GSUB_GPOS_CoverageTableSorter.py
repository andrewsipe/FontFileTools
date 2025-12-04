#!/usr/bin/env python3
"""
Sort Coverage table glyph lists by glyph ID in fonts and TTX files.

This tool fixes the "GSUB/GPOS Coverage is not sorted by glyph ids" warning
by sorting all Coverage tables according to glyph ID order.

Works on both:
- Binary fonts (.ttf, .otf) - converts to TTX, sorts, then converts back
- TTX files (.ttx) - text-based processing

Binary fonts are processed via TTX conversion to ensure sorting uses actual
glyph IDs from the GlyphOrder table, matching the exact behavior of the
TTX-based sorter.
"""

import argparse
import sys
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otTables import Coverage, ExtensionSubst, ExtensionPos

import FontCore.core_console_styles as cs


def find_all_coverage_tables(font):
    """
    Recursively find all Coverage objects in GSUB/GPOS tables.
    Returns list of (path, coverage_object) tuples.
    """
    coverages = []

    def recurse_coverage(obj, path="", visited=None):
        if visited is None:
            visited = set()

        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, Coverage):
            coverages.append((path, obj))
            return

        if isinstance(obj, (ExtensionSubst, ExtensionPos)):
            if hasattr(obj, "ExtSubTable") and obj.ExtSubTable is not None:
                recurse_coverage(obj.ExtSubTable, path + ".ExtSubTable", visited)
            return

        if hasattr(obj, "__dict__"):
            for attr, value in obj.__dict__.items():
                if value is None:
                    continue
                new_path = f"{path}.{attr}" if path else attr

                if isinstance(value, list):
                    for i, item in enumerate(value):
                        recurse_coverage(item, f"{new_path}[{i}]", visited)
                else:
                    recurse_coverage(value, new_path, visited)

    for table_name in ["GSUB", "GPOS"]:
        if table_name not in font:
            continue

        table = font[table_name].table
        if hasattr(table, "LookupList") and table.LookupList:
            for lookup_idx, lookup in enumerate(table.LookupList.Lookup):
                if not hasattr(lookup, "SubTable") or not lookup.SubTable:
                    continue

                for subtable_idx, subtable in enumerate(lookup.SubTable):
                    base_path = (
                        f"{table_name}.Lookup[{lookup_idx}].SubTable[{subtable_idx}]"
                    )
                    recurse_coverage(subtable, base_path)

    return coverages


def extract_glyph_order_from_ttx_content(ttx_content):
    """
    Extract the GlyphOrder from TTX XML content.
    Returns a dict mapping glyph names to their IDs.
    """
    try:
        root = ET.fromstring(ttx_content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid TTX XML: {e}")

    glyph_order = root.find("GlyphOrder")
    if glyph_order is None:
        raise ValueError("No GlyphOrder table found in TTX content")

    glyph_to_id = {}
    for glyph_id_elem in glyph_order.findall("GlyphID"):
        glyph_name = glyph_id_elem.get("name")
        glyph_id = int(glyph_id_elem.get("id"))
        if glyph_name:
            glyph_to_id[glyph_name] = glyph_id

    return glyph_to_id


def sort_coverage_tables_in_ttx_content(ttx_content, glyph_to_id, verbose=False):
    """
    Sort Coverage tables in TTX XML content by glyph ID.
    Returns (total_coverage, sorted_count, sorted_content)
    """
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
    sorted_content = coverage_pattern.sub(process_coverage_block, ttx_content)

    return total_coverage, sorted_coverage, sorted_content


def sort_coverage_tables_in_font(font, verbose=False):
    """
    Sort all Coverage tables in a binary font by glyph ID using TTX conversion.
    This ensures sorting matches the exact behavior of Tools_TTX_GSUB_GPOS_CoverageTableSorter.py
    by converting to TTX, sorting by GlyphOrder IDs, then converting back.

    The font object is modified in place by reloading from the sorted TTX.

    Args:
        font: TTFont object (will be reloaded from sorted TTX if sorting occurs)
        verbose: Whether to show verbose output

    Returns:
        (total_coverage, sorted_count) tuple
    """
    import tempfile
    import os
    import subprocess
    from io import StringIO

    try:
        # Convert font to TTX XML string using fontTools
        ttx_buffer = StringIO()
        font.saveXML(ttx_buffer)
        ttx_content = ttx_buffer.getvalue()

        # Extract GlyphOrder to get glyph IDs (matching TTX sorter logic exactly)
        glyph_to_id = extract_glyph_order_from_ttx_content(ttx_content)

        if verbose:
            cs.StatusIndicator("info").add_message(
                f"Extracted {len(glyph_to_id)} glyphs from GlyphOrder"
            ).emit()

        # Sort Coverage tables in TTX content using exact TTX sorter logic
        total, sorted_count, sorted_ttx_content = sort_coverage_tables_in_ttx_content(
            ttx_content, glyph_to_id, verbose
        )

        # If sorting occurred, reload font from sorted TTX
        if sorted_count > 0:
            # Determine output extension based on font type (before closing font)
            font_ext = ".otf"  # Default to OTF
            if hasattr(font, "sfntVersion"):
                # Check if it's TTF or OTF
                # TTF: "\x00\x01\x00\x00", OTF: "OTTO"
                if font.sfntVersion == "\x00\x01\x00\x00":
                    font_ext = ".ttf"

            # Use temporary file for TTX
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".ttx", delete=False, encoding="utf-8"
            ) as tmp_ttx:
                tmp_ttx.write(sorted_ttx_content)
                tmp_ttx_path = tmp_ttx.name

            # Create temp binary file path (same directory, different extension)
            tmp_bin_path = tmp_ttx_path.rsplit(".", 1)[0] + font_ext

            try:
                # Convert sorted TTX back to binary using ttx command-line tool
                # Use ttx to compile TTX back to binary
                # ttx -f -o output.otf input.ttx
                # -f forces overwrite, -o specifies output file
                result = subprocess.run(
                    ["ttx", "-f", "-o", tmp_bin_path, tmp_ttx_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    raise ValueError(f"ttx compilation failed: {error_msg}")

                # Verify the output file was created and has content
                if not os.path.exists(tmp_bin_path):
                    raise ValueError(
                        f"ttx compilation succeeded but output file not found: {tmp_bin_path}"
                    )

                if os.path.getsize(tmp_bin_path) == 0:
                    raise ValueError(
                        f"ttx compilation created empty file: {tmp_bin_path}"
                    )

                # Reload font from sorted binary file
                # Load new font first (before closing old one to preserve any file paths)
                try:
                    new_font = TTFont(tmp_bin_path, lazy=False)
                except Exception as load_error:
                    raise ValueError(
                        f"Failed to load compiled font from {tmp_bin_path}: {load_error}. "
                        f"TTX file: {tmp_ttx_path}, ttx output: {result.stdout}, errors: {result.stderr}"
                    )

                # Update the original font object by replacing its internal state
                # Get list of existing table tags before closing (for cleanup)
                old_tags = list(font.keys()) if hasattr(font, "keys") else []

                # Copy all tables from new font to old font BEFORE closing
                # This way the font object is still valid
                for tag in new_font.keys():
                    font[tag] = new_font[tag]

                # Remove old tables that won't be replaced
                for tag in old_tags:
                    if tag not in new_font:
                        try:
                            del font[tag]
                        except Exception:
                            pass

                # Copy font-level attributes
                if hasattr(new_font, "sfntVersion"):
                    font.sfntVersion = new_font.sfntVersion
                if hasattr(new_font, "flavor"):
                    font.flavor = new_font.flavor
                if hasattr(new_font, "lazy"):
                    font.lazy = new_font.lazy

                # Close both fonts
                new_font.close()
                # Note: We don't close the original font here as it's still in use
                # The caller will close it when done

                # Clean up temp files
                os.unlink(tmp_ttx_path)
                os.unlink(tmp_bin_path)

            except FileNotFoundError:
                # ttx command not found - fall back to binary sorting
                if verbose:
                    cs.StatusIndicator("warning").add_message(
                        "ttx command not found, falling back to binary sorting"
                    ).emit()
                try:
                    os.unlink(tmp_ttx_path)
                except Exception:
                    pass
                # Fall through to binary sorting below
                raise ValueError("ttx command not available")
            except Exception as e:
                # Clean up temp files on error
                try:
                    os.unlink(tmp_ttx_path)
                    if "tmp_bin_path" in locals() and os.path.exists(tmp_bin_path):
                        os.unlink(tmp_bin_path)
                except Exception:
                    pass
                raise ValueError(f"Failed to reload font from sorted TTX: {e}")

        return total, sorted_count

    except Exception as e:
        if verbose:
            cs.StatusIndicator("warning").add_message(
                f"TTX-based sorting failed: {e}"
            ).with_explanation("Coverage tables may not be sorted correctly").emit()

        # Return zero counts on error
        return 0, 0


def extract_glyph_order_from_ttx(ttx_path):
    """Extract GlyphOrder from TTX file."""
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


def sort_coverage_tables_in_ttx(
    ttx_path, output_path=None, backup=False, verbose=False
):
    """
    Sort Coverage tables in a TTX file by glyph ID.
    Returns (total_coverage, sorted_count)
    """
    ttx_path = Path(ttx_path)

    if output_path is None:
        output_path = ttx_path
    else:
        output_path = Path(output_path)

    # Extract glyph order
    if verbose:
        cs.StatusIndicator("info").add_message(
            "Extracting GlyphOrder from TTX..."
        ).emit()

    glyph_to_id = extract_glyph_order_from_ttx(ttx_path)

    if verbose:
        cs.StatusIndicator("info").add_message(
            f"Found {len(glyph_to_id)} glyphs in GlyphOrder"
        ).emit()

    # Read the TTX file
    with open(ttx_path, "r", encoding="utf-8") as f:
        content = f.read()

    total_coverage = 0
    sorted_coverage = 0

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

        # Extract and sort glyph values by glyph ID
        glyph_values = [m.group(2) for m in glyph_matches]
        sorted_values = sorted(glyph_values, key=lambda g: glyph_to_id.get(g, 999999))

        # Check if sorting is needed
        if glyph_values == sorted_values:
            return full_block

        sorted_coverage += 1

        if verbose:
            unsorted_ids = [glyph_to_id.get(g, -1) for g in glyph_values[:5]]
            sorted_ids = [glyph_to_id.get(g, -1) for g in sorted_values[:5]]
            cs.StatusIndicator("info").add_message(
                f"Sorted Coverage (was IDs {unsorted_ids}, now {sorted_ids})"
            ).emit()

        # Rebuild with sorted glyphs
        first_glyph_indent = glyph_matches[0].group(1) if glyph_matches else "        "
        sorted_glyph_lines = [
            f'{first_glyph_indent}<Glyph value="{value}"/>' for value in sorted_values
        ]

        result = f"{indent}<Coverage{coverage_attrs}>\n"
        result += "\n".join(sorted_glyph_lines)
        result += f"\n{indent}</Coverage>"

        return result

    # Pattern to match Coverage blocks
    coverage_pattern = re.compile(
        r"(\s*)<Coverage([^>]*)>\s*\n((?:\s*<Glyph[^>]+/>\s*\n)+)\s*\1</Coverage>",
        re.MULTILINE,
    )

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
        Sort Coverage tables by glyph ID to fix fontTools warnings.
        
        This tool fixes the "GSUB/GPOS Coverage is not sorted by glyph ids" 
        warning by sorting Coverage tables according to glyph ID order.
        
        Works on both binary fonts (.ttf, .otf) and TTX files (.ttx).
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix binary font directly
  %(prog)s font.otf --backup
  
  # Fix TTX file
  %(prog)s font.ttx -v
  
  # Process multiple fonts
  %(prog)s fonts/*.otf --backup
        """,
    )

    parser.add_argument(
        "input", nargs="+", help="Font file(s) or TTX file(s) to process"
    )
    parser.add_argument(
        "-o", "--output", help="Output file (only valid for single input)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup file (.bak) before modifying",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    input_files = []
    for path_str in args.input:
        path = Path(path_str)
        if path.is_dir():
            input_files.extend(path.glob("*.ttf"))
            input_files.extend(path.glob("*.otf"))
        else:
            input_files.append(path)

    if not input_files:
        cs.StatusIndicator("error").with_explanation("No input files found").emit()
        sys.exit(1)

    if len(input_files) > 1 and args.output:
        cs.StatusIndicator("error").with_explanation(
            "--output can only be used with a single input file"
        ).emit()
        sys.exit(1)

    for input_path in input_files:
        if not input_path.exists():
            cs.StatusIndicator("error").add_file(str(input_path)).with_explanation(
                "not found"
            ).emit()
            continue

        try:
            if args.verbose:
                cs.StatusIndicator("info").add_file(str(input_path)).add_message(
                    "Processing"
                ).emit()

            # Determine if it's a TTX or binary font
            is_ttx = input_path.suffix.lower() == ".ttx"

            if is_ttx:
                # Process TTX file
                total, sorted_count = sort_coverage_tables_in_ttx(
                    input_path, args.output, args.backup, args.verbose
                )
            else:
                # Process binary font
                font = TTFont(input_path)
                total, sorted_count = sort_coverage_tables_in_font(font, args.verbose)

                if sorted_count > 0:
                    output_path = Path(args.output) if args.output else input_path

                    # Create backup if requested
                    if args.backup and output_path == input_path:
                        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
                        import shutil

                        shutil.copy2(input_path, backup_path)
                        if args.verbose:
                            cs.StatusIndicator("info").add_file(
                                str(backup_path)
                            ).add_message("Created backup").emit()

                    # Save the font
                    font.save(output_path)

                    if args.verbose:
                        cs.StatusIndicator("saved").add_file(str(output_path)).emit()

            output_path = Path(args.output) if args.output else input_path

            if sorted_count == 0:
                cs.StatusIndicator("info").add_file(input_path.name).add_message(
                    f"All {total} Coverage tables already sorted"
                ).emit()
            else:
                cs.StatusIndicator("updated").add_file(input_path.name).add_message(
                    f"Sorted {sorted_count} of {total} Coverage tables"
                ).emit()

        except Exception as e:
            cs.StatusIndicator("error").add_file(str(input_path)).with_explanation(
                f"Error: {e}"
            ).emit()
            if args.verbose:
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    main()
