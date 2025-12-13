#!/usr/bin/env python3
"""
Simple Font Vendor ID Sorter
Scans font files alphabetically, extracts OS/2 achVendID, and moves them to directories as encountered.
"""

import shutil
import argparse
from pathlib import Path
import importlib.util
import re

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
import sys
from pathlib import Path as PathLib

# ruff: noqa: E402
_project_root = PathLib(__file__).parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import FontCore.core_console_styles as cs


def get_vendor_id(font_path):
    """Extract achVendID robustly and return sanitized vendor id or codes 'UKWN'/'ERROR'."""
    font = None
    coll = None
    try:
        suffix = font_path.suffix.lower()
        # Use TTCollection for .ttc/.otc
        if suffix in (".ttc", ".otc"):
            from fontTools.ttLib import TTCollection

            coll = TTCollection(str(font_path))
            # policy: use first member
            font = coll[0]
        else:
            from fontTools.ttLib import TTFont

            font = TTFont(str(font_path))

        os2_table = font.get("OS/2") if font is not None else None

        if not os2_table:
            vendor = "UKWN"
        else:
            raw = getattr(os2_table, "achVendID", None)
            if raw is None:
                vendor = "UKWN"
            else:
                if isinstance(raw, bytes):
                    vendor = raw.decode("latin-1", "ignore")
                else:
                    vendor = str(raw)
                vendor = vendor.replace("\x00", "").strip()
                if not vendor:
                    vendor = "UKWN"

        # sanitize vendor for safe directory name: replace unsafe chars with '_'
        vendor = "".join(c for c in vendor if c.isprintable())
        vendor = re.sub(r"[^A-Za-z0-9 ._-]", "_", vendor).strip()
        vendor = re.sub(r"\s+", "_", vendor)
        if not vendor:
            vendor = "UKWN"

        return vendor

    except Exception as e:
        cs.StatusIndicator("error").add_file(str(font_path)).with_explanation(
            f"Error reading: {e}"
        ).emit()
        return "ERROR"  # Error reading file

    finally:
        try:
            if coll is not None and hasattr(coll, "close"):
                coll.close()
            elif font is not None and hasattr(font, "close"):
                font.close()
        except Exception:
            pass


def is_font_file(file_path):
    """Check if file is a supported font format."""
    font_extensions = {".ttf", ".otf", ".ttc", ".otc", ".woff", ".woff2"}
    return file_path.suffix.lower() in font_extensions


def sort_fonts(source_dir):
    """
    Sort fonts by vendor ID in alphabetical order.

    Args:
        source_dir: Directory containing font files (non-recursive)
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        cs.StatusIndicator("error").add_file(source_dir).with_explanation(
            "Source directory does not exist"
        ).emit()
        return

    # Get all font files in directory (not recursive), ignore hidden files
    font_files = [
        f
        for f in source_path.iterdir()
        if f.is_file() and not f.name.startswith(".") and is_font_file(f)
    ]

    if not font_files:
        cs.StatusIndicator("info").add_message(
            "No font files found in the directory"
        ).emit()
        return

    # Sort files alphabetically
    font_files.sort(key=lambda x: x.name.lower())

    cs.StatusIndicator("info").add_message(
        f"Found {len(font_files)} font files in: {source_path}"
    ).emit()
    cs.StatusIndicator("info").add_message("Processing files alphabetically...").emit()
    cs.emit("-" * 50)

    processed = 0
    errors = 0

    for font_file in font_files:
        processed += 1
        cs.StatusIndicator("info").add_file(font_file.name).add_message(
            "Processing"
        ).emit()

        # Get vendor ID
        vendor_id = get_vendor_id(font_file)
        cs.StatusIndicator("info").add_field("Vendor ID", vendor_id).emit()

        # Track errors for summary
        if vendor_id == "ERROR":
            errors += 1

        # Create vendor directory if it doesn't exist
        vendor_dir = source_path / vendor_id
        if not vendor_dir.exists():
            vendor_dir.mkdir(parents=True, exist_ok=True)
            cs.StatusIndicator("info").add_message(
                f"Created directory: {vendor_id}/"
            ).emit()

        # Move the file
        dest_path = vendor_dir / font_file.name

        # Handle filename conflicts
        counter = 1
        while dest_path.exists():
            name_parts = font_file.stem, counter, font_file.suffix
            dest_path = vendor_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
            counter += 1

        try:
            shutil.move(str(font_file), str(dest_path))
            cs.StatusIndicator("updated").add_message(
                f"Moved to {vendor_id}/{dest_path.name}"
            ).emit()
        except Exception as e:
            cs.StatusIndicator("error").add_file(font_file.name).with_explanation(
                f"Error moving: {e}"
            ).emit()
            # If move fails, try to move to ERROR directory instead
            try:
                error_dir = source_path / "ERROR"
                if not error_dir.exists():
                    error_dir.mkdir(parents=True, exist_ok=True)
                    cs.StatusIndicator("info").add_message(
                        "Created directory: ERROR/"
                    ).emit()
                error_dest = error_dir / font_file.name
                counter = 1
                while error_dest.exists():
                    name_parts = font_file.stem, counter, font_file.suffix
                    error_dest = (
                        error_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                    )
                    counter += 1
                shutil.move(str(font_file), str(error_dest))
                cs.StatusIndicator("updated").add_message(
                    f"Moved to ERROR/{error_dest.name} due to move failure"
                ).emit()
            except Exception as move_error:
                cs.StatusIndicator("error").with_explanation(
                    f"Critical error: Could not move file anywhere: {move_error}"
                ).emit()
            errors += 1

    # Print summary
    cs.emit("\n" + "=" * 50)
    cs.StatusIndicator("info").add_message("SUMMARY").emit()
    cs.emit("=" * 50)
    cs.StatusIndicator("info").add_field("Total font files processed", processed).emit()
    cs.StatusIndicator("info").add_field("Files with errors", errors).emit()

    # Show created directories
    vendor_dirs = [
        d for d in source_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    if vendor_dirs:
        cs.StatusIndicator("info").add_message(
            f"Created {len(vendor_dirs)} directories:"
        ).emit()
        for vendor_dir in sorted(vendor_dirs):
            file_count = len([f for f in vendor_dir.iterdir() if f.is_file()])
            if vendor_dir.name == "ERROR":
                cs.StatusIndicator("warning").add_file(vendor_dir.name).add_field(
                    "files", f"{file_count} (review needed)"
                ).emit()
            else:
                cs.StatusIndicator("info").add_file(vendor_dir.name).add_field(
                    "files", file_count
                ).emit()


def main():
    parser = argparse.ArgumentParser(
        description="Sort font files by OS/2 achVendID (alphabetical processing)"
    )
    parser.add_argument("source_dir", help="Directory containing font files to sort")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Show installation instructions for dependencies",
    )

    args = parser.parse_args()

    if args.install_deps:
        cs.StatusIndicator("info").add_message(
            "To install required dependencies, run:"
        ).emit()
        cs.StatusIndicator("info").add_message("pip install fonttools").emit()
        return

    if importlib.util.find_spec("fontTools.ttLib") is None:
        cs.StatusIndicator("error").with_explanation(
            "fonttools library not found"
        ).emit()
        cs.StatusIndicator("info").add_message(
            "Install it with: pip install fonttools"
        ).emit()
        cs.StatusIndicator("info").add_message(
            "Or run this script with --install-deps to see installation instructions"
        ).emit()
        return

    sort_fonts(args.source_dir)


if __name__ == "__main__":
    main()
