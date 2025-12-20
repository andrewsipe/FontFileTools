#!/usr/bin/env python3
"""
UPM Rescaler

Changes a font's units-per-em (UPM) value and rescales all associated metrics
using fonttools' scaleUpem function.

Supported formats: TTF, OTF, WOFF, WOFF2 (binary formats only)

Limitations:
- AAT and Graphite tables are not supported by scaleUpem and will cause the
  font to be skipped with a warning
- CFF/CFF2 fonts will be de-subroutinized as a side effect

Usage:
    python Tools_UPM_Rescaler.py <new_upem> <font_path> [options]

    python Tools_UPM_Rescaler.py 2048 font.otf
    python Tools_UPM_Rescaler.py 1000 fonts/ -r --dry-run
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

try:
    from fontTools.ttLib import TTFont, TTLibError
    from fontTools.ttLib.scaleUpem import scale_upem
except ImportError:
    print("Error: fonttools is required. Install with: pip install fonttools")
    sys.exit(1)

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
# ruff: noqa: E402
_project_root = Path(__file__).parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import FontCore.core_console_styles as cs
from FontCore.core_file_collector import iter_font_files

SUPPORTED_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}

# AAT table tags
AAT_TABLES = {"morx", "mort"}

# Graphite table tags
GRAPHITE_TABLES = {"Silf", "Glat", "Gloc", "Sill"}


def has_aat_tables(font: TTFont) -> bool:
    """Check if font contains AAT tables."""
    return any(tag in font for tag in AAT_TABLES)


def has_graphite_tables(font: TTFont) -> bool:
    """Check if font contains Graphite tables."""
    return any(tag in font for tag in GRAPHITE_TABLES)


def is_cff_font(font: TTFont) -> bool:
    """Check if font uses CFF or CFF2 outlines."""
    return "CFF " in font or "CFF2" in font


def get_current_upem(font: TTFont) -> int:
    """Get current UPM value from font."""
    if "head" not in font:
        raise ValueError("Font missing 'head' table")
    return font["head"].unitsPerEm


def rescale_font_upem(
    font_path: Path,
    new_upem: int,
    dry_run: bool = False,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Rescale font UPM using fonttools scaleUpem.

    Args:
        font_path: Path to font file
        new_upem: New UPM value to apply
        dry_run: If True, don't modify files
        verbose: If True, show detailed information
        output_dir: Optional output directory (None = overwrite in place)

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        # Load font
        font = TTFont(str(font_path))

        # Get current UPM
        current_upem = get_current_upem(font)

        # Check if UPM is already the target value
        if current_upem == new_upem:
            cs.StatusIndicator("unchanged").add_file(str(font_path)).add_field(
                "UPM", current_upem
            ).with_explanation("UPM already matches target value").emit()
            return True, None

        # Check for AAT tables
        if has_aat_tables(font):
            error_msg = "Font contains AAT tables (not supported by scaleUpem)"
            cs.StatusIndicator("skipped").add_file(str(font_path)).with_explanation(
                error_msg
            ).emit()
            return False, error_msg

        # Check for Graphite tables
        if has_graphite_tables(font):
            error_msg = "Font contains Graphite tables (not supported by scaleUpem)"
            cs.StatusIndicator("skipped").add_file(str(font_path)).with_explanation(
                error_msg
            ).emit()
            return False, error_msg

        # Inform about CFF de-subroutinization if applicable
        cff_warning = None
        if is_cff_font(font):
            cff_warning = "CFF/CFF2 font will be de-subroutinized"
            if verbose:
                cs.StatusIndicator("info").add_file(str(font_path)).with_explanation(
                    cff_warning
                ).emit()

        # Show before/after info
        if verbose or dry_run:
            cs.StatusIndicator("info").add_file(str(font_path)).add_field(
                "Current UPM", current_upem
            ).add_field("Target UPM", new_upem).emit()

        # Perform rescaling if not dry-run
        if not dry_run:
            try:
                scale_upem(font, new_upem)
            except Exception as e:
                error_msg = f"scaleUpem failed: {e}"
                cs.StatusIndicator("error").add_file(str(font_path)).with_explanation(
                    error_msg
                ).emit()
                return False, error_msg

            # Determine output path
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / font_path.name
            else:
                output_path = font_path

            # Verify UPM was changed correctly (check before saving)
            new_current_upem = get_current_upem(font)
            if new_current_upem != new_upem:
                error_msg = f"UPM verification failed: expected {new_upem}, got {new_current_upem}"
                cs.StatusIndicator("error").add_file(str(font_path)).with_explanation(
                    error_msg
                ).emit()
                return False, error_msg

            # Save font (fonttools handles WOFF/WOFF2 automatically)
            try:
                font.save(str(output_path))
            except Exception as e:
                error_msg = f"Failed to save font: {e}"
                cs.StatusIndicator("error").add_file(str(font_path)).with_explanation(
                    error_msg
                ).emit()
                return False, error_msg

            # Success
            cs.StatusIndicator("updated").add_file(
                str(output_path) if output_dir else str(font_path)
            ).add_field("UPM", f"{current_upem} → {new_current_upem}").emit()
        else:
            # Dry-run mode
            cs.StatusIndicator("preview").add_file(str(font_path)).add_field(
                "Would change UPM", f"{current_upem} → {new_upem}"
            ).emit()
            if cff_warning:
                cs.StatusIndicator("info").with_explanation(cff_warning).emit()

        return True, None

    except TTLibError as e:
        error_msg = f"Font loading error: {e}"
        cs.StatusIndicator("error").add_file(str(font_path)).with_explanation(
            error_msg
        ).emit()
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        cs.StatusIndicator("error").add_file(str(font_path)).with_explanation(
            error_msg
        ).emit()
        return False, error_msg
    finally:
        # Ensure font is closed
        try:
            if "font" in locals():
                font.close()
        except Exception:
            pass


def validate_upem(upem_str: str) -> int:
    """Validate and convert UPM value."""
    try:
        upem = int(upem_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid UPM value: {upem_str} (must be an integer)"
        )

    if upem <= 0:
        raise argparse.ArgumentTypeError(f"UPM must be positive, got: {upem}")

    if upem > 65535:
        raise argparse.ArgumentTypeError(f"UPM too large (max 65535), got: {upem}")

    # Common UPM values are powers of 2, but allow any reasonable value
    if upem < 16:
        raise argparse.ArgumentTypeError(f"UPM too small (min 16), got: {upem}")

    return upem


def collect_font_files(paths: list[str], recursive: bool = False) -> list[str]:
    """Collect font files from paths."""
    font_files = []

    for path_str in paths:
        path_obj = Path(path_str).expanduser().resolve()

        if path_obj.is_file():
            if path_obj.suffix.lower() in SUPPORTED_EXTENSIONS:
                font_files.append(str(path_obj))
            else:
                cs.StatusIndicator("warning").add_file(str(path_obj)).with_explanation(
                    "Not a supported font file"
                ).emit()
        elif path_obj.is_dir():
            # Use iter_font_files for consistent behavior
            for font_path in iter_font_files(
                paths=[path_obj],
                recursive=recursive,
                allowed_extensions=SUPPORTED_EXTENSIONS,
                include_uppercase=True,
            ):
                font_files.append(font_path)
        else:
            cs.StatusIndicator("warning").add_file(str(path_obj)).with_explanation(
                "Path does not exist"
            ).emit()

    return sorted(set(font_files))


def main() -> None:
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description="Rescale font UPM (units-per-em) and all associated metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Rescale single font to 2048 UPM:
    %(prog)s 2048 font.otf

  Rescale all fonts in directory recursively:
    %(prog)s 1000 fonts/ -r

  Preview changes without modifying files:
    %(prog)s 2048 fonts/ -r --dry-run

  Save to different directory:
    %(prog)s 2048 fonts/ -r -o output/

LIMITATIONS:
  - AAT and Graphite tables are not supported (fonts will be skipped)
  - CFF/CFF2 fonts will be de-subroutinized as a side effect
  - Only binary formats supported (TTF, OTF, WOFF, WOFF2)

For more information, see:
  https://fonttools.readthedocs.io/en/latest/ttLib/scaleUpem.html
        """,
    )

    parser.add_argument(
        "new_upem",
        type=validate_upem,
        help="New UPM value (typically 1000 or 2048, range: 16-65535)",
    )

    parser.add_argument(
        "paths",
        nargs="+",
        help="Font files or directories to process",
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory for processed fonts (default: overwrite in place)",
    )

    args = parser.parse_args()

    # Check fonttools availability
    try:
        from fontTools.ttLib import TTFont
        from fontTools.ttLib.scaleUpem import scale_upem
    except ImportError:
        cs.StatusIndicator("error").with_explanation(
            "fonttools is required. Install with: pip install fonttools"
        ).emit()
        sys.exit(1)

    # Collect font files
    font_files = collect_font_files(args.paths, args.recursive)

    if not font_files:
        cs.StatusIndicator("error").with_explanation(
            "No font files found to process"
        ).emit()
        sys.exit(1)

    # Show operation summary
    cs.StatusIndicator("info").add_message(
        f"Found {cs.fmt_count(len(font_files))} font file(s) to process"
    ).emit()
    cs.StatusIndicator("info").add_field("Target UPM", args.new_upem).emit()

    if args.dry_run:
        cs.StatusIndicator("warning").with_explanation(
            "Dry run mode - no changes will be made"
        ).emit()

    if args.verbose:
        cs.StatusIndicator("info").with_explanation("Verbose mode enabled").emit()

    # Process fonts
    success_count = 0
    skipped_count = 0
    error_count = 0

    # Use progress bar for batch processing
    console = cs.get_console()
    progress = None
    task = None

    if len(font_files) > 1 and console:
        progress = cs.create_progress_bar(console=console)
        task = progress.add_task(
            "Processing fonts...",
            total=len(font_files),
        )
        progress.start()

    try:
        for i, font_path_str in enumerate(font_files, 1):
            font_path = Path(font_path_str)

            if progress and task is not None:
                progress.update(
                    task,
                    description=f"Processing fonts... [{i}/{len(font_files)}]",
                    completed=i,
                )

            success, error_msg = rescale_font_upem(
                font_path=font_path,
                new_upem=args.new_upem,
                dry_run=args.dry_run,
                verbose=args.verbose,
                output_dir=args.output_dir,
            )

            if success:
                success_count += 1
            elif error_msg:
                if "AAT" in error_msg or "Graphite" in error_msg:
                    skipped_count += 1
                else:
                    error_count += 1
            else:
                error_count += 1

    finally:
        if progress:
            progress.stop()

    # Summary
    cs.emit("")
    cs.StatusIndicator("info").add_message("Processing complete").emit()
    cs.StatusIndicator("info").add_field(
        "Processed", cs.fmt_count(success_count)
    ).emit()

    if skipped_count > 0:
        cs.StatusIndicator("info").add_field(
            "Skipped", cs.fmt_count(skipped_count)
        ).emit()

    if error_count > 0:
        cs.StatusIndicator("error").add_field(
            "Errors", cs.fmt_count(error_count)
        ).emit()

    # Exit with error code if any failures
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
