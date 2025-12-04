#!/usr/bin/env python3
"""
Font Advance Width Scanner and Fixer

This script scans font files for glyphs with problematic advance widths
(likely corrupted negative values showing as large positive numbers) and
provides options to fix them.

Usage:
    python Tools_AdvancedWidth_DetectandFix.py /path/to/fonts/directory
    python Tools_AdvancedWidth_DetectandFix.py single_font.ttf

Requirements:
    pip install fonttools
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import shutil
import logging

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
_project_root = Path(__file__).parent
while not (_project_root / "FontCore").exists() and _project_root.parent != _project_root:
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import FontCore.core_console_styles as cs
from FontCore.core_file_collector import iter_font_files

try:
    from fontTools.ttLib import TTFont
    from fontTools.pens.boundsPen import BoundsPen
except ImportError:
    cs.StatusIndicator("error").with_explanation(
        "fonttools is required. Install with: pip install fonttools"
    ).emit()
    sys.exit(1)

# Suppress fontTools warnings about advance widths during load
logging.getLogger("fontTools.ttLib.tables._h_m_t_x").setLevel(logging.ERROR)


class FontAdvanceWidthFixer:
    def __init__(self, threshold: int = 32000):
        # Suspicious advance width threshold (likely corrupted negative values)
        self.SUSPICIOUS_THRESHOLD = (
            threshold  # Base threshold, may be overridden by stats
        )
        self.MAX_REASONABLE_WIDTH = 10000  # Maximum reasonable advance width

        # Common problematic glyphs and their typical characteristics (fallback only)
        self.GLYPH_RECOMMENDATIONS = {
            "fraction": {"typical_width": 200, "description": "fraction slash"},
            "Dotaccent": {
                "typical_width": 250,
                "description": "dot accent mark (spacing)",
            },
            "dotaccent": {
                "typical_width": 250,
                "description": "dot accent mark (spacing)",
            },
            "hungarumlaut": {
                "typical_width": 350,
                "description": "double acute accent (spacing)",
            },
            "ogonek": {"typical_width": 250, "description": "ogonek mark (spacing)"},
            "caron": {"typical_width": 300, "description": "caron mark (spacing)"},
            "breve": {"typical_width": 300, "description": "breve mark (spacing)"},
            "ring": {"typical_width": 250, "description": "ring accent (spacing)"},
            "macron": {"typical_width": 350, "description": "macron mark (spacing)"},
            "circumflex": {
                "typical_width": 300,
                "description": "circumflex accent (spacing)",
            },
            "tilde": {"typical_width": 300, "description": "tilde accent (spacing)"},
            "dieresis": {
                "typical_width": 300,
                "description": "dieresis/umlaut (spacing)",
            },
            "acute": {"typical_width": 250, "description": "acute accent (spacing)"},
            "grave": {"typical_width": 250, "description": "grave accent (spacing)"},
            "cedilla": {"typical_width": 250, "description": "cedilla mark (spacing)"},
        }

    def scan_font(
        self, font_path: Path
    ) -> Tuple[Dict[str, int], Dict[str, Union[str, int]]]:
        """
        Scan a font file for problematic advance widths using statistical analysis.

        Returns:
            Tuple of (problematic_glyphs_dict, font_info_dict)
        """
        try:
            font = TTFont(str(font_path))
            problematic_glyphs = {}

            # Get horizontal metrics table
            if "hmtx" not in font:
                font.close()
                return {}, {"error": "No horizontal metrics table found"}

            hmtx = font["hmtx"]
            glyph_set = font.getGlyphSet()

            # Collect all advance widths for statistical analysis
            all_widths = []
            glyph_widths = {}

            for glyph_name in glyph_set.keys():
                if glyph_name in hmtx.metrics:
                    advance_width, _ = hmtx.metrics[glyph_name]
                    all_widths.append(advance_width)
                    glyph_widths[glyph_name] = advance_width

            # Calculate statistics (excluding obvious outliers)
            valid_widths = [w for w in all_widths if 0 < w < 10000]
            if valid_widths:
                sorted_widths = sorted(valid_widths)
                median_width = sorted_widths[len(sorted_widths) // 2]
                avg_width = sum(valid_widths) / len(valid_widths)

                # Dynamic threshold: max of (median * 10, base threshold, 65535 trigger)
                # 65535 is likely unsigned 16-bit overflow (-1 stored as 0xFFFF)
                # Dynamic threshold: Use median * 10, but flag anything >= 65535 separately
                suspicious_threshold = max(median_width * 10, self.SUSPICIOUS_THRESHOLD)
            else:
                # Fallback if no valid widths found
                median_width = 0
                avg_width = 0
                suspicious_threshold = self.SUSPICIOUS_THRESHOLD

            # Flag problematic glyphs
            for glyph_name, width in glyph_widths.items():
                if width > suspicious_threshold or width >= 32767:
                    problematic_glyphs[glyph_name] = width

            units_per_em = font["head"].unitsPerEm if "head" in font else 1000

            font_info = {
                "family_name": self._get_font_name(font),
                "total_glyphs": len(glyph_set),
                "units_per_em": units_per_em,
                "median_width": median_width if valid_widths else "Unknown",
                "avg_width": int(avg_width) if valid_widths else "Unknown",
                "threshold_used": suspicious_threshold,
            }

            font.close()
            return problematic_glyphs, font_info

        except Exception as e:
            return {}, {"error": str(e)}

    def _get_font_name(self, font: TTFont) -> str:
        """Extract font family name from font tables."""
        if "name" not in font:
            return "Unknown"

        name_table = font["name"]
        for record in name_table.names:
            if record.nameID == 1:  # Family name
                try:
                    return record.toUnicodeString()
                except Exception:
                    continue
        return "Unknown"

    def _is_combining_mark(self, glyph_name: str) -> bool:
        """Check if glyph is a combining diacritical mark."""
        combining_prefixes = [
            "uni03",  # Combining diacriticals Unicode range
            "uni04",  # Combining diacriticals Unicode range
            "gravecomb",
            "acutecomb",
            "tildecomb",
            "macroncomb",
            "dotaccentcomb",
            "dieresiscomb",
            "caroncomb",
            "brevecomb",
            "ringcomb",
            "cedillacomb",
            "ogonekcomb",
        ]

        combining_names = {
            "Dotaccent",
            "dotaccent",
            "hungarumlaut",
            "ogonek",
            "caron",
            "breve",
            "ring",
            "macron",
            "grave",
            "acute",
            "circumflex",
            "tilde",
            "dieresis",
            "cedilla",
        }

        # Check if it ends with 'comb' (combining marks always end with this)
        if glyph_name.endswith("comb"):
            return True

        # Check prefixes
        if any(glyph_name.startswith(prefix) for prefix in combining_prefixes):
            return True

        # Check if it's in the combining names list BUT not a spacing modifier
        if glyph_name in combining_names:
            # Make sure it's not a spacing modifier version
            return not self._is_spacing_modifier(glyph_name)

        return False

    def _is_zero_width_character(self, glyph_name: str) -> bool:
        """Characters that should always be zero-width."""
        zero_width_chars = {
            "zerowidthspace",
            "uni200B",  # Zero-width space
            "zerowidthnonjoiner",
            "uni200C",  # ZWNJ
            "zerowidthjoiner",
            "uni200D",  # ZWJ
            "zerowidthnobreakspace",
            "uniFEFF",  # Zero-width no-break space
            "softhyphen",
            "uni00AD",  # Soft hyphen (often zero-width)
        }
        return glyph_name in zero_width_chars

    def _is_control_character(self, glyph_name: str) -> bool:
        """Control characters should be zero-width."""
        # Unicode control characters (U+0000-U+001F, U+007F-U+009F)
        if glyph_name.startswith("uni00") and len(glyph_name) == 7:
            try:
                code_point = glyph_name[3:7]
                code_int = int(code_point, 16)
                if code_int <= 0x1F or (0x7F <= code_int <= 0x9F):
                    return True
            except ValueError:
                pass
        return glyph_name in {"NULL", "CR", "nonmarkingreturn"}

    def _is_spacing_modifier(self, glyph_name: str) -> bool:
        """Check if glyph is a spacing modifier letter (NOT zero-width).

        These look like combining marks but are standalone spacing characters.
        """
        spacing_modifiers = {
            "circumflex",
            "caron",
            "breve",
            "dotaccent",
            "ring",
            "tilde",
            "macron",
            "dieresis",
            "acute",
            "grave",
            "cedilla",
            "ogonek",
            "hungarumlaut",
        }

        # Spacing modifiers have these names but DON'T end in 'comb'
        return glyph_name in spacing_modifiers and not glyph_name.endswith("comb")

    def _get_unicode_category(self, glyph_name: str) -> Optional[str]:
        """Get Unicode category if glyph name is in uni notation."""
        if glyph_name.startswith("uni") and len(glyph_name) >= 7:
            try:
                import unicodedata

                # Handle both uni0000 (4 hex digits) and uni00000 (5 hex digits)
                hex_digits = (
                    glyph_name[3:7] if len(glyph_name) == 7 else glyph_name[3:8]
                )
                code_point = int(hex_digits, 16)
                char = chr(code_point)
                return unicodedata.category(char)
            except (ValueError, OverflowError):
                pass
        return None

    def _glyph_bounds_width(self, font: TTFont, glyph_name: str) -> Optional[int]:
        """Calculate advance width based on actual glyph geometry."""
        try:
            glyph_set = font.getGlyphSet()
            if glyph_name not in glyph_set:
                return None

            pen = BoundsPen(glyph_set)
            glyph_set[glyph_name].draw(pen)

            if pen.bounds is None:
                return None

            xMin, yMin, xMax, yMax = pen.bounds
            bbox_width = xMax - xMin

            # Add ~15% padding for side bearings
            return int(bbox_width * 1.15)

        except Exception:
            return None

    def _infer_base_glyph(self, glyph_name: str) -> Optional[str]:
        """Try to infer base glyph name from accented glyph name."""
        # Common patterns: remove combining marks
        # e.g., "Aacute" -> "A", "agrave" -> "a"
        common_marks = [
            "acute",
            "grave",
            "circumflex",
            "tilde",
            "macron",
            "breve",
            "dotaccent",
            "dieresis",
            "ring",
            "caron",
            "hungarumlaut",
            "ogonek",
        ]

        for mark in common_marks:
            if glyph_name.endswith(mark):
                base = glyph_name[: -len(mark)]
                if base:
                    return base

        return None

    def calculate_recommended_width(
        self,
        font: TTFont,
        glyph_name: str,
        font_units_per_em: int,
        font_median_width: Optional[int] = None,
    ) -> int:
        """
        Calculate a recommended advance width using multiple strategies.

        Args:
            font: TTFont object for geometry access
            glyph_name: Name of the glyph
            font_units_per_em: Units per em of the font
            font_median_width: Median width of font (for fallback)

        Returns:
            Recommended advance width
        """
        # Strategy 0: Zero-width characters (highest priority)
        if self._is_zero_width_character(glyph_name):
            return 0

        # Strategy 0.5: Control characters should be zero-width
        if self._is_control_character(glyph_name):
            return 0

        # Strategy 1: Combining marks should be zero-width
        if self._is_combining_mark(glyph_name):
            return 0

        # Strategy 1.5: Spacing modifiers need actual width (NOT zero)
        if self._is_spacing_modifier(glyph_name):
            geometric_width = self._glyph_bounds_width(font, glyph_name)
            if geometric_width is not None:
                return max(100, min(geometric_width, self.MAX_REASONABLE_WIDTH))
            # Fallback for spacing modifiers
            return 300

        # Strategy 2: Special case for fraction slash
        if glyph_name in {"fraction", "uni2044"}:
            hmtx = font.get("hmtx")
            if hmtx and "slash" in hmtx.metrics:
                slash_width, _ = hmtx.metrics["slash"]
                if 0 < slash_width < 10000:
                    # Fraction slash is typically ~35% the width of regular slash
                    return int(slash_width * 0.35)
            # Fallback
            return int(font_median_width * 0.3) if font_median_width else 200

        # Strategy 3: Try to calculate from actual glyph geometry
        geometric_width = self._glyph_bounds_width(font, glyph_name)
        if geometric_width is not None:
            return max(50, min(geometric_width, self.MAX_REASONABLE_WIDTH))

        # Strategy 4: Look for similar glyphs (base glyph for accented characters)
        base_glyph = self._infer_base_glyph(glyph_name)
        if base_glyph:
            hmtx = font.get("hmtx")
            if hmtx and base_glyph in hmtx.metrics:
                base_width, _ = hmtx.metrics[base_glyph]
                if 0 < base_width < 10000:
                    return base_width

        # Strategy 5: Use Unicode category as fallback
        unicode_category = self._get_unicode_category(glyph_name)
        if unicode_category:
            # Mn = Mark, nonspacing; Cf = Format; Cc = Control
            if unicode_category in {"Mn", "Cf", "Cc"}:
                return 0
            # Zs = Space separator - use space width from font
            if unicode_category == "Zs":
                hmtx = font.get("hmtx")
                if hmtx and "space" in hmtx.metrics:
                    space_width, _ = hmtx.metrics["space"]
                    if 0 < space_width < 10000:
                        return space_width

        # Strategy 6: Use median width of font as fallback
        if font_median_width and font_median_width > 0:
            return max(50, min(font_median_width, self.MAX_REASONABLE_WIDTH))

        # Strategy 7: Use hardcoded recommendations (last resort)
        base_recommendation = 300
        if glyph_name.lower() in self.GLYPH_RECOMMENDATIONS:
            base_recommendation = self.GLYPH_RECOMMENDATIONS[glyph_name.lower()][
                "typical_width"
            ]

        # Scale based on units per em (assuming 1000 as baseline)
        if isinstance(font_units_per_em, int) and font_units_per_em > 0:
            scaled_width = int(base_recommendation * (font_units_per_em / 1000))
            return max(50, min(scaled_width, self.MAX_REASONABLE_WIDTH))

        return base_recommendation

    def fix_font(
        self,
        font_path: Path,
        problematic_glyphs: Dict[str, int],
        font_info: Dict[str, Union[str, int]],
        custom_widths: Optional[Dict[str, int]] = None,
        backup: bool = True,
    ) -> bool:
        """
        Fix problematic advance widths in a font file.

        Args:
            font_path: Path to the font file
            problematic_glyphs: Dictionary of glyph_name -> current_width
            font_info: Font information dict (includes median_width, units_per_em)
            custom_widths: Optional dictionary of glyph_name -> desired_width
            backup: Whether to create a backup of the original file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup if requested
            if backup:
                backup_path = font_path.with_suffix(font_path.suffix + ".backup")
                shutil.copy2(font_path, backup_path)
                cs.StatusIndicator("saved").add_file(backup_path.name).with_explanation(
                    "Backup created"
                ).emit()

            font = TTFont(str(font_path))
            hmtx = font["hmtx"]
            units_per_em = (
                font_info.get("units_per_em")
                if isinstance(font_info.get("units_per_em"), int)
                else (font["head"].unitsPerEm if "head" in font else 1000)
            )
            median_width = (
                font_info.get("median_width")
                if isinstance(font_info.get("median_width"), int)
                else None
            )

            fixed_count = 0
            for glyph_name, current_width in problematic_glyphs.items():
                if glyph_name in hmtx.metrics:
                    # Get current metrics
                    advance_width, left_side_bearing = hmtx.metrics[glyph_name]

                    # Determine new width
                    if custom_widths and glyph_name in custom_widths:
                        new_width = custom_widths[glyph_name]
                    else:
                        new_width = self.calculate_recommended_width(
                            font, glyph_name, units_per_em, median_width
                        )

                    # Update metrics
                    hmtx.metrics[glyph_name] = (new_width, left_side_bearing)
                    fixed_count += 1

                    cs.StatusIndicator("updated").add_values(
                        old_value=str(current_width), new_value=str(new_width)
                    ).add_field("glyph", glyph_name).emit()

            # Save the modified font
            font.save(str(font_path))
            font.close()

            cs.StatusIndicator("updated").add_file(font_path.name).add_field(
                "fixed_glyphs", fixed_count
            ).emit()
            return True

        except Exception as e:
            cs.StatusIndicator("error").add_file(font_path.name).with_explanation(
                f"Error fixing: {e}"
            ).emit()
            return False

    def scan_directory(
        self, directory_path: Path, recursive: bool = False
    ) -> Dict[Path, Tuple[Dict[str, int], Dict[str, Union[str, int]]]]:
        """
        Scan all font files in a directory using core_file_collector with progress.

        Returns:
            Dictionary mapping font paths to (problematic_glyphs, font_info)
        """
        results = {}
        font_files: List[str] = []

        # Collect font files with progress bar
        console = cs.get_console()
        show_progress = console and cs.RICH_AVAILABLE

        if show_progress:
            progress = cs.create_progress_bar()
            task = progress.add_task("Collecting font files...", total=None)
            progress.start()

            try:
                for file_path in iter_font_files(
                    paths=[directory_path],
                    recursive=recursive,
                    allowed_extensions={".ttf", ".otf"},
                    on_progress=lambda info: progress.update(
                        task,
                        description=f"Scanning directories... ({info.get('matches_found', 0)} fonts found)",
                    ),
                ):
                    font_files.append(file_path)
            finally:
                progress.stop()

            # Now scan collected files with progress
            if font_files:
                task = progress.add_task("Analyzing fonts...", total=len(font_files))
                progress.start()

                try:
                    for font_path_str in font_files:
                        font_path = Path(font_path_str)
                        problematic_glyphs, font_info = self.scan_font(font_path)
                        if problematic_glyphs or "error" in font_info:
                            results[font_path] = (problematic_glyphs, font_info)
                        progress.update(task, advance=1)
                finally:
                    progress.stop()
        else:
            # Fallback without progress bar
            for file_path in iter_font_files(
                paths=[directory_path],
                recursive=recursive,
                allowed_extensions={".ttf", ".otf"},
            ):
                font_path = Path(file_path)
                problematic_glyphs, font_info = self.scan_font(font_path)
                if problematic_glyphs or "error" in font_info:
                    results[font_path] = (problematic_glyphs, font_info)

        return results

    def interactive_fix(
        self,
        font_path: Path,
        problematic_glyphs: Dict[str, int],
        font_info: Dict[str, Union[str, int]],
    ):
        """
        Interactive fixing interface for a single font.
        """
        if "error" in font_info:
            cs.StatusIndicator("error").with_explanation(font_info["error"]).emit()
            return

        cs.StatusIndicator("info").add_message(
            f"Font: {font_info.get('family_name', 'Unknown')}"
        ).emit()
        cs.StatusIndicator("info").add_file(font_path.name).emit()
        cs.StatusIndicator("info").add_field(
            "Units per EM", font_info.get("units_per_em", "Unknown")
        ).emit()
        cs.StatusIndicator("info").add_field(
            "Problematic glyphs found", len(problematic_glyphs)
        ).emit()

        if not problematic_glyphs:
            cs.StatusIndicator("info").add_message(
                "No problematic glyphs found!"
            ).emit()
            return

        # Load font for width calculation
        try:
            font = TTFont(str(font_path))
        except Exception as e:
            cs.StatusIndicator("error").with_explanation(
                f"Failed to load font: {e}"
            ).emit()
            return

        # Display problematic glyphs with recommendations
        cs.StatusIndicator("warning").add_message("Problematic glyphs:").emit()
        custom_widths = {}
        units_per_em = (
            font_info.get("units_per_em")
            if isinstance(font_info.get("units_per_em"), int)
            else 1000
        )
        median_width = (
            font_info.get("median_width")
            if isinstance(font_info.get("median_width"), int)
            else None
        )

        for glyph_name, current_width in problematic_glyphs.items():
            recommended = self.calculate_recommended_width(
                font, glyph_name, units_per_em, median_width
            )
            description = ""
            if glyph_name.lower() in self.GLYPH_RECOMMENDATIONS:
                description = f" ({self.GLYPH_RECOMMENDATIONS[glyph_name.lower()]['description']})"

            cs.StatusIndicator("warning").add_field(
                "glyph", f"{glyph_name}{description}"
            ).add_values(
                old_value=str(current_width), new_value=f"recommended: {recommended}"
            ).emit()

        font.close()

        cs.StatusIndicator("info").add_message("Options:").emit()
        cs.StatusIndicator("info").add_item("1. Fix all with recommended values").emit()
        cs.StatusIndicator("info").add_item("2. Fix with custom values").emit()
        cs.StatusIndicator("info").add_item("3. Skip this font").emit()

        choice = input("Choose option (1-3): ").strip()

        if choice == "1":
            self.fix_font(font_path, problematic_glyphs, font_info, backup=True)
        elif choice == "2":
            cs.StatusIndicator("info").add_message(
                "Enter custom widths (press Enter for recommended value):"
            ).emit()
            # Reload font for recommendations
            font = TTFont(str(font_path))
            for glyph_name, current_width in problematic_glyphs.items():
                recommended = self.calculate_recommended_width(
                    font, glyph_name, units_per_em, median_width
                )
                custom_input = input(
                    f"  {glyph_name} (recommended {recommended}): "
                ).strip()

                if custom_input:
                    try:
                        custom_widths[glyph_name] = int(custom_input)
                    except ValueError:
                        print(
                            f"    Invalid input, using recommended value: {recommended}"
                        )
                        custom_widths[glyph_name] = recommended
                else:
                    custom_widths[glyph_name] = recommended
            font.close()

            self.fix_font(
                font_path, problematic_glyphs, font_info, custom_widths, backup=True
            )
        else:
            cs.StatusIndicator("skipped").add_file(font_path.name).emit()


def main():
    parser = argparse.ArgumentParser(
        description="Scan and fix problematic advance widths in font files"
    )
    parser.add_argument("path", help="Path to font file or directory containing fonts")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive mode (prompt for each font). Default is auto-fix.",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Don't create backup files when fixing"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=32000,
        help="Base threshold for suspicious advance widths (default: 32000). "
        "Actual threshold may be higher based on font statistics.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories (directory mode only)",
    )

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        cs.StatusIndicator("error").with_explanation(
            f"Path '{path}' does not exist"
        ).emit()
        sys.exit(1)

    fixer = FontAdvanceWidthFixer(threshold=args.threshold)

    if path.is_file():
        # Single font file
        cs.StatusIndicator("info").add_message("Scanning single font").add_file(
            path.name
        ).emit()
        problematic_glyphs, font_info = fixer.scan_font(path)

        if problematic_glyphs:
            if args.interactive:
                fixer.interactive_fix(path, problematic_glyphs, font_info)
            else:
                # Auto-fix by default
                cs.StatusIndicator("info").add_message(
                    f"Found {len(problematic_glyphs)} problematic glyph(s)"
                ).emit()
                fixer.fix_font(
                    path, problematic_glyphs, font_info, backup=not args.no_backup
                )
        else:
            if "error" in font_info:
                cs.StatusIndicator("error").with_explanation(font_info["error"]).emit()
            else:
                cs.StatusIndicator("info").add_message(
                    "No problematic glyphs found"
                ).add_file(path.name).emit()

    elif path.is_dir():
        # Directory of fonts
        cs.StatusIndicator("info").add_message("Scanning directory").add_file(
            str(path)
        ).emit()
        results = fixer.scan_directory(path, recursive=args.recursive)

        if not results:
            cs.StatusIndicator("info").add_message(
                "No fonts with problematic advance widths found"
            ).emit()
            return

        cs.StatusIndicator("info").add_message(
            f"Found {len(results)} font(s) with issues"
        ).emit()

        if args.interactive:
            # Interactive mode
            for font_path, (problematic_glyphs, font_info) in results.items():
                fixer.interactive_fix(font_path, problematic_glyphs, font_info)

                if len(results) > 1:
                    continue_choice = (
                        input("\nContinue to next font? (y/n): ").strip().lower()
                    )
                    if continue_choice != "y":
                        break
        else:
            # Auto-fix all fonts by default
            fixed_count = 0
            error_count = 0

            for font_path, (problematic_glyphs, font_info) in results.items():
                if problematic_glyphs and "error" not in font_info:
                    cs.StatusIndicator("info").add_message("Fixing").add_file(
                        font_path.name
                    ).emit()
                    if fixer.fix_font(
                        font_path,
                        problematic_glyphs,
                        font_info,
                        backup=not args.no_backup,
                    ):
                        fixed_count += 1
                    else:
                        error_count += 1
                elif "error" in font_info:
                    error_count += 1

            # Summary
            cs.StatusIndicator("success").add_message(
                "Processing complete"
            ).with_summary_block(
                updated=fixed_count,
                unchanged=len(results) - fixed_count - error_count,
                errors=error_count,
            ).emit()

    else:
        cs.StatusIndicator("error").with_explanation(
            f"'{path}' is neither a file nor a directory"
        ).emit()
        sys.exit(1)


if __name__ == "__main__":
    main()
