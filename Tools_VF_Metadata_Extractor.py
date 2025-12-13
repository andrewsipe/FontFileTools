#!/usr/bin/env python3
"""
Variable Font Metadata Extraction & Analysis Tool

Extract comprehensive naming data from a large corpus of variable fonts to analyze
patterns and improve the naming normalization logic in core_name_policies.py and
VariableFont_Instancer.py.

Usage:
  python VF_Metadata_Extractor.py ~/FontLibrary/VariableFonts/ -o output/
  python VF_Metadata_Extractor.py ~/FontLibrary/ -filter "Google Fonts" -o google_analysis/
  python VF_Metadata_Extractor.py ~/FontLibrary/ --abbreviations-only -o abbrev_dict.json
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from fontTools.ttLib import TTFont

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
from FontCore.core_file_collector import collect_font_files
from FontCore.core_error_handling import ErrorTracker, ErrorContext

logger = cs.get_logger(__name__)
console = cs.get_console()


@dataclass
class NameTableRecord:
    """Single name table record with platform info."""

    nameID: int
    platformID: int
    encodingID: int
    languageID: int
    string: str


@dataclass
class STATAxisValue:
    """STAT AxisValue record."""

    format: int
    axis_index: int
    axis_tag: str
    value: float
    value_name_id: int
    value_name: str
    flags: int
    is_elidable: bool
    linked_value: Optional[float] = None


@dataclass
class STATAxis:
    """STAT DesignAxis record."""

    tag: str
    name_id: int
    name: str
    ordering: int


@dataclass
class FvarAxis:
    """fvar axis record."""

    tag: str
    name_id: int
    name: str
    min_value: float
    default_value: float
    max_value: float
    flags: int


@dataclass
class FvarInstance:
    """fvar named instance."""

    subfamily_name_id: int
    subfamily_name: str
    coordinates: Dict[str, float]
    postscript_name_id: Optional[int] = None
    postscript_name: Optional[str] = None


@dataclass
class FontMetadata:
    """Complete font metadata."""

    font_path: str
    family_name: str
    subfamily_name: str
    full_name: str
    postscript_name: str

    # Name table data
    name_records: List[NameTableRecord]
    high_name_ids: List[int]  # nameIDs > 255

    # STAT table data
    stat_axes: List[STATAxis]
    stat_values: List[STATAxisValue]
    elided_fallback_name_id: Optional[int]
    elided_fallback_name: Optional[str]

    # fvar table data
    fvar_axes: List[FvarAxis]
    fvar_instances: List[FvarInstance]

    # Analysis data
    stat_derived_names: Dict[str, str]  # coordinates -> STAT-derived name
    naming_comparisons: List[Dict[str, Any]]  # STAT vs fvar comparisons
    abbreviation_mappings: Dict[str, str]  # detected abbreviations


class VFMetadataExtractor:
    """Extract and analyze variable font metadata."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_tracker = ErrorTracker()
        self.font_metadata: List[FontMetadata] = []

    def extract_from_directory(
        self,
        font_dir: Path,
        recursive: bool = True,
        filter_pattern: Optional[str] = None,
    ) -> None:
        """Extract metadata from all variable fonts in directory."""
        cs.fmt_header("Variable Font Metadata Extraction", console)

        # Collect font files
        if font_dir.is_file():
            # Single file
            font_files = [str(font_dir)]
        else:
            # Directory
            font_files = collect_font_files(
                [str(font_dir)],
                recursive=recursive,
                allowed_extensions={".ttf", ".otf", ".woff2"},
            )

        if filter_pattern:
            font_files = [f for f in font_files if filter_pattern.lower() in f.lower()]

        cs.emit(f"Found {len(font_files)} font files to analyze")

        # Process each font
        for i, font_path in enumerate(font_files, 1):
            try:
                cs.emit(f"[{i}/{len(font_files)}] Processing: {Path(font_path).name}")
                metadata = self._extract_single_font(font_path)
                if metadata:
                    self.font_metadata.append(metadata)

            except Exception as e:
                logger.error(f"Failed to process {font_path}: {e}")
                self.error_tracker.add_from_exception(
                    context=ErrorContext.CONSTRUCTION,
                    message=f"Failed to extract metadata from {font_path}",
                    details=str(e),
                )
                continue

        cs.emit(f"\nSuccessfully processed {len(self.font_metadata)} fonts")

        # Generate analysis
        self._generate_analysis()

    def _extract_single_font(self, font_path: str) -> Optional[FontMetadata]:
        """Extract metadata from a single font file."""
        try:
            font = TTFont(font_path)

            # Check if it's a variable font
            if "fvar" not in font:
                logger.debug(f"Skipping non-variable font: {font_path}")
                return None

            logger.debug(f"Processing variable font: {font_path}")

            # Extract name table
            name_records, high_name_ids = self._extract_name_table(font)

            # Extract STAT table
            stat_axes, stat_values, elided_fallback = self._extract_stat_table(font)

            # Extract fvar table
            fvar_axes, fvar_instances = self._extract_fvar_table(font)

            # Build STAT-derived names for fvar instances
            stat_derived_names = self._build_stat_derived_names(
                fvar_instances, stat_axes, stat_values
            )

            # Compare STAT vs fvar naming
            naming_comparisons = self._compare_stat_fvar_naming(
                fvar_instances, stat_derived_names
            )

            # Detect abbreviations
            abbreviation_mappings = self._detect_abbreviations(
                stat_values, fvar_instances
            )

            return FontMetadata(
                font_path=font_path,
                family_name=font["name"].getDebugName(1) or "Unknown",
                subfamily_name=font["name"].getDebugName(2) or "Unknown",
                full_name=font["name"].getDebugName(4) or "Unknown",
                postscript_name=font["name"].getDebugName(6) or "Unknown",
                name_records=name_records,
                high_name_ids=high_name_ids,
                stat_axes=stat_axes,
                stat_values=stat_values,
                elided_fallback_name_id=elided_fallback[0] if elided_fallback else None,
                elided_fallback_name=elided_fallback[1] if elided_fallback else None,
                fvar_axes=fvar_axes,
                fvar_instances=fvar_instances,
                stat_derived_names=stat_derived_names,
                naming_comparisons=naming_comparisons,
                abbreviation_mappings=abbreviation_mappings,
            )

        except Exception as e:
            logger.error(f"Error extracting from {font_path}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _extract_name_table(
        self, font: TTFont
    ) -> Tuple[List[NameTableRecord], List[int]]:
        """Extract all name table records."""
        name_records = []
        high_name_ids = []

        if "name" not in font:
            return name_records, high_name_ids

        name_table = font["name"]

        for record in name_table.names:
            # Handle different fontTools versions
            encoding_id = getattr(record, "encodingID", getattr(record, "encoding", 0))
            language_id = getattr(record, "languageID", getattr(record, "langID", 0))

            name_record = NameTableRecord(
                nameID=record.nameID,
                platformID=record.platformID,
                encodingID=encoding_id,
                languageID=language_id,
                string=record.toUnicode(),
            )
            name_records.append(name_record)

            if record.nameID > 255:
                high_name_ids.append(record.nameID)

        return name_records, sorted(set(high_name_ids))

    def _extract_stat_table(
        self, font: TTFont
    ) -> Tuple[List[STATAxis], List[STATAxisValue], Optional[Tuple[int, str]]]:
        """Extract STAT table data."""
        stat_axes = []
        stat_values = []
        elided_fallback = None

        if "STAT" not in font:
            return stat_axes, stat_values, elided_fallback

        stat_table = font["STAT"].table

        # Extract axes
        if hasattr(stat_table, "DesignAxisRecord") and stat_table.DesignAxisRecord:
            for axis in stat_table.DesignAxisRecord.Axis:
                axis_name = font["name"].getDebugName(axis.AxisNameID) or "Unknown"
                stat_axes.append(
                    STATAxis(
                        tag=axis.AxisTag,
                        name_id=axis.AxisNameID,
                        name=axis_name,
                        ordering=axis.AxisOrdering,
                    )
                )

        # Extract axis values
        if hasattr(stat_table, "AxisValueArray") and stat_table.AxisValueArray:
            for av in stat_table.AxisValueArray.AxisValue:
                # Find axis tag by index position
                axis_tag = "unknown"
                if 0 <= av.AxisIndex < len(stat_axes):
                    axis_tag = stat_axes[av.AxisIndex].tag

                value_name = font["name"].getDebugName(av.ValueNameID) or "Unknown"
                is_elidable = bool(av.Flags & 0x0002)  # ElidableAxisValueName

                stat_value = STATAxisValue(
                    format=av.Format,
                    axis_index=av.AxisIndex,
                    axis_tag=axis_tag,
                    value=av.Value,
                    value_name_id=av.ValueNameID,
                    value_name=value_name,
                    flags=av.Flags,
                    is_elidable=is_elidable,
                )

                # Handle Format 3 (linked values)
                if av.Format == 3 and hasattr(av, "LinkedValue"):
                    stat_value.linked_value = av.LinkedValue

                stat_values.append(stat_value)

        # Extract elided fallback
        if (
            hasattr(stat_table, "ElidedFallbackNameID")
            and stat_table.ElidedFallbackNameID
        ):
            fallback_name = font["name"].getDebugName(stat_table.ElidedFallbackNameID)
            elided_fallback = (stat_table.ElidedFallbackNameID, fallback_name)

        return stat_axes, stat_values, elided_fallback

    def _extract_fvar_table(
        self, font: TTFont
    ) -> Tuple[List[FvarAxis], List[FvarInstance]]:
        """Extract fvar table data."""
        fvar_axes = []
        fvar_instances = []

        if "fvar" not in font:
            return fvar_axes, fvar_instances

        fvar_table = font["fvar"]

        # Extract axes
        for axis in fvar_table.axes:
            axis_name = font["name"].getDebugName(axis.axisNameID) or "Unknown"
            fvar_axes.append(
                FvarAxis(
                    tag=axis.axisTag,
                    name_id=axis.axisNameID,
                    name=axis_name,
                    min_value=axis.minValue,
                    default_value=axis.defaultValue,
                    max_value=axis.maxValue,
                    flags=axis.flags,
                )
            )

        # Extract named instances
        for inst in fvar_table.instances:
            subfamily_name = (
                font["name"].getDebugName(inst.subfamilyNameID) or "Unknown"
            )

            # Build coordinates dict
            coordinates = {}
            if hasattr(inst, "coordinates") and inst.coordinates is not None:
                if isinstance(inst.coordinates, dict):
                    # coordinates is already a dict
                    coordinates = inst.coordinates
                else:
                    # coordinates is a list, map by index
                    for i, axis in enumerate(fvar_axes):
                        if i < len(inst.coordinates):
                            coordinates[axis.tag] = inst.coordinates[i]

            postscript_name = None
            if hasattr(inst, "postscriptNameID") and inst.postscriptNameID != 0xFFFF:
                postscript_name = font["name"].getDebugName(inst.postscriptNameID)

            fvar_instances.append(
                FvarInstance(
                    subfamily_name_id=inst.subfamilyNameID,
                    subfamily_name=subfamily_name,
                    coordinates=coordinates,
                    postscript_name_id=getattr(inst, "postscriptNameID", None),
                    postscript_name=postscript_name,
                )
            )

        return fvar_axes, fvar_instances

    def _build_stat_derived_names(
        self,
        fvar_instances: List[FvarInstance],
        stat_axes: List[STATAxis],
        stat_values: List[STATAxisValue],
    ) -> Dict[str, str]:
        """Build STAT-derived names for fvar instances."""
        stat_derived_names = {}

        # Create axis value mapping
        axis_values = {}
        for sv in stat_values:
            if sv.axis_tag not in axis_values:
                axis_values[sv.axis_tag] = {}
            axis_values[sv.axis_tag][sv.value] = sv.value_name

        for inst in fvar_instances:
            # Build name parts
            name_parts = []

            for axis in stat_axes:
                if axis.tag in inst.coordinates:
                    value = inst.coordinates[axis.tag]

                    # Find closest STAT value
                    if axis.tag in axis_values:
                        closest_name = self._find_closest_stat_name(
                            value, axis_values[axis.tag]
                        )
                        if closest_name:
                            name_parts.append(closest_name)

            # Create coordinates key
            coords_key = ",".join(
                [f"{k}={v}" for k, v in sorted(inst.coordinates.items())]
            )
            stat_derived_names[coords_key] = (
                " ".join(name_parts) if name_parts else "Regular"
            )

        return stat_derived_names

    def _find_closest_stat_name(
        self, value: float, stat_names: Dict[float, str], epsilon: float = 0.5
    ) -> Optional[str]:
        """Find the closest STAT name for a given value."""
        for stat_value, name in stat_names.items():
            if abs(value - stat_value) <= epsilon:
                return name
        return None

    def _compare_stat_fvar_naming(
        self, fvar_instances: List[FvarInstance], stat_derived_names: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Compare STAT vs fvar naming."""
        comparisons = []

        for inst in fvar_instances:
            coords_key = ",".join(
                [f"{k}={v}" for k, v in sorted(inst.coordinates.items())]
            )
            stat_name = stat_derived_names.get(coords_key, "Unknown")

            comparison = {
                "coordinates": coords_key,
                "fvar_name": inst.subfamily_name,
                "stat_name": stat_name,
                "identical": stat_name == inst.subfamily_name,
                "fvar_length": len(inst.subfamily_name),
                "stat_length": len(stat_name),
            }
            comparisons.append(comparison)

        return comparisons

    def _detect_abbreviations(
        self, stat_values: List[STATAxisValue], fvar_instances: List[FvarInstance]
    ) -> Dict[str, str]:
        """Detect abbreviation patterns."""
        abbreviations = {}

        # Collect all names
        all_names = set()
        for sv in stat_values:
            all_names.add(sv.value_name)
        for inst in fvar_instances:
            all_names.add(inst.subfamily_name)

        # Simple abbreviation detection
        common_abbrevs = {
            "Condensed": ["Cnd", "Cond"],
            "Extended": ["Ext", "Extd"],
            "Slanted": ["Slnt", "Slant"],
            "Italic": ["Ital", "It"],
            "Bold": ["Bd", "B"],
            "Light": ["Lt", "L"],
            "Medium": ["Med", "Md"],
            "Regular": ["Reg", "R"],
            "Thin": ["Th", "T"],
            "Black": ["Blk", "Bk"],
        }

        for full, abbrevs in common_abbrevs.items():
            for abbrev in abbrevs:
                if abbrev in all_names and full in all_names:
                    abbreviations[abbrev] = full

        return abbreviations

    def _generate_analysis(self) -> None:
        """Generate analysis reports."""
        cs.emit("\nGenerating analysis reports...")

        # Export full metadata
        self._export_json()

        # Generate CSV reports
        self._export_instances_csv()
        self._export_patterns_csv()
        self._export_abbreviations_csv()
        self._export_stat_values_csv()

        # Generate markdown report
        self._export_summary_report()

        cs.emit(f"Analysis complete. Results saved to: {self.output_dir}")

    def _export_json(self) -> None:
        """Export complete metadata to JSON."""
        json_data = {
            "extraction_info": {
                "total_fonts": len(self.font_metadata),
                "extraction_timestamp": str(Path().cwd()),
            },
            "fonts": [asdict(font) for font in self.font_metadata],
        }

        json_path = self.output_dir / "metadata_full.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        cs.emit(f"  ✓ Full metadata: {json_path}")

    def _export_instances_csv(self) -> None:
        """Export instance analysis to CSV."""
        csv_path = self.output_dir / "instances_analysis.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Font Path",
                    "Family",
                    "Coordinates",
                    "fvar Name",
                    "STAT Name",
                    "Identical",
                    "fvar Length",
                    "STAT Length",
                ]
            )

            for font in self.font_metadata:
                for comparison in font.naming_comparisons:
                    writer.writerow(
                        [
                            font.font_path,
                            font.family_name,
                            comparison["coordinates"],
                            comparison["fvar_name"],
                            comparison["stat_name"],
                            comparison["identical"],
                            comparison["fvar_length"],
                            comparison["stat_length"],
                        ]
                    )

        cs.emit(f"  ✓ Instance analysis: {csv_path}")

    def _export_patterns_csv(self) -> None:
        """Export naming patterns to CSV."""
        csv_path = self.output_dir / "naming_patterns.csv"

        # Collect all axis values and their names
        axis_patterns = {}
        for font in self.font_metadata:
            for stat_value in font.stat_values:
                key = f"{stat_value.axis_tag}={stat_value.value}"
                if key not in axis_patterns:
                    axis_patterns[key] = []
                axis_patterns[key].append(stat_value.value_name)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Axis Value", "Names", "Count", "Most Common"])

            for axis_value, names in axis_patterns.items():
                from collections import Counter

                name_counts = Counter(names)
                most_common = name_counts.most_common(1)[0][0] if name_counts else ""

                writer.writerow(
                    [axis_value, "; ".join(set(names)), len(names), most_common]
                )

        cs.emit(f"  ✓ Naming patterns: {csv_path}")

    def _export_abbreviations_csv(self) -> None:
        """Export abbreviation mappings to CSV."""
        csv_path = self.output_dir / "abbreviations.csv"

        # Collect all abbreviations
        all_abbrevs = {}
        for font in self.font_metadata:
            all_abbrevs.update(font.abbreviation_mappings)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Abbreviation", "Full Name", "Frequency"])

            for abbrev, full in all_abbrevs.items():
                # Count frequency across fonts
                freq = sum(
                    1
                    for font in self.font_metadata
                    if abbrev in font.abbreviation_mappings
                )
                writer.writerow([abbrev, full, freq])

        cs.emit(f"  ✓ Abbreviations: {csv_path}")

    def _export_stat_values_csv(self) -> None:
        """Export STAT values to CSV."""
        csv_path = self.output_dir / "stat_values.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Font", "Axis Tag", "Value", "Name", "Is Elidable", "Format"]
            )

            for font in self.font_metadata:
                for stat_value in font.stat_values:
                    writer.writerow(
                        [
                            Path(font.font_path).name,
                            stat_value.axis_tag,
                            stat_value.value,
                            stat_value.value_name,
                            stat_value.is_elidable,
                            stat_value.format,
                        ]
                    )

        cs.emit(f"  ✓ STAT values: {csv_path}")

    def _export_summary_report(self) -> None:
        """Generate human-readable summary report."""
        report_path = self.output_dir / "summary_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Variable Font Metadata Analysis Report\n\n")
            f.write(f"**Analysis Date:** {Path().cwd()}\n")
            f.write(f"**Total Fonts Analyzed:** {len(self.font_metadata)}\n\n")

            # Font summary
            f.write("## Font Summary\n\n")
            for font in self.font_metadata:
                f.write(f"- **{Path(font.font_path).name}**\n")
                f.write(f"  - Family: {font.family_name}\n")
                f.write(f"  - STAT Axes: {len(font.stat_axes)}\n")
                f.write(f"  - fvar Instances: {len(font.fvar_instances)}\n")
                f.write(f"  - High Name IDs: {len(font.high_name_ids)}\n\n")

            # Naming consistency
            f.write("## Naming Consistency Analysis\n\n")
            total_instances = sum(
                len(font.fvar_instances) for font in self.font_metadata
            )
            identical_count = sum(
                sum(1 for comp in font.naming_comparisons if comp["identical"])
                for font in self.font_metadata
            )
            consistency_pct = (
                (identical_count / total_instances * 100) if total_instances > 0 else 0
            )

            f.write(f"- **Total Instances:** {total_instances}\n")
            f.write(
                f"- **STAT/fvar Identical:** {identical_count} ({consistency_pct:.1f}%)\n"
            )
            f.write(
                f"- **STAT/fvar Different:** {total_instances - identical_count} ({100 - consistency_pct:.1f}%)\n\n"
            )

            # Common abbreviations
            f.write("## Common Abbreviations Detected\n\n")
            all_abbrevs = {}
            for font in self.font_metadata:
                for abbrev, full in font.abbreviation_mappings.items():
                    if abbrev not in all_abbrevs:
                        all_abbrevs[abbrev] = []
                    all_abbrevs[abbrev].append(full)

            for abbrev, fulls in all_abbrevs.items():
                f.write(f"- **{abbrev}** → {', '.join(set(fulls))}\n")

            f.write("\n## Recommendations\n\n")
            f.write("Based on this analysis:\n\n")
            f.write("1. **Naming Consistency:** ")
            if consistency_pct > 80:
                f.write("Good consistency between STAT and fvar naming.\n")
            elif consistency_pct > 60:
                f.write(
                    "Moderate consistency - some fonts may need STAT table improvements.\n"
                )
            else:
                f.write(
                    "Poor consistency - many fonts have misaligned STAT and fvar naming.\n"
                )

            f.write("2. **Abbreviation Usage:** ")
            if all_abbrevs:
                f.write("Common abbreviations detected that could be standardized.\n")
            else:
                f.write("No significant abbreviation patterns found.\n")

            f.write(
                "3. **Next Steps:** Use the CSV exports for detailed analysis and pattern recognition.\n"
            )

        cs.emit(f"  ✓ Summary report: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract and analyze variable font metadata for naming pattern recognition"
    )
    parser.add_argument("font_dir", help="Directory containing variable fonts")
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for analysis results"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Scan directories recursively"
    )
    parser.add_argument("--filter", help="Filter fonts by pattern (case-insensitive)")
    parser.add_argument(
        "--abbreviations-only",
        action="store_true",
        help="Only generate abbreviation dictionary",
    )

    args = parser.parse_args()

    # Validate input
    font_dir = Path(args.font_dir)
    if not font_dir.exists():
        cs.emit(f"Error: Path {font_dir} does not exist")
        return 1

    output_dir = Path(args.output)

    # Create extractor
    extractor = VFMetadataExtractor(output_dir)

    # Extract metadata
    extractor.extract_from_directory(
        font_dir, recursive=args.recursive, filter_pattern=args.filter
    )

    # Show error summary if any
    error_summary = extractor.error_tracker.get_summary()
    if error_summary["total_errors"] > 0:
        cs.emit(
            f"\n⚠️  {error_summary['total_errors']} errors occurred during extraction"
        )
        extractor.error_tracker.print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
