#!/usr/bin/env python3
"""
Scan variable fonts for STAT table axis value records of a given format.

STAT axis value formats:
  1 — Name/value pair (single axis, exact value)
  2 — Range (single axis, min/nominal/max + linked value)
  3 — Linked values (single axis, exact value + linked value)
  4 — Multi-axis values (pins multiple axes to a named style, e.g. "Bold Italic")

Usage:
    python scan_stat_format4.py /path/to/fonts/
    python scan_stat_format4.py /path/to/fonts/ --format 1
    python scan_stat_format4.py /path/to/fonts/ --format 4 --ext .ttf .otf .ttc
"""

import argparse
import sys
from pathlib import Path

try:
    from fontTools.ttLib import TTFont, TTCollection
except ImportError:
    sys.exit("fontTools is not installed. Run: pip install fonttools")


def axis_tag(stat, axis_index: int) -> str:
    """Resolve an axis index to its 4-char tag, falling back gracefully."""
    try:
        return stat.DesignAxisRecord.Axis[axis_index].AxisTag
    except (AttributeError, IndexError):
        return f"axis[{axis_index}]"


def describe_axis_value(av, stat) -> list[str]:
    """Return human-readable detail lines for an axis value record."""
    if av.Format == 1:
        return [f"{axis_tag(stat, av.AxisIndex)}={av.Value}"]
    elif av.Format == 2:
        return [
            f"{axis_tag(stat, av.AxisIndex)}  "
            f"nominal={av.NominalValue}  "
            f"range=[{av.RangeMinValue}, {av.RangeMaxValue}]"
        ]
    elif av.Format == 3:
        return [f"{axis_tag(stat, av.AxisIndex)}={av.Value}  linked={av.LinkedValue}"]
    elif av.Format == 4:
        return [
            f"{axis_tag(stat, r.AxisIndex)}={r.Value}"
            for r in av.AxisValueRecord
        ]
    return [f"(unknown format {av.Format})"]


def scan_font(path: Path, target_format: int) -> list[dict]:
    """Return a list of findings for axis value records matching target_format."""
    findings = []

    # Handle .ttc / .otc collections
    if path.suffix.lower() in (".ttc", ".otc"):
        try:
            collection = TTCollection(path)
            fonts = list(collection.fonts)
        except Exception as e:
            print(f"  [WARN] Could not open collection {path.name}: {e}")
            return findings
    else:
        try:
            fonts = [TTFont(path, lazy=True)]
        except Exception as e:
            print(f"  [WARN] Could not open {path.name}: {e}")
            return findings

    for font_index, font in enumerate(fonts):
        label = f"{path.name}" + (f" [face {font_index}]" if len(fonts) > 1 else "")

        if "STAT" not in font:
            continue

        stat = font["STAT"].table
        if not stat or not hasattr(stat, "AxisValueArray") or stat.AxisValueArray is None:
            continue

        for i, av in enumerate(stat.AxisValueArray.AxisValue):
            if av.Format != target_format:
                continue

            name_id = av.ValueNameID
            try:
                name = font["name"].getDebugName(name_id) or f"nameID={name_id}"
            except Exception:
                name = f"nameID={name_id}"

            findings.append({
                "font": label,
                "axis_value_index": i,
                "name": name,
                "details": describe_axis_value(av, stat),
            })

        font.close()

    return findings


def main():
    parser = argparse.ArgumentParser(description="Find STAT axis value records by format in fonts.")
    parser.add_argument("path", help="Font file or directory to scan")
    parser.add_argument(
        "--format",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="STAT axis value format to search for (default: 4)",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=[".ttf", ".otf", ".ttc", ".otc"],
        help="File extensions to include (default: .ttf .otf .ttc .otc)",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        sys.exit(f"Path not found: {root}")

    extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext}

    font_files = sorted(root.rglob("*")) if root.is_dir() else [root]
    font_files = [f for f in font_files if f.is_file() and f.suffix.lower() in extensions]

    if not font_files:
        sys.exit(f"No font files found at {root} with extensions {extensions}")

    print(f"Scanning {len(font_files)} font file(s) for STAT format={args.format} axis values...\n")

    total_findings = []
    fonts_with_hits = 0

    for font_path in font_files:
        findings = scan_font(font_path, args.format)
        if findings:
            fonts_with_hits += 1
            for f in findings:
                print(f"✔  {f['font']}")
                print(f"   Axis value #{f['axis_value_index']:>3}  \"{f['name']}\"")
                for detail in f["details"]:
                    print(f"   {detail}")
                print()
            total_findings.extend(findings)

    print("─" * 60)
    if total_findings:
        print(
            f"Found {len(total_findings)} format={args.format} axis value record(s) "
            f"across {fonts_with_hits} font file(s)."
        )
    else:
        print(f"No format={args.format} axis value records found.")


if __name__ == "__main__":
    main()