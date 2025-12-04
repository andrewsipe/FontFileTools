#!/usr/bin/env python3
"""
GASP Table Filler

Search fonts/TTX for empty or missing 'gasp' tables and fill them with a
recommended Version 1 profile per the OpenType spec.

Defaults to the 4-range sample profile:
  - <= 8    : 0x000A (GASP_DOGRAY | GASP_SYMMETRIC_SMOOTHING)
  - <= 16   : 0x0005 (GASP_GRIDFIT | GASP_SYMMETRIC_GRIDFIT)
  - <= 19   : 0x0007 (GASP_GRIDFIT | GASP_DOGRAY | GASP_SYMMETRIC_GRIDFIT)
  - <= 0xFFFF : 0x000F (All four flags set)

Optionally, a simpler 2-range profile is available via --simple:
  - <= 8    : 0x000A
  - <= 0xFFFF : 0x000F

Supported formats: TTF, OTF, WOFF, WOFF2, TTX
"""

import sys
import argparse
import glob
from pathlib import Path
import xml.etree.ElementTree as ET

from fontTools.ttLib import TTFont, newTable

import FontCore.core_console_styles as cs

# Keep label aliases for parity with other scripts
updated_label = cs.UPDATED_LABEL
UNCHANGED_LABEL = cs.UNCHANGED_LABEL

SUPPORTED_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2", ".ttx"}

# Optional lxml for better XML handling
try:
    from lxml import etree as LET

    LXML_AVAILABLE = True
except Exception:
    LXML_AVAILABLE = False


# ---- GASP constants and default profiles ----
GASP_GRIDFIT = 0x0001
GASP_DOGRAY = 0x0002
GASP_SYMMETRIC_GRIDFIT = 0x0004
GASP_SYMMETRIC_SMOOTHING = 0x0008

DEFAULT_PROFILE = [
    (8, GASP_DOGRAY | GASP_SYMMETRIC_SMOOTHING),  # 0x000A
    (16, GASP_GRIDFIT | GASP_SYMMETRIC_GRIDFIT),  # 0x0005
    (19, GASP_GRIDFIT | GASP_DOGRAY | GASP_SYMMETRIC_GRIDFIT),  # 0x0007
    (
        0xFFFF,
        GASP_GRIDFIT | GASP_DOGRAY | GASP_SYMMETRIC_GRIDFIT | GASP_SYMMETRIC_SMOOTHING,
    ),  # 0x000F
]

SIMPLE_PROFILE = [
    (8, GASP_DOGRAY | GASP_SYMMETRIC_SMOOTHING),
    (
        0xFFFF,
        GASP_GRIDFIT | GASP_DOGRAY | GASP_SYMMETRIC_GRIDFIT | GASP_SYMMETRIC_SMOOTHING,
    ),
]


def build_profile(use_simple: bool) -> list[tuple[int, int]]:
    return SIMPLE_PROFILE if use_simple else DEFAULT_PROFILE


# ---- TTX helpers ----
def _get_or_create_child(elem, tag: str):
    child = elem.find(f".//{tag}") if tag not in {"gaspRange"} else None
    if child is None:
        if LXML_AVAILABLE:
            child = LET.Element(tag)
        else:
            child = ET.Element(tag)
        elem.append(child)
    return child


def _clear_gasp_ranges(gasp_elem) -> None:
    for child in list(gasp_elem):
        if child.tag == "gaspRange":
            gasp_elem.remove(child)


def _count_gasp_ranges(gasp_elem) -> int:
    return sum(1 for c in list(gasp_elem) if c.tag == "gaspRange")


def process_ttx_file(
    filepath: str, overwrite: bool, do_add: bool, do_remove: bool, use_simple: bool
) -> bool:
    try:
        if LXML_AVAILABLE:
            parser = LET.XMLParser(remove_blank_text=False, remove_comments=False)
            tree = LET.parse(filepath, parser)
            root = tree.getroot()
        else:
            tree = ET.parse(filepath)
            root = tree.getroot()

        gasp = root.find(".//gasp")
        if do_remove:
            if gasp is None:
                cs.StatusIndicator("unchanged").add_file(filepath).with_explanation(
                    "No 'gasp' table to remove"
                ).emit()
                return False
            try:
                root.remove(gasp)
            except Exception:
                # some TTX structures may nest tables; fallback to clearing children
                for child in list(root):
                    if child is gasp:
                        root.remove(child)
                        break
            if LXML_AVAILABLE:
                tree.write(
                    filepath, encoding="utf-8", xml_declaration=True, pretty_print=False
                )
            else:
                tree.write(filepath, encoding="utf-8", xml_declaration=True)
            cs.StatusIndicator("updated").add_file(filepath).with_explanation(
                "Removed 'gasp' table"
            ).emit()
            return True

        created = False
        if gasp is None:
            if not do_add:
                cs.StatusIndicator("unchanged").add_file(filepath).with_explanation(
                    "No 'gasp' table present"
                ).emit()
                return False
            gasp = LET.Element("gasp") if LXML_AVAILABLE else ET.Element("gasp")
            root.append(gasp)
            created = True

        existing_ranges = _count_gasp_ranges(gasp)
        if existing_ranges > 0 and not overwrite:
            cs.StatusIndicator("unchanged").add_file(filepath).with_explanation(
                f"Existing 'gasp' ranges present ({existing_ranges})"
            ).emit()
            return False

        # (Re)build table
        _clear_gasp_ranges(gasp)

        version_elem = gasp.find("version")
        if version_elem is None:
            version_elem = (
                LET.Element("version") if LXML_AVAILABLE else ET.Element("version")
            )
            gasp.insert(0, version_elem)
        version_elem.set("value", "1")

        num_ranges_elem = gasp.find("numRanges")
        if num_ranges_elem is None:
            num_ranges_elem = (
                LET.Element("numRanges") if LXML_AVAILABLE else ET.Element("numRanges")
            )
            gasp.insert(1, num_ranges_elem)

        profile = build_profile(use_simple)
        num_ranges_elem.set("value", str(len(profile)))

        for rng_max, behavior in profile:
            e = LET.Element("gaspRange") if LXML_AVAILABLE else ET.Element("gaspRange")
            e.set("rangeMaxPPEM", str(rng_max))
            e.set("rangeGaspBehavior", f"0x{behavior:04X}")
            gasp.append(e)

        if LXML_AVAILABLE:
            tree.write(
                filepath, encoding="utf-8", xml_declaration=True, pretty_print=False
            )
        else:
            tree.write(filepath, encoding="utf-8", xml_declaration=True)

        if created and existing_ranges == 0:
            cs.StatusIndicator("saved").add_file(filepath).with_explanation(
                "Added 'gasp' table"
            ).emit()
        elif existing_ranges == 0:
            cs.StatusIndicator("updated").add_file(filepath).with_explanation(
                "Filled empty 'gasp' ranges"
            ).emit()
        else:
            cs.StatusIndicator("updated").add_file(filepath).with_explanation(
                "Rebuilt 'gasp' ranges"
            ).emit()
        return True
    except Exception as e:
        cs.StatusIndicator("error").add_file(filepath).with_explanation(
            f"Error processing TTX file: {e}"
        ).emit()
        return False


# ---- Binary helpers ----
def _describe_gasp_dict(d: dict[int, int]) -> str:
    if not d:
        return "{}"
    parts = []
    for k in sorted(d.keys()):
        parts.append(f"{k}:0x{d[k]:04X}")
    return "{" + ", ".join(parts) + "}"


def process_binary_font(
    filepath: str, overwrite: bool, do_add: bool, do_remove: bool, use_simple: bool
) -> bool:
    try:
        font = TTFont(filepath)
        try:
            profile = build_profile(use_simple)
            target = {rng: beh for rng, beh in profile}

            if do_remove:
                if "gasp" not in font:
                    cs.StatusIndicator("unchanged").add_file(filepath).with_explanation(
                        "No 'gasp' table to remove"
                    ).emit()
                    return False
                try:
                    del font["gasp"]
                except Exception:
                    # fallback: set to empty and let save drop it
                    font["gasp"].gaspRange = {}
                    del font["gasp"]
                font.save(filepath)
                cs.StatusIndicator("updated").add_file(filepath).with_explanation(
                    "Removed 'gasp' table"
                ).emit()
                return True

            if "gasp" not in font:
                if not do_add:
                    cs.StatusIndicator("unchanged").add_file(filepath).with_explanation(
                        "No 'gasp' table present"
                    ).emit()
                    return False
                gasp_table = newTable("gasp")
                gasp_table.version = 1
                gasp_table.gaspRange = target
                font["gasp"] = gasp_table
                font.save(filepath)
                cs.StatusIndicator("saved").add_file(filepath).with_explanation(
                    "Added 'gasp' table"
                ).emit()
                return True

            gasp_table = font["gasp"]
            existing = getattr(gasp_table, "gaspRange", {}) or {}
            if existing and not overwrite:
                cs.StatusIndicator("unchanged").add_file(filepath).with_explanation(
                    f"Existing 'gasp' ranges present: {_describe_gasp_dict(existing)}"
                ).emit()
                return False

            gasp_table.version = 1
            gasp_table.gaspRange = target
            font.save(filepath)
            if not existing:
                cs.StatusIndicator("updated").add_file(filepath).with_explanation(
                    "Filled empty 'gasp' ranges"
                ).emit()
            else:
                cs.StatusIndicator("updated").add_file(filepath).with_explanation(
                    "Rebuilt 'gasp' ranges"
                ).emit()
            return True
        finally:
            try:
                font.close()
            except Exception:
                pass
    except Exception as e:
        cs.StatusIndicator("error").add_file(filepath).with_explanation(
            f"Error processing font file: {e}"
        ).emit()
        return False


def process_file(
    filepath: str, overwrite: bool, do_add: bool, do_remove: bool, use_simple: bool
) -> bool:
    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        cs.StatusIndicator("warning").add_file(filepath).with_explanation(
            "Skipping unsupported file"
        ).emit()
        return False
    cs.StatusIndicator("info").add_file(filepath).add_message("Processing").emit()
    if ext == ".ttx":
        return process_ttx_file(filepath, overwrite, do_add, do_remove, use_simple)
    return process_binary_font(filepath, overwrite, do_add, do_remove, use_simple)


def collect_font_files(paths: list[str], recursive: bool = False) -> list[str]:
    font_files: list[str] = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.is_file():
            if path_obj.suffix.lower() in SUPPORTED_EXTENSIONS:
                font_files.append(str(path_obj))
            else:
                cs.StatusIndicator("warning").add_file(path).with_explanation(
                    "is not a supported font file"
                ).emit()
        elif path_obj.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                if recursive:
                    pattern = str(path_obj / f"**/*{ext}")
                    font_files.extend(glob.glob(pattern, recursive=True))
                    pattern_upper = str(path_obj / f"**/*{ext.upper()}")
                    font_files.extend(glob.glob(pattern_upper, recursive=True))
                else:
                    pattern = str(path_obj / f"*{ext}")
                    font_files.extend(glob.glob(pattern))
                    pattern_upper = str(path_obj / f"*{ext.upper()}")
                    font_files.extend(glob.glob(pattern_upper))
        else:
            cs.StatusIndicator("warning").add_file(path).with_explanation(
                "does not exist"
            ).emit()
    return sorted(set(font_files))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage 'gasp' tables in font files (fill empty by default)",
        epilog="Supported formats: TTF, OTF, WOFF, WOFF2, TTX",
    )

    parser.add_argument("paths", nargs="+", help="Font files or directories to process")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning directories",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing non-empty 'gasp' tables",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Create and fill a 'gasp' table when none exists",
    )
    parser.add_argument(
        "--remove", action="store_true", help="Remove the 'gasp' table entirely"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use a simpler 2-range profile instead of the 4-range default",
    )

    args = parser.parse_args()

    # Check fonttools availability (should be, since we import above)
    import importlib.util

    if importlib.util.find_spec("fontTools") is None:
        cs.StatusIndicator("error").with_explanation(
            "fonttools is required. Install with: pip install fonttools"
        ).emit()
        sys.exit(1)

    font_files = collect_font_files(args.paths, args.recursive)
    if not font_files:
        cs.StatusIndicator("error").with_explanation(
            "No font files found to process"
        ).emit()
        sys.exit(1)

    cs.StatusIndicator("info").add_message(
        f"Found {len(font_files)} font files to process:"
    ).emit()
    for file in font_files:
        cs.StatusIndicator("info").add_file(file).emit()

    profile = build_profile(args.simple)
    # default behavior per request: only fill when existing gasp table is present but empty
    # --add allows creating one when missing; --remove deletes it.

    if args.dry_run:
        cs.StatusIndicator("warning").with_explanation(
            "Dry run mode - no changes will be made"
        ).emit()
        for file in font_files:
            ext = Path(file).suffix.lower()
            action = ""
            try:
                if ext == ".ttx":
                    tree = ET.parse(file)
                    root = tree.getroot()
                    gasp = root.find(".//gasp")
                    if gasp is None:
                        if args.remove:
                            action = "Skip (no 'gasp' to remove)"
                        elif args.add:
                            action = "Add 'gasp' table"
                        else:
                            action = "Skip (missing 'gasp')"
                    else:
                        existing = sum(1 for c in list(gasp) if c.tag == "gaspRange")
                        if args.remove:
                            action = "Remove 'gasp' table"
                        elif existing == 0:
                            action = "Fill empty 'gasp'"
                        elif args.overwrite:
                            action = "Overwrite 'gasp' ranges"
                        else:
                            action = f"Skip (has {existing} ranges)"
                else:
                    font = TTFont(file)
                    if "gasp" not in font:
                        if args.remove:
                            action = "Skip (no 'gasp' to remove)"
                        elif args.add:
                            action = "Add 'gasp' table"
                        else:
                            action = "Skip (missing 'gasp')"
                    else:
                        existing = getattr(font["gasp"], "gaspRange", {}) or {}
                        if args.remove:
                            action = "Remove 'gasp' table"
                        elif not existing:
                            action = "Fill empty 'gasp'"
                        elif args.overwrite:
                            action = "Overwrite 'gasp' ranges"
                        else:
                            action = f"Skip (has {len(existing)} ranges)"
                    font.close()
            except Exception:
                action = "Error reading; skipping"

            desc = ", ".join([f"{rng}:0x{beh:04X}" for rng, beh in profile])
            cs.StatusIndicator("info").add_file(file).add_message("Would set:").emit()
            cs.StatusIndicator("info").add_field("Action", action).emit()
            cs.StatusIndicator("info").add_field("Profile", f"{{{desc}}}").emit()
        return

    # Confirmation prompt
    try:
        resp = (
            cs.prompt_input(
                f"About to modify [bold blue]{len(font_files)}[/] file(s). Proceed? (y/n): "
            )
            .strip()
            .lower()
        )
    except Exception:
        resp = ""
    if resp not in ("y", "yes"):
        cs.StatusIndicator("warning").with_explanation("Aborted by user").emit()
        sys.exit(2)

    cs.emit("")
    cs.StatusIndicator("info").add_message("Processing files...").emit()
    success_count = 0
    for file in font_files:
        if process_file(file, args.overwrite, args.add, args.remove, args.simple):
            success_count += 1

    cs.StatusIndicator("saved").add_message(
        f"Completed: {success_count}/{len(font_files)} files processed successfully"
    ).emit()


if __name__ == "__main__":
    main()
