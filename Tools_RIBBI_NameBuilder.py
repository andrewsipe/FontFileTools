#!/usr/bin/env python3
"""
RIBBI Variant Setter

Sets name IDs 1, 2, 4, 16, 17 consistently for Regular, Italic, Bold, Bold Italic
variants. By default, auto-detects subfamily from metrics and updates name IDs
accordingly without changing weight. With overrides -R/-I/-B/-BI, forces the
variant, sets OS/2.usWeightClass (400 for Regular/Italic, 700 for Bold/Bold Italic),
and synchronizes OS/2.fsSelection and head.macStyle bits.

Supported files: TTF, OTF, WOFF, WOFF2, TTX. Can process files and directories.
"""

import sys
import argparse
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord

import FontCore.core_console_styles as cs

try:
    from rich.console import Console

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if RICH_AVAILABLE:
    console = Console()
else:
    console = None

SUPPORTED_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2", ".ttx"}


# ---------- Constructors (aligned with existing scripts) ----------


def construct_family_name(
    family: str, modifier: str | None, style: str | None, slope: str | None
) -> str:
    parts: list[str] = []
    if family:
        parts.append(family)
    if modifier:
        parts.append(modifier)
    if style and style != "Regular":
        parts.append(style)
    if slope and slope != "Italic":
        parts.append(slope)
    return " ".join(parts)


def construct_full_name(
    family: str, modifier: str | None, style: str | None, slope: str | None
) -> str:
    parts: list[str] = []
    if family:
        parts.append(family)
    if modifier:
        parts.append(modifier)
    if style and style != "Regular":
        parts.append(style)
    if slope:
        parts.append(slope)
    return " ".join(parts)


def construct_typographic_subfamily(
    modifier: str | None, style: str | None, slope: str | None
) -> str:
    parts: list[str] = []
    if modifier:
        parts.append(modifier)
    if style:
        parts.append(style)
    if slope:
        parts.append(slope)
    if not parts:
        parts.append("Regular")
    return " ".join(parts)


# ---------- Metrics helpers ----------


def _format_bits_16(value: int) -> str:
    bits = f"{value:016b}"
    return f"{bits[:8]} {bits[8:]}"


def _parse_bits_16(value: str | None) -> int:
    if not value:
        return 0
    s = str(value).strip()
    try:
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        if len(s) == 17 and s[8] == " " and all(c in "01 " for c in s):
            return int(s.replace(" ", ""), 2)
        return int(s, 0)
    except Exception:
        return 0


def get_font_metrics_ttx(root) -> dict:
    metrics = {"is_italic": False, "is_bold": False}
    italic_angle_val = 0.0
    fs_selection_val = 0
    mac_style_val = 0

    post_table = root.find(".//post")
    if post_table is not None:
        italic_angle = post_table.find(".//italicAngle")
        if italic_angle is not None and italic_angle.get("value"):
            try:
                italic_angle_val = float(italic_angle.get("value"))
            except Exception:
                italic_angle_val = 0.0

    os2_table = root.find(".//OS_2")
    if os2_table is not None:
        fs_selection = os2_table.find(".//fsSelection")
        if fs_selection is not None and fs_selection.get("value"):
            fs_selection_val = _parse_bits_16(fs_selection.get("value"))
        weight_class = os2_table.find(".//usWeightClass")
        if weight_class is not None and weight_class.get("value"):
            try:
                weight_value = int(weight_class.get("value"))
                metrics["is_bold"] = weight_value == 700
            except Exception:
                pass

    head_table = root.find(".//head")
    if head_table is not None:
        mac_style = head_table.find(".//macStyle")
        if mac_style is not None and mac_style.get("value"):
            mac_style_val = _parse_bits_16(mac_style.get("value"))

    metrics["is_italic"] = bool(
        (fs_selection_val & 0x01) or (mac_style_val & 0x02) or (italic_angle_val != 0.0)
    )
    return metrics


def get_font_metrics_binary(font: TTFont) -> dict:
    metrics = {"is_italic": False, "is_bold": False}
    os2_table = font["OS/2"] if "OS/2" in font else None
    head_table = font["head"] if "head" in font else None
    post_table = font["post"] if "post" in font else None
    fs_selection = getattr(os2_table, "fsSelection", 0) if os2_table else 0
    mac_style = getattr(head_table, "macStyle", 0) if head_table else 0
    italic_angle = getattr(post_table, "italicAngle", 0.0) if post_table else 0.0
    weight_value = getattr(os2_table, "usWeightClass", 400) if os2_table else 400

    metrics["is_italic"] = bool(
        (fs_selection & 0x01) or (mac_style & 0x02) or (italic_angle != 0.0)
    )
    metrics["is_bold"] = weight_value == 700
    return metrics


def determine_subfamily(metrics: dict) -> str:
    if metrics["is_bold"] and metrics["is_italic"]:
        return "Bold Italic"
    if metrics["is_bold"]:
        return "Bold"
    if metrics["is_italic"]:
        return "Italic"
    return "Regular"


def compute_ribbi_flags(subfamily: str) -> tuple[int, int]:
    sub = (subfamily or "").strip().lower()
    is_bold = "bold" in sub
    is_italic = "italic" in sub
    fs_sel = 0
    if is_italic:
        fs_sel |= 0x0001
    if is_bold:
        fs_sel |= 0x0020
    if not is_bold and not is_italic:
        fs_sel |= 0x0040
    mac = 0
    if is_bold:
        mac |= 0x01
    if is_italic:
        mac |= 0x02
    return fs_sel, mac


# ---------- Dedup/order/whitespace helpers ----------


def deduplicate_name_records_ttx(
    name_table_elem, target_name_id: str, target_string: str
) -> None:
    matches = []
    for nr in list(name_table_elem.findall("namerecord")):
        if (
            nr.get("nameID") == target_name_id
            and nr.get("platformID") == "3"
            and nr.get("platEncID") == "1"
            and nr.get("langID") == "0x409"
        ):
            matches.append(nr)
    if not matches:
        return
    keep = matches[0]
    keep.text = f"\n      {target_string}\n    "
    for nr in matches[1:]:
        name_table_elem.remove(nr)


def deduplicate_name_records_binary(
    name_table, target_name_id: int, target_string: str
) -> None:
    filtered = []
    kept_one = False
    for record in list(name_table.names):
        if (
            record.nameID == target_name_id
            and record.platformID == 3
            and record.platEncID == 1
            and record.langID == 0x409
        ):
            if not kept_one:
                try:
                    current = (
                        record.toUnicode()
                        if hasattr(record, "toUnicode")
                        else str(record.string)
                    )
                except Exception:
                    current = str(record.string)
                if current != target_string:
                    record.string = target_string
                filtered.append(record)
                kept_one = True
            else:
                continue
        else:
            filtered.append(record)
    name_table.names = filtered


def _insert_namerecord_in_order(name_table, new_record) -> None:
    try:
        target_id = int(new_record.get("nameID"))
    except Exception:
        target_id = 999999
    insert_at = len(list(name_table))
    idx = 0
    for child in list(name_table):
        if child.tag != "namerecord":
            idx += 1
            continue
        try:
            child_id = int(child.get("nameID", "999999"))
        except Exception:
            child_id = 999999
        if child_id > target_id:
            insert_at = idx
            break
        idx += 1
    name_table.insert(insert_at, new_record)


def _adjust_ttx_whitespace(name_table) -> None:
    name_table.text = "\n    "
    children = [c for c in list(name_table) if c.tag == "namerecord"]
    total = len(children)
    for i, child in enumerate(children):
        child.tail = "\n    " if i < total - 1 else "\n  "


# ---------- Update helpers ----------


def update_ttx_name_record(name_table, name_id_str: str, new_value: str) -> None:
    nr = name_table.find(
        f'.//namerecord[@nameID="{name_id_str}"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
    )
    if nr is not None:
        old_text = nr.text.strip() if nr.text else ""
        if old_text != new_value:
            nr.text = f"\n      {new_value}\n    "
            cs.StatusIndicator("updated").add_message(
                f"nameID={name_id_str}"
            ).add_values(old_value=old_text, new_value=new_value).emit()
    else:
        new_record = ET.Element("namerecord")
        new_record.set("nameID", name_id_str)
        new_record.set("platformID", "3")
        new_record.set("platEncID", "1")
        new_record.set("langID", "0x409")
        new_record.text = f"\n      {new_value}\n    "
        _insert_namerecord_in_order(name_table, new_record)
        cs.StatusIndicator("updated").add_message(
            f"Created new nameID={name_id_str}"
        ).add_field("value", new_value).emit()


def update_binary_name_record(name_table, name_id: int, new_value: str) -> None:
    for record in name_table.names:
        if (
            record.nameID == name_id
            and record.platformID == 3
            and record.platEncID == 1
            and record.langID == 0x409
        ):
            try:
                old_text = (
                    record.toUnicode()
                    if hasattr(record, "toUnicode")
                    else str(record.string)
                )
            except Exception:
                old_text = str(record.string)
            if old_text != new_value:
                record.string = new_value
                cs.StatusIndicator("updated").add_message(
                    f"nameID={name_id}"
                ).add_values(old_value=old_text, new_value=new_value).emit()
            return
    new_record = NameRecord()
    new_record.nameID = name_id
    new_record.platformID = 3
    new_record.platEncID = 1
    new_record.langID = 0x409
    new_record.string = new_value
    name_table.names.append(new_record)
    cs.StatusIndicator("updated").add_message(
        f"Created new nameID={name_id}: {new_value}"
    ).emit()


# ---------- Core processing ----------


def _apply_ribbi_flags_ttx(root, subfamily: str) -> None:
    fs_bits_set, mac_bits_set = compute_ribbi_flags(subfamily)
    os2_table = root.find(".//OS_2")
    if os2_table is None:
        os2_table = ET.SubElement(root, "OS_2")
    fs_sel_elem = os2_table.find(".//fsSelection")
    current_fs = (
        _parse_bits_16(fs_sel_elem.get("value")) if fs_sel_elem is not None else 0
    )
    mask_clear_fs = 0x0001 | 0x0020 | 0x0040
    new_fs = (current_fs & ~mask_clear_fs) | fs_bits_set
    if fs_sel_elem is None:
        fs_sel_elem = ET.SubElement(os2_table, "fsSelection")
    fs_sel_elem.set("value", _format_bits_16(new_fs))

    head_table = root.find(".//head")
    if head_table is None:
        head_table = ET.SubElement(root, "head")
    mac_style_elem = head_table.find(".//macStyle")
    current_mac = (
        _parse_bits_16(mac_style_elem.get("value")) if mac_style_elem is not None else 0
    )
    mask_clear_mac = 0x01 | 0x02
    new_mac = (current_mac & ~mask_clear_mac) | mac_bits_set
    if mac_style_elem is None:
        mac_style_elem = ET.SubElement(head_table, "macStyle")
    mac_style_elem.set("value", _format_bits_16(new_mac))


def _apply_ribbi_flags_binary(font: TTFont, subfamily: str) -> None:
    try:
        fs_bits_set, mac_bits_set = compute_ribbi_flags(subfamily)
        if "OS/2" in font and hasattr(font["OS/2"], "fsSelection"):
            current_fs = getattr(font["OS/2"], "fsSelection", 0)
            mask_clear_fs = 0x0001 | 0x0020 | 0x0040
            font["OS/2"].fsSelection = (current_fs & ~mask_clear_fs) | fs_bits_set
        if "head" in font and hasattr(font["head"], "macStyle"):
            current_mac = getattr(font["head"], "macStyle", 0)
            mask_clear_mac = 0x01 | 0x02
            font["head"].macStyle = (current_mac & ~mask_clear_mac) | mac_bits_set
    except Exception:
        pass


def _compute_id_values(
    family: str,
    modifier: str | None,
    target_subfamily: str,
    is_italic: bool,
    slope_override: str | None,
) -> tuple[str, str, str, str]:
    # Returns (id1, id2, id4, id17). ID16 will be family (base)
    if target_subfamily == "Bold" or target_subfamily == "Bold Italic":
        style_for_id = "Bold"
    else:
        style_for_id = "Regular"
    # ID1 MUST NOT include slope, even if oblique override
    id1_val = construct_family_name(family, modifier, style_for_id, None)
    # ID2 equals subfamily
    id2_val = target_subfamily
    # ID4 includes slope
    id4_val = construct_full_name(
        family,
        modifier,
        style_for_id,
        (
            slope_override
            if (is_italic and slope_override)
            else ("Italic" if is_italic else None)
        ),
    )
    # ID17 typographic subfamily: modifier + style + slope (ensure Regular baseline)
    style_for_typo = style_for_id
    slope_for_typo = (
        slope_override
        if (is_italic and slope_override)
        else ("Italic" if is_italic else None)
    )
    id17_val = construct_typographic_subfamily(modifier, style_for_typo, slope_for_typo)
    return id1_val, id2_val, id4_val, id17_val


def process_ttx_file(
    filepath: str,
    family: str,
    modifier: str | None,
    override_subfamily: str | None,
    slope_override: str | None,
) -> bool:
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        metrics = get_font_metrics_ttx(root)
        detected_subfamily = determine_subfamily(metrics)
        target_subfamily = (
            override_subfamily if override_subfamily else detected_subfamily
        )
        is_italic = "Italic" in target_subfamily

        # Compute ID values
        id1_val, id2_val, id4_val, id17_val = _compute_id_values(
            family, modifier, target_subfamily, is_italic, slope_override
        )

        name_table = root.find(".//name")
        if name_table is None:
            cs.StatusIndicator("warning").add_file(filepath).with_explanation(
                "No name table found"
            ).emit()
            return False

        update_ttx_name_record(name_table, "1", id1_val)
        update_ttx_name_record(name_table, "2", id2_val)
        update_ttx_name_record(name_table, "4", id4_val)
        update_ttx_name_record(name_table, "16", family)
        update_ttx_name_record(name_table, "17", id17_val)

        deduplicate_name_records_ttx(name_table, "1", id1_val)
        deduplicate_name_records_ttx(name_table, "2", id2_val)
        deduplicate_name_records_ttx(name_table, "4", id4_val)
        deduplicate_name_records_ttx(name_table, "16", family)
        deduplicate_name_records_ttx(name_table, "17", id17_val)
        _adjust_ttx_whitespace(name_table)

        # RIBBI flags
        _apply_ribbi_flags_ttx(root, target_subfamily)

        # usWeightClass when override provided
        if override_subfamily:
            os2_table = root.find(".//OS_2")
            if os2_table is None:
                os2_table = ET.SubElement(root, "OS_2")
            weight_elem = os2_table.find(".//usWeightClass")
            if weight_elem is None:
                weight_elem = ET.SubElement(os2_table, "usWeightClass")
            weight_val = "700" if "Bold" in override_subfamily else "400"
            weight_elem.set("value", weight_val)

        tree.write(filepath, encoding="utf-8", xml_declaration=True)
        return True
    except Exception as e:
        cs.StatusIndicator("error").add_file(filepath).with_explanation(
            f"Error processing TTX file: {e}"
        ).emit()
        return False


def process_binary_font(
    filepath: str,
    family: str,
    modifier: str | None,
    override_subfamily: str | None,
    slope_override: str | None,
) -> bool:
    try:
        font = TTFont(filepath)
        if "name" not in font:
            cs.StatusIndicator("warning").add_file(filepath).with_explanation(
                "No name table found"
            ).emit()
            return False

        metrics = get_font_metrics_binary(font)
        detected_subfamily = determine_subfamily(metrics)
        target_subfamily = (
            override_subfamily if override_subfamily else detected_subfamily
        )
        is_italic = "Italic" in target_subfamily

        id1_val, id2_val, id4_val, id17_val = _compute_id_values(
            family, modifier, target_subfamily, is_italic, slope_override
        )

        name_table = font["name"]
        update_binary_name_record(name_table, 1, id1_val)
        update_binary_name_record(name_table, 2, id2_val)
        update_binary_name_record(name_table, 4, id4_val)
        update_binary_name_record(name_table, 16, family)
        update_binary_name_record(name_table, 17, id17_val)

        deduplicate_name_records_binary(name_table, 1, id1_val)
        deduplicate_name_records_binary(name_table, 2, id2_val)
        deduplicate_name_records_binary(name_table, 4, id4_val)
        deduplicate_name_records_binary(name_table, 16, family)
        deduplicate_name_records_binary(name_table, 17, id17_val)

        _apply_ribbi_flags_binary(font, target_subfamily)

        # usWeightClass when override provided
        if override_subfamily and "OS/2" in font:
            target_weight = 700 if "Bold" in override_subfamily else 400
            try:
                font["OS/2"].usWeightClass = target_weight
            except Exception:
                pass

        font.save(filepath)
        font.close()
        return True
    except Exception as e:
        cs.StatusIndicator("error").add_file(filepath).with_explanation(
            f"Error processing font file: {e}"
        ).emit()
        return False


def process_file(
    filepath: str,
    family: str,
    modifier: str | None,
    override_subfamily: str | None,
    slope_override: str | None,
) -> bool:
    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        cs.StatusIndicator("warning").add_file(filepath).with_explanation(
            "Skipping unsupported file"
        ).emit()
        return False
    cs.StatusIndicator("info").add_file(filepath).add_message("Processing").emit()
    if ext == ".ttx":
        return process_ttx_file(
            filepath, family, modifier, override_subfamily, slope_override
        )
    return process_binary_font(
        filepath, family, modifier, override_subfamily, slope_override
    )


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
        description="Set nameID 1/2/4/16/17 for RIBBI variants; sync RIBBI flags and weight",
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
        "-f",
        "--family",
        default="FontName",
        help="Family name for ID1/4/16 (default: 'FontName')",
    )
    parser.add_argument(
        "-m", "--modifier", help="Optional modifier (e.g., 'Condensed', 'Extended')"
    )
    parser.add_argument(
        "-sl",
        "--slope",
        help="Optional slope override for italic variants (e.g., 'Oblique'); affects ID4 and ID17 when used with -I or -BI",
    )

    # Mutually exclusive RIBBI overrides
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-R",
        "--Regular",
        action="store_true",
        help="Force Regular (sets usWeightClass=400)",
    )
    group.add_argument(
        "-I",
        "--Italic",
        action="store_true",
        help="Force Italic (sets usWeightClass=400)",
    )
    group.add_argument(
        "-B", "--Bold", action="store_true", help="Force Bold (sets usWeightClass=700)"
    )
    group.add_argument(
        "-BI",
        "--BoldItalic",
        action="store_true",
        help="Force Bold Italic (sets usWeightClass=700)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # fonttools presence
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
        f"Found {len(font_files)} font file(s) to process:"
    ).emit()
    for file in font_files:
        cs.StatusIndicator("info").add_file(file).emit()

    def _override_to_subfamily() -> str | None:
        if args.BoldItalic:
            return "Bold Italic"
        if args.Bold:
            return "Bold"
        if args.Italic:
            return "Italic"
        if args.Regular:
            return "Regular"
        return None

    override_subfamily = _override_to_subfamily()
    slope_override = args.slope if (args.Italic or args.BoldItalic) else None

    # Preview
    if args.dry_run:
        cs.StatusIndicator("warning").with_explanation(
            "Dry run mode - no changes will be made"
        ).emit()
        for file in font_files:
            try:
                ext = Path(file).suffix.lower()
                if ext == ".ttx":
                    tree = ET.parse(file)
                    root = tree.getroot()
                    metrics = get_font_metrics_ttx(root)
                else:
                    font = TTFont(file)
                    metrics = get_font_metrics_binary(font)
                    font.close()
                detected = determine_subfamily(metrics)
                target = override_subfamily if override_subfamily else detected
                is_italic = "Italic" in target
                id1_val, id2_val, id4_val, id17_val = _compute_id_values(
                    args.family, args.modifier, target, is_italic, slope_override
                )
                cs.StatusIndicator("info", dry_run=True).add_file(file).add_message(
                    "Would set:"
                ).emit()
                cs.StatusIndicator("info", dry_run=True).add_field(
                    "nameID=1", id1_val
                ).emit()
                cs.StatusIndicator("info", dry_run=True).add_field(
                    "nameID=2", id2_val
                ).emit()
                cs.StatusIndicator("info", dry_run=True).add_field(
                    "nameID=4", id4_val
                ).emit()
                cs.StatusIndicator("info", dry_run=True).add_field(
                    "nameID=16", args.family
                ).emit()
                cs.StatusIndicator("info", dry_run=True).add_field(
                    "nameID=17", id17_val
                ).emit()
                if override_subfamily:
                    w = 700 if "Bold" in override_subfamily else 400
                    cs.StatusIndicator("info", dry_run=True).add_field(
                        "usWeightClass", w
                    ).emit()
            except Exception as e:
                cs.StatusIndicator("warning").add_file(file).with_explanation(
                    f"Dry-run preview failed: {e}"
                ).emit()
        return

    # Confirmation prompt
    try:
        resp = (
            input(f"\nAbout to modify {len(font_files)} file(s). Proceed? [y/N]: ")
            .strip()
            .lower()
        )
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        cs.StatusIndicator("warning").with_explanation("Aborted by user").emit()
        sys.exit(2)

    cs.StatusIndicator("info").add_message("Processing files...").emit()
    success_count = 0
    for file in font_files:
        if process_file(
            file, args.family, args.modifier, override_subfamily, slope_override
        ):
            success_count += 1

    cs.StatusIndicator("saved").add_message(
        f"Completed: {success_count}/{len(font_files)} files processed successfully"
    ).emit()


if __name__ == "__main__":
    main()
