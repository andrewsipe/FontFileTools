#!/usr/bin/env python3
"""
RIBBI Cleanup Script

Cleans naming across core RIBBI family (Regular, Regular Italic, Bold, Bold Italic):

- Weight checks:
  • usWeightClass == 400 → Regular
  • usWeightClass == 700 → Bold

- ID1 (Font Family): remove style tokens "Regular", "Bold", "Italic", "Oblique".
  • Do NOT remove "Bold" when part of safe words: ExtraBold, SemiBold, DemiBold, UltraBold
  • If tokenized "Ultra Bold" exists, only remove the "Bold" word if weight==700

- ID2 (Subfamily): same behavior as ID2 replacer → set to one of
  "Regular", "Italic", "Bold", "Bold Italic" based on usWeightClass and italic detection.
  Also update OS/2.fsSelection and head.macStyle RIBBI bits non-destructively.

- ID3 (Unique identifier): ensure the 3rd segment (postscript-like) contains "-Regular"
  when weight==400. If italic and weight==400 ensure "-RegularItalic". If already present, no change.

- ID4 (Full name): remove the word "Regular" regardless of weight.

- ID6 (PostScript name): ensure "-Regular" for weight==400. If italic and weight==400 ensure
  "-RegularItalic" (insert Regular before Italic if necessary). Sanitizes name but keeps '?' and '!'.

- ID17 (Typographic Subfamily): for weight==400 ensure "Regular" present.
  If italic and weight==400 set to "Regular Italic"; if weight==700 set to "Bold" or "Bold Italic".

Supports TTF, OTF, WOFF, WOFF2, TTX; files, dirs, and -r recursion.
"""

import sys
import argparse
import glob
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._n_a_m_e import NameRecord

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
import sys
_project_root = Path(__file__).parent
while not (_project_root / "FontCore").exists() and _project_root.parent != _project_root:
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
SAFE_BOLD_COMPOUNDS = {"extrabold", "semibold", "demibold", "ultrabold"}


def _format_bits_16(value: int) -> str:
    bits = f"{value:016b}"
    return f"{bits[:8]} {bits[8:]}"


def _parse_bits_16(value: str) -> int:
    if value is None:
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


def _is_italic_ttx(root: ET.Element) -> bool:
    italic_angle = 0.0
    fs_selection_val = 0
    mac_style_val = 0

    post_table = root.find(".//post")
    if post_table is not None:
        italic_angle_elem = post_table.find(".//italicAngle")
        if italic_angle_elem is not None and italic_angle_elem.get("value"):
            try:
                italic_angle = float(italic_angle_elem.get("value"))
            except Exception:
                italic_angle = 0.0

    os2_table = root.find(".//OS_2")
    if os2_table is not None:
        fs_sel_elem = os2_table.find(".//fsSelection")
        if fs_sel_elem is not None:
            fs_selection_val = _parse_bits_16(fs_sel_elem.get("value"))

    head_table = root.find(".//head")
    if head_table is not None:
        mac_style_elem = head_table.find(".//macStyle")
        if mac_style_elem is not None:
            mac_style_val = _parse_bits_16(mac_style_elem.get("value"))

    return bool(
        (fs_selection_val & 0x0001) or (mac_style_val & 0x02) or (italic_angle != 0.0)
    )


def _is_italic_binary(font: TTFont) -> bool:
    os2_table = font["OS/2"] if "OS/2" in font else None
    head_table = font["head"] if "head" in font else None
    post_table = font["post"] if "post" in font else None
    fs_selection = getattr(os2_table, "fsSelection", 0) if os2_table else 0
    mac_style = getattr(head_table, "macStyle", 0) if head_table else 0
    italic_angle = getattr(post_table, "italicAngle", 0.0) if post_table else 0.0
    return bool((fs_selection & 0x0001) or (mac_style & 0x02) or (italic_angle != 0.0))


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


def _cleanup_id1_text(text: str, weight: int) -> str:
    if not text:
        return text
    original = text
    # Remove Regular, Italic, Oblique as whole words
    for token in ["Regular", "Italic", "Oblique"]:
        text = re.sub(rf"(?i)\b{token}\b", "", text)

    # Handle Bold
    # 1) do not touch compounds like ExtraBold, SemiBold, DemiBold, UltraBold
    def _preserve_compounds(m: re.Match) -> str:
        word = m.group(0)
        if word.lower() in SAFE_BOLD_COMPOUNDS:
            return word
        # Not a safe compound; leave as-is for now
        return word

    text = re.sub(
        r"(?i)\b(ExtraBold|SemiBold|DemiBold|UltraBold)\b", _preserve_compounds, text
    )

    # 2) remove standalone 'Bold' tokens
    # Special case: 'Ultra Bold' -> only remove 'Bold' if weight == 700
    def _remove_bold_token(m: re.Match) -> str:
        before = m.group(1) or ""
        after = m.group(3) or ""
        # Check if pattern is 'Ultra Bold'
        if before.strip().lower().endswith("ultra"):
            return before + ("" if weight == 700 else " Bold") + after
        # General standalone 'Bold' token → remove
        return before + after

    # Pattern captures optional preceding word and following space
    text = re.sub(r"(?i)(\b\w+\s+)?\b(Bold)\b(\s+)?", _remove_bold_token, text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text if text != original else text


def _sanitize_postscript(name: str) -> str:
    name = name.replace(" ", "")
    return re.sub(r"[^A-Za-z0-9\-\._\?\!]", "-", name)


def _ensure_regular_in_ps(ps: str, italic: bool) -> str:
    # Add -Regular (or -RegularItalic) if missing
    if italic:
        if re.search(r"(?i)-regularitalic$", ps):
            return ps
        if ps.lower().endswith("italic"):
            # Insert Regular before Italic
            base = re.sub(r"(?i)italic$", "", ps)
            return base + "RegularItalic"
        if not re.search(r"(?i)-regular$", ps):
            return ps + "RegularItalic"
        return ps + "Italic" if ps.lower().endswith("-regular") else ps
    else:
        if re.search(r"(?i)-regular$", ps):
            return ps
        return ps + "-Regular" if not ps.lower().endswith("-regular") else ps


def _ensure_regular_in_filename_segment(name3: str, italic: bool) -> str:
    # Ensure third segment has -Regular or -RegularItalic
    parts = name3.split(";")
    if len(parts) != 3:
        return name3
    third = parts[2]
    if italic:
        if re.search(r"(?i)-regularitalic$", third):
            return name3
        if third.lower().endswith("italic"):
            base = re.sub(r"(?i)italic$", "", third)
            third_new = base + "RegularItalic"
        else:
            if re.search(r"(?i)-regular$", third):
                third_new = third + "Italic"
            else:
                third_new = third + "-RegularItalic"
    else:
        if re.search(r"(?i)-regular$", third):
            return name3
        third_new = third + "-Regular"
    parts[2] = third_new
    return ";".join(parts)


def _fix_bold_in_filename_segment(name3: str, italic: bool) -> str:
    """When weight is Bold (700), remove any trailing -Regular after -Bold.
    Handles cases like 'Name-Bold-Regular' and 'Name-Bold-RegularItalic'.
    """
    parts = name3.split(";")
    if len(parts) != 3:
        return name3
    third = parts[2]
    # Replace Bold-RegularItalic → BoldItalic, then Bold-Regular → Bold
    third_new = re.sub(r"(?i)-Bold-?RegularItalic$", "-BoldItalic", third)
    third_new = re.sub(r"(?i)-Bold-?Regular$", "-Bold", third_new)
    parts[2] = third_new
    return ";".join(parts)


def _has_slope_suffix(token: str) -> bool:
    return bool(re.search(r"(?i)(Oblique|Italic)$", token or ""))


def _collapse_regular_duplicates(value: str) -> str:
    if not value:
        return value
    s = value
    # Collapse double Regular
    s = re.sub(r"(?i)RegularRegular", "Regular", s)
    s = re.sub(r"(?i)-RegularRegular", "-Regular", s)
    s = re.sub(r"(?i)RegularRegularItalic$", "RegularItalic", s)
    s = re.sub(r"(?i)ObliqueRegularItalic$", "ObliqueItalic", s)
    return s


def _get_us_weight_ttx(root: ET.Element) -> int:
    os2 = root.find(".//OS_2")
    if os2 is None:
        return 400
    wt = os2.find(".//usWeightClass")
    try:
        return int(wt.get("value")) if wt is not None else 400
    except Exception:
        return 400


def _set_id_text_ttx(
    root: ET.Element, name_id: str, text_value: str, filepath: str
) -> None:
    name_table = root.find(".//name")
    if name_table is None:
        cs.StatusIndicator("error").add_file(filepath).with_explanation(
            "No name table found"
        ).emit()
        return
    nr = name_table.find(
        f'.//namerecord[@nameID="{name_id}"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
    )
    if nr is not None:
        old = nr.text.strip() if nr.text else ""
        if old != text_value:
            nr.text = f"\n      {text_value}\n    "
            cs.StatusIndicator("updated").add_file(filepath).add_message(
                f"nameID={name_id}"
            ).add_values(old_value=old, new_value=text_value).emit()
    else:
        new_record = ET.Element("namerecord")
        new_record.set("nameID", str(name_id))
        new_record.set("platformID", "3")
        new_record.set("platEncID", "1")
        new_record.set("langID", "0x409")
        new_record.text = f"\n      {text_value}\n    "
        _insert_namerecord_in_order(name_table, new_record)
        cs.StatusIndicator("updated").add_file(filepath).add_message(
            f"Created new nameID={name_id}"
        ).add_field("value", text_value).emit()
    deduplicate_name_records_ttx(name_table, str(name_id), text_value)
    _adjust_ttx_whitespace(name_table)


def _update_id2_and_flags_ttx(
    root: ET.Element, new_subfamily: str, filepath: str
) -> None:
    _set_id_text_ttx(root, "2", new_subfamily, filepath)
    os2 = root.find(".//OS_2")
    if os2 is None:
        os2 = ET.SubElement(root, "OS_2")
    fs_sel = os2.find(".//fsSelection")
    current_fs = _parse_bits_16(fs_sel.get("value")) if fs_sel is not None else 0
    # compute bits
    sub = new_subfamily.lower()
    fs_bits = (
        (0x0001 if "italic" in sub else 0)
        | (0x0020 if "bold" in sub else 0)
        | (0x0040 if ("bold" not in sub and "italic" not in sub) else 0)
    )
    mask_clear_fs = 0x0001 | 0x0020 | 0x0040
    new_fs = (current_fs & ~mask_clear_fs) | fs_bits
    if fs_sel is None:
        fs_sel = ET.SubElement(os2, "fsSelection")
    fs_sel.set("value", _format_bits_16(new_fs))

    head = root.find(".//head")
    if head is None:
        head = ET.SubElement(root, "head")
    mac_style = head.find(".//macStyle")
    current_mac = _parse_bits_16(mac_style.get("value")) if mac_style is not None else 0
    mac_bits = (0x01 if "bold" in sub else 0) | (0x02 if "italic" in sub else 0)
    mask_clear_mac = 0x01 | 0x02
    new_mac = (current_mac & ~mask_clear_mac) | mac_bits
    if mac_style is None:
        mac_style = ET.SubElement(head, "macStyle")
    mac_style.set("value", _format_bits_16(new_mac))


def process_ttx_file(filepath: str, force_weight: int | None) -> bool:
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        changed = False

        weight = _get_us_weight_ttx(root)
        # Apply forced weight if requested
        if force_weight is not None and force_weight in (400, 700):
            os2 = root.find(".//OS_2")
            if os2 is None:
                os2 = ET.SubElement(root, "OS_2")
            us_wt = os2.find(".//usWeightClass")
            old = us_wt.get("value") if us_wt is not None else None
            if us_wt is None:
                us_wt = ET.SubElement(os2, "usWeightClass")
            if str(old) != str(force_weight):
                cs.StatusIndicator("updated").add_file(filepath).add_message(
                    "OS/2.usWeightClass"
                ).add_values(
                    old_value=str(old) if old is not None else "<missing>",
                    new_value=str(force_weight),
                ).emit()
                us_wt.set("value", str(force_weight))
                changed = True
            weight = force_weight
        italic = _is_italic_ttx(root)
        allow = weight in (400, 700)

        # ID1 cleanup (only when allowed; only update existing)
        name_table = root.find(".//name")
        if allow and name_table is not None:
            nr1 = name_table.find(
                './/namerecord[@nameID="1"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
            )
            if nr1 is not None and nr1.text is not None:
                old1 = nr1.text.strip()
                new1 = _cleanup_id1_text(old1, weight)
                if new1 != old1:
                    nr1.text = f"\n      {new1}\n    "
                    cs.StatusIndicator("updated").add_file(filepath).add_message(
                        "nameID=1"
                    ).add_values(old_value=old1, new_value=new1).emit()
                    changed = True

        # ID2 (only when allowed)
        if allow:
            if weight == 700:
                sub = "Bold Italic" if italic else "Bold"
            else:
                sub = "Italic" if italic else "Regular"
            _update_id2_and_flags_ttx(root, sub, filepath)
            changed = True

        # ID4 remove 'Regular'
        if name_table is not None:
            nr4 = name_table.find(
                './/namerecord[@nameID="4"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
            )
            if nr4 is not None and nr4.text:
                old4 = nr4.text.strip()
                new4 = re.sub(r"(?i)\bRegular\b", "", old4)
                new4 = re.sub(r"\s+", " ", new4).strip()
                if new4 != old4:
                    nr4.text = f"\n      {new4}\n    "
                    cs.StatusIndicator("updated").add_file(filepath).add_message(
                        "nameID=4"
                    ).add_values(old_value=old4, new_value=new4).emit()
                    changed = True

        # ID3 ensure -Regular / -RegularItalic for weight 400
        if name_table is not None and weight == 400:
            nr3 = name_table.find(
                './/namerecord[@nameID="3"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
            )
            if nr3 is not None and nr3.text:
                old3 = nr3.text.strip()
                new3 = _ensure_regular_in_filename_segment(old3, italic)
                if new3 != old3:
                    nr3.text = f"\n      {new3}\n    "
                    cs.StatusIndicator("updated").add_file(filepath).add_message(
                        "nameID=3"
                    ).add_values(old_value=old3, new_value=new3).emit()
                    changed = True

        # ID3 remove '-Regular' after Bold for weight 700
        if name_table is not None and weight == 700:
            nr3b = name_table.find(
                './/namerecord[@nameID="3"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
            )
            if nr3b is not None and nr3b.text:
                old3b = nr3b.text.strip()
                new3b = _fix_bold_in_filename_segment(old3b, italic)
                if new3b != old3b:
                    nr3b.text = f"\n      {new3b}\n    "
                    cs.StatusIndicator("updated").add_file(filepath).add_message(
                        "nameID=3"
                    ).add_values(old_value=old3b, new_value=new3b).emit()
                    changed = True

        # ID6 ensure -Regular / -RegularItalic for weight 400 in PS name
        if name_table is not None and weight == 400:
            nr6 = name_table.find(
                './/namerecord[@nameID="6"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
            )
            if nr6 is not None and nr6.text:
                old6 = nr6.text.strip()
                ps = _sanitize_postscript(old6)
                newps = _ensure_regular_in_ps(ps, italic)
                if newps != old6:
                    nr6.text = f"\n      {newps}\n    "
                    cs.StatusIndicator("updated").add_file(filepath).add_message(
                        "nameID=6"
                    ).add_values(old_value=old6, new_value=newps).emit()
                    changed = True

        # ID6 remove '-Regular' after Bold for weight 700 in PS name
        if name_table is not None and weight == 700:
            nr6b = name_table.find(
                './/namerecord[@nameID="6"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
            )
            if nr6b is not None and nr6b.text:
                old6b = nr6b.text.strip()
                psb = _sanitize_postscript(old6b)
                new6b = re.sub(r"(?i)-Bold-?RegularItalic$", "-BoldItalic", psb)
                new6b = re.sub(r"(?i)-Bold-?Regular$", "-Bold", new6b)
                if new6b != old6b:
                    nr6b.text = f"\n      {new6b}\n    "
                    cs.StatusIndicator("updated").add_file(filepath).add_message(
                        "nameID=6"
                    ).add_values(old_value=old6b, new_value=new6b).emit()
                    changed = True

        # ID17 set per RIBBI
        if name_table is not None:
            nr17 = name_table.find(
                './/namerecord[@nameID="17"][@platformID="3"][@platEncID="1"][@langID="0x409"]'
            )
            if weight == 700:
                id17_val = "Bold Italic" if italic else "Bold"
            else:
                id17_val = "Regular Italic" if italic else "Regular"
            if nr17 is not None and nr17.text:
                old17 = nr17.text.strip()
                if old17 != id17_val:
                    nr17.text = f"\n      {id17_val}\n    "
                    cs.StatusIndicator("updated").add_file(filepath).add_message(
                        "nameID=17"
                    ).add_values(old_value=old17, new_value=id17_val).emit()
                    changed = True
            else:
                _set_id_text_ttx(root, "17", id17_val, filepath)
                changed = True

        if changed:
            tree.write(filepath, encoding="utf-8", xml_declaration=True)
        return True
    except Exception as e:
        cs.StatusIndicator("error").add_file(filepath).with_explanation(
            f"Error processing TTX file: {e}"
        ).emit()
        return False


def process_binary_font(filepath: str, force_weight: int | None) -> bool:
    try:
        font = TTFont(filepath)
        changed = False

        weight = getattr(font["OS/2"], "usWeightClass", 400) if "OS/2" in font else 400
        if force_weight is not None and force_weight in (400, 700) and "OS/2" in font:
            old_w = getattr(font["OS/2"], "usWeightClass", None)
            if old_w != force_weight:
                cs.StatusIndicator("updated").add_file(filepath).add_message(
                    "OS/2.usWeightClass"
                ).add_values(old_value=str(old_w), new_value=str(force_weight)).emit()
                font["OS/2"].usWeightClass = force_weight
            weight = force_weight
        italic = _is_italic_binary(font)

        # ID1 cleanup
        if "name" in font:
            name_table = font["name"]
            for record in list(name_table.names):
                if (
                    record.nameID == 1
                    and record.platformID == 3
                    and record.platEncID == 1
                    and record.langID == 0x409
                ):
                    try:
                        old1 = (
                            record.toUnicode()
                            if hasattr(record, "toUnicode")
                            else str(record.string)
                        )
                    except Exception:
                        old1 = str(record.string)
                    new1 = _cleanup_id1_text(old1, weight)
                    if new1 != old1:
                        record.string = new1
                        cs.StatusIndicator("updated").add_file(filepath).add_message(
                            "nameID=1"
                        ).add_values(old_value=old1, new_value=new1).emit()
                        changed = True
                    break

        # ID2 + flags (only when allowed)
        allow = weight in (400, 700)
        if weight == 700:
            sub = "Bold Italic" if italic else "Bold"
        else:
            sub = "Italic" if italic else "Regular"
        if allow and "name" in font:
            name_table = font["name"]
            found2 = False
            for record in name_table.names:
                if (
                    record.nameID == 2
                    and record.platformID == 3
                    and record.platEncID == 1
                    and record.langID == 0x409
                ):
                    try:
                        old2 = (
                            record.toUnicode()
                            if hasattr(record, "toUnicode")
                            else str(record.string)
                        )
                    except Exception:
                        old2 = str(record.string)
                    if old2 != sub:
                        record.string = sub
                        cs.StatusIndicator("updated").add_file(filepath).add_message(
                            "nameID=2"
                        ).add_values(old_value=old2, new_value=sub).emit()
                        changed = True
                    found2 = True
                    break
            if not found2:
                nr = NameRecord()
                nr.nameID = 2
                nr.platformID = 3
                nr.platEncID = 1
                nr.langID = 0x409
                nr.string = sub
                name_table.names.append(nr)
                cs.StatusIndicator("updated").add_file(filepath).add_message(
                    f"Created new nameID=2"
                ).add_field("value", sub).emit()
                changed = True

        # Flags
        try:
            if "OS/2" in font:
                os2 = font["OS/2"]
                current_fs = getattr(os2, "fsSelection", 0)
                fs_bits = (
                    (0x0001 if "italic" in sub.lower() else 0)
                    | (0x0020 if "bold" in sub.lower() else 0)
                    | (
                        0x0040
                        if ("bold" not in sub.lower() and "italic" not in sub.lower())
                        else 0
                    )
                )
                mask_clear_fs = 0x0001 | 0x0020 | 0x0040
                if allow:
                    os2.fsSelection = (current_fs & ~mask_clear_fs) | fs_bits
            if "head" in font:
                head = font["head"]
                current_mac = getattr(head, "macStyle", 0)
                mac_bits = (0x01 if "bold" in sub.lower() else 0) | (
                    0x02 if "italic" in sub.lower() else 0
                )
                mask_clear_mac = 0x01 | 0x02
                if allow:
                    head.macStyle = (current_mac & ~mask_clear_mac) | mac_bits
        except Exception:
            pass

        # ID4 remove Regular (only when allowed)
        if allow and "name" in font:
            name_table = font["name"]
            for record in list(name_table.names):
                if (
                    record.nameID == 4
                    and record.platformID == 3
                    and record.platEncID == 1
                    and record.langID == 0x409
                ):
                    try:
                        old4 = (
                            record.toUnicode()
                            if hasattr(record, "toUnicode")
                            else str(record.string)
                        )
                    except Exception:
                        old4 = str(record.string)
                    new4 = re.sub(r"(?i)\bRegular\b", "", old4)
                    new4 = re.sub(r"\s+", " ", new4).strip()
                    if new4 != old4:
                        record.string = new4
                        cs.StatusIndicator("updated").add_file(filepath).add_message(
                            "nameID=4"
                        ).add_values(old_value=old4, new_value=new4).emit()
                        changed = True
                    break

        # ID3 third segment ensure -Regular/-RegularItalic when weight==400 (only update existing)
        if allow and "name" in font and weight == 400:
            name_table = font["name"]
            for record in list(name_table.names):
                if (
                    record.nameID == 3
                    and record.platformID == 3
                    and record.platEncID == 1
                    and record.langID == 0x409
                ):
                    try:
                        old3 = (
                            record.toUnicode()
                            if hasattr(record, "toUnicode")
                            else str(record.string)
                        )
                    except Exception:
                        old3 = str(record.string)
                    new3 = _ensure_regular_in_filename_segment(old3, italic)
                    if new3 != old3:
                        record.string = new3
                        cs.StatusIndicator("updated").add_file(filepath).add_message(
                            "nameID=3"
                        ).add_values(old_value=old3, new_value=new3).emit()
                        changed = True
                    break

        # ID6 ensure -Regular/-RegularItalic when weight==400 (only update existing)
        if allow and "name" in font and weight == 400:
            name_table = font["name"]
            for record in list(name_table.names):
                if (
                    record.nameID == 6
                    and record.platformID == 3
                    and record.platEncID == 1
                    and record.langID == 0x409
                ):
                    try:
                        old6 = (
                            record.toUnicode()
                            if hasattr(record, "toUnicode")
                            else str(record.string)
                        )
                    except Exception:
                        old6 = str(record.string)
                    ps = _sanitize_postscript(old6)
                    new6 = _ensure_regular_in_ps(ps, italic)
                    if new6 != old6:
                        record.string = new6
                        cs.StatusIndicator("updated").add_file(filepath).add_message(
                            "nameID=6"
                        ).add_values(old_value=old6, new_value=new6).emit()
                        changed = True
                    break

        # ID3/ID6 remove '-Regular' after Bold when weight==700
        if "name" in font and weight == 700:
            name_table = font["name"]
            for record in list(name_table.names):
                if (
                    record.platformID == 3
                    and record.platEncID == 1
                    and record.langID == 0x409
                ):
                    try:
                        val = (
                            record.toUnicode()
                            if hasattr(record, "toUnicode")
                            else str(record.string)
                        )
                    except Exception:
                        val = str(record.string)
                    if record.nameID == 3:
                        newv = re.sub(r"(?i)-Bold-?RegularItalic$", "-BoldItalic", val)
                        newv = re.sub(r"(?i)-Bold-?Regular$", "-Bold", newv)
                        if newv != val:
                            record.string = newv
                            cs.StatusIndicator("updated").add_file(
                                filepath
                            ).add_message("nameID=3").add_values(
                                old_value=val, new_value=newv
                            ).emit()
                            changed = True
                    elif record.nameID == 6:
                        psb = _sanitize_postscript(val)
                        newv = re.sub(r"(?i)-Bold-?RegularItalic$", "-BoldItalic", psb)
                        newv = re.sub(r"(?i)-Bold-?Regular$", "-Bold", newv)
                        if newv != val:
                            record.string = newv
                            cs.StatusIndicator("updated").add_file(
                                filepath
                            ).add_message("nameID=6").add_values(
                                old_value=val, new_value=newv
                            ).emit()
                            changed = True

        # ID17 set per RIBBI (only when allowed)
        if allow and "name" in font:
            id17_val = (
                ("Bold Italic" if italic else "Bold")
                if weight == 700
                else ("Regular Italic" if italic else "Regular")
            )
            name_table = font["name"]
            found17 = False
            for record in list(name_table.names):
                if (
                    record.nameID == 17
                    and record.platformID == 3
                    and record.platEncID == 1
                    and record.langID == 0x409
                ):
                    try:
                        old17 = (
                            record.toUnicode()
                            if hasattr(record, "toUnicode")
                            else str(record.string)
                        )
                    except Exception:
                        old17 = str(record.string)
                    if old17 != id17_val:
                        record.string = id17_val
                        cs.StatusIndicator("updated").add_file(filepath).add_message(
                            "nameID=17"
                        ).add_values(old_value=old17, new_value=id17_val).emit()
                        changed = True
                    found17 = True
                    break
            if not found17:
                nr = NameRecord()
                nr.nameID = 17
                nr.platformID = 3
                nr.platEncID = 1
                nr.langID = 0x409
                nr.string = id17_val
                name_table.names.append(nr)
                cs.StatusIndicator("updated").add_file(filepath).add_message(
                    f"Created new nameID=17"
                ).add_field("value", id17_val).emit()
                changed = True

        if changed:
            font.save(filepath)
        font.close()
        return True
    except Exception as e:
        cs.StatusIndicator("error").add_file(filepath).with_explanation(
            f"Error processing font file: {e}"
        ).emit()
        return False


def process_file(filepath: str, force_weight: int | None = None) -> bool:
    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        cs.StatusIndicator("warning").add_file(filepath).with_explanation(
            "Skipping unsupported file"
        ).emit()
        return False
    cs.StatusIndicator("info").add_file(filepath).add_message("Processing").emit()
    if ext == ".ttx":
        return process_ttx_file(filepath, force_weight)
    return process_binary_font(filepath, force_weight)


def collect_font_files(paths, recursive: bool = False):
    font_files = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.is_file():
            if path_obj.suffix.lower() in SUPPORTED_EXTENSIONS:
                font_files.append(str(path_obj))
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


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup RIBBI naming across name IDs (1,2,3,4,6,17)",
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
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    # Force flags
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-b",
        "--bold",
        action="store_true",
        help="Force Bold (sets usWeightClass=700 and applies Bold/Bold Italic normalization)",
    )
    group.add_argument(
        "-reg",
        "--regular",
        action="store_true",
        help="Force Regular (sets usWeightClass=400 and applies Regular/Regular Italic normalization)",
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

    if RICH_AVAILABLE:
        cs.StatusIndicator("info").add_message(
            f"Found {len(font_files)} font file(s) to process:"
        ).emit()
        for f in font_files:
            cs.StatusIndicator("info").add_file(f).emit()

    if args.dry_run:
        cs.StatusIndicator("warning").with_explanation(
            "Dry run mode - no changes will be made"
        ).emit()
        # Summarize intended actions per file
        for file in font_files:
            ext = Path(file).suffix.lower()
            try:
                if ext == ".ttx":
                    tree = ET.parse(file)
                    root = tree.getroot()
                    weight = (
                        700
                        if args.bold
                        else (400 if args.regular else _get_us_weight_ttx(root))
                    )
                    italic = _is_italic_ttx(root)
                else:
                    font = TTFont(file)
                    weight = (
                        700
                        if args.bold
                        else (
                            400
                            if args.regular
                            else (
                                getattr(font["OS/2"], "usWeightClass", 400)
                                if "OS/2" in font
                                else 400
                            )
                        )
                    )
                    italic = _is_italic_binary(font)
                    font.close()
                ribbi = ("Bold" if weight == 700 else "Regular") + (
                    " Italic" if italic else ""
                )
                cs.StatusIndicator("info", dry_run=True).add_file(file).add_message(
                    f"Would normalize to RIBBI='{ribbi}'"
                ).emit()
            except Exception:
                pass
        return

    # Confirm
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
    success = 0
    for file in font_files:
        if process_file(
            file, force_weight=(700 if args.bold else (400 if args.regular else None))
        ):
            success += 1
    cs.StatusIndicator("saved").add_message(
        f"Completed: {success}/{len(font_files)} files processed successfully"
    ).emit()


if __name__ == "__main__":
    main()
