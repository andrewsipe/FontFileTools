#!/usr/bin/env python3
"""
Fix VarStore RegionAxisCount mismatch in a TTX file with minimal formatting changes.

- Counts axes from the fvar table and updates VarRegionList's RegionAxisCount comment.
- Ensures each Region has the correct number of VarRegionAxis children; adds missing ones
  with neutral (0.0) coordinates.
- Uses core_ttx_table_io.load_ttx/write_ttx to preserve existing whitespace and comments.
"""

import re
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

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

try:
    from FontCore.core_ttx_table_io import load_ttx, write_ttx  # type: ignore
except Exception:
    load_ttx = None  # type: ignore
    write_ttx = None  # type: ignore

try:  # optional lxml import for factories when available
    from lxml import etree as LET  # type: ignore
except Exception:
    LET = None  # type: ignore


def _count_fvar_axes(root) -> int:
    fvar = root.find(".//fvar")
    if fvar is None:
        return 0
    # Use descendant Axis elements to be robust
    axes = list(fvar.findall(".//Axis"))
    return len(axes)


def _update_regionaxiscount_comment(
    var_region_list, num_axes: int, using_lxml: bool
) -> bool:
    """Update the RegionAxisCount comment in-place, preserving surrounding spaces.

    Returns True if a change was applied.
    """
    changed = False
    # lxml: we can select comments via xpath('comment()')
    if using_lxml and hasattr(var_region_list, "xpath"):
        try:
            comments = var_region_list.xpath("comment()")  # type: ignore[attr-defined]
        except Exception:
            comments = []
    else:
        # ET fallback: comments appear as child nodes with tag == ET.Comment
        comments = [
            c for c in list(var_region_list) if getattr(c, "tag", None) == ET.Comment
        ]

    for c in comments:
        try:
            txt = c.text or ""
            new_txt = re.sub(
                r"(RegionAxisCount\s*=\s*)(\d+)", r"\1" + str(num_axes), txt
            )
            if new_txt != txt:
                c.text = new_txt
                changed = True
        except Exception:
            continue
    return changed


def _ensure_regionaxiscount_comment(
    var_region_list, num_axes: int, using_lxml: bool
) -> bool:
    """Ensure a RegionAxisCount comment exists; create if missing.

    Returns True if created.
    """
    # Check existing
    has = False
    if using_lxml and hasattr(var_region_list, "xpath"):
        try:
            for c in var_region_list.xpath("comment()"):  # type: ignore[attr-defined]
                if isinstance(c.text, str) and "RegionAxisCount" in c.text:
                    has = True
                    break
        except Exception:
            has = False
    else:
        for c in list(var_region_list):
            if (
                getattr(c, "tag", None) == ET.Comment
                and isinstance(c.text, str)
                and "RegionAxisCount" in c.text
            ):
                has = True
                break
    if has:
        return False

    # Create a comment node with conservative indentation
    comment_text = f" RegionAxisCount={num_axes} "
    c = (
        LET.Comment(comment_text)
        if using_lxml and LET is not None
        else ET.Comment(comment_text)
    )
    # Put it as the first child for visibility
    try:
        var_region_list.insert(0, c)
        # Indentation heuristics: match first child's text/tail if possible
        if len(list(var_region_list)) > 1:
            first_child = list(var_region_list)[1]
            # Align comment.tail with first_child.text if it's pure whitespace
            if isinstance(first_child.text, str) and first_child.text.strip() == "":
                c.tail = first_child.text
            else:
                c.tail = "\n      "
        else:
            c.tail = "\n      "
    except Exception:
        # Fallback append
        var_region_list.append(c)
        c.tail = "\n      "
    return True


def _copy_indent_like(sample_el) -> Tuple[str, str]:
    """Return (child_indent, tail_indent) inferred from a sample element.

    child_indent is the element.text whitespace before first child.
    tail_indent is the element.tail whitespace used after the element.
    Fallback to common TTX indents if absent.
    """
    child_indent = None
    tail_indent = None
    try:
        child_indent = (
            sample_el.text
            if isinstance(sample_el.text, str) and sample_el.text.strip() == ""
            else None
        )
    except Exception:
        child_indent = None
    try:
        tail_indent = (
            sample_el.tail
            if isinstance(sample_el.tail, str) and sample_el.tail.strip() == ""
            else None
        )
    except Exception:
        tail_indent = None
    return child_indent or "\n        ", tail_indent or "\n      "


def _append_var_region_axis(
    region_el, index_value: int, template_axis=None, *, using_lxml: bool
) -> None:
    # Create VarRegionAxis with neutral coordinates
    if using_lxml and LET is not None:
        new_axis = LET.SubElement(region_el, "VarRegionAxis")  # type: ignore[attr-defined]
    else:
        new_axis = ET.SubElement(region_el, "VarRegionAxis")
    new_axis.set("index", str(index_value))

    # Determine indentation based on template or region
    child_indent, tail_indent = _copy_indent_like(template_axis or region_el)
    new_axis.text = child_indent

    for tag in ("StartCoord", "PeakCoord", "EndCoord"):
        if using_lxml and LET is not None:
            c = LET.SubElement(new_axis, tag)  # type: ignore[attr-defined]
        else:
            c = ET.SubElement(new_axis, tag)
        c.set("value", "0.0")
        # For each child, try to use the same tail spacing as in template children if available
        if template_axis is not None:
            try:
                t_child = template_axis.find(tag)
                if (
                    t_child is not None
                    and isinstance(t_child.tail, str)
                    and t_child.tail.strip() == ""
                ):
                    c.tail = t_child.tail
                else:
                    c.tail = "\n        "
            except Exception:
                c.tail = "\n        "
        else:
            c.tail = "\n        "

    new_axis.tail = tail_indent


def _get_fvar_axis_tags_ttx(root) -> list[str]:
    """Return list of fvar axis tags from TTX (best-effort, robust to value/text)."""
    tags: list[str] = []
    fvar = root.find(".//fvar")
    if fvar is None:
        return tags
    for axis in list(fvar.findall(".//Axis")):
        tag_el = axis.find("AxisTag")
        tag = None
        if tag_el is not None:
            tag = tag_el.get("value") or (tag_el.text or "").strip()
        if tag:
            tags.append(tag)
    return tags


def _ensure_avar_axes(root, using_lxml: bool) -> bool:
    """Ensure <avar> has an identity segment for each fvar axis.

    Returns True if any changes were made.
    """
    avar = root.find(".//avar")
    if avar is None:
        return False

    # Collect existing axis tags in <avar> by looking at children
    existing: set[str] = set()
    sample_axis = None
    for child in list(avar):
        try:
            # Common patterns: <axis tag="wght">, <axis axisTag="wght">, <segment axis="wght">
            axis_tag = child.get("tag") or child.get("axisTag") or child.get("axis")
            if axis_tag:
                existing.add(axis_tag)
                if sample_axis is None:
                    sample_axis = child
                continue
            # Fallback: <AxisTag>text</AxisTag> inside child
            at = child.find("AxisTag")
            if at is not None:
                t = at.get("value") or (at.text or "").strip()
                if t:
                    existing.add(t)
                    if sample_axis is None:
                        sample_axis = child
        except Exception:
            continue

    needed_tags = [t for t in _get_fvar_axis_tags_ttx(root) if t not in existing]
    if not needed_tags:
        return False

    # Determine axis element name and mapping element/attr names from a sample, else defaults
    axis_elem_name = sample_axis.tag if sample_axis is not None else "axis"
    sample_map = None
    map_elem_name = "map"
    from_attr = "from"
    to_attr = "to"
    if sample_axis is not None:
        for ch in list(sample_axis):
            # First child element is assumed to be a mapping element
            sample_map = ch
            break
    if sample_map is not None:
        map_elem_name = sample_map.tag
        # Normalize attribute names if sample uses different casing
        for candidate in ("from", "From", "src"):
            if sample_map.get(candidate) is not None:
                from_attr = candidate
                break
        for candidate in ("to", "To", "dst"):
            if sample_map.get(candidate) is not None:
                to_attr = candidate
                break

    # Indentation heuristics from avar container
    child_indent, tail_indent = _copy_indent_like(avar)
    axis_child_indent = "\n          "
    axis_tail_indent = "\n      "
    try:
        # If there is an existing axis, borrow its indents
        if sample_axis is not None:
            axis_child_indent, axis_tail_indent = _copy_indent_like(sample_axis)
    except Exception:
        pass

    changed = False
    for tag in needed_tags:
        # Create new axis element
        if using_lxml and LET is not None:
            axis_el = LET.Element(axis_elem_name)  # type: ignore[attr-defined]
        else:
            axis_el = ET.Element(axis_elem_name)
        # Set an identifying attribute; prefer the one present in sample
        if sample_axis is not None:
            if sample_axis.get("tag") is not None:
                axis_el.set("tag", tag)
            elif sample_axis.get("axisTag") is not None:
                axis_el.set("axisTag", tag)
            elif sample_axis.get("axis") is not None:
                axis_el.set("axis", tag)
            else:
                axis_el.set("axis", tag)
        else:
            axis_el.set("axis", tag)

        axis_el.text = axis_child_indent

        # Add identity mappings -1→-1, 0→0, 1→1
        for f, t in ((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)):
            if using_lxml and LET is not None:
                map_el = LET.SubElement(axis_el, map_elem_name)  # type: ignore[attr-defined]
            else:
                map_el = ET.SubElement(axis_el, map_elem_name)
            map_el.set(from_attr, str(f))
            map_el.set(to_attr, str(t))
            map_el.tail = "\n            "

        axis_el.tail = axis_tail_indent

        try:
            # Insert before closing; keep order after existing children
            avar.append(axis_el)
            changed = True
        except Exception:
            continue

    return changed


def fix_varstore_axes(ttx_file_path: str, *, ensure_avar_axes: bool = False) -> bool:
    """Fix the VarStore RegionAxisCount to match the fvar axes with minimal diff."""

    if load_ttx is not None:
        tree, root, using_lxml = load_ttx(ttx_file_path)
    else:
        # Fallback to stdlib ET (may not preserve whitespace as well)
        tree = ET.parse(ttx_file_path)
        root = tree.getroot()
        using_lxml = False

    num_axes = _count_fvar_axes(root)
    if num_axes <= 0:
        cs.StatusIndicator("error").with_explanation(
            "No fvar axes found; aborting"
        ).emit()
        return False
    cs.StatusIndicator("info").add_message(
        f"Found {num_axes} axes in fvar table"
    ).emit()

    var_region_lists = list(root.findall(".//VarRegionList"))
    any_change = False

    for vrl in var_region_lists:
        created = _ensure_regionaxiscount_comment(vrl, num_axes, using_lxml)
        if created:
            cs.StatusIndicator("updated").add_message(
                f"Inserted RegionAxisCount={num_axes} comment"
            ).emit()
            any_change = True
        if _update_regionaxiscount_comment(vrl, num_axes, using_lxml):
            cs.StatusIndicator("updated").add_message(
                f"Updated RegionAxisCount to {num_axes}"
            ).emit()
            any_change = True

        # For each Region ensure correct number of VarRegionAxis elements
        for region in list(vrl.findall("Region")):
            axes = list(region.findall("VarRegionAxis"))
            current = len(axes)
            # If there are too many axes compared to fvar, prune extras (keep first num_axes)
            if current > num_axes:
                for extra in axes[num_axes:]:
                    try:
                        region.remove(extra)
                        any_change = True
                    except Exception:
                        pass
                cs.StatusIndicator("updated").add_message(
                    f"Region {region.get('index')} pruned from {current} to {num_axes} axes"
                ).emit()
                axes = list(region.findall("VarRegionAxis"))
                current = len(axes)
            if current < num_axes:
                cs.StatusIndicator("info").add_message(
                    f"Region {region.get('index')} has {current} axes, needs {num_axes}"
                ).emit()
                template = axes[0] if axes else None
                for i in range(current, num_axes):
                    _append_var_region_axis(region, i, template, using_lxml=using_lxml)
                    any_change = True
            # Normalize @index attributes to 0..num_axes-1 order
            axes_now = list(region.findall("VarRegionAxis"))
            for i, ax in enumerate(axes_now):
                try:
                    if ax.get("index") != str(i):
                        ax.set("index", str(i))
                        any_change = True
                except Exception:
                    continue

    # Ensure avar axes (optional)
    if ensure_avar_axes:
        if _ensure_avar_axes(root, using_lxml):
            cs.StatusIndicator("updated").add_message(
                "Ensured avar identity mappings for missing axes"
            ).emit()
            any_change = True

    if not any_change:
        cs.StatusIndicator("info").add_message("No changes needed").emit()
        return True

    # Backup original
    path_obj = Path(ttx_file_path)
    backup_path = path_obj.with_suffix(".ttx.backup")
    try:
        path_obj.rename(backup_path)
        cs.StatusIndicator("saved").add_file(str(backup_path)).with_explanation(
            "Created backup"
        ).emit()
    except Exception as e:
        cs.StatusIndicator("warning").with_explanation(
            f"failed to create backup ({e}); proceeding to write"
        ).emit()
        backup_path = None  # type: ignore

    # Write updated file
    try:
        if write_ttx is not None:
            write_ttx(tree, str(path_obj), using_lxml)
        else:
            tree.write(str(path_obj), encoding="UTF-8", xml_declaration=True)
        cs.StatusIndicator("saved").add_file(str(path_obj)).with_explanation(
            "Fixed TTX file saved"
        ).emit()
        return True
    except Exception as e:
        cs.StatusIndicator("error").with_explanation(f"Error writing TTX: {e}").emit()
        # Attempt to restore backup
        try:
            if backup_path is not None and not path_obj.exists():
                backup_path.rename(path_obj)
        except Exception:
            pass
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix VarStore RegionAxisCount and optionally ensure avar axes"
    )
    parser.add_argument("ttx", help="Path to TTX file")
    parser.add_argument(
        "--ensure-avar-axes",
        action="store_true",
        help="If <avar> exists, ensure identity segment for each fvar axis",
    )
    args = parser.parse_args()

    if fix_varstore_axes(args.ttx, ensure_avar_axes=args.ensure_avar_axes):
        cs.StatusIndicator("saved").add_message("TTX file has been fixed!").emit()
        cs.StatusIndicator("info").add_message(
            "You can now try converting it again with:"
        ).emit()
        cs.StatusIndicator("info").add_message(f"  ttx {args.ttx}").emit()
    else:
        cs.StatusIndicator("error").add_message("Failed to fix the TTX file").emit()
