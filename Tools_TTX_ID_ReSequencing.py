#!/usr/bin/env python3
import argparse
from pathlib import Path
from lxml import etree


def resequence(
    file_path,
    table,
    target,
    start,
    new,
    range_str=None,
    resequence_mode=True,
    keep_first=True,
    from_first=True,
    elem_tag=None,
):
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(file_path, parser)
    root = tree.getroot()

    # Compute offset
    start = int(start)
    new = int(new)
    offset = new - start

    # Find the table(s)
    for tbl in root.findall(f".//{table}"):
        # Determine range for this table: if no explicit range, go from start to max value present
        if range_str:
            r_start, r_end = map(int, range_str.split("-"))
        else:
            r_start = start
            max_val = None
            for elem in tbl.iter():
                if elem_tag is not None and elem.tag != elem_tag:
                    continue
                if target in elem.attrib:
                    try:
                        v = int(elem.attrib[target])
                    except Exception:
                        continue
                    if max_val is None or v > max_val:
                        max_val = v
            r_end = max_val if max_val is not None else start

        # Collect candidate elements in document order
        all_candidates = []
        for elem in tbl.iter():
            if elem_tag is not None and elem.tag != elem_tag:
                continue
            if target in elem.attrib:
                try:
                    value = int(elem.attrib[target])
                except Exception:
                    continue
                all_candidates.append((elem, value))

        # Choose selection strategy
        if resequence_mode and from_first and not range_str:
            # Find first occurrence of `start` and take that and all subsequent candidates
            in_range_elems = []
            found = False
            for elem, value in all_candidates:
                if not found and value == start:
                    found = True
                if found:
                    in_range_elems.append((elem, value))
        else:
            # Value-range based selection
            in_range_elems = [
                (e, v) for (e, v) in all_candidates if r_start <= v <= r_end
            ]

        if resequence_mode:
            # Assign sequential values starting at `new`, ignoring prior gaps/ordering
            if keep_first and in_range_elems:
                # Preserve the first element's existing value; start sequencing after it
                anchor_val = in_range_elems[0][1]
                # first stays as-is (write back explicitly to be safe)
                in_range_elems[0][0].attrib[target] = str(anchor_val)
                next_val = anchor_val + 1
                for elem, _old in in_range_elems[1:]:
                    elem.attrib[target] = str(next_val)
                    next_val += 1
            else:
                next_val = new
                for elem, _old in in_range_elems:
                    elem.attrib[target] = str(next_val)
                    next_val += 1
        else:
            # Offset mode: preserve relative distances, shift by computed offset
            for elem, value in in_range_elems:
                elem.attrib[target] = str(value + offset)

    # Write back (don’t pretty print, keep formatting intact)
    Path(file_path).write_bytes(
        etree.tostring(tree, encoding="utf-8", xml_declaration=True, pretty_print=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resequence numeric IDs in TTX XML font files (name/STAT/fvar), minimal flags by default."
    )
    parser.add_argument("file", help="Path to XML/TTX file to update.")

    # Presets for common targets
    parser.add_argument(
        "--preset",
        choices=["name", "stat", "fvar"],
        help="Quick target selection: name→namerecord/@nameID, stat→ValueNameID/LinkedValueNameID/@value, fvar→NamedInstance/@subfamilyNameID,@postscriptNameID",
    )

    # Custom targeting (used when --preset is not provided)
    parser.add_argument("--table", help="Table name (e.g., name, STAT, fvar).")
    parser.add_argument(
        "--target", help="Attribute to update (e.g., nameID, value, subfamilyNameID)."
    )
    parser.add_argument(
        "--elem", help="Restrict to elements with this tag (e.g., ValueNameID)."
    )

    parser.add_argument("--start", required=True, help="Anchor/original start value.")
    parser.add_argument(
        "--new",
        required=True,
        help="First value to assign (resequence) or new start (offset mode).",
    )

    parser.add_argument(
        "--range",
        help="Optional range (e.g., 272-292). If omitted, resequence from the first occurrence of --start forward.",
    )

    # Mode controls: defaults are resequence + from-first + keep-first
    parser.add_argument(
        "--offset",
        action="store_true",
        help="Use offset mode (shift values by new-start) instead of resequencing.",
    )
    parser.add_argument(
        "--no-keep-first",
        action="store_true",
        help="When resequencing, do not keep the first anchor value; assign --new to the first element.",
    )
    parser.add_argument(
        "--whole-range",
        action="store_true",
        help="When resequencing without --range, use value range [--start..max] instead of from-first selection.",
    )

    args = parser.parse_args()

    # Determine mode defaults
    reseq_mode = not args.offset
    keep_first = not args.no_keep_first
    from_first = not args.whole_range

    # Dispatch based on preset or custom
    def run_pair(table: str, elem: str | None, target: str) -> None:
        resequence(
            args.file,
            table,
            target,
            args.start,
            args.new,
            args.range,
            reseq_mode,
            keep_first,
            from_first,
            elem,
        )

    if args.preset == "name":
        run_pair("name", "namerecord", "nameID")
    elif args.preset == "stat":
        # ValueNameID and LinkedValueNameID values; also try Axis/@axisNameID variants
        run_pair("STAT", "ValueNameID", "value")
        run_pair("STAT", "LinkedValueNameID", "value")
        run_pair("STAT", "Axis", "axisNameID")
        run_pair("STAT", "Axis", "AxisNameID")
    elif args.preset == "fvar":
        run_pair("fvar", "NamedInstance", "subfamilyNameID")
        run_pair("fvar", "NamedInstance", "postscriptNameID")
    else:
        # Custom
        if not (args.table and args.target):
            raise SystemExit(
                "Custom mode requires --table and --target when --preset is not used"
            )
        run_pair(args.table, args.elem, args.target)
