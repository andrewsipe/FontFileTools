# Product refinement notes — FontFileTools

Captured during the 2026-08-21 declutter pass. Use for a later product/release pass. **Not** user-facing docs.

## What was archived (declutter)

| Archived path | Was | Why archived |
|---------------|-----|--------------|
| `_misc/_archive/Tools_FontFixer.py` | Monolithic fixer | Replaced by **FontFixer** package (earlier in this cleanup) |
| `_misc/_archive/FontFileTools/MorisawaDownloader.py` | One-shot Morisawa CSS/font grabber | Hardcoded `stg.morisawafonts.net` job URLs + fixed `OUTPUT_DIR`; not a reusable CLI. Extraction belongs with **FontExtractor2** if revisited |

## Declutter verdict for the rest

**No other safe archives.** Remaining scripts are single-purpose utilities with distinct jobs. This package is intentionally a grab-bag (same pattern as Filename_Tools).

## Active tree (grouped)

### General / reusable

| Script | Role |
|--------|------|
| `Tools_ListGenerator.py` | Filename list export |
| `Tools_GASP_TableFixer.py` | GASP table fixes |
| `CoverageSorter.py` | Sort GSUB/GPOS Coverage by glyph ID (simple single-file) |
| `Tools_TTX_ID_ReSequencing.py` | TTX ID resequence |
| `Tools_AdvancedWidth_DetectandFix.py` | Advanced width detect/fix |
| `Tools_UPM_Rescaler.py` | unitsPerEm rescale |
| `Tools_ReCenter_A_Glyph.py` | Recenter a glyph |
| `Tools_RIBBI_NameBuilder.py` / `Tools_RIBBI_NameCleaner.py` | RIBBI name build/clean |
| `Tools_VariableFonts_VarStoreAxesFixer.py` | VarStore axes fix |
| `Tools_VF_Metadata_Extractor.py` | VF metadata extract / abbrev dict |
| `Tools_FileSorter_ByVendorAbbreviation.py` | Sort files by OS/2 achVendID |
| `VF_AxisFormat_Scanner.py` | VF axis format scan |
| `fix_corrupt_gvar.py` / `fix_oversized_glyf_hints.py` | Targeted VF/glyf repairs |

### Foundry- / workflow-specific (keep; product may split later)

| Script | Role |
|--------|------|
| `HoeflerSpaceCombiner.py` | Merge `{name}` + `{name}-Space` pairs → `combined/` |
| `JungLeeMergeTool.py` | Merge Jung Lee A/B/(C) split fonts → `merged/` |

## Overlaps to resolve on the product pass

1. **Coverage sorting** — `CoverageSorter.py` (here) vs `Opentype_Tools/opentype_coverage_sorter.py` (FontCore-integrated, batch CLI). Prefer one public entry; keep the other as thin wrapper or archive after parity check.
2. **Vendor folder sort** — `Tools_FileSorter_ByVendorAbbreviation.py` vs FileRenamer / FontVault organize-by-vendor flows.
3. **RIBBI naming** — Builder/Cleaner vs **FontNameID** NameID policies; clarify ownership.
4. **VF metadata / axis tools** — overlap with Variable_Instancer / VarFontStudio tooling narrative.
5. **Foundry scripts** — ship as optional “recipes” extras, separate package, or keep undocumented personal utilities.
6. **README drift (fixed in declutter)** — still listed missing `Tools_GSUB_GPOS_CoverageTableSorter.py` / `Tools_TTX_GSUB_GPOS_CoverageTableSorter.py`; actual file is `CoverageSorter.py`. Undocumented scripts existed (UPM, fix_*, VF_Axis*, JungLee, Hoefler).

## Missing from old README (were on disk)

- `Tools_UPM_Rescaler.py`
- `CoverageSorter.py` (replaced the two GSUB/GPOS sorter names)
- `VF_AxisFormat_Scanner.py`
- `fix_corrupt_gvar.py`, `fix_oversized_glyf_hints.py`
- `HoeflerSpaceCombiner.py`, `JungLeeMergeTool.py`

## Do not lose

- JungLee merge rules (stem ends with A/B/C; OT layout combine from A base).
- Hoefler Space-pair combine behavior.
- Anything unique in archived `Tools_FontFixer.py` is covered under FontFixer `PRODUCT_REFINEMENT_NOTES.md` (CLI parity already checked).
