# Font File Tools

Specialized utilities for font processing, fixing, and analysis. Grab-bag package — each script is a standalone CLI.

Older `Tools_FontFixer.py` and a one-shot Morisawa downloader live under `_misc/_archive/`. See `PRODUCT_REFINEMENT_NOTES.md` for product-pass notes.

## Scripts

### Lists / organize
| Script | Role |
|--------|------|
| `Tools_ListGenerator.py` | Write a filename list (no extensions) |
| `Tools_FileSorter_ByVendorAbbreviation.py` | Sort files by OS/2 `achVendID` |

### Table / outline fixes
| Script | Role |
|--------|------|
| `Tools_GASP_TableFixer.py` | GASP table fixes |
| `CoverageSorter.py` | Sort GSUB/GPOS Coverage by glyph ID |
| `Tools_TTX_ID_ReSequencing.py` | Resequence IDs in TTX |
| `Tools_AdvancedWidth_DetectandFix.py` | Advanced width detect/fix |
| `Tools_UPM_Rescaler.py` | Rescale unitsPerEm |
| `Tools_ReCenter_A_Glyph.py` | Recenter a glyph |
| `fix_corrupt_gvar.py` | Repair corrupt gvar |
| `fix_oversized_glyf_hints.py` | Oversized glyf hints |

### Names
| Script | Role |
|--------|------|
| `Tools_RIBBI_NameBuilder.py` | Build RIBBI names |
| `Tools_RIBBI_NameCleaner.py` | Clean RIBBI names |

### Variable fonts
| Script | Role |
|--------|------|
| `Tools_VariableFonts_VarStoreAxesFixer.py` | VarStore axes fix |
| `Tools_VF_Metadata_Extractor.py` | Extract VF metadata / abbrev dict |
| `VF_AxisFormat_Scanner.py` | Scan VF axis formats |

### Foundry-specific workflows
| Script | Role |
|--------|------|
| `HoeflerSpaceCombiner.py` | Merge `{name}` + `{name}-Space` pairs |
| `JungLeeMergeTool.py` | Merge Jung Lee A/B/(C) split fonts |

## Quick examples

```bash
cd FontFileTools
python Tools_ListGenerator.py /path/to/fonts
python CoverageSorter.py font.otf
python Tools_UPM_Rescaler.py 1000 fonts/ -r --dry-run
python JungLeeMergeTool.py /path/to/converted/fonts
```

## Dependencies

See `requirements.txt` (`fonttools`, `rich`, `lxml`, optional `fontFeatures`). Shared helpers via `FontCore/`.

## Related

- [FontFixer](https://github.com/andrewsipe/FontFixer) — single-pass structural fixer (replaces archived Tools_FontFixer)
- [Opentype_Tools](https://github.com/andrewsipe/OpentypeFeaturesGenerator) — OT feature tooling (also has a Coverage sorter)
- [FontNameID](https://github.com/andrewsipe/FontNameID) — NameID editing
