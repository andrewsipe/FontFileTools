# Tools

Various utility tools for font processing, fixing, and analysis.

## Overview

A collection of specialized tools for specific font processing tasks, including OpenType table manipulation, font fixing, metadata extraction, and more.

## Scripts

### `Tools_ListGenerator.py`
Generate a simple text list of font filenames (without extensions) for analysis.

**Usage:**
```bash
python Tools_ListGenerator.py /path/to/fonts
# Creates filename_list.txt in the input directory
```

### `Tools_FontFixer.py`
Comprehensive font fixing tool with multiple fix strategies.

### `Tools_GASP_TableFixer.py`
Fix GASP (Grid-fitting and Scan-conversion Procedure) table issues.

### `Tools_GSUB_GPOS_CoverageTableSorter.py`
Sort coverage tables in GSUB/GPOS OpenType tables.

**Usage:**
```bash
python Tools_GSUB_GPOS_CoverageTableSorter.py font.otf
```

### `Tools_TTX_GSUB_GPOS_CoverageTableSorter.py`
Sort coverage tables using TTX (XML) format.

### `Tools_TTX_ID_ReSequencing.py`
Resequence ID numbers in TTX files.

### `Tools_AdvancedWidth_DetectandFix.py`
Detect and fix advanced width issues in fonts.

### `Tools_ReCenter_A_Glyph.py`
Recenter a specific glyph in a font.

### `Tools_RIBBI_NameBuilder.py`
Build RIBBI (Regular, Italic, Bold, Bold Italic) names.

### `Tools_RIBBI_NameCleaner.py`
Clean RIBBI names in font metadata.

### `Tools_VariableFonts_VarStoreAxesFixer.py`
Fix variable font VarStore axes issues.

### `Tools_VF_Metadata_Extractor.py`
Extract metadata from variable fonts.

### `Tools_FileSorter_ByVendorAbbreviation.py`
Sort font files by vendor abbreviations.

## OpenType Features Generator

The OpenType Features Generator (`Opentype_FeaturesGenerator.py`) is available as a separate repository. See [OpentypeFeaturesGenerator](https://github.com/andrewsipe/OpentypeFeaturesGenerator) for documentation.

## Common Options

Most tools support:
- `-R, --recursive` - Process directories recursively
- `--dry-run` - Preview changes without modifying files
- `-V, --verbose` - Show detailed processing information

## Dependencies

See `requirements.txt`:
- `fontFeatures` - Font feature processing (optional but recommended)
- `lxml` - XML processing for TTX tools
- `rich` - Console output formatting
- Core dependencies (fonttools) provided by included `core/` library

## Installation

1. Clone this repository:
```bash
git clone https://github.com/andrewsipe/FontFileTools.git
cd FontFileTools
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Related Tools

- [OpentypeFeaturesGenerator](https://github.com/andrewsipe/OpentypeFeaturesGenerator) - OpenType feature generation
- [FontNameID](https://github.com/andrewsipe/FontNameID) - Font metadata editing
- [FontMetricsNormalizer](https://github.com/andrewsipe/FontMetricsNormalizer) - Font metrics normalization

