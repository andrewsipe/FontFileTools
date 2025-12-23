#!/usr/bin/env python3
"""
DEPRECATED: This script has been refactored into a modular structure.

New location: FontFixer/main.py

This file is kept for reference only. Please use the new modular version.

---

Original documentation:

Standalone Font Fixer using fonttools

A high-performance font validation and correction tool that applies comprehensive
OpenType font fixes in a single pass. Designed to replace sequential ftcli command
workflows with efficient batch processing.

Architecture:
    - Single-pass processing: Opens each font once, applies all fixes, writes once
    - Handler-based design: Modular table-specific validators and fixers
    - Parallel processing: Multi-core support for large font collections

Performance:
    - ~10-12x faster than sequential ftcli-fix-loop for equivalent operations
    - Memory-efficient: Suitable for directories with 1000+ font files

Handler Coverage:
    - OS/2 table: Version upgrade, embedding permissions, monospace detection
    - Style consistency: Italic angle, fsSelection, macStyle synchronization
    - Glyph fixes: .notdef structure, nbsp presence and metrics
    - Kerning: Legacy kern table cleanup when GPOS exists
    - Name table: Platform-specific record filtering

Dependencies:
    - fonttools (pip install fonttools)

Example:
    # Process all fonts in directory with 8 workers
    python Tools_FontFixer.py -j 8 fonts/

    # Only run specific handlers
    python Tools_FontFixer.py --handlers os2,style fonts/
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, ClassVar, Callable, NamedTuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from functools import wraps
from contextlib import contextmanager
import math
import re
import struct
import unicodedata
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    from fontTools.ttLib import TTFont, TTLibError
    from fontTools.pens.statisticsPen import StatisticsPen
    from fontTools.ttLib.tables.O_S_2f_2 import Panose
except ImportError:
    print("Error: fonttools library not found.")
    print("Install with: pip install fonttools")
    sys.exit(1)

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

try:
    from rich.markup import escape
except ImportError:
    # Fallback if Rich is not available
    def escape(text: str) -> str:
        return text


# Get the themed console singleton
console = cs.get_console()

# ============================================================================
# CONSTANTS
# ============================================================================

# Name table platform constants
PLATFORM_WINDOWS = 3
ENCODING_UNICODE_BMP = 1
LANG_ENGLISH_US = 0x409

# OS/2 fsSelection bit positions
FS_SELECTION_ITALIC = 0
FS_SELECTION_BOLD = 5
FS_SELECTION_REGULAR = 6
FS_SELECTION_USE_TYPO_METRICS = 7
FS_SELECTION_WWS = 8
FS_SELECTION_OBLIQUE = 9

# head.macStyle bit positions
MAC_STYLE_BOLD = 0
MAC_STYLE_ITALIC = 1

# Standard weight classes
WEIGHT_BOLD = 700

# Signed 16-bit integer limits (used for bounds, caretSlopeRun, etc.)
SIGNED_16BIT_MIN = -32768
SIGNED_16BIT_MAX = 32767

# ============================================================================
# HANDLER SPECIFICATION
# ============================================================================


@dataclass(frozen=True)
class HandlerSpec:
    """Specification for a font table handler."""

    full_name: str  # "OS/2"
    short_name: str  # "os2"
    description: str  # "OS/2 table (version, fsType...)"

    # Class-level registry
    _registry: ClassVar[dict[str, "HandlerSpec"]] = {}

    def __post_init__(self):
        HandlerSpec._registry[self.short_name] = self

    @classmethod
    def get(cls, short_name: str) -> Optional["HandlerSpec"]:
        """Get handler spec by short name."""
        return cls._registry.get(short_name)

    @classmethod
    def all_short_names(cls) -> list[str]:
        """Get all registered short names."""
        return list(cls._registry.keys())


# Define all handlers
HANDLER_OS2 = HandlerSpec(
    "OS/2", "os2", "OS/2 table (version, fsType, monospace, fsSelection)"
)
HANDLER_STYLE = HandlerSpec(
    "post+hhea+OS/2+head (style consistency)",
    "style",
    "Style consistency (italic angle, fsSelection, macStyle)",
)
HANDLER_GLYPH = HandlerSpec(
    "glyf/CFF + cmap + hmtx (glyphs)", "glyph", "Glyph fixes (.notdef, nbsp)"
)
HANDLER_KERN = HandlerSpec(
    "kern+GPOS (kerning)", "kern", "Kerning cleanup (remove legacy kern if GPOS exists)"
)
HANDLER_NAME = HandlerSpec(
    "name (naming)",
    "name",
    "Name table cleanup (Windows English only, remove problematic IDs)",
)

# All handler names (for backward compatibility during transition)
ALL_HANDLERS = [
    HANDLER_OS2.full_name,
    HANDLER_STYLE.full_name,
    HANDLER_GLYPH.full_name,
    HANDLER_KERN.full_name,
    HANDLER_NAME.full_name,
]


def _get_handler_spec_by_full_name(full_name: str) -> Optional[HandlerSpec]:
    """Get HandlerSpec by full_name (for backward compatibility)."""
    for spec in HandlerSpec._registry.values():
        if spec.full_name == full_name:
            return spec
    return None


# ============================================================================
# RESULT DATACLASS
# ============================================================================


@dataclass
class FontFixResult:
    """Result of font validation/fixing operation."""

    file: str
    success: bool = False
    was_modified: bool = False

    # Handler tracking
    handlers_run: list[str] = field(default_factory=list)
    handlers_changed: list[str] = field(default_factory=list)
    handlers_unchanged: list[str] = field(default_factory=list)

    # Detailed results
    validations: dict[str, dict[str, Any]] = field(default_factory=dict)
    changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # Quarantine info
    quarantined: bool = False
    quarantine_path: Optional[str] = None

    # Output info
    output_path: Optional[str] = None

    def add_error(self, error: str, include_traceback: bool = False):
        """Add error message with optional traceback."""
        self.errors.append(error)
        if include_traceback:
            self.errors.append(traceback.format_exc())

    def add_exception(
        self, exc: Exception, context: str = "", include_traceback: bool = False
    ):
        """Add formatted exception as error."""
        error_type = type(exc).__name__
        error_str = str(exc) if str(exc) else "No error message provided"

        if context:
            msg = f"{context}: {error_type}: {error_str}"
        else:
            msg = f"{error_type}: {error_str}"

        self.add_error(msg, include_traceback)

    def mark_handler_run(self, handler_name: str, changed: bool):
        """Record handler execution result."""
        self.handlers_run.append(handler_name)
        if changed:
            self.handlers_changed.append(handler_name)
        else:
            self.handlers_unchanged.append(handler_name)

    def to_dict(self) -> dict:
        """Convert to dictionary for legacy compatibility."""
        return asdict(self)


# ============================================================================
# BITFIELD SPECIFICATION
# ============================================================================


@dataclass(frozen=True)
class BitfieldSpec:
    """Specification for OpenType bitfield flags."""

    name: str
    bit_position: int
    description: str = ""

    def is_set(self, value: int) -> bool:
        """Check if this bit is set in value."""
        return bool(value & (1 << self.bit_position))

    def set(self, value: int) -> int:
        """Return value with this bit set."""
        return value | (1 << self.bit_position)

    def clear(self, value: int) -> int:
        """Return value with this bit cleared."""
        return value & ~(1 << self.bit_position)


@dataclass
class BitfieldGroup:
    """Collection of related bitfield flags."""

    name: str
    flags: Dict[str, BitfieldSpec]

    def get_changed_flags(self, old_value: int, new_value: int) -> tuple[list, list]:
        """Return (added_flags, removed_flags) as readable names."""
        changed_bits = old_value ^ new_value
        added = []
        removed = []

        for flag_name, spec in self.flags.items():
            if spec.is_set(changed_bits):
                if spec.is_set(new_value):
                    added.append(flag_name)
                else:
                    removed.append(flag_name)

        return added, removed

    def format_change(self, old_value: int, new_value: int) -> str:
        """Return human-readable change description."""
        added, removed = self.get_changed_flags(old_value, new_value)
        parts = [f"added {f}" for f in added] + [f"removed {f}" for f in removed]
        delta = ", ".join(parts) if parts else "no change"
        return f"0x{old_value:04X} -> 0x{new_value:04X} [{delta}]"


# Define bitfield groups
FS_SELECTION = BitfieldGroup(
    "fsSelection",
    {
        "ITALIC": BitfieldSpec("ITALIC", 0, "Italic font style"),
        "BOLD": BitfieldSpec("BOLD", 5, "Bold font weight"),
        "REGULAR": BitfieldSpec("REGULAR", 6, "Regular style"),
        "USE_TYPO_METRICS": BitfieldSpec("USE_TYPO_METRICS", 7, "Use typo metrics"),
        "WWS": BitfieldSpec("WWS", 8, "WWS family conformance"),
        "OBLIQUE": BitfieldSpec("OBLIQUE", 9, "Oblique style"),
    },
)

MAC_STYLE = BitfieldGroup(
    "macStyle",
    {
        "BOLD": BitfieldSpec("BOLD", 0, "Bold weight"),
        "ITALIC": BitfieldSpec("ITALIC", 1, "Italic style"),
    },
)


# ============================================================================
# NAME TABLE UTILITY FUNCTIONS
# ============================================================================


def keep_windows_english_only(font: TTFont) -> int:
    """
    Keep only Windows English/Latin name records (platformID=3, platEncID=1, langID=0x409).
    Remove all other platform-specific name records.

    This ensures consistent name table behavior across platforms and reduces
    font file size by eliminating redundant name data.

    Args:
        font: TTFont object to modify

    Returns:
        Number of name records removed

    Note:
        Windows platform (platformID=3) with Unicode BMP encoding (platEncID=1)
        and US English language (langID=0x409) is the modern standard for
        cross-platform font compatibility.
    """
    if "name" not in font:
        return 0

    name_table = font["name"]
    original_count = len(name_table.names)

    # Keep only Windows Unicode BMP English US records
    # platformID=3 (Windows), platEncID=1 (Unicode BMP), langID=0x409 (English US)
    kept = [
        rec
        for rec in name_table.names
        if (
            rec.platformID == PLATFORM_WINDOWS
            and rec.platEncID == ENCODING_UNICODE_BMP
            and rec.langID == LANG_ENGLISH_US
        )
    ]

    name_table.names = kept
    return original_count - len(name_table.names)


def delete_specific_nameids(font: TTFont, name_ids: set[int]) -> int:
    """
    Remove name records with specified nameIDs.

    This function removes problematic nameIDs that can cause issues in certain
    font processing workflows or applications.

    Args:
        font: TTFont object to modify
        name_ids: Set of nameIDs to remove

    Returns:
        Number of name records removed

    Note:
        Commonly removed nameIDs include: 13 (License), 14 (License Info URL),
        18 (Compatible Full), 19 (Sample Text), 200-203 (WWS), and 55555 (custom).
    """
    if "name" not in font:
        return 0

    name_table = font["name"]
    before_count = len(name_table.names)
    name_table.names = [rec for rec in name_table.names if rec.nameID not in name_ids]
    return before_count - len(name_table.names)


# ============================================================================
# DECORATORS
# ============================================================================


def conditional_fix(*validation_keys: str):
    """
    Decorator: Only execute fix if corresponding validation(s) failed.

    Args:
        *validation_keys: One or more validation check names that must fail
                         for this fix to run

    Usage:
        @conditional_fix("version_current")
        def _fix_version(self) -> bool:
            # Only runs if "version_current" validation failed
            ...
    """

    def decorator(fix_method: Callable) -> Callable:
        @wraps(fix_method)
        def wrapper(self) -> bool:
            # Check if ANY of the validations failed
            should_run = any(
                not self.validations.get(key, {}).get("valid", True)
                for key in validation_keys
            )

            if not should_run:
                return False

            try:
                return fix_method(self)
            except (TTLibError, AttributeError, IndexError, ValueError) as e:
                if self.verbose:
                    self.log(f"Skipping {fix_method.__name__} due to error: {e}")
                return False

        return wrapper

    return decorator


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def clamp_signed_16bit(value: int) -> int:
    """
    Clamp a value to signed 16-bit integer range (-32768 to 32767).

    Args:
        value: The value to clamp

    Returns:
        Clamped value within valid range
    """
    if value > SIGNED_16BIT_MAX:
        return SIGNED_16BIT_MAX
    elif value < SIGNED_16BIT_MIN:
        return SIGNED_16BIT_MIN
    return value


# ============================================================================
# TABLE HANDLERS
# ============================================================================


class TableHandler(ABC):
    """
    Base class for font table validation and correction.

    Each handler manages one or more related OpenType table fixes,
    providing validation and correction capabilities.
    """

    def __init__(self, font: TTFont, verbose: bool = False):
        self.font = font
        self.verbose = verbose
        self.changes = {}  # Track all changes made
        self.validations = {}  # Track validation results

    @abstractmethod
    def validate(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate table state.

        Returns:
            Dictionary mapping check names to validation results.
            Each result contains:
                - 'valid' (bool): Whether the check passes validation standards
                - 'message' (str): Human-readable description of the validation state
        """
        pass

    @abstractmethod
    def fix(self) -> bool:
        """
        Apply corrections based on validation results.

        Returns:
            True if any changes were made, False otherwise.
        """
        pass

    @abstractmethod
    def get_table_name(self) -> str:
        """
        Return the handler name (describes what this handler manages).

        Returns:
            Handler name string.
        """
        pass

    def get_changes(self) -> Dict:
        """Return dict of changes made."""
        return self.changes

    def get_validations(self) -> Dict:
        """Return dict of validation results."""
        return self.validations

    def _track_change(
        self,
        property_name: str,
        old_value,
        new_value,
        changed: bool = None,
        info_only: bool = False,
    ):
        """
        Helper to track a property change.

        Args:
            property_name: Name of the property
            old_value: Original value
            new_value: New value
            changed: Whether value changed (auto-detected if None)
            info_only: If True, this is informational only (not a change)
        """
        if changed is None:
            changed = old_value != new_value

        self.changes[property_name] = {
            "old": old_value,
            "new": new_value,
            "changed": changed,
            "info_only": info_only,
        }

    def _track_bitfield_change(
        self,
        property_name: str,
        old_value: int,
        new_value: int,
        bitfield_group: BitfieldGroup,
    ) -> None:
        """
        Track a bitfield change with formatted description.

        Args:
            property_name: Name of the property
            old_value: Original bitfield value
            new_value: New bitfield value
            bitfield_group: BitfieldGroup instance for formatting
        """
        if old_value == new_value:
            return

        change_desc = bitfield_group.format_change(old_value, new_value)
        # Store formatted description as the new value
        self._track_change(property_name, f"0x{old_value:04X}", change_desc, True)

    def _track_multiple_changes(
        self,
        changes: list[
            Union[
                tuple[str, Any, Any],  # (name, old, new)
                tuple[str, Any, Any, bool],  # (name, old, new, changed)
            ]
        ],
    ) -> None:
        """
        Track multiple changes at once.

        Args:
            changes: List of (property_name, old_value, new_value) or
                     (property_name, old_value, new_value, changed) tuples
        """
        for change in changes:
            if len(change) == 4:
                prop, old, new_val, changed = change
                self._track_change(prop, old, new_val, changed)
            else:
                prop, old, new_val = change
                self._track_change(prop, old, new_val)

    def _track_info_changes(
        self,
        info_items: Union[dict[str, Any], list[tuple[str, Any]]],
    ) -> None:
        """
        Track informational-only properties (no actual changes).

        Args:
            info_items: Dict of {property_name: value} or list of (property_name, value) tuples
        """
        if isinstance(info_items, dict):
            items = info_items.items()
        else:
            items = info_items

        for prop, value in items:
            self._track_change(prop, value, value, False, info_only=True)

    def _track_validation(self, check_name: str, is_valid: bool, message: str = ""):
        """
        Track a validation check result.

        Args:
            check_name: Identifier for this validation check
            is_valid: True if the check passes validation standards
            message: Human-readable description of the validation state
        """
        self.validations[check_name] = {"valid": is_valid, "message": message}

    def validate_field(
        self,
        check_name: str,
        table_name: str,
        field_path: str,  # Can be "version" or nested "panose.bProportion"
        validator: Callable[[Any], Tuple[bool, str]],
        missing_table_msg: Optional[str] = None,
        missing_field_msg: Optional[str] = None,
    ) -> bool:
        """
        Generic field validation with comprehensive error handling.

        Args:
            check_name: Validation check identifier
            table_name: OpenType table name (e.g., "OS/2")
            field_path: Field name or dot-notation path (e.g., "panose.bProportion")
            validator: Function that takes field value and returns (is_valid, message)
            missing_table_msg: Override message when table doesn't exist
            missing_field_msg: Override message when field doesn't exist

        Returns:
            True if validation passed, False otherwise
        """
        # Check table exists
        if table_name not in self.font:
            msg = missing_table_msg or f"{table_name} table missing"
            self._track_validation(check_name, False, msg)
            return False

        try:
            # Navigate field path
            obj = self.font[table_name]
            for part in field_path.split("."):
                if not hasattr(obj, part):
                    msg = (
                        missing_field_msg or f"{table_name}.{field_path} field missing"
                    )
                    self._track_validation(check_name, False, msg)
                    return False
                obj = getattr(obj, part)

            # Run validator
            is_valid, message = validator(obj)
            self._track_validation(check_name, is_valid, message)
            return is_valid

        except (TTLibError, AttributeError, TypeError, IndexError, ValueError) as e:
            msg = f"Cannot validate {table_name}.{field_path}: {str(e)}"
            self._track_validation(check_name, False, msg)
            return False

    def validate_condition(
        self,
        check_name: str,
        condition: Callable[[], Tuple[bool, str]],
        error_context: str = "",
    ) -> bool:
        """
        Validate a complex condition with error handling.

        Args:
            check_name: Validation check identifier
            condition: Function that returns (is_valid, message)
            error_context: Context description for error messages

        Returns:
            True if validation passed, False otherwise
        """
        try:
            is_valid, message = condition()
            self._track_validation(check_name, is_valid, message)
            return is_valid
        except Exception as e:
            msg = f"Validation error{' in ' + error_context if error_context else ''}: {str(e)}"
            self._track_validation(check_name, False, msg)
            return False

    def log(self, message: str):
        """Print verbose messages."""
        if self.verbose:
            cs.StatusIndicator("info").add_message(message).emit(console)

    def track_changes(self) -> "ChangeBuilder":
        """Start building a change tracking group."""
        return ChangeBuilder(self)


# ============================================================================
# CHANGE BUILDER
# ============================================================================


class ChangeBuilder:
    """Fluent API for building complex change tracking."""

    def __init__(self, handler: "TableHandler"):
        self.handler = handler
        self.changes: list[tuple] = []

    def add(
        self,
        name: str,
        old: Any,
        new: Any,
        changed: Optional[bool] = None,
        info_only: bool = False,
    ) -> "ChangeBuilder":
        """Add a change (chainable)."""
        if changed is None:
            changed = old != new
        self.changes.append((name, old, new, changed, info_only))
        return self

    def add_if_changed(self, name: str, old: Any, new: Any) -> "ChangeBuilder":
        """Only add if value actually changed."""
        if old != new:
            self.changes.append((name, old, new, True, False))
        return self

    def add_info(self, name: str, value: Any) -> "ChangeBuilder":
        """Add info-only property."""
        self.changes.append((name, None, value, False, True))
        return self

    def commit(self) -> bool:
        """Commit all changes and return whether any were made."""
        for name, old, new, changed, info_only in self.changes:
            self.handler._track_change(name, old, new, changed, info_only)
        return any(c[3] for c in self.changes)  # Any with changed=True


# ============================================================================
# FONT STYLE ANALYZER
# ============================================================================


class FontStyleAnalyzer:
    """
    Centralized font style analysis.

    Determines:
    - Italic angle
    - Is italic vs oblique
    - Is bold
    - Is regular
    """

    def __init__(self, font: TTFont):
        self.font = font
        self._italic_angle = None
        self._is_italic = None
        self._is_oblique = None
        self._is_bold = None

    @property
    def italic_angle(self) -> float:
        """Calculate italic angle (cached)."""
        if self._italic_angle is None:
            self._italic_angle = self._calculate_italic_angle()
        return self._italic_angle

    @property
    def is_italic(self) -> bool:
        """Determine if font is italic (cached)."""
        if self._is_italic is None:
            self._analyze_style()
        return self._is_italic

    @property
    def is_oblique(self) -> bool:
        """Determine if font is oblique (cached)."""
        if self._is_oblique is None:
            self._analyze_style()
        return self._is_oblique

    @property
    def is_bold(self) -> bool:
        """Determine if font is bold (cached)."""
        if self._is_bold is None:
            self._analyze_style()
        return self._is_bold

    def _calculate_italic_angle(self, min_slant: float = 2.0) -> float:
        """
        Calculate italic angle using StatisticsPen on 'H' glyph.

        Based on FoundryTools logic:
        - Uses StatisticsPen to calculate slant from 'H' glyph
        - Returns 0 if abs(angle) < min_slant (considers upright)
        - Returns calculated angle otherwise
        """
        try:
            glyph_set = self.font.getGlyphSet()
            if "H" not in glyph_set:
                return 0.0

            pen = StatisticsPen(glyph_set)
            glyph_set["H"].draw(pen)

            # StatisticsPen.slant is the tangent of the angle
            # Convert to degrees: angle = -arctan(slant) in degrees
            slant = pen.slant
            angle = -math.degrees(math.atan(slant)) if slant else 0.0

            # Round to reasonable precision
            angle = round(angle, 2)

            # If abs angle is below minimum, consider it upright
            if abs(angle) < min_slant:
                return 0.0

            return angle

        except Exception:
            return 0.0

    def _analyze_style(self):
        """Analyze font style from names and metrics."""
        is_slanted = abs(self.italic_angle) >= 2.0

        self._is_italic = False
        self._is_oblique = False
        self._is_bold = False

        # Check for oblique/italic in names
        if is_slanted and "name" in self.font:
            for record in self.font["name"].names:
                if record.nameID in [1, 2, 4, 6]:  # Family, Subfamily, Full, PostScript
                    name_str = record.toUnicode().lower()
                    # Check for "oblique" first (more specific)
                    if re.search(r"\boblique\b", name_str):
                        self._is_oblique = True
                        break
                    # Check for "italic" (but not if already found oblique)
                    elif re.search(r"\bitalic\b", name_str):
                        self._is_italic = True

            # Default to italic if slanted but no explicit name found
            if not self._is_oblique and not self._is_italic:
                self._is_italic = True

        # Check for bold
        if "OS/2" in self.font:
            os2 = self.font["OS/2"]
            if hasattr(os2, "usWeightClass") and os2.usWeightClass == 700:
                self._is_bold = True
            else:
                # Fallback to name check
                if "name" in self.font:
                    for record in self.font["name"].names:
                        if record.nameID in [1, 2, 4, 6]:
                            name_str = record.toUnicode().lower()
                            # Use word boundary regex to avoid false matches
                            if re.search(r"\bbold\b", name_str):
                                self._is_bold = True
                                break


class OS2TableHandler(TableHandler):
    """
    Handles OS/2 table validation and correction.

    Responsibilities:
    - Table version management (upgrade to version 4)
    - Embedding permissions (fsType = 0 for installable)
    - Width classification (monospace detection, xAvgCharWidth)
    - Selection flags (USE_TYPO_METRICS, WWS bits)
    - Metrics (sxHeight, sCapHeight, ulCodePageRange)
    """

    # Unicode codepoints allowed to be zero-width
    ZERO_WIDTH_ALLOWED = {
        0x0000,
        0x0008,
        0x0009,
        0x000D,
        0x001D,
        0x00AD,
        0x034F,
        0x061C,
        0x180E,
        0x200B,
        0x200C,
        0x200D,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2060,
        0x2061,
        0x2062,
        0x2063,
        0x2064,
        0x206A,
        0x206B,
        0x206C,
        0x206D,
        0x206E,
        0x206F,
        0xFEFF,
        0xFFF9,
        0xFFFA,
        0xFFFB,
    }

    # Unicode ranges for combining marks (zero-width allowed)
    COMBINING_MARK_RANGES = [
        (0x0300, 0x036F),
        (0x1AB0, 0x1AFF),
        (0x1DC0, 0x1DFF),
        (0x20D0, 0x20FF),
        (0xFE20, 0xFE2F),
    ]

    def get_table_name(self) -> str:
        return HANDLER_OS2.full_name

    def is_monospace(self) -> bool:
        """Check if font is truly monospaced."""
        try:
            if "hmtx" not in self.font:
                return False

            # Build glyph → codepoint mapping
            glyph_to_cp = self._build_glyph_codepoint_map()

            # Collect non-zero widths (excluding allowed zero-width glyphs)
            non_zero_widths = set()
            hmtx = self.font["hmtx"]

            for glyph_name in self.font.getGlyphOrder():
                width = hmtx[glyph_name][0]

                if width == 0:
                    # Check if zero-width is allowed for this glyph
                    if self._is_zero_width_allowed(glyph_name, glyph_to_cp):
                        continue

                if width > 0:
                    non_zero_widths.add(width)

            # Analyze width distribution
            return self._is_width_distribution_monospace(non_zero_widths)
        except Exception:
            return False

    def _build_glyph_codepoint_map(self) -> dict[str, int]:
        """Build reverse cmap: glyph_name -> codepoint."""
        if "cmap" not in self.font:
            return {}

        cmap = self.font.getBestCmap()
        if not cmap:
            return {}

        return {v: k for k, v in cmap.items()}

    def _is_zero_width_allowed(
        self, glyph_name: str, glyph_to_cp: dict[str, int]
    ) -> bool:
        """Check if glyph is allowed to be zero-width."""
        codepoint = glyph_to_cp.get(glyph_name)

        # Check explicit allowed list
        if codepoint in self.ZERO_WIDTH_ALLOWED:
            return True

        # Check combining mark ranges
        if codepoint and self._is_combining_mark(codepoint):
            return True

        # Check glyph name patterns
        if self._is_combining_mark_name(glyph_name):
            return True

        return False

    def _is_combining_mark(self, codepoint: int) -> bool:
        """Check if codepoint is in combining mark ranges."""
        # Check explicit ranges
        for start, end in self.COMBINING_MARK_RANGES:
            if start <= codepoint <= end:
                return True

        # Check Unicode category
        try:
            category = unicodedata.category(chr(codepoint))
            # Mn = Nonspacing Mark, Me = Enclosing Mark, Cf = Format
            return category in ("Mn", "Me", "Cf")
        except (ValueError, OverflowError):
            return False

    def _is_combining_mark_name(self, glyph_name: str) -> bool:
        """Check if glyph name suggests combining mark."""
        glyph_lower = glyph_name.lower()

        patterns = [
            "comb",
            "grave",
            "acute",
            "tilde",
            "dieresis",
        ]

        if any(pattern in glyph_lower for pattern in patterns):
            return True

        # Check uni03xx and uni04xx patterns
        if glyph_lower.startswith(("uni03", "uni04", "uni20d")):
            return True

        return False

    def _is_width_distribution_monospace(self, widths: set[int]) -> bool:
        """
        Determine if width distribution indicates monospace font.

        Rules:
        - 1 unique width: monospace
        - 2 widths where one is exactly double: CJK monospace
        - Otherwise: proportional
        """
        if len(widths) == 1:
            return True

        if len(widths) == 2:
            return self._is_cjk_monospace(widths)

        return False

    def _is_cjk_monospace(self, widths: set[int]) -> bool:
        """
        Check if font is CJK monospace (narrow + double-width glyphs).

        CJK monospace fonts have:
        - Two widths: narrow and exactly double
        - Significant number of glyphs at each width
        """
        assert len(widths) == 2, "CJK monospace check requires exactly 2 unique widths"
        widths_list = sorted(widths)

        # Check if wider is exactly double
        if widths_list[1] != widths_list[0] * 2:
            return False

        # Count glyphs at each width
        hmtx = self.font["hmtx"]
        narrow_count = sum(1 for w in hmtx.values() if w[0] == widths_list[0])
        wide_count = sum(1 for w in hmtx.values() if w[0] == widths_list[1])

        # If we have a significant number of both widths, it's likely CJK
        return narrow_count > 10 and wide_count > 10

    def validate(self) -> Dict[str, Dict[str, Any]]:
        """Validate OS/2 table state."""
        if "OS/2" not in self.font:
            self._track_validation("table_exists", False, "OS/2 table missing")
            return self.validations

        try:
            os2 = self.font["OS/2"]
        except (TTLibError, AttributeError, IndexError, ValueError) as e:
            # OS/2 table is corrupted and cannot be read
            error_msg = str(e) if str(e) else "Unknown error"
            self._track_validation(
                "table_readable",
                False,
                f"OS/2 table is corrupted and cannot be read: {error_msg}",
            )
            return self.validations

        # Fix corrupted panose field (should be an object, not a string)
        try:
            if hasattr(os2, "panose"):
                # Check if panose is corrupted (should be an object, not a string)
                if isinstance(os2.panose, str):
                    # Reconstruct panose as a proper object
                    new_panose = Panose()
                    # Try to decode the string if it looks like raw bytes
                    if len(os2.panose) >= 10:
                        try:
                            # Interpret as bytes
                            panose_bytes = os2.panose.encode("latin1")[:10]
                            new_panose.bFamilyType = (
                                panose_bytes[0] if len(panose_bytes) > 0 else 0
                            )
                            new_panose.bSerifStyle = (
                                panose_bytes[1] if len(panose_bytes) > 1 else 0
                            )
                            new_panose.bWeight = (
                                panose_bytes[2] if len(panose_bytes) > 2 else 0
                            )
                            new_panose.bProportion = (
                                panose_bytes[3] if len(panose_bytes) > 3 else 0
                            )
                            new_panose.bContrast = (
                                panose_bytes[4] if len(panose_bytes) > 4 else 0
                            )
                            new_panose.bStrokeVariation = (
                                panose_bytes[5] if len(panose_bytes) > 5 else 0
                            )
                            new_panose.bArmStyle = (
                                panose_bytes[6] if len(panose_bytes) > 6 else 0
                            )
                            new_panose.bLetterForm = (
                                panose_bytes[7] if len(panose_bytes) > 7 else 0
                            )
                            new_panose.bMidline = (
                                panose_bytes[8] if len(panose_bytes) > 8 else 0
                            )
                            new_panose.bXHeight = (
                                panose_bytes[9] if len(panose_bytes) > 9 else 0
                            )
                        except Exception:
                            # If decode fails, use all zeros
                            pass

                    os2.panose = new_panose

                    if self.verbose:
                        self.log(
                            "Fixed corrupted panose field (was string, now proper object)"
                        )
        except Exception as e:
            if self.verbose:
                self.log(f"Could not fix panose field: {e}")

        # Version validation
        self.validate_field(
            "version_current",
            "OS/2",
            "version",
            lambda v: (v >= 4, f"Current: {v}, Expected: 4 (minimum required)"),
        )

        # Embedding validation
        self.validate_field(
            "fstype_installable",
            "OS/2",
            "fsType",
            lambda v: (v == 0, f"Current: {v}, Expected: 0 (installable fonts)"),
        )

        # Monospace validation
        self.validate_condition(
            "monospace_consistent",
            lambda: self._validate_monospace_flags(),
            "monospace detection",
        )

        # fsSelection flags validation
        self.validate_condition(
            "fsselection_flags_set",
            lambda: self._validate_fsselection_flags(),
            "fsSelection flags",
        )

        return self.validations

    def _validate_monospace_flags(self) -> Tuple[bool, str]:
        """Helper: validate monospace flags consistency."""
        is_mono = self.is_monospace()
        is_correct = self._is_monospace_flags_correct(is_mono)
        return (is_correct, f"Status: monospace={is_mono}, Compliant: {is_correct}")

    def _validate_fsselection_flags(self) -> Tuple[bool, str]:
        """Helper: validate fsSelection flags."""
        os2 = self.font["OS/2"]
        version = getattr(os2, "version", 0)
        if version < 4:
            return (
                False,
                f"Check: fsSelection flags (FAIL - requires OS/2 v4+, current: {version})",
            )

        fs_selection = getattr(os2, "fsSelection", 0)
        has_use_typo = FS_SELECTION.flags["USE_TYPO_METRICS"].is_set(fs_selection)
        has_wws = FS_SELECTION.flags["WWS"].is_set(fs_selection)
        is_valid = has_use_typo and has_wws
        return (
            is_valid,
            f"Status: USE_TYPO_METRICS={has_use_typo}, WWS={has_wws}, Compliant: {is_valid}",
        )

    def _is_monospace_flags_correct(self, is_mono: bool) -> bool:
        """
        Verify that monospace-related flags are consistent with actual glyph metrics.

        Checks:
            - OS/2.panose.bProportion should be 9 for monospace fonts
            - post.isFixedPitch should be 1 for monospace fonts

        Args:
            is_mono: Whether the font is actually monospace based on glyph analysis

        Returns:
            True if flags correctly reflect the monospace state

        Note:
            For proportional fonts, these flags should NOT be set. This method
            returns False if any discrepancy is found.
        """
        if not is_mono:
            # For proportional fonts, check that flags are cleared
            if "OS/2" in self.font and hasattr(self.font["OS/2"], "panose"):
                if self.font["OS/2"].panose.bProportion == 9:
                    return False
            if "post" in self.font and self.font["post"].isFixedPitch != 0:
                return False
            return True
        else:
            # For monospace fonts, check that flags are set
            if "OS/2" in self.font and hasattr(self.font["OS/2"], "panose"):
                if self.font["OS/2"].panose.bProportion != 9:
                    return False
            if "post" in self.font and self.font["post"].isFixedPitch != 1:
                return False
            return True

    def fix(self) -> bool:
        """Apply all OS/2 fixes based on validation results."""
        if "OS/2" not in self.font:
            return False

        # Check if table is readable - if validation failed due to corruption, skip fixes
        if not self.validations.get("table_readable", {}).get("valid", True):
            # Table is corrupted and cannot be fixed
            return False

        # Verify OS/2 table is accessible before attempting fixes
        try:
            _ = self.font["OS/2"]
        except (TTLibError, AttributeError, IndexError, ValueError):
            # Table became unreadable, skip fixes
            return False

        return any(
            [
                self._fix_version(),
                self._fix_fstype(),
                self._fix_monospace(),
                self._fix_fsselection_flags(),
            ]
        )

    @conditional_fix("version_current")
    def _fix_version(self) -> bool:
        """Upgrade OS/2 table to version 4."""
        os2 = self.font["OS/2"]
        old_version = os2.version

        if old_version >= 4:
            return False

        old_sxHeight = getattr(os2, "sxHeight", None)
        old_sCapHeight = getattr(os2, "sCapHeight", None)
        old_cp1 = getattr(os2, "ulCodePageRange1", None)
        old_cp2 = getattr(os2, "ulCodePageRange2", None)

        os2.version = 4

        # Initialize fields added in version 1 (if upgrading from v0)
        if old_version < 1:
            if not hasattr(os2, "ulCodePageRange1"):
                os2.ulCodePageRange1 = 0
            if not hasattr(os2, "ulCodePageRange2"):
                os2.ulCodePageRange2 = 0

        # Initialize fields added in version 2 (if upgrading from v0 or v1)
        if old_version < 2:
            if not hasattr(os2, "sxHeight"):
                os2.sxHeight = 0
            if not hasattr(os2, "sCapHeight"):
                os2.sCapHeight = 0
            if not hasattr(os2, "usDefaultChar"):
                os2.usDefaultChar = 0
            if not hasattr(os2, "usBreakChar"):
                os2.usBreakChar = 32
            if not hasattr(os2, "usMaxContext"):
                os2.usMaxContext = 0

        # Calculate metrics if missing
        self._calculate_metrics()

        # Recalculate code page ranges if needed
        if os2.ulCodePageRange1 == 0 and os2.ulCodePageRange2 == 0:
            try:
                os2.recalcCodePageRanges(self.font)
            except Exception:
                pass

        # Track changes using ChangeBuilder
        self.track_changes().add("version", old_version, 4).add_if_changed(
            "sxHeight", old_sxHeight or 0, os2.sxHeight
        ).add_if_changed(
            "sCapHeight", old_sCapHeight or 0, os2.sCapHeight
        ).add_if_changed(
            "ulCodePageRange",
            f"({old_cp1 or 0}, {old_cp2 or 0})",
            f"({os2.ulCodePageRange1}, {os2.ulCodePageRange2})",
        ).commit()

        return True

    def _calculate_metrics(self):
        """Calculate sxHeight and sCapHeight from glyphs."""
        os2 = self.font["OS/2"]

        # Calculate sxHeight from 'x' glyph if available
        if os2.sxHeight == 0:
            try:
                glyph_set = self.font.getGlyphSet()
                if "x" in glyph_set:
                    from fontTools.pens.boundsPen import BoundsPen

                    pen = BoundsPen(glyph_set)
                    glyph_set["x"].draw(pen)
                    if pen.bounds:
                        os2.sxHeight = int(round(pen.bounds[3]))
            except Exception:
                pass

        # Calculate sCapHeight from 'H' glyph if available
        if os2.sCapHeight == 0:
            try:
                glyph_set = self.font.getGlyphSet()
                if "H" in glyph_set:
                    from fontTools.pens.boundsPen import BoundsPen

                    pen = BoundsPen(glyph_set)
                    glyph_set["H"].draw(pen)
                    if pen.bounds:
                        os2.sCapHeight = int(round(pen.bounds[3]))
            except Exception:
                pass

    @conditional_fix("fstype_installable")
    def _fix_fstype(self) -> bool:
        """Set fsType to 0 (installable)."""
        os2 = self.font["OS/2"]
        old_fstype = os2.fsType

        if old_fstype == 0:
            return False

        os2.fsType = 0
        self._track_change("fsType", old_fstype, 0, True)
        return True

    @conditional_fix("monospace_consistent")
    def _fix_monospace(self) -> bool:
        """Fix monospace attributes."""
        os2 = self.font["OS/2"]
        is_mono = self.is_monospace()

        old_proportion = (
            getattr(os2.panose, "bProportion", None) if hasattr(os2, "panose") else None
        )
        old_spacing = (
            getattr(os2.panose, "bSpacing", None) if hasattr(os2, "panose") else None
        )
        old_post_fixed = self.font["post"].isFixedPitch if "post" in self.font else None
        old_cff_fixed = None
        old_xavg = os2.xAvgCharWidth

        changed = False

        if is_mono:
            # Set monospace flags
            if hasattr(os2, "panose") and os2.panose.bFamilyType in [2, 3, 4, 5]:
                if os2.panose.bProportion != 9:
                    os2.panose.bProportion = 9
                    changed = True

            if hasattr(os2, "panose") and os2.panose.bFamilyType in [3, 5]:
                if os2.panose.bSpacing != 3:
                    os2.panose.bSpacing = 3
                    changed = True

            if "post" in self.font and self.font["post"].isFixedPitch != 1:
                self.font["post"].isFixedPitch = 1
                changed = True

            # Handle CFF (skip CFF2 for variable fonts)
            if "CFF " in self.font and "CFF2" not in self.font:
                try:
                    cff = self.font["CFF "].cff
                    if hasattr(cff, "topDictIndex") and len(cff.topDictIndex) > 0:
                        top_dict = cff.topDictIndex[0]
                        old_cff_fixed = getattr(
                            top_dict, "isFixedPitch", False
                        )  # Capture BEFORE
                        if not old_cff_fixed:
                            top_dict.isFixedPitch = True
                            changed = True
                except Exception:
                    old_cff_fixed = None
                    pass
        else:
            # Clear monospace flags
            if hasattr(os2, "panose") and os2.panose.bProportion == 9:
                os2.panose.bProportion = 0
                changed = True

            if "post" in self.font and self.font["post"].isFixedPitch != 0:
                self.font["post"].isFixedPitch = 0
                changed = True

            # Clear CFF (skip CFF2 for variable fonts)
            if "CFF " in self.font and "CFF2" not in self.font:
                try:
                    cff = self.font["CFF "].cff
                    if hasattr(cff, "topDictIndex") and len(cff.topDictIndex) > 0:
                        top_dict = cff.topDictIndex[0]
                        old_cff_fixed = getattr(
                            top_dict, "isFixedPitch", False
                        )  # Capture BEFORE
                        if old_cff_fixed:
                            top_dict.isFixedPitch = False
                            changed = True
                except Exception:
                    old_cff_fixed = None
                    pass

        # Recalculate xAvgCharWidth
        if "hmtx" in self.font:
            try:
                os2.recalcAvgCharWidth(self.font)
                if os2.xAvgCharWidth != old_xavg:
                    changed = True
            except (AttributeError, KeyError, TypeError, struct.error):
                # Font has unusual metrics table structure, skip recalculation
                if self.verbose:
                    self.log(
                        "Could not recalculate xAvgCharWidth (unusual metrics table)"
                    )
                pass
            except Exception as e:
                # Catch struct.error and other unexpected errors
                if self.verbose:
                    self.log(f"Could not recalculate xAvgCharWidth: {e}")
                pass

        if changed:
            # Track changes using ChangeBuilder
            builder = self.track_changes()
            builder.add_info("is_monospace", is_mono)

            if old_proportion is not None:
                builder.add_if_changed(
                    "bProportion", old_proportion, os2.panose.bProportion
                )
            if old_spacing is not None:
                builder.add_if_changed("bSpacing", old_spacing, os2.panose.bSpacing)
            if old_post_fixed is not None:
                builder.add_if_changed(
                    "post.isFixedPitch", old_post_fixed, self.font["post"].isFixedPitch
                )
            if "CFF " in self.font:
                try:
                    cff = self.font["CFF "].cff
                    new_cff = getattr(cff.topDictIndex[0], "isFixedPitch", False)
                    if old_cff_fixed is not None:
                        builder.add_if_changed(
                            "CFF.isFixedPitch", old_cff_fixed, new_cff
                        )
                except Exception:
                    pass
            builder.add_if_changed("xAvgCharWidth", old_xavg, os2.xAvgCharWidth)
            builder.commit()

        return changed

    @conditional_fix("fsselection_flags_set")
    def _fix_fsselection_flags(self) -> bool:
        """Set USE_TYPO_METRICS and WWS flags."""
        os2 = self.font["OS/2"]

        if os2.version < 4:
            return False

        original_fs = os2.fsSelection
        os2.fsSelection = FS_SELECTION.flags["USE_TYPO_METRICS"].set(os2.fsSelection)
        os2.fsSelection = FS_SELECTION.flags["WWS"].set(os2.fsSelection)

        if os2.fsSelection != original_fs:
            self._track_bitfield_change(
                "fsSelection", original_fs, os2.fsSelection, FS_SELECTION
            )
            return True

        return False


class StyleConsistencyHandler(TableHandler):
    """
    Ensures consistency across post, hhea, OS/2, and head tables for style.

    Responsibilities:
    - post.italicAngle matches calculated angle
    - hhea.caretSlope matches italic angle
    - OS/2.fsSelection matches style (italic, oblique, bold, regular)
    - head.macStyle matches style (bold, italic)
    """

    def __init__(self, font: TTFont, verbose: bool = False):
        super().__init__(font, verbose)
        self.style_analyzer = FontStyleAnalyzer(font)

    def get_table_name(self) -> str:
        return HANDLER_STYLE.full_name

    def validate(self) -> Dict[str, Dict[str, Any]]:
        """Validate style consistency across tables."""
        # Validate post.italicAngle
        self.validate_field(
            "italic_angle_correct",
            "post",
            "italicAngle",
            lambda v: (
                abs(v - self.style_analyzer.italic_angle) < 0.1,
                f"Current: {v}°, Expected: {self.style_analyzer.italic_angle}°",
            ),
        )

        # Validate hhea.caretSlope
        self.validate_condition(
            "caret_slope_correct", lambda: self._validate_caret_slope(), "caret slope"
        )

        # Validate OS/2.fsSelection
        self.validate_condition(
            "fsselection_style_match",
            lambda: self._validate_fsselection_style(),
            "fsSelection style flags",
        )

        # Validate head.macStyle
        self.validate_condition(
            "macstyle_correct", lambda: self._validate_macstyle(), "macStyle flags"
        )

        return self.validations

    def _validate_caret_slope(self) -> Tuple[bool, str]:
        """Helper: validate hhea caret slope."""
        if "hhea" not in self.font or "head" not in self.font:
            return (True, "hhea or head table missing")

        hhea = self.font["hhea"]
        head = self.font["head"]
        calculated = self.style_analyzer.italic_angle

        if calculated == 0:
            expected_rise = 1
            expected_run = 0
        else:
            upm = head.unitsPerEm
            expected_rise = upm
            expected_run = round(math.tan(math.radians(-1 * calculated)) * upm)

        caret_correct = (
            hhea.caretSlopeRise == expected_rise and hhea.caretSlopeRun == expected_run
        )
        return (
            caret_correct,
            f"Current: rise={hhea.caretSlopeRise}, run={hhea.caretSlopeRun}; "
            f"Expected: rise={expected_rise}, run={expected_run}",
        )

    def _validate_fsselection_style(self) -> Tuple[bool, str]:
        """Helper: validate fsSelection style flags."""
        if "OS/2" not in self.font:
            return (True, "OS/2 table missing")

        os2 = self.font["OS/2"]
        fs_italic = FS_SELECTION.flags["ITALIC"].is_set(os2.fsSelection)
        fs_oblique = (
            FS_SELECTION.flags["OBLIQUE"].is_set(os2.fsSelection)
            if os2.version >= 4
            else False
        )
        fs_bold = FS_SELECTION.flags["BOLD"].is_set(os2.fsSelection)

        style_matches = (
            fs_italic == self.style_analyzer.is_italic
            and fs_oblique == self.style_analyzer.is_oblique
            and fs_bold == self.style_analyzer.is_bold
        )

        return (
            style_matches,
            f"Status: Italic={fs_italic} (expected {self.style_analyzer.is_italic}), "
            f"Oblique={fs_oblique} (expected {self.style_analyzer.is_oblique}), "
            f"Bold={fs_bold} (expected {self.style_analyzer.is_bold}), "
            f"Compliant: {style_matches}",
        )

    def _validate_macstyle(self) -> Tuple[bool, str]:
        """Helper: validate macStyle flags."""
        if "head" not in self.font:
            return (True, "head table missing")

        head = self.font["head"]
        mac_bold = MAC_STYLE.flags["BOLD"].is_set(head.macStyle)
        mac_italic = MAC_STYLE.flags["ITALIC"].is_set(head.macStyle)

        mac_correct = mac_bold == self.style_analyzer.is_bold and mac_italic == (
            self.style_analyzer.is_italic or self.style_analyzer.is_oblique
        )

        return (
            mac_correct,
            f"Status: bold={mac_bold} (expected {self.style_analyzer.is_bold}), "
            f"italic={mac_italic} (expected {self.style_analyzer.is_italic or self.style_analyzer.is_oblique}), "
            f"Compliant: {mac_correct}",
        )

    def fix(self) -> bool:
        """Fix style consistency issues."""
        return any(
            [
                self._fix_italic_angle(),
                self._fix_caret_slope(),
                self._fix_style_flags(),
            ]
        )

    @conditional_fix("italic_angle_correct")
    def _fix_italic_angle(self) -> bool:
        """Fix post.italicAngle."""
        if "post" not in self.font:
            return False

        post = self.font["post"]
        calculated = self.style_analyzer.italic_angle
        old_italic_angle = post.italicAngle

        if abs(calculated - post.italicAngle) > 0.1:
            post.italicAngle = calculated
            self._track_change(
                "italic_angle", f"{old_italic_angle} deg", f"{calculated} deg", True
            )
            return True

        return False

    @conditional_fix("caret_slope_correct")
    def _fix_caret_slope(self) -> bool:
        """Fix hhea.caretSlope."""
        if "hhea" not in self.font or "head" not in self.font:
            return False

        hhea = self.font["hhea"]
        head = self.font["head"]
        calculated = self.style_analyzer.italic_angle

        original_rise = hhea.caretSlopeRise
        original_run = hhea.caretSlopeRun

        if calculated == 0:
            hhea.caretSlopeRise = 1
            hhea.caretSlopeRun = 0
        else:
            upm = head.unitsPerEm
            hhea.caretSlopeRise = upm
            # Calculate run but clamp to valid range for signed 16-bit integer
            calculated_run = round(math.tan(math.radians(-1 * calculated)) * upm)
            original_calculated = calculated_run

            # Clamp to signed 16-bit range
            hhea.caretSlopeRun = clamp_signed_16bit(calculated_run)

            if hhea.caretSlopeRun != original_calculated and self.verbose:
                self.log(
                    f"Clamped caretSlopeRun from {original_calculated} to {hhea.caretSlopeRun} "
                    f"(signed 16-bit range: {SIGNED_16BIT_MIN} to {SIGNED_16BIT_MAX})"
                )

        if hhea.caretSlopeRise != original_rise or hhea.caretSlopeRun != original_run:
            self._track_change(
                "caret_slope_rise",
                original_rise if original_rise is not None else "N/A",
                hhea.caretSlopeRise,
                original_rise is not None and hhea.caretSlopeRise != original_rise,
            )
            self._track_change(
                "caret_slope_run",
                original_run if original_run is not None else "N/A",
                hhea.caretSlopeRun,
                original_run is not None and hhea.caretSlopeRun != original_run,
            )
            return True

        return False

    @conditional_fix("fsselection_style_match", "macstyle_correct")
    def _fix_style_flags(self) -> bool:
        """Fix OS/2.fsSelection and head.macStyle."""
        if "OS/2" not in self.font or "head" not in self.font:
            return False

        os2 = self.font["OS/2"]
        head = self.font["head"]

        original_fs = os2.fsSelection
        original_mac = head.macStyle

        # Apply fixes
        self._apply_fsselection_style(os2)
        self._apply_macstyle(head)

        # Track changes
        any_changed = self._track_style_flag_changes(
            original_fs, os2.fsSelection, original_mac, head.macStyle
        )

        return any_changed

    def _apply_fsselection_style(self, os2: Any) -> None:
        """Set fsSelection bits based on style analysis."""
        # Clear style bits
        os2.fsSelection = FS_SELECTION.flags["ITALIC"].clear(os2.fsSelection)
        os2.fsSelection = FS_SELECTION.flags["BOLD"].clear(os2.fsSelection)
        os2.fsSelection = FS_SELECTION.flags["REGULAR"].clear(os2.fsSelection)
        if os2.version >= 4:
            os2.fsSelection = FS_SELECTION.flags["OBLIQUE"].clear(os2.fsSelection)

        # Set appropriate bits
        if self.style_analyzer.is_italic:
            os2.fsSelection = FS_SELECTION.flags["ITALIC"].set(os2.fsSelection)

        if self.style_analyzer.is_oblique:
            if os2.version >= 4:
                os2.fsSelection = FS_SELECTION.flags["OBLIQUE"].set(os2.fsSelection)
            else:
                # Verbose warning: oblique bit requires OS/2 v4+
                if self.verbose:
                    self.log(
                        f"Font is oblique but OS/2 version {os2.version} < 4, cannot set oblique bit"
                    )

        if self.style_analyzer.is_bold:
            os2.fsSelection = FS_SELECTION.flags["BOLD"].set(os2.fsSelection)

        if (
            not self.style_analyzer.is_bold
            and not self.style_analyzer.is_italic
            and not self.style_analyzer.is_oblique
        ):
            os2.fsSelection = FS_SELECTION.flags["REGULAR"].set(os2.fsSelection)

    def _apply_macstyle(self, head: Any) -> None:
        """Set macStyle bits based on style analysis."""
        head.macStyle = 0

        if self.style_analyzer.is_bold:
            head.macStyle = MAC_STYLE.flags["BOLD"].set(head.macStyle)

        if self.style_analyzer.is_italic or self.style_analyzer.is_oblique:
            head.macStyle = MAC_STYLE.flags["ITALIC"].set(head.macStyle)

    def _track_style_flag_changes(
        self, old_fs: int, new_fs: int, old_mac: int, new_mac: int
    ) -> bool:
        """Track all style flag changes with detailed deltas."""
        any_changed = False

        if new_fs != old_fs:
            self._track_bitfield_change("fsSelection", old_fs, new_fs, FS_SELECTION)
            any_changed = True

        if new_mac != old_mac:
            self._track_bitfield_change("macStyle", old_mac, new_mac, MAC_STYLE)
            any_changed = True

        if any_changed:
            # Track info changes
            os2 = self.font["OS/2"]
            self._track_info_changes(
                [
                    ("is_italic", self.style_analyzer.is_italic),
                    ("is_oblique", self.style_analyzer.is_oblique),
                    ("is_bold", self.style_analyzer.is_bold),
                    (
                        "is_regular",
                        FS_SELECTION.flags["REGULAR"].is_set(os2.fsSelection),
                    ),
                ]
            )

        return any_changed


class GlyphHandler(TableHandler):
    """
    Handles glyph-level fixes.

    Responsibilities:
    - .notdef glyph presence and structure
    - nbsp (U+00A0) presence and metrics
    """

    def get_table_name(self) -> str:
        return HANDLER_GLYPH.full_name

    def validate(self) -> Dict[str, Dict[str, Any]]:
        """Validate glyph issues."""
        self.validate_condition(
            "notdef_valid", lambda: self._validate_notdef_helper(), ".notdef validation"
        )
        self.validate_condition(
            "nbsp_present", lambda: self._validate_nbsp_helper(), "nbsp validation"
        )
        return self.validations

    def _validate_notdef_helper(self) -> Tuple[bool, str]:
        """Helper: check if .notdef is valid."""
        is_empty = False

        if "glyf" in self.font:
            glyf = self.font["glyf"]
            if ".notdef" in glyf:
                notdef = glyf[".notdef"]
                is_empty = notdef.numberOfContours == 0
        elif "CFF " in self.font:
            try:
                cff = self.font["CFF "].cff
                top_dict = cff.topDictIndex[0]
                char_strings = top_dict.CharStrings
                if ".notdef" in char_strings:
                    is_empty = self._is_cff_notdef_empty(char_strings[".notdef"])
            except Exception:
                pass

        is_valid = not is_empty
        return (
            is_valid,
            f"Status: .notdef glyph {'empty' if is_empty else 'valid'}, Compliant: {is_valid}",
        )

    def _validate_nbsp_helper(self) -> Tuple[bool, str]:
        """Helper: check if nbsp exists and has correct width."""
        if "cmap" not in self.font:
            return (True, "No cmap table")

        cmap = self.font.getBestCmap()
        if cmap is None:
            return (True, "No cmap")

        nbsp_present = 0x00A0 in cmap
        if not nbsp_present:
            return (
                False,
                "Status: nbsp (U+00A0) missing, Compliant: False",
            )

        # Also validate width if both space and nbsp exist
        if "hmtx" in self.font and 0x0020 in cmap:
            hmtx = self.font["hmtx"]
            space_glyph = cmap[0x0020]
            nbsp_glyph = cmap[0x00A0]
            space_width = hmtx[space_glyph][0]
            nbsp_width = hmtx[nbsp_glyph][0]
            width_correct = space_width == nbsp_width

            # Track width validation separately
            self._track_validation(
                "nbsp_width_correct",
                width_correct,
                f"Current: {nbsp_width}, Expected: {space_width}"
                if not width_correct
                else f"Current: {nbsp_width}, Expected: {space_width} (matches)",
            )
        else:
            self._track_validation("nbsp_width_correct", True, "Cannot validate width")

        return (
            True,
            "Status: nbsp (U+00A0) present, Compliant: True",
        )

    def fix(self) -> bool:
        """Fix glyph issues."""
        any_changed = False

        if not self.validations.get("notdef_valid", {}).get("valid", True):
            any_changed |= self._fix_notdef()

        if not self.validations.get("nbsp_present", {}).get("valid", True):
            any_changed |= self._fix_nbsp_missing()

        if not self.validations.get("nbsp_width_correct", {}).get("valid", True):
            any_changed |= self._fix_nbsp_width()

        return any_changed

    def _fix_notdef(self) -> bool:
        """Create .notdef glyph if empty."""
        if "glyf" in self.font:
            return self._fix_notdef_truetype()
        elif "CFF " in self.font:
            return self._fix_notdef_cff()
        return False

    def _fix_notdef_truetype(self) -> bool:
        """Fix .notdef for TrueType fonts."""
        glyf = self.font["glyf"]
        if ".notdef" not in glyf:
            return False

        notdef = glyf[".notdef"]
        if notdef.numberOfContours != 0:
            return False  # Already has contours

        # For variable fonts, try copying from space/nbsp first to avoid gvar issues
        is_variable = "gvar" in self.font
        if is_variable and self._copy_notdef_from_fallback(is_cff=False):
            return True

        # If we didn't copy, create new glyph (but skip for variable fonts)
        if is_variable:
            return False  # Skip fixing .notdef in variable fonts to avoid gvar table issues

        return self._create_notdef_truetype()

    def _fix_notdef_cff(self) -> bool:
        """Fix .notdef for CFF fonts."""
        try:
            cff = self.font["CFF "].cff
            top_dict = cff.topDictIndex[0]
            char_strings = top_dict.CharStrings

            if ".notdef" not in char_strings:
                return False

            cs = char_strings[".notdef"]
            if not self._is_cff_notdef_empty(cs):
                return False  # Already has content

            # For variable fonts, try copying from space/nbsp first to avoid gvar issues
            is_variable = "gvar" in self.font
            if is_variable and self._copy_notdef_from_fallback(is_cff=True):
                return True

            # If we didn't copy, create new glyph (but skip for variable fonts)
            if is_variable:
                return False  # Skip fixing .notdef in variable fonts to avoid gvar table issues

            return self._create_notdef_cff()
        except Exception:
            return False

    def _copy_notdef_from_fallback(self, is_cff: bool) -> bool:
        """Try copying .notdef from space (U+0020) or nbsp (U+00A0)."""
        if "cmap" not in self.font:
            return False

        cmap = self.font.getBestCmap()
        if not cmap:
            return False

        # Try space first, then nbsp
        for codepoint in [0x0020, 0x00A0]:
            if codepoint not in cmap:
                continue

            source_glyph = cmap[codepoint]

            if is_cff:
                if self._copy_notdef_cff(source_glyph):
                    self._track_notdef_copy(source_glyph, is_cff=True)
                    return True
            else:
                if self._copy_notdef_truetype(source_glyph):
                    self._track_notdef_copy(source_glyph, is_cff=False)
                    return True

        return False

    def _copy_notdef_truetype(self, source_glyph: str) -> bool:
        """Copy TrueType glyph to .notdef."""
        glyf = self.font["glyf"]
        if source_glyph not in glyf:
            return False

        source = glyf[source_glyph]
        if source.numberOfContours == 0:
            return False  # Don't copy empty glyphs

        import copy

        glyf[".notdef"] = copy.deepcopy(source)

        # Remove gvar variations for .notdef
        self._remove_gvar_for_notdef()

        # Copy metrics
        if "hmtx" in self.font and source_glyph in self.font["hmtx"]:
            source_width = self.font["hmtx"][source_glyph][0]
            source_lsb = self.font["hmtx"][source_glyph][1]
            self.font["hmtx"][".notdef"] = (source_width, source_lsb)

        return True

    def _copy_notdef_cff(self, source_glyph: str) -> bool:
        """Copy CFF charstring to .notdef."""
        try:
            cff = self.font["CFF "].cff
            char_strings = cff.topDictIndex[0].CharStrings

            if source_glyph not in char_strings:
                return False

            source_cs = char_strings[source_glyph]

            # Check source is not empty
            program = source_cs.program
            if len(program) <= 2 and program[-1] == "endchar":
                return False

            import copy

            char_strings[".notdef"] = copy.deepcopy(source_cs)

            # Remove gvar variations for .notdef
            self._remove_gvar_for_notdef()

            # Copy metrics
            if "hmtx" in self.font and source_glyph in self.font["hmtx"]:
                source_width = self.font["hmtx"][source_glyph][0]
                source_lsb = self.font["hmtx"][source_glyph][1]
                self.font["hmtx"][".notdef"] = (source_width, source_lsb)

            return True
        except Exception:
            return False

    def _create_notdef_truetype(self) -> bool:
        """Create new .notdef glyph for TrueType fonts."""
        from fontTools.pens.ttGlyphPen import TTGlyphPen

        upm = self.font["head"].unitsPerEm
        width = int(upm * 0.5)
        height = int(upm * 0.75)
        thickness = max(int(upm * 0.05), 50)

        pen = TTGlyphPen(self.font.getGlyphSet())
        self._draw_notdef_outline(pen, width, height, thickness)

        glyf = self.font["glyf"]
        glyf[".notdef"] = pen.glyph()

        # Update hmtx
        if "hmtx" in self.font:
            self.font["hmtx"][".notdef"] = (width, thickness)

        self.track_changes().add(".notdef_empty", True, False).add(
            ".notdef_width", 0, width
        ).add_info(".notdef_contours", 2).commit()

        return True

    def _create_notdef_cff(self) -> bool:
        """Create new .notdef glyph for CFF fonts."""
        from fontTools.pens.t2CharStringPen import T2CharStringPen

        upm = self.font["head"].unitsPerEm
        width = int(upm * 0.5)
        height = int(upm * 0.75)
        thickness = max(int(upm * 0.05), 50)

        # Get advance width from hmtx if available
        if "hmtx" in self.font and ".notdef" in self.font["hmtx"]:
            width = self.font["hmtx"][".notdef"][0]

        cff = self.font["CFF "].cff
        char_strings = cff.topDictIndex[0].CharStrings
        glyph_set = self.font.getGlyphSet()

        pen = T2CharStringPen(width=width, glyphSet=glyph_set)
        self._draw_notdef_outline(pen, width, height, thickness)

        new_charstring = pen.getCharString()

        # Preserve private dictionary reference
        original_charstring = char_strings[".notdef"]
        if hasattr(original_charstring, "private"):
            new_charstring.private = original_charstring.private

        char_strings[".notdef"] = new_charstring

        # Update hmtx
        if "hmtx" in self.font:
            self.font["hmtx"][".notdef"] = (width, thickness)

        self.track_changes().add(".notdef_empty", True, False).add(
            ".notdef_width", 0, width
        ).add_info(".notdef_contours", 2).commit()

        return True

    def _draw_notdef_outline(
        self, pen: Any, width: int, height: int, thickness: int
    ) -> None:
        """Draw standard .notdef outline (two rectangles)."""
        # Outer rectangle
        pen.moveTo((thickness, thickness))
        pen.lineTo((thickness, height - thickness))
        pen.lineTo((width - thickness, height - thickness))
        pen.lineTo((width - thickness, thickness))
        pen.closePath()

        # Inner rectangle
        pen.moveTo((thickness * 2, thickness * 2))
        pen.lineTo((width - thickness * 2, thickness * 2))
        pen.lineTo((width - thickness * 2, height - thickness * 2))
        pen.lineTo((thickness * 2, height - thickness * 2))
        pen.closePath()

    def _is_cff_notdef_empty(self, charstring: Any) -> bool:
        """Check if CFF .notdef charstring is empty."""
        try:
            program = charstring.program
            # Empty glyph: only endchar or width + endchar
            return len(program) <= 2 and program[-1] == "endchar"
        except Exception:
            # Fallback to bytecode check
            try:
                return len(charstring.bytecode) < 10
            except Exception:
                return False

    def _remove_gvar_for_notdef(self) -> None:
        """Remove .notdef variations from gvar table."""
        if "gvar" not in self.font:
            return

        try:
            gvar = self.font["gvar"]
            gid = self.font.getGlyphID(".notdef")
            if gid in gvar.variations:
                del gvar.variations[gid]
        except (KeyError, AttributeError, IndexError):
            pass

    def _track_notdef_copy(self, source_glyph: str, is_cff: bool) -> None:
        """Track .notdef changes when copied from another glyph."""
        width = 0
        contours = 0

        if "hmtx" in self.font and source_glyph in self.font["hmtx"]:
            width = self.font["hmtx"][source_glyph][0]

        if not is_cff and "glyf" in self.font:
            if source_glyph in self.font["glyf"]:
                contours = self.font["glyf"][source_glyph].numberOfContours

        self.track_changes().add(".notdef_empty", True, False).add_info(
            ".notdef_contours", contours
        ).add(".notdef_width", 0, width).commit()

    def _fix_nbsp_missing(self) -> bool:
        """Add nbsp if missing."""
        if "cmap" not in self.font:
            return False

        cmap = self.font.getBestCmap()
        if cmap is None or 0x00A0 in cmap:
            return False

        # Add nbsp by copying space (U+0020) if it exists
        if 0x0020 in cmap:
            space_glyph = cmap[0x0020]
            # Add mapping to cmap tables
            for table in self.font["cmap"].tables:
                if 0x0020 in table.cmap:
                    table.cmap[0x00A0] = space_glyph

            self._track_change("nbsp_exists", False, True, True)
            self._track_change("nbsp_glyph", None, space_glyph, True)
            return True

        return False

    def _fix_nbsp_width(self) -> bool:
        """Fix nbsp width to match space."""
        if "cmap" not in self.font or "hmtx" not in self.font:
            return False

        cmap = self.font.getBestCmap()
        if not cmap or 0x0020 not in cmap or 0x00A0 not in cmap:
            return False

        hmtx = self.font["hmtx"]
        space_glyph = cmap[0x0020]
        nbsp_glyph = cmap[0x00A0]

        space_width = hmtx[space_glyph][0]
        nbsp_width = hmtx[nbsp_glyph][0]

        if space_width != nbsp_width:
            # Copy space metrics to nbsp
            hmtx[nbsp_glyph] = hmtx[space_glyph]
            self._track_change(
                "space_width", space_width, space_width, False, info_only=True
            )
            self._track_change("nbsp_width", nbsp_width, space_width, True)
            return True

        return False


class KernHandler(TableHandler):
    """
    Handles kern table cleanup.

    Implements fix 10:
    - Remove legacy kern table if GPOS exists
    """

    def get_table_name(self) -> str:
        return HANDLER_KERN.full_name

    def validate(self) -> Dict[str, Dict[str, Any]]:
        """Validate kerning table state."""
        has_gpos = "GPOS" in self.font
        has_kern = "kern" in self.font

        self._track_validation(
            "kern_redundant",
            not (has_gpos and has_kern),
            f"Status: GPOS={has_gpos}, kern={has_kern}, Compliant: {not (has_gpos and has_kern)}",
        )

        return self.validations

    def fix(self) -> bool:
        """Remove legacy kern table if needed."""
        if not self.validations.get("kern_redundant", {}).get("valid", True):
            if "GPOS" in self.font and "kern" in self.font:
                del self.font["kern"]
                self._track_change("has_kern", True, False, True)
                self._track_change("has_GPOS", True, True, False, info_only=True)
                return True
        return False


class NameTableHandler(TableHandler):
    """
    Handles name table cleanup.

    Responsibilities:
    - Keep only Windows English names
    - Remove specific problematic nameIDs
    """

    def get_table_name(self) -> str:
        return HANDLER_NAME.full_name

    def validate(self) -> Dict[str, Dict[str, Any]]:
        """Validate name table state."""
        if "name" not in self.font:
            self._track_validation("table_exists", False, "name table missing")
            return self.validations

        self.validate_condition(
            "only_windows_english",
            lambda: self._validate_windows_english_only(),
            "name table platform",
        )
        self.validate_condition(
            "no_problematic_ids",
            lambda: self._validate_problematic_ids(),
            "name table IDs",
        )

        return self.validations

    def _validate_windows_english_only(self) -> Tuple[bool, str]:
        """Helper: validate only Windows English names."""
        name_table = self.font["name"]
        non_windows_count = sum(
            1
            for rec in name_table.names
            if not (rec.platformID == 3 and rec.platEncID == 1 and rec.langID == 0x409)
        )
        is_valid = non_windows_count == 0
        return (
            is_valid,
            f"Current: {non_windows_count} non-Windows-English name(s), Expected: 0",
        )

    def _validate_problematic_ids(self) -> Tuple[bool, str]:
        """Helper: validate no problematic nameIDs."""
        name_table = self.font["name"]
        problematic_ids = {13, 14, 18, 19, 200, 201, 202, 203, 55555}
        problematic_count = sum(
            1 for rec in name_table.names if rec.nameID in problematic_ids
        )
        is_valid = problematic_count == 0
        return (
            is_valid,
            f"Current: {problematic_count} problematic nameID(s), Expected: 0",
        )

    def fix(self) -> bool:
        """Clean up name table."""
        any_changed = False

        if not self.validations.get("only_windows_english", {}).get("valid", True):
            any_changed |= self._keep_windows_english_only()

        if not self.validations.get("no_problematic_ids", {}).get("valid", True):
            any_changed |= self._delete_problematic_ids()

        return any_changed

    @conditional_fix("only_windows_english")
    def _keep_windows_english_only(self) -> bool:
        """Keep only Windows English names."""
        original_count = len(self.font["name"].names) if "name" in self.font else 0
        removed = keep_windows_english_only(self.font)

        if removed > 0:
            self._track_change(
                "total_names", original_count, original_count - removed, removed > 0
            )
            self._track_change("removed_names", 0, removed, removed > 0)
            return True

        return False

    @conditional_fix("no_problematic_ids")
    def _delete_problematic_ids(self) -> bool:
        """Delete problematic nameIDs."""
        ids_to_remove = {13, 14, 18, 19, 200, 201, 202, 203, 55555}
        specific_removed = delete_specific_nameids(self.font, ids_to_remove)

        if specific_removed > 0:
            self._track_change(
                "count_removed", 0, specific_removed, specific_removed > 0
            )
            self._track_change(
                "nameIDs_removed",
                "present",
                ", ".join(map(str, sorted(ids_to_remove))),
                specific_removed > 0,
                info_only=True,
            )
            return True

        return False


class FontFixer:
    """Orchestrates font validation and correction using table handlers."""

    def __init__(
        self,
        verbose: bool = False,
        enabled_handlers: Optional[list] = None,
        quarantine_enabled: bool = True,
    ):
        self.verbose = verbose
        # enabled_handlers is a list of handler names (HANDLER_OS2, etc.) to run, None means all
        self.enabled_handlers = enabled_handlers
        self.quarantine_enabled = quarantine_enabled

    def log(self, message: str):
        """Print verbose messages."""
        if self.verbose:
            cs.StatusIndicator("info").add_message(message).emit(console)

    def _validate_glyph_bounds(self, font: TTFont) -> list[str]:
        """
        Check if any glyph has bounding box coordinates outside signed 16-bit range.

        Args:
            font: The TTFont object to check

        Returns:
            List of glyph names with out-of-range bounds, empty if all valid
        """
        problematic_glyphs = []
        if "glyf" not in font:
            return problematic_glyphs

        try:
            glyf = font["glyf"]
            for glyph_name in font.getGlyphOrder():
                if glyph_name not in glyf:
                    continue

                glyph = glyf[glyph_name]
                # Check if glyph has bounding box data
                if hasattr(glyph, "xMin") and hasattr(glyph, "xMax"):
                    # Check each coordinate
                    if (
                        glyph.xMin < SIGNED_16BIT_MIN
                        or glyph.xMin > SIGNED_16BIT_MAX
                        or glyph.xMax < SIGNED_16BIT_MIN
                        or glyph.xMax > SIGNED_16BIT_MAX
                    ):
                        problematic_glyphs.append(glyph_name)
                        continue

                if hasattr(glyph, "yMin") and hasattr(glyph, "yMax"):
                    if (
                        glyph.yMin < SIGNED_16BIT_MIN
                        or glyph.yMin > SIGNED_16BIT_MAX
                        or glyph.yMax < SIGNED_16BIT_MIN
                        or glyph.yMax > SIGNED_16BIT_MAX
                    ):
                        if glyph_name not in problematic_glyphs:
                            problematic_glyphs.append(glyph_name)

        except Exception as e:
            # If validation fails, log but don't block processing
            if self.verbose:
                self.log(f"Error validating glyph bounds: {e}")

        return problematic_glyphs

    def _is_variable_font(self, font: Optional[Any]) -> bool:
        """
        Detect if a font is a variable font by checking for variable font tables.

        Variable font tables:
        - fvar: Required for variable fonts (font variations)
        - gvar: Glyph variations (TrueType outlines)
        - cvar: CVT variations (TrueType hinting)
        - avar: Axis variations (axis mapping)
        - HVAR: Horizontal metrics variations
        - VVAR: Vertical metrics variations
        - MVAR: Metrics variations

        Note: STAT table is not variable-specific - newer static fonts may contain it.

        Args:
            font: TTFont object or None

        Returns:
            True if font is variable (has fvar table, or other var tables indicating variable font), False otherwise
        """
        if font is None:
            return False
        try:
            # fvar is the definitive indicator of a variable font
            if "fvar" in font:
                return True

            # Other variable font tables (typically exist alongside fvar)
            # If they exist without fvar, the font is problematic but still variable-like
            var_tables = ["gvar", "cvar", "avar", "HVAR", "VVAR", "MVAR"]
            return any(table in font for table in var_tables)
        except Exception:
            return False

    def _is_corruption_type(
        self,
        exc_type: type,
        error_str: str,
        exc: Optional[Exception] = None,
        context: str = "",
        font: Optional[Any] = None,
    ) -> bool:
        """
        Determine if exception type/message indicates corruption.

        Enhanced to inspect tracebacks and consider context (especially save operations
        for variable fonts).

        Args:
            exc_type: Exception type
            error_str: Error message string
            exc: Optional exception object for traceback inspection
            context: Context string (e.g., "while saving", "during font loading")
            font: Optional TTFont object to check if variable font

        Returns:
            True if exception indicates font corruption, False otherwise
        """
        corruption_patterns = {
            AssertionError: ["gvar", "TupleVariation", "table", "glyph", "font"],
            TTLibError: ["*"],  # All TTLibError = corruption
            IndexError: ["table", "glyph"],
            ValueError: ["table", "struct", "format h", "xmax", "xmin", "ymax", "ymin"],
            struct.error: ["*"],
            AttributeError: ["table", "font", "glyph", "OS/2", "cmap"],
        }

        patterns = corruption_patterns.get(exc_type, [])
        if "*" in patterns:
            return True

        # Check error message for keywords
        error_lower = error_str.lower()
        if any(keyword in error_lower for keyword in patterns):
            return True

        # If error message is generic/empty, inspect traceback
        if exc is not None and (
            not error_str or error_str == "No error message provided"
        ):
            try:
                tb_str = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                tb_lower = tb_str.lower()

                # Check traceback for font-related keywords
                font_keywords = [
                    "gvar",
                    "fvar",
                    "tuplevariation",
                    "table",
                    "glyph",
                    "font",
                    "ttfont",
                    "ttlib",
                    "cmap",
                    "os/2",
                    "hmtx",
                    "vmtx",
                ]
                if any(keyword in tb_lower for keyword in font_keywords):
                    return True
            except Exception:
                pass

        # Special handling for save-time errors
        if "saving" in context.lower() or "save" in context.lower():
            # For variable fonts, any error during save is likely corruption
            if font is not None and self._is_variable_font(font):
                # Variable fonts are more sensitive - treat save errors as corruption
                if exc_type in (IndexError, AssertionError, AttributeError, ValueError):
                    return True

            # For any font, if error occurs during save and matches suspicious types
            # and traceback shows font table operations, treat as corruption
            if exc is not None and exc_type in (
                IndexError,
                AssertionError,
                AttributeError,
            ):
                try:
                    tb_str = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                    tb_lower = tb_str.lower()
                    # Check if traceback shows font table operations
                    if any(
                        keyword in tb_lower
                        for keyword in [
                            "ttfont",
                            "ttlib",
                            "table",
                            "compile",
                            "getTableData",
                        ]
                    ):
                        return True
                except Exception:
                    pass

        return False

    def _format_bounds_overflow_error(self, error_str: str) -> Optional[str]:
        """
        Format bounding box overflow errors with clearer explanation.

        Args:
            error_str: The original error message

        Returns:
            Formatted error message if this is a bounds overflow error, None otherwise
        """
        error_lower = error_str.lower()
        # Check if this is a bounding box overflow error
        if "format h" in error_lower and any(
            coord in error_lower for coord in ["xmax", "xmin", "ymax", "ymin"]
        ):
            # Try to extract the value and coordinate name
            # Pattern: "Value 33416 does not fit in format h for xMax"
            match = re.search(
                r"Value\s+(\d+)\s+does\s+not\s+fit\s+in\s+format\s+h\s+for\s+(\w+)",
                error_str,
                re.IGNORECASE,
            )
            if match:
                value = match.group(1)
                coord = match.group(2)
                return (
                    f"Glyph bounding box overflow (requires glyph coordinate modification to fix): "
                    f"Value {value} does not fit in signed 16-bit range for {coord}. "
                    f"This font needs glyph-level coordinate adjustment, which is beyond this script's scope."
                )
            else:
                # Fallback if regex doesn't match
                return (
                    f"Glyph bounding box overflow (requires glyph coordinate modification to fix): {error_str}. "
                    f"This font needs glyph-level coordinate adjustment, which is beyond this script's scope."
                )
        return None

    @contextmanager
    def corruption_handler(
        self,
        font_path: Path,
        result: FontFixResult,
        context: str = "",
        quarantine_dir: Optional[Path] = None,
        input_root: Optional[Path] = None,
        font: Optional[Any] = None,
    ):
        """
        Context manager for detecting and handling font corruption.

        If a corruption-related exception occurs:
        - Adds formatted error to result
        - Quarantines file if enabled
        - SWALLOWS the exception (does not re-raise)
        - Sets result.success = False

        If a non-corruption exception occurs:
        - Re-raises the exception for caller to handle

        Args:
            font_path: Path to the font file
            result: FontFixResult to update
            context: Context string describing where error occurred
            quarantine_dir: Quarantine directory path
            input_root: Root input directory for relative path calculation
            font: Optional TTFont object for variable font detection

        Usage:
            with self.corruption_handler(font_path, result, "during saving", ..., font=font):
                font.save(output_path)
                result.success = True  # Only reached if no exception
                result.was_modified = True

            # Check result.success after block - will be False if corrupted
        """
        try:
            yield
        except Exception as e:
            error_str = str(e) if str(e) else "No error message provided"

            # Check if this is corruption (pass exception object and context for enhanced detection)
            is_corrupt = self._is_corruption_type(
                type(e), error_str, exc=e, context=context, font=font
            )

            if is_corrupt:
                # Check if this is a bounding box overflow error that needs special formatting
                bounds_error = self._format_bounds_overflow_error(error_str)
                if bounds_error:
                    # Use the formatted bounds error message
                    context_msg = f" {context}" if context else ""
                    error_msg = f"Font corruption detected{context_msg}: {bounds_error}"
                else:
                    # Build standard error message
                    context_msg = f" {context}" if context else ""
                    error_msg = f"Font corruption detected{context_msg}: {type(e).__name__}: {error_str}"

                # Quarantine if enabled
                if self.quarantine_enabled and input_root and quarantine_dir:
                    qpath = self._quarantine_font(font_path, quarantine_dir, input_root)
                    if qpath:
                        result.quarantined = True
                        result.quarantine_path = str(qpath)
                        error_msg += f" (quarantined to {qpath})"
                    elif self.verbose:
                        # Log quarantine failure for debugging
                        self.log(
                            f"Warning: Failed to quarantine {font_path} (quarantine_dir={quarantine_dir}, input_root={input_root})"
                        )

                result.add_error(error_msg, include_traceback=self.verbose)
                result.success = False

                # Don't re-raise - corruption is handled
            else:
                # Not corruption - re-raise for caller to handle
                raise

    def _build_fix_details_from_changes(self, changes: dict) -> list:
        """Build detailed fix information from changes dictionary."""
        details_parts = []

        for prop_name, info in changes.items():
            old_val = info.get("old")
            new_val = info.get("new")
            changed = info.get("changed", False)
            info_only = info.get("info_only", False)

            # Skip info_only properties in non-verbose mode
            if info_only and not self.verbose:
                continue

            # Skip unchanged properties in non-verbose mode
            if not changed and not self.verbose:
                continue

            if changed:
                # Check if new_val already contains delta information in brackets
                if "[" in str(new_val) and "]" in str(new_val):
                    # Delta information already embedded, show as-is
                    details_parts.append(f"- {prop_name}: {old_val} -> {new_val}")
                else:
                    # Standard format
                    details_parts.append(f"- {prop_name}: {old_val} -> {new_val}")
            else:
                # Unchanged property, show in verbose mode with checkmark
                details_parts.append(
                    (f"- {prop_name}: {new_val} [success]✓ OK[/success]")
                )

        return details_parts

    def log_handler_result(
        self, handler_name: str, changed: bool, changes: Dict = None
    ):
        """Log the result of a handler execution."""
        if not changed:
            return  # Don't show unchanged handlers here - will show in consolidated message

        indicator = cs.StatusIndicator("updated", dry_run=False).add_message(
            handler_name
        )

        # Add each detail line as an indented item
        if changes:
            details = self._build_fix_details_from_changes(changes)
            for detail_line in details:
                indicator.add_item(detail_line)

        indicator.emit(console)

    def _is_corruption_error(self, error: Exception, error_str: str) -> bool:
        """Determine if an error indicates font corruption (wrapper for _is_corruption_type)."""
        return self._is_corruption_type(type(error), error_str)

    def _quarantine_font(
        self, font_path: Path, quarantine_dir: Optional[Path], input_root: Path
    ) -> Optional[Path]:
        """
        Move a corrupted font to quarantine directory.

        Args:
            font_path: Path to the font file to quarantine
            quarantine_dir: Root quarantine directory
            input_root: Root of input directory (for preserving relative paths)

        Returns:
            Path to quarantined file if successful, None otherwise
        """
        # Debug logging
        if self.verbose:
            self.log(
                f"Quarantine check: enabled={self.quarantine_enabled}, dir={quarantine_dir}, input_root={input_root}"
            )

        if not self.quarantine_enabled:
            if self.verbose:
                self.log(f"Quarantine disabled, skipping {font_path}")
            return None

        if quarantine_dir is None:
            if self.verbose:
                self.log(f"Quarantine directory is None, cannot quarantine {font_path}")
            return None

        try:
            # Calculate relative path from input root to preserve directory structure
            try:
                relative_path = font_path.relative_to(input_root)
                if self.verbose:
                    self.log(f"Calculated relative path: {relative_path}")
            except ValueError:
                # Font is not under input root, use just the filename
                relative_path = Path(font_path.name)
                if self.verbose:
                    self.log(
                        f"Font not under input_root, using filename: {relative_path}"
                    )

            # Build quarantine path
            quarantine_path = quarantine_dir / relative_path

            # Handle path collisions using tilde format
            if quarantine_path.exists():
                stem = quarantine_path.stem
                suffix = quarantine_path.suffix
                counter = 1
                while quarantine_path.exists():
                    quarantine_path = (
                        quarantine_path.parent / f"{stem}~{counter:03d}{suffix}"
                    )
                    counter += 1
                if self.verbose:
                    self.log(f"Quarantine path collision, using: {quarantine_path}")

            # Create parent directories
            quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                self.log(f"Created quarantine directory: {quarantine_path.parent}")

            # Move file to quarantine
            import shutil

            shutil.move(str(font_path), str(quarantine_path))

            if self.verbose:
                self.log(f"Successfully quarantined {font_path} to {quarantine_path}")

            return quarantine_path
        except Exception as e:
            if self.verbose:
                self.log(f"Failed to quarantine {font_path}: {e}")
                import traceback

                self.log(f"Quarantine error traceback: {traceback.format_exc()}")
            return None

    def fix_font(
        self,
        font_path: Path,
        output_dir: Optional[Path] = None,
        validate_only: bool = False,
        quarantine_dir: Optional[Path] = None,
        input_root: Optional[Path] = None,
    ) -> FontFixResult:
        """
        Apply all fixes to a single font file.

        Returns FontFixResult with status and details.
        """
        result = FontFixResult(file=str(font_path))

        font = None
        try:
            # Open font once
            font = TTFont(font_path, recalcBBoxes=True, recalcTimestamp=False)
            original_flavor = font.flavor  # Store original flavor

            # Validate glyph bounding boxes are within signed 16-bit range
            # This catches issues early before processing handlers
            if "glyf" in font:
                problematic_glyphs = self._validate_glyph_bounds(font)
                if problematic_glyphs:
                    # Close font and reopen without recalcBBoxes to preserve original bounds
                    font.close()
                    if self.verbose:
                        self.log(
                            f"Detected {len(problematic_glyphs)} glyph(s) with out-of-range bounds, "
                            f"reopening without recalcBBoxes: {', '.join(problematic_glyphs[:5])}"
                            + (
                                f" and {len(problematic_glyphs) - 5} more"
                                if len(problematic_glyphs) > 5
                                else ""
                            )
                        )
                    font = TTFont(font_path, recalcBBoxes=False, recalcTimestamp=False)
                    font.flavor = original_flavor

            # Create all handlers in dependency order
            handlers = [
                OS2TableHandler(font, self.verbose),
                StyleConsistencyHandler(font, self.verbose),
                GlyphHandler(font, self.verbose),
                KernHandler(font, self.verbose),
                NameTableHandler(font, self.verbose),
            ]

            # Filter handlers if specific ones are enabled
            if self.enabled_handlers:
                handlers = [
                    h for h in handlers if h.get_table_name() in self.enabled_handlers
                ]

            any_changed = False

            # Run each handler
            for handler in handlers:
                try:
                    handler_name = handler.get_table_name()
                    result.handlers_run.append(handler_name)

                    # Validate
                    handler_validations = handler.validate()
                    result.validations[handler_name] = handler_validations

                    # Fix (skip if validate_only)
                    if validate_only:
                        handler_changed = False
                    else:
                        handler_changed = handler.fix()
                    if handler_changed:
                        any_changed = True
                        handler_changes = handler.get_changes()
                        result.changes[handler_name] = handler_changes
                        result.mark_handler_run(handler_name, True)

                        # Log handler result
                        self.log_handler_result(handler_name, True, handler_changes)
                    else:
                        result.mark_handler_run(handler_name, False)

                except Exception as e:
                    handler_name = handler.get_table_name()
                    result.add_exception(
                        e,
                        f"Handler '{handler_name}' failed",
                        include_traceback=self.verbose,
                    )

            # Check if any handler detected fatal corruption during validation
            has_fatal_corruption = False
            for handler_name, validations in result.validations.items():
                for check_name, validation in validations.items():
                    if check_name in ["table_readable"] and not validation.get(
                        "valid", True
                    ):
                        has_fatal_corruption = True
                        break
                if has_fatal_corruption:
                    break

            if has_fatal_corruption:
                # Quarantine immediately without attempting save
                result.add_error("Font has corrupted tables detected during validation")
                result.success = False
                result.was_modified = False

                if input_root and quarantine_dir:
                    quarantine_path = self._quarantine_font(
                        font_path, quarantine_dir, input_root
                    )
                    if quarantine_path:
                        result.quarantined = True
                        result.quarantine_path = str(quarantine_path)

                # Skip attempting to save this font
                any_changed = False

            # Only save if fixes were actually applied
            if any_changed:
                # Determine output path
                if output_dir:
                    output_path = output_dir / font_path.name
                    # Handle path collisions using tilde format
                    if output_path.exists() and output_path != font_path:
                        stem = output_path.stem
                        suffix = output_path.suffix
                        counter = 1
                        while output_path.exists():
                            output_path = output_dir / f"{stem}~{counter:03d}{suffix}"
                            counter += 1
                    # Create parent directories if needed
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = font_path

                # Restore flavor before saving
                font.flavor = original_flavor

                # Detect if this is a variable font for enhanced error detection
                is_variable = self._is_variable_font(font)
                if is_variable and self.verbose:
                    self.log(
                        "Detected variable font, using enhanced corruption detection"
                    )

                # Attempt to save with improved error handling
                with self.corruption_handler(
                    font_path,
                    result,
                    "while saving",
                    quarantine_dir,
                    input_root,
                    font=font,
                ):
                    font.save(str(output_path))
                    result.success = True
                    result.was_modified = True
                    result.output_path = str(output_path)
            else:
                result.success = True
                result.was_modified = False

        except TTLibError as e:
            # Font loading or table access error
            error_str = str(e) if str(e) else "No error message provided"
            context = "during font loading"

            if self._is_corruption_type(
                TTLibError, error_str, exc=e, context=context, font=font
            ):
                # Handle corruption directly
                if self.quarantine_enabled and input_root and quarantine_dir:
                    qpath = self._quarantine_font(font_path, quarantine_dir, input_root)
                    if qpath:
                        result.quarantined = True
                        result.quarantine_path = str(qpath)
                    elif self.verbose:
                        self.log(
                            f"Warning: Failed to quarantine {font_path} (TTLibError)"
                        )
                result.add_error(
                    f"Font corruption detected {context}: {type(e).__name__}: {error_str}",
                    include_traceback=self.verbose,
                )
            else:
                result.add_error(
                    f"Font table error {context}: {type(e).__name__}: {error_str}",
                    include_traceback=self.verbose,
                )
            result.success = False
        except Exception as e:
            # Build error message with context
            error_type = type(e).__name__
            error_str = str(e) if str(e) else "No error message provided"

            # Determine context - where did the error occur?
            try:
                if "any_changed" in locals() and any_changed:
                    if "output_path" in locals():
                        context = f"while saving to {output_path.name}"
                    else:
                        context = "while saving font"
                elif font is not None:
                    context = "during font processing"
                else:
                    context = "during font loading"
            except (NameError, KeyError):
                # Fallback if variable checks fail
                context = "during font processing"

            # Check if corruption-related (pass exception object and context for enhanced detection)
            if self._is_corruption_type(
                type(e), error_str, exc=e, context=context, font=font
            ):
                # Handle corruption directly
                if self.quarantine_enabled and input_root and quarantine_dir:
                    qpath = self._quarantine_font(font_path, quarantine_dir, input_root)
                    if qpath:
                        result.quarantined = True
                        result.quarantine_path = str(qpath)
                    elif self.verbose:
                        self.log(
                            f"Warning: Failed to quarantine {font_path} (Exception: {error_type})"
                        )
                result.add_error(
                    f"Font corruption detected {context}: {error_type}: {error_str}",
                    include_traceback=(isinstance(e, AssertionError) or self.verbose),
                )
            else:
                result.add_error(
                    f"Fatal error {context}: {error_type}: {error_str}",
                    include_traceback=(isinstance(e, AssertionError) or self.verbose),
                )
            result.success = False
        finally:
            # Ensure font is always closed to prevent resource leaks
            if font is not None:
                try:
                    font.close()
                except Exception:
                    # Ignore errors during close - font may already be closed or corrupted
                    pass

        return result


# ============================================================================
# PROCESSING CONFIGURATION
# ============================================================================


class ProcessingConfig(NamedTuple):
    """Configuration for font processing run."""

    input_path: Path
    output_dir: Optional[Path]
    recursive: bool
    num_workers: int
    verbose: bool
    dry_run: bool
    validate_only: bool
    enabled_handlers: Optional[list[str]]
    quarantine_enabled: bool
    quarantine_dir: Optional[Path]
    input_root: Path


# ============================================================================
# ARGUMENT PARSING
# ============================================================================


def _parse_handler_selection(args) -> Optional[list[str]]:
    """Parse --handlers and --skip-handlers arguments."""
    if args.handlers and args.skip_handlers:
        cs.StatusIndicator("error").add_message(
            "--handlers and --skip-handlers cannot be used together"
        ).emit(console)
        sys.exit(1)

    if args.handlers:
        return _parse_enabled_handlers(args.handlers)

    if args.skip_handlers:
        return _parse_skipped_handlers(args.skip_handlers)

    return None  # All handlers enabled


def _parse_enabled_handlers(handlers_str: str) -> list[str]:
    """Parse comma-separated handler list."""
    handler_list = [h.strip().lower() for h in handlers_str.split(",")]
    invalid = [h for h in handler_list if h not in HandlerSpec.all_short_names()]

    if invalid:
        cs.StatusIndicator("error").add_message(
            f"Invalid handler(s): {', '.join(invalid)}. "
            f"Available: {', '.join(HandlerSpec.all_short_names())}"
        ).emit(console)
        sys.exit(1)

    return [HandlerSpec.get(h).full_name for h in handler_list]


def _parse_skipped_handlers(handlers_str: str) -> list[str]:
    """Parse comma-separated skip list."""
    skip_list = [h.strip().lower() for h in handlers_str.split(",")]
    invalid = [h for h in skip_list if h not in HandlerSpec.all_short_names()]

    if invalid:
        cs.StatusIndicator("error").add_message(
            f"Invalid handler(s): {', '.join(invalid)}. "
            f"Available: {', '.join(HandlerSpec.all_short_names())}"
        ).emit(console)
        sys.exit(1)

    all_handlers_set = set(ALL_HANDLERS)
    skip_handlers_set = {HandlerSpec.get(h).full_name for h in skip_list}
    enabled = list(all_handlers_set - skip_handlers_set)

    if not enabled:
        cs.StatusIndicator("error").add_message("Cannot skip all handlers").emit(
            console
        )
        sys.exit(1)

    return enabled


def parse_and_validate_arguments() -> ProcessingConfig:
    """Parse command-line arguments and validate configuration."""
    parser = argparse.ArgumentParser(
        description="Apply all font fixes in a single pass using only fonttools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Process all fonts in a directory (non-recursive):
    %(prog)s fonts/

  Process recursively with 8 parallel workers:
    %(prog)s -r -j 8 fonts/

  Save fixed fonts to a different directory:
    %(prog)s -o output/ fonts/

  Only run specific handlers (comma-separated):
    %(prog)s --handlers os2,style fonts/

  Skip specific handlers:
    %(prog)s --skip-handlers name fonts/

  Preview what would be changed without modifying files:
    %(prog)s --validate-only -V fonts/MyFont.ttf

AVAILABLE HANDLERS:
  Handler      Description
  -----------  ----------------------------------------------------
  os2          OS/2 table version, embedding permissions, monospace
               detection, USE_TYPO_METRICS and WWS flags

  style        Style consistency across post, hhea, OS/2, and head
               tables (italic angle, caret slope, fsSelection, macStyle)

  glyph        Glyph-level fixes: .notdef structure, nbsp (U+00A0)
               presence and width matching space character

  kern         Legacy kern table removal when modern GPOS table exists

  name         Name table cleanup: Windows English records only,
               removal of problematic nameIDs (13,14,18,19,200-203,55555)

DEPENDENCIES:
  This tool requires only fonttools:
    pip install fonttools

For more information, see: https://github.com/fonttools/fonttools
        """,
    )

    parser.add_argument(
        "input_path", type=Path, help="Font file or directory containing font files"
    )

    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Process directories recursively"
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory (default: overwrite originals)",
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, use 0 for CPU count)",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes",
    )

    parser.add_argument(
        "--handlers",
        type=str,
        help="Comma-separated list of handlers to run. Available: os2, style, glyph, kern, name. Default: all",
    )

    parser.add_argument(
        "--skip-handlers",
        type=str,
        help="Comma-separated list of handlers to skip. Available: os2, style, glyph, kern, name",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate fonts and report issues without applying any fixes",
    )

    parser.add_argument(
        "--no-quarantine",
        action="store_true",
        help="Disable automatic quarantine of corrupted fonts (quarantine enabled by default)",
    )

    args = parser.parse_args()

    # Validate handler selection
    enabled_handlers = _parse_handler_selection(args)

    # Determine paths
    input_root = args.input_path if args.input_path.is_dir() else args.input_path.parent
    quarantine_enabled = not args.no_quarantine
    quarantine_dir = input_root / "_quarantine" if quarantine_enabled else None

    # Determine workers
    num_workers = mp.cpu_count() if args.jobs == 0 else args.jobs

    # Create output directory if needed
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    return ProcessingConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        recursive=args.recursive,
        num_workers=num_workers,
        verbose=args.verbose,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        enabled_handlers=enabled_handlers,
        quarantine_enabled=quarantine_enabled,
        quarantine_dir=quarantine_dir,
        input_root=input_root,
    )


# ============================================================================
# MULTIPROCESSING WRAPPER
# ============================================================================


def process_font_wrapper(
    args: Tuple[
        Path,
        Optional[Path],
        bool,
        Optional[list],
        bool,
        Optional[Path],
        Optional[Path],
        bool,
    ],
) -> Dict:
    """Wrapper for multiprocessing."""
    (
        font_path,
        output_dir,
        verbose,
        enabled_handlers,
        validate_only,
        quarantine_dir,
        input_root,
        quarantine_enabled,
    ) = args
    fixer = FontFixer(
        verbose=verbose,
        enabled_handlers=enabled_handlers,
        quarantine_enabled=quarantine_enabled,
    )
    result = fixer.fix_font(
        font_path, output_dir, validate_only, quarantine_dir, input_root
    )
    return result.to_dict()  # Convert to dict for multiprocessing compatibility


def discover_and_validate_fonts(config: ProcessingConfig) -> list[Path]:
    """Find all font files to process."""
    font_files = collect_font_files([config.input_path], config.recursive)
    font_paths = [Path(f) for f in font_files]

    if not font_paths:
        cs.StatusIndicator("error").add_message("No font files found.").emit(console)
        sys.exit(0)

    return font_paths


def display_preflight_info(config: ProcessingConfig, font_paths: list[Path]):
    """Display header, checklist, and file list."""
    cs.fmt_header("Font Fixer (fonttools)", console=console)
    cs.emit("")

    # Show which handlers will run
    if config.enabled_handlers is None:
        handler_display = HandlerSpec.all_short_names()
    else:
        handler_display = [
            spec.short_name
            for h in config.enabled_handlers
            if (spec := _get_handler_spec_by_full_name(h)) is not None
        ]

    # Build operations list with handler descriptions
    operations = []
    for short_name in handler_display:
        spec = HandlerSpec.get(short_name)
        description = spec.description if spec else short_name
        operations.append(f"{short_name}: {description}")

    cs.fmt_preflight_checklist("Font Fixer", operations, console=console)

    # Show file list
    cs.emit("")
    cs.StatusIndicator("info").add_message(
        f"Found {cs.fmt_count(len(font_paths))} font file(s) to process:"
    ).emit(console)
    for path in font_paths:
        cs.emit(f"  - {cs.fmt_file_compact(str(path))}", console=console)

    if config.dry_run:
        # DRY prefix will be added automatically by StatusIndicator when dry_run=True
        # This script exits early in dry-run mode, so no processing messages are shown
        sys.exit(0)


def main():
    # Parse and validate arguments
    config = parse_and_validate_arguments()

    # Discover fonts
    font_paths = discover_and_validate_fonts(config)

    # Display preflight info
    display_preflight_info(config, font_paths)

    cs.emit("")
    cs.StatusIndicator("info").add_message(
        f"Processing with {cs.fmt_count(config.num_workers)} worker(s)..."
    ).emit(console)
    cs.emit("")

    # Process fonts
    results = process_all_fonts(config, font_paths)

    # Display summary
    display_summary(config, results)


def _display_handler_list(config: ProcessingConfig):
    """Display which handlers will run."""
    if config.enabled_handlers:
        handler_display = [
            spec.short_name
            for h in config.enabled_handlers
            if (spec := _get_handler_spec_by_full_name(h)) is not None
        ]
    else:
        handler_display = HandlerSpec.all_short_names()

    cs.StatusIndicator("info").add_message(
        f"Running handlers: {', '.join(handler_display)}"
    ).emit(console)


def _display_result(result: Dict, font_path: Path):
    """Display result for a single font."""
    if result["success"]:
        # Show NO CHANGE message for unchanged handlers
        if result.get("handlers_unchanged"):
            unchanged_display = [
                spec.short_name
                for h in result["handlers_unchanged"]
                if (spec := _get_handler_spec_by_full_name(h)) is not None
            ]
            cs.StatusIndicator("unchanged").add_message(
                f"Already compliant: {', '.join(unchanged_display)}"
            ).emit(console)

        # Show SAVED status if file was modified
        if result.get("was_modified") and result.get("output_path"):
            cs.StatusIndicator("saved").add_file(
                result["output_path"], filename_only=True
            ).emit(console)
    else:
        # Show quarantine status if file was quarantined
        if result.get("quarantined"):
            cs.StatusIndicator("error").add_file(
                str(font_path), filename_only=True
            ).with_explanation(
                f"Quarantined: {result.get('quarantine_path', 'unknown')}; "
                + "; ".join(result["errors"])
            ).emit(console)
        else:
            cs.StatusIndicator("error").add_file(
                str(font_path), filename_only=True
            ).with_explanation("; ".join(result["errors"])).emit(console)


def process_all_fonts(config: ProcessingConfig, font_paths: list[Path]) -> list[Dict]:
    """Process all fonts (sequential or parallel)."""
    if config.num_workers == 1:
        return _process_sequential(config, font_paths)
    else:
        return _process_parallel(config, font_paths)


def _process_sequential(config: ProcessingConfig, font_paths: list[Path]) -> list[Dict]:
    """Process fonts one at a time."""
    results = []

    for idx, font_path in enumerate(font_paths, 1):
        cs.StatusIndicator("parsing").add_message(
            f"File {idx} of {len(font_paths)} |"
        ).add_file(str(font_path), filename_only=True).emit(console)

        _display_handler_list(config)

        fixer = FontFixer(
            verbose=config.verbose,
            enabled_handlers=config.enabled_handlers,
            quarantine_enabled=config.quarantine_enabled,
        )

        result = fixer.fix_font(
            font_path,
            config.output_dir,
            config.validate_only,
            config.quarantine_dir,
            config.input_root,
        )

        result_dict = result.to_dict()  # Convert for display and storage
        results.append(result_dict)
        _display_result(result_dict, font_path)

    return results


def _process_parallel(config: ProcessingConfig, font_paths: list[Path]) -> list[Dict]:
    """Process fonts in parallel."""
    work_items = [
        (
            font_path,
            config.output_dir,
            config.verbose,
            config.enabled_handlers,
            config.validate_only,
            config.quarantine_dir,
            config.input_root,
            config.quarantine_enabled,
        )
        for font_path in font_paths
    ]

    results = []

    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {
            executor.submit(process_font_wrapper, item): item[0] for item in work_items
        }

        completed = 0
        for future in as_completed(futures):
            font_path = futures[future]
            try:
                result = future.result()
                results.append(result)

                completed += 1
                cs.StatusIndicator("parsing").add_message(
                    f"File {completed} of {len(font_paths)} |"
                ).add_file(str(font_path), filename_only=True).emit(console)

                _display_result(result, font_path)
            except Exception as e:
                error_msg = f"{str(font_path)}: {type(e).__name__}: {str(e)}"
                if config.verbose:
                    error_msg = f"{error_msg}\n{traceback.format_exc()}"
                cs.StatusIndicator("error").add_file(
                    str(font_path), filename_only=True
                ).with_explanation(error_msg).emit(console)
                results.append(
                    {
                        "file": str(font_path),
                        "success": False,
                        "errors": [error_msg],
                    }
                )

    return results


def display_summary(config: ProcessingConfig, results: list[Dict]):
    """Display processing summary."""
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    unchanged = sum(
        1 for r in results if r["success"] and not r.get("was_modified", False)
    )
    updated = sum(1 for r in results if r["success"] and r.get("was_modified", False))
    quarantined = sum(1 for r in results if r.get("quarantined", False))

    cs.fmt_processing_summary(
        dry_run=False,
        updated=updated,
        unchanged=unchanged,
        errors=failed,
        console=console,
    )

    # Show quarantined count if any files were quarantined
    if quarantined > 0 and config.quarantine_dir:
        cs.emit("", console=console)
        cs.StatusIndicator("info").add_message(
            f"Quarantined {cs.fmt_count(quarantined)} corrupted font file(s) to: {config.quarantine_dir}"
        ).emit(console)

    # Show most common fixes applied
    if updated > 0 and results:
        _display_common_fixes(results)

    # Show handler-level statistics
    if results:
        _display_handler_statistics(results)

    # Show failed files if any
    if failed > 0:
        _display_failed_files(results)


def _display_common_fixes(results: list[Dict]):
    """Display most common fixes applied."""
    common_fixes = {}
    for result in results:
        # Only count fixes from successfully processed and saved files
        # This ensures consistency with the "updated" count
        if not result.get("success", False) or not result.get("was_modified", False):
            continue

        for handler, changes in result.get("changes", {}).items():
            for prop in changes.keys():
                if changes[prop].get("changed"):
                    spec = _get_handler_spec_by_full_name(handler)
                    handler_short = spec.short_name if spec else handler
                    key = f"{handler_short}.{prop}"
                    common_fixes[key] = common_fixes.get(key, 0) + 1

    if common_fixes:
        top_fixes = sorted(common_fixes.items(), key=lambda x: x[1], reverse=True)[:5]
        cs.emit("", console=console)
        cs.StatusIndicator("info").add_message("Most common fixes applied:").emit(
            console
        )
        for fix, count in top_fixes:
            cs.emit(f"  · {fix}: {count} file(s)", console=console)


def _display_handler_statistics(results: list[Dict]):
    """Display handler-level statistics."""
    handler_changed_counts = {}
    handler_unchanged_counts = {}

    for result in results:
        # Only count handlers from successfully processed files
        # This ensures statistics match the "updated" count which only includes successful saves
        if not result.get("success", False):
            continue

        for handler in result.get("handlers_changed", []):
            spec = _get_handler_spec_by_full_name(handler)
            handler_display = spec.short_name if spec else handler
            handler_changed_counts[handler_display] = (
                handler_changed_counts.get(handler_display, 0) + 1
            )

        for handler in result.get("handlers_unchanged", []):
            spec = _get_handler_spec_by_full_name(handler)
            handler_display = spec.short_name if spec else handler
            handler_unchanged_counts[handler_display] = (
                handler_unchanged_counts.get(handler_display, 0) + 1
            )

    if handler_changed_counts or handler_unchanged_counts:
        cs.emit("")

        all_handlers = set(handler_changed_counts.keys()) | set(
            handler_unchanged_counts.keys()
        )
        total_updates = sum(handler_changed_counts.values())
        total_stable = sum(handler_unchanged_counts.values())

        indicator = cs.StatusIndicator("success").add_message("Handler Statistics")
        indicator.add_item(
            f"Handlers evaluated: {cs.fmt_count(len(all_handlers))} | "
            f"Made changes: {cs.fmt_count(total_updates)} | "
            f"No changes: {cs.fmt_count(total_stable)}"
        )

        for handler_name in sorted(all_handlers):
            changed = handler_changed_counts.get(handler_name, 0)
            unchanged = handler_unchanged_counts.get(handler_name, 0)

            if changed and unchanged:
                detail = f"{handler_name}: {cs.fmt_count(changed)} updated, {cs.fmt_count(unchanged)} unchanged"
            elif changed:
                detail = f"{handler_name}: {cs.fmt_count(changed)} updated"
            else:
                detail = f"{handler_name}: {cs.fmt_count(unchanged)} unchanged"

            indicator.add_item(detail, indent_level=2)

        indicator.emit(console)


def _display_failed_files(results: list[Dict]):
    """Display list of failed files."""
    cs.emit("")
    cs.StatusIndicator("error").add_message("Failed files:").emit(console)
    for result in results:
        if not result["success"]:
            cs.StatusIndicator("error").add_file(
                str(result["file"]), filename_only=False
            ).with_explanation("; ".join(result["errors"])).emit(console)

    # Exit with appropriate code
    failed = sum(1 for r in results if not r["success"])
    cs.emit("")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
