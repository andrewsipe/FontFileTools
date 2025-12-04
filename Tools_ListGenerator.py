#!/usr/bin/env python3
"""
Font Filename List Generator

Recursively scans a directory for font files and generates a simple text list
of filenames (without extensions) for analysis purposes.

Supported formats: TTF, OTF, WOFF, WOFF2
Excludes: TTX files

Usage:
    python FontFilename_ListGenerator.py [directory_path]

    If no directory is provided, the script will prompt for input.
    The output file 'filename_list.txt' will be created in the input directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Set

import core.core_console_styles as cs
from core.core_file_collector import iter_font_files

# Font extensions to include (excluding TTX as specified)
FONT_EXTENSIONS: Set[str] = {".ttf", ".otf", ".woff", ".woff2"}

# Output filename
OUTPUT_FILENAME = "filename_list.txt"


def get_input_directory() -> Path:
    """Get directory path from command line or prompt user."""
    if len(sys.argv) > 1:
        # Directory provided as command line argument
        input_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        # Prompt user for directory
        cs.StatusIndicator("info").add_message("Font Filename List Generator").emit()
        cs.StatusIndicator("info").add_message(
            "This script will scan a directory recursively for font files"
        ).emit()
        cs.StatusIndicator("info").add_message(
            "and create a simple text list of filenames (without extensions)."
        ).emit()
        cs.emit("")

        while True:
            directory_input = cs.prompt_input(
                "Enter directory path (or drag & drop folder): "
            ).strip()

            if not directory_input:
                cs.StatusIndicator("warning").with_explanation(
                    "Please enter a directory path"
                ).emit()
                continue

            # Remove quotes if present (common with drag & drop)
            directory_input = directory_input.strip("\"'")
            input_path = Path(directory_input).expanduser().resolve()

            if not input_path.exists():
                cs.StatusIndicator("error").add_file(str(input_path)).with_explanation(
                    "Directory does not exist"
                ).emit()
                continue

            if not input_path.is_dir():
                cs.StatusIndicator("error").add_file(str(input_path)).with_explanation(
                    "Path is not a directory"
                ).emit()
                continue

            break

    return input_path


def collect_font_filenames_with_progress(directory: Path) -> List[str]:
    """Collect font files from directory with progress indication and return list of filenames without extensions."""
    cs.StatusIndicator("info").add_message(
        f"Scanning directory: {cs.fmt_file(str(directory), filename_only=False)}"
    ).emit()

    # Create progress bar for discovery phase
    console = cs.get_console()
    progress = cs.create_progress_bar(console=console)

    filenames = []

    def on_progress(progress_data: dict):
        """Update progress bar with discovery statistics"""
        files_scanned = progress_data.get("files_scanned", 0)
        matches_found = progress_data.get("matches_found", 0)

        # Update task description with current stats
        progress.update(
            task,
            description=f"Scanning directories... {cs.fmt_count(matches_found)} fonts found from {cs.fmt_count(files_scanned)} files",
            completed=files_scanned,
        )

    with progress:
        # Create indeterminate task for discovery phase
        task = progress.add_task(
            "Discovering font files...",
            total=None,  # Indeterminate
        )

        # Use the new progress-enabled iterator
        font_files = iter_font_files(
            paths=[directory],
            recursive=True,
            allowed_extensions=FONT_EXTENSIONS,
            include_uppercase=True,
            on_progress=on_progress,
        )

        # Collect filenames as we iterate
        for font_path in font_files:
            filename = Path(font_path).stem  # Gets filename without extension
            filenames.append(filename)

    # Sort for consistent output
    filenames.sort()

    cs.StatusIndicator("info").add_message(
        f"Found {cs.fmt_count(len(filenames))} font files"
    ).emit()

    return filenames


def write_filename_list_with_progress(filenames: List[str], output_dir: Path) -> Path:
    """Write filenames to text file with progress indication."""
    output_path = output_dir / OUTPUT_FILENAME
    temp_path = output_dir / f".{OUTPUT_FILENAME}.tmp"

    cs.StatusIndicator("info").add_message(
        f"Writing filename list to: {cs.fmt_file(str(output_path), filename_only=False)}"
    ).emit()

    # Create progress bar for writing phase
    console = cs.get_console()
    progress = cs.create_progress_bar(console=console)

    try:
        with progress:
            # Create determinate task for writing phase
            task = progress.add_task("Writing filenames...", total=len(filenames))

            # Write to temporary file first (atomic operation)
            with open(temp_path, "w", encoding="utf-8") as f:
                for i, filename in enumerate(filenames):
                    f.write(f"{filename}\n")
                    # Update progress every 100 files or at the end
                    if i % 100 == 0 or i == len(filenames) - 1:
                        progress.update(
                            task,
                            description=f"Writing filenames... {cs.fmt_count(i + 1)}/{cs.fmt_count(len(filenames))}",
                            completed=i + 1,
                        )

            # Atomic rename to final destination
            temp_path.rename(output_path)

        cs.StatusIndicator("saved").add_file(str(output_path)).add_field(
            "filenames", len(filenames)
        ).emit()
        return output_path

    except PermissionError:
        cs.StatusIndicator("error").add_file(str(output_path)).with_explanation(
            "Permission denied: Cannot write to file"
        ).emit()
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise
    except Exception as e:
        cs.StatusIndicator("error").with_explanation(f"Error writing file: {e}").emit()
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise


def main():
    """Main script execution."""
    try:
        # Get input directory
        input_directory = get_input_directory()

        # Collect font filenames with progress indication
        filenames = collect_font_filenames_with_progress(input_directory)

        if not filenames:
            cs.StatusIndicator("warning").with_explanation(
                "No font files found in directory"
            ).emit()
            cs.StatusIndicator("info").add_message(
                f"Supported formats: {', '.join(sorted(FONT_EXTENSIONS))}"
            ).emit()
            return

        # Write output file with progress indication
        output_path = write_filename_list_with_progress(filenames, input_directory)

        # Show summary
        cs.StatusIndicator("saved").add_message(
            "Filename list generation completed!"
        ).emit()
        cs.StatusIndicator("info").add_file(str(output_path)).add_message(
            "Output file"
        ).emit()
        cs.StatusIndicator("info").add_field(
            "Total filenames", cs.fmt_count(len(filenames))
        ).emit()

        # Show preview of first few filenames
        if len(filenames) > 0:
            cs.StatusIndicator("info").add_message("Preview of filenames:").emit()
            preview_count = min(5, len(filenames))
            for i, filename in enumerate(filenames[:preview_count], 1):
                cs.emit(f"  {cs.fmt_count(i)}. {cs.fmt_value(filename)}")

            if len(filenames) > preview_count:
                cs.emit(
                    f"  ... and {cs.fmt_count(len(filenames) - preview_count)} more"
                )

    except KeyboardInterrupt:
        cs.StatusIndicator("info").add_message("Operation cancelled by user").emit()
        # Clean up any temporary files
        temp_path = input_directory / f".{OUTPUT_FILENAME}.tmp"
        if temp_path.exists():
            temp_path.unlink()
    except Exception as e:
        cs.StatusIndicator("error").with_explanation(f"Script error: {e}").emit()
        # Clean up any temporary files
        temp_path = input_directory / f".{OUTPUT_FILENAME}.tmp"
        if temp_path.exists():
            temp_path.unlink()
        sys.exit(1)


if __name__ == "__main__":
    main()
