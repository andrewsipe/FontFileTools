#!/usr/bin/env python3
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.misc.transform import Transform
import sys
import os
import unicodedata
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text
from rich.prompt import Prompt, Confirm
import glob
from typing import List, Tuple

# Initialize Rich console
console = Console()


def get_unicode_info(char_or_code):
    """
    Get Unicode information for a character or code point.

    Parameters:
    - char_or_code: A character (e.g., '('), a Unicode code point as int (e.g., 40),
                   or a string representation of a code point (e.g., "U+0028", "0x28", "40")

    Returns:
    - (code_point, character, name) tuple, or (code_point, None, None) if input is just a code point
    """
    if isinstance(char_or_code, int):
        # If input is already an integer code point
        try:
            char = chr(char_or_code)
            name = unicodedata.name(char, f"U+{char_or_code:04X}")
            return char_or_code, char, name
        except (ValueError, TypeError):
            return char_or_code, None, None

    elif isinstance(char_or_code, str):
        if len(char_or_code) == 1:
            # Single character
            code_point = ord(char_or_code)
            try:
                name = unicodedata.name(char_or_code)
                return code_point, char_or_code, name
            except ValueError:
                return code_point, char_or_code, f"U+{code_point:04X}"

        # Handle string representations of code points
        char_or_code = char_or_code.strip().upper()

        # Handle "U+XXXX" format
        if char_or_code.startswith("U+"):
            try:
                code_point = int(char_or_code[2:], 16)
                return get_unicode_info(code_point)
            except ValueError:
                pass

        # Handle "0xXX" format
        if char_or_code.startswith("0X"):
            try:
                code_point = int(char_or_code[2:], 16)
                return get_unicode_info(code_point)
            except ValueError:
                pass

        # Handle decimal format
        try:
            code_point = int(char_or_code)
            return get_unicode_info(code_point)
        except ValueError:
            pass

    # If we get here, the input couldn't be processed
    return None, None, None


def list_available_glyphs(font_path, filter_text=None):
    """
    List all available glyphs in the font with their Unicode code points and names.

    Parameters:
    - font_path: Path to the font file
    - filter_text: Optional text to filter glyph names or descriptions (case-insensitive)

    Returns:
    - A list of (code_point, character, name, glyph_name) tuples
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Loading font glyphs..."),
        transient=True,
    ) as progress:
        task = progress.add_task("Loading", total=None)

        font = TTFont(font_path)
        cmap = font.getBestCmap()

        # Get all mappings
        glyph_info = []
        for code_point, glyph_name in cmap.items():
            try:
                char = chr(code_point)
                try:
                    name = unicodedata.name(char)
                except ValueError:
                    name = f"U+{code_point:04X}"
            except (ValueError, OverflowError):
                char = "□"  # Placeholder for unprintable characters
                name = f"U+{code_point:04X}"

            glyph_info.append((code_point, char, name, glyph_name))

        # Sort by code point
        glyph_info.sort(key=lambda x: x[0])
        progress.update(task, completed=100)

    # Filter if requested
    if filter_text:
        filter_text = filter_text.lower()
        glyph_info = [
            info
            for info in glyph_info
            if (
                filter_text in info[2].lower()
                or filter_text in info[3].lower()
                or (info[1] and filter_text in info[1].lower())
            )
        ]

    return glyph_info


def print_glyph_table(glyph_info):
    """Print a formatted table of glyph information using Rich."""
    if not glyph_info:
        console.print("[bold red]No glyphs match your criteria.[/bold red]")
        return

    # Create a Rich table
    table = Table(title="Available Glyphs", box=box.ROUNDED)

    # Add columns
    table.add_column("Code Point", style="cyan")
    table.add_column("Character", style="magenta")
    table.add_column("Unicode Name", style="green")
    table.add_column("Glyph Name", style="yellow")

    # Add rows
    for code, char, name, glyph_name in glyph_info:
        table.add_row(f"U+{code:04X}", char, name, glyph_name)

    # Print the table
    console.print(table)
    console.print(f"[bold]Total:[/bold] {len(glyph_info)} glyphs")


def resolve_glyph_identifier(font_path, identifier):
    """
    Resolve a glyph identifier to a Unicode code point.

    Parameters:
    - font_path: Path to the font file
    - identifier: Can be a character, Unicode code point, or common name

    Returns:
    - (code_point, glyph_name) tuple, or (None, None) if not found
    """
    # Try to get Unicode info first
    code_point, char, name = get_unicode_info(identifier)

    if code_point is not None:
        # We have a valid code point, check if it exists in the font
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        if code_point in cmap:
            return code_point, cmap[code_point]

    # If not found or not a valid code point, try to match by name
    # This is useful for common names like "left_parenthesis" or "hyphen"
    if isinstance(identifier, str):
        search_term = identifier.lower()

        # Get all glyphs with their names
        glyph_info = list_available_glyphs(font_path)

        # Search by Unicode name
        for code, _, unicode_name, glyph_name in glyph_info:
            if unicode_name and search_term in unicode_name.lower():
                return code, glyph_name

        # Search by glyph name
        for code, _, _, glyph_name in glyph_info:
            if glyph_name and search_term in glyph_name.lower():
                return code, glyph_name

    return None, None


def recenter_multiple_glyphs(
    font_path, glyph_identifiers, output_path=None, debug=False
):
    """
    Recenter multiple glyphs horizontally within their advance widths.

    Parameters:
    - font_path: Path to the input font file
    - glyph_identifiers: List of glyph identifiers (code points, characters, or names)
    - output_path: Path for the modified font (if None, will append '_recentered' to original name)
    - debug: If True, will output detailed information about the transformation process

    Returns:
    - Tuple of (success_count, total_count)
    """
    if output_path is None:
        # Create default output filename if none provided
        directory, filename = os.path.split(font_path)
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{base}_recentered{ext}")

    console.print(
        f"[bold blue]Processing [cyan]{len(glyph_identifiers)}[/cyan] glyphs...[/bold blue]"
    )

    # Load the font only once
    font = TTFont(font_path)
    cmap = font.getBestCmap()

    success_count = 0
    processed = []

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Recentering glyphs...", total=len(glyph_identifiers)
        )

        for idx, glyph_identifier in enumerate(glyph_identifiers):
            progress.update(
                task,
                description=f"[cyan]Processing glyph {idx + 1}/{len(glyph_identifiers)}...",
            )

            # Resolve the glyph identifier to a Unicode code point
            glyph_code, resolved_glyph_name = resolve_glyph_identifier(
                font_path, glyph_identifier
            )

            if glyph_code is None:
                console.print(
                    f"  [yellow]⚠️ Skipping '{glyph_identifier}' - not found in font[/yellow]"
                )
                continue

            if glyph_code in processed:
                console.print(
                    f"  [yellow]⚠️ Skipping '{glyph_identifier}' - already processed[/yellow]"
                )
                continue

            # Get the glyph name
            glyph_name = cmap[glyph_code]

            # Process this glyph
            try:
                success = recenter_single_glyph(font, glyph_code, glyph_name, debug)
                if success:
                    success_count += 1
                    processed.append(glyph_code)
            except Exception as e:
                console.print(
                    f"  [red]❌ Error processing '{glyph_identifier}': {str(e)}[/red]"
                )

            progress.update(task, advance=1)

    # Save the modified font with all changes
    try:
        font.save(output_path)
        console.print(
            f"\n[bold green]✓ Font with {success_count} recentered glyphs saved to:[/bold green] {output_path}"
        )
    except Exception as e:
        console.print(f"[bold red]❌ Error saving font: {str(e)}[/bold red]")
        return 0, len(glyph_identifiers)

    return success_count, len(glyph_identifiers)


def recenter_single_glyph(font, glyph_code, glyph_name, debug=False):
    """
    Recenter a single glyph within the provided font object.

    Parameters:
    - font: The TTFont object
    - glyph_code: Unicode code point
    - glyph_name: Glyph name
    - debug: Enable debug output

    Returns:
    - True if successful, False otherwise
    """
    try:
        unicode_char = chr(glyph_code)
        try:
            unicode_name = unicodedata.name(unicode_char)
        except ValueError:
            unicode_name = f"U+{glyph_code:04X}"
    except (ValueError, OverflowError):
        unicode_char = "□"
        unicode_name = f"U+{glyph_code:04X}"

    if debug:
        glyph_text = Text()
        glyph_text.append("Processing glyph: ", style="bold green")
        glyph_text.append(f"{glyph_name}", style="yellow")
        glyph_text.append(" for ")
        glyph_text.append(f"'{unicode_char}'", style="magenta")
        glyph_text.append(f" ({unicode_name}, ", style="cyan")
        glyph_text.append(f"U+{glyph_code:04X}", style="cyan bold")
        glyph_text.append(")")
        console.print(glyph_text)

    # Get the glyph set to calculate bounds
    glyph_set = font.getGlyphSet()

    # Calculate the glyph's bounding box
    bounds_pen = BoundsPen(glyph_set)
    glyph_set[glyph_name].draw(bounds_pen)

    if bounds_pen.bounds is None:
        if debug:
            console.print(
                f"  [yellow]⚠️ Glyph '{glyph_name}' has no contours or is empty.[/yellow]"
            )
        return False

    # Get the current bounds
    xmin, ymin, xmax, ymax = bounds_pen.bounds
    glyph_width = xmax - xmin

    # Get the advance width from horizontal metrics
    hmtx = font["hmtx"]
    advance_width, fonttools_lsb = hmtx[glyph_name]

    # In FontLab, the LSB is the xmin value
    fontlab_lsb = xmin

    if debug:
        # Create a table for current metrics
        metrics_table = Table(title=f"Metrics for {glyph_name}", box=box.ROUNDED)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        metrics_table.add_row("Bounds (xmin, xmax)", f"{xmin}, {xmax}")
        metrics_table.add_row("Glyph width", f"{glyph_width}")
        metrics_table.add_row("Advance width", f"{advance_width}")
        metrics_table.add_row("LSB (FontTools)", f"{fonttools_lsb}")
        metrics_table.add_row("LSB (FontLab)", f"{fontlab_lsb}")

        console.print(metrics_table)

    # Calculate ideal left side bearing to center the glyph
    ideal_lsb = (advance_width - glyph_width) / 2

    # Calculate the shift amount needed (based on FontLab's LSB definition)
    shift_amount = int(ideal_lsb - fontlab_lsb)

    # Calculate the right side bearing for display
    rsb = advance_width - xmax
    ideal_rsb = advance_width - (ideal_lsb + glyph_width)

    if debug:
        # Create a table for centering calculations
        center_table = Table(title="Centering Calculations", box=box.ROUNDED)
        center_table.add_column("Metric", style="cyan")
        center_table.add_column("Value", style="yellow")

        center_table.add_row("Ideal LSB for centering", f"{ideal_lsb:.2f}")
        center_table.add_row("Shift amount needed", f"{shift_amount}")
        center_table.add_row("Current RSB", f"{rsb}")
        center_table.add_row("Ideal RSB after centering", f"{ideal_rsb:.2f}")

        console.print(center_table)

    # If there's no significant shift needed, we're done
    if abs(shift_amount) < 1:
        if debug:
            console.print(
                "  [green]✓ Glyph is already centered. No changes needed.[/green]"
            )
        return True

    # Apply the transformation
    if "glyf" in font:
        # For TrueType fonts (TTF)
        glyf = font["glyf"]
        glyph = glyf[glyph_name]

        if debug:
            console.print("  [blue]Processing TTF glyph...[/blue]")

        if not glyph.isComposite():
            # For simple glyphs with coordinates
            if hasattr(glyph, "coordinates") and glyph.coordinates is not None:
                # Apply the shift to each coordinate
                for i in range(len(glyph.coordinates)):
                    x, y = glyph.coordinates[i]
                    glyph.coordinates[i] = (x + shift_amount, y)

                # The glyph needs to recalculate its bounds
                glyph.recalcBounds(glyf)

                if debug:
                    console.print(
                        f"  [green]✓ Shifted simple glyph coordinates by {shift_amount} units[/green]"
                    )
        else:
            # For composite glyphs
            if debug:
                console.print(
                    f"  [blue]Processing composite glyph with {len(glyph.components)} components[/blue]"
                )

            # Shift each component
            for comp in glyph.components:
                comp.x += shift_amount

            # Recalculate bounds for composite glyph
            glyph.recalcBounds(glyf)

            if debug:
                console.print(
                    f"  [green]✓ Shifted composite glyph components by {shift_amount} units[/green]"
                )

        # Update the horizontal metrics (left side bearing)
        new_fonttools_lsb = fonttools_lsb + shift_amount
        hmtx[glyph_name] = (advance_width, int(new_fonttools_lsb))

    elif "CFF " in font:
        # For CFF/OpenType fonts (OTF)
        if debug:
            console.print("  [blue]Processing CFF/OpenType glyph...[/blue]")

        try:
            # Get CFF and glyph context
            cff = font["CFF "].cff
            top_dict = cff.topDictIndex[0]
            char_strings = top_dict.CharStrings
            glyph_set = font.getGlyphSet()

            if glyph_name not in char_strings:
                console.print(
                    f"  [yellow]⚠️ Glyph '{glyph_name}' not found in CFF CharStrings.[/yellow]"
                )
                return False

            # Create the transform matrix with the shift amount
            transform = Transform(1, 0, 0, 1, shift_amount, 0)

            # Record original glyph drawing
            rec_pen = RecordingPen()
            char_strings[glyph_name].draw(rec_pen)

            # Get advance width (required)
            width = hmtx[glyph_name][0]

            # Create charstring pen with required width and glyph set
            cs_pen = T2CharStringPen(width=width, glyphSet=glyph_set)

            # Apply transformation using TransformPen
            transform_pen = TransformPen(cs_pen, transform)
            rec_pen.replay(transform_pen)

            # Get the new charstring
            new_charstring = cs_pen.getCharString()

            # Preserve the private dictionary reference from the original CharString
            original_charstring = char_strings[glyph_name]
            if hasattr(original_charstring, "private"):
                new_charstring.private = original_charstring.private

            # Replace the glyph with the transformed one
            char_strings[glyph_name] = new_charstring

            # Update metrics
            new_fonttools_lsb = fonttools_lsb + shift_amount
            hmtx[glyph_name] = (advance_width, int(new_fonttools_lsb))

            if debug:
                console.print(
                    f"  [green]✓ Updated CFF glyph and metrics: FontTools LSB from {fonttools_lsb} to {int(new_fonttools_lsb)}[/green]"
                )

        except Exception as e:
            console.print(f"  [red]❌ Error processing CFF glyph: {e}[/red]")
            if debug:
                import traceback

                console.print(traceback.format_exc(), style="red")
            return False

    return True


def process_multiple_fonts(
    font_paths: List[str], glyph_identifier=None, output_dir=None, debug=False
) -> Tuple[int, int]:
    """
    Process multiple font files with the same glyph recentering parameters.

    Parameters:
    - font_paths: List of font file paths to process
    - glyph_identifier: Glyph identifier to recenter in each font
    - output_dir: Directory to save output files (if None, save in same directory as input)
    - debug: Enable debug output

    Returns:
    - Tuple of (success_count, total_count)
    """
    success_count = 0

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing fonts...", total=len(font_paths))

        for idx, font_path in enumerate(font_paths):
            font_filename = os.path.basename(font_path)
            progress.update(
                task,
                description=f"[cyan]Processing font {idx + 1}/{len(font_paths)}: {font_filename}...",
            )

            # Determine output path
            if output_dir:
                # Create output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Use the same filename but in the output directory
                base_filename = os.path.basename(font_path)
                output_path = os.path.join(output_dir, base_filename)

                # Don't overwrite input file
                if os.path.normpath(output_path) == os.path.normpath(font_path):
                    directory, filename = os.path.split(output_path)
                    base, ext = os.path.splitext(filename)
                    output_path = os.path.join(directory, f"{base}_recentered{ext}")
            else:
                # Create default output filename in the same directory
                directory, filename = os.path.split(font_path)
                base, ext = os.path.splitext(filename)
                output_path = os.path.join(directory, f"{base}_recentered{ext}")

            try:
                console.print(f"\n[bold blue]Processing font:[/bold blue] {font_path}")

                # Handle different processing modes while avoiding nested progress bars
                if glyph_identifier == "all":
                    # Process all glyphs in the font without using interactive mode
                    # and without creating a new progress bar
                    success = process_all_glyphs_in_font(
                        font_path, output_path, debug, progress
                    )
                elif isinstance(glyph_identifier, str) and "," in glyph_identifier:
                    # Process a batch of glyphs without creating a new progress bar
                    glyph_identifiers = [g.strip() for g in glyph_identifier.split(",")]
                    success = process_batch_glyphs_in_font(
                        font_path, glyph_identifiers, output_path, debug, progress
                    )
                else:
                    # Process a single glyph or use interactive mode
                    success = recenter_glyph_for_batch(
                        font_path, glyph_identifier, output_path, debug
                    )

                if success:
                    success_count += 1
                    console.print(
                        f"[green]✓ Successfully processed: {font_path}[/green]"
                    )
                else:
                    console.print(f"[yellow]⚠️ Failed to process: {font_path}[/yellow]")
            except Exception as e:
                console.print(f"[red]❌ Error processing {font_path}: {str(e)}[/red]")

            progress.update(task, advance=1)

    return success_count, len(font_paths)


def process_all_glyphs_in_font(
    font_path, output_path=None, debug=False, parent_progress=None
) -> bool:
    """
    Process all glyphs in a font without creating nested progress bars.

    Parameters:
    - font_path: Path to the font file
    - output_path: Path for the output font file
    - debug: Enable debug output
    - parent_progress: Parent progress bar (if called from batch processing)

    Returns:
    - True if successful, False otherwise
    """
    try:
        # Create default output path if none provided
        if output_path is None:
            directory, filename = os.path.split(font_path)
            base, ext = os.path.splitext(filename)
            output_path = os.path.join(directory, f"{base}_recentered{ext}")

        # Load the font
        font = TTFont(font_path)
        cmap = font.getBestCmap()

        # Get all code points in the font
        code_points = list(cmap.keys())
        success_count = 0

        console.print(
            f"[bold blue]Processing [cyan]{len(code_points)}[/cyan] glyphs in {os.path.basename(font_path)}...[/bold blue]"
        )

        # Process each glyph without progress bar (or use parent progress if provided)
        for code_point in code_points:
            glyph_name = cmap[code_point]

            try:
                success = recenter_single_glyph(font, code_point, glyph_name, debug)
                if success:
                    success_count += 1
            except Exception as e:
                if debug:
                    console.print(
                        f"  [red]Error processing glyph {glyph_name}: {e}[/red]"
                    )

        # Save the font with all changes
        font.save(output_path)
        console.print(
            f"[bold green]✓ Font with {success_count} recentered glyphs saved to:[/bold green] {output_path}"
        )
        return True

    except Exception as e:
        console.print(f"[bold red]❌ Error: {str(e)}[/bold red]")
        return False


def process_batch_glyphs_in_font(
    font_path, glyph_identifiers, output_path=None, debug=False, parent_progress=None
) -> bool:
    """
    Process a batch of glyphs in a font without creating nested progress bars.

    Parameters:
    - font_path: Path to the font file
    - glyph_identifiers: List of glyph identifiers to process
    - output_path: Path for the output font file
    - debug: Enable debug output
    - parent_progress: Parent progress bar (if called from batch processing)

    Returns:
    - True if successful, False otherwise
    """
    try:
        # Create default output path if none provided
        if output_path is None:
            directory, filename = os.path.split(font_path)
            base, ext = os.path.splitext(filename)
            output_path = os.path.join(directory, f"{base}_recentered{ext}")

        # Load the font
        font = TTFont(font_path)
        cmap = font.getBestCmap()

        success_count = 0
        processed = []

        console.print(
            f"[bold blue]Processing [cyan]{len(glyph_identifiers)}[/cyan] glyphs in {os.path.basename(font_path)}...[/bold blue]"
        )

        # Process each identified glyph without a new progress bar
        for glyph_identifier in glyph_identifiers:
            # Resolve the glyph identifier to a Unicode code point
            glyph_code, resolved_glyph_name = resolve_glyph_identifier(
                font_path, glyph_identifier
            )

            if glyph_code is None:
                if debug:
                    console.print(
                        f"  [yellow]⚠️ Skipping '{glyph_identifier}' - not found in font[/yellow]"
                    )
                continue

            if glyph_code in processed:
                if debug:
                    console.print(
                        f"  [yellow]⚠️ Skipping '{glyph_identifier}' - already processed[/yellow]"
                    )
                continue

            # Get the glyph name
            if glyph_code in cmap:
                glyph_name = cmap[glyph_code]

                # Process this glyph
                try:
                    success = recenter_single_glyph(font, glyph_code, glyph_name, debug)
                    if success:
                        success_count += 1
                        processed.append(glyph_code)
                except Exception as e:
                    if debug:
                        console.print(
                            f"  [red]❌ Error processing '{glyph_identifier}': {str(e)}[/red]"
                        )
            else:
                if debug:
                    console.print(
                        f"  [yellow]⚠️ Code point {glyph_code} not in font[/yellow]"
                    )

        # Save the font with all changes
        font.save(output_path)
        console.print(
            f"[bold green]✓ Font with {success_count} recentered glyphs saved to:[/bold green] {output_path}"
        )
        return success_count > 0

    except Exception as e:
        console.print(f"[bold red]❌ Error: {str(e)}[/bold red]")
        return False


def recenter_glyph_for_batch(
    font_path, glyph_identifier=None, output_path=None, debug=False
):
    """
    Version of recenter_glyph that doesn't use progress bars, for use in batch processing.

    Parameters:
    - font_path: Path to the input font file
    - glyph_identifier: Unicode code point, character, or name of the glyph to recenter
    - output_path: Path for the modified font (if None, will append '_recentered' to original name)
    - debug: If True, will output detailed information about the transformation process

    Returns:
    - True if successful, False otherwise
    """
    # Handle special cases to avoid interactive mode or progress bars
    if glyph_identifier is None:
        # When in batch mode and no identifier is provided, we can't do interactive selection
        console.print(
            "[yellow]⚠️ No glyph specified for batch processing. Please provide specific glyphs for batch mode.[/yellow]"
        )
        return False

    # Process a single glyph
    if output_path is None:
        # Create default output filename if none provided
        directory, filename = os.path.split(font_path)
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{base}_recentered{ext}")

    # Resolve the glyph identifier to a Unicode code point
    try:
        glyph_code, resolved_glyph_name = resolve_glyph_identifier(
            font_path, glyph_identifier
        )
    except Exception as e:
        console.print(f"[red]❌ Error resolving glyph identifier: {str(e)}[/red]")
        return False

    if glyph_code is None:
        console.print(
            f"[yellow]⚠️ Could not find glyph matching '{glyph_identifier}' in the font.[/yellow]"
        )
        return False

    # Load the font
    try:
        font = TTFont(font_path)
    except Exception as e:
        console.print(f"[red]❌ Error loading font: {str(e)}[/red]")
        return False

    # Get the glyph name for the specified character code
    cmap = font.getBestCmap()
    if glyph_code not in cmap:
        console.print(
            f"[yellow]⚠️ Character code {glyph_code} not found in font.[/yellow]"
        )
        return False

    glyph_name = cmap[glyph_code]

    if debug:
        # Display glyph info
        try:
            unicode_char = chr(glyph_code)
            try:
                unicode_name = unicodedata.name(unicode_char)
            except ValueError:
                unicode_name = f"U+{glyph_code:04X}"

            console.print(
                f"Processing glyph: {glyph_name} ('{unicode_char}', {unicode_name}, U+{glyph_code:04X})"
            )
        except (ValueError, OverflowError):
            console.print(f"Processing glyph: {glyph_name} (U+{glyph_code:04X})")

    # Process the single glyph
    success = recenter_single_glyph(font, glyph_code, glyph_name, debug)

    if not success:
        console.print("[yellow]⚠️ Failed to recenter glyph.[/yellow]")
        return False

    # Save the modified font
    try:
        font.save(output_path)
    except Exception as e:
        console.print(f"[red]❌ Error saving font: {str(e)}[/red]")
        return False

    if debug:
        console.print(f"[green]✓ Recentered glyph saved to: {output_path}[/green]")

    return True


def recenter_glyph(font_path, glyph_identifier=None, output_path=None, debug=False):
    """
    Recenter a glyph horizontally within its advance width.

    Parameters:
    - font_path: Path to the input font file
    - glyph_identifier: Unicode code point, character, or name of the glyph to recenter
    - output_path: Path for the modified font (if None, will append '_recentered' to original name)
    - debug: If True, will output detailed information about the transformation process

    Returns:
    - True if successful, False otherwise
    """
    # If no glyph identifier is provided, let the user select glyphs interactively
    if glyph_identifier is None:
        # Let user select interactively from a list or all glyphs
        glyph_identifiers = select_glyphs_interactively(font_path)
        if not glyph_identifiers:
            return False

        # Process multiple glyphs
        success_count, total = recenter_multiple_glyphs(
            font_path, glyph_identifiers, output_path, debug
        )
        return success_count > 0

    # Check if the glyph_identifier is "all" to process all glyphs
    if glyph_identifier == "all":
        # Process all glyphs using the dedicated function that doesn't create nested progress bars
        return process_all_glyphs_in_font(font_path, output_path, debug)

    # Check if this is a batch of glyphs (comma-separated list)
    if isinstance(glyph_identifier, str) and "," in glyph_identifier:
        glyph_identifiers = [g.strip() for g in glyph_identifier.split(",")]
        return process_batch_glyphs_in_font(
            font_path, glyph_identifiers, output_path, debug
        )

    # Process a single glyph (original functionality)
    if output_path is None:
        # Create default output filename if none provided
        directory, filename = os.path.split(font_path)
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{base}_recentered{ext}")

    # Resolve the glyph identifier to a Unicode code point
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Resolving glyph identifier..."),
        transient=True,
    ) as progress:
        task = progress.add_task("Resolving", total=None)
        glyph_code, resolved_glyph_name = resolve_glyph_identifier(
            font_path, glyph_identifier
        )
        progress.update(task, completed=100)

    if glyph_code is None:
        console.print(
            f"[bold red]Could not find glyph matching '{glyph_identifier}' in the font.[/bold red]"
        )
        console.print(
            "Use the [bold yellow]--list[/bold yellow] option to see available glyphs."
        )
        return False

    # Load the font
    font = TTFont(font_path)

    # Get the glyph name for the specified character code
    cmap = font.getBestCmap()
    if glyph_code not in cmap:
        console.print(
            f"[bold red]Character code {glyph_code} not found in font.[/bold red]"
        )
        return False

    glyph_name = cmap[glyph_code]
    unicode_char = chr(glyph_code)

    try:
        unicode_name = unicodedata.name(unicode_char)
    except ValueError:
        unicode_name = f"U+{glyph_code:04X}"

    glyph_text = Text()
    glyph_text.append("Found glyph: ", style="bold green")
    glyph_text.append(f"{glyph_name}", style="yellow")
    glyph_text.append(" for ")
    glyph_text.append(f"'{unicode_char}'", style="magenta")
    glyph_text.append(f" ({unicode_name}, ", style="cyan")
    glyph_text.append(f"U+{glyph_code:04X}", style="cyan bold")
    glyph_text.append(")")

    console.print(glyph_text)

    # Process the single glyph
    success = recenter_single_glyph(font, glyph_code, glyph_name, debug)

    if not success:
        console.print("[bold red]Failed to recenter glyph.[/bold red]")
        return False

    # Save the modified font
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Saving font..."),
    ) as progress:
        task = progress.add_task("Saving", total=100)
        progress.update(task, advance=50)

        try:
            font.save(output_path)
            progress.update(task, advance=50)
        except Exception as e:
            console.print(f"[bold red]Error saving font: {str(e)}[/bold red]")
            return False

    console.print(
        f"\n[bold green]✓ Recentered glyph saved to:[/bold green] {output_path}"
    )

    # Verify the change
    console.print("\n[bold blue]Verifying results...[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Verifying changes..."),
        transient=True,
    ) as progress:
        task = progress.add_task("Verifying", total=None)

        verify_font = TTFont(output_path)
        verify_gs = verify_font.getGlyphSet()
        verify_bounds_pen = BoundsPen(verify_gs)
        verify_gs[glyph_name].draw(verify_bounds_pen)

        progress.update(task, completed=100)

    if verify_bounds_pen.bounds:
        new_xmin, new_ymin, new_xmax, new_ymax = verify_bounds_pen.bounds
        new_width = new_xmax - new_xmin
        new_fonttools_lsb = verify_font["hmtx"][glyph_name][1]

        # Get the advance width from the updated font
        advance_width = verify_font["hmtx"][glyph_name][0]

        # Calculate left and right spaces (FontLab style)
        new_fontlab_lsb = new_xmin
        new_rsb = advance_width - new_xmax

        # Create a verification table
        verify_table = Table(title="Verification Results", box=box.ROUNDED)
        verify_table.add_column("Metric", style="cyan")
        verify_table.add_column("Value", style="green")

        verify_table.add_row("New bounds (xmin, xmax)", f"{new_xmin}, {new_xmax}")
        verify_table.add_row("New glyph width", f"{new_width}")
        verify_table.add_row("Advance width", f"{advance_width}")
        verify_table.add_row("New FontTools LSB", f"{new_fonttools_lsb}")
        verify_table.add_row("New FontLab LSB", f"{new_fontlab_lsb}")
        verify_table.add_row("New RSB", f"{new_rsb}")

        console.print(verify_table)

        # Create a final spacing table
        final_spacing = Table(title="Final Spacing Balance", box=box.ROUNDED)
        final_spacing.add_column("Side", style="cyan")
        final_spacing.add_column("Value", style="green")

        final_spacing.add_row("Left spacing", f"{new_fontlab_lsb}")
        final_spacing.add_row("Right spacing", f"{new_rsb}")
        final_spacing.add_row("Difference", f"{abs(new_fontlab_lsb - new_rsb)}")

        console.print(final_spacing)

        # Calculate the ideal balanced LSB for reference
        ideal_balanced_lsb = (advance_width - new_width) / 2
        error = abs(new_fontlab_lsb - ideal_balanced_lsb)

        console.print(
            f"Ideal balanced LSB would be: [yellow]{ideal_balanced_lsb:.2f}[/yellow]"
        )

        if error > 1:
            console.print(f"Centering error: [yellow]{error:.2f}[/yellow] units")
            console.print(
                "[dim]Note: Small centering error may be due to rounding.[/dim]"
            )
        else:
            console.print(f"Centering error: [green]{error:.2f}[/green] units")

    return True


def select_glyphs_interactively(font_path):
    """Allow user to interactively select glyphs to process."""
    console.print("\n[bold blue]Interactive Glyph Selection[/bold blue]")

    # Ask if user wants to filter the glyphs
    filter_first = Confirm.ask("Would you like to filter the glyphs first?")

    if filter_first:
        filter_text = Prompt.ask(
            "Enter search term (e.g., 'parenthesis', 'bracket', etc.)"
        )
        glyph_info = list_available_glyphs(font_path, filter_text)
    else:
        # Ask if user wants to see all glyphs (could be a lot)
        show_all = Confirm.ask(
            "Would you like to see all available glyphs? (This could be a long list)"
        )

        if show_all:
            glyph_info = list_available_glyphs(font_path)
        else:
            # Without seeing the list, user can directly specify glyph identifiers
            direct_input = Prompt.ask(
                "Enter glyph identifiers separated by commas\n(e.g., '(,),{,}' or 'U+0028,U+0029,U+007B,U+007D')"
            )
            return [g.strip() for g in direct_input.split(",")]

    # Print the filtered or complete glyph table
    print_glyph_table(glyph_info)

    # Ask if user wants to process all displayed glyphs
    if len(glyph_info) > 0:
        process_all_displayed = Confirm.ask("Process all displayed glyphs?")

        if process_all_displayed:
            return [info[0] for info in glyph_info]  # Return all code points

    # Let user select specific glyphs from the displayed list
    selection_input = Prompt.ask(
        "Enter the indices, code points, or characters to process separated by commas\n"
        "(e.g., '1,3,5' or 'U+0028,)' or '(,)')"
    )

    selected_glyphs = []

    # Process the selection input
    for item in selection_input.split(","):
        item = item.strip()

        # Try to parse as an index into the displayed table
        try:
            idx = int(item) - 1  # Convert to 0-based index
            if 0 <= idx < len(glyph_info):
                selected_glyphs.append(glyph_info[idx][0])  # Add the code point
                continue
        except ValueError:
            pass

        # Otherwise treat as a glyph identifier
        selected_glyphs.append(item)

    return selected_glyphs


def find_font_files(directory_path: str, recursive: bool = False) -> List[str]:
    """
    Find all font files in the specified directory.

    Parameters:
    - directory_path: Path to the directory to search
    - recursive: If True, search in subdirectories as well

    Returns:
    - List of font file paths
    """
    font_extensions = [".ttf", ".otf", ".woff", ".woff2"]

    if not os.path.isdir(directory_path):
        console.print(
            f"[bold red]Error:[/bold red] {directory_path} is not a directory."
        )
        return []

    font_files = []

    search_pattern = os.path.join(directory_path, "**" if recursive else "*")

    for ext in font_extensions:
        # Use recursive glob pattern if recursive=True
        pattern = f"{search_pattern}/*{ext}" if recursive else f"{search_pattern}{ext}"
        font_files.extend(glob.glob(pattern, recursive=recursive))

    if not font_files:
        console.print(f"[yellow]No font files found in {directory_path}[/yellow]")
    else:
        console.print(
            f"[green]Found {len(font_files)} font files in {directory_path}[/green]"
        )

    return sorted(font_files)


def main():
    console.print(
        Panel.fit(
            "[bold]Glyph Recentering Tool[/bold]\nHorizontally centers glyphs within their advance width",
            border_style="blue",
        )
    )

    # Parse command line arguments
    if len(sys.argv) < 2:
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print(
            "  python recenter_glyph.py <font_file|directory> [glyph_identifier] [output_file|output_dir] [options]"
        )
        console.print("\n[bold yellow]Options:[/bold yellow]")
        console.print("  --debug       Enable debug output")
        console.print("  --list        List all available glyphs in the font")
        console.print("  --find TEXT   Find glyphs matching TEXT")
        console.print("  --all         Process all glyphs in the font")
        console.print("  --batch GLYPHS Process multiple glyphs (comma-separated)")
        console.print("  --interactive Start interactive mode (if no glyph specified)")
        console.print("  --dir         Process all fonts in the specified directory")
        console.print("  --recursive   When used with --dir, include subdirectories")
        console.print(
            "  --output-dir DIR  Specify output directory for processed fonts"
        )

        console.print("\n[bold yellow]Examples:[/bold yellow]")
        examples_table = Table(show_header=False, box=box.SIMPLE)
        examples_table.add_column("Command", style="green")
        examples_table.add_column("Description", style="dim")

        examples_table.add_row(
            "python recenter_glyph.py myfont.ttf", "Start in interactive mode"
        )
        examples_table.add_row(
            "python recenter_glyph.py myfont.ttf '('", "Recenter by character"
        )
        examples_table.add_row(
            "python recenter_glyph.py myfont.ttf --all", "Recenter all glyphs"
        )
        examples_table.add_row(
            "python recenter_glyph.py myfont.ttf --batch '(,),[,],{,}'",
            "Recenter brackets",
        )
        examples_table.add_row(
            "python recenter_glyph.py fonts_dir/ --dir",
            "Process all fonts in directory",
        )
        examples_table.add_row(
            "python recenter_glyph.py fonts_dir/ --dir --recursive",
            "Process all fonts recursively",
        )
        examples_table.add_row(
            "python recenter_glyph.py fonts_dir/ '(' --dir", "Recenter '(' in all fonts"
        )
        examples_table.add_row(
            "python recenter_glyph.py myfont.ttf --list", "List all glyphs"
        )
        examples_table.add_row(
            "python recenter_glyph.py myfont.ttf --find bracket", "Find brackets"
        )

        console.print(examples_table)

        console.print("\n[bold yellow]Common glyph identifiers:[/bold yellow]")
        common_table = Table(show_header=False, box=box.SIMPLE)
        common_table.add_column("Glyph", style="cyan")
        common_table.add_column("Identifiers", style="green")

        common_table.add_row("Left Parenthesis", "'(' or 40 or U+0028")
        common_table.add_row("Right Parenthesis", "')' or 41 or U+0029")
        common_table.add_row("Left Square Bracket", "'[' or 91 or U+005B")
        common_table.add_row("Right Square Bracket", "']' or 93 or U+005D")
        common_table.add_row("Left Curly Brace", "'{' or 123 or U+007B")
        common_table.add_row("Right Curly Brace", "'}' or 125 or U+007D")
        common_table.add_row("Hyphen", "'-' or 45 or U+002D")
        common_table.add_row("En Dash", "'–' or 8211 or U+2013")
        common_table.add_row("Em Dash", "'—' or 8212 or U+2014")

        console.print(common_table)
        sys.exit(1)

    input_path = sys.argv[1]
    args = sys.argv[2:]

    # Process special flags
    debug = "--debug" in args
    if debug:
        args.remove("--debug")

    process_dir = "--dir" in args
    if process_dir:
        args.remove("--dir")

    recursive = "--recursive" in args
    if recursive:
        args.remove("--recursive")

    # Extract output directory if specified
    output_dir = None
    output_dir_index = -1
    try:
        output_dir_index = args.index("--output-dir")
        if output_dir_index < len(args) - 1:
            output_dir = args[output_dir_index + 1]
            # Remove --output-dir and its value
            args = args[:output_dir_index] + args[output_dir_index + 2 :]
        else:
            console.print(
                "[bold red]Error:[/bold red] --output-dir option requires a directory path."
            )
            sys.exit(1)
    except ValueError:
        pass  # --output-dir not in args

    # Check for --list option
    if "--list" in args:
        if process_dir:
            console.print(
                "[bold red]Error:[/bold red] --list cannot be used with --dir option."
            )
            sys.exit(1)

        console.print(
            f"[bold blue]Loading glyphs from[/bold blue] [yellow]{input_path}[/yellow]"
        )
        glyph_info = list_available_glyphs(input_path)
        print_glyph_table(glyph_info)
        sys.exit(0)

    # Check for --find option
    find_index = -1
    try:
        find_index = args.index("--find")
        if find_index < len(args) - 1:
            if process_dir:
                console.print(
                    "[bold red]Error:[/bold red] --find cannot be used with --dir option."
                )
                sys.exit(1)

            search_text = args[find_index + 1]
            console.print(
                f"[bold blue]Searching for glyphs matching[/bold blue] [yellow]{search_text}[/yellow]"
            )
            glyph_info = list_available_glyphs(input_path, search_text)
            print_glyph_table(glyph_info)
            if glyph_info:
                console.print(
                    "\nTo recenter one of these glyphs, use the character or code point as the glyph identifier."
                )
            sys.exit(0)
        else:
            console.print(
                "[bold red]Error:[/bold red] --find option requires a search term."
            )
            sys.exit(1)
    except ValueError:
        pass  # --find not in args, continue normal operation

    # Normal operation setup
    glyph_identifier = None  # Default to interactive mode
    process_all = "--all" in args
    interactive_mode = "--interactive" in args

    if process_all:
        args.remove("--all")
        glyph_identifier = "all"

    if interactive_mode:
        args.remove("--interactive")

    # Check for batch processing
    batch_index = -1
    try:
        batch_index = args.index("--batch")
        if batch_index < len(args) - 1:
            glyph_identifier = args[batch_index + 1]
            # Remove --batch and its value
            args = args[:batch_index] + args[batch_index + 2 :]
        else:
            console.print(
                "[bold red]Error:[/bold red] --batch option requires a comma-separated list of glyphs."
            )
            sys.exit(1)
    except ValueError:
        pass  # --batch not in args

    # Process remaining positional arguments
    remaining_args = [arg for arg in args if not arg.startswith("--")]

    if len(remaining_args) >= 1 and not process_all and glyph_identifier is None:
        glyph_identifier = remaining_args[0]

    output_path = None
    if len(remaining_args) >= 2:
        output_path = remaining_args[1]

    # Directory processing mode
    if process_dir:
        if not os.path.isdir(input_path):
            console.print(
                f"[bold red]Error:[/bold red] {input_path} is not a directory."
            )
            sys.exit(1)

        # Find all font files in the directory
        font_files = find_font_files(input_path, recursive)

        if not font_files:
            console.print(f"[bold red]No font files found in {input_path}[/bold red]")
            sys.exit(1)

        # Show operation details for directory processing
        operation_table = Table(title="Directory Processing Settings", box=box.ROUNDED)
        operation_table.add_column("Parameter", style="cyan")
        operation_table.add_column("Value", style="yellow")

        operation_table.add_row("Input directory", input_path)
        operation_table.add_row("Recursive", "Yes" if recursive else "No")
        operation_table.add_row("Font files found", str(len(font_files)))

        if glyph_identifier == "all":
            operation_table.add_row("Glyphs to recenter", "ALL GLYPHS")
        elif glyph_identifier and "," in glyph_identifier:
            operation_table.add_row("Glyphs to recenter", f"Batch: {glyph_identifier}")
        elif glyph_identifier:
            operation_table.add_row("Glyph to recenter", str(glyph_identifier))
        else:
            operation_table.add_row("Glyph selection", "Interactive")

        operation_table.add_row(
            "Output directory",
            output_dir if output_dir else "Same as input (with _recentered suffix)",
        )
        operation_table.add_row("Debug mode", "Enabled" if debug else "Disabled")

        console.print(operation_table)

        # Process all fonts in the directory
        success_count, total = process_multiple_fonts(
            font_files, glyph_identifier, output_dir, debug
        )

        # Show summary
        console.print(
            f"\n[bold]Summary:[/bold] Successfully processed {success_count} out of {total} fonts."
        )

        if success_count > 0:
            console.print(
                "\n[bold green]✓ Directory processing completed successfully.[/bold green]"
            )
        else:
            console.print("\n[bold red]❌ Failed to process any fonts.[/bold red]")
    else:
        # Single file mode - check if the font file exists
        if not os.path.isfile(input_path):
            console.print(
                f"[bold red]Error:[/bold red] Font file '{input_path}' not found."
            )
            sys.exit(1)

        # Show operation details for single file processing
        operation_table = Table(title="Operation Settings", box=box.ROUNDED)
        operation_table.add_column("Parameter", style="cyan")
        operation_table.add_column("Value", style="yellow")

        operation_table.add_row("Input font", input_path)

        if process_all:
            operation_table.add_row("Glyphs to recenter", "ALL GLYPHS")
        elif glyph_identifier and "," in glyph_identifier:
            operation_table.add_row("Glyphs to recenter", f"Batch: {glyph_identifier}")
        elif glyph_identifier:
            operation_table.add_row("Glyph to recenter", str(glyph_identifier))
        else:
            operation_table.add_row("Glyph selection", "Interactive")

        operation_table.add_row(
            "Output file", output_path if output_path else "auto-generated filename"
        )
        operation_table.add_row("Debug mode", "Enabled" if debug else "Disabled")

        console.print(operation_table)

        # Process the single font file
        success = recenter_glyph(input_path, glyph_identifier, output_path, debug)

        if success:
            console.print(
                "\n[bold green]✓ Glyph recentering completed successfully.[/bold green]"
            )
        else:
            console.print("\n[bold red]❌ Failed to recenter glyph(s).[/bold red]")


if __name__ == "__main__":
    main()
