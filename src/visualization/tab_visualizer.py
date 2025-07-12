"""
Visualization tools for guitar tablature.

This module provides tools for rendering guitar tablature in various formats,
including ASCII, SVG, PNG, and interactive HTML visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TabNote:
    """Representation of a note in a tab."""
    string: int
    fret: int
    time_position: float
    duration: float
    is_bend: bool = False
    is_hammer_on: bool = False
    is_pull_off: bool = False
    is_slide: bool = False
    bend_value: float = 0.0


class TabVisualizer:
    """
    Visualization tools for guitar tablature.
    """
    
    def __init__(
        self,
        num_strings: int = 6,
        tuning: Optional[List[str]] = None,
        font_path: Optional[str] = None
    ):
        """
        Initialize the tab visualizer.
        
        Args:
            num_strings: Number of strings on the instrument
            tuning: Tuning of the instrument as a list of note names
            font_path: Path to a custom font for rendering
        """
        self.num_strings = num_strings
        
        # Set default tuning if not provided
        if tuning is None:
            if num_strings == 6:
                # Standard guitar tuning
                self.tuning = ['E', 'B', 'G', 'D', 'A', 'E']
            elif num_strings == 4:
                # Standard bass tuning
                self.tuning = ['G', 'D', 'A', 'E']
            else:
                # Generic tuning
                self.tuning = [f'String {i+1}' for i in range(num_strings)]
        else:
            self.tuning = tuning
        
        # Try to find a monospace font for rendering
        self.font_path = font_path
        if self.font_path is None:
            # Default fonts to try
            font_paths = [
                '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',  # Linux
                '/System/Library/Fonts/Monaco.ttf',  # macOS
                'C:\\Windows\\Fonts\\consola.ttf',   # Windows
                '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',  # Linux alternative
                '/usr/share/fonts/TTF/DejaVuSansMono.ttf',  # Arch Linux
                '/usr/share/fonts/dejavu/DejaVuSansMono.ttf'  # Other Linux distros
            ]
            
            for path in font_paths:
                if os.path.exists(path):
                    self.font_path = path
                    break

            # If no font found, log a warning
            if self.font_path is None:
                logger.warning("No suitable font found. Will use default font.")

    def parse_tab_file(self, tab_file_path: str) -> List[TabNote]:
        """
        Parse a tab file into a list of TabNote objects.
        
        Args:
            tab_file_path: Path to the tab file
            
        Returns:
            List of TabNote objects
        """
        # Check file extension
        ext = os.path.splitext(tab_file_path)[1].lower()
        
        if ext == '.tab':
            # Parse text-based tab
            with open(tab_file_path, 'r') as f:
                tab_string = f.read()
            return self.parse_tab_string(tab_string)
        elif ext == '.mid' or ext == '.midi':
            # Parse MIDI file
            return self.parse_midi_file(tab_file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def parse_tab_string(self, tab_string: str) -> List[TabNote]:
        """
        Parse a string representation of a tab into a list of TabNote objects.
        
        Args:
            tab_string: String representation of a tablature
            
        Returns:
            List of TabNote objects
        """
        # This is a simplified parser for standard tab notation
        lines = tab_string.strip().split('\n')
        
        # Filter out non-tab lines
        tab_lines = [line for line in lines if '-' in line]
        
        if not tab_lines:
            return []
        
        # Parse notes from the tab
        notes = []
        
        # Scan each string for notes
        for string_idx, string_line in enumerate(tab_lines):
            current_pos = 0
            
            for pos_idx, char in enumerate(string_line):
                if char.isdigit():
                    # Check if it's a multi-digit fret number
                    if pos_idx + 1 < len(string_line) and string_line[pos_idx + 1].isdigit():
                        fret_num = int(char + string_line[pos_idx + 1])
                        pos_idx += 1  # Skip the next character
                    else:
                        fret_num = int(char)
                    
                    # Check for technique markings (bend, hammer-on, etc.)
                    is_bend = False
                    is_hammer_on = False
                    is_pull_off = False
                    is_slide = False
                    bend_value = 0.0
                    
                    if pos_idx + 1 < len(string_line):
                        if string_line[pos_idx + 1] == 'b':
                            is_bend = True
                            # Look for bend value
                            if pos_idx + 2 < len(string_line) and string_line[pos_idx + 2].isdigit():
                                bend_value = float(string_line[pos_idx + 2])
                        elif string_line[pos_idx + 1] == 'h':
                            is_hammer_on = True
                        elif string_line[pos_idx + 1] == 'p':
                            is_pull_off = True
                        elif string_line[pos_idx + 1] == '/':
                            is_slide = True
                    
                    # Create a TabNote
                    notes.append(TabNote(
                        string=string_idx,
                        fret=fret_num,
                        time_position=pos_idx,  # Use position in the string as time
                        duration=1.0,  # Default duration
                        is_bend=is_bend,
                        is_hammer_on=is_hammer_on,
                        is_pull_off=is_pull_off,
                        is_slide=is_slide,
                        bend_value=bend_value
                    ))
        
        return notes
    
    def parse_midi_file(self, midi_file_path: str) -> List[TabNote]:
        """
        Parse a MIDI file into a list of TabNote objects.
        
        Args:
            midi_file_path: Path to the MIDI file
            
        Returns:
            List of TabNote objects
        """
        try:
            import pretty_midi
        except ImportError:
            logger.error("pretty_midi package not found. Install with: pip install pretty_midi")
            return []
        
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        
        # Extract notes
        notes = []
        
        # Standard guitar tuning in MIDI note numbers
        # E2 (40), A2 (45), D3 (50), G3 (55), B3 (59), E4 (64)
        tuning_midi = [40, 45, 50, 55, 59, 64][:self.num_strings]
        
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                # Find the best string/fret combination
                string_idx, fret = self._find_best_string_fret(note.pitch, tuning_midi)
                
                notes.append(TabNote(
                    string=string_idx,
                    fret=fret,
                    time_position=note.start * 4,  # Convert to quarter notes
                    duration=note.end - note.start,
                    is_bend=False,
                    is_hammer_on=False,
                    is_pull_off=False,
                    is_slide=False,
                    bend_value=0.0
                ))
        
        return notes
    
    def _find_best_string_fret(self, pitch: int, tuning_midi: List[int]) -> Tuple[int, int]:
        """
        Find the best string and fret to play a given MIDI pitch.
        
        Args:
            pitch: MIDI pitch number
            tuning_midi: List of MIDI note numbers for each string
            
        Returns:
            Tuple of (string_idx, fret)
        """
        best_string = 0
        best_fret = 0
        best_score = float('inf')
        
        for string, open_note in enumerate(tuning_midi):
            # Check if the note can be played on this string
            if pitch >= open_note:
                fret = pitch - open_note
                
                # Prefer lower frets and higher strings
                score = fret + 0.1 * string
                
                # Check if fret is within reasonable range
                if fret <= 24 and score < best_score:
                    best_string = string
                    best_fret = fret
                    best_score = score
        
        return best_string, best_fret
    
    def render_tab_as_text(self, notes: List[TabNote], measures_per_line: int = 4) -> str:
        """
        Render tab notes as ASCII tablature.
        
        Args:
            notes: List of TabNote objects
            measures_per_line: Number of measures per line
            
        Returns:
            ASCII representation of the tablature
        """
        if not notes:
            return "No notes to render"
        
        # Sort notes by time position
        sorted_notes = sorted(notes, key=lambda n: n.time_position)
        
        # Determine the time span
        max_time = max(note.time_position for note in sorted_notes)
        
        # Determine measures
        measure_width = 16  # Characters per measure
        total_measures = int(max_time / 4) + 1  # 4 beats per measure
        
        # Initialize the tab lines with proper string labels and spacing
        tab_lines = []
        for string_name in self.tuning:
            tab_lines.append(f"{string_name}|")
        
        # Add measure dividers with cleaner formatting
        for measure in range(total_measures):
            # Check if we need to start a new line
            if measure > 0 and measure % measures_per_line == 0:
                # Add a newline and restart the tab lines
                full_tab = "\n".join(tab_lines) + "\n\n"
                tab_lines = []
                for string_name in self.tuning:
                    tab_lines.append(f"{string_name}|")
            
            # Add the measure divider with proper spacing
            for i in range(self.num_strings):
                tab_lines[i] += "-" * (measure_width - 1) + "|"
        
        # Place notes in the tab with proper spacing
        for note in sorted_notes:
            # Calculate position in the tab
            measure = int(note.time_position / 4)
            position_in_measure = note.time_position % 4
            
            # Calculate the line in the tab (which set of measures)
            line = measure // measures_per_line
            
            # Calculate the position in the line
            measure_in_line = measure % measures_per_line
            
            # Calculate the character position
            char_pos = len(str(self.tuning[0])) + 1  # Account for tuning label and first |
            char_pos += measure_in_line * measure_width  # Account for measures
            char_pos += int(position_in_measure * (measure_width - 1) / 4)  # Position within measure
            
            # Place the note in the tab
            line_idx = note.string
            
            # Make sure the tab line is long enough
            while len(tab_lines[line_idx]) <= char_pos:
                tab_lines[line_idx] += "-"
            
            # Insert the fret number
            fret_str = str(note.fret)
            
            # Check if we need to overwrite characters
            for i, c in enumerate(fret_str):
                if char_pos + i < len(tab_lines[line_idx]):
                    tab_line_chars = list(tab_lines[line_idx])
                    line_char = tab_line_chars[char_pos + i]
                    
                    # Don't overwrite measure dividers
                    if line_char != '|':
                        tab_line_chars[char_pos + i] = c
                        tab_lines[line_idx] = ''.join(tab_line_chars)
            
            # Add technique markings
            technique_pos = char_pos + len(fret_str)
            if note.is_bend and technique_pos < len(tab_lines[line_idx]):
                if tab_lines[line_idx][technique_pos] == '-':
                    tab_line_chars = list(tab_lines[line_idx])
                    tab_line_chars[technique_pos] = 'b'
                    tab_lines[line_idx] = ''.join(tab_line_chars)
            elif note.is_hammer_on and technique_pos < len(tab_lines[line_idx]):
                if tab_lines[line_idx][technique_pos] == '-':
                    tab_line_chars = list(tab_lines[line_idx])
                    tab_line_chars[technique_pos] = 'h'
                    tab_lines[line_idx] = ''.join(tab_line_chars)
            elif note.is_pull_off and technique_pos < len(tab_lines[line_idx]):
                if tab_lines[line_idx][technique_pos] == '-':
                    tab_line_chars = list(tab_lines[line_idx])
                    tab_line_chars[technique_pos] = 'p'
                    tab_lines[line_idx] = ''.join(tab_line_chars)
            elif note.is_slide and technique_pos < len(tab_lines[line_idx]):
                if tab_lines[line_idx][technique_pos] == '-':
                    tab_line_chars = list(tab_lines[line_idx])
                    tab_line_chars[technique_pos] = '/'
                    tab_lines[line_idx] = ''.join(tab_line_chars)
        
        # Join the lines to form the complete tab
        full_tab = "\n".join(tab_lines)
        
        return full_tab
    
    def render_tab_as_image(
        self, 
        notes: List[TabNote], 
        output_path: Optional[str] = None,
        width: int = 1200,
        height: int = 400,
        measures_per_line: int = 4,
        show_grid: bool = True
    ) -> Image.Image:
        """
        Render tab notes as a PNG image.
        
        Args:
            notes: List of TabNote objects
            output_path: Path to save the image (optional)
            width: Width of the image
            height: Height of the image
            measures_per_line: Number of measures per line
            show_grid: Whether to show measure grid lines
            
        Returns:
            PIL Image object
        """
        if not notes:
            # Create an empty image with text
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            try:
                if self.font_path:
                    font = ImageFont.truetype(self.font_path, 20)
                else:
                    font = ImageFont.load_default()
            except (OSError, IOError):
                font = ImageFont.load_default()
            
            draw.text((width // 2 - 50, height // 2), "No notes to render", fill='black', font=font)
            
            if output_path:
                img.save(output_path)
            
            return img
        
        # Sort notes by time position
        sorted_notes = sorted(notes, key=lambda n: n.time_position)
        
        # Determine the time span
        max_time = max(note.time_position for note in sorted_notes) + 4  # Add a bit of padding
        
        # Determine measures
        measure_width = width // measures_per_line
        total_measures = int(max_time / 4) + 1  # 4 beats per measure
        
        # Calculate the number of lines needed
        num_lines = (total_measures + measures_per_line - 1) // measures_per_line
        
        # Calculate the total height needed
        line_height = 100  # Height per line of tab
        total_height = num_lines * line_height + 50  # Add some padding
        
        # Adjust height if necessary
        actual_height = max(height, total_height)
        
        # Create the image
        img = Image.new('RGB', (width, actual_height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, 14)
                title_font = ImageFont.truetype(self.font_path, 20)
            else:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()
        except (OSError, IOError):
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Draw title
        draw.text((width // 2 - 50, 10), "Guitar Tablature", fill='black', font=title_font)
        
        # Draw each line of the tab
        for line in range(num_lines):
            y_offset = 50 + line * line_height
            
            # Draw string lines
            for string in range(self.num_strings):
                y = y_offset + string * 15
                draw.line([(30, y), (width - 30, y)], fill='black', width=1)
                
                # Draw tuning label
                draw.text((10, y - 7), self.tuning[string], fill='black', font=font)
            
            # Draw measure dividers
            if show_grid:
                for measure in range(measures_per_line + 1):
                    x = 30 + measure * measure_width
                    draw.line([(x, y_offset - 10), (x, y_offset + self.num_strings * 15 + 10)], 
                             fill='black', width=1)
        
        # Draw notes
        for note in sorted_notes:
            # Calculate position in the tab
            measure = int(note.time_position / 4)
            position_in_measure = note.time_position % 4
            
            # Calculate the line and position in the image
            line = measure // measures_per_line
            measure_in_line = measure % measures_per_line
            
            # Calculate pixel position
            x = 30 + measure_in_line * measure_width + (position_in_measure / 4) * measure_width
            y = 50 + line * line_height + note.string * 15
            
            # Draw the fret number
            draw.text((x - 4, y - 7), str(note.fret), fill='black', font=font)
            
            # Draw technique markings
            if note.is_bend:
                draw.text((x + 8, y - 7), f"b{note.bend_value}", fill='blue', font=font)
            elif note.is_hammer_on:
                draw.text((x + 8, y - 7), "h", fill='green', font=font)
            elif note.is_pull_off:
                draw.text((x + 8, y - 7), "p", fill='red', font=font)
            elif note.is_slide:
                draw.text((x + 8, y - 7), "/", fill='purple', font=font)
        
        # Save the image if output path is provided
        if output_path:
            img.save(output_path)
        
        return img
    
    def render_tab_as_html(
        self, 
        notes: List[TabNote], 
        output_path: Optional[str] = None,
        include_playback: bool = True,
        midi_path: Optional[str] = None
    ) -> str:
        """
        Render tab notes as an interactive HTML visualization.
        
        Args:
            notes: List of TabNote objects
            output_path: Path to save the HTML file (optional)
            include_playback: Whether to include playback functionality
            midi_path: Path to a MIDI file for playback (optional)
            
        Returns:
            HTML string representation of the tablature
        """
        if not notes:
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Guitar Tab Visualization</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                </style>
            </head>
            <body>
                <h1>Guitar Tab Visualization</h1>
                <p>No notes to render</p>
            </body>
            </html>
            """
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(html)
            
            return html
        
        # Generate the ASCII tab representation
        ascii_tab = self.render_tab_as_text(notes)
        
        # Generate an image representation and convert to base64
        img = self.render_tab_as_image(notes)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Create the HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Guitar Tab Visualization</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .tab-container {{ margin-top: 20px; }}
                .tab-ascii {{ font-family: monospace; white-space: pre; margin-top: 20px; }}
                .tab-image {{ margin-top: 20px; }}
                .playback-controls {{ margin-top: 20px; display: {'block' if include_playback else 'none'}; }}
                button {{ padding: 8px 16px; margin-right: 10px; }}
            </style>
        </head>
        <body>
            <h1>Guitar Tab Visualization</h1>
            
            <div class="tab-container">
                <h2>Graphical Tab</h2>
                <div class="tab-image">
                    <img src="data:image/png;base64,{img_str}" alt="Guitar Tab Visualization">
                </div>
                
                <h2>ASCII Tab</h2>
                <div class="tab-ascii">
{ascii_tab}
                </div>
            </div>
        """
        
        # Add playback controls if requested
        if include_playback and midi_path:
            # Create base64 representation of the MIDI file
            with open(midi_path, 'rb') as f:
                midi_data = f.read()
            midi_b64 = base64.b64encode(midi_data).decode()
            
            html += f"""
            <div class="playback-controls">
                <h2>Playback</h2>
                <button id="play-btn">Play</button>
                <button id="stop-btn">Stop</button>
                
                <script src="https://cdn.jsdelivr.net/npm/jsmidgen@0.1.5/lib/jsmidgen.js"></script>
                <script>
                    // MIDI playback functionality
                    const midiB64 = "{midi_b64}";
                    const midiData = atob(midiB64);
                    let midiPlayer = null;
                    
                    document.getElementById('play-btn').addEventListener('click', function() {{
                        if (midiPlayer) {{
                            midiPlayer.resume();
                        }} else {{
                            // Convert base64 MIDI to ArrayBuffer
                            const byteNumbers = new Array(midiData.length);
                            for (let i = 0; i < midiData.length; i++) {{
                                byteNumbers[i] = midiData.charCodeAt(i);
                            }}
                            const byteArray = new Uint8Array(byteNumbers);
                            
                            // Load and play MIDI
                            const blob = new Blob([byteArray], {{type: 'audio/midi'}});
                            const url = URL.createObjectURL(blob);
                            
                            const audio = new Audio(url);
                            audio.play();
                            midiPlayer = audio;
                        }}
                    }});
                    
                    document.getElementById('stop-btn').addEventListener('click', function() {{
                        if (midiPlayer) {{
                            midiPlayer.pause();
                            midiPlayer.currentTime = 0;
                        }}
                    }});
                </script>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        # Save the HTML if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
        
        return html


def visualize_tab(
    tab_file_path: str, 
    output_format: str = 'text',
    output_path: Optional[str] = None,
    midi_path: Optional[str] = None,
    **kwargs
) -> Union[str, Image.Image]:
    """
    Visualize a tab file in various formats.
    
    Args:
        tab_file_path: Path to the tab file
        output_format: Format to render ('text', 'image', or 'html')
        output_path: Path to save the output (optional)
        midi_path: Path to MIDI file for HTML playback (optional)
        **kwargs: Additional arguments for the specific renderer
        
    Returns:
        Rendered tab in the specified format
    """
    # Create a visualizer - filter out midi_path from kwargs
    visualizer_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['midi_path']}
    visualizer = TabVisualizer(**visualizer_kwargs)
    
    # Parse the tab file
    notes = visualizer.parse_tab_file(tab_file_path)
    
    # Render in the requested format
    if output_format == 'text':
        return visualizer.render_tab_as_text(notes, **kwargs)
    elif output_format == 'image':
        return visualizer.render_tab_as_image(notes, output_path=output_path, **kwargs)
    elif output_format == 'html':
        # Only pass midi_path to render_tab_as_html
        return visualizer.render_tab_as_html(notes, output_path=output_path, 
                                          midi_path=midi_path, **kwargs)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def compare_tabs(
    generated_tab_path: str, 
    reference_tab_path: str,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Create a visual comparison of a generated tab against a reference tab.
    
    Args:
        generated_tab_path: Path to the generated tab file
        reference_tab_path: Path to the reference tab file
        output_path: Path to save the comparison image (optional)
        
    Returns:
        PIL Image showing the comparison
    """
    # Create a visualizer
    visualizer = TabVisualizer()
    
    # Parse the tab files
    generated_notes = visualizer.parse_tab_file(generated_tab_path)
    reference_notes = visualizer.parse_tab_file(reference_tab_path)
    
    # Render both tabs as images
    gen_img = visualizer.render_tab_as_image(generated_notes)
    ref_img = visualizer.render_tab_as_image(reference_notes)
    
    # Create a comparison image
    width = max(gen_img.width, ref_img.width)
    height = gen_img.height + ref_img.height + 40  # Add space for labels
    
    comparison = Image.new('RGB', (width, height), color='white')
    
    # Add labels and images
    draw = ImageDraw.Draw(comparison)
    
    try:
        if visualizer.font_path:
            font = ImageFont.truetype(visualizer.font_path, 16)
        else:
            font = ImageFont.load_default()
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Add reference tab
    draw.text((10, 10), "Reference Tab", fill='black', font=font)
    comparison.paste(ref_img, (0, 30))
    
    # Add generated tab
    draw.text((10, ref_img.height + 30), "Generated Tab", fill='black', font=font)
    comparison.paste(gen_img, (0, ref_img.height + 50))
    
    # Save the comparison if output path is provided
    if output_path:
        comparison.save(output_path)
    
    return comparison