"""
Format guitar fingering paths as ASCII tablature.

This module handles the conversion of optimized fingering paths to readable
guitar tablature format.
"""

from typing import List
from .optimizer import FretPosition
from .extractor import MidiNote, midi_number_to_note_name


class TablatureFormatter:
    """
    Formats guitar fingering as ASCII tablature.
    """
    
    def __init__(self, width: int = 80):
        """
        Initialize the formatter.
        
        Args:
            width: Character width for each line of tablature
        """
        self.width = width
        self.string_names = ['e', 'B', 'G', 'D', 'A', 'E']  # High to low
    
    def format(self, path: List[FretPosition], midi_sequence: List[MidiNote] = None, clean: bool = False) -> str:
        """
        Format fingering path as ASCII guitar tablature.
        
        Args:
            path: List of FretPosition objects
            midi_sequence: Optional list of MidiNote objects for metadata
            clean: If True, return only the tab without headers/footers
            
        Returns:
            ASCII tablature string
        """
        if not path:
            return "No tablature to display."
        
        num_strings = 6
        
        # Build tablature as one continuous line (horizontal)
        tab_lines = [f"{self.string_names[i]}|" for i in range(num_strings)]
        
        for idx, pos in enumerate(path):
            # Convert string index (0=low E) to display index (0=high e)
            display_string = num_strings - 1 - pos.string
            
            fret_str = str(pos.fret)
            
            # Add to appropriate string line
            for i in range(num_strings):
                if i == display_string:
                    tab_lines[i] += f"{fret_str:>2}-"
                else:
                    tab_lines[i] += "---"
        
        # Close all lines
        for i in range(num_strings):
            tab_lines[i] += "|"
        
        tablature = "\n".join(tab_lines)
        
        # Return clean version if requested
        if clean:
            return tablature
        
        # Add header
        header = "=" * self.width + "\n"
        header += "GUITAR TABLATURE (Optimized via Dijkstra's Algorithm)\n"
        header += "=" * self.width + "\n\n"
        
        # Add footer with statistics
        footer = "\n\n" + "=" * self.width + "\n"
        footer += f"Total notes: {len(path)}\n"
        
        if len(path) > 1:
            total_fret_distance = sum(abs(path[i].fret - path[i+1].fret) 
                                     for i in range(len(path) - 1))
            total_string_jumps = sum(abs(path[i].string - path[i+1].string) 
                                    for i in range(len(path) - 1))
            avg_fret_distance = total_fret_distance / (len(path) - 1)
            avg_string_jumps = total_string_jumps / (len(path) - 1)
            
            footer += f"Average fret movement: {avg_fret_distance:.2f}\n"
            footer += f"Average string jumps: {avg_string_jumps:.2f}\n"
        
        footer += "=" * self.width
        
        return header + tablature + footer
    
    def format_detailed(self, path: List[FretPosition], 
                       midi_sequence: List[MidiNote]) -> str:
        """
        Format with detailed note information.
        
        Args:
            path: List of FretPosition objects
            midi_sequence: List of MidiNote objects
            
        Returns:
            Detailed tablature with note names and timing
        """
        output = self.format(path, midi_sequence)
        
        # Add detailed note list
        output += "\n\nDetailed Note Information:\n"
        output += "-" * self.width + "\n"
        
        string_names_full = ['E (low)', 'A', 'D', 'G', 'B', 'E (high)']
        
        for i, (note, pos) in enumerate(zip(midi_sequence, path)):
            note_name = midi_number_to_note_name(note.midi_number)
            duration = note.offset - note.onset
            output += f"{i+1:3d}. {note_name:>4} | "
            output += f"String {pos.string + 1} ({string_names_full[pos.string]:>8}), "
            output += f"Fret {pos.fret:>2} | "
            output += f"Time: {note.onset:.2f}-{note.offset:.2f}s ({duration:.2f}s)\n"
        
        return output
    
    def save(self, path: List[FretPosition], output_path: str,
             midi_sequence: List[MidiNote] = None, detailed: bool = False) -> None:
        """
        Save tablature to file.
        
        Args:
            path: List of FretPosition objects
            output_path: Path to save tablature
            midi_sequence: Optional list of MidiNote objects
            detailed: Whether to include detailed information
        """
        if detailed and midi_sequence:
            content = self.format_detailed(path, midi_sequence)
        else:
            content = self.format(path, midi_sequence)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"Saved tablature to: {output_path}")


def print_tablature(path: List[FretPosition], midi_sequence: List[MidiNote] = None) -> None:
    """
    Convenience function to print tablature to console.
    
    Args:
        path: List of FretPosition objects
        midi_sequence: Optional list of MidiNote objects
    """
    formatter = TablatureFormatter()
    print(formatter.format(path, midi_sequence))
