"""
Tab generator module for Tab-Gen-AI.
Converts processed audio features into tablature notation.
"""

import logging
import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class Note:
    """Representation of a single note in a tab."""
    string: int  # Guitar string number (0-5 for standard 6-string)
    fret: int    # Fret number (0 = open string)
    start_time: float  # Start time in seconds
    duration: float    # Duration in seconds
    velocity: float = 1.0  # Relative volume/intensity (0.0-1.0)
    
    # Optional technique information
    bend: Optional[float] = None  # Bend amount in semitones
    vibrato: bool = False
    slide_to: Optional[int] = None  # Target fret if sliding
    hammer_on: bool = False
    pull_off: bool = False
    
    def __str__(self):
        """String representation of the note."""
        s = f"String {self.string+1}, Fret {self.fret}, Time: {self.start_time:.2f}s, Duration: {self.duration:.2f}s"
        techniques = []
        if self.bend:
            techniques.append(f"bend {self.bend}")
        if self.vibrato:
            techniques.append("vibrato")
        if self.slide_to is not None:
            techniques.append(f"slide to {self.slide_to}")
        if self.hammer_on:
            techniques.append("hammer-on")
        if self.pull_off:
            techniques.append("pull-off")
            
        if techniques:
            s += f" [{', '.join(techniques)}]"
        return s

@dataclass
class TabSegment:
    """A segment of tablature, typically a measure or phrase."""
    notes: List[Note]
    start_time: float
    end_time: float
    
    @property
    def duration(self):
        return self.end_time - self.start_time

class TabGenerator:
    """
    Converts audio features into tablature notation.
    """
    def __init__(self, config):
        """
        Initialize the tab generator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.instrument_config = self._get_instrument_config(config['tab']['instruments'][0])
        logger.info(f"Initialized TabGenerator for {self.instrument_config['name']}")
        
    def _get_instrument_config(self, instrument_config):
        """Get the configuration for the specified instrument."""
        return instrument_config
    
    def generate_tab(self, features):
        """
        Generate tablature from extracted audio features.
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            list: List of TabSegment objects representing the full tab
        """
        logger.info("Generating tablature from audio features")
        
        # Extract key information from features
        pitches = features['pitch']
        onsets = features['onsets']
        
        # Convert detected pitches to notes on the fretboard
        notes = self._assign_string_and_fret(pitches, onsets)
        
        # Group notes into segments (e.g., measures)
        segments = self._create_segments(notes)
        
        return segments
    
    def _assign_string_and_fret(self, pitches, onsets):
        """
        Assign detected pitches to specific strings and frets.
        
        This is a critical and complex part of tab generation that determines
        the most playable way to represent the detected notes.
        
        Args:
            pitches: Pitch information from audio features
            onsets: Onset information from audio features
            
        Returns:
            list: List of Note objects with string and fret assignments
        """
        logger.info("Assigning strings and frets")
        
        # This is a simplified implementation
        # A real implementation would use more sophisticated algorithms
        
        notes = []
        
        # Get tuning frequencies
        tuning = self.instrument_config['tuning']
        tuning_midi = [self._note_name_to_midi(note) for note in tuning]
        
        # Process detected pitches at each onset
        if 'f0' in pitches:  # pYIN output
            f0 = pitches['f0']
            voiced = pitches['voiced']
            
            # Convert f0 to MIDI note numbers
            midi_notes = librosa.hz_to_midi(f0)
            
            # For each onset, find the closest pitch
            for i, onset_time in enumerate(onsets['times']):
                if i < len(onsets['times']) - 1:
                    end_time = onsets['times'][i+1]
                else:
                    # For the last onset, estimate duration
                    end_time = onset_time + 0.5  # default duration
                
                # Find frame index for this onset
                frame_idx = librosa.time_to_frames(onset_time, sr=self.config['audio']['sample_rate'], 
                                                  hop_length=self.config['audio']['hop_length'])
                
                if frame_idx < len(midi_notes) and voiced[frame_idx]:
                    midi_note = midi_notes[frame_idx]
                    
                    # Find best string and fret
                    string, fret = self._find_best_string_and_fret(midi_note, tuning_midi)
                    
                    if string is not None:
                        note = Note(
                            string=string,
                            fret=fret,
                            start_time=onset_time,
                            duration=end_time - onset_time
                        )
                        notes.append(note)
        
        return notes
    
    def _note_name_to_midi(self, note_name):
        """Convert a note name (e.g., 'E2') to MIDI note number."""
        # This is a simple implementation
        note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        
        # Extract note and octave
        if len(note_name) == 2:
            note = note_name[0]
            octave = int(note_name[1])
        else:
            note = note_name[0:2]  # For sharps/flats
            octave = int(note_name[2])
            
        # Calculate MIDI note number
        midi_note = note_map[note] + (octave + 1) * 12
        return midi_note
    
    def _find_best_string_and_fret(self, midi_note, tuning_midi):
        """
        Find the best string and fret to play a given MIDI note.
        
        Args:
            midi_note: MIDI note number
            tuning_midi: List of MIDI note numbers for each string
            
        Returns:
            tuple: (string_index, fret_number)
        """
        best_string = None
        best_fret = None
        best_score = float('inf')
        
        for string, open_note in enumerate(tuning_midi):
            # Check if the note can be played on this string
            if midi_note >= open_note:
                fret = int(round(midi_note - open_note))
                
                # Simple scoring - prefer lower frets and higher strings
                # This is a very basic heuristic
                score = fret + 0.1 * string
                
                # Check if fret is within reasonable range (e.g., 0-24)
                if fret <= 24 and score < best_score:
                    best_string = string
                    best_fret = fret
                    best_score = score
        
        return best_string, best_fret
    
    def _create_segments(self, notes, segment_duration=4.0):
        """
        Group notes into logical segments (e.g., measures).
        
        Args:
            notes: List of Note objects
            segment_duration: Target duration for each segment in seconds
            
        Returns:
            list: List of TabSegment objects
        """
        if not notes:
            return []
            
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        
        # Find total duration
        start_time = sorted_notes[0].start_time
        end_time = max(n.start_time + n.duration for n in sorted_notes)
        
        # Create segments
        segments = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + segment_duration, end_time)
            
            # Get notes that fall within this segment
            segment_notes = [
                n for n in sorted_notes 
                if n.start_time >= current_start and n.start_time < current_end
            ]
            
            segments.append(TabSegment(
                notes=segment_notes,
                start_time=current_start,
                end_time=current_end
            ))
            
            current_start = current_end
            
        return segments
    
    def export_tab(self, segments, format='txt', output_path=None):
        """
        Export tablature to the specified format.
        
        Args:
            segments: List of TabSegment objects
            format: Output format ('txt', 'gp5', etc.)
            output_path: Path to save the output file
            
        Returns:
            str or bool: Tab content as string (for txt) or success flag
        """
        logger.info(f"Exporting tab in {format} format")
        
        if format == 'txt':
            return self._export_txt(segments, output_path)
        elif format == 'gp5':
            return self._export_gp5(segments, output_path)
        else:
            logger.error(f"Unsupported output format: {format}")
            return False
    
    def _export_txt(self, segments, output_path=None):
        """Export tab to text format."""
        num_strings = self.instrument_config['strings']
        
        # Create empty tab with the right number of strings
        tab_lines = ['-' * 80 for _ in range(num_strings)]
        
        # Place each note on the appropriate string
        for segment in segments:
            for note in segment.notes:
                string_idx = note.string
                fret_str = str(note.fret)
                
                # Find the right position based on start time
                # This is a very simplified approach
                position = int(note.start_time * 10) + 1
                
                # Make sure we have enough space
                if position >= len(tab_lines[0]):
                    for i in range(num_strings):
                        tab_lines[i] += '-' * (position - len(tab_lines[i]) + len(fret_str))
                
                # Place the fret number
                line = list(tab_lines[string_idx])
                for i, c in enumerate(fret_str):
                    line[position + i] = c
                tab_lines[string_idx] = ''.join(line)
        
        # Combine all strings into the final tab
        tab_txt = '\n'.join(tab_lines)
        
        # Write to file if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(tab_txt)
                
        return tab_txt
    
    def _export_gp5(self, segments, output_path):
        """
        Export tab to Guitar Pro 5 format.
        
        This is a placeholder - actual implementation would require
        a library like PyGuitarPro.
        """
        logger.warning("Guitar Pro export not fully implemented")
        return False