"""
Tab Generator module for AxTone.
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
        
        # Define guitar type profiles for electric and acoustic guitars
        self.guitar_profiles = {
            'electric': {
                'name': 'Electric Guitar',
                'strings': 6,
                # Standard tuning for electric guitar
                'tuning': ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'],
                # Typical characteristics for an electric guitar
                'max_frets': 24,
                'techniques': ['bend', 'vibrato', 'hammer-on', 'pull-off', 'slide', 'tapping'],
                'preferences': {
                    'fret_preference': 'economy',  # Prefer economical fingerings
                    'position_shifts': 'minimize',  # Minimize position shifts
                    'string_preference': 'natural'  # Added string preference
                }
            },
            'acoustic': {
                'name': 'Acoustic Guitar',
                'strings': 6,
                # Standard tuning for acoustic guitar
                'tuning': ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'],
                # Typical characteristics for an acoustic guitar
                'max_frets': 20,
                'techniques': ['hammer-on', 'pull-off', 'slide'],
                'preferences': {
                    'fret_preference': 'lower',  # Prefer lower frets for acoustic
                    'position_shifts': 'natural',  # Allow more natural position shifts
                    'string_preference': 'natural'  # Added string preference
                }
            }
        }
        
        # Default to standard guitar
        self.current_guitar_type = 'acoustic'
        
        # Initialize note to string/fret mapping for standard tuning
        self.note_to_string_fret = self._initialize_note_mapping()
        
    def configure_for_guitar_type(self, guitar_type: str):
        """
        Configure the tab generator for a specific guitar type.
        
        Args:
            guitar_type: Type of guitar ('electric' or 'acoustic')
            
        Returns:
            self: Returns self for method chaining
        """
        if guitar_type not in self.guitar_profiles:
            raise ValueError(f"Unsupported guitar type: {guitar_type}")
            
        self.current_guitar_type = guitar_type
        profile = self.guitar_profiles[guitar_type]
        
        # Update the instrument configuration with the selected profile
        self.instrument_config.update({
            'name': profile['name'],
            'strings': profile['strings'],
            'tuning': profile['tuning'],
            'max_frets': profile['max_frets']
        })
        
        logger.info(f"TabGenerator configured for {profile['name']}")
        
        # Update note to string/fret mapping for the new guitar type
        self.note_to_string_fret = self._initialize_note_mapping()
        
        # Return the updated configuration for chaining
        return self
    
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
        logger.info(f"Generating tablature from audio features for {self.current_guitar_type} guitar")
        
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
        
        # Get preferences specific to the current guitar type
        preferences = self.guitar_profiles[self.current_guitar_type]['preferences']
        fret_preference = preferences['fret_preference']
        string_preference = preferences['string_preference']
        
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
                    
                    # Find best string and fret, taking into account guitar type preferences
                    string, fret = self._find_best_string_and_fret(
                        midi_note, 
                        tuning_midi,
                        fret_preference,
                        string_preference
                    )
                    
                    if string is not None:
                        # For electric guitar, we might add more techniques
                        techniques = {}
                        
                        if self.current_guitar_type == 'electric':
                            # Occasionally add vibrato for electric guitar
                            if fret > 5 and np.random.random() < 0.2:
                                techniques['vibrato'] = True
                            
                            # Occasionally add bend for electric guitar high notes
                            if fret > 12 and string < 3 and np.random.random() < 0.15:
                                techniques['bend'] = 0.5  # 1/4 step bend
                        
                        note = Note(
                            string=string,
                            fret=fret,
                            start_time=onset_time,
                            duration=end_time - onset_time,
                            **techniques
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
    
    def _find_best_string_and_fret(self, midi_note, tuning_midi, fret_preference='economy', string_preference='natural'):
        """
        Find the best string and fret to play a given MIDI note.
        
        Args:
            midi_note: MIDI note number
            tuning_midi: List of MIDI note numbers for each string
            fret_preference: Strategy for fret selection ('lower', 'economy', 'position')
            string_preference: Strategy for string selection ('natural', 'higher', 'lower')
            
        Returns:
            tuple: (string_index, fret_number)
        """
        best_string = None
        best_fret = None
        best_score = float('inf')
        
        # Get max frets from the current instrument config
        max_frets = self.instrument_config.get('max_frets', 24)
        
        for string, open_note in enumerate(tuning_midi):
            # Check if the note can be played on this string
            if midi_note >= open_note:
                fret = int(round(midi_note - open_note))
                
                # Check if fret is within range for this guitar type
                if fret > max_frets:
                    continue
                
                # Calculate score based on preferences
                score = 0
                
                # Fret preference scoring
                if fret_preference == 'lower':
                    # Strongly prefer lower frets (acoustic guitar)
                    score += fret * 1.5
                elif fret_preference == 'economy':
                    # Balanced approach for electric guitar
                    score += fret * 0.8
                elif fret_preference == 'position':
                    # Position-based playing (for solos)
                    # Prefer notes in position (e.g., around fret 12)
                    target_position = 12
                    score += abs(fret - target_position) * 0.5
                
                # String preference scoring
                if string_preference == 'higher':
                    # Prefer higher strings (thinner) for lead parts
                    score += (5 - string) * 0.2  # Reverse string order (string 0 is highest)
                elif string_preference == 'lower':
                    # Prefer lower strings (thicker) for rhythm parts
                    score += string * 0.2
                elif string_preference == 'natural':
                    # Balanced approach for acoustic
                    # Slightly prefer middle strings
                    score += abs(string - 2) * 0.1
                
                # Update best if this is better
                if score < best_score:
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
        logger.info(f"Exporting tab in {format} format for {self.current_guitar_type} guitar")
        
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
        
        # Add header for guitar type
        header = f"Guitar Type: {self.current_guitar_type.capitalize()}\n"
        header += f"Tuning: {', '.join(self.instrument_config['tuning'])}\n\n"
        
        # Place each note on the appropriate string
        for segment in segments:
            for note in segment.notes:
                string_idx = note.string
                
                # Handle techniques
                fret_str = str(note.fret)
                if note.bend:
                    fret_str += "b"  # Mark bends with 'b'
                if note.vibrato:
                    fret_str += "~"  # Mark vibrato with '~'
                if note.slide_to is not None:
                    fret_str += "/"  # Mark slides with '/'
                if note.hammer_on:
                    fret_str += "h"  # Mark hammer-ons with 'h'
                if note.pull_off:
                    fret_str += "p"  # Mark pull-offs with 'p'
                
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
        tab_txt = header + '\n'.join(tab_lines)
        
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
    
    def generate_tab_from_notes(self, notes):
        """
        Generate tablature from a list of note dictionaries.
        
        Args:
            notes: List of note dictionaries with pitch, start, end, and velocity
            
        Returns:
            List of measures, each containing a list of note positions
        """
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda x: x['start'])
        
        # Group notes into measures (assuming 4/4 time signature)
        measure_duration = 4.0  # 4 beats per measure
        measures = []
        current_measure = {'notes': []}
        
        # Get preferences for the current guitar type
        preferences = self.guitar_profiles[self.current_guitar_type]['preferences']
        fret_preference = preferences['fret_preference']
        string_preference = preferences['string_preference']
        
        for note in sorted_notes:
            # Calculate which measure this note belongs to
            measure_idx = int(note['start'] / measure_duration)
            
            # Ensure we have enough measures
            while len(measures) <= measure_idx:
                # If we have a non-empty current measure, add it to measures
                if current_measure['notes']:
                    measures.append(current_measure)
                    current_measure = {'notes': []}
            
            # Find the best string/fret position for this note
            midi_note = note['pitch']
            
            # Look up in our mapping
            if midi_note in self.note_to_string_fret:
                string, fret = self.note_to_string_fret[midi_note]
            else:
                # If not in mapping, find closest match
                closest_pitch = min(
                    self.note_to_string_fret.keys(),
                    key=lambda p: abs(p - midi_note)
                )
                string, fret = self.note_to_string_fret[closest_pitch]
            
            # Calculate relative time within the measure
            relative_time = note['start'] % measure_duration
            
            # Apply guitar-specific techniques
            techniques = {}
            
            # Electric guitars have more technique options
            if self.current_guitar_type == 'electric':
                # Higher chance for techniques on higher frets
                if fret > 12 and note['velocity'] > 100:
                    # Possibly add vibrato for held notes
                    if note['end'] - note['start'] > 0.5 and np.random.random() < 0.3:
                        techniques['vibrato'] = True
                        
                    # Possibly add bend for higher strings
                    if string < 3 and np.random.random() < 0.2:
                        techniques['bend'] = np.random.choice([0.5, 1.0])  # 1/4 or 1/2 step bend
            
            # Add note to the current measure with any techniques
            note_entry = {
                'string': string,
                'fret': fret,
                'time': relative_time,
                'duration': note['end'] - note['start']
            }
            
            # Add techniques if any
            note_entry.update(techniques)
            
            current_measure['notes'].append(note_entry)
        
        # Add the last measure if it's not empty
        if current_measure['notes']:
            measures.append(current_measure)
        
        # Add guitar type information to each measure
        for measure in measures:
            measure['guitar_type'] = self.current_guitar_type
        
        return measures
    
    def _initialize_note_mapping(self):
        """
        Initialize the mapping of notes to string and fret positions for the current tuning.
        
        Returns:
            Dictionary mapping MIDI note numbers to tuples of (string, fret)
        """
        note_map = {}
        
        # Get the current tuning based on guitar type
        tuning = self.guitar_profiles[self.current_guitar_type]['tuning']
        max_frets = self.guitar_profiles[self.current_guitar_type]['max_frets']
        
        # Convert tuning to MIDI note numbers
        tuning_midi = [self._note_name_to_midi(note) for note in tuning]
        
        # For each string
        for string_idx, open_note in enumerate(tuning_midi):
            # For each fret position on this string
            for fret in range(max_frets + 1):  # Include open string (fret 0)
                midi_note = open_note + fret
                
                # Store the optimal string/fret position for this note
                # If multiple positions are possible, prefer lower frets
                if midi_note not in note_map or note_map[midi_note][1] > fret:
                    note_map[midi_note] = (string_idx, fret)
        
        return note_map