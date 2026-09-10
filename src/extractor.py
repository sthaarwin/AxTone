"""
Audio to MIDI extraction using Basic Pitch and Librosa.

This module handles the conversion of audio files (.mp3, .wav) to MIDI note sequences.
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    warnings.warn("Basic Pitch not available. Install with: pip install basic-pitch")

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False
    warnings.warn("Pretty MIDI not available. Install with: pip install pretty-midi")


@dataclass
class MidiNote:
    """Represents a MIDI note with timing information."""
    midi_number: int  # MIDI note number (e.g., 60 = C4)
    onset: float      # Start time in seconds
    offset: float     # End time in seconds
    velocity: int = 80  # Note velocity (0-127)
    
    def __repr__(self):
        return f"MidiNote(midi={self.midi_number}, onset={self.onset:.3f}, offset={self.offset:.3f})"


class AudioExtractor:
    """
    Extract MIDI notes from audio files using pitch detection.
    
    Supports multiple backends:
    - Basic Pitch (ML-based, most accurate)
    - Librosa pYIN (traditional DSP)
    """
    
    def __init__(self, method: str = 'basic_pitch', min_note_duration: float = 0.1):
        """
        Initialize the audio extractor.
        
        Args:
            method: Extraction method ('basic_pitch' or 'pyin')
            min_note_duration: Minimum note duration in seconds
        """
        self.method = method
        self.min_note_duration = min_note_duration
        
        if method == 'basic_pitch' and not BASIC_PITCH_AVAILABLE:
            raise ImportError("Basic Pitch is not installed. Use: pip install basic-pitch")
    
    def load_audio(self, audio_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
        """
        Load audio file and normalize.
        
        Args:
            audio_path: Path to audio file (.mp3, .wav, etc.)
            sr: Target sample rate
            
        Returns:
            (audio_data, sample_rate)
        """
        print(f"Loading audio: {audio_path}")
        audio, sample_rate = librosa.load(audio_path, sr=sr, mono=True)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return audio, sample_rate
    
    def extract_with_basic_pitch(self, audio_path: str) -> List[MidiNote]:
        """
        Extract MIDI notes using Basic Pitch (ML-based).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of MidiNote objects
        """
        if not BASIC_PITCH_AVAILABLE:
            raise ImportError("Basic Pitch is not installed")
        
        print("Extracting MIDI with Basic Pitch...")
        
        # Run Basic Pitch inference
        model_output, midi_data, note_events = predict(
            audio_path,
            ICASSP_2022_MODEL_PATH
        )
        
        # Convert to MidiNote objects
        notes = []
        for start_time, end_time, pitch, velocity, _ in note_events:
            if end_time - start_time >= self.min_note_duration:
                notes.append(MidiNote(
                    midi_number=int(pitch),
                    onset=start_time,
                    offset=end_time,
                    velocity=int(velocity * 127)
                ))
        
        # Sort by onset time
        notes.sort(key=lambda n: n.onset)
        
        print(f"Extracted {len(notes)} notes")
        return notes
    
    def extract_with_pyin(self, audio_path: str) -> List[MidiNote]:
        """
        Extract MIDI notes using pYIN pitch detection.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of MidiNote objects
        """
        print("Extracting MIDI with pYIN...")
        
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Extract pitch using pYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # Lowest guitar note
            fmax=librosa.note_to_hz('C6'),  # Highest common guitar note
            sr=sr
        )
        
        # Convert to MIDI notes
        notes = self._f0_to_midi_notes(f0, voiced_flag, sr)
        
        print(f"Extracted {len(notes)} notes")
        return notes
    
    def _f0_to_midi_notes(self, f0: np.ndarray, voiced_flag: np.ndarray, 
                          sr: int) -> List[MidiNote]:
        """
        Convert fundamental frequency estimates to MIDI notes.
        
        Args:
            f0: Fundamental frequency array
            voiced_flag: Boolean array indicating voiced segments
            sr: Sample rate
            
        Returns:
            List of MidiNote objects
        """
        notes = []
        hop_length = 512
        frame_duration = hop_length / sr
        
        # Find note segments
        current_note = None
        note_start = None
        
        for i, (freq, is_voiced) in enumerate(zip(f0, voiced_flag)):
            if is_voiced and not np.isnan(freq):
                midi_num = librosa.hz_to_midi(freq)
                midi_rounded = int(np.round(midi_num))
                
                if current_note is None:
                    # Start new note
                    current_note = midi_rounded
                    note_start = i * frame_duration
                elif abs(midi_rounded - current_note) > 0.5:
                    # Pitch changed, end previous note and start new one
                    note_end = i * frame_duration
                    if note_end - note_start >= self.min_note_duration:
                        notes.append(MidiNote(
                            midi_number=current_note,
                            onset=note_start,
                            offset=note_end
                        ))
                    current_note = midi_rounded
                    note_start = i * frame_duration
            else:
                # Unvoiced segment
                if current_note is not None:
                    note_end = i * frame_duration
                    if note_end - note_start >= self.min_note_duration:
                        notes.append(MidiNote(
                            midi_number=current_note,
                            onset=note_start,
                            offset=note_end
                        ))
                    current_note = None
                    note_start = None
        
        # Close final note
        if current_note is not None and note_start is not None:
            note_end = len(f0) * frame_duration
            if note_end - note_start >= self.min_note_duration:
                notes.append(MidiNote(
                    midi_number=current_note,
                    onset=note_start,
                    offset=note_end
                ))
        
        return notes
    
    def extract(self, audio_path: str) -> List[MidiNote]:
        """
        Extract MIDI notes from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of MidiNote objects sorted by onset time
        """
        if self.method == 'basic_pitch':
            return self.extract_with_basic_pitch(audio_path)
        elif self.method == 'pyin':
            return self.extract_with_pyin(audio_path)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
    
    def save_midi(self, notes: List[MidiNote], output_path: str) -> None:
        """
        Save MIDI notes to a .mid file. First tries pretty_midi, then falls back to mido.
        """
        if PRETTY_MIDI_AVAILABLE:
            midi = pretty_midi.PrettyMIDI()
            guitar = pretty_midi.Instrument(program=24)
            for note in notes:
                midi_note = pretty_midi.Note(
                    velocity=note.velocity, pitch=note.midi_number,
                    start=note.onset, end=note.offset
                )
                guitar.notes.append(midi_note)
            midi.instruments.append(guitar)
            midi.write(output_path)
            print(f"Saved MIDI to: {output_path} (using pretty_midi)")
            return

        # Fallback: Use mido (already in requirements.txt)
        import mido
        mid = mido.MidiFile(type=0)
        mid.ticks_per_beat = 480
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Initial meta messages
        track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        track.append(mido.Message('program_change', program=24, time=0, channel=0))

        # Build list of events
        events = []
        for note in notes:
            start_ticks = int(note.onset * 960)
            end_ticks = int(note.offset * 960)
            events.append((start_ticks, 'note_on', note.midi_number, note.velocity))
            events.append((end_ticks, 'note_off', note.midi_number, 0))

        # Sort by absolute ticks, then convert to delta time
        events.sort(key=lambda x: x[0])
        last_tick = 0
        for abs_tick, msg_type, pitch, vel in events:
            delta = abs_tick - last_tick
            track.append(mido.Message(msg_type, note=pitch, velocity=vel, time=delta, channel=0))
            last_tick = abs_tick

        mid.save(output_path)
        print(f"Saved MIDI to: {output_path} (using mido)")

def consolidate_notes(
    notes: List[MidiNote],
    min_duration: float = 0.08,
    merge_gap: float = 0.05,
) -> List[MidiNote]:
    """
    Remove micro-notes and merge near-consecutive same-pitch notes.
    """
    if not notes: return notes
    notes = sorted(notes, key=lambda n: n.onset)

    # 1: remove tiny notes
    notes = [n for n in notes if (n.offset - n.onset) >= min_duration]
    if not notes: return notes

    # 2: merge consecutive notes of same pitch
    merged: List[MidiNote] = []
    current = notes[0]

    for nxt in notes[1:]:
        same_pitch = nxt.midi_number == current.midi_number
        small_gap  = (nxt.onset - current.offset) <= merge_gap

        if same_pitch and small_gap:
            current = MidiNote(
                midi_number=current.midi_number,
                onset=current.onset,
                offset=nxt.offset,
                velocity=max(current.velocity, nxt.velocity),
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def midi_number_to_note_name(midi_number: int) -> str:
    """Convert MIDI note number to note name (e.g., 60 -> C4)."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]
    return f"{note}{octave}"


# Example usage
if __name__ == "__main__":
    # Test with a sample file
    extractor = AudioExtractor(method='pyin')
    
    # This would require an actual audio file
    # notes = extractor.extract("path/to/vocal.mp3")
    # extractor.save_midi(notes, "output.mid")
    
    print("Audio extractor ready!")
    print(f"Basic Pitch available: {BASIC_PITCH_AVAILABLE}")
    print(f"Pretty MIDI available: {PRETTY_MIDI_AVAILABLE}")
