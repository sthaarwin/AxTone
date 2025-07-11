"""
Tab Generation AI Pipeline Module

This module integrates the tab generation AI pipeline from the notebooks into
the core AxTone application. It provides a unified interface for transforming
audio files into guitar tablature using AI/ML techniques.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional

# Import other core modules
from src.core.tab_generator import TabGenerator, TabSegment, Note
from src.utils import import_pretty_midi

logger = logging.getLogger(__name__)

class TabAIPipeline:
    """
    Main class that orchestrates the AI-based tab generation pipeline.
    This implements the workflow from the tab_gen_ai_pipeline notebook.
    """
    
    def __init__(self, config):
        """
        Initialize the AI pipeline with configuration.
        
        Args:
            config: Configuration dictionary with paths and parameters
        """
        self.config = config
        
        # Set up paths
        self.raw_data_path = config.get('paths', {}).get('raw_data', 'data/raw/')
        self.processed_stems_path = config.get('paths', {}).get('processed_stems', 'data/processed/stems/')
        self.features_path = config.get('paths', {}).get('features', 'data/processed/features/')
        self.midi_output_path = config.get('paths', {}).get('midi_output', 'data/processed/midi/')
        self.tab_output_path = config.get('paths', {}).get('tab_output', 'data/outputs/')
        
        # Create directories if they don't exist
        os.makedirs(self.processed_stems_path, exist_ok=True)
        os.makedirs(self.features_path, exist_ok=True)
        os.makedirs(self.midi_output_path, exist_ok=True)
        os.makedirs(self.tab_output_path, exist_ok=True)
        
        # Initialize the TabGenerator
        self.tab_generator = TabGenerator(config)
        
        logger.info("Initialized TabAIPipeline")
    
    def process_file(self, input_file_path: str, output_path: Optional[str] = None) -> str:
        """
        Process a single audio file through the complete pipeline.
        
        Args:
            input_file_path: Path to the input audio file
            output_path: Path for the output tab file (optional)
            
        Returns:
            Path to the generated tab file
        """
        logger.info(f"Processing file: {input_file_path}")
        
        # 1. Load audio file
        audio_data = self.load_audio_file(input_file_path)
        if not audio_data:
            raise ValueError(f"Failed to load audio file: {input_file_path}")
        
        # 2. Preprocess audio
        filename = os.path.basename(input_file_path)
        processed_audio = self.preprocess_audio(audio_data, filename)
        
        # 3. Extract features
        features = self.extract_features(processed_audio)
        
        # 4. Generate MIDI
        midi_path = self.generate_midi(features, os.path.splitext(filename)[0])
        
        # 5. Convert MIDI to tab notation
        if output_path is None:
            output_path = os.path.join(
                self.tab_output_path, 
                f"{os.path.splitext(filename)[0]}.tab"
            )
        
        tab_path = self.midi_to_tab(midi_path, output_path)
        
        logger.info(f"Generated tab file: {tab_path}")
        return tab_path
    
    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file using librosa.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            y, sr = librosa.load(file_path, sr=None)  # Keep native sample rate
            logger.info(f"Loaded {file_path}: {len(y)/sr:.2f}s, {sr}Hz")
            return (y, sr)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def preprocess_audio(self, audio_data: Tuple[np.ndarray, int], filename: str) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio data with noise reduction and normalization.
        
        Args:
            audio_data: Tuple of (audio_data, sample_rate)
            filename: Original filename
            
        Returns:
            Tuple of (processed_audio_data, sample_rate)
        """
        y, sr = audio_data
        
        # 1. Normalize audio
        y_normalized = librosa.util.normalize(y)
        
        # 2. Simple noise reduction (high-pass filter to remove low frequency noise)
        y_filtered = librosa.effects.preemphasis(y_normalized)
        
        # Save processed audio
        output_filename = os.path.splitext(filename)[0] + "_processed.wav"
        output_path = os.path.join(self.processed_stems_path, output_filename)
        sf.write(output_path, y_filtered, sr)
        
        logger.info(f"Preprocessed audio saved to {output_path}")
        return (y_filtered, sr)
    
    def extract_features(self, audio_data: Tuple[np.ndarray, int]) -> Dict:
        """
        Extract features from audio data.
        
        Args:
            audio_data: Tuple of (audio_data, sample_rate)
            
        Returns:
            Dictionary of features (mel_spectrogram, chromagram, mfccs)
        """
        y, sr = audio_data
        
        file_features = {}
        
        # 1. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        file_features['mel_spectrogram'] = mel_spec_db
        
        # 2. Chromagram
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
        file_features['chromagram'] = chromagram
        
        # 3. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        file_features['mfccs'] = mfccs
        
        # Save features as numpy arrays
        base_filename = os.path.splitext(os.path.basename(self.processed_stems_path))[0]
        feature_path = os.path.join(self.features_path, base_filename)
        os.makedirs(feature_path, exist_ok=True)
        
        for feature_name, feature_data in file_features.items():
            np.save(os.path.join(feature_path, f"{feature_name}.npy"), feature_data)
        
        logger.info(f"Features extracted and saved to {feature_path}")
        return file_features
    
    def generate_midi(self, features: Dict, base_filename: str) -> str:
        """
        Generate MIDI file from extracted features.
        
        Args:
            features: Dictionary of features
            base_filename: Base filename for output
            
        Returns:
            Path to the generated MIDI file
        """
        from midiutil.MidiFile import MIDIFile
        
        # Create the output filename
        output_path = os.path.join(self.midi_output_path, f"{base_filename}.mid")
        
        # Create a MIDI file
        midi = MIDIFile(1)  # One track
        track = 0
        time = 0
        
        # Add track name and tempo
        midi.addTrackName(track, time, f"Generated from {base_filename}")
        midi.addTempo(track, time, 120)
        
        # Add notes based on the chromagram
        chromagram = features['chromagram']
        duration = 0.5  # in beats
        
        for t in range(0, chromagram.shape[1], 2):  # step by 2 frames for less density
            time = t / 4  # Convert frame to musical time
            
            # Find the most prominent notes at this time
            if t < chromagram.shape[1]:
                chroma_frame = chromagram[:, t]
                prominent_notes = np.where(chroma_frame > np.mean(chroma_frame))[0]
                
                # Add these notes to the MIDI file
                for note in prominent_notes:
                    pitch = 60 + note  # Middle C (60) + chroma bin
                    velocity = int(min(127, 50 + 77 * (chroma_frame[note] / np.max(chroma_frame))))
                    midi.addNote(track, 0, pitch, time, duration, velocity)
        
        # Write the MIDI file
        with open(output_path, 'wb') as output_file:
            midi.writeFile(output_file)
        
        logger.info(f"MIDI file generated: {output_path}")
        return output_path
    
    def midi_to_tab(self, midi_path: str, output_path: str) -> str:
        """
        Convert MIDI file to tablature notation.
        
        Args:
            midi_path: Path to the MIDI file
            output_path: Path for the output tab file
            
        Returns:
            Path to the generated tab file
        """
        # Import pretty_midi without the deprecation warning
        pretty_midi = import_pretty_midi()
        
        # Load the MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Extract notes from the MIDI file
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
        
        # Use the TabGenerator to create tablature
        raw_tab = self.tab_generator.generate_tab_from_notes(notes)
        
        # Convert the dictionary-based tab format to TabSegment objects
        segments = []
        for measure in raw_tab:
            measure_notes = []
            for note_dict in measure['notes']:
                measure_notes.append(Note(
                    string=note_dict['string'],
                    fret=note_dict['fret'],
                    start_time=note_dict['time'],
                    duration=note_dict['duration'],
                    velocity=1.0  # Default velocity
                ))
            
            # Create a TabSegment for this measure
            start_time = min(note.start_time for note in measure_notes) if measure_notes else 0
            end_time = max(note.start_time + note.duration for note in measure_notes) if measure_notes else 1
            
            segments.append(TabSegment(
                notes=measure_notes,
                start_time=start_time,
                end_time=end_time
            ))
        
        # Export the tab to the desired format
        self.tab_generator.export_tab(segments, format='txt', output_path=output_path)
        
        logger.info(f"Tab file generated: {output_path}")
        return output_path
    
    def process_batch(self, input_dir: str, output_dir: str) -> List[str]:
        """
        Process a batch of audio files.
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save output files
            
        Returns:
            List of paths to generated tab files
        """
        output_files = []
        
        for filename in os.listdir(input_dir):
            if filename.endswith(('.wav', '.mp3', '.ogg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(
                    output_dir, 
                    f"{os.path.splitext(filename)[0]}.tab"
                )
                
                try:
                    tab_path = self.process_file(input_path, output_path)
                    output_files.append(tab_path)
                except Exception as e:
                    logger.error(f"Error processing {input_path}: {e}")
        
        return output_files