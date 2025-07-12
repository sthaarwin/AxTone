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
from src.utils.source_separation import GuitarSourceSeparator

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
        os.makedirs(os.path.join(self.tab_output_path, 'electric'), exist_ok=True)
        os.makedirs(os.path.join(self.tab_output_path, 'acoustic'), exist_ok=True)
        
        # Initialize the TabGenerator
        self.tab_generator = TabGenerator(config)
        
        # Initialize the source separator with method from config
        self.source_separation_method = config.get('audio', {}).get('source_separation', {}).get('method', 'spectral')
        self.source_separator = GuitarSourceSeparator(method=self.source_separation_method)
        
        logger.info(f"Initialized TabAIPipeline with {self.source_separation_method} source separation")
    
    def detect_guitar_type(self, audio_data: Tuple[np.ndarray, int], file_path: str = None) -> str:
        """
        Detect if the audio is from electric or acoustic guitar based on spectral features.
        
        Args:
            audio_data: Tuple of (audio_data, sample_rate)
            file_path: Path to the audio file (used for path-based detection)
            
        Returns:
            String indicating guitar type: 'electric' or 'acoustic'
        """
        # First try to detect based on file path if available
        if file_path:
            path_lower = file_path.lower()
            if 'electric' in path_lower:
                return 'electric'
            elif 'acoustic' in path_lower:
                return 'acoustic'
            
        # If path-based detection fails, use spectral features
        y, sr = audio_data
        
        # Extract spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Higher spectral centroid and zero crossing rate typically indicate electric guitar
        # due to the presence of more high-frequency content and distortion
        if np.mean(spectral_centroid) > 3000 and np.mean(zcr) > 0.1:
            return 'electric'
        else:
            return 'acoustic'
    
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
        
        # Detect guitar type (electric or acoustic)
        guitar_type = self.detect_guitar_type(audio_data, input_file_path)
        logger.info(f"Detected guitar type: {guitar_type}")
        
        # 2. Preprocess audio
        filename = os.path.basename(input_file_path)
        processed_audio = self.preprocess_audio(audio_data, filename, guitar_type)
        
        # 3. Extract features
        features = self.extract_features(processed_audio, guitar_type)
        
        # 4. Generate MIDI
        midi_path = self.generate_midi(features, os.path.splitext(filename)[0])
        
        # 5. Convert MIDI to tab notation
        if output_path is None:
            # Save to the appropriate guitar type folder
            output_path = os.path.join(
                self.tab_output_path,
                guitar_type,
                f"{os.path.splitext(filename)[0]}.tab"
            )
        
        tab_path = self.midi_to_tab(midi_path, output_path, guitar_type)
        
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
    
    def preprocess_audio(self, audio_data: Tuple[np.ndarray, int], filename: str, guitar_type: str = None) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio data with source separation, noise reduction and normalization.
        
        Args:
            audio_data: Tuple of (audio_data, sample_rate)
            filename: Original filename
            guitar_type: Type of guitar ('electric' or 'acoustic')
            
        Returns:
            Tuple of (processed_audio_data, sample_rate)
        """
        y, sr = audio_data
        
        # First save the original audio to a temporary file for source separation
        temp_file = os.path.join(self.processed_stems_path, f"{os.path.splitext(filename)[0]}_temp.wav")
        sf.write(temp_file, y, sr)
        
        # Apply source separation if enabled
        use_source_separation = self.config.get('audio', {}).get('source_separation', {}).get('enabled', True)
        
        if use_source_separation:
            logger.info(f"Applying {self.source_separation_method} source separation")
            # Use our source separator to isolate the guitar
            guitar_path = self.source_separator.separate(temp_file, self.processed_stems_path)
            
            # Load the separated guitar audio
            y_guitar, sr = librosa.load(guitar_path, sr=None)
            
            # Remove temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            logger.info(f"Guitar track isolated: {guitar_path}")
            y = y_guitar
        
        # Apply standard preprocessing
        # 1. Normalize audio
        y_normalized = librosa.util.normalize(y)
        
        # 2. Apply appropriate preprocessing based on guitar type
        if guitar_type == 'electric':
            # For electric guitars, apply more aggressive high-pass filtering
            # to emphasize the bright character and reduce low-end rumble
            y_filtered = librosa.effects.preemphasis(y_normalized, coef=0.97)
        else:
            # For acoustic guitars, use gentler filtering to preserve the natural tone
            y_filtered = librosa.effects.preemphasis(y_normalized, coef=0.95)
        
        # Save processed audio
        # Include guitar type in the filename for better organization
        output_filename = f"{os.path.splitext(filename)[0]}_{guitar_type}_processed.wav"
        output_path = os.path.join(self.processed_stems_path, output_filename)
        sf.write(output_path, y_filtered, sr)
        
        logger.info(f"Preprocessed audio saved to {output_path}")
        return (y_filtered, sr)
    
    def extract_features(self, audio_data: Tuple[np.ndarray, int], guitar_type: str = None) -> Dict:
        """
        Extract features from audio data.
        
        Args:
            audio_data: Tuple of (audio_data, sample_rate)
            guitar_type: Type of guitar ('electric' or 'acoustic')
            
        Returns:
            Dictionary of features (mel_spectrogram, chromagram, mfccs)
        """
        y, sr = audio_data
        
        file_features = {}
        
        # Add guitar type to features for model training
        file_features['guitar_type'] = guitar_type
        
        # 1. Mel Spectrogram - adjust parameters based on guitar type
        if guitar_type == 'electric':
            # For electric guitars, use more mel bands to capture the harmonics
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        else:
            # For acoustic guitars, fewer mel bands but more focus on lower frequencies
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=6000)
            
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
        # Include guitar type in the feature path for organized training data
        if guitar_type:
            feature_path = os.path.join(self.features_path, guitar_type, base_filename)
        else:
            feature_path = os.path.join(self.features_path, base_filename)
            
        os.makedirs(feature_path, exist_ok=True)
        
        for feature_name, feature_data in file_features.items():
            if feature_name != 'guitar_type':  # Don't try to save string as numpy array
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
        
        # Get guitar type if available
        guitar_type = features.get('guitar_type', '')
        
        # Create the output filename (include guitar type if available)
        if guitar_type:
            output_path = os.path.join(self.midi_output_path, f"{base_filename}_{guitar_type}.mid")
        else:
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
    
    def midi_to_tab(self, midi_path: str, output_path: str, guitar_type: str = None) -> str:
        """
        Convert MIDI file to tablature notation.
        
        Args:
            midi_path: Path to the MIDI file
            output_path: Path for the output tab file
            guitar_type: Type of guitar ('electric' or 'acoustic')
            
        Returns:
            Path to the generated tab file
        """
        # Import pretty_midi without the deprecation warning
        logger.info(f"Starting MIDI to tab conversion for {midi_path}")
        pretty_midi = import_pretty_midi()
        
        # Load the MIDI file
        logger.info(f"Loading MIDI file: {midi_path}")
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Extract notes from the MIDI file
        logger.info(f"Extracting notes from MIDI file")
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
        
        original_note_count = len(notes)
        logger.info(f"Extracted {original_note_count} notes from MIDI file")
        
        # Apply note filtering and quantization to reduce computational load
        notes = self._filter_and_quantize_notes(notes)
        logger.info(f"After filtering and quantization: {len(notes)} notes to process (reduced by {original_note_count - len(notes)} notes)")
        
        # Configure the TabGenerator based on guitar type
        if guitar_type == 'electric':
            # For electric guitar, we might use a different tuning or string config
            self.tab_generator.configure_for_guitar_type('electric')
        elif guitar_type == 'acoustic':
            # For acoustic guitar, we might use standard tuning
            self.tab_generator.configure_for_guitar_type('acoustic')
        
        # Use the TabGenerator to create tablature
        logger.info(f"Generating tab notation from {len(notes)} notes")
        raw_tab = self.tab_generator.generate_tab_from_notes(notes)
        
        logger.info(f"Generated raw tab with {len(raw_tab)} measures")
        
        # Convert the dictionary-based tab format to TabSegment objects
        logger.info(f"Converting raw tab to TabSegment objects")
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
        
        logger.info(f"Created {len(segments)} TabSegment objects")
        
        # Export the tab to the desired format
        logger.info(f"Exporting tab to {output_path}")
        self.tab_generator.export_tab(segments, format='txt', output_path=output_path)
        
        # Generate PNG visualization of the tab
        png_output_path = output_path.replace('.txt', '.png').replace('.tab', '.png')
        try:
            from src.visualization.tab_visualizer import TabVisualizer
            
            # Convert TabSegment objects to TabNote objects for visualization
            tab_notes = []
            for i, segment in enumerate(segments):
                for note in segment.notes:
                    tab_notes.append({
                        'string': note.string,
                        'fret': note.fret,
                        'time_position': segment.start_time * 10 + i * 40,  # Scale time for visualization
                        'duration': note.duration,
                        'is_bend': note.bend is not None,
                        'is_hammer_on': note.hammer_on,
                        'is_pull_off': note.pull_off,
                        'is_slide': note.slide_to is not None,
                        'bend_value': note.bend if note.bend else 0.0
                    })
            
            # Initialize the tab visualizer with the correct tuning
            tuning = self.tab_generator.instrument_config['tuning']
            num_strings = self.tab_generator.instrument_config['strings']
            visualizer = TabVisualizer(num_strings=num_strings, tuning=tuning)
            
            # Render the tab as an image
            logger.info(f"Generating PNG visualization: {png_output_path}")
            visualizer.render_tab_as_image(tab_notes, output_path=png_output_path)
            
            # Also generate a JSON representation for potential further processing
            json_output_path = output_path.replace('.txt', '.json').replace('.tab', '.json')
            with open(json_output_path, 'w') as f:
                import json
                json.dump([{
                    'string': note['string'],
                    'fret': note['fret'],
                    'time': note['time_position'],
                    'duration': note['duration']
                } for note in tab_notes], f, indent=2)
            
            logger.info(f"Tab visualizations generated: {png_output_path} and {json_output_path}")
        except Exception as e:
            logger.error(f"Error generating tab visualization: {e}")
        
        logger.info(f"Tab file generated: {output_path}")
        return output_path
    
    def _filter_and_quantize_notes(self, notes):
        """
        Filter and quantize notes to reduce computational load
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            Filtered and quantized list of notes
        """
        # If we have fewer than 500 notes, no need to filter
        if len(notes) < 500:
            return notes
            
        # Sort by start time first
        notes = sorted(notes, key=lambda x: x['start'])
        
        # 1. Quantize timing to reduce the number of unique time positions
        # Define a grid size (e.g., 16th notes at 120bpm)
        grid_size = 0.125  # in seconds
        
        for note in notes:
            # Quantize start and end times
            note['start'] = round(note['start'] / grid_size) * grid_size
            note['end'] = round(note['end'] / grid_size) * grid_size
            
            # Ensure minimum duration
            if note['end'] <= note['start']:
                note['end'] = note['start'] + grid_size
        
        # 2. Remove near-duplicate notes (same pitch at very similar times)
        filtered_notes = []
        note_groups = {}  # Group by quantized start time and pitch
        
        for note in notes:
            key = (note['start'], note['pitch'])
            if key not in note_groups:
                note_groups[key] = []
            note_groups[key].append(note)
        
        # For each group, keep only the note with highest velocity
        for group in note_groups.values():
            if group:
                # Keep the note with highest velocity
                best_note = max(group, key=lambda x: x['velocity'])
                filtered_notes.append(best_note)
        
        # 3. If still too many notes, perform additional filtering for very dense sections
        if len(filtered_notes) > 1000:
            # Find regions with very high note density and thin them out
            filtered_notes = self._thin_dense_regions(filtered_notes)
            
        return filtered_notes
    
    def _thin_dense_regions(self, notes):
        """
        Thin out regions with very high note density
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            Thinned list of notes
        """
        # Find the total duration of the piece
        if not notes:
            return notes
            
        min_time = min(note['start'] for note in notes)
        max_time = max(note['end'] for note in notes)
        duration = max_time - min_time
        
        # If the piece is shorter than 10 seconds, no need to thin
        if duration < 10:
            return notes
        
        # Divide the piece into 1-second segments and count notes in each
        segment_size = 1.0  # 1 second
        segments = {}
        
        for note in notes:
            segment_idx = int((note['start'] - min_time) / segment_size)
            if segment_idx not in segments:
                segments[segment_idx] = []
            segments[segment_idx].append(note)
        
        # Find the average number of notes per segment
        avg_notes_per_segment = len(notes) / (duration / segment_size)
        
        # For segments with more than 2x the average, keep only the most important notes
        thinned_notes = []
        
        for segment_idx, segment_notes in segments.items():
            if len(segment_notes) > 2 * avg_notes_per_segment and len(segment_notes) > 20:
                # Too many notes in this segment, thin it out
                # Sort by velocity (importance)
                segment_notes.sort(key=lambda x: x['velocity'], reverse=True)
                # Keep only the top 50% of notes by velocity, or at least 20 notes
                keep_count = max(20, len(segment_notes) // 2)
                thinned_notes.extend(segment_notes[:keep_count])
            else:
                # This segment is fine, keep all notes
                thinned_notes.extend(segment_notes)
                
        return thinned_notes
    
    def process_batch(self, input_dir: str, output_dir: str = None) -> List[str]:
        """
        Process a batch of audio files.
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save output files (optional)
            
        Returns:
            List of paths to generated tab files
        """
        output_files = []
        
        # Check if the input directory is for a specific guitar type
        guitar_type = None
        if 'electric' in input_dir.lower():
            guitar_type = 'electric'
        elif 'acoustic' in input_dir.lower():
            guitar_type = 'acoustic'
        
        # Set the output directory based on guitar type if not specified
        if output_dir is None and guitar_type:
            output_dir = os.path.join(self.tab_output_path, guitar_type)
        elif output_dir is None:
            output_dir = self.tab_output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if filename.endswith(('.wav', '.mp3', '.ogg')):
                input_path = os.path.join(input_dir, filename)
                
                # If we know the guitar type, include it in the output filename
                if guitar_type:
                    output_path = os.path.join(
                        output_dir, 
                        f"{os.path.splitext(filename)[0]}_{guitar_type}.tab"
                    )
                else:
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