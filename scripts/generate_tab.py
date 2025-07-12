"""
Inference script for generating guitar tab from audio files.

This script loads a trained TabTranscriptionModel and generates
tablature from input audio files.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import librosa
import matplotlib.pyplot as plt

from src.core.ai.models.tab_models import TabTranscriptionModel
from src.core.tab_generator import TabGenerator, Note, TabSegment
from src.data.dataset import GuitarSetDataset
from src.evaluation.metrics import evaluate_tab

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> TabTranscriptionModel:
    """
    Load a trained TabTranscriptionModel from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint
        device: PyTorch device to load the model to
        
    Returns:
        Loaded model
    """
    # Load checkpoint with weights_only=False to handle the pickle error
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model configuration
    config = checkpoint.get('config', {})
    
    # Create model with same parameters
    model = TabTranscriptionModel(
        input_channels=6,  # 6 strings for guitar
        mel_bins=config.get('mel_bins', 128),
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        max_fret=config.get('max_fret', 24)
    )
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model


def preprocess_audio(
    audio_path: str,
    sr: int = 44100,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 128,
    mono: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Preprocess an audio file for tab generation.
    
    Args:
        audio_path: Path to the audio file
        sr: Sample rate
        hop_length: Hop length for feature extraction
        n_fft: FFT size
        n_mels: Number of mel bins
        mono: Whether to convert audio to mono
        
    Returns:
        Dictionary of preprocessed features
    """
    logger.info(f"Preprocessing audio file: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, mono=mono)
    
    # If we only have mono audio, create a simulated hexaphonic signal
    # by duplicating the mono signal to 6 channels
    if mono or (isinstance(y, np.ndarray) and len(y.shape) == 1):
        # For real inference, we don't have true hexaphonic data
        # So we just duplicate the mono signal to all 6 strings
        y = np.tile(y, (6, 1))
    # If we have stereo audio (2 channels), convert to 6 channels
    elif isinstance(y, np.ndarray) and y.shape[0] == 2:
        # Convert stereo to mono first
        y_mono = np.mean(y, axis=0)
        
        # Apply some audio enhancements to better isolate guitar sounds
        # 1. Apply a bandpass filter to focus on guitar frequency range (80Hz-1.2kHz)
        y_filtered = librosa.effects.trim(y_mono, top_db=20)[0]  # Trim silence
        y_filtered = librosa.effects.harmonic(y_filtered)  # Extract harmonic component
        
        # Apply bandpass filter for guitar frequency range
        y_filtered = librosa.effects.preemphasis(y_filtered)  # Boost higher frequencies
        
        # 2. Normalize the audio
        y_filtered = librosa.util.normalize(y_filtered)
        
        # Then duplicate to 6 channels
        y = np.tile(y_filtered, (6, 1))
        
        # Log some debug info
        logger.info(f"Audio duration: {len(y_filtered)/sr:.2f} seconds")
        logger.info(f"Audio shape after processing: {y.shape}")
    
    # Make sure we have 6 channels (one per string)
    if y.shape[0] != 6:
        raise ValueError(f"Expected 6 audio channels, got {y.shape[0]}")
    
    # Extract mel spectrograms for each string
    mel_specs = []
    for string_idx in range(6):
        string_audio = y[string_idx]
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=string_audio, 
            sr=sr, 
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_specs.append(mel_spec_db)
    
    # Stack mel spectrograms from all strings
    mel_spectrograms = np.stack(mel_specs)
    
    # Convert to PyTorch tensor and add batch dimension
    mel_spectrograms_tensor = torch.FloatTensor(mel_spectrograms).unsqueeze(0)
    
    return {
        'mel_spectrograms': mel_spectrograms_tensor,
        'audio': y,
        'sample_rate': sr,
        'hop_length': hop_length
    }


def postprocess_predictions(
    string_probs: torch.Tensor,
    fret_probs: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    threshold: float = 0.5,
    min_note_duration: float = 0.05  # reduced from 0.1 to 0.05 seconds
) -> List[Dict]:
    """
    Convert model predictions to a list of notes with timing information.
    
    Args:
        string_probs: String activation probabilities [batch, time, strings]
        fret_probs: Fret position probabilities [batch, time, strings, frets]
        hop_length: Hop length used for feature extraction
        sample_rate: Audio sample rate
        threshold: Threshold for string activation
        min_note_duration: Minimum note duration in seconds
        
    Returns:
        List of note dictionaries with timing, string, and fret information
    """
    # Get string and fret predictions
    string_preds = (string_probs > threshold).cpu().numpy()[0]  # [time, strings]
    fret_preds = torch.argmax(fret_probs, dim=3).cpu().numpy()[0]  # [time, strings]
    
    # Log prediction stats for debugging
    active_strings = np.sum(string_preds)
    total_frames = string_preds.shape[0] * string_preds.shape[1]
    activation_percentage = (active_strings / total_frames) * 100
    logger.info(f"Active string frames: {active_strings}/{total_frames} ({activation_percentage:.2f}%)")
    
    # Convert frame indices to time in seconds
    frame_times = np.arange(string_preds.shape[0]) * hop_length / sample_rate
    
    # Initialize list to store notes
    notes = []
    
    # For each string, find segments where the string is played
    for string_idx in range(string_preds.shape[1]):
        # Get string activation for this string
        string_activation = string_preds[:, string_idx]
        
        # Count activations for this string
        string_active_frames = np.sum(string_activation)
        if string_active_frames > 0:
            logger.info(f"String {string_idx+1} has {string_active_frames} active frames")
        
        # Find segments where string is played (consecutive frames)
        segments = []
        current_segment = None
        
        for frame_idx, is_active in enumerate(string_activation):
            if is_active:
                # String is played in this frame
                if current_segment is None:
                    # Start a new segment
                    current_segment = {
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'frets': [fret_preds[frame_idx, string_idx]]
                    }
                else:
                    # Extend the current segment
                    current_segment['end_frame'] = frame_idx
                    current_segment['frets'].append(fret_preds[frame_idx, string_idx])
            else:
                # String is not played in this frame
                if current_segment is not None:
                    # End the current segment
                    segments.append(current_segment)
                    current_segment = None
        
        # Add the last segment if it exists
        if current_segment is not None:
            segments.append(current_segment)
        
        # Convert segments to notes
        for segment in segments:
            start_time = frame_times[segment['start_frame']]
            end_time = frame_times[segment['end_frame']]
            duration = end_time - start_time
            
            # Skip very short notes
            if duration < min_note_duration:
                continue
            
            # Get the most common fret in this segment
            frets = segment['frets']
            fret = max(set(frets), key=frets.count)
            
            # Create a note
            note = {
                'string': string_idx,
                'fret': fret,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            }
            
            notes.append(note)
    
    # Sort notes by start time
    notes = sorted(notes, key=lambda x: x['start_time'])
    
    return notes


def render_tab(
    notes: List[Dict],
    num_strings: int = 6,
    resolution: float = 0.25,  # Time resolution in seconds
    max_duration: Optional[float] = None
) -> str:
    """
    Render notes as ASCII tab notation.
    
    Args:
        notes: List of note dictionaries
        num_strings: Number of strings on the guitar
        resolution: Time resolution for rendering (seconds per column)
        max_duration: Maximum duration to render (None = use all notes)
        
    Returns:
        ASCII tab notation as string
    """
    if not notes:
        return "No notes detected."
    
    # Determine the duration of the tab
    if max_duration is None:
        max_time = max(note['end_time'] for note in notes)
    else:
        max_time = max_duration
    
    # Calculate the number of columns needed
    num_columns = int(max_time / resolution) + 1
    
    # Initialize the tab as a 2D array
    tab = [['-' for _ in range(num_columns)] for _ in range(num_strings)]
    
    # Add the standard guitar tuning header
    tuning = ['E', 'B', 'G', 'D', 'A', 'E']
    
    # Place each note in the tab
    for note in notes:
        string_idx = note['string']
        fret = note['fret']
        start_idx = int(note['start_time'] / resolution)
        
        # Skip notes that fall outside the tab
        if start_idx >= num_columns:
            continue
        
        # Convert fret number to string representation
        fret_str = str(fret)
        
        # Place the fret number in the tab
        if start_idx < num_columns:
            # Handle multi-digit fret numbers
            if fret >= 10:
                if start_idx + 1 < num_columns:
                    tab[string_idx][start_idx] = fret_str[0]
                    tab[string_idx][start_idx + 1] = fret_str[1]
            else:
                tab[string_idx][start_idx] = fret_str
    
    # Convert the tab to a string
    tab_lines = []
    
    # Add tuning header
    for i, string in enumerate(tuning):
        tab_lines.append(f"{string}|-" + ''.join(tab[i]))
    
    return '\n'.join(tab_lines)


def save_tab(tab: str, output_path: str):
    """
    Save tab to a text file.
    
    Args:
        tab: Tab string
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        f.write(tab)
    
    logger.info(f"Tab saved to {output_path}")


def export_json(notes: List[Dict], output_path: str):
    """
    Export notes as JSON.
    
    Args:
        notes: List of note dictionaries
        output_path: Output file path
    """
    # Create a JSON-serializable structure
    # Convert any NumPy types to Python native types
    serializable_notes = []
    for note in notes:
        serializable_note = {}
        for key, value in note.items():
            # Convert NumPy types to Python native types
            if isinstance(value, np.integer):
                serializable_note[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_note[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_note[key] = value.tolist()
            else:
                serializable_note[key] = value
        serializable_notes.append(serializable_note)
    
    data = {
        'notes': serializable_notes
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Notes exported to {output_path}")


def generate_tab(args):
    """Generate tablature from audio files."""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Process each input file
    for input_path in args.input_files:
        logger.info(f"Processing file: {input_path}")
        
        # Get base filename for output
        basename = os.path.splitext(os.path.basename(input_path))[0]
        
        # Determine output paths
        if args.output_dir:
            output_dir = Path(args.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            tab_path = output_dir / f"{basename}.txt"
            json_path = output_dir / f"{basename}.json"
        else:
            input_dir = os.path.dirname(input_path)
            tab_path = os.path.join(input_dir, f"{basename}.txt")
            json_path = os.path.join(input_dir, f"{basename}.json")
        
        # Preprocess audio
        features = preprocess_audio(
            input_path,
            sr=args.sample_rate,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            n_mels=args.mel_bins,
            mono=args.mono
        )
        
        # Run inference
        with torch.no_grad():
            mel_spectrograms = features['mel_spectrograms'].to(device)
            predictions = model.predict(mel_spectrograms)
            
            string_probs = predictions['string_probs']
            fret_probs = predictions['fret_probs']
        
        # Postprocess predictions to get notes
        notes = postprocess_predictions(
            string_probs,
            fret_probs,
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            threshold=args.threshold,
            min_note_duration=args.min_note_duration
        )
        
        # Render the tab
        tab = render_tab(
            notes,
            num_strings=6,
            resolution=args.resolution
        )
        
        # Save the tab
        save_tab(tab, tab_path)
        
        # Export notes as JSON if requested
        if args.export_json:
            export_json(notes, json_path)
        
        # Generate visualization if requested
        if args.visualize:
            # Convert to TabGenerator format
            tab_notes = [
                Note(
                    string=note['string'],
                    fret=note['fret'],
                    start_time=note['start_time'],
                    duration=note['duration']
                )
                for note in notes
            ]
            
            tab_segment = TabSegment(
                notes=tab_notes,
                start_time=min(note['start_time'] for note in notes) if notes else 0,
                end_time=max(note['end_time'] for note in notes) if notes else 0
            )
            
            # Create a TabGenerator instance
            config = {'tab': {'instruments': [{'name': 'Guitar', 'strings': 6, 'tuning': ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']}]}}
            tab_generator = TabGenerator(config)
            
            # Determine the guitar type (default to acoustic)
            guitar_type = args.guitar_type
            if guitar_type in ('electric', 'acoustic'):
                tab_generator.configure_for_guitar_type(guitar_type)
            
            # Export the tab using the TabGenerator
            tab_txt = tab_generator.export_tab([tab_segment], format='txt', output_path=tab_path)
            
            # Also generate a visualization
            viz_path = os.path.splitext(tab_path)[0] + ".png"
            
            # Simple matplotlib visualization
            plt.figure(figsize=(12, 8))
            
            # Plot the notes
            for note in notes:
                string_idx = note['string']
                fret = note['fret']
                start_time = note['start_time']
                duration = note['duration']
                
                # Plot a rectangle for each note
                plt.fill_between(
                    [start_time, start_time + duration],
                    [string_idx - 0.4, string_idx - 0.4],
                    [string_idx + 0.4, string_idx + 0.4],
                    color='blue',
                    alpha=0.6
                )
                
                # Add fret number text
                plt.text(start_time + duration/2, string_idx, str(fret),
                         ha='center', va='center', color='white', fontweight='bold')
            
            # Customize the plot
            plt.yticks(range(6), ['E', 'B', 'G', 'D', 'A', 'E'])
            plt.xlabel('Time (seconds)')
            plt.ylabel('String')
            plt.title(f'Guitar Tab: {basename}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig(viz_path)
            plt.close()
            logger.info(f"Visualization saved to {viz_path}")
        
        logger.info(f"Generated tab for {basename}")


def main():
    """Main function to parse arguments and generate tabs."""
    parser = argparse.ArgumentParser(description='Generate guitar tabs from audio files')
    
    # Input/output arguments
    parser.add_argument('--input-files', nargs='+', required=True,
                        help='Path(s) to input audio file(s)')
    parser.add_argument('--model-path', required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files (default: same as input)')
    
    # Audio processing arguments
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--hop-length', type=int, default=512,
                        help='Hop length for feature extraction')
    parser.add_argument('--n-fft', type=int, default=2048,
                        help='FFT size for feature extraction')
    parser.add_argument('--mel-bins', type=int, default=128,
                        help='Number of mel frequency bins')
    parser.add_argument('--mono', action='store_true',
                        help='Process audio as mono (duplicate to all strings)')
    
    # Tab generation arguments
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for string activation')
    parser.add_argument('--min-note-duration', type=float, default=0.1,
                        help='Minimum note duration in seconds')
    parser.add_argument('--resolution', type=float, default=0.25,
                        help='Time resolution for tab rendering (seconds per column)')
    parser.add_argument('--guitar-type', type=str, default='acoustic',
                        choices=['acoustic', 'electric'],
                        help='Type of guitar (acoustic or electric)')
    
    # Output options
    parser.add_argument('--export-json', action='store_true',
                        help='Export notes as JSON')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of the tab')
    
    # Other arguments
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate tabs
    generate_tab(args)


if __name__ == '__main__':
    main()