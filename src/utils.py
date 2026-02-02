"""
Utility functions for audio processing and analysis.
"""

import numpy as np
import librosa
from typing import Tuple, Optional
import os


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio signal
        target_db: Target dB level
        
    Returns:
        Normalized audio
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    # Calculate target RMS
    target_rms = 10 ** (target_db / 20)
    
    # Scale audio
    scaling_factor = target_rms / rms
    normalized = audio * scaling_factor
    
    # Clip to prevent distortion
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def trim_silence(audio: np.ndarray, sr: int, 
                 top_db: float = 30.0) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        top_db: Threshold in dB below reference to consider as silence
        
    Returns:
        Trimmed audio
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def preprocess_audio(audio_path: str, 
                     output_path: Optional[str] = None,
                     sr: int = 22050,
                     normalize: bool = True,
                     trim: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to input audio file
        output_path: Optional path to save preprocessed audio
        sr: Target sample rate
        normalize: Whether to normalize audio
        trim: Whether to trim silence
        
    Returns:
        (preprocessed_audio, sample_rate)
    """
    print(f"Preprocessing: {audio_path}")
    
    # Load audio
    audio, sample_rate = librosa.load(audio_path, sr=sr, mono=True)
    print(f"  Loaded: {len(audio)} samples at {sample_rate} Hz ({len(audio)/sample_rate:.2f}s)")
    
    # Trim silence
    if trim:
        audio = trim_silence(audio, sample_rate)
        print(f"  Trimmed: {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")
    
    # Normalize
    if normalize:
        audio = normalize_audio(audio)
        print(f"  Normalized to -20 dB")
    
    # Save if output path provided
    if output_path:
        import soundfile as sf
        sf.write(output_path, audio, sample_rate)
        print(f"  Saved to: {output_path}")
    
    return audio, sample_rate


def get_audio_stats(audio: np.ndarray, sr: int) -> dict:
    """
    Calculate audio statistics.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'duration': len(audio) / sr,
        'sample_rate': sr,
        'num_samples': len(audio),
        'rms': np.sqrt(np.mean(audio ** 2)),
        'peak': np.max(np.abs(audio)),
        'zero_crossing_rate': np.mean(librosa.zero_crossings(audio)),
    }
    
    # Calculate spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    stats['spectral_centroid_mean'] = np.mean(spectral_centroids)
    stats['spectral_centroid_std'] = np.std(spectral_centroids)
    
    return stats


def validate_audio_file(audio_path: str) -> bool:
    """
    Check if audio file is valid and readable.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return False
    
    # Check file extension
    valid_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    _, ext = os.path.splitext(audio_path)
    
    if ext.lower() not in valid_extensions:
        print(f"Warning: Unusual file extension: {ext}")
        print(f"Supported: {', '.join(valid_extensions)}")
    
    # Try to load
    try:
        audio, sr = librosa.load(audio_path, duration=1.0)  # Load 1 second to test
        if len(audio) == 0:
            print("Error: Audio file is empty")
            return False
        print(f"✓ Audio file valid: {audio_path}")
        return True
    except Exception as e:
        print(f"Error loading audio: {e}")
        return False


def convert_mp3_to_wav(mp3_path: str, wav_path: str, sr: int = 22050) -> bool:
    """
    Convert MP3 to WAV format.
    
    Args:
        mp3_path: Path to input MP3 file
        wav_path: Path to output WAV file
        sr: Target sample rate
        
    Returns:
        True if successful
    """
    try:
        import soundfile as sf
        
        # Load MP3
        audio, sample_rate = librosa.load(mp3_path, sr=sr)
        
        # Save as WAV
        sf.write(wav_path, audio, sample_rate)
        
        print(f"Converted: {mp3_path} -> {wav_path}")
        return True
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False


def analyze_pitch_range(audio_path: str) -> dict:
    """
    Analyze the pitch range of an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with pitch statistics
    """
    audio, sr = librosa.load(audio_path)
    
    # Extract pitch
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    
    # Filter out unvoiced segments
    f0_voiced = f0[voiced_flag]
    f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
    
    if len(f0_voiced) == 0:
        return {'error': 'No pitched content detected'}
    
    # Convert to MIDI
    midi_notes = librosa.hz_to_midi(f0_voiced)
    
    from .extractor import midi_number_to_note_name
    
    return {
        'min_freq': np.min(f0_voiced),
        'max_freq': np.max(f0_voiced),
        'mean_freq': np.mean(f0_voiced),
        'min_midi': int(np.min(midi_notes)),
        'max_midi': int(np.max(midi_notes)),
        'min_note': midi_number_to_note_name(int(np.min(midi_notes))),
        'max_note': midi_number_to_note_name(int(np.max(midi_notes))),
        'range_semitones': int(np.max(midi_notes) - np.min(midi_notes))
    }


# Example usage
if __name__ == "__main__":
    print("Audio utilities loaded!")
    print("Available functions:")
    print("  - normalize_audio()")
    print("  - trim_silence()")
    print("  - preprocess_audio()")
    print("  - get_audio_stats()")
    print("  - validate_audio_file()")
    print("  - convert_mp3_to_wav()")
    print("  - analyze_pitch_range()")
