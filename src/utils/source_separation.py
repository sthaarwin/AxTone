"""
Source separation utilities for AxTone.

This module provides functions to isolate guitar tracks from mixed audio
using various source separation techniques including:
1. Spectral masking approaches
2. Deep learning models (Demucs, Spleeter)
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class GuitarSourceSeparator:
    """
    Class to handle source separation specifically optimized for guitar tracks.
    """
    def __init__(self, method="spectral", device=None):
        """
        Initialize the source separator.
        
        Args:
            method (str): Method to use for separation.
                Options: 'spectral', 'demucs', 'spleeter'
            device (str): Device to use for processing. 
                Options: 'cuda', 'cpu', or None (auto-detect)
        """
        self.method = method
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize specific models based on method
        if method == "demucs":
            try:
                import demucs.separate
                self.has_demucs = True
                logger.info("Loaded Demucs for source separation")
            except ImportError:
                logger.warning("Demucs not found. Install with 'pip install demucs'")
                self.has_demucs = False
                self.method = "spectral"  # Fallback to spectral method
                
        elif method == "spleeter":
            try:
                import tensorflow as tf
                from spleeter.separator import Separator
                self.has_spleeter = True
                # Create separator - 4stems separates vocals, drums, bass, and other
                self.separator = Separator('spleeter:4stems')
                logger.info("Loaded Spleeter for source separation")
            except ImportError:
                logger.warning("Spleeter not found. Install with 'pip install spleeter'")
                self.has_spleeter = False
                self.method = "spectral"  # Fallback to spectral method
        
        # Spectral method doesn't need any special imports
        logger.info(f"Source separator initialized with {self.method} method on {self.device}")
    
    def separate(self, input_path, output_dir=None):
        """
        Separate guitar track from mixed audio.
        
        Args:
            input_path (str): Path to input audio file
            output_dir (str): Directory to save output files
                If None, will use the same directory as the input
                
        Returns:
            str: Path to the separated guitar audio file
        """
        # Create output directory if needed
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_guitar.wav")
        
        # Run separation using the selected method
        if self.method == "demucs" and self.has_demucs:
            return self._separate_demucs(input_path, output_dir)
        elif self.method == "spleeter" and self.has_spleeter:
            return self._separate_spleeter(input_path, output_dir)
        else:
            logger.info(f"Using spectral method for separation.")
            return self._separate_spectral(input_path, output_path)
    
    def _separate_spectral(self, input_path, output_path):
        """
        Separate using spectral masking techniques.
        
        Args:
            input_path (str): Path to input audio
            output_path (str): Path to output audio
            
        Returns:
            str: Path to isolated guitar audio
        """
        logger.info(f"Applying spectral separation to {input_path}")
        
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None, mono=True)
            
            # Calculate spectrogram
            n_fft = 2048
            D = librosa.stft(y, n_fft=n_fft)
            D_mag, D_phase = librosa.magphase(D)
            
            # Apply harmonic-percussive source separation
            # This will isolate harmonic content (typically includes guitar)
            D_harmonic, D_percussive = librosa.decompose.hpss(D_mag)
            
            # Filter to enhance guitar frequency range (typically 80Hz-1.2kHz)
            # Create a simple filter to boost guitar frequencies
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Create a mask with the same shape as the spectrogram
            guitar_mask = np.ones_like(D_harmonic)
            
            # Find bin indices for our frequency ranges
            low_freq_bins = np.where(freq_bins < 80)[0]
            high_freq_bins = np.where(freq_bins > 5000)[0]
            
            # Apply mask to specific frequency ranges
            if len(low_freq_bins) > 0:
                guitar_mask[low_freq_bins, :] *= 0.2
                
            if len(high_freq_bins) > 0:
                guitar_mask[high_freq_bins, :] *= 0.3
            
            # Apply mask to harmonic content
            D_guitar = D_harmonic * guitar_mask
            
            # Reconstruct audio
            y_guitar = librosa.istft(D_guitar * D_phase)
            
            # Normalize audio
            y_guitar = librosa.util.normalize(y_guitar)
            
            # Save the isolated guitar track
            sf.write(output_path, y_guitar, sr)
            
            logger.info(f"Saved spectral separated guitar to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error during spectral separation: {e}")
            # If we can't do spectral separation, just copy the original file
            logger.warning(f"Falling back to original audio file")
            import shutil
            shutil.copy(input_path, output_path)
            return output_path
    
    def _separate_demucs(self, input_path, output_dir):
        """
        Separate using Demucs deep learning model.
        
        Args:
            input_path (str): Path to input audio
            output_dir (str): Directory for output files
            
        Returns:
            str: Path to isolated guitar audio
        """
        import demucs.separate
        
        logger.info(f"Applying Demucs separation to {input_path}")
        
        # Run separation - Demucs will create a separated folder
        demucs.separate.main(["--out", output_dir, "--name", "htdemucs", input_path])
        
        # Find the 'other' stem which usually contains guitar
        # For htdemucs model, it should be in htdemucs/track_name/other.wav
        basename = os.path.splitext(os.path.basename(input_path))[0]
        guitar_path = os.path.join(output_dir, "htdemucs", basename, "other.wav")
        
        # Rename to our standard format
        output_path = os.path.join(output_dir, f"{basename}_guitar.wav")
        if os.path.exists(guitar_path):
            os.rename(guitar_path, output_path)
        else:
            logger.error(f"Demucs separation failed to create {guitar_path}")
            return self._separate_spectral(input_path, output_path)
        
        logger.info(f"Saved Demucs separated guitar to {output_path}")
        return output_path
    
    def _separate_spleeter(self, input_path, output_dir):
        """
        Separate using Spleeter deep learning model.
        
        Args:
            input_path (str): Path to input audio
            output_dir (str): Directory for output files
            
        Returns:
            str: Path to isolated guitar audio
        """
        logger.info(f"Applying Spleeter separation to {input_path}")
        
        # Run separation
        self.separator.separate_to_file(
            input_path, 
            output_dir
        )
        
        # Spleeter creates a folder with the input filename
        basename = os.path.splitext(os.path.basename(input_path))[0]
        # Guitars are typically in the 'other' stem
        other_path = os.path.join(output_dir, basename, "other.wav")
        
        # Rename to our standard format
        output_path = os.path.join(output_dir, f"{basename}_guitar.wav")
        if os.path.exists(other_path):
            os.rename(other_path, output_path)
        else:
            logger.error(f"Spleeter separation failed to create {other_path}")
            return self._separate_spectral(input_path, output_path)
        
        logger.info(f"Saved Spleeter separated guitar to {output_path}")
        return output_path


def isolate_guitar(input_path, output_dir=None, method="spectral"):
    """
    Convenience function to isolate guitar from a mixed audio file.
    
    Args:
        input_path (str): Path to input audio file
        output_dir (str): Directory to save output files
            If None, will use the same directory as the input
        method (str): Method to use for separation
            Options: 'spectral', 'demucs', 'spleeter'
            
    Returns:
        str: Path to the separated guitar audio file
    """
    separator = GuitarSourceSeparator(method=method)
    return separator.separate(input_path, output_dir)