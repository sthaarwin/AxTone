"""
Core audio processing functionality for AxTone.
"""

import os
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles loading and processing audio files for feature extraction.
    """
    def __init__(self, config):
        """
        Initialize the audio processor with configuration.
        
        Args:
            config: Configuration dictionary containing audio processing parameters
        """
        self.config = config
        self.sr = config['audio']['sample_rate']
        self.hop_length = config['audio']['hop_length']
        self.n_fft = config['audio']['n_fft']
        self.fmin = config['audio']['fmin']
        self.fmax = config['audio']['fmax']
        
        logger.info(f"Initialized AudioProcessor with sample rate: {self.sr} Hz")
    
    def load_audio(self, file_path):
        """
        Load an audio file with the configured sample rate.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        logger.info(f"Loading audio file: {file_path}")
        try:
            y, sr = librosa.load(file_path, sr=self.sr)
            logger.info(f"Loaded audio: {len(y)/sr:.2f} seconds, {sr} Hz")
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise
    
    def extract_features(self, audio_data, sr):
        """
        Extract all relevant features from the audio data.
        
        Args:
            audio_data: Audio time series
            sr: Sample rate
            
        Returns:
            dict: Dictionary of extracted features
        """
        logger.info("Extracting features from audio")
        features = {}
        
        # Extract pitch information
        if self.config['features']['pitch_tracking']['algorithm'] == 'pyin':
            features['pitch'] = self._extract_pitch_pyin(audio_data, sr)
        else:
            features['pitch'] = self._extract_pitch_basic(audio_data, sr)
        
        # Extract onset information
        features['onsets'] = self._extract_onsets(audio_data, sr)
        
        # Extract harmonic content
        features['chroma'] = self._extract_chroma(audio_data, sr)
        
        # Extract spectral features
        features['spectrogram'] = self._extract_spectrogram(audio_data, sr)
        
        return features
    
    def _extract_pitch_basic(self, y, sr):
        """Basic pitch extraction using librosa's piptrack."""
        logger.debug("Extracting basic pitch information")
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr,
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        return {'pitches': pitches, 'magnitudes': magnitudes}
    
    def _extract_pitch_pyin(self, y, sr):
        """More accurate pitch extraction using pYIN algorithm."""
        logger.debug("Extracting pitch using pYIN algorithm")
        # Note: This is a placeholder. Actual pYIN implementation would be more complex.
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr
        )
        return {'f0': f0, 'voiced': voiced_flag, 'probabilities': voiced_probs}
    
    def _extract_onsets(self, y, sr):
        """Extract note onsets from audio."""
        logger.debug("Extracting note onsets")
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr,
            hop_length=self.hop_length
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=sr,
            hop_length=self.hop_length,
            backtrack=True,
            threshold=self.config['features']['onset_detection']['threshold']
        )
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=self.hop_length)
        return {'frames': onsets, 'times': onset_times, 'envelope': onset_env}
    
    def _extract_chroma(self, y, sr):
        """Extract chromagram for harmonic analysis."""
        logger.debug("Extracting chromagram")
        chroma = librosa.feature.chroma_cqt(
            y=y, 
            sr=sr,
            hop_length=self.hop_length,
            fmin=self.fmin
        )
        return chroma
    
    def _extract_spectrogram(self, y, sr):
        """Extract mel spectrogram for general spectral features."""
        logger.debug("Extracting mel spectrogram")
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        # Convert to dB scale
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB
    
    def preprocess_for_model(self, features):
        """
        Preprocess extracted features for model input.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            dict: Preprocessed features ready for model input
        """
        # This is a placeholder for more complex preprocessing
        logger.info("Preprocessing features for model input")
        
        # Normalize spectral features
        norm_spec = (features['spectrogram'] - features['spectrogram'].mean()) / features['spectrogram'].std()
        
        # Package features for model
        model_input = {
            'spectrogram': norm_spec,
            'chroma': features['chroma'],
            'onsets': features['onsets']['frames']
        }
        
        return model_input