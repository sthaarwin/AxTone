"""
Dataset module for managing labeled audio-tab pairs.

This module provides tools for loading, processing, and preparing
datasets of paired audio and tablature for model training.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import json
import jams
import scipy.signal

logger = logging.getLogger(__name__)

class AudioTabPair:
    """Represents a single paired audio and tablature sample."""
    
    def __init__(self, audio_path: str, tab_path: str, metadata: Optional[Dict] = None):
        """
        Initialize an audio-tab pair.
        
        Args:
            audio_path: Path to the audio file
            tab_path: Path to the corresponding tablature file
            metadata: Optional metadata about the pair (artist, song, etc.)
        """
        self.audio_path = audio_path
        self.tab_path = tab_path
        self.metadata = metadata or {}
        
        # Verify files exist
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(tab_path):
            raise FileNotFoundError(f"Tab file not found: {tab_path}")
    
    def load_audio(self, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load the audio file and return the waveform and sample rate."""
        y, sr = librosa.load(self.audio_path, sr=sr)
        return y, sr
    
    def load_tab(self) -> str:
        """Load the tablature file and return its contents as a string."""
        with open(self.tab_path, 'r') as f:
            return f.read()
    
    def __repr__(self) -> str:
        return f"AudioTabPair(audio={os.path.basename(self.audio_path)}, tab={os.path.basename(self.tab_path)})"


class TabDataset(Dataset):
    """PyTorch dataset for audio-tab pairs."""
    
    def __init__(
        self, 
        audio_dir: str, 
        tab_dir: str, 
        transform=None, 
        sample_rate: Optional[int] = None,
        max_audio_length: Optional[int] = None,
        metadata_file: Optional[str] = None
    ):
        """
        Initialize dataset with directories for audio files and corresponding tablature.
        
        Args:
            audio_dir: Directory containing audio files
            tab_dir: Directory containing tab files
            transform: Optional transform to apply to features
            sample_rate: Sample rate to use for audio loading
            max_audio_length: Maximum number of samples in audio (for padding/truncating)
            metadata_file: Optional path to JSON file with additional metadata
        """
        self.audio_dir = audio_dir
        self.tab_dir = tab_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        
        # Load metadata if provided
        self.metadata = {}
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        # Find all matching audio and tab pairs
        self.pairs = self._find_pairs()
        logger.info(f"Found {len(self.pairs)} audio-tab pairs")
    
    def _find_pairs(self) -> List[AudioTabPair]:
        """Find all matching audio and tab file pairs."""
        pairs = []
        
        # Get all audio files
        audio_files = [f for f in os.listdir(self.audio_dir) 
                      if f.endswith(('.wav', '.mp3', '.ogg'))]
        
        # For each audio file, look for a matching tab file
        for audio_file in audio_files:
            base_name = os.path.splitext(audio_file)[0]
            
            # Check for tab file with the same name
            tab_file = f"{base_name}.tab"
            tab_path = os.path.join(self.tab_dir, tab_file)
            
            if os.path.exists(tab_path):
                audio_path = os.path.join(self.audio_dir, audio_file)
                
                # Get metadata for this pair if available
                metadata = self.metadata.get(base_name, {})
                
                pairs.append(AudioTabPair(audio_path, tab_path, metadata))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx) -> Dict:
        """Get a dataset item by index."""
        pair = self.pairs[idx]
        
        # Load audio and tab
        audio, sr = pair.load_audio(self.sample_rate)
        tab = pair.load_tab()
        
        # Preprocess audio
        features = self._extract_features(audio, sr)
        
        # Preprocess tab (convert to tensor representation)
        tab_tensor = self._convert_tab_to_tensor(tab)
        
        # Apply transforms if available
        if self.transform:
            features = self.transform(features)
        
        return {
            'features': features,
            'tab': tab_tensor,
            'metadata': pair.metadata,
            'audio_path': pair.audio_path,
            'tab_path': pair.tab_path
        }
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract features from audio data."""
        features = {}
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        features['mel_spectrogram'] = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract chromagram
        features['chromagram'] = librosa.feature.chroma_cqt(y=audio, sr=sr)
        
        # Extract MFCCs
        features['mfccs'] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        return features
    
    def _convert_tab_to_tensor(self, tab_string: str) -> torch.Tensor:
        """
        Convert tab string to tensor representation.
        
        Parses the tab string into a tensor with dimensions:
        [num_strings, time_steps] where each value represents the fret number.
        """
        # Simple parsing of standard tab format
        lines = tab_string.strip().split('\n')
        
        # Remove any header or non-tab lines
        tab_lines = [line for line in lines if '-' in line]
        
        # Determine number of strings and time steps
        num_strings = len(tab_lines)
        max_length = max(len(line) for line in tab_lines)
        
        # Initialize tensor with -1 (representing no note played)
        tab_tensor = torch.full((num_strings, max_length), -1)
        
        # Parse each string
        for string_idx, string_line in enumerate(tab_lines):
            for pos_idx, char in enumerate(string_line):
                if char.isdigit():
                    # Single digit fret number
                    tab_tensor[string_idx, pos_idx] = int(char)
                elif pos_idx < max_length - 1 and char.isdigit() and string_line[pos_idx+1].isdigit():
                    # Two-digit fret number
                    fret_num = int(char + string_line[pos_idx+1])
                    tab_tensor[string_idx, pos_idx] = fret_num
                    # Skip the next position as it's part of this fret number
                    pos_idx += 1
                    
        return tab_tensor


class GuitarSetDataset(Dataset):
    """Dataset class for the GuitarSet dataset."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_fft: int = 2048,
        sequence_length: int = 256,
        split_ratio: Dict[str, float] = {'train': 0.8, 'val': 0.1, 'test': 0.1},
        random_seed: int = 42
    ):
        """
        Initialize the GuitarSet dataset.
        
        Args:
            root_dir: Root directory of the GuitarSet dataset
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transform to apply to features
            sample_rate: Sample rate to use for audio loading
            hop_length: Hop length for feature extraction
            n_fft: FFT size for feature extraction
            sequence_length: Length of sequences for training
            split_ratio: Ratio of train/val/test splits
            random_seed: Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sequence_length = sequence_length
        
        # Find all player folders
        player_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        # Find all audio and annotation files
        self.audio_files = []
        self.jams_files = []
        
        for player_dir in player_dirs:
            audio_dir = player_dir / 'audio'
            anno_dir = player_dir / 'annotation'
            
            if not audio_dir.exists() or not anno_dir.exists():
                continue
                
            # Find hexaphonic audio files (.wav)
            hex_audio_files = list(audio_dir.glob('*_hex.wav'))
            
            # Find corresponding JAMS files
            for audio_file in hex_audio_files:
                base_name = audio_file.stem.replace('_hex', '')
                jams_file = anno_dir / f"{base_name}.jams"
                
                if jams_file.exists():
                    self.audio_files.append(str(audio_file))
                    self.jams_files.append(str(jams_file))
        
        # Split into train/val/test
        import random
        random.seed(random_seed)
        
        # Create indices for the full dataset
        indices = list(range(len(self.audio_files)))
        random.shuffle(indices)
        
        # Calculate split sizes
        train_size = int(len(indices) * split_ratio['train'])
        val_size = int(len(indices) * split_ratio['val'])
        
        # Get indices for the requested split
        if split == 'train':
            self.indices = indices[:train_size]
        elif split == 'val':
            self.indices = indices[train_size:train_size + val_size]
        elif split == 'test':
            self.indices = indices[train_size + val_size:]
        else:
            raise ValueError(f"Invalid split: {split}")
            
        logger.info(f"Initialized GuitarSet {split} dataset with {len(self.indices)} samples")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx) -> Dict:
        """Get a dataset item by index."""
        # Get the real index from our split indices
        real_idx = self.indices[idx]
        
        # Get audio and JAMS file paths
        audio_path = self.audio_files[real_idx]
        jams_path = self.jams_files[real_idx]
        
        # Load hexaphonic audio (6 channels, one per string)
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
        
        # Load JAMS annotation
        jam = jams.load(jams_path)
        
        # Extract features from audio
        features = self._extract_features(audio, sr)
        
        # Process JAMS annotations to get tab targets
        tab_targets = self._process_annotations(jam)
        
        # Create a sample
        sample = {
            'features': features,
            'tab_targets': tab_targets,
            'audio_path': audio_path,
            'jams_path': jams_path
        }
        
        # Apply transforms if available
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract features from hexaphonic audio data.
        
        Args:
            audio: Hexaphonic audio data (6 channels)
            sr: Sample rate
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Ensure we have 6 channels (one per string)
        if audio.shape[0] != 6:
            raise ValueError(f"Expected 6 audio channels, got {audio.shape[0]}")
        
        # Extract mel spectrograms for each string
        mel_specs = []
        for string_idx in range(6):
            string_audio = audio[string_idx]
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=string_audio, 
                sr=sr, 
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=128
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_specs.append(mel_spec_db)
        
        # Stack mel spectrograms from all strings
        features['mel_spectrograms'] = np.stack(mel_specs)
        
        # Also extract features from the mixed audio (sum of all strings)
        mixed_audio = np.sum(audio, axis=0)
        
        # Extract onset envelope
        onset_env = librosa.onset.onset_strength(
            y=mixed_audio, 
            sr=sr,
            hop_length=self.hop_length
        )
        features['onset_envelope'] = onset_env
        
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(
            y=mixed_audio, 
            sr=sr,
            hop_length=self.hop_length
        )
        features['chroma'] = chroma
        
        return features
    
    def _process_annotations(self, jam) -> Dict[str, np.ndarray]:
        """
        Process JAMS annotations to extract tab targets.
        
        Args:
            jam: JAMS annotation object
            
        Returns:
            Dictionary of tab targets
        """
        # Initialize targets
        targets = {}
        
        # Get annotation data for string/fret positions
        note_data = []
        
        # Look for the note annotation data
        for annotation in jam.annotations:
            if annotation.namespace == 'note_midi':
                for obs in annotation.data:
                    # Extract relevant information
                    time = obs.time
                    duration = obs.duration
                    pitch = obs.value
                    string = obs.value_annotations.get('string', None)
                    fret = obs.value_annotations.get('fret', None)
                    
                    if string is not None and fret is not None:
                        note_data.append({
                            'time': time,
                            'duration': duration,
                            'pitch': pitch,
                            'string': string,
                            'fret': fret
                        })
        
        # Sort notes by time
        note_data.sort(key=lambda x: x['time'])
        
        # Convert note data to frame-level targets
        if note_data:
            # Find the maximum time to determine the number of frames
            max_time = max(note['time'] + note['duration'] for note in note_data)
            num_frames = int(max_time * self.sample_rate / self.hop_length) + 1
            
            # Initialize frame-level targets
            # For each frame, we need to know which string is being played and at what fret
            string_targets = np.full((6, num_frames), -1, dtype=np.int32)  # -1 means string not played
            fret_targets = np.full((6, num_frames), -1, dtype=np.int32)    # -1 means no fret
            
            # Fill in the targets
            for note in note_data:
                # Convert time to frame index
                start_frame = int(note['time'] * self.sample_rate / self.hop_length)
                end_frame = int((note['time'] + note['duration']) * self.sample_rate / self.hop_length)
                
                # Ensure we don't go beyond the array bounds
                end_frame = min(end_frame, num_frames - 1)
                
                # String index is 0-based (string 1 = index 0, string 6 = index 5)
                string_idx = int(note['string']) - 1
                
                # Set the string and fret targets for this note's duration
                for frame in range(start_frame, end_frame + 1):
                    string_targets[string_idx, frame] = 1  # String is played
                    fret_targets[string_idx, frame] = int(note['fret'])
            
            targets['string_targets'] = string_targets
            targets['fret_targets'] = fret_targets
            
            # Also create a combined representation for sequence modeling
            # Each frame will have a 6-element vector, one per string, with the fret number
            # We use -1 to indicate the string is not played
            tab_sequence = np.full((num_frames, 6), -1, dtype=np.int32)
            
            for string_idx in range(6):
                for frame in range(num_frames):
                    if string_targets[string_idx, frame] == 1:
                        tab_sequence[frame, string_idx] = fret_targets[string_idx, frame]
            
            targets['tab_sequence'] = tab_sequence
            
            # Trim to sequence length or pad if necessary
            if num_frames < self.sequence_length:
                # Pad
                pad_length = self.sequence_length - num_frames
                targets['string_targets'] = np.pad(
                    targets['string_targets'],
                    ((0, 0), (0, pad_length)),
                    mode='constant',
                    constant_values=-1
                )
                targets['fret_targets'] = np.pad(
                    targets['fret_targets'],
                    ((0, 0), (0, pad_length)),
                    mode='constant',
                    constant_values=-1
                )
                targets['tab_sequence'] = np.pad(
                    targets['tab_sequence'],
                    ((0, pad_length), (0, 0)),
                    mode='constant',
                    constant_values=-1
                )
            elif num_frames > self.sequence_length:
                # Randomly select a window of sequence_length frames
                start_idx = np.random.randint(0, num_frames - self.sequence_length)
                targets['string_targets'] = targets['string_targets'][:, start_idx:start_idx + self.sequence_length]
                targets['fret_targets'] = targets['fret_targets'][:, start_idx:start_idx + self.sequence_length]
                targets['tab_sequence'] = targets['tab_sequence'][start_idx:start_idx + self.sequence_length]
        else:
            # No notes found, create empty targets
            targets['string_targets'] = np.full((6, self.sequence_length), -1, dtype=np.int32)
            targets['fret_targets'] = np.full((6, self.sequence_length), -1, dtype=np.int32)
            targets['tab_sequence'] = np.full((self.sequence_length, 6), -1, dtype=np.int32)
        
        return targets


def create_dataloader(
    audio_dir: str, 
    tab_dir: str, 
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a PyTorch DataLoader for audio-tab pairs.
    
    Args:
        audio_dir: Directory containing audio files
        tab_dir: Directory containing tab files
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments for TabDataset
        
    Returns:
        PyTorch DataLoader
    """
    dataset = TabDataset(audio_dir, tab_dir, **dataset_kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def create_guitarset_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for the GuitarSet dataset.
    
    Args:
        root_dir: Root directory of the GuitarSet dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments for GuitarSetDataset
        
    Returns:
        Dictionary with train, validation, and test DataLoaders
    """
    # Create datasets for each split
    train_dataset = GuitarSetDataset(root_dir, split='train', **dataset_kwargs)
    val_dataset = GuitarSetDataset(root_dir, split='val', **dataset_kwargs)
    test_dataset = GuitarSetDataset(root_dir, split='test', **dataset_kwargs)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def split_dataset(
    dataset_dir: str,
    audio_dir: str = 'audio',
    tab_dir: str = 'tabs',
    output_dir: str = 'split_dataset',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Split a dataset into training, validation and test sets.
    
    Args:
        dataset_dir: Root directory of the dataset
        audio_dir: Subdirectory containing audio files
        tab_dir: Subdirectory containing tab files
        output_dir: Directory to save the split dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with paths to each split
    """
    import random
    import shutil
    
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Set random seed
    random.seed(random_seed)
    
    # Create output directories
    train_audio_dir = os.path.join(output_dir, 'train', audio_dir)
    train_tab_dir = os.path.join(output_dir, 'train', tab_dir)
    val_audio_dir = os.path.join(output_dir, 'val', audio_dir)
    val_tab_dir = os.path.join(output_dir, 'val', tab_dir)
    test_audio_dir = os.path.join(output_dir, 'test', audio_dir)
    test_tab_dir = os.path.join(output_dir, 'test', tab_dir)
    
    for directory in [train_audio_dir, train_tab_dir, val_audio_dir, 
                      val_tab_dir, test_audio_dir, test_tab_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Find all audio files
    audio_files = [f for f in os.listdir(os.path.join(dataset_dir, audio_dir)) 
                   if f.endswith(('.wav', '.mp3', '.ogg'))]
    
    # Shuffle the files
    random.shuffle(audio_files)
    
    # Calculate split indices
    n_files = len(audio_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    train_files = audio_files[:n_train]
    val_files = audio_files[n_train:n_train+n_val]
    test_files = audio_files[n_train+n_val:]
    
    # Function to copy files for a split
    def copy_files_for_split(files, src_audio_dir, src_tab_dir, 
                           dst_audio_dir, dst_tab_dir):
        copied_files = []
        for audio_file in files:
            base_name = os.path.splitext(audio_file)[0]
            tab_file = f"{base_name}.tab"
            
            src_audio_path = os.path.join(src_audio_dir, audio_file)
            src_tab_path = os.path.join(src_tab_dir, tab_file)
            
            # Only copy if both files exist
            if os.path.exists(src_audio_path) and os.path.exists(src_tab_path):
                dst_audio_path = os.path.join(dst_audio_dir, audio_file)
                dst_tab_path = os.path.join(dst_tab_dir, tab_file)
                
                shutil.copy2(src_audio_path, dst_audio_path)
                shutil.copy2(src_tab_path, dst_tab_path)
                
                copied_files.append(base_name)
        
        return copied_files
    
    # Copy files for each split
    train_copied = copy_files_for_split(
        train_files, 
        os.path.join(dataset_dir, audio_dir),
        os.path.join(dataset_dir, tab_dir),
        train_audio_dir,
        train_tab_dir
    )
    
    val_copied = copy_files_for_split(
        val_files,
        os.path.join(dataset_dir, audio_dir),
        os.path.join(dataset_dir, tab_dir),
        val_audio_dir,
        val_tab_dir
    )
    
    test_copied = copy_files_for_split(
        test_files,
        os.path.join(dataset_dir, audio_dir),
        os.path.join(dataset_dir, tab_dir),
        test_audio_dir,
        test_tab_dir
    )
    
    # Log the results
    logger.info(f"Dataset split complete:")
    logger.info(f"  Training: {len(train_copied)} files")
    logger.info(f"  Validation: {len(val_copied)} files")
    logger.info(f"  Test: {len(test_copied)} files")
    
    return {
        'train': train_copied,
        'val': val_copied,
        'test': test_copied,
        'train_dir': os.path.join(output_dir, 'train'),
        'val_dir': os.path.join(output_dir, 'val'),
        'test_dir': os.path.join(output_dir, 'test')
    }