#!/usr/bin/env python3
"""
Create a mock GuitarSet dataset structure for testing.

This script creates a minimal GuitarSet dataset structure to allow
the training script to run. Use this for development and testing 
when the full dataset cannot be downloaded.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
import numpy as np
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mock_guitarset")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create a mock GuitarSet dataset structure for testing.'
    )
    parser.add_argument(
        '--output-dir',
        default='datasets/guitarset',
        help='Directory to create the mock dataset'
    )
    parser.add_argument(
        '--num-players',
        type=int,
        default=3,
        help='Number of player directories to create'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of audio samples per player'
    )
    return parser.parse_args()

def create_mock_audio_file(filepath, duration=5.0, sample_rate=44100):
    """
    Create a mock hexaphonic guitar audio WAV file.
    
    Args:
        filepath: Path to save the WAV file
        duration: Duration of the audio in seconds
        sample_rate: Sample rate of the audio
    """
    try:
        import soundfile as sf
        
        # Generate some random noise for each string (6 channels)
        num_samples = int(duration * sample_rate)
        mock_audio = np.random.randn(num_samples, 6) * 0.1
        
        # Add some "notes" (sine waves) to make it more guitar-like
        for string in range(6):
            freq = 110 * (2 ** (string / 12))  # Approximate guitar string frequencies
            t = np.linspace(0, duration, num_samples)
            sine_wave = 0.5 * np.sin(2 * np.pi * freq * t)
            decay = np.exp(-t / 2)
            mock_audio[:, string] += sine_wave * decay
        
        # Save as WAV
        sf.write(filepath, mock_audio, sample_rate)
        return True
    except ImportError:
        # If soundfile is not available, create an empty file
        logger.warning("soundfile module not available, creating empty WAV file")
        with open(filepath, 'wb') as f:
            # Write a minimal WAV header
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x06\x00\x44\xac\x00\x00\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00')
        return True
    except Exception as e:
        logger.error(f"Failed to create mock audio file: {e}")
        return False

def create_mock_jams_file(filepath, duration=5.0):
    """
    Create a mock JAMS annotation file.
    
    Args:
        filepath: Path to save the JAMS file
        duration: Duration of the annotation in seconds
    """
    try:
        # Create a minimal JAMS file structure with standard namespaces
        jams_data = {
            "file_metadata": {
                "title": "Mock GuitarSet Annotation",
                "duration": duration
            },
            "annotations": [
                {
                    "namespace": "note_midi",
                    "annotation_metadata": {
                        "data_source": "program",
                        "annotation_tools": "mock generator",
                        "annotator": {
                            "name": "auto"
                        }
                    },
                    "data": [
                        {"time": 0.0, "duration": 0.5, "value": 40, "confidence": 1.0},
                        {"time": 1.0, "duration": 0.5, "value": 45, "confidence": 1.0},
                        {"time": 2.0, "duration": 0.5, "value": 50, "confidence": 1.0}
                    ]
                },
                {
                    "namespace": "note",
                    "annotation_metadata": {
                        "data_source": "program",
                        "annotation_tools": "mock generator",
                        "annotator": {
                            "name": "auto"
                        }
                    },
                    "data": [
                        {"time": 0.0, "duration": 0.5, "value": {"pitch": 40, "string": 5, "fret": 0}, "confidence": 1.0},
                        {"time": 1.0, "duration": 0.5, "value": {"pitch": 45, "string": 4, "fret": 5}, "confidence": 1.0},
                        {"time": 2.0, "duration": 0.5, "value": {"pitch": 50, "string": 3, "fret": 7}, "confidence": 1.0}
                    ]
                }
            ],
            "sandbox": {
                "guitarset": {
                    "string_fret_annotations": [
                        {"time": 0.0, "duration": 0.5, "string": 5, "fret": 0, "confidence": 1.0},
                        {"time": 1.0, "duration": 0.5, "string": 4, "fret": 5, "confidence": 1.0},
                        {"time": 2.0, "duration": 0.5, "string": 3, "fret": 7, "confidence": 1.0}
                    ]
                }
            }
        }
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(jams_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to create mock JAMS file: {e}")
        return False

def create_mock_guitarset(output_dir, num_players=3, num_samples=5):
    """
    Create a mock GuitarSet dataset.
    
    Args:
        output_dir: Directory to create the dataset
        num_players: Number of player directories to create
        num_samples: Number of audio samples per player
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Creating mock GuitarSet dataset in {output_dir} with {num_players} players")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    success = True
    for i in range(1, num_players + 1):
        player_id = f"player{i:02d}"
        player_dir = os.path.join(output_dir, player_id)
        
        # Create player directory
        os.makedirs(player_dir, exist_ok=True)
        
        # Create audio and annotation directories
        audio_dir = os.path.join(player_dir, "audio")
        anno_dir = os.path.join(player_dir, "annotation")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)
        
        # Create mock files
        for j in range(1, num_samples + 1):
            sample_id = f"{player_id}_{j:02d}"
            
            # Create audio file
            audio_file = os.path.join(audio_dir, f"{sample_id}_hex.wav")
            if not create_mock_audio_file(audio_file):
                success = False
            
            # Create annotation file
            anno_file = os.path.join(anno_dir, f"{sample_id}.jams")
            if not create_mock_jams_file(anno_file):
                success = False
        
        logger.info(f"Created {num_samples} samples for {player_id}")
    
    if success:
        logger.info(f"Mock GuitarSet dataset created successfully at {output_dir}")
        logger.info(f"Use this path with the training script: --data-dir {output_dir}")
    else:
        logger.error("Failed to create some mock files")
    
    return success

def main():
    """Main function."""
    args = parse_args()
    
    # Convert to absolute path
    output_dir = os.path.abspath(args.output_dir)
    
    logger.info(f"Output directory: {output_dir}")
    
    try:
        create_mock_guitarset(output_dir, args.num_players, args.num_samples)
    except KeyboardInterrupt:
        logger.info("Creation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating mock GuitarSet: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()