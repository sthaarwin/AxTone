#!/usr/bin/env python3
"""
Organize partially downloaded GuitarSet dataset.

This script organizes the partially downloaded GuitarSet dataset files
into the proper directory structure for use with AxTone.
"""

import os
import sys
import argparse
import logging
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("organize_guitarset")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Organize partially downloaded GuitarSet dataset.'
    )
    parser.add_argument(
        '--temp-dir',
        default='/tmp/guitarset_download',
        help='Directory with downloaded ZIP files'
    )
    parser.add_argument(
        '--output-dir',
        default='datasets/guitarset',
        help='Directory to save the organized GuitarSet dataset'
    )
    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='Only extract ZIP files without organizing'
    )
    return parser.parse_args()

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(zip_path):
        logger.warning(f"ZIP file not found: {zip_path}")
        return False
        
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract the zip file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extraction complete: {extract_to}")
        return True
    except zipfile.BadZipFile:
        logger.error(f"Failed to extract {zip_path}: File is not a valid ZIP archive")
        return False
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False

def organize_dataset(temp_dir, output_dir):
    """
    Organize the dataset into the expected structure.
    
    Args:
        temp_dir: Temporary directory with extracted files
        output_dir: Output directory for the organized dataset
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Organizing dataset to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check available directories
    available_dirs = []
    
    # List of possible source directories
    source_dirs = [
        {"name": "annotation", "path": os.path.join(temp_dir, "annotation")},
        {"name": "audio_hex", "path": os.path.join(temp_dir, "audio_hex-pickup")},
        {"name": "audio_mono", "path": os.path.join(temp_dir, "audio_mono-mic")},
        {"name": "audio_pickup", "path": os.path.join(temp_dir, "audio_mono-pickup_mix")}
    ]
    
    # Check which directories exist
    for source_dir in source_dirs:
        if os.path.exists(source_dir["path"]):
            available_dirs.append(source_dir)
            logger.info(f"Found {source_dir['name']} directory: {source_dir['path']}")
    
    if not available_dirs:
        logger.error(f"No valid data directories found in {temp_dir}")
        return False
    
    # Find annotation directory which is required for player structure
    annotation_dir = None
    for source_dir in available_dirs:
        if source_dir["name"] == "annotation":
            annotation_dir = source_dir["path"]
            break
    
    if not annotation_dir:
        logger.error("Annotation directory not found. Cannot determine player structure.")
        return False
    
    # List all players from annotation directory
    try:
        players = sorted([d for d in os.listdir(annotation_dir) 
                        if os.path.isdir(os.path.join(annotation_dir, d))])
        
        if not players:
            logger.error(f"No player directories found in {annotation_dir}")
            return False
            
    except Exception as e:
        logger.error(f"Error listing player directories: {e}")
        return False
    
    logger.info(f"Found {len(players)} players: {', '.join(players)}")
    
    # Organize by player
    for player in players:
        player_dir = os.path.join(output_dir, player)
        os.makedirs(player_dir, exist_ok=True)
        
        # Create audio and annotation directories
        player_audio_dir = os.path.join(player_dir, "audio")
        player_anno_dir = os.path.join(player_dir, "annotation")
        os.makedirs(player_audio_dir, exist_ok=True)
        os.makedirs(player_anno_dir, exist_ok=True)
        
        # Copy annotation files
        source_anno_dir = os.path.join(annotation_dir, player)
        if os.path.exists(source_anno_dir):
            anno_files_copied = 0
            for anno_file in os.listdir(source_anno_dir):
                if anno_file.endswith(".jams"):
                    shutil.copy2(
                        os.path.join(source_anno_dir, anno_file),
                        os.path.join(player_anno_dir, anno_file)
                    )
                    anno_files_copied += 1
            
            logger.info(f"Copied {anno_files_copied} annotation files for player {player}")
        else:
            logger.warning(f"Annotation directory not found for player {player}")
        
        # Copy audio files from each available source
        for source_dir in available_dirs:
            if source_dir["name"] in ["audio_hex", "audio_mono", "audio_pickup"]:
                source_audio_dir = os.path.join(source_dir["path"], player)
                
                if not os.path.exists(source_audio_dir):
                    logger.warning(f"Audio directory {source_dir['name']} not found for player {player}")
                    continue
                    
                audio_files_copied = 0
                for audio_file in os.listdir(source_audio_dir):
                    if audio_file.endswith(".wav"):
                        shutil.copy2(
                            os.path.join(source_audio_dir, audio_file),
                            os.path.join(player_audio_dir, audio_file)
                        )
                        audio_files_copied += 1
                
                logger.info(f"Copied {audio_files_copied} {source_dir['name']} audio files for player {player}")
    
    logger.info(f"Dataset organization complete: {output_dir}")
    return True

def main():
    """Main function to organize GuitarSet dataset."""
    args = parse_args()
    
    # Convert to absolute paths
    output_dir = os.path.abspath(args.output_dir)
    temp_dir = os.path.abspath(args.temp_dir)
    
    logger.info(f"Temp directory with ZIPs: {temp_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # First, check if any ZIP files need extraction
        zip_files = [
            os.path.join(temp_dir, "annotation.zip"),
            os.path.join(temp_dir, "audio_mono-mic.zip"),
            os.path.join(temp_dir, "audio_mono-pickup_mix.zip")
        ]
        
        # Extract all available ZIP files
        for zip_path in zip_files:
            if os.path.exists(zip_path):
                extract_zip(zip_path, temp_dir)
        
        # Stop here if extract-only flag is set
        if args.extract_only:
            logger.info("Extract-only mode: Skipping organization step")
            return
        
        # Organize dataset
        organization_success = organize_dataset(temp_dir, output_dir)
        
        if organization_success:
            logger.info(f"GuitarSet organization complete. Dataset available at: {output_dir}")
            logger.info(f"Use this path with the training script: --data-dir {output_dir}")
        else:
            logger.error("GuitarSet organization failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Organization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error organizing GuitarSet: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()