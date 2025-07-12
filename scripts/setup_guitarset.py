#!/usr/bin/env python3
"""
Download and setup GuitarSet dataset script.

This script downloads the GuitarSet dataset from the official source
and prepares it for use with the tab generation model.
"""

import os
import sys
import argparse
import logging
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup_guitarset")

# Updated GuitarSet download URLs with correct Zenodo format
# Zenodo changed URL format in newer versions
AUDIO_MONO_URL = "https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1"
AUDIO_HEX_URL_ORIGINAL = "https://zenodo.org/records/3371780/files/audio_hex-pickup_original.zip?download=1"
AUDIO_MONO_PICKUP_MIX_URL = "https://zenodo.org/records/3371780/files/audio_mono-pickup_mix.zip?download=1"
ANNOTATIONS_URL = "https://zenodo.org/records/3371780/files/annotation.zip?download=1"

# Alternative URLs (in case primary URLs fail)
# GuitarSet is also available from other sources
ALT_AUDIO_MONO_URL = "https://guitarset.weebly.com/uploads/1/3/4/0/13407324/audio_mono-mic.zip"
ALT_AUDIO_HEX_URL = "https://guitarset.weebly.com/uploads/1/3/4/0/13407324/audio_hex-pickup.zip"
ALT_ANNOTATIONS_URL = "https://guitarset.weebly.com/uploads/1/3/4/0/13407324/annotation.zip"

# GitHub URLs not used anymore as they're not reliable
GITHUB_AUDIO_MONO_URL = ALT_AUDIO_MONO_URL
GITHUB_AUDIO_HEX_URL = ALT_AUDIO_HEX_URL  
GITHUB_ANNOTATIONS_URL = ALT_ANNOTATIONS_URL

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download and setup GuitarSet dataset.'
    )
    parser.add_argument(
        '--output-dir',
        default='datasets/guitarset',
        help='Directory to save the GuitarSet dataset'
    )
    parser.add_argument(
        '--temp-dir',
        default='/tmp/guitarset_download',
        help='Temporary directory for downloads'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force download even if files already exist'
    )
    parser.add_argument(
        '--skip-mono',
        action='store_true',
        help='Skip downloading mono audio (smaller download)'
    )
    parser.add_argument(
        '--clean-temp',
        action='store_true',
        help='Clean temporary files after successful setup'
    )
    return parser.parse_args()

def download_file(url, output_path, force=False, use_alternative=False):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        force: Whether to overwrite existing file
        use_alternative: Whether to try alternative URL if primary fails
        
    Returns:
        Path to the downloaded file
    """
    if os.path.exists(output_path) and not force:
        logger.info(f"File already exists: {output_path}")
        # Verify file is valid
        if output_path.endswith('.zip'):
            try:
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    # Just check the file list
                    zip_ref.namelist()
                logger.info(f"Verified valid ZIP file: {output_path}")
                return output_path
            except zipfile.BadZipFile:
                logger.warning(f"Existing file is not a valid ZIP archive: {output_path}")
                logger.info(f"Removing invalid file and re-downloading")
                os.remove(output_path)
        else:
            return output_path
    
    logger.info(f"Downloading {url} to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try alternative URL if available and requested
    urls_to_try = [url]
    if use_alternative:
        # Get the filename from the output path
        filename = os.path.basename(output_path)
        
        # Handle audio hex pickup special case with split files
        if "audio_hex-pickup_part" in filename:
            if "part1" in filename:
                alt_url = ALT_AUDIO_HEX_URL
                urls_to_try.append(alt_url)
            # Part2 doesn't have a separate alternative, it's combined in the ALT_AUDIO_HEX_URL
        elif "audio_mono-mic" in filename:
            alt_url = ALT_AUDIO_MONO_URL
            urls_to_try.append(alt_url)
        elif "annotation" in filename:
            alt_url = ALT_ANNOTATIONS_URL
            urls_to_try.append(alt_url)
    
    # Try each URL in turn
    for try_url in urls_to_try:
        try:
            response = requests.get(try_url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            with open(output_path, 'wb') as file, tqdm(
                desc=os.path.basename(output_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)
            
            logger.info(f"Download complete: {output_path}")
            
            # Verify the downloaded file is a valid zip
            if output_path.endswith('.zip'):
                try:
                    with zipfile.ZipFile(output_path, 'r') as zip_ref:
                        # Just check the file list
                        zip_ref.namelist()
                    logger.info(f"Verified valid ZIP file: {output_path}")
                except zipfile.BadZipFile:
                    logger.error(f"Downloaded file is not a valid ZIP archive: {output_path}")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    continue  # Try next URL if available
            
            return output_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {try_url}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            # Continue to next URL if available
    
    # If we get here, all URLs failed
    return None

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
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
    
    # Check source directories
    audio_hex_dir = os.path.join(temp_dir, "audio_hex-pickup")
    annotation_dir = os.path.join(temp_dir, "annotation")
    
    if not os.path.exists(audio_hex_dir):
        logger.error(f"Missing required directory: {audio_hex_dir}")
        return False
        
    if not os.path.exists(annotation_dir):
        logger.error(f"Missing required directory: {annotation_dir}")
        return False
    
    # List all players
    try:
        players = sorted([d for d in os.listdir(audio_hex_dir) 
                        if os.path.isdir(os.path.join(audio_hex_dir, d))])
        
        if not players:
            logger.error(f"No player directories found in {audio_hex_dir}")
            return False
            
    except Exception as e:
        logger.error(f"Error listing player directories: {e}")
        return False
    
    # Organize by player
    for player in players:
        player_dir = os.path.join(output_dir, player)
        os.makedirs(player_dir, exist_ok=True)
        
        # Create audio and annotation directories
        player_audio_dir = os.path.join(player_dir, "audio")
        player_anno_dir = os.path.join(player_dir, "annotation")
        os.makedirs(player_audio_dir, exist_ok=True)
        os.makedirs(player_anno_dir, exist_ok=True)
        
        # Copy hexaphonic audio files
        source_audio_dir = os.path.join(audio_hex_dir, player)
        if not os.path.exists(source_audio_dir):
            logger.warning(f"Audio directory not found for player {player}: {source_audio_dir}")
            continue
            
        audio_files_copied = 0
        for audio_file in os.listdir(source_audio_dir):
            if audio_file.endswith(".wav") and "_hex" in audio_file:
                shutil.copy2(
                    os.path.join(source_audio_dir, audio_file),
                    os.path.join(player_audio_dir, audio_file)
                )
                audio_files_copied += 1
        
        logger.info(f"Copied {audio_files_copied} audio files for player {player}")
        
        # Copy annotation files
        source_anno_dir = os.path.join(annotation_dir, player)
        if not os.path.exists(source_anno_dir):
            logger.warning(f"Annotation directory not found for player {player}: {source_anno_dir}")
            continue
            
        anno_files_copied = 0
        for anno_file in os.listdir(source_anno_dir):
            if anno_file.endswith(".jams"):
                shutil.copy2(
                    os.path.join(source_anno_dir, anno_file),
                    os.path.join(player_anno_dir, anno_file)
                )
                anno_files_copied += 1
        
        logger.info(f"Copied {anno_files_copied} annotation files for player {player}")
    
    logger.info(f"Dataset organization complete: {output_dir}")
    logger.info(f"Found {len(players)} players: {', '.join(players)}")
    return True

def setup_guitarset(output_dir, temp_dir, force=False, skip_mono=False, clean_temp=False):
    """
    Download and setup the GuitarSet dataset.
    
    Args:
        output_dir: Directory to save the organized dataset
        temp_dir: Temporary directory for downloads
        force: Whether to force download even if files exist
        skip_mono: Skip downloading mono audio (smaller download)
        clean_temp: Clean temporary files after successful setup
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Setting up GuitarSet dataset")
    
    # Create directories
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define downloads with primary and alternative URLs
    downloads = [
        {
            "primary_url": AUDIO_HEX_URL_ORIGINAL,
            "alt_url": ALT_AUDIO_HEX_URL,
            "path": os.path.join(temp_dir, "audio_hex-pickup_original.zip"),
            "required": True
        },
        {
            "primary_url": ANNOTATIONS_URL,
            "alt_url": ALT_ANNOTATIONS_URL,
            "path": os.path.join(temp_dir, "annotation.zip"),
            "required": True
        }
    ]
    
    if not skip_mono:
        downloads.append({
            "primary_url": AUDIO_MONO_URL,
            "alt_url": ALT_AUDIO_MONO_URL,
            "path": os.path.join(temp_dir, "audio_mono-mic.zip"),
            "required": False
        })
        
        # Add mono pickup mix as an optional download
        downloads.append({
            "primary_url": AUDIO_MONO_PICKUP_MIX_URL,
            "alt_url": None,  # No alternative for this one
            "path": os.path.join(temp_dir, "audio_mono-pickup_mix.zip"),
            "required": False
        })
    # Download all files
    all_downloads_successful = True
    
    for download in downloads:
        result = download_file(download["primary_url"], download["path"], force)
        if result is None and download["alt_url"]:
            # Try alternative URL if available
            logger.info(f"Trying alternative URL: {download['alt_url']}")
            result = download_file(download["alt_url"], download["path"], force)
        
        if result is None and download["required"]:
            all_downloads_successful = False
            logger.error(f"Failed to download required file: {download['path']}")
    
    # Check if required downloads succeeded
    required_files = [d["path"] for d in downloads if d["required"]]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        return False
    
    # Extract files
    files_to_extract = [f for f in required_files if os.path.exists(f)]
    # Add optional files if they exist
    for download in downloads:
        if not download["required"] and os.path.exists(download["path"]):
            files_to_extract.append(download["path"])
    
    extract_success = True
    for path in files_to_extract:
        if not extract_zip(path, temp_dir):
            extract_success = False
    
    if not extract_success:
        logger.error("Failed to extract one or more files")
        return False
    
    # Organize dataset
    organization_success = organize_dataset(temp_dir, output_dir)
    
    if organization_success:
        logger.info(f"GuitarSet setup complete. Dataset available at: {output_dir}")
        logger.info(f"Use this path with the training script: --data-dir {output_dir}")
        
        # Clean up temporary files if requested
        if clean_temp and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary files in {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                logger.info("Temporary files cleaned up successfully")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")
        
        return True
    else:
        logger.error("GuitarSet setup failed")
        return False

def main():
    """Main function to set up GuitarSet dataset."""
    args = parse_args()
    
    # Convert to absolute paths
    output_dir = os.path.abspath(args.output_dir)
    temp_dir = os.path.abspath(args.temp_dir)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Temporary directory: {temp_dir}")
    
    try:
        setup_guitarset(output_dir, temp_dir, args.force, args.skip_mono, args.clean_temp)
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error setting up GuitarSet: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()