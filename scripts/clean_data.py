#!/usr/bin/env python3
"""
Clean Data Script

This script cleans all generated data files in the AxTone project,
preserving only raw input files. This is useful when you want to
reset your workspace to reprocess files or start fresh.
"""

import os
import shutil
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("clean_data")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Clean generated data files in the AxTone project.'
    )
    parser.add_argument(
        '--keep-outputs', '-k',
        action='store_true',
        help='Keep final output tab files in data/outputs'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force deletion without confirmation'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Path to the data directory (default: data)'
    )
    return parser.parse_args()

def clean_data(data_dir, keep_outputs=False, force=False):
    """
    Clean all data files except those in the raw folder.
    
    Args:
        data_dir: Path to the data directory
        keep_outputs: Whether to preserve files in the outputs folder
        force: Whether to proceed without confirmation
    """
    # Ensure data directory exists
    if not os.path.isdir(data_dir):
        logger.error(f"Data directory '{data_dir}' not found")
        return False
    
    # Get list of folders to clean
    folders_to_clean = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item != 'raw':
            if item == 'outputs' and keep_outputs:
                logger.info(f"Skipping outputs folder as requested")
                continue
            folders_to_clean.append(item_path)
    
    if not folders_to_clean:
        logger.info("No folders to clean")
        return True
    
    # Display what will be cleaned
    logger.info("The following folders will be cleaned:")
    for folder in folders_to_clean:
        logger.info(f"  - {folder}")
    
    # Confirm action if not forced
    if not force:
        confirm = input("Do you want to proceed? [y/N] ")
        if not confirm.lower().startswith('y'):
            logger.info("Operation cancelled")
            return False
    
    # Clean each folder
    for folder in folders_to_clean:
        try:
            logger.info(f"Cleaning folder: {folder}")
            # Option 1: Remove the entire folder and recreate it
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
            logger.info(f"Recreated empty folder: {folder}")
        except Exception as e:
            logger.error(f"Error cleaning {folder}: {e}")
    
    logger.info("Cleaning complete")
    return True

if __name__ == "__main__":
    args = parse_args()
    clean_data(args.data_dir, args.keep_outputs, args.force)