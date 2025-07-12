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
import glob

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
        '--keep-models', '-m',
        action='store_true',
        help='Keep trained models in models/trained'
    )
    parser.add_argument(
        '--clean-cache', '-c',
        action='store_true',
        help='Clean __pycache__ directories and .pyc files'
    )
    parser.add_argument(
        '--clean-notebooks', '-n',
        action='store_true',
        help='Clean notebook checkpoints and outputs'
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


def clean_data(data_dir, keep_outputs=False, keep_models=False, clean_cache=False, 
               clean_notebooks=False, force=False):
    """
    Clean all data files except those in the raw folder.
    
    Args:
        data_dir: Path to the data directory
        keep_outputs: Whether to preserve files in the outputs folder
        keep_models: Whether to preserve trained models
        clean_cache: Whether to clean Python cache files
        clean_notebooks: Whether to clean notebook checkpoints
        force: Whether to proceed without confirmation
    """
    # Ensure data directory exists
    if not os.path.isdir(data_dir):
        logger.error(f"Data directory '{data_dir}' not found")
        return False
    
    # Get list of folders to clean in data directory
    folders_to_clean = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item != 'raw':
            if item == 'outputs' and keep_outputs:
                logger.info(f"Skipping outputs folder as requested")
                # Ensure electric and acoustic folders exist in outputs
                electric_path = os.path.join(item_path, 'electric')
                acoustic_path = os.path.join(item_path, 'acoustic')
                os.makedirs(electric_path, exist_ok=True)
                os.makedirs(acoustic_path, exist_ok=True)
                logger.info(f"Ensured 'electric' and 'acoustic' folders exist in outputs")
                continue
            folders_to_clean.append(item_path)
    
    # Add models/trained to cleaning list if not keeping models
    if not keep_models:
        models_dir = os.path.join(os.path.dirname(data_dir), 'models', 'trained')
        if os.path.isdir(models_dir):
            folders_to_clean.append(models_dir)
    
    # Find all __pycache__ directories if cleaning cache
    cache_files = []
    if clean_cache:
        for root, dirs, files in os.walk(os.path.dirname(data_dir)):
            # Add __pycache__ directories
            for dir in dirs:
                if dir == '__pycache__':
                    cache_files.append(os.path.join(root, dir))
            # Add .pyc files
            for file in files:
                if file.endswith('.pyc'):
                    cache_files.append(os.path.join(root, file))
    
    # Find notebook checkpoints if cleaning notebooks
    notebook_files = []
    if clean_notebooks:
        # Find .ipynb_checkpoints directories
        for root, dirs, files in os.walk(os.path.dirname(data_dir)):
            for dir in dirs:
                if dir == '.ipynb_checkpoints':
                    notebook_files.append(os.path.join(root, dir))
    
    if not folders_to_clean and not cache_files and not notebook_files:
        logger.info("No folders or files to clean")
        return True
    
    # Display what will be cleaned
    logger.info("The following items will be cleaned:")
    for folder in folders_to_clean:
        logger.info(f"  - {folder}")
    
    if cache_files:
        logger.info("Python cache files to be cleaned:")
        for cache in cache_files[:5]:  # Show only first 5 to avoid overwhelming output
            logger.info(f"  - {cache}")
        if len(cache_files) > 5:
            logger.info(f"  - ... and {len(cache_files) - 5} more")
    
    if notebook_files:
        logger.info("Notebook checkpoints to be cleaned:")
        for nb in notebook_files:
            logger.info(f"  - {nb}")
    
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
            # Remove the entire folder and recreate it
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
            logger.info(f"Recreated empty folder: {folder}")
        except Exception as e:
            logger.error(f"Error cleaning {folder}: {e}")
    
    # Clean cache files
    for cache in cache_files:
        try:
            if os.path.isdir(cache):
                shutil.rmtree(cache)
                logger.info(f"Removed cache directory: {cache}")
            else:
                os.remove(cache)
                logger.info(f"Removed cache file: {cache}")
        except Exception as e:
            logger.error(f"Error cleaning {cache}: {e}")
    
    # Clean notebook files
    for nb in notebook_files:
        try:
            shutil.rmtree(nb)
            logger.info(f"Removed notebook checkpoint: {nb}")
        except Exception as e:
            logger.error(f"Error cleaning {nb}: {e}")
    
    logger.info("Cleaning complete")
    return True


if __name__ == "__main__":
    args = parse_args()
    clean_data(args.data_dir, args.keep_outputs, args.keep_models, 
               args.clean_cache, args.clean_notebooks, args.force)