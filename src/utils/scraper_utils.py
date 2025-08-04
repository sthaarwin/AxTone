"""
Updated tab scraper utilities integrated with AxTone project.

This module extends the existing tab_scraper.py in the utils directory
to work seamlessly with the AxTone project structure.
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TabScraperConfig:
    """Configuration manager for the tab scraper."""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration."""
        if config_file is None:
            config_file = "configs/tab_scraper_config.yaml"
        
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file or create default."""
        default_config = {
            'scraper': {
                'headless': True,
                'download_timeout': 15,
                'request_delay': 1,
                'max_pages_per_query': 5,
                'default_tab_types': ['Tabs', 'Chords']
            },
            'processing': {
                'max_sequence_length': 1024,
                'min_sequence_length': 10,
                'standard_tuning': ['E', 'A', 'D', 'G', 'B', 'E']
            },
            'dataset': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1
            },
            'output': {
                'base_dir': 'data',
                'scraped_dir': 'scraped_tabs',
                'processed_dir': 'processed_tabs'
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                import yaml
                with open(self.config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                # Merge with defaults
                self._deep_update(default_config, loaded_config)
                return default_config
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, section: str, key: str = None, default=None):
        """Get configuration value."""
        if key is None:
            return self.config.get(section, default)
        return self.config.get(section, {}).get(key, default)
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            import yaml
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")


def clean_scraped_tabs(scraped_dir: str) -> Dict:
    """
    Clean and organize scraped tabs.
    
    Args:
        scraped_dir: Directory containing scraped tabs
        
    Returns:
        Cleaning statistics
    """
    scraped_path = Path(scraped_dir)
    
    if not scraped_path.exists():
        logger.error(f"Scraped directory {scraped_dir} does not exist")
        return {'error': 'Directory not found'}
    
    stats = {
        'total_files': 0,
        'cleaned_files': 0,
        'removed_files': 0,
        'renamed_files': 0
    }
    
    # Process each subdirectory
    for tab_type_dir in scraped_path.iterdir():
        if not tab_type_dir.is_dir():
            continue
            
        logger.info(f"Cleaning {tab_type_dir.name} directory")
        
        for file_path in tab_type_dir.iterdir():
            if file_path.suffix not in ['.tab', '.json', '.png']:
                continue
                
            stats['total_files'] += 1
            
            # Check file size (remove very small files)
            if file_path.stat().st_size < 100:  # Less than 100 bytes
                file_path.unlink()
                stats['removed_files'] += 1
                logger.debug(f"Removed small file: {file_path.name}")
                continue
            
            # Clean filename if needed
            original_name = file_path.name
            clean_name = _clean_filename(original_name)
            
            if clean_name != original_name:
                new_path = file_path.parent / clean_name
                if not new_path.exists():
                    file_path.rename(new_path)
                    stats['renamed_files'] += 1
                    logger.debug(f"Renamed: {original_name} -> {clean_name}")
                else:
                    # If target exists, remove the duplicate
                    file_path.unlink()
                    stats['removed_files'] += 1
                    logger.debug(f"Removed duplicate: {original_name}")
                    continue
            
            stats['cleaned_files'] += 1
    
    logger.info(f"Cleaning complete: {stats}")
    return stats


def _clean_filename(filename: str) -> str:
    """Clean a filename for better organization."""
    # Remove problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace multiple spaces with single space
    filename = re.sub(r'\s+', ' ', filename)
    # Remove leading/trailing whitespace
    filename = filename.strip()
    # Limit length
    name_part, ext = os.path.splitext(filename)
    if len(name_part) > 100:
        name_part = name_part[:100]
    return name_part + ext


def validate_dataset(dataset_dir: str) -> Dict:
    """
    Validate a generated dataset for training readiness.
    
    Args:
        dataset_dir: Directory containing the dataset
        
    Returns:
        Validation report
    """
    dataset_path = Path(dataset_dir)
    
    validation_report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check directory structure
    required_dirs = ['scraped_tabs', 'processed_tabs']
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            validation_report['errors'].append(f"Missing directory: {dir_name}")
            validation_report['valid'] = False
    
    # Check for processed splits
    processed_dir = dataset_path / 'processed_tabs'
    if processed_dir.exists():
        split_dirs = ['train', 'val', 'test']
        for split_dir in split_dirs:
            split_path = processed_dir / split_dir
            if not split_path.exists():
                validation_report['warnings'].append(f"Missing split directory: {split_dir}")
            else:
                # Count files in each split
                tensor_files = list(split_path.glob("*.pt"))
                validation_report['statistics'][f'{split_dir}_files'] = len(tensor_files)
    
    # Check metadata files
    metadata_files = ['dataset_report.json', 'processing_metadata.json']
    for metadata_file in metadata_files:
        metadata_path = dataset_path / metadata_file
        if not metadata_path.exists():
            validation_report['warnings'].append(f"Missing metadata file: {metadata_file}")
    
    # Validate tensor files
    processed_dir = dataset_path / 'processed_tabs'
    if processed_dir.exists():
        tensor_files = list(processed_dir.glob("*.pt"))
        if len(tensor_files) == 0:
            validation_report['errors'].append("No processed tensor files found")
            validation_report['valid'] = False
        else:
            # Sample a few files to check format
            import torch
            valid_tensors = 0
            for tensor_file in tensor_files[:5]:  # Check first 5 files
                try:
                    data = torch.load(tensor_file)
                    if 'tensor' in data and 'metadata' in data:
                        valid_tensors += 1
                    else:
                        validation_report['warnings'].append(f"Invalid tensor format: {tensor_file.name}")
                except Exception as e:
                    validation_report['warnings'].append(f"Failed to load tensor: {tensor_file.name}")
            
            validation_report['statistics']['total_tensor_files'] = len(tensor_files)
            validation_report['statistics']['valid_tensor_files_sampled'] = valid_tensors
    
    # Overall validation
    if len(validation_report['errors']) == 0:
        validation_report['valid'] = True
        logger.info("Dataset validation passed")
    else:
        validation_report['valid'] = False
        logger.error(f"Dataset validation failed with {len(validation_report['errors'])} errors")
    
    return validation_report


def create_dataset_summary(dataset_dir: str) -> Dict:
    """
    Create a comprehensive summary of the dataset.
    
    Args:
        dataset_dir: Directory containing the dataset
        
    Returns:
        Dataset summary
    """
    dataset_path = Path(dataset_dir)
    
    summary = {
        'dataset_path': str(dataset_path),
        'created_at': None,
        'scraped_stats': {},
        'processed_stats': {},
        'splits': {},
        'total_files': 0,
        'total_size_mb': 0
    }
    
    # Load metadata if available
    report_file = dataset_path / 'dataset_report.json'
    if report_file.exists():
        with open(report_file, 'r') as f:
            report_data = json.load(f)
            summary.update(report_data.get('summary', {}))
    
    # Calculate directory sizes
    for dir_path in dataset_path.rglob('*'):
        if dir_path.is_file():
            summary['total_files'] += 1
            summary['total_size_mb'] += dir_path.stat().st_size / (1024 * 1024)
    
    # Count files in each split
    processed_dir = dataset_path / 'processed_tabs'
    if processed_dir.exists():
        for split_name in ['train', 'val', 'test']:
            split_dir = processed_dir / split_name
            if split_dir.exists():
                tensor_files = list(split_dir.glob("*.pt"))
                summary['splits'][split_name] = len(tensor_files)
    
    summary['total_size_mb'] = round(summary['total_size_mb'], 2)
    
    return summary


# Integration functions for AxTone project
def integrate_with_guitarset(scraped_dataset_dir: str, guitarset_dir: str, output_dir: str) -> Dict:
    """
    Integrate scraped dataset with existing GuitarSet data.
    
    Args:
        scraped_dataset_dir: Directory containing scraped tabs
        guitarset_dir: Directory containing GuitarSet dataset
        output_dir: Directory to save integrated dataset
        
    Returns:
        Integration statistics
    """
    logger.info("Starting dataset integration with GuitarSet")
    
    # This is a placeholder for now - would need to implement
    # logic to combine scraped tabs with audio data from GuitarSet
    
    integration_stats = {
        'scraped_files': 0,
        'guitarset_files': 0,
        'matched_pairs': 0,
        'output_files': 0
    }
    
    # Count files in each dataset
    scraped_path = Path(scraped_dataset_dir)
    if scraped_path.exists():
        integration_stats['scraped_files'] = len(list(scraped_path.rglob("*.tab")))
    
    guitarset_path = Path(guitarset_dir)
    if guitarset_path.exists():
        integration_stats['guitarset_files'] = len(list(guitarset_path.rglob("*.wav")))
    
    logger.info(f"Integration stats: {integration_stats}")
    return integration_stats


if __name__ == "__main__":
    # Example usage
    config = TabScraperConfig()
    print("Loaded configuration:")
    print(json.dumps(config.config, indent=2))