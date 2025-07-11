#!/usr/bin/env python3
"""
AxTone: Main entry point for the application.
This script provides a command-line interface for tab generation.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core modules
from src.core.audio_processor import AudioProcessor
from src.core.tab_generator import TabGenerator
# Import the new AI pipeline
from src.core.ai import TabAIPipeline

def setup_arg_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='AxTone - AI-powered guitar tablature generator'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process a single file
    process_parser = subparsers.add_parser('process', help='Process a single audio file')
    process_parser.add_argument('--input', '-i', required=True, help='Input audio file path')
    process_parser.add_argument('--output', '-o', help='Output tab file path')
    process_parser.add_argument('--model', '-m', default='default', help='Model to use')
    process_parser.add_argument('--ai', action='store_true', help='Use AI pipeline instead of traditional approach')
    
    # Process a batch of files
    batch_parser = subparsers.add_parser('batch', help='Process a directory of audio files')
    batch_parser.add_argument('--input', '-i', required=True, help='Input directory path')
    batch_parser.add_argument('--output', '-o', required=True, help='Output directory path')
    batch_parser.add_argument('--model', '-m', default='default', help='Model to use')
    batch_parser.add_argument('--ai', action='store_true', help='Use AI pipeline instead of traditional approach')
    
    # Train a new model
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--config', '-c', required=True, help='Training configuration file')
    train_parser.add_argument('--output', '-o', required=True, help='Output directory for the model')
    
    return parser

def load_config(model_name='default'):
    """Load configuration from file."""
    config_path = os.path.join('configs', f'{model_name}_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration")
        return {
            'audio': {
                'sample_rate': 44100,
                'hop_length': 512
            },
            'tab': {
                'instruments': [
                    {
                        'name': 'guitar',
                        'strings': 6,
                        'tuning': ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
                    }
                ]
            },
            'paths': {
                'raw_data': 'data/raw/',
                'processed_stems': 'data/processed/stems/',
                'features': 'data/processed/features/',
                'midi_output': 'data/processed/midi/',
                'tab_output': 'data/outputs/'
            }
        }

def process_file(input_path, output_path, model_name, use_ai=False):
    """Process a single audio file and generate tab."""
    config = load_config(model_name)
    
    if use_ai:
        # Use the AI pipeline
        pipeline = TabAIPipeline(config)
        return pipeline.process_file(input_path, output_path)
    else:
        # Use the traditional approach
        audio_processor = AudioProcessor(config)
        tab_generator = TabGenerator(config)
        
        # Process the audio
        features = audio_processor.process_file(input_path)
        
        # Generate the tab
        tab = tab_generator.generate_tab(features)
        
        # Export the tab
        if output_path is None:
            filename = os.path.basename(input_path)
            output_path = os.path.join(
                config['paths']['tab_output'],
                f"{os.path.splitext(filename)[0]}.tab"
            )
        
        tab_generator.export_tab(tab, format='txt', output_path=output_path)
        logger.info(f"Tab exported to {output_path}")
        return output_path

def process_batch(input_dir, output_dir, model_name, use_ai=False):
    """Process all audio files in a directory."""
    config = load_config(model_name)
    
    if use_ai:
        # Use the AI pipeline
        pipeline = TabAIPipeline(config)
        return pipeline.process_batch(input_dir, output_dir)
    else:
        # Use the traditional approach
        audio_processor = AudioProcessor(config)
        tab_generator = TabGenerator(config)
        
        output_files = []
        
        for filename in os.listdir(input_dir):
            if filename.endswith(('.wav', '.mp3', '.ogg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(
                    output_dir, 
                    f"{os.path.splitext(filename)[0]}.tab"
                )
                
                try:
                    # Process the audio
                    features = audio_processor.process_file(input_path)
                    
                    # Generate the tab
                    tab = tab_generator.generate_tab(features)
                    
                    # Export the tab
                    tab_generator.export_tab(tab, format='txt', output_path=output_path)
                    logger.info(f"Tab exported to {output_path}")
                    output_files.append(output_path)
                except Exception as e:
                    logger.error(f"Error processing {input_path}: {e}")
        
        return output_files

def train_model(config_path, output_dir):
    """Train a new model using the provided configuration."""
    try:
        with open(config_path, 'r') as f:
            training_config = yaml.safe_load(f)
        
        # This would be implemented with actual training code
        logger.info(f"Training new model with config {config_path}")
        logger.info(f"Model will be saved to {output_dir}")
        
        # Placeholder for actual training code
        logger.warning("Model training not yet implemented")
        
        return None
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

def main():
    """Main entry point for the application."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    if args.command == 'process':
        process_file(args.input, args.output, args.model, args.ai)
    elif args.command == 'batch':
        process_batch(args.input, args.output, args.model, args.ai)
    elif args.command == 'train':
        train_model(args.config, args.output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()