#!/usr/bin/env python
"""
AxTone Improvements Demo

This script demonstrates the major improvements made to the AxTone project:
1. Enhanced source separation
2. Custom model training capabilities
3. Advanced music theory-based evaluation metrics
4. Improved tablature visualization

Usage:
    python demo_improvements.py --input <input_audio_file> --output <output_dir>
"""

import os
import sys
import argparse
import logging
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import time

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("axtone_demo")

# Import project modules
from src.core.ai.tab_ai_pipeline import TabAIPipeline
from src.utils.source_separation import GuitarSourceSeparator
from src.evaluation.metrics import evaluate_tab, TabMetrics
from src.visualization.tab_visualizer import TabVisualizer, visualize_tab, compare_tabs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Demonstrate AxTone improvements'
    )
    
    parser.add_argument('--input', '-i', required=True,
                      help='Path to input audio file')
    parser.add_argument('--output', '-o', required=True,
                      help='Output directory for generated files')
    parser.add_argument('--config', '-c', default='configs/default_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--separation', '-s', default='spectral',
                      choices=['spectral', 'demucs', 'spleeter'],
                      help='Source separation method to use')
    parser.add_argument('--reference', '-r',
                      help='Path to reference tab file (optional)')
    parser.add_argument('--visualize', '-v', action='store_true',
                      help='Generate visualizations')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def demo_source_separation(input_file, output_dir, method='spectral'):
    """
    Demonstrate the improved source separation.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save output files
        method: Source separation method to use
        
    Returns:
        Path to separated guitar audio
    """
    logger.info(f"Demonstrating source separation using {method} method")
    
    # Create a source separator
    separator = GuitarSourceSeparator(method=method)
    
    # Create output directory
    separation_dir = os.path.join(output_dir, 'separation')
    os.makedirs(separation_dir, exist_ok=True)
    
    # Separate guitar track
    guitar_path = separator.separate(input_file, separation_dir)
    
    logger.info(f"Source separation complete: {guitar_path}")
    return guitar_path


def demo_tab_generation(input_file, output_dir, config):
    """
    Demonstrate the tab generation pipeline with improvements.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save output files
        config: Configuration dictionary
        
    Returns:
        Path to generated tab
    """
    logger.info("Demonstrating tab generation pipeline")
    
    # Update config with output directories
    config['paths'] = config.get('paths', {})
    config['paths']['processed_stems'] = os.path.join(output_dir, 'stems')
    config['paths']['features'] = os.path.join(output_dir, 'features')
    config['paths']['midi_output'] = os.path.join(output_dir, 'midi')
    config['paths']['tab_output'] = os.path.join(output_dir, 'tabs')
    
    # Create pipeline
    pipeline = TabAIPipeline(config)
    
    # Process file
    tab_path = pipeline.process_file(input_file)
    
    logger.info(f"Tab generation complete: {tab_path}")
    return tab_path


def demo_evaluation(generated_tab, reference_tab=None):
    """
    Demonstrate the improved evaluation metrics.
    
    Args:
        generated_tab: Path to generated tab file
        reference_tab: Path to reference tab file (optional)
        
    Returns:
        Evaluation metrics
    """
    logger.info("Demonstrating evaluation metrics")
    
    if reference_tab is None or not os.path.exists(reference_tab):
        logger.warning("No reference tab provided, using self-evaluation")
        # Create a modified version of the generated tab for demonstration
        with open(generated_tab, 'r') as f:
            tab_content = f.read()
        
        # Create a temporary reference with some modifications
        reference_tab = os.path.join(os.path.dirname(generated_tab), 'temp_reference.tab')
        with open(reference_tab, 'w') as f:
            # Modify some fret numbers to simulate differences
            modified_content = tab_content.replace('3', '4').replace('5', '6')
            f.write(modified_content)
    
    # Evaluate the tab
    metrics = evaluate_tab(generated_tab, reference_tab)
    
    # Print metrics
    logger.info("Evaluation results:")
    logger.info(f"  Note precision: {metrics.note_precision:.4f}")
    logger.info(f"  Note recall: {metrics.note_recall:.4f}")
    logger.info(f"  Note F1 score: {metrics.note_f1:.4f}")
    logger.info(f"  Timing accuracy: {metrics.timing_accuracy:.4f}")
    logger.info(f"  Pitch accuracy: {metrics.pitch_accuracy:.4f}")
    logger.info(f"  Fret distance: {metrics.fret_distance:.4f}")
    logger.info(f"  Playability score: {metrics.playability_score:.4f}")
    logger.info(f"  Overall score: {metrics.overall_score:.4f}")
    
    return metrics


def demo_visualization(tab_path, midi_path, output_dir):
    """
    Demonstrate the improved tab visualization.
    
    Args:
        tab_path: Path to tab file
        midi_path: Path to MIDI file
        output_dir: Directory to save output files
        
    Returns:
        Paths to visualization files
    """
    logger.info("Demonstrating tab visualization")
    
    # Create output directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate visualizations in different formats
    text_output = visualize_tab(
        tab_path, 
        output_format='text'
    )
    
    image_path = os.path.join(viz_dir, 'tab_image.png')
    image = visualize_tab(
        tab_path, 
        output_format='image',
        output_path=image_path
    )
    
    html_path = os.path.join(viz_dir, 'tab_interactive.html')
    html = visualize_tab(
        tab_path, 
        output_format='html',
        output_path=html_path,
        midi_path=midi_path
    )
    
    # Save text visualization
    text_path = os.path.join(viz_dir, 'tab_text.txt')
    with open(text_path, 'w') as f:
        f.write(text_output)
    
    # Generate comparison visualization if needed
    comparison_path = None
    reference_tab = os.path.join(os.path.dirname(tab_path), 'temp_reference.tab')
    if os.path.exists(reference_tab):
        comparison_path = os.path.join(viz_dir, 'tab_comparison.png')
        comparison = compare_tabs(
            tab_path,
            reference_tab,
            output_path=comparison_path
        )
    
    logger.info(f"Visualization files saved to {viz_dir}")
    return {
        'text': text_path,
        'image': image_path,
        'html': html_path,
        'comparison': comparison_path
    }


def demo_model_training_setup():
    """
    Demonstrate the model training setup.
    
    Note: This only shows the configuration and setup, but doesn't
    actually train a model, which requires a dataset.
    """
    logger.info("Demonstrating model training setup")
    
    # Example training configuration
    training_config = {
        'dataset': {
            'dataset_dir': 'data/dataset',
            'split_dataset': True,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'batch_size': 16,
            'num_workers': 4
        },
        'model': {
            'type': 'transformer',
            'input_dim': 128,
            'num_strings': 6,
            'max_fret': 24,
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1
        },
        'training': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'scheduler': 'reducelronplateau',
            'scheduler_factor': 0.5,
            'scheduler_patience': 5,
            'num_epochs': 100,
            'clip_grad_norm': True,
            'clip_grad_value': 1.0,
            'log_interval': 10
        }
    }
    
    # Print training configuration
    logger.info("Example training configuration:")
    logger.info(f"  Model type: {training_config['model']['type']}")
    logger.info(f"  Input dimension: {training_config['model']['input_dim']}")
    logger.info(f"  Batch size: {training_config['dataset']['batch_size']}")
    logger.info(f"  Learning rate: {training_config['training']['learning_rate']}")
    logger.info(f"  Number of epochs: {training_config['training']['num_epochs']}")
    
    # Command to run training
    train_cmd = "python scripts/train_model.py --config training_config.yaml --output models/trained"
    logger.info(f"To train a model, you would run: {train_cmd}")
    
    return training_config


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Demo 1: Source Separation
    guitar_path = demo_source_separation(args.input, args.output, args.separation)
    
    # Update config with source separation settings
    config['audio'] = config.get('audio', {})
    config['audio']['source_separation'] = {
        'enabled': True,
        'method': args.separation
    }
    
    # Demo 2: Tab Generation with the improved pipeline
    tab_path = demo_tab_generation(args.input, args.output, config)
    
    # Find MIDI file
    midi_dir = os.path.join(args.output, 'midi')
    midi_files = [os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith('.mid')]
    midi_path = midi_files[0] if midi_files else None
    
    # Demo 3: Evaluation
    metrics = demo_evaluation(tab_path, args.reference)
    
    # Demo 4: Visualization (if requested)
    viz_paths = None
    if args.visualize and midi_path:
        viz_paths = demo_visualization(tab_path, midi_path, args.output)
    
    # Demo 5: Model Training Setup
    training_config = demo_model_training_setup()
    
    # End timing
    end_time = time.time()
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("AxTone Improvements Demo Summary")
    logger.info("="*50)
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Source separation method: {args.separation}")
    logger.info(f"Generated tab: {tab_path}")
    logger.info(f"Overall quality score: {metrics.overall_score:.4f}")
    
    if viz_paths:
        logger.info(f"Visualizations saved to: {os.path.dirname(viz_paths['text'])}")
    
    logger.info("="*50)
    logger.info("Demo complete!")


if __name__ == "__main__":
    main()