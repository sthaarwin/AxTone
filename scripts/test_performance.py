#!/usr/bin/env python3
"""
Test script to benchmark the performance improvements in AxTone
Compares performance between original and optimized versions for tab generation
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main pipeline
from src.core.ai.tab_ai_pipeline import TabAIPipeline
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        sys.exit(1)

def create_shorter_sample(input_file, output_file, duration=10):
    """Create a shorter audio sample for testing."""
    import librosa
    import soundfile as sf
    
    # Load the audio file
    logger.info(f"Loading audio file: {input_file}")
    y, sr = librosa.load(input_file, sr=None)
    
    # Trim to the specified duration
    samples = int(duration * sr)
    if len(y) > samples:
        y = y[:samples]
    
    # Save the trimmed audio
    logger.info(f"Saving trimmed audio to: {output_file}")
    sf.write(output_file, y, sr)
    
    return output_file

def process_file(input_file, output_file, config):
    """Process a single file and measure the time it takes."""
    # Initialize the pipeline
    pipeline = TabAIPipeline(config)
    
    # Process the file and measure time
    start_time = time.time()
    output_path = pipeline.process_file(input_file, output_file)
    end_time = time.time()
    
    processing_time = end_time - start_time
    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    
    return processing_time, output_path

def main():
    parser = argparse.ArgumentParser(description='Benchmark AxTone performance improvements')
    parser.add_argument('--input', '-i', help='Input audio file path', default='datasets/guitarset/audio_mono-pickup_mix/03_Jazz1-130-D_solo_mix.wav')
    parser.add_argument('--duration', '-d', type=int, help='Duration in seconds for the test sample', default=10)
    parser.add_argument('--model', '-m', default='default', help='Model to use')
    args = parser.parse_args()
    
    # Load the configuration
    config = load_config(args.model)
    
    # Create directories for test outputs
    os.makedirs('data/test_performance', exist_ok=True)
    
    # Create a shorter sample for testing
    test_sample_path = os.path.join('data/test_performance', 'test_sample.wav')
    create_shorter_sample(args.input, test_sample_path, args.duration)
    
    # Process the sample file
    output_path = os.path.join('data/test_performance', 'test_output.txt')
    processing_time, actual_output_path = process_file(test_sample_path, output_path, config)
    
    logger.info(f"Performance test completed")
    logger.info(f"Input file: {test_sample_path} (duration: {args.duration}s)")
    logger.info(f"Output file: {actual_output_path}")
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    
    # Provide some context about the performance
    if processing_time < 60:
        logger.info("Performance looks good! The processing is running efficiently.")
    elif processing_time < 120:
        logger.info("Performance is acceptable. Further optimizations might be possible.")
    else:
        logger.info("Processing is still taking some time. Consider additional optimizations.")

if __name__ == '__main__':
    main()