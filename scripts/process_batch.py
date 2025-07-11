#!/usr/bin/env python3
"""
Batch processing script for AxTone
Usage: python process_batch.py [input_directory] [output_directory]
"""

import os
import sys

def process_batch(input_dir, output_dir):
    """
    Process all audio files in a directory
    """
    # TODO: Implement batch processing logic
    print(f"Processing files from {input_dir} to {output_dir}")
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_batch.py [input_directory] [output_directory]")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    process_batch(input_dir, output_dir)