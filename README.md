# AxTone

An AI-powered system for generating musical tablature from audio files.

## Overview

AxTone uses machine learning and signal processing to automatically convert audio recordings of guitar (and other string instruments) into accurate tablature notation. The system analyzes audio to identify notes, timing, and playing techniques, then generates readable tablature.

## Features

- Audio file processing and feature extraction
- Note detection and pitch tracking
- String and fret assignment
- Tab notation generation in multiple formats
- Support for various string instruments
- AI-powered tablature generation pipeline

## Project Structure

```
AxTone/
│
├── data/                 # Input/Output
│   ├── raw/              # Raw audio files
│   ├── processed/        # Processed intermediates
│   │   ├── stems/        # Separated instruments
│   │   ├── features/     # Extracted features
│   │   └── midi/         # Intermediate MIDI
│   └── outputs/          # Final tabs in all formats
│
├── src/                  # Main code
│   ├── core/             # Pipeline stages
│   │   └── ai/           # AI-based tab generation
│   ├── utils/            # Helper functions
│   ├── evaluation/       # Quality metrics
│   └── visualization/    # Debugging tools
│
├── models/               # ML models
│   ├── pretrained/       # Downloaded models
│   └── trained/          # custom trained models
│
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── docs/                 # Documentation
├── notebooks/            # Experimental code
│
├── configs/              # Configuration files
├── requirements/         # Split requirements
│   ├── base.txt          # Core deps
│   ├── dev.txt           # Development tools
│   └── ml.txt            # ML-specific
│
├── main.py               # CLI entry point
└── README.md 
```

## Installation

```bash
# Clone the repository
git clone https://github.com/sthaarwin/axtone.git
cd axtone

# Install dependencies
pip install -r requirements/base.txt
pip install -r requirements/ml.txt  # If using ML features

# Install development dependencies (optional)
pip install -r requirements/dev.txt
```

## Usage

```bash
# Process a single audio file with traditional approach
python main.py process --input data/raw/test.mp3 --output data/outputs/test.tab

# Process a single file using the AI pipeline
python main.py process --input data/raw/test.mp3 --output data/outputs/test.tab --ai

# Process a directory of files
python main.py batch --input data/raw --output data/outputs

# Process a directory using the AI pipeline
python main.py batch --input data/raw --output data/outputs --ai

# Train a custom model (requires training configuration)
python main.py train --config configs/training_config.yaml --output models/trained
```

## Cleaning Up

You can clean up generated data files while preserving raw inputs:

```bash
# Clean all generated data (will prompt for confirmation)
python scripts/clean_data.py

# Force cleaning without confirmation
python scripts/clean_data.py --force

# Keep final output tabs while cleaning intermediate files
python scripts/clean_data.py --keep-outputs
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
