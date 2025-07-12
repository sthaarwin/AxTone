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
- Neural network-based guitar tab transcription trained on GuitarSet
- Hexaphonic audio processing for advanced multi-string detection
- Visualization of generated tablature

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
│       ├── acoustic/     # Tabs for acoustic guitar
│       └── electric/     # Tabs for electric guitar
│
├── src/                  # Main code
│   ├── core/             # Pipeline stages
│   │   ├── ai/           # AI-based tab generation
│   │   │   └── models/   # Neural network models
│   │   └── tab_generator.py  # Tab generation utilities
│   ├── data/             # Dataset handling
│   ├── evaluation/       # Quality metrics
│   ├── utils/            # Helper functions
│   └── visualization/    # Tab visualization tools
│
├── models/               # ML models
│   ├── pretrained/       # Downloaded models
│   └── trained/          # Custom trained models
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

# Download GuitarSet dataset for training (optional)
mkdir -p datasets
cd datasets
git clone https://github.com/marl/guitarset.git
cd ..
```

## Usage

### Basic Usage

```bash
# Process a single audio file with traditional approach
python main.py process --input data/raw/test.mp3 --output data/outputs/test.tab

# Process a single file using the AI pipeline
python main.py process --input data/raw/test.mp3 --output data/outputs/test.tab --ai

# Process a directory of files
python main.py batch --input data/raw --output data/outputs

# Process a directory using the AI pipeline
python main.py batch --input data/raw --output data/outputs --ai
```

### Guitar Tab Generation with Neural Networks

```bash
# Train a tab transcription model on GuitarSet
python scripts/train_model.py --data-dir datasets/guitarset --output-dir models/trained --epochs 100

# Generate tabs from a guitar recording using a trained model
python scripts/generate_tab.py --input-files your_guitar_recording.wav --model-path models/trained/tab_model_TIMESTAMP/models/best_model.pt --visualize

# Generate tabs with customized parameters
python scripts/generate_tab.py --input-files your_guitar_recording.wav --model-path models/trained/tab_model_TIMESTAMP/models/best_model.pt --threshold 0.4 --resolution 0.25 --min-note-duration 0.1 --export-json --visualize
```

## Guitar Tab Transcription System

The neural network-based tab transcription system uses the GuitarSet dataset to train models that can convert guitar audio into tablature. Key components include:

### Data Processing
- **GuitarSet Dataset**: Uses hexaphonic audio (one channel per string) and detailed annotations
- **Feature Extraction**: Converts audio to mel-spectrograms for each string
- **Annotation Alignment**: Aligns note onset, pitch, string, and fret data with audio frames

### Models
- **TabTranscriptionModel**: CNN-LSTM architecture for processing spectrograms and predicting string/fret positions
- **TabCRNN**: Alternative Convolutional RNN model for tab transcription experiments

### Evaluation
- **String/Fret Accuracy**: Measures the accuracy of string and fret predictions
- **Timing Error**: Evaluates the timing precision of note onsets
- **F1 Score**: Balances precision and recall for comprehensive evaluation

### Output Formats
- **ASCII Tab**: Readable guitar tablature format
- **JSON**: Structured format with timing, string, and fret information
- **Visualization**: Visual representation of the generated tab

## Cleaning Up

You can clean up generated data files while preserving raw inputs:

```bash
# Clean all generated data (will prompt for confirmation)
python scripts/clean_data.py

# Force cleaning without confirmation
python scripts/clean_data.py --force

# Keep final output tabs while cleaning intermediate files
python scripts/clean_data.py --keep-outputs

# Keep trained models when cleaning other data
python scripts/clean_data.py --keep-models

# Clean Python cache files and notebook checkpoints
python scripts/clean_data.py --clean-cache --clean-notebooks
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [GuitarSet](https://github.com/marl/guitarset) - Dataset for guitar transcription
- [librosa](https://librosa.org/) - Audio processing library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [JAMS](https://github.com/marl/jams) - JSON Annotated Music Specification
