# AxTone 🎸

A system for converting audio melodies into optimized guitar tablature using advanced pitch detection and graph algorithms.

## Overview

AxTone is a complete vocal-to-guitar-tab converter that extracts MIDI notes from audio files and generates optimized, playable guitar tablature using **Dijkstra's Algorithm**. The system:

1. **Extracts pitch** from audio files (MP3, WAV) using PYIN pitch detection
2. **Optimizes fingering** by treating the guitar fretboard as a graph where each possible finger position is a node, and transitions between positions have costs based on playability
3. **Generates tablature** in readable ASCII format with performance statistics
4. **Web interface** with interactive fretboard and audio playback

## Features

- 🎵 **Audio-to-MIDI Conversion**: Extract melodies from audio files (MP3, WAV) using PYIN pitch detection
- 🎯 **Graph-Based Optimization**: Uses Dijkstra's algorithm to find the easiest fingering path
- 🎵 **Intelligent Cost Function**: Considers fret distance, string jumps, stretch penalties, and open string preferences
- 🎸 **Multiple Tuning Support**: Works with standard tuning and custom tunings (Drop-D, Drop-C, Open-G, DADGAD)
- 📊 **ASCII Tablature Output**: Generates readable guitar tab format
- 🔧 **Customizable Parameters**: Adjust cost weights, minimum note duration, and preprocessing options
- 📈 **Performance Statistics**: Shows average fret movement and string jumps
- 💾 **MIDI Export**: Optionally save extracted MIDI files for further editing
- 🌐 **Web Interface**: Modern Next.js frontend with drag-and-drop file upload and interactive fretboard

## Installation

```bash
# Clone the repository
git clone https://github.com/sthaarwin/axtone.git
cd axtone

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command-Line Usage (Audio to Tab)

Convert any audio file to guitar tablature:

```bash
# Basic usage
python main.py vocals.mp3

# Specify output file
python main.py vocals.mp3 --output my_tab.txt

# Use Drop-D tuning
python main.py vocals.mp3 --tuning drop-d

# With audio preprocessing and MIDI export
python main.py vocals.mp3 --preprocess --save-midi --detailed
```

### Python API Usage (Direct MIDI)

```python
from src.fretboard_optimizer import FretboardOptimizer, MidiNote

# Create a simple melody (C major scale)
melody = [
    MidiNote(60, 0.0, 0.5),   # C4
    MidiNote(62, 0.5, 1.0),   # D4
    MidiNote(64, 1.0, 1.5),   # E4
    MidiNote(65, 1.5, 2.0),   # F4
    MidiNote(67, 2.0, 2.5),   # G4
    MidiNote(69, 2.5, 3.0),   # A4
    MidiNote(71, 3.0, 3.5),   # B4
    MidiNote(72, 3.5, 4.0),   # C5
]

# Optimize and generate tablature
optimizer = FretboardOptimizer()
path = optimizer.optimize(melody)

# Format and display
from src.formatter import TablatureFormatter
formatter = TablatureFormatter()
print(formatter.format(path, melody))
```

**Output:**
```
================================================================================
GUITAR TABLATURE (Optimized via Dijkstra's Algorithm)
================================================================================

e|--- 0--- 1--- 3--- 5--- 7--- 8---10---12---|
B|--- 1--- 3--- 5--- 6--- 8---10---12---13---|
G|--- 0--- 2--- 4--- 5--- 7--- 9---11---12---|
D|--- 2--- 4--- 5--- 7--- 9---10---12---14---|
A|--- 3--- 5--- 7--- 8---10---12---14---15---|
E|--- 0--- 2--- 3--- 5--- 7--- 8---10---12---|

================================================================================
Total notes: 8
Average fret movement: 1.71
Average string jumps: 0.00
================================================================================
```

## System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│ Audio File  │─────>│  Extractor   │─────>│ MIDI Notes  │
│ (.mp3/.wav) │      │  (AI Pitch)  │      │  Sequence   │
└─────────────┘      └──────────────┘      └─────────────┘
                                                   │
                                                   v
                                           ┌─────────────┐
                                           │  Optimizer  │
                                           │  (Dijkstra) │
                                           └─────────────┘
                                                   │
                                                   v
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Tab File   │<─────│  Formatter   │<─────│  Path Data  │
│  (.txt)     │      │  (ASCII)     │      │             │
└─────────────┘      └──────────────┘      └─────────────┘
```

## How It Works

### 1. Pitch Extraction (Audio → MIDI)

The system supports two pitch detection methods:

- **Basic Pitch** (Recommended): Spotify's neural network model for accurate polyphonic pitch detection
- **PYIN**: Probabilistic YIN algorithm for monophonic pitch tracking

Both methods convert audio to a sequence of `MidiNote(midi_number, start_time, end_time)` objects.

### 2. Graph Construction

For each MIDI note in the sequence, the system identifies all possible (string, fret) positions on the guitar:

- **Nodes**: Each possible finger position for a note
- **Edges**: Transitions between consecutive notes
- **Weights**: Cost based on difficulty of transition

### 2. Cost Function

The transition cost between two positions considers:

| Factor | Formula | Weight | Description |
|--------|---------|--------|-------------|
| **Fret Distance** | `|fret₁ - fret₂|` | 1.0 | Physical distance to move |
| **String Jump** | `|string₁ - string₂| × 0.5` | 0.5 | Penalty for changing strings |
| **Stretch Penalty** | `100.0` if `|fret₁ - fret₂| > 4` | 100.0 | Massive cost for unplayable stretches |
| **Open String Bonus** | `-0.3` if `fret₂ = 0` | -0.3 | Slight preference for open strings |

### 3. Dijkstra's Algorithm

The system finds the shortest path through all notes, minimizing total fingering difficulty:

```
START → Note1_Position → Note2_Position → ... → NoteN_Position → END
```

### 4. Tablature Formatting

The optimized path is formatted into ASCII tablature with:
- Standard 6-line guitar tab notation
- Fret numbers aligned to timing
- Performance statistics (fret movement, string jumps)
- Optional detailed note information (MIDI numbers, timing, positions)

## Advanced Usage

### Command-Line Options

```bash
# Full options
python main.py input.mp3 \
    --output tab.txt \              # Output file path
    --method basic_pitch \          # Pitch detection method
    --tuning drop-d \               # Guitar tuning preset
    --min-duration 0.1 \            # Minimum note duration (seconds)
    --preprocess \                  # Normalize and trim audio
    --save-midi \                   # Save extracted MIDI file
    --detailed                      # Include detailed note info
```

### Available Tunings

- **standard**: E2-A2-D3-G3-B3-E4 (40-45-50-55-59-64)
- **drop-d**: D2-A2-D3-G3-B3-E4 (38-45-50-55-59-64)
- **drop-c**: C2-G2-C3-F3-A3-D4 (36-43-48-53-57-62)
- **open-g**: D2-G2-D3-G3-B3-D4 (38-43-50-55-59-62)
- **dadgad**: D2-A2-D3-G3-A3-D4 (38-45-50-55-45-50)

Custom tunings via MIDI numbers:
```bash
python main.py input.mp3 --tuning "40,45,50,55,59,64"
```

### Custom Tuning (Python API)

```python
# Drop-D tuning: D2, A2, D3, G3, B3, E4
drop_d_tuning = [38, 45, 50, 55, 59, 64]
optimizer = FretboardOptimizer(tuning=drop_d_tuning)
```

### Adjust Cost Parameters

```python
optimizer = FretboardOptimizer()

# Prefer staying on same string
optimizer.STRING_JUMP_WEIGHT = 2.0

# Stricter stretch limit
optimizer.STRETCH_THRESHOLD = 3

# Stronger open string preference
optimizer.OPEN_STRING_BONUS = -1.0
```

### Get All Possible Positions

```python
# Find all ways to play G4 (MIDI 67)
positions = optimizer.get_possible_positions(67)
for pos in positions:
    print(f"String {pos.string + 1}, Fret {pos.fret}")
```

### Audio Extraction API

```python
from src.extractor import AudioExtractor

# Extract MIDI from audio
extractor = AudioExtractor(method='basic_pitch', min_note_duration=0.1)
midi_notes = extractor.extract('vocals.mp3')

# Save as MIDI file
extractor.save_midi(midi_notes, 'output.mid')
```

## Examples

### Run Complete Audio Conversion

```bash
# Process a vocal recording
python main.py your_melody.mp3

# With all options
python main.py your_melody.mp3 \
    --method basic_pitch \
    --tuning drop-d \
    --preprocess \
    --save-midi \
    --detailed \
    --output my_custom_tab.txt
```

### Python API Examples

See [QUICK_REFERENCE.py](QUICK_REFERENCE.py) for common usage patterns and examples.

### Jupyter Notebooks

Explore interactive tutorials:

```bash
jupyter notebook notebooks/fretboard_optimizer_tutorial.ipynb
```

## Testing

Run the comprehensive test suite:

```bash
# All tests
python -m unittest tests/test_fretboard_optimizer.py -v

# Dijkstra algorithm tests
python -m unittest tests/test_dijkstra.py -v

# Specific test
python -m unittest tests.test_fretboard_optimizer.TestFretboardOptimizer.test_optimize_simple_scale -v
```

## Project Structure

```
axtone/
├── main.py                        # CLI entry point for audio conversion
├── src/
│   ├── __init__.py
│   ├── fretboard_optimizer.py    # Core optimizer (Dijkstra)
│   ├── optimizer.py               # Optimizer wrapper
│   ├── extractor.py               # Audio-to-MIDI extraction
│   ├── formatter.py               # Tablature formatting
│   └── utils.py                   # Audio preprocessing utilities
├── tests/
│   ├── test_fretboard_optimizer.py
│   └── test_dijkstra.py
├── data/
│   ├── raw/                       # Input audio files
│   ├── processed/                 # Preprocessed audio
│   └── output/                    # Generated tablature & MIDI
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── fretboard_optimizer_tutorial.ipynb
│   ├── 01_dsp_test.ipynb
│   └── 02_graph_test.ipynb
├── requirements.txt
├── README.md
├── USAGE.md                       # Detailed usage guide
├── QUICK_REFERENCE.py             # Quick reference cheat sheet
└── ARCHITECTURE.py                # System architecture diagrams
```

## Technical Details

### Guitar Standard Tuning (MIDI Numbers)

| String | Note | MIDI Number |
|--------|------|-------------|
| 1 (High E) | E4 | 64 |
| 2 | B3 | 59 |
| 3 | G3 | 55 |
| 4 | D3 | 50 |
| 5 | A2 | 45 |
| 6 (Low E) | E2 | 40 |

### Complexity

- **Time Complexity**: O((V + E) log V) where V = number of positions, E = number of transitions
- **Space Complexity**: O(V + E) for graph storage

For a typical melody of N notes with ~5 positions per note:
- V ≈ 5N nodes
- E ≈ 25N edges
- Runtime: O(N log N)

## Future Enhancements

- [ ] ~~Integration with audio-to-MIDI conversion~~ ✅ **COMPLETED** (Basic Pitch & PYIN)
- [ ] Real-time audio input processing
- [ ] Machine learning for personalized fingering preferences
- [ ] Export to GP, TuxGuitar, or MusicXML formats
- [ ] Chord support (polyphonic optimization)
- [ ] Fretting hand position visualization
- [ ] Web interface for easier access
- [ ] Mobile app for on-the-go conversion

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dijkstra's algorithm for shortest path finding
- Music Information Retrieval (MIR) community
- Guitar pedagogical research on optimal fingering

