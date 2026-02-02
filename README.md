# AxTone 🎸

An AI-powered system for generating optimized guitar tablature from MIDI melodies using advanced graph algorithms.

## Overview

AxTone converts vocal melodies (represented as MIDI notes) into playable guitar tablature by finding the optimal fingering path using **Dijkstra's Algorithm**. The system treats the guitar fretboard as a graph where each possible finger position is a node, and transitions between positions have costs based on playability.

## Features

- 🎯 **Graph-Based Optimization**: Uses Dijkstra's algorithm to find the easiest fingering path
- 🎵 **Intelligent Cost Function**: Considers fret distance, string jumps, stretch penalties, and open string preferences
- 🎸 **Multiple Tuning Support**: Works with standard tuning and custom tunings (Drop-D, etc.)
- 📊 **ASCII Tablature Output**: Generates readable guitar tab format
- 🔧 **Customizable Parameters**: Adjust cost weights to match your playing style
- 📈 **Performance Statistics**: Shows average fret movement and string jumps

## Installation

```bash
# Clone the repository
git clone https://github.com/sthaarwin/axtone.git
cd axtone

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

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
path, tablature = optimizer.optimize(melody)

print(tablature)
```

**Output:**
```
================================================================================
GUITAR TABLATURE (Optimized via Dijkstra's Algorithm)
================================================================================

e|--- 0--- 2--- 0--- 1--- 2--- 0--- 1---|
B|--- 1--- 3--- 0--- 1--- 3--- 0--- 1---|
G|--- 0--- 2--- 0--- 2--- 0--- 2--- 0---|
D|--- 2--- 0--- 2--- 3--- 0--- 2--- 3---|
A|--- 3--- 0--- 2--- 3--- 5--- 2--- 3---|
E|--- 0--- 2--- 0--- 1--- 3--- 0--- 1---|

================================================================================
Total notes: 8
Average fret movement: 1.43
Average string jumps: 0.71
================================================================================
```

## How It Works

### 1. Graph Construction

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

## Advanced Usage

### Custom Tuning

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

## Examples

Run the comprehensive example suite:

```bash
python examples.py
```

This includes:
- Simple scales
- Melodies with octave jumps
- Chromatic passages
- Custom tunings
- Cost function comparisons
- Popular melodies ("Twinkle Twinkle Little Star")

## Testing

```bash
python -m unittest tests/test_fretboard_optimizer.py -v
```

## Project Structure

```
axtone/
├── src/
│   ├── __init__.py
│   └── fretboard_optimizer.py    # Main optimizer class
├── tests/
│   └── test_fretboard_optimizer.py
├── data/
│   ├── raw/                       # Input MIDI files
│   ├── processed/                 # Processed data
│   └── output/                    # Generated tablature
├── notebooks/                     # Jupyter notebooks for analysis
├── examples.py                    # Example demonstrations
├── requirements.txt
└── README.md
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

- [ ] Integration with audio-to-MIDI conversion (using Basic Pitch)
- [ ] Machine learning for personalized fingering preferences
- [ ] Export to GP, TuxGuitar, or MusicXML formats
- [ ] Chord support (polyphonic optimization)
- [ ] Fretting hand position visualization
- [ ] Real-time MIDI input processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dijkstra's algorithm for shortest path finding
- Music Information Retrieval (MIR) community
- Guitar pedagogical research on optimal fingering

## Citation

If you use this code in your research, please cite:

```bibtex
@software{axtone2026,
  title={AxTone: Graph-Based Guitar Tablature Optimization},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/axtone}
}
```

---

Made with ❤️ for guitarists and music technologists
