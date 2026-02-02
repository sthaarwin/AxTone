# FretboardOptimizer Usage Guide

## Quick Start

### 1. Basic Usage

```python
from src.fretboard_optimizer import FretboardOptimizer, MidiNote

# Create your melody
melody = [
    MidiNote(midi_number=60, onset=0.0, offset=0.5),  # C4
    MidiNote(midi_number=62, onset=0.5, offset=1.0),  # D4
    MidiNote(midi_number=64, onset=1.0, offset=1.5),  # E4
]

# Optimize
optimizer = FretboardOptimizer()
path, tablature = optimizer.optimize(melody)

# Display
print(tablature)
```

### 2. Understanding the Output

```
e|--- 0--- 2---|
B|--- 1--- 3---|
G|--- 0--- 2---|
D|--- 2--- 0---|
A|--- 3--- 0---|
E|--- 0--- 2---|
```

- Each column represents one note
- Numbers = fret positions (0 = open string)
- Dashes = string not played

## API Reference

### Classes

#### `MidiNote`
Represents a single MIDI note with timing.

**Parameters:**
- `midi_number` (int): MIDI note number (0-127)
- `onset` (float): Start time in seconds
- `offset` (float): End time in seconds

**Example:**
```python
note = MidiNote(60, 0.0, 0.5)  # Middle C, half second
```

#### `FretPosition`
Represents a specific position on the guitar.

**Attributes:**
- `string` (int): String number (0=low E, 5=high E)
- `fret` (int): Fret number (0=open)
- `midi_note` (int): MIDI note this position produces

#### `FretboardOptimizer`
Main optimizer class using Dijkstra's algorithm.

**Constructor:**
```python
FretboardOptimizer(
    tuning=None,      # List of MIDI numbers for each string
    max_fret=22       # Maximum fret to consider
)
```

**Methods:**

##### `get_possible_positions(midi_note: int) -> List[FretPosition]`
Find all ways to play a specific note.

```python
positions = optimizer.get_possible_positions(67)  # G4
for pos in positions:
    print(f"String {pos.string}, Fret {pos.fret}")
```

##### `build_graph(midi_sequence: List[MidiNote]) -> None`
Construct the graph of all possible fingering paths.

```python
optimizer.build_graph(melody)
print(f"Created {len(optimizer.graph)} nodes")
```

##### `find_best_path() -> Optional[List[FretPosition]]`
Find optimal path using Dijkstra's algorithm.

```python
path = optimizer.find_best_path()
if path:
    print(f"Found path with {len(path)} positions")
```

##### `format_tablature(path: List[FretPosition], width: int = 80) -> str`
Convert path to ASCII tablature.

```python
tab = optimizer.format_tablature(path, width=100)
print(tab)
```

##### `optimize(midi_sequence: List[MidiNote]) -> Tuple[List[FretPosition], str]`
Complete pipeline: build graph → find path → format tab.

```python
path, tablature = optimizer.optimize(melody)
```

### Cost Function Parameters

You can customize these attributes after creating the optimizer:

```python
optimizer = FretboardOptimizer()

# Fret distance weight (default: 1.0)
optimizer.FRET_DISTANCE_WEIGHT = 1.5

# String jump penalty (default: 0.5)
optimizer.STRING_JUMP_WEIGHT = 2.0

# Stretch penalty (default: 100.0)
optimizer.STRETCH_PENALTY = 200.0

# Stretch threshold in frets (default: 4)
optimizer.STRETCH_THRESHOLD = 3

# Open string bonus (default: -0.3)
optimizer.OPEN_STRING_BONUS = -1.0
```

## Advanced Usage

### Custom Tunings

```python
# Drop-D tuning
drop_d = [38, 45, 50, 55, 59, 64]  # D2-A2-D3-G3-B3-E4
optimizer = FretboardOptimizer(tuning=drop_d)

# Open G tuning
open_g = [38, 43, 50, 55, 59, 62]  # D2-G2-D3-G3-B3-D4
optimizer = FretboardOptimizer(tuning=open_g)

# 7-string guitar
seven_string = [35, 40, 45, 50, 55, 59, 64]  # B1-E2-A2-D3-G3-B3-E4
optimizer = FretboardOptimizer(tuning=seven_string)
```

### Analyzing Results

```python
path, tab = optimizer.optimize(melody)

# Calculate statistics
if len(path) > 1:
    # Average fret movement
    fret_moves = [abs(path[i].fret - path[i+1].fret) 
                  for i in range(len(path)-1)]
    avg_fret_move = sum(fret_moves) / len(fret_moves)
    
    # Average string jumps
    string_jumps = [abs(path[i].string - path[i+1].string) 
                    for i in range(len(path)-1)]
    avg_string_jump = sum(string_jumps) / len(string_jumps)
    
    print(f"Average fret movement: {avg_fret_move:.2f}")
    print(f"Average string jumps: {avg_string_jump:.2f}")
```

### Detailed Path Analysis

```python
from src.fretboard_optimizer import midi_number_to_note_name

path, _ = optimizer.optimize(melody)

for i, (note, pos) in enumerate(zip(melody, path)):
    note_name = midi_number_to_note_name(note.midi_number)
    print(f"{i+1}. {note_name} → String {pos.string + 1}, Fret {pos.fret}")
    
    if i > 0:
        cost = optimizer.calculate_transition_cost(path[i-1], pos)
        print(f"   Transition cost: {cost:.2f}")
```

## Integration Examples

### From MIDI File

```python
import mido

# Load MIDI file
mid = mido.MidiFile('melody.mid')

# Extract notes
notes = []
time = 0.0

for msg in mid.tracks[0]:
    time += msg.time
    if msg.type == 'note_on' and msg.velocity > 0:
        onset = time
        # Find corresponding note_off or next note_on
        # ... (simplified)
        offset = onset + 0.5
        notes.append(MidiNote(msg.note, onset, offset))

# Optimize
optimizer = FretboardOptimizer()
path, tablature = optimizer.optimize(notes)
```

### From Pretty MIDI

```python
import pretty_midi

# Load audio file -> MIDI (using Basic Pitch or similar)
midi_data = pretty_midi.PrettyMIDI('melody.mid')

# Extract melody notes
notes = []
for instrument in midi_data.instruments:
    for note in instrument.notes:
        notes.append(MidiNote(
            midi_number=note.pitch,
            onset=note.start,
            offset=note.end
        ))

# Sort by onset time
notes.sort(key=lambda n: n.onset)

# Optimize
optimizer = FretboardOptimizer()
path, tablature = optimizer.optimize(notes)
```

### Save Results

```python
# Save tablature to file
with open('output/tablature.txt', 'w') as f:
    f.write(tablature)

# Save path details as JSON
import json

path_data = [
    {
        'note': midi_number_to_note_name(note.midi_number),
        'midi': note.midi_number,
        'string': pos.string,
        'fret': pos.fret,
        'onset': note.onset,
        'offset': note.offset
    }
    for note, pos in zip(melody, path)
]

with open('output/fingering.json', 'w') as f:
    json.dump(path_data, indent=2, fp=f)
```

## Optimization Tips

### For Beginners
```python
optimizer = FretboardOptimizer()
# Prefer open strings more strongly
optimizer.OPEN_STRING_BONUS = -1.0
# Reduce stretch threshold
optimizer.STRETCH_THRESHOLD = 3
# Increase string jump penalty (stay on same string more)
optimizer.STRING_JUMP_WEIGHT = 1.5
```

### For Advanced Players
```python
optimizer = FretboardOptimizer()
# Allow larger stretches
optimizer.STRETCH_THRESHOLD = 5
# Reduce stretch penalty
optimizer.STRETCH_PENALTY = 50.0
# Allow more string jumping
optimizer.STRING_JUMP_WEIGHT = 0.3
```

### For Specific Genres

**Classical/Fingerstyle:**
```python
optimizer.STRING_JUMP_WEIGHT = 0.8  # Moderate string changes
optimizer.OPEN_STRING_BONUS = -0.5   # Favor open strings
```

**Rock/Shred:**
```python
optimizer.STRING_JUMP_WEIGHT = 0.2  # Allow fast string changes
optimizer.STRETCH_THRESHOLD = 6      # Allow wider stretches
```

**Jazz:**
```python
optimizer.STRING_JUMP_WEIGHT = 0.4
optimizer.FRET_DISTANCE_WEIGHT = 0.8  # Prefer position playing
```

## Troubleshooting

### No Path Found

```python
path, tab = optimizer.optimize(melody)

if path is None:
    # Check if notes are in guitar range
    for note in melody:
        positions = optimizer.get_possible_positions(note.midi_number)
        if not positions:
            print(f"Note {note.midi_number} cannot be played!")
    
    # Try increasing max_fret
    optimizer = FretboardOptimizer(max_fret=24)
```

### Unrealistic Fingering

```python
# Analyze the path
for i in range(len(path) - 1):
    fret_dist = abs(path[i].fret - path[i+1].fret)
    if fret_dist > 5:
        print(f"Warning: Large stretch at position {i}")
    
    # Adjust cost parameters
    optimizer.STRETCH_PENALTY = 200.0
```

### Performance Issues

```python
import time

# For long melodies, track progress
melody_chunks = [melody[i:i+50] for i in range(0, len(melody), 50)]

results = []
for chunk in melody_chunks:
    start = time.time()
    path, tab = optimizer.optimize(chunk)
    print(f"Chunk processed in {time.time() - start:.2f}s")
    results.append(path)
```

## Best Practices

1. **Always validate MIDI notes are in playable range (40-84)**
2. **Sort notes by onset time before optimizing**
3. **Use custom tunings for specific musical contexts**
4. **Adjust cost parameters based on player skill level**
5. **Test with simple melodies first**
6. **Visualize results to ensure they make sense**

## Common MIDI Note Numbers

| Note | MIDI | Typical Guitar Position |
|------|------|------------------------|
| E2   | 40   | Low E string, open |
| A2   | 45   | A string, open |
| D3   | 50   | D string, open |
| G3   | 55   | G string, open |
| B3   | 59   | B string, open |
| E4   | 64   | High E string, open |
| C4   | 60   | A string, fret 3 |
| G4   | 67   | D string, fret 5 |
| C5   | 72   | B string, fret 1 |

## Helper Functions

```python
# MIDI number to note name
from src.fretboard_optimizer import midi_number_to_note_name

print(midi_number_to_note_name(60))  # "C4"
print(midi_number_to_note_name(69))  # "A4"

# Note name to MIDI (not built-in, but easy to add)
def note_name_to_midi(note_name):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 
             'F#', 'G', 'G#', 'A', 'A#', 'B']
    note = note_name[:-1]
    octave = int(note_name[-1])
    return notes.index(note) + (octave + 1) * 12

print(note_name_to_midi("C4"))  # 60
```

## Additional Resources

- Run `python demo.py` for a quick demonstration
- Run `python examples.py` for comprehensive examples
- Check `notebooks/fretboard_optimizer_tutorial.ipynb` for interactive tutorial
- Run tests: `python -m unittest tests.test_fretboard_optimizer`

## Support

For issues or questions:
1. Check the examples in `examples.py`
2. Review the test cases in `tests/test_fretboard_optimizer.py`
3. Consult the detailed README.md
4. Open an issue on GitHub
