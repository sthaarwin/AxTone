"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FRETBOARD OPTIMIZER - QUICK REFERENCE                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT IT DOES:
  Converts MIDI melodies → Optimized guitar tablature using Dijkstra's Algorithm

BASIC USAGE:
  from src.fretboard_optimizer import FretboardOptimizer, MidiNote
  
  melody = [MidiNote(60, 0.0, 0.5), MidiNote(62, 0.5, 1.0)]
  optimizer = FretboardOptimizer()
  path, tablature = optimizer.optimize(melody)
  print(tablature)

KEY CLASSES:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ MidiNote(midi_number, onset, offset)                                    │
  │   - midi_number: 0-127 (60 = middle C)                                  │
  │   - onset: start time in seconds                                        │
  │   - offset: end time in seconds                                         │
  └─────────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ FretPosition(string, fret, midi_note)                                   │
  │   - string: 0-5 (0=low E, 5=high E)                                     │
  │   - fret: 0-22 (0=open string)                                          │
  │   - midi_note: MIDI number this position produces                       │
  └─────────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ FretboardOptimizer(tuning=None, max_fret=22)                            │
  │   Methods:                                                              │
  │     • get_possible_positions(midi_note) → List[FretPosition]            │
  │     • build_graph(midi_sequence)        → None                          │
  │     • find_best_path()                  → Optional[List[FretPosition]]  │
  │     • format_tablature(path, width=80)  → str                           │
  │     • optimize(midi_sequence)           → (path, tablature)             │
  └─────────────────────────────────────────────────────────────────────────┘

COST FUNCTION PARAMETERS (Customizable):
  ┌──────────────────────────┬──────────┬─────────────────────────────────┐
  │ Parameter                │ Default  │ Description                     │
  ├──────────────────────────┼──────────┼─────────────────────────────────┤
  │ FRET_DISTANCE_WEIGHT     │ 1.0      │ Cost per fret moved             │
  │ STRING_JUMP_WEIGHT       │ 0.5      │ Cost per string jumped          │
  │ STRETCH_PENALTY          │ 100.0    │ Cost for unplayable stretches   │
  │ STRETCH_THRESHOLD        │ 4        │ Max frets before penalty        │
  │ OPEN_STRING_BONUS        │ -0.3     │ Bonus for open strings          │
  └──────────────────────────┴──────────┴─────────────────────────────────┘
  
  Example customization:
    optimizer.STRING_JUMP_WEIGHT = 2.0  # Prefer staying on same string
    optimizer.STRETCH_THRESHOLD = 3     # Stricter stretch limit

STANDARD TUNING (MIDI):
  ┌────────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ String │   1  │   2  │   3  │   4  │   5  │   6  │
  │ Note   │  e   │  B   │  G   │  D   │  A   │  E   │
  │ MIDI   │  64  │  59  │  55  │  50  │  45  │  40  │
  └────────┴──────┴──────┴──────┴──────┴──────┴──────┘
  
  Drop-D: [38, 45, 50, 55, 59, 64]
  Open-G: [38, 43, 50, 55, 59, 62]

COMMON TASKS:

  1. Simple optimization:
     ┌───────────────────────────────────────────────────────────────────┐
     │ melody = [MidiNote(60, 0, 0.5), MidiNote(62, 0.5, 1.0)]           │
     │ optimizer = FretboardOptimizer()                                  │
     │ path, tab = optimizer.optimize(melody)                            │
     │ print(tab)                                                        │
     └───────────────────────────────────────────────────────────────────┘
  
  2. Check all positions for a note:
     ┌───────────────────────────────────────────────────────────────────┐
     │ positions = optimizer.get_possible_positions(67)  # G4            │
     │ for pos in positions:                                             │
     │     print(f"String {pos.string}, Fret {pos.fret}")                │
     └───────────────────────────────────────────────────────────────────┘
  
  3. Custom tuning:
     ┌───────────────────────────────────────────────────────────────────┐
     │ drop_d = [38, 45, 50, 55, 59, 64]                                 │
     │ optimizer = FretboardOptimizer(tuning=drop_d)                     │
     └───────────────────────────────────────────────────────────────────┘
  
  4. Analyze transition cost:
     ┌───────────────────────────────────────────────────────────────────┐
     │ pos1 = FretPosition(2, 5, 55)                                     │
     │ pos2 = FretPosition(2, 8, 58)                                     │
     │ cost = optimizer.calculate_transition_cost(pos1, pos2)            │
     │ print(f"Transition cost: {cost:.2f}")                             │
     └───────────────────────────────────────────────────────────────────┘
  
  5. Convert MIDI number to note name:
     ┌───────────────────────────────────────────────────────────────────┐
     │ from src.fretboard_optimizer import midi_number_to_note_name      │
     │ print(midi_number_to_note_name(60))  # "C4"                       │
     └───────────────────────────────────────────────────────────────────┘

RUNNING EXAMPLES:
  
  Demo (simple):          python demo.py
  All examples:           python examples.py
  Main module:            python src/fretboard_optimizer.py
  Unit tests:             python -m unittest tests.test_fretboard_optimizer
  Interactive tutorial:   jupyter notebook notebooks/fretboard_optimizer_tutorial.ipynb

ALGORITHM OVERVIEW:
  
  1. Graph Construction
     ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
     │   Note 1     │       │   Note 2     │       │   Note 3     │
     │ ┌──────────┐ │       │ ┌──────────┐ │       │ ┌──────────┐ │
     │ │String 3  │─┼───────┼→│String 2  │─┼───────┼→│String 4  │ │
     │ │Fret 5    │ │       │ │Fret 7    │ │       │ │Fret 5    │ │
     │ └──────────┘ │       │ └──────────┘ │       │ └──────────┘ │
     │ ┌──────────┐ │       │ ┌──────────┐ │       │ ┌──────────┐ │
     │ │String 4  │─┼───────┼→│String 3  │─┼───────┼→│String 2  │ │
     │ │Fret 0    │ │       │ │Fret 2    │ │       │ │Fret 10   │ │
     │ └──────────┘ │       │ └──────────┘ │       │ └──────────┘ │
     └──────────────┘       └──────────────┘       └──────────────┘
  
  2. Cost Calculation
     Cost = (|Δfret| × 1.0) + (|Δstring| × 0.5) + penalties + bonuses
  
  3. Dijkstra's Algorithm
     Find shortest path from START → through all notes → END

MIDI NOTE REFERENCE:
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ C3   │ E3   │ G3   │ C4   │ E4   │ G4   │ C5   │ E5   │ G5   │
  │ 48   │ 52   │ 55   │ 60   │ 64   │ 67   │ 72   │ 76   │ 79   │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

TYPICAL GUITAR RANGE:
  Low E (E2) = MIDI 40  ──────────────────────  High E (E5) = MIDI 76

PROJECT STRUCTURE:
  axtone/
  ├── src/
  │   ├── __init__.py
  │   └── fretboard_optimizer.py    ← Main implementation
  ├── tests/
  │   └── test_fretboard_optimizer.py
  ├── notebooks/
  │   └── fretboard_optimizer_tutorial.ipynb
  ├── demo.py                        ← Quick demo
  ├── examples.py                    ← Comprehensive examples
  ├── README.md                      ← Full documentation
  └── USAGE.md                       ← Detailed usage guide

TROUBLESHOOTING:
  
  • "No path found"
    → Check if notes are in playable range (40-84)
    → Try increasing max_fret: FretboardOptimizer(max_fret=24)
  
  • "Unrealistic fingering"
    → Increase STRETCH_PENALTY: optimizer.STRETCH_PENALTY = 200.0
    → Decrease STRETCH_THRESHOLD: optimizer.STRETCH_THRESHOLD = 3
  
  • "Too many string jumps"
    → Increase STRING_JUMP_WEIGHT: optimizer.STRING_JUMP_WEIGHT = 2.0

TIPS:
  ✓ Sort melody by onset time before optimizing
  ✓ Validate MIDI notes are in playable range
  ✓ Adjust cost parameters for player skill level
  ✓ Test with simple melodies first
  ✓ Use custom tunings for specific genres

╔══════════════════════════════════════════════════════════════════════════════╗
║  For more information: README.md, USAGE.md, or run examples.py              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# This file can be printed or displayed as a reference
if __name__ == "__main__":
    print(__doc__)
