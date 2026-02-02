"""
System Architecture and Algorithm Flow Diagram
"""

ARCHITECTURE_DIAGRAM = r"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    FRETBOARD OPTIMIZER - SYSTEM ARCHITECTURE                ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                               INPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐              │
│  │  MIDI Files   │    │  Audio Files  │    │  Manual Input │              │
│  │   (.mid)      │    │   (.wav/mp3)  │    │  (MidiNote)   │              │
│  └───────┬───────┘    └───────┬───────┘    └───────┬───────┘              │
│          │                    │                    │                        │
│          │              ┌─────▼──────┐             │                        │
│          │              │ Audio→MIDI │             │                        │
│          │              │ (Optional) │             │                        │
│          │              └─────┬──────┘             │                        │
│          │                    │                    │                        │
│          └────────────────────┴────────────────────┘                        │
│                                │                                            │
│                        ┌───────▼────────┐                                   │
│                        │ List[MidiNote] │                                   │
│                        │  - midi_number │                                   │
│                        │  - onset       │                                   │
│                        │  - offset      │                                   │
│                        └───────┬────────┘                                   │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                            PROCESSING LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ FretboardOptimizer                                                 │    │
│  │                                                                    │    │
│  │  Step 1: POSITION GENERATION                                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │ For each MIDI note:                                      │     │    │
│  │  │   get_possible_positions(midi_note)                      │     │    │
│  │  │   → Generates FretPosition(string, fret, midi_note)      │     │    │
│  │  │                                                           │     │    │
│  │  │   Example: G4 (MIDI 67)                                  │     │    │
│  │  │     String 6, Fret 3  ─┐                                 │     │    │
│  │  │     String 5, Fret 8   ├─ 6 possible positions          │     │    │
│  │  │     String 4, Fret 12  │                                 │     │    │
│  │  │     String 3, Fret 0   │                                 │     │    │
│  │  │     String 2, Fret 5   │                                 │     │    │
│  │  │     String 1, Fret 10 ─┘                                 │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │                                │                                  │    │
│  │  Step 2: GRAPH CONSTRUCTION                                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │ build_graph(midi_sequence)                               │     │    │
│  │  │                                                           │     │    │
│  │  │ Creates adjacency list:                                  │     │    │
│  │  │                                                           │     │    │
│  │  │   Note[i] Position A ──┬─→ Note[i+1] Position X (cost)   │     │    │
│  │  │                        ├─→ Note[i+1] Position Y (cost)   │     │    │
│  │  │                        └─→ Note[i+1] Position Z (cost)   │     │    │
│  │  │                                                           │     │    │
│  │  │   Note[i] Position B ──┬─→ Note[i+1] Position X (cost)   │     │    │
│  │  │                        └─→ Note[i+1] Position Y (cost)   │     │    │
│  │  │                                                           │     │    │
│  │  │ Graph structure: {node: [(neighbor, cost), ...]}         │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │                                │                                  │    │
│  │  Step 3: COST CALCULATION                                        │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │ calculate_transition_cost(pos1, pos2)                    │     │    │
│  │  │                                                           │     │    │
│  │  │ Cost = Σ of:                                             │     │    │
│  │  │   • Fret Distance:    |fret₁ - fret₂| × 1.0             │     │    │
│  │  │   • String Jump:      |string₁ - string₂| × 0.5         │     │    │
│  │  │   • Stretch Penalty:  100.0 if |Δfret| > 4              │     │    │
│  │  │   • Open String Bonus: -0.3 if fret₂ = 0                │     │    │
│  │  │                                                           │     │    │
│  │  │ Example:                                                 │     │    │
│  │  │   pos1: String 3, Fret 5                                 │     │    │
│  │  │   pos2: String 2, Fret 7                                 │     │    │
│  │  │   cost = |5-7|×1.0 + |3-2|×0.5 = 2.0 + 0.5 = 2.5        │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │                                │                                  │    │
│  │  Step 4: DIJKSTRA'S ALGORITHM                                    │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │ find_best_path()                                         │     │    │
│  │  │                                                           │     │    │
│  │  │ 1. Initialize:                                           │     │    │
│  │  │    - distances[START] = 0                                │     │    │
│  │  │    - distances[others] = ∞                               │     │    │
│  │  │    - priority_queue = [(0, START)]                       │     │    │
│  │  │                                                           │     │    │
│  │  │ 2. While queue not empty:                                │     │    │
│  │  │    - Pop node with minimum distance                      │     │    │
│  │  │    - For each neighbor:                                  │     │    │
│  │  │        new_dist = current_dist + edge_cost               │     │    │
│  │  │        if new_dist < distances[neighbor]:                │     │    │
│  │  │            distances[neighbor] = new_dist                │     │    │
│  │  │            previous[neighbor] = current                  │     │    │
│  │  │            push (new_dist, neighbor) to queue            │     │    │
│  │  │                                                           │     │    │
│  │  │ 3. Reconstruct path from END to START                    │     │    │
│  │  │                                                           │     │    │
│  │  │ Complexity: O((V + E) log V)                             │     │    │
│  │  │   V = #positions (~5N for N notes)                       │     │    │
│  │  │   E = #transitions (~25N)                                │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │                                │                                  │    │
│  │  Step 5: TABLATURE FORMATTING                                    │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │ format_tablature(path)                                   │     │    │
│  │  │                                                           │     │    │
│  │  │ Converts List[FretPosition] to ASCII tab format          │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                 │                                           │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                              OUTPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────┐    ┌────────────────────────────┐          │
│  │   List[FretPosition]       │    │   ASCII Tablature          │          │
│  │                            │    │                            │          │
│  │  [FretPosition(            │    │  e|--- 0--- 2--- 0---|    │          │
│  │    string=2,               │    │  B|--- 1--- 3--- 1---|    │          │
│  │    fret=5,                 │    │  G|--- 0--- 2--- 0---|    │          │
│  │    midi_note=55            │    │  D|--- 2--- 0--- 2---|    │          │
│  │  ), ...]                   │    │  A|--- 3--- 0--- 3---|    │          │
│  │                            │    │  E|--- 0--- 2--- 0---|    │          │
│  │                            │    │                            │          │
│  │  Can be used for:          │    │  + Statistics              │          │
│  │  - Further processing      │    │  + Header/Footer           │          │
│  │  - Visualization           │    │                            │          │
│  │  - Export to other formats │    │                            │          │
│  └────────────────────────────┘    └────────────────────────────┘          │
│                                                                             │
│  ┌────────────────────────────┐    ┌────────────────────────────┐          │
│  │   Save to File             │    │   Display to User          │          │
│  │  - tablature.txt           │    │  - Console output          │          │
│  │  - fingering.json          │    │  - Jupyter notebook        │          │
│  │  - musicxml (future)       │    │  - Web interface           │          │
│  └────────────────────────────┘    └────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════╗
║                            EXAMPLE DATA FLOW                                ║
╚════════════════════════════════════════════════════════════════════════════╝

Input: C-E-G (C major chord arpeggio)

  [MidiNote(60, 0.0, 0.5), MidiNote(64, 0.5, 1.0), MidiNote(67, 1.0, 1.5)]
                                      │
  ┌───────────────────────────────────▼───────────────────────────────────┐
  │                       Position Generation                             │
  ├───────────────────────────────────────────────────────────────────────┤
  │  C4 (60): [(6,8), (5,3), (4,10), (3,5), (2,1)]  ← 5 positions         │
  │  E4 (64): [(6,12), (5,7), (4,2), (3,9), (2,5), (1,0)]  ← 6 positions  │
  │  G4 (67): [(6,3), (5,10), (4,5), (3,0), (2,8)]  ← 5 positions         │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼───────────────────────────────────┐
  │                         Graph Construction                            │
  ├───────────────────────────────────────────────────────────────────────┤
  │  Total nodes: 16 (5 + 6 + 5)                                          │
  │  Total edges: 5×6 + 6×5 = 60 edges                                    │
  │                                                                        │
  │  Sample edges:                                                        │
  │    C4(3,5) → E4(4,2): cost = |5-2| + |3-4|×0.5 = 3.5                  │
  │    C4(3,5) → E4(2,5): cost = |5-5| + |3-2|×0.5 = 0.5  ← Low cost!     │
  │    E4(2,5) → G4(3,0): cost = |5-0| + |2-3|×0.5 + (-0.3) = 4.7         │
  │    E4(3,9) → G4(3,0): cost = 9 + 0 + (-0.3) = 8.7     ← High cost     │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼───────────────────────────────────┐
  │                         Dijkstra's Algorithm                          │
  ├───────────────────────────────────────────────────────────────────────┤
  │  START → C4(3,5) → E4(2,5) → G4(3,0) → END                            │
  │                                                                        │
  │  Total cost: 0 + 1.2 + 0.5 + 4.7 + 0 = 6.4  ← Optimal path!           │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼───────────────────────────────────┐
  │                        Tablature Generation                           │
  ├───────────────────────────────────────────────────────────────────────┤
  │  e|----------|                                                        │
  │  B|--- 5-----|                                                        │
  │  G|--- 5--- 0|                                                        │
  │  D|----------|                                                        │
  │  A|----------|                                                        │
  │  E|----------|                                                        │
  │                                                                        │
  │  Notes:                                                               │
  │    1. C4 → String 3 (G), Fret 5                                       │
  │    2. E4 → String 2 (B), Fret 5                                       │
  │    3. G4 → String 3 (G), Fret 0 (open)                                │
  └───────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════╗
║                           PERFORMANCE METRICS                               ║
╚════════════════════════════════════════════════════════════════════════════╝

For a typical melody with N notes:

  Nodes (V):        ~5N   (average 5 positions per note)
  Edges (E):        ~25N  (5×5 transitions between consecutive notes)
  
  Time Complexity:  O((V + E) log V) = O(N log N)
  Space Complexity: O(V + E) = O(N)
  
  Typical performance:
    10 notes:   < 10ms
    50 notes:   < 50ms
    100 notes:  < 150ms
    500 notes:  < 1s

╔════════════════════════════════════════════════════════════════════════════╗
║                          KEY DESIGN DECISIONS                               ║
╚════════════════════════════════════════════════════════════════════════════╝

1. Graph Representation
   ✓ Adjacency list (not matrix) - sparse graph, memory efficient
   ✓ Nodes include note index - prevents mixing positions across notes
   ✓ Virtual START/END nodes - simplifies path finding

2. Cost Function
   ✓ Linear combination of factors - simple, interpretable
   ✓ Tunable weights - customizable for different players
   ✓ Negative costs allowed - can favor certain positions
   ✓ Stretch penalty - ensures playability

3. Algorithm Choice
   ✓ Dijkstra over A* - no clear heuristic for guitar fingering
   ✓ Priority queue (heapq) - efficient implementation
   ✓ Path reconstruction - stores previous nodes

4. Extensibility
   ✓ Pluggable cost function - easy to customize
   ✓ Multiple tunings - supports different guitar setups
   ✓ Dataclasses - clean, type-safe data structures
"""

if __name__ == "__main__":
    print(ARCHITECTURE_DIAGRAM)
