"""
MIDI to optimal guitar fingering using Dijkstra's Algorithm.

This module implements a hand-position-aware graph optimization to find the most
playable fingering path through a sequence of MIDI notes on a guitar fretboard.

Key improvement over naive fret-distance approach:
  - Each state is (note_index, FretPosition, hand_position) where hand_position
    is the fret covered by the index finger.
  - Transitions *within* a hand position cost only the finger movement (cheap).
  - Transitions that *shift* the hand position cost proportionally to the shift
    distance, scaled up when the time between notes is short (fast tempo).
  - This models real guitar technique: commit to a position, play notes there,
    then deliberately shift — rather than penalising every individual fret step
    the same way.

The public graph dict uses (note_idx, FretPosition) nodes (backward compatible
with existing tests).  The internal _graph uses (note_idx, FretPosition,
hand_pos) nodes for the actual Dijkstra run.
"""

import heapq
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from .extractor import MidiNote


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FretPosition:
    """Represents a specific fret position on the guitar."""
    string: int      # String number (0 = low E, 5 = high E)
    fret: int        # Fret number  (0 = open string)
    midi_note: int   # MIDI note number this position produces

    def __hash__(self):
        return hash((self.string, self.fret, self.midi_note))

    def __eq__(self, other):
        return (self.string, self.fret, self.midi_note) == (other.string, other.fret, other.midi_note)

    def __lt__(self, other):
        """Needed so heapq can break ties without comparing full node tuples."""
        return (self.string, self.fret, self.midi_note) < (other.string, other.fret, other.midi_note)

    def __repr__(self):
        return f"FretPosition(string={self.string}, fret={self.fret}, midi={self.midi_note})"


# ── Optimizer ─────────────────────────────────────────────────────────────────

class FretboardOptimizer:
    """
    Optimises guitar fingering using Dijkstra's algorithm on a hand-position-
    expanded state space.

    State space
    -----------
    Each node in the *internal* graph is a triple:
        (note_index, FretPosition, hand_position)

    where  hand_position  is the fret covered by the index finger.  A player can
    comfortably reach up to  MAX_STRETCH  additional frets from that anchor.

    Cost model
    ----------
    Within the same hand position:
        cost = |fret₁ − fret₂| × FRET_DISTANCE_WEIGHT
               + |string₁ − string₂| × STRING_JUMP_WEIGHT

    Hand-position shift (index finger moves):
        cost = |hp₂ − hp₁| × POSITION_SHIFT_WEIGHT
               × (1 + TIME_PRESSURE_WEIGHT / max(time_gap, 0.1))
               + |string₁ − string₂| × STRING_JUMP_WEIGHT

    Open-string bonus (applied to destination):
        cost += OPEN_STRING_BONUS   (negative → reduces cost)
    """

    # Standard guitar tuning (MIDI): E2 A2 D3 G3 B3 E4
    STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

    MAX_FRET    = 22
    MAX_STRETCH = 4   # frets reachable from hand position (index → pinky)

    # ── Cost weights (all publicly settable) ──────────────────────────────────
    # Within-position finger movement
    FRET_DISTANCE_WEIGHT  = 1.0
    STRING_JUMP_WEIGHT    = 0.5
    # Hand-position shift
    POSITION_SHIFT_WEIGHT = 2.5   # base cost per fret of shift
    TIME_PRESSURE_WEIGHT  = 0.3   # how much a tight time gap inflates shift cost
    # Misc
    OPEN_STRING_BONUS     = -0.3
    # Legacy (kept so existing tests that set these still work)
    STRETCH_PENALTY       = 100.0
    STRETCH_THRESHOLD     = 4

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self, tuning: Optional[List[int]] = None, max_fret: int = 22):
        """
        Initialise the FretboardOptimizer.

        Args:
            tuning:   List of MIDI note numbers for each open string (low→high).
                      Defaults to standard tuning.
            max_fret: Maximum fret number to consider (default 22).
        """
        self.tuning   = tuning if tuning is not None else self.STANDARD_TUNING
        self.max_fret = max_fret

        # Public (backward-compatible)
        self.graph             = {}  # (note_idx, FretPosition) → [(node, cost)]
        self.midi_sequence     = []
        self.positions_by_note = {}  # note_idx → [FretPosition]

        # Private (hp-expanded state space)
        self._graph            = {}  # (note_idx, FretPosition, hp) → [(node, cost)]
        self._nodes_by_note    = {}  # note_idx → [(FretPosition, hp)]

        # Results populated after find_best_path()
        self.hand_positions: List[int] = []   # hp for each note in the optimal path

    # ── Fret / hand-position helpers ──────────────────────────────────────────

    def get_possible_positions(self, midi_note: int) -> List[FretPosition]:
        """
        Return all (string, fret) positions that produce *midi_note*.

        Args:
            midi_note: MIDI note number to look up.

        Returns:
            List of FretPosition objects (may be empty for out-of-range notes).
        """
        positions = []
        for string_idx, open_string_midi in enumerate(self.tuning):
            fret = midi_note - open_string_midi
            if 0 <= fret <= self.max_fret:
                positions.append(FretPosition(string=string_idx, fret=fret,
                                              midi_note=midi_note))
        return positions

    def get_hand_positions_for_fret(self, fret: int) -> List[int]:
        """
        Return all valid hand positions (index-finger frets) that can
        comfortably reach *fret* within MAX_STRETCH semitones.

        Example: fret=7, MAX_STRETCH=4  →  [3, 4, 5, 6, 7]
        """
        min_hp = max(0, fret - self.MAX_STRETCH)
        return list(range(min_hp, fret + 1))

    # ── Cost functions ────────────────────────────────────────────────────────

    def _hp_cost(self,
                 pos1: FretPosition, hp1: int,
                 pos2: FretPosition, hp2: int,
                 time_gap: float = 0.5) -> float:
        """
        Hand-position-aware transition cost (used internally).

        Args:
            pos1 / hp1: Starting fret position and hand position.
            pos2 / hp2: Destination fret position and hand position.
            time_gap:   Time between the two notes in seconds.

        Returns:
            Non-negative float cost (lower is better).
        """
        hand_shift  = abs(hp2 - hp1)
        string_jump = abs(pos1.string - pos2.string)

        if hand_shift == 0:
            # Same hand position: only finger-to-finger movement
            fret_distance = abs(pos1.fret - pos2.fret)
            cost = fret_distance * self.FRET_DISTANCE_WEIGHT
        else:
            # Hand position shift: one-time cost, harder at fast tempo
            time_factor = 1.0 / max(time_gap, 0.1)
            cost = hand_shift * self.POSITION_SHIFT_WEIGHT * (
                1.0 + time_factor * self.TIME_PRESSURE_WEIGHT
            )

        cost += string_jump * self.STRING_JUMP_WEIGHT

        # Slight preference for open strings (destination)
        if pos2.fret == 0:
            cost += self.OPEN_STRING_BONUS

        return max(0.0, cost)

    def calculate_transition_cost(self,
                                  pos1: FretPosition,
                                  pos2: FretPosition) -> float:
        """
        Public transition-cost API (backward compatible with existing tests).

        Uses the classic fret-distance model with a large stretch penalty.
        The internal optimiser uses :meth:`_hp_cost` instead.

        Args:
            pos1: Starting position.
            pos2: Destination position.

        Returns:
            Non-negative float cost.
        """
        fret_distance = abs(pos1.fret - pos2.fret)
        string_jump   = abs(pos1.string - pos2.string)

        cost = (fret_distance * self.FRET_DISTANCE_WEIGHT +
                string_jump   * self.STRING_JUMP_WEIGHT)

        if fret_distance > self.STRETCH_THRESHOLD:
            cost += self.STRETCH_PENALTY

        if pos2.fret == 0:
            cost += self.OPEN_STRING_BONUS

        return max(0.0, cost)

    # ── Graph construction ────────────────────────────────────────────────────

    def build_graph(self, midi_sequence: List[MidiNote]) -> None:
        """
        Build both the public and internal graph representations.

        Public graph  – nodes: (note_idx, FretPosition)
                        edges: minimum hp-cost over all hand-position pairs
                        (backward compatible with tests that inspect self.graph)

        Internal graph – nodes: (note_idx, FretPosition, hand_pos)
                         edges: exact hp-cost for each hand-position pairing
                         (used by find_best_path for real optimisation)

        Args:
            midi_sequence: List of MidiNote objects in chronological order.
        """
        self.midi_sequence     = midi_sequence
        self.graph             = {}
        self._graph            = {}
        self.positions_by_note = {}
        self._nodes_by_note    = {}

        if not midi_sequence:
            return

        # ── Node generation ───────────────────────────────────────────────────
        for note_idx, midi_note in enumerate(midi_sequence):
            positions = self.get_possible_positions(midi_note.midi_number)
            self.positions_by_note[note_idx] = positions

            hp_nodes: List[Tuple[FretPosition, int]] = []
            for pos in positions:
                self.graph[(note_idx, pos)] = []
                for hp in self.get_hand_positions_for_fret(pos.fret):
                    self._graph[(note_idx, pos, hp)] = []
                    hp_nodes.append((pos, hp))

            self._nodes_by_note[note_idx] = hp_nodes

        # ── Edge generation ───────────────────────────────────────────────────
        for note_idx in range(len(midi_sequence) - 1):
            # Time between this note and the next (seconds)
            time_gap = (midi_sequence[note_idx + 1].onset
                        - midi_sequence[note_idx].onset)
            # Clamp to a small positive value to avoid division by zero
            time_gap = max(time_gap, 0.01)

            curr_positions = self.positions_by_note[note_idx]
            next_positions = self.positions_by_note[note_idx + 1]

            curr_nodes = self._nodes_by_note[note_idx]
            next_nodes = self._nodes_by_note[note_idx + 1]

            # Pre-compute min hp-cost between each (pos1, pos2) pair for the
            # public graph.
            for pos1 in curr_positions:
                hps1 = self.get_hand_positions_for_fret(pos1.fret)

                for pos2 in next_positions:
                    hps2 = self.get_hand_positions_for_fret(pos2.fret)

                    # Minimum cost over all hp combos → public graph edge
                    min_cost = min(
                        self._hp_cost(pos1, h1, pos2, h2, time_gap)
                        for h1 in hps1
                        for h2 in hps2
                    )
                    self.graph[(note_idx, pos1)].append(
                        ((note_idx + 1, pos2), min_cost)
                    )

            # Exact hp-specific costs → internal graph edges
            for (pos1, hp1) in curr_nodes:
                src = (note_idx, pos1, hp1)
                for (pos2, hp2) in next_nodes:
                    dst  = (note_idx + 1, pos2, hp2)
                    cost = self._hp_cost(pos1, hp1, pos2, hp2, time_gap)
                    self._graph[src].append((dst, cost))

    # ── Path finding ──────────────────────────────────────────────────────────

    def find_best_path(self) -> Optional[List[FretPosition]]:
        """
        Find the optimal fingering path using Dijkstra's algorithm on the
        hand-position-expanded internal graph.

        After this call, :attr:`hand_positions` is populated with the index-
        finger fret for each note in the returned path.

        Returns:
            List of FretPosition objects (optimal path), or None if no path
            exists (e.g. empty melody or all notes out of range).
        """
        self.hand_positions = []

        if not self.midi_sequence or not self._graph:
            return None

        # ── Virtual start / end nodes ─────────────────────────────────────────
        START = ('START', None, None)
        END   = ('END',   None, None)

        self._graph[START] = []
        for (pos, hp) in self._nodes_by_note[0]:
            node = (0, pos, hp)
            # Soft preference: lower hand positions, middle strings
            start_cost = hp * 0.05 + abs(pos.string - 2.5) * 0.2
            self._graph[START].append((node, start_cost))

        self._graph[END] = []
        last_idx = len(self.midi_sequence) - 1
        for (pos, hp) in self._nodes_by_note[last_idx]:
            node = (last_idx, pos, hp)
            if node not in self._graph:
                self._graph[node] = []
            self._graph[node].append((END, 0.0))

        # ── Dijkstra ──────────────────────────────────────────────────────────
        distances = {n: float('inf') for n in self._graph}
        distances[START] = 0.0
        previous: dict = {n: None for n in self._graph}

        counter = 0  # tie-breaker: avoids comparing incompatible node types
        pq      = [(0.0, counter, START)]
        visited = set()

        while pq:
            cur_dist, _, cur_node = heapq.heappop(pq)

            if cur_node in visited:
                continue
            visited.add(cur_node)

            if cur_node == END:
                break

            for neighbour, edge_cost in self._graph.get(cur_node, []):
                if neighbour in visited:
                    continue
                new_dist = cur_dist + edge_cost
                if new_dist < distances.get(neighbour, float('inf')):
                    distances[neighbour] = new_dist
                    previous[neighbour]  = cur_node
                    counter += 1
                    heapq.heappush(pq, (new_dist, counter, neighbour))

        # ── Path reconstruction ───────────────────────────────────────────────
        if distances.get(END, float('inf')) == float('inf'):
            return None

        fret_path: List[FretPosition] = []
        hp_path:   List[int]          = []

        current = END
        while current is not None:
            if current not in (END, START):
                _, pos, hp = current
                fret_path.append(pos)
                hp_path.append(hp)
            current = previous.get(current)

        fret_path.reverse()
        hp_path.reverse()

        self.hand_positions = hp_path
        return fret_path

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def format_tablature(self,
                         path: List[FretPosition],
                         width: int = 80) -> str:
        """
        Format *path* as ASCII guitar tablature (delegates to TablatureFormatter).

        Args:
            path:  List of FretPosition objects.
            width: Character width for header/footer separators.

        Returns:
            ASCII tablature string.
        """
        from .formatter import TablatureFormatter
        formatter = TablatureFormatter(width=width)
        hand_positions = (self.hand_positions
                          if len(self.hand_positions) == len(path) else None)
        return formatter.format(path, hand_positions=hand_positions)

    def optimize(self,
                 midi_sequence: List[MidiNote]
                 ) -> Tuple[Optional[List[FretPosition]], str]:
        """
        Full pipeline: build graph → find path → format tablature.

        Args:
            midi_sequence: List of MidiNote objects in chronological order.

        Returns:
            ``(path, tablature)`` where *path* is None on failure.
        """
        print(f"Optimizing fingering for {len(midi_sequence)} notes "
              f"(hand-position-aware Dijkstra)...")
        self.build_graph(midi_sequence)
        path = self.find_best_path()

        if path:
            print(f"Found optimal path with {len(path)} positions")
            tablature = self.format_tablature(path)
        else:
            print("No valid fingering path found")
            tablature = "No valid fingering path found."

        return path, tablature
