"""
FretboardOptimizer: Convert MIDI melody to optimized guitar tablature using Dijkstra's Algorithm.

This module implements a graph-based approach to find the optimal fingering path
through a sequence of MIDI notes on a guitar fretboard.
"""

import heapq
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class MidiNote:
    """Represents a MIDI note with timing information."""
    midi_number: int  # MIDI note number (e.g., 60 = C4)
    onset: float      # Start time in seconds
    offset: float     # End time in seconds
    
    def __repr__(self):
        return f"MidiNote(midi={self.midi_number}, onset={self.onset:.3f}, offset={self.offset:.3f})"


@dataclass
class FretPosition:
    """Represents a specific fret position on the guitar."""
    string: int  # String number (0=low E, 5=high E)
    fret: int    # Fret number (0=open string)
    midi_note: int  # MIDI note number this position produces
    
    def __hash__(self):
        return hash((self.string, self.fret, self.midi_note))
    
    def __eq__(self, other):
        return (self.string, self.fret, self.midi_note) == (other.string, other.fret, other.midi_note)
    
    def __lt__(self, other):
        """Less than comparison for heap operations."""
        return (self.string, self.fret, self.midi_note) < (other.string, other.fret, other.midi_note)
    
    def __repr__(self):
        return f"FretPosition(string={self.string}, fret={self.fret}, midi={self.midi_note})"


class FretboardOptimizer:
    """
    Optimizes guitar fingering using Dijkstra's algorithm to find the easiest path
    through a sequence of MIDI notes.
    """
    
    # Standard guitar tuning (MIDI numbers): E2, A2, D3, G3, B3, E4
    STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # Low E to High E
    
    # Maximum fret to consider (most guitars have 12-24 frets)
    MAX_FRET = 22
    
    # Cost function parameters
    FRET_DISTANCE_WEIGHT = 1.0
    STRING_JUMP_WEIGHT = 0.5
    STRETCH_PENALTY = 100.0  # Applied when fret distance > 4
    STRETCH_THRESHOLD = 4
    OPEN_STRING_BONUS = -0.3  # Negative cost = bonus
    
    def __init__(self, tuning: Optional[List[int]] = None, max_fret: int = 22):
        """
        Initialize the FretboardOptimizer.
        
        Args:
            tuning: List of MIDI note numbers for each string (low to high).
                   Defaults to standard tuning.
            max_fret: Maximum fret number to consider.
        """
        self.tuning = tuning if tuning is not None else self.STANDARD_TUNING
        self.max_fret = max_fret
        self.graph = {}  # Adjacency list: node -> [(neighbor, cost), ...]
        self.midi_sequence = []
        self.positions_by_note = {}  # Maps note index to list of FretPositions
        
    def get_possible_positions(self, midi_note: int) -> List[FretPosition]:
        """
        Find all possible (string, fret) positions for a given MIDI note.
        
        Args:
            midi_note: MIDI note number
            
        Returns:
            List of FretPosition objects
        """
        positions = []
        
        for string_idx, open_string_midi in enumerate(self.tuning):
            # Calculate fret needed on this string
            fret = midi_note - open_string_midi
            
            # Check if this position is valid
            if 0 <= fret <= self.max_fret:
                positions.append(FretPosition(
                    string=string_idx,
                    fret=fret,
                    midi_note=midi_note
                ))
        
        return positions
    
    def calculate_transition_cost(self, pos1: FretPosition, pos2: FretPosition) -> float:
        """
        Calculate the cost of transitioning from one fret position to another.
        
        Cost factors:
        - Fret distance: |fret1 - fret2|
        - String jump: |string1 - string2| × 0.5
        - Stretch penalty: Massive cost if fret distance > 4
        - Open string bonus: Slight preference for open strings
        
        Args:
            pos1: Starting position
            pos2: Ending position
            
        Returns:
            Transition cost (lower is better)
        """
        # Calculate basic distances
        fret_distance = abs(pos1.fret - pos2.fret)
        string_jump = abs(pos1.string - pos2.string)
        
        # Base cost
        cost = (fret_distance * self.FRET_DISTANCE_WEIGHT + 
                string_jump * self.STRING_JUMP_WEIGHT)
        
        # Stretch penalty (unplayable positions)
        if fret_distance > self.STRETCH_THRESHOLD:
            cost += self.STRETCH_PENALTY
        
        # Open string bonus (only for the destination)
        if pos2.fret == 0:
            cost += self.OPEN_STRING_BONUS
        
        return max(0, cost)  # Ensure non-negative
    
    def build_graph(self, midi_sequence: List[MidiNote]) -> None:
        """
        Build the graph representation of all possible fingering paths.
        
        Each node represents a (string, fret) position for a specific note.
        Edges connect positions of consecutive notes with weighted costs.
        
        Args:
            midi_sequence: List of MidiNote objects in chronological order
        """
        self.midi_sequence = midi_sequence
        self.graph = {}
        self.positions_by_note = {}
        
        if not midi_sequence:
            return
        
        # Generate all possible positions for each note
        for note_idx, midi_note in enumerate(midi_sequence):
            positions = self.get_possible_positions(midi_note.midi_number)
            self.positions_by_note[note_idx] = positions
            
            # Initialize graph nodes
            for pos in positions:
                node_id = (note_idx, pos)
                self.graph[node_id] = []
        
        # Create edges between consecutive notes
        for note_idx in range(len(midi_sequence) - 1):
            current_positions = self.positions_by_note[note_idx]
            next_positions = self.positions_by_note[note_idx + 1]
            
            for pos1 in current_positions:
                node1 = (note_idx, pos1)
                
                for pos2 in next_positions:
                    node2 = (note_idx + 1, pos2)
                    cost = self.calculate_transition_cost(pos1, pos2)
                    
                    # Add edge from current position to next position
                    self.graph[node1].append((node2, cost))
    
    def find_best_path(self) -> Optional[List[FretPosition]]:
        """
        Find the optimal fingering path through the note sequence using Dijkstra's algorithm.
        
        Returns:
            List of FretPosition objects representing the optimal path,
            or None if no path exists.
        """
        if not self.midi_sequence or not self.graph:
            return None
        
        # Create virtual start and end nodes
        start_node = ('START', None)
        end_node = ('END', None)
        
        # Add start node connected to all positions of the first note
        self.graph[start_node] = []
        for pos in self.positions_by_note[0]:
            node = (0, pos)
            # Start cost: slight preference for middle strings and lower frets
            start_cost = pos.fret * 0.1 + abs(pos.string - 2.5) * 0.2
            self.graph[start_node].append((node, start_cost))
        
        # Add end node connected from all positions of the last note
        self.graph[end_node] = []
        last_idx = len(self.midi_sequence) - 1
        for pos in self.positions_by_note[last_idx]:
            node = (last_idx, pos)
            if node not in self.graph:
                self.graph[node] = []
            self.graph[node].append((end_node, 0))
        
        # Dijkstra's algorithm
        distances = {node: float('inf') for node in self.graph}
        distances[start_node] = 0
        previous = {node: None for node in self.graph}
        
        # Priority queue: (distance, node)
        pq = [(0, start_node)]
        visited = set()
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Found the shortest path to end
            if current_node == end_node:
                break
            
            # Explore neighbors
            for neighbor, edge_cost in self.graph.get(current_node, []):
                if neighbor in visited:
                    continue
                
                new_dist = current_dist + edge_cost
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))
        
        # Reconstruct path
        if distances[end_node] == float('inf'):
            return None  # No path found
        
        path = []
        current = end_node
        
        while current is not None:
            if current != end_node and current != start_node:
                note_idx, position = current
                path.append(position)
            current = previous[current]
        
        path.reverse()
        return path
    
    def format_tablature(self, path: List[FretPosition], width: int = 80) -> str:
        """
        Format the fingering path as ASCII guitar tablature.
        
        Args:
            path: List of FretPosition objects representing the fingering
            width: Character width for each line of tablature
            
        Returns:
            ASCII tablature string
        """
        if not path:
            return "No tablature to display."
        
        # String names (high to low for display)
        string_names = ['e', 'B', 'G', 'D', 'A', 'E']
        
        # Initialize tablature lines (6 strings)
        num_strings = 6
        tab_lines = [f"{string_names[i]}|" for i in range(num_strings)]
        
        # Track position in tablature
        positions_per_line = (width - 2) // 4  # Rough estimate
        
        for idx, pos in enumerate(path):
            # Convert string index (0=low E) to display index (0=high e)
            display_string = num_strings - 1 - pos.string
            
            # Format fret number
            fret_str = str(pos.fret)
            
            # Add to appropriate string line
            for i in range(num_strings):
                if i == display_string:
                    tab_lines[i] += f"{fret_str:>2}-"
                else:
                    tab_lines[i] += "---"
            
            # Add line break if needed
            if (idx + 1) % positions_per_line == 0 and idx < len(path) - 1:
                for i in range(num_strings):
                    tab_lines[i] += "|\n" + f"{string_names[i]}|"
        
        # Close all lines
        for i in range(num_strings):
            tab_lines[i] += "|"
        
        # Join all strings
        tablature = "\n".join(tab_lines)
        
        # Add header
        header = "=" * width + "\n"
        header += "GUITAR TABLATURE (Optimized via Dijkstra's Algorithm)\n"
        header += "=" * width + "\n\n"
        
        # Add footer with statistics
        footer = "\n\n" + "=" * width + "\n"
        footer += f"Total notes: {len(path)}\n"
        
        # Calculate some statistics
        if len(path) > 1:
            total_fret_distance = sum(abs(path[i].fret - path[i+1].fret) 
                                     for i in range(len(path) - 1))
            total_string_jumps = sum(abs(path[i].string - path[i+1].string) 
                                    for i in range(len(path) - 1))
            avg_fret_distance = total_fret_distance / (len(path) - 1)
            avg_string_jumps = total_string_jumps / (len(path) - 1)
            
            footer += f"Average fret movement: {avg_fret_distance:.2f}\n"
            footer += f"Average string jumps: {avg_string_jumps:.2f}\n"
        
        footer += "=" * width
        
        return header + tablature + footer
    
    def optimize(self, midi_sequence: List[MidiNote]) -> Tuple[Optional[List[FretPosition]], str]:
        """
        Complete optimization pipeline: build graph, find path, format tablature.
        
        Args:
            midi_sequence: List of MidiNote objects
            
        Returns:
            Tuple of (optimal_path, tablature_string)
        """
        self.build_graph(midi_sequence)
        path = self.find_best_path()
        
        if path is None:
            return None, "No valid fingering path found."
        
        tablature = self.format_tablature(path)
        return path, tablature


def midi_number_to_note_name(midi_number: int) -> str:
    """Convert MIDI note number to note name (e.g., 60 -> C4)."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]
    return f"{note}{octave}"


# Example usage demonstration
if __name__ == "__main__":
    # Create a simple melody: C major scale ascending
    test_melody = [
        MidiNote(60, 0.0, 0.5),   # C4
        MidiNote(62, 0.5, 1.0),   # D4
        MidiNote(64, 1.0, 1.5),   # E4
        MidiNote(65, 1.5, 2.0),   # F4
        MidiNote(67, 2.0, 2.5),   # G4
        MidiNote(69, 2.5, 3.0),   # A4
        MidiNote(71, 3.0, 3.5),   # B4
        MidiNote(72, 3.5, 4.0),   # C5
    ]
    
    # Create optimizer and run
    optimizer = FretboardOptimizer()
    path, tablature = optimizer.optimize(test_melody)
    
    # Display results
    print(tablature)
    
    if path:
        print("\n\nDetailed Path:")
        for i, (note, pos) in enumerate(zip(test_melody, path)):
            note_name = midi_number_to_note_name(note.midi_number)
            print(f"{i+1}. {note_name} (MIDI {note.midi_number}) -> "
                  f"String {pos.string + 1}, Fret {pos.fret}")
