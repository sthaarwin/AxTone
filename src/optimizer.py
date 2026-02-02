"""
MIDI to optimal guitar fingering using Dijkstra's Algorithm.

This module implements graph-based optimization to find the easiest fingering path
through a sequence of MIDI notes on a guitar fretboard.
"""

import heapq
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .extractor import MidiNote


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
    
    # Maximum fret to consider
    MAX_FRET = 22
    
    # Cost function parameters
    FRET_DISTANCE_WEIGHT = 1.0
    STRING_JUMP_WEIGHT = 0.5
    STRETCH_PENALTY = 100.0
    STRETCH_THRESHOLD = 4
    OPEN_STRING_BONUS = -0.3
    
    def __init__(self, tuning: Optional[List[int]] = None, max_fret: int = 22):
        """
        Initialize the FretboardOptimizer.
        
        Args:
            tuning: List of MIDI note numbers for each string (low to high).
            max_fret: Maximum fret number to consider.
        """
        self.tuning = tuning if tuning is not None else self.STANDARD_TUNING
        self.max_fret = max_fret
        self.graph = {}
        self.midi_sequence = []
        self.positions_by_note = {}
        
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
            fret = midi_note - open_string_midi
            
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
        
        Args:
            pos1: Starting position
            pos2: Ending position
            
        Returns:
            Transition cost (lower is better)
        """
        fret_distance = abs(pos1.fret - pos2.fret)
        string_jump = abs(pos1.string - pos2.string)
        
        cost = (fret_distance * self.FRET_DISTANCE_WEIGHT + 
                string_jump * self.STRING_JUMP_WEIGHT)
        
        # Stretch penalty
        if fret_distance > self.STRETCH_THRESHOLD:
            cost += self.STRETCH_PENALTY
        
        # Open string bonus
        if pos2.fret == 0:
            cost += self.OPEN_STRING_BONUS
        
        return max(0, cost)
    
    def build_graph(self, midi_sequence: List[MidiNote]) -> None:
        """
        Build the graph representation of all possible fingering paths.
        
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
                    self.graph[node1].append((node2, cost))
    
    def find_best_path(self) -> Optional[List[FretPosition]]:
        """
        Find the optimal fingering path using Dijkstra's algorithm.
        
        Returns:
            List of FretPosition objects representing the optimal path
        """
        if not self.midi_sequence or not self.graph:
            return None
        
        # Virtual start and end nodes
        start_node = ('START', None)
        end_node = ('END', None)
        
        # Add start node
        self.graph[start_node] = []
        for pos in self.positions_by_note[0]:
            node = (0, pos)
            start_cost = pos.fret * 0.1 + abs(pos.string - 2.5) * 0.2
            self.graph[start_node].append((node, start_cost))
        
        # Add end node
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
        
        pq = [(0, start_node)]
        visited = set()
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            if current_node == end_node:
                break
            
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
            return None
        
        path = []
        current = end_node
        
        while current is not None:
            if current != end_node and current != start_node:
                note_idx, position = current
                path.append(position)
            current = previous[current]
        
        path.reverse()
        return path
    
    def optimize(self, midi_sequence: List[MidiNote]) -> Optional[List[FretPosition]]:
        """
        Complete optimization: build graph and find best path.
        
        Args:
            midi_sequence: List of MidiNote objects
            
        Returns:
            List of FretPosition objects (optimal path)
        """
        print(f"Optimizing fingering for {len(midi_sequence)} notes...")
        self.build_graph(midi_sequence)
        path = self.find_best_path()
        
        if path:
            print(f"Found optimal path with {len(path)} positions")
        else:
            print("No valid fingering path found")
        
        return path
