"""
Unit tests for FretboardOptimizer.

Tests the graph construction, Dijkstra's algorithm, cost functions,
and tablature formatting.
"""

import unittest
from src.fretboard_optimizer import (
    FretboardOptimizer, 
    MidiNote, 
    FretPosition,
    midi_number_to_note_name
)


class TestMidiNote(unittest.TestCase):
    """Test MidiNote dataclass."""
    
    def test_creation(self):
        note = MidiNote(60, 0.0, 0.5)
        self.assertEqual(note.midi_number, 60)
        self.assertEqual(note.onset, 0.0)
        self.assertEqual(note.offset, 0.5)
    
    def test_repr(self):
        note = MidiNote(60, 0.0, 0.5)
        repr_str = repr(note)
        self.assertIn("60", repr_str)
        self.assertIn("0.000", repr_str)


class TestFretPosition(unittest.TestCase):
    """Test FretPosition dataclass."""
    
    def test_creation(self):
        pos = FretPosition(string=2, fret=5, midi_note=62)
        self.assertEqual(pos.string, 2)
        self.assertEqual(pos.fret, 5)
        self.assertEqual(pos.midi_note, 62)
    
    def test_equality(self):
        pos1 = FretPosition(2, 5, 62)
        pos2 = FretPosition(2, 5, 62)
        pos3 = FretPosition(2, 6, 62)
        self.assertEqual(pos1, pos2)
        self.assertNotEqual(pos1, pos3)
    
    def test_hashable(self):
        pos1 = FretPosition(2, 5, 62)
        pos2 = FretPosition(2, 5, 62)
        pos_set = {pos1, pos2}
        self.assertEqual(len(pos_set), 1)


class TestFretboardOptimizer(unittest.TestCase):
    """Test FretboardOptimizer class."""
    
    def setUp(self):
        self.optimizer = FretboardOptimizer()
    
    def test_initialization(self):
        self.assertEqual(len(self.optimizer.tuning), 6)
        self.assertEqual(self.optimizer.max_fret, 22)
        self.assertEqual(self.optimizer.tuning[0], 40)  # Low E
        self.assertEqual(self.optimizer.tuning[5], 64)  # High E
    
    def test_custom_tuning(self):
        drop_d = [38, 45, 50, 55, 59, 64]
        optimizer = FretboardOptimizer(tuning=drop_d)
        self.assertEqual(optimizer.tuning[0], 38)  # Low D
    
    def test_get_possible_positions(self):
        # Test middle C (MIDI 60)
        positions = self.optimizer.get_possible_positions(60)
        
        # C4 should be playable on multiple strings
        self.assertGreater(len(positions), 0)
        self.assertLess(len(positions), 7)  # Can't be on all 6 strings
        
        # All positions should produce the same MIDI note
        for pos in positions:
            self.assertEqual(pos.midi_note, 60)
            self.assertGreaterEqual(pos.fret, 0)
            self.assertLessEqual(pos.fret, 22)
    
    def test_get_possible_positions_open_string(self):
        # Test open strings
        positions = self.optimizer.get_possible_positions(40)  # Low E
        has_open_string = any(pos.fret == 0 for pos in positions)
        self.assertTrue(has_open_string)
    
    def test_get_possible_positions_high_note(self):
        # Very high note might not be playable
        positions = self.optimizer.get_possible_positions(100)
        # Should have very few or no positions
        self.assertLessEqual(len(positions), 2)
    
    def test_calculate_transition_cost_same_position(self):
        pos = FretPosition(2, 5, 62)
        cost = self.optimizer.calculate_transition_cost(pos, pos)
        self.assertGreaterEqual(cost, 0)
        self.assertLess(cost, 1)  # Should be very small
    
    def test_calculate_transition_cost_fret_distance(self):
        pos1 = FretPosition(2, 5, 62)
        pos2 = FretPosition(2, 8, 65)  # Same string, 3 frets away
        cost = self.optimizer.calculate_transition_cost(pos1, pos2)
        self.assertGreater(cost, 0)
        self.assertLess(cost, 10)  # Reasonable cost
    
    def test_calculate_transition_cost_stretch_penalty(self):
        pos1 = FretPosition(2, 3, 53)
        pos2 = FretPosition(2, 10, 60)  # 7 frets away - big stretch!
        cost = self.optimizer.calculate_transition_cost(pos1, pos2)
        # Should have large penalty
        self.assertGreater(cost, 50)
    
    def test_calculate_transition_cost_open_string_bonus(self):
        pos1 = FretPosition(2, 5, 55)
        pos2_fretted = FretPosition(3, 5, 60)
        pos2_open = FretPosition(3, 0, 55)
        
        cost_fretted = self.optimizer.calculate_transition_cost(pos1, pos2_fretted)
        cost_open = self.optimizer.calculate_transition_cost(pos1, pos2_open)
        
        # Open string should have lower cost
        self.assertLess(cost_open, cost_fretted)
    
    def test_build_graph_empty(self):
        self.optimizer.build_graph([])
        self.assertEqual(len(self.optimizer.graph), 0)
    
    def test_build_graph_single_note(self):
        melody = [MidiNote(60, 0.0, 0.5)]
        self.optimizer.build_graph(melody)
        
        # Should have nodes for each possible position
        self.assertGreater(len(self.optimizer.graph), 0)
        self.assertEqual(len(self.optimizer.positions_by_note), 1)
    
    def test_build_graph_two_notes(self):
        melody = [
            MidiNote(60, 0.0, 0.5),
            MidiNote(62, 0.5, 1.0)
        ]
        self.optimizer.build_graph(melody)
        
        # Should have positions for both notes
        self.assertEqual(len(self.optimizer.positions_by_note), 2)
        
        # First note positions should have edges to second note positions
        first_positions = self.optimizer.positions_by_note[0]
        self.assertGreater(len(first_positions), 0)
        
        for pos in first_positions:
            node = (0, pos)
            edges = self.optimizer.graph[node]
            self.assertGreater(len(edges), 0)  # Should have edges to next note
    
    def test_find_best_path_empty(self):
        path = self.optimizer.find_best_path()
        self.assertIsNone(path)
    
    def test_find_best_path_single_note(self):
        melody = [MidiNote(60, 0.0, 0.5)]
        self.optimizer.build_graph(melody)
        path = self.optimizer.find_best_path()
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0].midi_note, 60)
    
    def test_find_best_path_scale(self):
        # C major scale
        melody = [
            MidiNote(60, 0.0, 0.5),
            MidiNote(62, 0.5, 1.0),
            MidiNote(64, 1.0, 1.5),
            MidiNote(65, 1.5, 2.0),
            MidiNote(67, 2.0, 2.5),
        ]
        self.optimizer.build_graph(melody)
        path = self.optimizer.find_best_path()
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), len(melody))
        
        # Verify each position matches the correct MIDI note
        for note, pos in zip(melody, path):
            self.assertEqual(pos.midi_note, note.midi_number)
    
    def test_format_tablature_empty(self):
        tab = self.optimizer.format_tablature([])
        self.assertIn("No tablature", tab)
    
    def test_format_tablature_single_note(self):
        path = [FretPosition(2, 5, 55)]
        tab = self.optimizer.format_tablature(path)
        
        self.assertIn("e|", tab)
        self.assertIn("E|", tab)
        self.assertIn("5", tab)
        self.assertIn("GUITAR TABLATURE", tab)
    
    def test_format_tablature_multiple_notes(self):
        path = [
            FretPosition(2, 5, 55),
            FretPosition(2, 7, 57),
            FretPosition(3, 5, 60),
        ]
        tab = self.optimizer.format_tablature(path)
        
        # Should contain fret numbers
        self.assertIn("5", tab)
        self.assertIn("7", tab)
        
        # Should have statistics
        self.assertIn("Total notes:", tab)
        self.assertIn("Average", tab)
    
    def test_optimize_complete_pipeline(self):
        melody = [
            MidiNote(60, 0.0, 0.5),
            MidiNote(62, 0.5, 1.0),
            MidiNote(64, 1.0, 1.5),
        ]
        
        path, tablature = self.optimizer.optimize(melody)
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)
        self.assertIn("GUITAR TABLATURE", tablature)
        self.assertIsInstance(tablature, str)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""
    
    def test_midi_to_note_name(self):
        self.assertEqual(midi_number_to_note_name(60), "C4")
        self.assertEqual(midi_number_to_note_name(69), "A4")
        self.assertEqual(midi_number_to_note_name(72), "C5")
        self.assertEqual(midi_number_to_note_name(61), "C#4")
        self.assertEqual(midi_number_to_note_name(48), "C3")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        self.optimizer = FretboardOptimizer()
    
    def test_very_low_note(self):
        # Note below guitar range
        positions = self.optimizer.get_possible_positions(30)
        # Might have 0 or very few positions
        self.assertIsInstance(positions, list)
    
    def test_very_high_note(self):
        # Note above typical guitar range
        positions = self.optimizer.get_possible_positions(90)
        # Should have very few or no positions
        self.assertLessEqual(len(positions), 3)
    
    def test_repeated_notes(self):
        # Same note repeated
        melody = [
            MidiNote(60, 0.0, 0.5),
            MidiNote(60, 0.5, 1.0),
            MidiNote(60, 1.0, 1.5),
        ]
        
        path, tablature = self.optimizer.optimize(melody)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)
    
    def test_wide_interval_jump(self):
        # Big interval jump
        melody = [
            MidiNote(40, 0.0, 0.5),  # Low E
            MidiNote(76, 0.5, 1.0),  # High E
        ]
        
        path, tablature = self.optimizer.optimize(melody)
        # Should still find a path, even if costly
        self.assertIsNotNone(path)


class TestCostFunctionCustomization(unittest.TestCase):
    """Test customization of cost function parameters."""
    
    def test_custom_stretch_threshold(self):
        optimizer = FretboardOptimizer()
        optimizer.STRETCH_THRESHOLD = 3  # Stricter stretch limit
        
        pos1 = FretPosition(2, 0, 50)
        pos2 = FretPosition(2, 5, 55)  # 5 fret stretch
        
        cost = optimizer.calculate_transition_cost(pos1, pos2)
        # Should have penalty since 5 > 3
        self.assertGreater(cost, 50)
    
    def test_custom_string_jump_weight(self):
        optimizer1 = FretboardOptimizer()
        optimizer1.STRING_JUMP_WEIGHT = 0.1  # Low penalty
        
        optimizer2 = FretboardOptimizer()
        optimizer2.STRING_JUMP_WEIGHT = 5.0  # High penalty
        
        pos1 = FretPosition(0, 5, 45)
        pos2 = FretPosition(5, 5, 64)  # Jump across all strings
        
        cost1 = optimizer1.calculate_transition_cost(pos1, pos2)
        cost2 = optimizer2.calculate_transition_cost(pos1, pos2)
        
        # Higher weight should produce higher cost
        self.assertLess(cost1, cost2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
