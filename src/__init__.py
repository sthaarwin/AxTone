"""
Vocal-to-Guitar-Tab: AI-powered vocal melody to guitar tablature conversion.

This package provides tools for converting audio files into optimized guitar tablature
using pitch detection and graph algorithms.
"""

from .extractor import AudioExtractor, MidiNote, midi_number_to_note_name, consolidate_notes
from .optimizer import FretboardOptimizer, FretPosition
from .formatter import TablatureFormatter, print_tablature
from . import utils

__version__ = "0.1.0"
__all__ = [
    "AudioExtractor",
    "FretboardOptimizer",
    "TablatureFormatter",
    "MidiNote",
    "FretPosition",
    "midi_number_to_note_name",
    "print_tablature",
    "utils"
]
