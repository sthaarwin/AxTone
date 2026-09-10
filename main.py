#!/usr/bin/env python3
"""
Vocal-to-Guitar-Tab: Main entry point

Convert audio files (.mp3, .wav) to guitar tablature.

Usage:
    python main.py input.mp3
    python main.py input.mp3 --output data/output/tab.txt
    python main.py input.mp3 --method basic_pitch --tuning drop-d
"""

import argparse
import os
import sys
from pathlib import Path

from src.extractor import AudioExtractor
from src.optimizer import FretboardOptimizer
from src.formatter import TablatureFormatter
from src import utils


# Preset tunings
TUNINGS = {
    'standard': [40, 45, 50, 55, 59, 64],  # E2-A2-D3-G3-B3-E4
    'drop-d': [38, 45, 50, 55, 59, 64],    # D2-A2-D3-G3-B3-E4
    'drop-c': [36, 43, 48, 53, 57, 62],    # C2-G2-C3-F3-A3-D4
    'open-g': [38, 43, 50, 55, 59, 62],    # D2-G2-D3-G3-B3-D4
    'dadgad': [38, 45, 50, 55, 45, 50],    # D2-A2-D3-G3-A3-D4
}


def main():
    parser = argparse.ArgumentParser(
        description='Convert vocal melodies to guitar tablature',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py vocals.mp3
  python main.py vocals.mp3 --output my_tab.txt
  python main.py vocals.mp3 --method basic_pitch
  python main.py vocals.mp3 --tuning drop-d
  python main.py vocals.mp3 --detailed
        """
    )
    
    parser.add_argument('input', help='Input audio file (.mp3, .wav, etc.)')
    parser.add_argument('-o', '--output', help='Output tablature file (default: auto-generated)')
    parser.add_argument('-m', '--method', choices=['basic_pitch', 'pyin'], 
                       default='pyin', help='Pitch detection method (default: pyin)')
    parser.add_argument('-t', '--tuning', default='standard',
                       help='Guitar tuning (standard, drop-d, drop-c, open-g, dadgad)')
    parser.add_argument('--save-midi', action='store_true',
                       help='Save extracted MIDI file')
    parser.add_argument('--detailed', action='store_true',
                       help='Include detailed note information in output')
    parser.add_argument('--min-duration', type=float, default=0.1,
                       help='Minimum note duration in seconds (default: 0.1)')
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess audio (normalize and trim)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("=" * 80)
    print("VOCAL-TO-GUITAR-TAB CONVERTER")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Method: {args.method}")
    print(f"Tuning: {args.tuning}")
    print("=" * 80)
    print()
    
    # Setup paths
    input_path = Path(args.input)
    base_name = input_path.stem
    
    if args.output:
        output_path = args.output
    else:
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{base_name}_tab.txt"
    
    # Preprocess audio if requested
    audio_path = args.input
    if args.preprocess:
        print("Step 1: Preprocessing audio...")
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / f"{base_name}_processed.wav"
        
        utils.preprocess_audio(
            args.input,
            output_path=str(processed_path),
            normalize=True,
            trim=True
        )
        audio_path = str(processed_path)
        print()
    
    # Step 1: Extract MIDI from audio
    print(f"Step {'2' if args.preprocess else '1'}: Extracting MIDI notes from audio...")
    extractor = AudioExtractor(method=args.method, min_note_duration=args.min_duration)
    
    try:
        midi_notes = extractor.extract(audio_path)
    except Exception as e:
        print(f"Error during MIDI extraction: {e}")
        sys.exit(1)
    
    if not midi_notes:
        print("Error: No notes detected in audio file")
        print("Try:")
        print("  - Using a clearer recording")
        print("  - Adjusting --min-duration parameter")
        print("  - Using --method basic_pitch for better accuracy")
        sys.exit(1)
    
    print(f"✓ Extracted {len(midi_notes)} notes")
    
    # Show pitch range
    midi_nums = [n.midi_number for n in midi_notes]
    from src.extractor import midi_number_to_note_name
    print(f"  Pitch range: {midi_number_to_note_name(min(midi_nums))} to {midi_number_to_note_name(max(midi_nums))}")
    print()
    
    # Save MIDI if requested
    if args.save_midi:
        midi_output = Path("data/output") / f"{base_name}.mid"
        midi_output.parent.mkdir(parents=True, exist_ok=True)
        extractor.save_midi(midi_notes, str(midi_output))
        print()
    
    # Step 2: Optimize fingering
    print(f"Step {'3' if args.preprocess else '2'}: Optimizing guitar fingering...")
    
    # Get tuning
    if args.tuning in TUNINGS:
        tuning = TUNINGS[args.tuning]
    else:
        try:
            # Try parsing as comma-separated MIDI numbers
            tuning = [int(x) for x in args.tuning.split(',')]
            if len(tuning) != 6:
                print(f"Error: Custom tuning must have 6 strings")
                sys.exit(1)
        except:
            print(f"Error: Unknown tuning '{args.tuning}'")
            print(f"Available: {', '.join(TUNINGS.keys())}")
            print(f"Or provide custom as comma-separated MIDI numbers")
            sys.exit(1)
    
    optimizer = FretboardOptimizer(tuning=tuning)
    
    try:
        path, _ = optimizer.optimize(midi_notes)  # optimize() returns (path, tablature)
    except Exception as e:
        print(f"Error during optimization: {e}")
        sys.exit(1)
    
    if not path:
        print("Error: Could not find valid fingering path")
        print("This might mean:")
        print("  - Notes are outside guitar range")
        print("  - Tuning doesn't support these notes")
        sys.exit(1)
    
    print()
    
    # Step 3: Format and save tablature
    print(f"Step {'4' if args.preprocess else '3'}: Generating tablature...")
    formatter = TablatureFormatter()
    
    formatter.save(
        path,
        output_path,
        midi_sequence=midi_notes,
        hand_positions=optimizer.hand_positions or None,
        detailed=args.detailed
    )
    
    print()
    print("=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"Tablature saved to: {output_path}")
    
    if args.save_midi:
        print(f"MIDI saved to: {midi_output}")
    
    print()
    print("Preview:")
    print("-" * 80)
    
    # Print preview (first 10 notes)
    preview_path = path[:min(10, len(path))]
    preview_notes = midi_notes[:min(10, len(midi_notes))]
    print(formatter.format(preview_path, preview_notes))
    
    if len(path) > 10:
        print(f"\n... ({len(path) - 10} more notes in full file)")
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
