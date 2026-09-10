"""
Format guitar fingering paths as ASCII tablature.

This module handles the conversion of optimized fingering paths to readable
guitar tablature format, including line-wrapping and hand-position markers.

Hand-position markers (e.g. ``[pos 5]``) are printed at the start of each
wrapped line when the optimizer has provided hand_positions, so the player
knows exactly where to anchor the index finger.
"""

from typing import List, Optional
from .optimizer import FretPosition
from .extractor import MidiNote, midi_number_to_note_name

# Number of notes printed per wrapped line
NOTES_PER_LINE = 16

# Roman numerals for hand positions (fret 1-22)
_ROMAN = [
    '', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX',
    'XX', 'XXI', 'XXII',
]


def _roman(n: int) -> str:
    """Return a roman-numeral position label for fret n (1-indexed)."""
    if 1 <= n < len(_ROMAN):
        return _ROMAN[n]
    return str(n)


class TablatureFormatter:
    """Formats guitar fingering as ASCII tablature with optional position markers."""

    def __init__(self, width: int = 80, notes_per_line: int = NOTES_PER_LINE):
        """
        Initialise the formatter.

        Args:
            width:          Character width for header/footer separator lines.
            notes_per_line: Notes per wrapped tab line (default 16).
        """
        self.width         = width
        self.notes_per_line = notes_per_line
        self.string_names  = ['e', 'B', 'G', 'D', 'A', 'E']  # high→low

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_tab_lines(
        self,
        path: List[FretPosition],
        hand_positions: Optional[List[int]] = None,
        midi_sequence: Optional[List[MidiNote]] = None,
    ) -> List[str]:
        num_strings  = 6
        output_lines: List[str] = []
        has_hp = (hand_positions is not None
                  and len(hand_positions) == len(path))
        has_midi = (midi_sequence is not None and len(midi_sequence) == len(path))

        chunks = [
            path[i: i + self.notes_per_line]
            for i in range(0, len(path), self.notes_per_line)
        ]
        hp_chunks = (
            [hand_positions[i: i + self.notes_per_line]
             for i in range(0, len(hand_positions), self.notes_per_line)]
            if has_hp else [None] * len(chunks)
        )

        prev_hp = None

        for chunk_idx, (chunk, hp_chunk) in enumerate(zip(chunks, hp_chunks)):
            # ── Position-shift header ─────────────────────────────────────────
            if has_hp and hp_chunk:
                chunk_start_hp = hp_chunk[0]
                if chunk_start_hp != prev_hp:
                    label = _roman(chunk_start_hp + 1)
                    output_lines.append(f"  [{label} pos]")
                prev_hp = hp_chunk[-1]

            # ── Six string lines for this chunk ───────────────────────────────
            tab_lines = [f"{self.string_names[s]}|" for s in range(num_strings)]

            for note_idx, pos in enumerate(chunk):
                global_idx = chunk_idx * self.notes_per_line + note_idx
                display_string = num_strings - 1 - pos.string
                fret_str = str(pos.fret)

                # Check for slides/bends from the previous note
                connector = "-"
                if has_midi and global_idx > 0:
                    prev_note = midi_sequence[global_idx - 1]
                    curr_note = midi_sequence[global_idx]
                    prev_pos = path[global_idx - 1]
                    
                    # If very close in time (< 0.05s gap) and on the same string
                    if curr_note.onset - prev_note.offset < 0.05 and pos.string == prev_pos.string:
                        fret_diff = pos.fret - prev_pos.fret
                        if 0 < fret_diff <= 2:
                            connector = "b"  # bend up 1-2 frets
                        elif fret_diff > 2:
                            connector = "/"  # slide up
                        elif fret_diff < 0:
                            connector = "\\" # slide down / release

                # Check if hand position shifts *within* this chunk
                if has_hp and hp_chunk and note_idx > 0:
                    if hp_chunk[note_idx] != hp_chunk[note_idx - 1]:
                        # Mark a shift with a small ↓ tick on the relevant string
                        for s in range(num_strings):
                            tab_lines[s] += f"-↓{connector}" if s == display_string else "---"
                        continue

                for s in range(num_strings):
                    if s == display_string:
                        tab_lines[s] += f"{connector}{fret_str:>2}-"
                    else:
                        tab_lines[s] += "----" if len(connector) + max(2, len(fret_str)) + 1 > 4 else "---"

            # Close each string line
            for s in range(num_strings):
                tab_lines[s] += "|"

            output_lines.extend(tab_lines)

            # Blank separator between chunks (except after the last)
            if chunk_idx < len(chunks) - 1:
                output_lines.append("")

        return output_lines

    # ── Public API ────────────────────────────────────────────────────────────

    def format(
        self,
        path: List[FretPosition],
        midi_sequence: Optional[List[MidiNote]] = None,
        hand_positions: Optional[List[int]] = None,
        clean: bool = False,
    ) -> str:
        """
        Format fingering path as ASCII guitar tablature.

        Args:
            path:           FretPosition list.
            midi_sequence:  Optional (unused in body; kept for API compat).
            hand_positions: Optional hand-position list from the optimizer.
                            When supplied, position-shift markers are rendered.
            clean:          If True, return only the tab body (no header/footer).

        Returns:
            ASCII tablature string with line-wrapping and optional position markers.
        """
        if not path:
            return "No tablature to display."

        tab_lines = self._build_tab_lines(path, hand_positions, midi_sequence)
        tablature = "\n".join(tab_lines)

        if clean:
            return tablature

        sep = "=" * self.width
        header = (
            f"{sep}\n"
            f"GUITAR TABLATURE (Hand-Position-Aware Dijkstra)\n"
            f"{sep}\n\n"
        )

        footer_parts = [f"\n\n{sep}", f"Total notes: {len(path)}"]
        if len(path) > 1:
            total_fret = sum(
                abs(path[i].fret - path[i + 1].fret)
                for i in range(len(path) - 1)
            )
            total_str = sum(
                abs(path[i].string - path[i + 1].string)
                for i in range(len(path) - 1)
            )
            footer_parts.append(
                f"Average fret movement: {total_fret / (len(path) - 1):.2f}"
            )
            footer_parts.append(
                f"Average string jumps:  {total_str / (len(path) - 1):.2f}"
            )
            if hand_positions and len(hand_positions) == len(path):
                shifts = sum(
                    1 for i in range(1, len(hand_positions))
                    if hand_positions[i] != hand_positions[i - 1]
                )
                footer_parts.append(f"Hand position shifts:  {shifts}")

        footer_parts.append(sep)
        footer = "\n".join(footer_parts)

        return header + tablature + footer

    def format_detailed(
        self,
        path: List[FretPosition],
        midi_sequence: List[MidiNote],
        hand_positions: Optional[List[int]] = None,
    ) -> str:
        """
        Format with per-note detail appended below the tab.

        Args:
            path:           FretPosition list.
            midi_sequence:  MidiNote list (same length as path).
            hand_positions: Optional hand-position list.

        Returns:
            Detailed tablature with note names, timing, and position info.
        """
        output = self.format(path, midi_sequence, hand_positions)

        output += "\n\nDetailed Note Information:\n"
        output += "-" * self.width + "\n"

        string_names_full = ['E (low)', 'A', 'D', 'G', 'B', 'E (high)']

        for i, (note, pos) in enumerate(zip(midi_sequence, path)):
            note_name = midi_number_to_note_name(note.midi_number)
            duration  = note.offset - note.onset
            hp_str = (f"  hp={hand_positions[i]}" if hand_positions else "")
            output += (
                f"{i + 1:3d}. {note_name:>4} | "
                f"String {pos.string + 1} ({string_names_full[pos.string]:>8}), "
                f"Fret {pos.fret:>2}{hp_str} | "
                f"Time: {note.onset:.2f}–{note.offset:.2f}s ({duration:.2f}s)\n"
            )

        return output

    def save(
        self,
        path: List[FretPosition],
        output_path: str,
        midi_sequence: Optional[List[MidiNote]] = None,
        hand_positions: Optional[List[int]] = None,
        detailed: bool = False,
    ) -> None:
        """
        Save tablature to a file.

        Args:
            path:           FretPosition list.
            output_path:    Destination file path.
            midi_sequence:  Optional MidiNote list.
            hand_positions: Optional hand-position list.
            detailed:       Whether to include per-note detail.
        """
        if detailed and midi_sequence:
            content = self.format_detailed(path, midi_sequence, hand_positions)
        else:
            content = self.format(path, midi_sequence, hand_positions)

        with open(output_path, "w") as f:
            f.write(content)

        print(f"Saved tablature to: {output_path}")


# ── Convenience ───────────────────────────────────────────────────────────────

def print_tablature(
    path: List[FretPosition],
    midi_sequence: Optional[List[MidiNote]] = None,
    hand_positions: Optional[List[int]] = None,
) -> None:
    """Print tablature to the console."""
    formatter = TablatureFormatter()
    print(formatter.format(path, midi_sequence, hand_positions))
