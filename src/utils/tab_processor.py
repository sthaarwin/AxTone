"""
Tab processing utilities for converting scraped tabs into training data format.

This module provides tools for parsing, cleaning, and converting guitar tablature
into tensor representations suitable for machine learning models.
"""

import re
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TabSection:
    """Represents a section of tablature with timing information."""
    lines: List[str]
    section_type: str  # 'verse', 'chorus', 'solo', 'intro', etc.
    time_signature: Optional[str] = None
    tempo: Optional[int] = None
    
class TabParser:
    """Parser for guitar tablature text."""
    
    def __init__(self):
        """Initialize the tab parser with standard patterns."""
        # Standard guitar tuning (EADGBE from low to high)
        self.standard_tuning = ['E', 'A', 'D', 'G', 'B', 'E']
        
        # Patterns for detecting different parts of a tab
        self.section_patterns = {
            'intro': re.compile(r'\b(intro|introduction)\b', re.IGNORECASE),
            'verse': re.compile(r'\b(verse|ver)\b', re.IGNORECASE),
            'chorus': re.compile(r'\b(chorus|cho)\b', re.IGNORECASE),
            'bridge': re.compile(r'\b(bridge|bri)\b', re.IGNORECASE),
            'solo': re.compile(r'\b(solo|lead|guitar solo)\b', re.IGNORECASE),
            'outro': re.compile(r'\b(outro|ending|end)\b', re.IGNORECASE),
        }
        
        # Pattern for detecting tab lines (contains dashes and numbers/letters)
        self.tab_line_pattern = re.compile(r'^[A-Ga-g]?\|?[-\d\w\|/\\()~^v<>phbr\s]+[-\d\w\|/\\()~^v<>phbr][-\d\w\|/\\()~^v<>phbr\s]*$')
        
        # Pattern for detecting tuning information
        self.tuning_pattern = re.compile(r'tuning[:\s]*([A-Ga-g][\s\-]*)+', re.IGNORECASE)
        
        # Pattern for detecting time signature
        self.time_sig_pattern = re.compile(r'(\d+)/(\d+)')
        
        # Pattern for detecting tempo
        self.tempo_pattern = re.compile(r'(?:tempo|bpm)[:\s]*(\d+)', re.IGNORECASE)
        
        # Special techniques mapping
        self.techniques = {
            'h': 'hammer_on',
            'p': 'pull_off',
            'b': 'bend',
            'r': 'release',
            '/': 'slide_up',
            '\\': 'slide_down',
            '~': 'vibrato',
            '^': 'bend',
            'v': 'vibrato',
            '()': 'ghost_note',
            'x': 'mute',
            'X': 'mute'
        }
    
    def parse_tab_file(self, file_path: str) -> Dict:
        """
        Parse a tab file and extract structured information.
        
        Args:
            file_path: Path to the tab file
            
        Returns:
            Dictionary containing parsed tab information
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return self.parse_tab_content(content)
    
    def parse_tab_content(self, content: str) -> Dict:
        """
        Parse tab content and extract structured information.
        
        Args:
            content: Tab content as string
            
        Returns:
            Dictionary containing parsed tab information
        """
        lines = content.split('\n')
        
        # Extract metadata
        metadata = self._extract_metadata(lines)
        
        # Find and group tab sections
        sections = self._find_tab_sections(lines)
        
        # Parse each section
        parsed_sections = []
        for section in sections:
            parsed_section = self._parse_section(section)
            if parsed_section:
                parsed_sections.append(parsed_section)
        
        return {
            'metadata': metadata,
            'sections': parsed_sections,
            'raw_content': content
        }
    
    def _extract_metadata(self, lines: List[str]) -> Dict:
        """Extract metadata from tab header."""
        metadata = {
            'tuning': self.standard_tuning.copy(),
            'time_signature': None,
            'tempo': None,
            'capo': None,
            'difficulty': None
        }
        
        # Look for metadata in first 20 lines
        header_lines = lines[:20]
        
        for line in header_lines:
            line = line.strip()
            
            # Tuning
            tuning_match = self.tuning_pattern.search(line)
            if tuning_match:
                tuning_str = tuning_match.group(1)
                tuning = re.findall(r'[A-Ga-g]', tuning_str)
                if len(tuning) == 6:
                    metadata['tuning'] = tuning
            
            # Time signature
            time_sig_match = self.time_sig_pattern.search(line)
            if time_sig_match:
                metadata['time_signature'] = f"{time_sig_match.group(1)}/{time_sig_match.group(2)}"
            
            # Tempo
            tempo_match = self.tempo_pattern.search(line)
            if tempo_match:
                metadata['tempo'] = int(tempo_match.group(1))
            
            # Capo
            if 'capo' in line.lower():
                capo_match = re.search(r'capo[:\s]*(\d+)', line, re.IGNORECASE)
                if capo_match:
                    metadata['capo'] = int(capo_match.group(1))
        
        return metadata
    
    def _find_tab_sections(self, lines: List[str]) -> List[TabSection]:
        """Find and group lines into tab sections."""
        sections = []
        current_section_lines = []
        current_section_type = 'unknown'
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines at the start of a section
            if not line and not current_section_lines:
                i += 1
                continue
            
            # Check if this line indicates a new section
            section_type = self._identify_section_type(line)
            if section_type and current_section_lines:
                # Save current section and start new one
                sections.append(TabSection(
                    lines=current_section_lines.copy(),
                    section_type=current_section_type
                ))
                current_section_lines = []
                current_section_type = section_type
            elif section_type:
                current_section_type = section_type
            
            # Check if this is a tab line
            if self._is_tab_line(line):
                current_section_lines.append(line)
            elif line and not section_type:
                # Non-empty, non-section line might be chord names or other info
                current_section_lines.append(line)
            
            i += 1
        
        # Add final section if it has content
        if current_section_lines:
            sections.append(TabSection(
                lines=current_section_lines,
                section_type=current_section_type
            ))
        
        return sections
    
    def _identify_section_type(self, line: str) -> Optional[str]:
        """Identify what type of section a line represents."""
        for section_type, pattern in self.section_patterns.items():
            if pattern.search(line):
                return section_type
        return None
    
    def _is_tab_line(self, line: str) -> bool:
        """Check if a line contains tablature notation."""
        # Must contain dashes and numbers/letters
        if not re.search(r'[-\d]', line):
            return False
        
        # Should match the general tab line pattern
        return bool(self.tab_line_pattern.match(line))
    
    def _parse_section(self, section: TabSection) -> Optional[Dict]:
        """Parse a single tab section into structured data."""
        if not section.lines:
            return None
        
        # Group lines into tab groups (usually 6 lines for 6 strings)
        tab_groups = self._group_tab_lines(section.lines)
        
        parsed_groups = []
        for group in tab_groups:
            parsed_group = self._parse_tab_group(group)
            if parsed_group:
                parsed_groups.append(parsed_group)
        
        if not parsed_groups:
            return None
        
        return {
            'section_type': section.section_type,
            'tab_groups': parsed_groups,
            'time_signature': section.time_signature,
            'tempo': section.tempo
        }
    
    def _group_tab_lines(self, lines: List[str]) -> List[List[str]]:
        """Group consecutive tab lines together."""
        groups = []
        current_group = []
        
        for line in lines:
            if self._is_tab_line(line):
                current_group.append(line)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                # Non-tab lines might be chord names or section markers
                if line.strip() and not any(pattern.search(line) for pattern in self.section_patterns.values()):
                    # Treat as a single-line group (chord progression, etc.)
                    groups.append([line])
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _parse_tab_group(self, lines: List[str]) -> Optional[Dict]:
        """Parse a group of tab lines (typically 6 strings)."""
        if not lines:
            return None
        
        # Clean and align the lines
        cleaned_lines = []
        for line in lines:
            # Remove string labels (E|, A|, etc.) if present
            line = re.sub(r'^[A-Ga-g]\|?', '', line)
            cleaned_lines.append(line.rstrip())
        
        # Find the maximum length
        max_length = max(len(line) for line in cleaned_lines) if cleaned_lines else 0
        if max_length == 0:
            return None
        
        # Parse positions and techniques using a smarter approach
        positions = []
        techniques = []
        
        # For each string, extract fret numbers and techniques
        string_positions = []
        for line in cleaned_lines:
            # Extract fret numbers from the line using regex
            # Look for numbers (including multi-digit) that are separated by dashes or other characters
            fret_pattern = r'(\d+|[h|p|b|r|/|\\|~|\^|v|x|X|\(\)])'
            matches = re.findall(fret_pattern, line)
            
            # Convert to positions with their character indices
            line_positions = []
            current_pos = 0
            
            for match in re.finditer(fret_pattern, line):
                value = match.group(1)
                char_pos = match.start()
                
                if value.isdigit():
                    line_positions.append((char_pos, int(value), None))
                else:
                    # It's a technique marker
                    line_positions.append((char_pos, -1, value))
            
            string_positions.append(line_positions)
        
        # Now align positions across all strings by character position
        if not string_positions or not any(string_positions):
            return None
        
        # Find all unique character positions where notes occur
        all_positions = set()
        for string_pos in string_positions:
            for char_pos, fret, technique in string_pos:
                all_positions.add(char_pos)
        
        # Sort positions chronologically
        sorted_positions = sorted(all_positions)
        
        # Create position data for each time point
        for pos_idx in sorted_positions:
            position_data = {}
            technique_data = {}
            
            # For each string, find what's happening at this position
            for string_idx, string_pos in enumerate(string_positions):
                # Find the closest position entry for this string
                closest_entry = None
                min_distance = float('inf')
                
                for char_pos, fret, technique in string_pos:
                    distance = abs(char_pos - pos_idx)
                    if distance < min_distance and distance <= 2:  # Allow small position variations
                        min_distance = distance
                        closest_entry = (fret, technique)
                
                if closest_entry:
                    fret, technique = closest_entry
                    if fret != -1:  # Valid fret number
                        position_data[string_idx] = fret
                    if technique:
                        technique_data[string_idx] = self.techniques.get(technique, technique)
                else:
                    # No note on this string at this position
                    position_data[string_idx] = -1
            
            positions.append(position_data)
            techniques.append(technique_data)
        
        return {
            'raw_lines': lines,
            'positions': positions,
            'techniques': techniques,
            'length': len(positions)
        }
    
    def to_tensor(self, parsed_tab: Dict, max_length: int = None) -> torch.Tensor:
        """
        Convert parsed tab to tensor representation.
        
        Args:
            parsed_tab: Parsed tab dictionary
            max_length: Maximum sequence length (for padding/truncating)
            
        Returns:
            Tensor of shape [num_strings, sequence_length] with fret numbers
        """
        all_positions = []
        
        # Collect all positions from all sections
        for section in parsed_tab['sections']:
            for group in section['tab_groups']:
                if 'positions' in group:
                    all_positions.extend(group['positions'])
        
        if not all_positions:
            # Return empty tensor if no positions found
            if max_length:
                return torch.full((6, max_length), -1, dtype=torch.long)
            else:
                return torch.full((6, 1), -1, dtype=torch.long)
        
        # Determine sequence length
        seq_length = len(all_positions)
        if max_length:
            seq_length = min(seq_length, max_length)
        
        # Create tensor
        tensor = torch.full((6, seq_length), -1, dtype=torch.long)
        
        # Fill tensor with fret positions
        for pos_idx in range(seq_length):
            position = all_positions[pos_idx]
            for string_idx in range(6):
                if string_idx in position:
                    fret = position[string_idx]
                    if 0 <= fret <= 24:  # Valid fret range
                        tensor[string_idx, pos_idx] = fret
        
        return tensor


class TabProcessor:
    """High-level processor for converting tabs to training data."""
    
    def __init__(self, output_dir: str = "data/processed_tabs"):
        """
        Initialize the tab processor.
        
        Args:
            output_dir: Directory to save processed tabs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = TabParser()
    
    def process_scraped_tabs(self, scraped_dir: str, max_length: int = 1024) -> Dict:
        """
        Process all scraped tabs into training format.
        
        Args:
            scraped_dir: Directory containing scraped tabs
            max_length: Maximum sequence length for tensors
            
        Returns:
            Processing statistics
        """
        scraped_path = Path(scraped_dir)
        
        # Find all tab files
        tab_files = []
        for tab_type_dir in scraped_path.iterdir():
            if tab_type_dir.is_dir():
                tab_files.extend(tab_type_dir.glob("*.tab"))
        
        logger.info(f"Found {len(tab_files)} tab files to process")
        
        processed_count = 0
        failed_count = 0
        total_sequences = 0
        
        # Process each tab file
        for tab_file in tab_files:
            try:
                # Parse the tab
                parsed_tab = self.parser.parse_tab_file(str(tab_file))
                
                # Convert to tensor
                tensor = self.parser.to_tensor(parsed_tab, max_length)
                
                # Skip if tensor is empty or too short
                if tensor.shape[1] < 10:
                    logger.debug(f"Skipping {tab_file.name}: too short")
                    continue
                
                # Create output filename
                output_name = tab_file.stem + ".pt"
                output_path = self.output_dir / output_name
                
                # Save tensor and metadata
                torch.save({
                    'tensor': tensor,
                    'metadata': parsed_tab['metadata'],
                    'source_file': str(tab_file),
                    'shape': tensor.shape
                }, output_path)
                
                processed_count += 1
                total_sequences += tensor.shape[1]
                
                logger.debug(f"Processed {tab_file.name} -> {tensor.shape}")
                
            except Exception as e:
                logger.error(f"Failed to process {tab_file.name}: {e}")
                failed_count += 1
        
        # Save processing statistics
        stats = {
            'total_files': len(tab_files),
            'processed': processed_count,
            'failed': failed_count,
            'total_sequences': total_sequences,
            'avg_sequence_length': total_sequences / processed_count if processed_count > 0 else 0,
            'max_length': max_length
        }
        
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing complete: {processed_count}/{len(tab_files)} files processed")
        return stats
    
    def create_training_dataset(self, processed_dir: str = None, 
                              train_split: float = 0.8, val_split: float = 0.1) -> Dict:
        """
        Create training/validation/test datasets from processed tabs.
        
        Args:
            processed_dir: Directory containing processed tensor files
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            
        Returns:
            Dataset split information
        """
        if processed_dir is None:
            processed_dir = self.output_dir
        
        processed_path = Path(processed_dir)
        
        # Find all processed tensor files
        tensor_files = list(processed_path.glob("*.pt"))
        
        if not tensor_files:
            raise ValueError(f"No processed tensor files found in {processed_dir}")
        
        # Shuffle files
        import random
        random.shuffle(tensor_files)
        
        # Calculate split sizes
        total_files = len(tensor_files)
        train_size = int(total_files * train_split)
        val_size = int(total_files * val_split)
        
        # Split files
        train_files = tensor_files[:train_size]
        val_files = tensor_files[train_size:train_size + val_size]
        test_files = tensor_files[train_size + val_size:]
        
        # Create split directories
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_dir = processed_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create symbolic links to avoid duplicating data
            for file in files:
                link_path = split_dir / file.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(file.resolve())
                    except OSError:
                        # If symlinks aren't supported, copy the file
                        import shutil
                        shutil.copy2(file, link_path)
        
        # Save split information
        split_info = {
            'total_files': total_files,
            'train_files': len(train_files),
            'val_files': len(val_files),
            'test_files': len(test_files),
            'train_split': train_split,
            'val_split': val_split,
            'test_split': 1 - train_split - val_split
        }
        
        split_info_file = processed_path / "split_info.json"
        with open(split_info_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Dataset split created: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        return split_info


def convert_tab_to_ascii_art(tensor: torch.Tensor, tuning: List[str] = None) -> str:
    """
    Convert a tab tensor back to ASCII tablature format.
    
    Args:
        tensor: Tab tensor of shape [num_strings, sequence_length]
        tuning: String tuning (default: standard tuning)
        
    Returns:
        ASCII tablature string
    """
    if tuning is None:
        tuning = ['E', 'A', 'D', 'G', 'B', 'E']
    
    num_strings, seq_length = tensor.shape
    
    # Create tab lines
    lines = []
    for string_idx in range(num_strings):
        string_name = tuning[string_idx] if string_idx < len(tuning) else f"S{string_idx}"
        line = f"{string_name}|"
        
        for pos_idx in range(seq_length):
            fret = tensor[string_idx, pos_idx].item()
            if fret == -1:
                line += "-"
            else:
                line += str(fret)
                
        lines.append(line)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    processor = TabProcessor()
    
    # Process scraped tabs
    stats = processor.process_scraped_tabs("data/scraped_tabs")
    print(f"Processed {stats['processed']} tabs")
    
    # Create training dataset
    split_info = processor.create_training_dataset()
    print(f"Created dataset with {split_info['train_files']} training files")