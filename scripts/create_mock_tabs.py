#!/usr/bin/env python3
"""
Test script to create mock tab data for testing the processing pipeline.
"""

import os
import json
from pathlib import Path

def create_mock_tab_data():
    """Create some mock tab data for testing the processing pipeline."""
    
    # Create demo dataset directory structure
    demo_dir = Path("data/demo_dataset/scraped_tabs")
    tabs_dir = demo_dir / "tabs"
    tabs_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample tab content
    mock_tabs = [
        {
            "filename": "Metallica - Enter Sandman (Ver 1).tab",
            "content": """Metallica - Enter Sandman
Tuning: E A D G B E

[Intro]
E|--0--0--0--0--0--0--0--0--|
B|--0--0--0--0--0--0--0--0--|
G|--0--0--0--0--0--0--0--0--|
D|--2--2--2--2--2--2--2--2--|
A|--2--2--2--2--2--2--2--2--|
E|--0--0--0--0--0--0--0--0--|

[Verse]
E|--0--0--0--0--3--3--3--3--|
B|--0--0--0--0--3--3--3--3--|
G|--0--0--0--0--0--0--0--0--|
D|--2--2--2--2--0--0--0--0--|
A|--2--2--2--2--2--2--2--2--|
E|--0--0--0--0--3--3--3--3--|
""",
            "metadata": {
                "artist": "Metallica",
                "title": "Enter Sandman",
                "version": "1",
                "rating": 4.8,
                "votes": 2547,
                "url": "https://tabs.ultimate-guitar.com/tab/metallica/enter_sandman_123456",
                "type": "Tabs"
            }
        },
        {
            "filename": "Led Zeppelin - Stairway To Heaven (Ver 1).tab",
            "content": """Led Zeppelin - Stairway To Heaven
Tuning: E A D G B E

[Intro]
E|--0--2--3--2--0-----------|
B|--3--3--3--3--3--3--1--0--|
G|--2--2--2--2--2--2--0--0--|
D|--0--0--0--0--0--0--2--0--|
A|--------------------------|
E|--------------------------|

[Verse]
E|--0--0--2--2--3--3--2--2--|
B|--1--1--3--3--3--3--3--3--|
G|--0--0--2--2--0--0--2--2--|
D|--2--2--0--0--0--0--0--0--|
A|--3--3--------------------|
E|--------------------------|
""",
            "metadata": {
                "artist": "Led Zeppelin",
                "title": "Stairway To Heaven",
                "version": "1",
                "rating": 4.9,
                "votes": 3245,
                "url": "https://tabs.ultimate-guitar.com/tab/led_zeppelin/stairway_to_heaven_123456",
                "type": "Tabs"
            }
        },
        {
            "filename": "Nirvana - Smells Like Teen Spirit (Ver 1).tab",
            "content": """Nirvana - Smells Like Teen Spirit
Tuning: E A D G B E

[Intro]
E|--------------------------------|
B|--------------------------------|
G|--------------------------------|
D|--5--5--5--5--8--8--7--7--5--5--|
A|--3--3--3--3--6--6--5--5--3--3--|
E|--------------------------------|

[Verse]
E|--0--0--0--0--0--0--0--0--|
B|--0--0--0--0--0--0--0--0--|
G|--1--1--1--1--6--6--6--6--|
D|--2--2--2--2--6--6--6--6--|
A|--2--2--2--2--4--4--4--4--|
E|--0--0--0--0--------------|
""",
            "metadata": {
                "artist": "Nirvana",
                "title": "Smells Like Teen Spirit",
                "version": "1",
                "rating": 4.7,
                "votes": 1892,
                "url": "https://tabs.ultimate-guitar.com/tab/nirvana/smells_like_teen_spirit_123456",
                "type": "Tabs"
            }
        }
    ]
    
    # Write tab files and metadata
    for tab_data in mock_tabs:
        # Write tab content
        tab_file = tabs_dir / tab_data["filename"]
        with open(tab_file, 'w', encoding='utf-8') as f:
            f.write(tab_data["content"])
        
        # Write metadata
        metadata_file = tabs_dir / (tab_data["filename"].replace('.tab', '.json'))
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(tab_data["metadata"], f, indent=2)
    
    # Create dataset metadata
    dataset_metadata = {
        "queries": ["metallica", "led zeppelin", "nirvana"],
        "tab_types": ["Tabs"],
        "total_downloaded": len(mock_tabs),
        "files": [str(tabs_dir / tab["filename"]) for tab in mock_tabs],
        "created_at": 1722768000,  # Mock timestamp
        "note": "Mock data for testing processing pipeline"
    }
    
    metadata_file = demo_dir / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print(f"✅ Created {len(mock_tabs)} mock tab files in {demo_dir}")
    print(f"📁 Files created:")
    for tab_data in mock_tabs:
        print(f"  - {tab_data['filename']}")
    
    return demo_dir

if __name__ == "__main__":
    create_mock_tab_data()