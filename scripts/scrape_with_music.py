"""
Script to scrape tabs and download corresponding music files.

This script combines tab scraping with music downloading to create
complete datasets with both tablature and audio.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.tab_scraper import TabScraper
from src.utils.music_downloader import MusicDownloader, install_ytdlp

logger = logging.getLogger(__name__)


def scrape_tabs_with_music(queries: List[str], output_dir: str = "data/complete_dataset", 
                          max_tabs_per_query: int = 10, download_music: bool = True) -> Dict:
    """
    Scrape tabs and download corresponding music.
    
    Args:
        queries: List of search queries (artist names, song names, etc.)
        output_dir: Directory to save the complete dataset
        max_tabs_per_query: Maximum tabs to scrape per query
        download_music: Whether to download music for each tab
        
    Returns:
        Dictionary with statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tabs_dir = output_path / "tabs"
    music_dir = output_path / "music"
    
    # Initialize scrapers
    tab_scraper = TabScraper(output_dir=str(tabs_dir))
    music_downloader = None
    
    if download_music:
        music_downloader = MusicDownloader(output_dir=str(music_dir))
        
        # Check if yt-dlp is available
        if not music_downloader.has_ytdlp:
            logger.info("Installing yt-dlp for music downloading...")
            if install_ytdlp():
                music_downloader = MusicDownloader(output_dir=str(music_dir))
            else:
                logger.error("Failed to install yt-dlp. Music downloading disabled.")
                download_music = False
    
    all_tabs = []
    stats = {
        'queries_processed': 0,
        'total_tabs_found': 0,
        'total_tabs_downloaded': 0,
        'total_music_downloaded': 0,
        'failed_downloads': 0,
        'complete_pairs': 0,  # Tabs with matching music
        'tab_only': 0,        # Tabs without music
    }
    
    # Process each query
    for query in queries:
        logger.info(f"Processing query: '{query}'")
        
        # Search and download tabs
        tab_results = tab_scraper.search_and_download(
            query=query,
            tab_types=['Tabs', 'Chords'],
            max_pages=3,
            max_downloads=max_tabs_per_query
        )
        
        stats['queries_processed'] += 1
        stats['total_tabs_found'] += tab_results['search_results']
        stats['total_tabs_downloaded'] += tab_results['downloaded']
        
        # Collect tab metadata for music downloading
        if download_music and tab_results['files']:
            # Extract metadata from downloaded tabs
            tab_metadata = []
            for tab_file in tab_results['files']:
                # Parse filename to extract artist and song
                filename = Path(tab_file).stem
                if ' - ' in filename:
                    parts = filename.split(' - ', 1)
                    artist = parts[0].strip()
                    song_part = parts[1].strip()
                    
                    # Remove version info like "(Ver 1)"
                    import re
                    song = re.sub(r'\s*\(Ver\s+\d+\)\s*', '', song_part).strip()
                    
                    tab_metadata.append({
                        'artist': artist,
                        'title': song,
                        'tab_file': tab_file,
                        'query': query
                    })
            
            all_tabs.extend(tab_metadata)
    
    # Download music for all tabs
    if download_music and all_tabs:
        logger.info(f"Downloading music for {len(all_tabs)} tabs...")
        
        music_results = music_downloader.batch_download_for_tabs(all_tabs)
        
        stats['total_music_downloaded'] = music_results['successful']
        stats['failed_downloads'] = music_results['failed']
        
        # Create paired dataset
        paired_data = []
        music_files = {f"{item['artist']} - {item['song']}": item['audio_path'] 
                      for item in music_results['downloaded_files']}
        
        for tab in all_tabs:
            key = f"{tab['artist']} - {tab['title']}"
            if key in music_files:
                paired_data.append({
                    'artist': tab['artist'],
                    'title': tab['title'],
                    'tab_file': tab['tab_file'],
                    'audio_file': music_files[key],
                    'query': tab['query']
                })
                stats['complete_pairs'] += 1
            else:
                stats['tab_only'] += 1
        
        # Save paired dataset index
        dataset_index = output_path / "dataset_index.json"
        with open(dataset_index, 'w') as f:
            json.dump({
                'metadata': {
                    'total_items': len(paired_data),
                    'queries': queries,
                    'created_at': str(pd.Timestamp.now() if 'pd' in globals() else 'unknown'),
                    'stats': stats
                },
                'items': paired_data
            }, f, indent=2)
        
        logger.info(f"Created dataset index: {dataset_index}")
    
    # Log final statistics
    logger.info("=" * 50)
    logger.info("SCRAPING AND DOWNLOAD COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Queries processed: {stats['queries_processed']}")
    logger.info(f"Tabs found: {stats['total_tabs_found']}")
    logger.info(f"Tabs downloaded: {stats['total_tabs_downloaded']}")
    
    if download_music:
        logger.info(f"Music downloaded: {stats['total_music_downloaded']}")
        logger.info(f"Complete pairs (tab + music): {stats['complete_pairs']}")
        logger.info(f"Tab-only files: {stats['tab_only']}")
        success_rate = stats['complete_pairs'] / stats['total_tabs_downloaded'] if stats['total_tabs_downloaded'] > 0 else 0
        logger.info(f"Pairing success rate: {success_rate:.1%}")
    
    return stats


def main():
    """Main function to parse arguments and run the scraper."""
    parser = argparse.ArgumentParser(description='Scrape tabs and download music')
    
    # Query arguments
    parser.add_argument('--queries', nargs='+', required=True,
                        help='Search queries (e.g., "metallica" "led zeppelin")')
    parser.add_argument('--output-dir', type=str, default='data/complete_dataset',
                        help='Output directory for the complete dataset')
    
    # Scraping options
    parser.add_argument('--max-tabs-per-query', type=int, default=10,
                        help='Maximum tabs to download per query')
    parser.add_argument('--no-music', action='store_true',
                        help='Skip music downloading (tabs only)')
    
    # Install options
    parser.add_argument('--install-ytdlp', action='store_true',
                        help='Install yt-dlp before running')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Install yt-dlp if requested
    if args.install_ytdlp:
        logger.info("Installing yt-dlp...")
        install_ytdlp()
    
    # Run the scraper
    stats = scrape_tabs_with_music(
        queries=args.queries,
        output_dir=args.output_dir,
        max_tabs_per_query=args.max_tabs_per_query,
        download_music=not args.no_music
    )
    
    print("\n" + "="*50)
    print("FINAL STATISTICS")
    print("="*50)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    main()