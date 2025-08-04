#!/usr/bin/env python3
"""
Dataset generation script for AxTone.

This script uses the integrated tab scraper to collect guitar tabs from Ultimate Guitar
and processes them into training datasets for the AxTone model.
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict

# Import our custom modules
from src.utils.tab_scraper import TabScraper, create_dataset_from_queries
from src.utils.tab_processor import TabProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_generation.log')
    ]
)
logger = logging.getLogger(__name__)

# Popular artists/songs for dataset generation
POPULAR_QUERIES = [
    # Classic Rock
    "led zeppelin", "pink floyd", "the beatles", "queen", "deep purple",
    "black sabbath", "ac dc", "guns n roses", "metallica", "iron maiden",
    
    # Modern Rock/Alternative
    "foo fighters", "radiohead", "pearl jam", "nirvana", "alice in chains",
    "soundgarden", "stone temple pilots", "red hot chili peppers", "linkin park",
    
    # Blues/Classic
    "stevie ray vaughan", "eric clapton", "bb king", "jimi hendrix", "cream",
    "allman brothers", "lynyrd skynyrd", "eagles", "fleetwood mac",
    
    # Popular Songs
    "stairway to heaven", "sweet child o mine", "comfortably numb", "hotel california",
    "smoke on the water", "paranoid", "enter sandman", "master of puppets",
    "thunderstruck", "back in black", "welcome to the jungle", "november rain",
    
    # Different Genres
    "john mayer", "dave matthews band", "green day", "blink 182", "sum 41",
    "the offspring", "system of a down", "rage against the machine", "audioslave",
    
    # Beginner-friendly
    "wonderwall oasis", "wish you were here", "good riddance", "boulevard of broken dreams",
    "zombie cranberries", "creep radiohead", "everlong foo fighters"
]

# Guitar techniques and styles for diversity
TECHNIQUE_QUERIES = [
    "fingerpicking", "classical guitar", "flamenco guitar", "jazz guitar",
    "blues guitar", "metal guitar", "acoustic guitar", "electric guitar",
    "guitar solo", "guitar riff", "chord progression", "strumming pattern"
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate training dataset for AxTone")
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/generated_dataset',
        help='Output directory for the dataset'
    )
    
    parser.add_argument(
        '--queries', 
        type=str, 
        nargs='+',
        default=None,
        help='Custom search queries (overrides default list)'
    )
    
    parser.add_argument(
        '--max-per-query', 
        type=int, 
        default=10,
        help='Maximum tabs to download per query'
    )
    
    parser.add_argument(
        '--total-limit', 
        type=int, 
        default=500,
        help='Total maximum number of tabs to download'
    )
    
    parser.add_argument(
        '--tab-types', 
        type=str, 
        nargs='+',
        default=['Tabs', 'Chords'],
        choices=['Tabs', 'Chords', 'Pro', 'Power', 'Bass Tabs', 'Ukulele Chords'],
        help='Types of tabs to scrape'
    )
    
    parser.add_argument(
        '--headless', 
        action='store_true',
        default=True,
        help='Run browser in headless mode'
    )
    
    parser.add_argument(
        '--no-headless', 
        action='store_false',
        dest='headless',
        help='Run browser with GUI (for debugging)'
    )
    
    parser.add_argument(
        '--skip-scraping', 
        action='store_true',
        help='Skip scraping phase and only process existing tabs'
    )
    
    parser.add_argument(
        '--skip-processing', 
        action='store_true',
        help='Skip processing phase and only scrape tabs'
    )
    
    parser.add_argument(
        '--max-length', 
        type=int, 
        default=1024,
        help='Maximum sequence length for processed tensors'
    )
    
    parser.add_argument(
        '--include-techniques', 
        action='store_true',
        help='Include technique-based queries for more diverse dataset'
    )
    
    return parser.parse_args()

def load_existing_metadata(output_dir: Path) -> Dict:
    """Load existing dataset metadata if it exists."""
    metadata_file = output_dir / 'dataset_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

def save_generation_config(output_dir: Path, args):
    """Save the configuration used for dataset generation."""
    config = {
        'output_dir': str(args.output_dir),
        'queries': args.queries,
        'max_per_query': args.max_per_query,
        'total_limit': args.total_limit,
        'tab_types': args.tab_types,
        'headless': args.headless,
        'max_length': args.max_length,
        'include_techniques': args.include_techniques,
        'timestamp': time.time(),
        'timestamp_human': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_file = output_dir / 'generation_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved generation config to {config_file}")

def scrape_dataset(args) -> Dict:
    """Scrape tabs using the configured parameters."""
    output_dir = Path(args.output_dir)
    scraped_dir = output_dir / 'scraped_tabs'
    
    # Determine queries to use
    if args.queries:
        queries = args.queries
    else:
        queries = POPULAR_QUERIES.copy()
        if args.include_techniques:
            queries.extend(TECHNIQUE_QUERIES)
    
    # Limit total queries based on total_limit and max_per_query
    max_queries = args.total_limit // args.max_per_query
    if len(queries) > max_queries:
        queries = queries[:max_queries]
        logger.info(f"Limited to {max_queries} queries to stay within total limit of {args.total_limit}")
    
    logger.info(f"Starting dataset scraping with {len(queries)} queries")
    logger.info(f"Tab types: {args.tab_types}")
    logger.info(f"Max per query: {args.max_per_query}")
    logger.info(f"Output directory: {scraped_dir}")
    
    # Create the dataset
    metadata = create_dataset_from_queries(
        queries=queries,
        output_dir=str(scraped_dir),
        tab_types=args.tab_types,
        max_per_query=args.max_per_query
    )
    
    # Update metadata with additional info
    metadata.update({
        'generation_args': vars(args),
        'total_queries': len(queries),
        'queries_used': queries
    })
    
    # Save updated metadata
    metadata_file = output_dir / 'scraping_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Scraping complete: {metadata['total_downloaded']} tabs downloaded")
    return metadata

def process_dataset(args, scraped_dir: Path = None) -> Dict:
    """Process scraped tabs into training format."""
    output_dir = Path(args.output_dir)
    
    if scraped_dir is None:
        scraped_dir = output_dir / 'scraped_tabs'
    
    processed_dir = output_dir / 'processed_tabs'
    
    logger.info(f"Starting tab processing")
    logger.info(f"Input directory: {scraped_dir}")
    logger.info(f"Output directory: {processed_dir}")
    logger.info(f"Max sequence length: {args.max_length}")
    
    # Create processor and process tabs
    processor = TabProcessor(str(processed_dir))
    
    # Process scraped tabs
    processing_stats = processor.process_scraped_tabs(
        str(scraped_dir), 
        max_length=args.max_length
    )
    
    # Create training splits
    split_info = processor.create_training_dataset()
    
    # Combine statistics
    result = {
        'processing_stats': processing_stats,
        'split_info': split_info,
        'processed_dir': str(processed_dir)
    }
    
    # Save processing metadata
    processing_metadata_file = output_dir / 'processing_metadata.json'
    with open(processing_metadata_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Processing complete: {processing_stats['processed']} tabs processed")
    logger.info(f"Dataset splits: {split_info['train_files']} train, {split_info['val_files']} val, {split_info['test_files']} test")
    
    return result

def generate_dataset_report(output_dir: Path):
    """Generate a comprehensive report of the dataset."""
    report_data = {}
    
    # Load all metadata files
    metadata_files = {
        'scraping': 'scraping_metadata.json',
        'processing': 'processing_metadata.json',
        'config': 'generation_config.json'
    }
    
    for key, filename in metadata_files.items():
        file_path = output_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                report_data[key] = json.load(f)
    
    # Generate summary statistics
    summary = {
        'dataset_creation_time': report_data.get('config', {}).get('timestamp_human', 'Unknown'),
        'total_scraped': report_data.get('scraping', {}).get('total_downloaded', 0),
        'total_processed': report_data.get('processing', {}).get('processing_stats', {}).get('processed', 0),
        'total_failed': report_data.get('processing', {}).get('processing_stats', {}).get('failed', 0),
        'train_files': report_data.get('processing', {}).get('split_info', {}).get('train_files', 0),
        'val_files': report_data.get('processing', {}).get('split_info', {}).get('val_files', 0),
        'test_files': report_data.get('processing', {}).get('split_info', {}).get('test_files', 0),
        'avg_sequence_length': report_data.get('processing', {}).get('processing_stats', {}).get('avg_sequence_length', 0),
        'max_sequence_length': report_data.get('processing', {}).get('processing_stats', {}).get('max_length', 0),
    }
    
    # Save report
    report = {
        'summary': summary,
        'detailed_metadata': report_data,
        'report_generated': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    report_file = output_dir / 'dataset_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary to console
    print("\n" + "="*60)
    print("DATASET GENERATION REPORT")
    print("="*60)
    print(f"Dataset created: {summary['dataset_creation_time']}")
    print(f"Total tabs scraped: {summary['total_scraped']}")
    print(f"Total tabs processed: {summary['total_processed']}")
    print(f"Processing failed: {summary['total_failed']}")
    print(f"Training files: {summary['train_files']}")
    print(f"Validation files: {summary['val_files']}")
    print(f"Test files: {summary['test_files']}")
    print(f"Average sequence length: {summary['avg_sequence_length']:.1f}")
    print(f"Max sequence length: {summary['max_sequence_length']}")
    print(f"\nDetailed report saved to: {report_file}")
    print("="*60)
    
    return report

def main():
    """Main function for dataset generation."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save generation configuration
    save_generation_config(output_dir, args)
    
    # Check if we should skip phases
    scraped_dir = output_dir / 'scraped_tabs'
    
    # Phase 1: Scraping
    if not args.skip_scraping:
        logger.info("=== PHASE 1: SCRAPING TABS ===")
        try:
            scraping_metadata = scrape_dataset(args)
            if scraping_metadata['total_downloaded'] == 0:
                logger.warning("No tabs were downloaded. Check your internet connection and queries.")
                return
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return
    else:
        logger.info("Skipping scraping phase")
        if not scraped_dir.exists():
            logger.error(f"Scraped directory {scraped_dir} does not exist. Cannot skip scraping.")
            return
    
    # Phase 2: Processing
    if not args.skip_processing:
        logger.info("=== PHASE 2: PROCESSING TABS ===")
        try:
            processing_metadata = process_dataset(args, scraped_dir)
            if processing_metadata['processing_stats']['processed'] == 0:
                logger.warning("No tabs were successfully processed.")
                return
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return
    else:
        logger.info("Skipping processing phase")
    
    # Phase 3: Generate report
    logger.info("=== PHASE 3: GENERATING REPORT ===")
    try:
        report = generate_dataset_report(output_dir)
        logger.info("Dataset generation completed successfully!")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")

if __name__ == "__main__":
    main()