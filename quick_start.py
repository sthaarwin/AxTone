#!/usr/bin/env python3
"""
Quick start script for AxTone dataset generation.

This script provides easy-to-use commands for generating training datasets
using the integrated tab scraper.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

def quick_demo():
    """Generate a small demo dataset quickly."""
    print("🎸 Generating demo dataset for AxTone...")
    
    from scripts.generate_dataset import main as generate_main
    
    # Override sys.argv for demo
    original_argv = sys.argv
    sys.argv = [
        'quick_start.py',
        '--output-dir', 'data/demo_dataset',
        '--queries', 'metallica', 'led zeppelin', 'nirvana',
        '--max-per-query', '3',
        '--total-limit', '10',
        '--tab-types', 'Tabs',
        '--max-length', '512'
    ]
    
    try:
        generate_main()
        print("✅ Demo dataset generated successfully!")
        print("📁 Check data/demo_dataset/ for results")
    except Exception as e:
        print(f"❌ Error generating demo dataset: {e}")
    finally:
        sys.argv = original_argv

def full_dataset():
    """Generate a full training dataset."""
    print("🎸 Generating full training dataset for AxTone...")
    
    from scripts.generate_dataset import main as generate_main
    
    # Override sys.argv for full generation
    original_argv = sys.argv
    sys.argv = [
        'quick_start.py',
        '--output-dir', 'data/full_dataset',
        '--max-per-query', '20',
        '--total-limit', '1000',
        '--tab-types', 'Tabs', 'Chords',
        '--include-techniques'
    ]
    
    try:
        generate_main()
        print("✅ Full dataset generated successfully!")
        print("📁 Check data/full_dataset/ for results")
    except Exception as e:
        print(f"❌ Error generating full dataset: {e}")
    finally:
        sys.argv = original_argv

def validate_existing(dataset_path):
    """Validate an existing dataset."""
    print(f"🔍 Validating dataset at {dataset_path}...")
    
    try:
        from src.utils.scraper_utils import validate_dataset, create_dataset_summary
        
        # Validate the dataset
        validation_report = validate_dataset(dataset_path)
        
        if validation_report['valid']:
            print("✅ Dataset validation passed!")
        else:
            print("❌ Dataset validation failed:")
            for error in validation_report['errors']:
                print(f"  - {error}")
        
        if validation_report['warnings']:
            print("⚠️  Warnings:")
            for warning in validation_report['warnings']:
                print(f"  - {warning}")
        
        # Show summary
        summary = create_dataset_summary(dataset_path)
        print(f"\n📊 Dataset Summary:")
        print(f"  - Total files: {summary['total_files']}")
        print(f"  - Total size: {summary['total_size_mb']} MB")
        if 'splits' in summary:
            for split, count in summary['splits'].items():
                print(f"  - {split.capitalize()} files: {count}")
        
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies for tab scraping...")
    
    import subprocess
    import platform
    
    try:
        # Install Python requirements first
        print("Installing Python requirements...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements/base.txt'], check=True)
        
        # Install additional selenium requirements
        print("Installing selenium and web scraping dependencies...")
        packages = [
            'selenium>=4.0.0',
            'requests>=2.25.0', 
            'beautifulsoup4>=4.9.0',
            'webdriver-manager>=3.8.0'  # This will handle geckodriver automatically
        ]
        
        for package in packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
        
        # Install geckodriver using webdriver-manager (cross-platform)
        print("Setting up geckodriver...")
        try:
            from webdriver_manager.firefox import GeckoDriverManager
            driver_path = GeckoDriverManager().install()
            print(f"✅ Geckodriver installed at: {driver_path}")
        except Exception as e:
            print(f"⚠️  Could not auto-install geckodriver: {e}")
            print("Manual installation instructions:")
            if platform.system() == "Linux":
                if os.path.exists("/usr/bin/apt"):
                    print("  sudo apt update && sudo apt install firefox-geckodriver")
                elif os.path.exists("/usr/bin/pacman"):
                    print("  sudo pacman -S geckodriver")
                else:
                    print("  Download from: https://github.com/mozilla/geckodriver/releases")
            elif platform.system() == "Darwin":
                print("  brew install geckodriver")
            elif platform.system() == "Windows":
                print("  Download from: https://github.com/mozilla/geckodriver/releases")
                print("  Add to PATH or place in project directory")
        
        print("✅ Dependencies installed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("Try installing manually:")
        print("  pip install selenium requests beautifulsoup4 webdriver-manager")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function for quick start commands."""
    parser = argparse.ArgumentParser(description="Quick start for AxTone dataset generation")
    
    parser.add_argument(
        'command',
        choices=['demo', 'full', 'validate', 'install'],
        help='Command to run'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Path to dataset (for validate command)'
    )
    
    args = parser.parse_args()
    
    print("🎸 AxTone Dataset Generation Quick Start")
    print("=" * 50)
    
    if args.command == 'demo':
        quick_demo()
    elif args.command == 'full':
        full_dataset()
    elif args.command == 'validate':
        if not args.dataset_path:
            print("❌ Please provide --dataset-path for validation")
            return
        validate_existing(args.dataset_path)
    elif args.command == 'install':
        install_dependencies()
    
    print("\n🎸 Done!")

if __name__ == "__main__":
    main()