"""
DEPRECATED: Legacy PyQt5-based tab scraper GUI.

This file has been replaced by the modern Selenium-based tab scraper
located in src/utils/tab_scraper.py. 

The new system provides:
- Headless browser automation with Selenium
- Better error handling and rate limiting
- Integration with the AxTone dataset generation pipeline
- No GUI dependencies

To use the new tab scraper, see:
- src/utils/tab_scraper.py (main scraper)
- src/utils/tab_processor.py (tab processing)
- scripts/generate_dataset.py (dataset generation)
- quick_start.py (easy commands)

Usage:
  python quick_start.py demo
  python quick_start.py full
  python scripts/generate_dataset.py --help
"""

# Legacy imports - no longer needed
# from PyQt5 import QtCore, QtGui, QtWidgets
# from src import utils
import sys
import os
import warnings

def main():
    """Show deprecation message and redirect to new system."""
    warnings.warn(
        "This PyQt5-based tab scraper is deprecated. "
        "Please use the new Selenium-based system in src/utils/tab_scraper.py",
        DeprecationWarning,
        stacklevel=2
    )
    
    print("🎸 AxTone Tab Scraper")
    print("=" * 50)
    print("⚠️  This PyQt5-based GUI scraper is deprecated.")
    print("✨ Use the new integrated system instead:")
    print()
    print("Quick start commands:")
    print("  python quick_start.py demo      # Generate demo dataset")
    print("  python quick_start.py full      # Generate full dataset")
    print("  python quick_start.py install   # Install dependencies")
    print()
    print("Advanced usage:")
    print("  python scripts/generate_dataset.py --help")
    print()
    print("📁 New scraper location: src/utils/tab_scraper.py")
    print("📁 Processing pipeline: src/utils/tab_processor.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
