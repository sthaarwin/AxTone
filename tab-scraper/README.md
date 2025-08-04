# DEPRECATED: Legacy Tab Scraper

⚠️ **This directory contains the old PyQt5-based GUI tab scraper which has been deprecated.**

## Migration to New System

The tab scraping functionality has been completely rewritten and integrated into the main AxTone project with modern technologies:

### Old System (Deprecated)
- PyQt5-based GUI application
- Manual tab selection and download
- Limited error handling
- Selenium with manual geckodriver setup

### New System (Current)
- **Headless automation** with Selenium 4.0+
- **Batch processing** with configurable queries
- **Automatic dataset generation** pipeline
- **Tab processing** into PyTorch tensors
- **Dataset validation** and quality checks
- **Cross-platform** geckodriver management

## How to Use the New System

### Quick Start
```bash
# Install dependencies
python quick_start.py install

# Generate demo dataset
python quick_start.py demo

# Generate full dataset
python quick_start.py full

# Validate existing dataset
python quick_start.py validate --dataset-path data/my_dataset
```

### Advanced Usage
```bash
# Custom dataset generation
python scripts/generate_dataset.py \
    --output-dir data/my_dataset \
    --queries "metallica" "led zeppelin" \
    --max-per-query 20 \
    --tab-types Tabs Chords

# Process existing tabs
python -c "
from src.utils.tab_processor import TabProcessor
processor = TabProcessor('output_dir')
stats = processor.process_scraped_tabs('input_dir')
print(f'Processed {stats[\"processed\"]} tabs')
"
```

## New File Locations

| Component | New Location |
|-----------|-------------|
| Tab Scraper | `src/utils/tab_scraper.py` |
| Tab Processor | `src/utils/tab_processor.py` |
| Dataset Generator | `scripts/generate_dataset.py` |
| Quick Commands | `quick_start.py` |
| Configuration | `configs/tab_scraper_config.yaml` |

## Benefits of the New System

1. **No GUI Dependencies**: Runs headlessly in any environment
2. **Automated Pipeline**: From scraping to training-ready tensors
3. **Better Error Handling**: Robust retry logic and validation
4. **Scalable**: Process thousands of tabs automatically
5. **Integrated**: Works seamlessly with AxTone training pipeline
6. **Modern**: Uses latest Selenium and webdriver-manager

## Migration Complete

All functionality from this legacy scraper has been reimplemented in the new system with significant improvements. This directory is kept for reference only.

**Use the new system in the main project root for all tab scraping needs.**