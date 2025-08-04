"""
Modern tab scraper module for collecting guitar tablature data.

This module provides tools for scraping guitar tabs from Ultimate Guitar
and processing them for use in training datasets.
"""

import os
import re
import json
import requests
import logging
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TabResult:
    """Represents a single tab search result."""
    tab_type: str
    artist: str
    title: str
    rating: float
    votes: int
    url: str
    version: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'type': self.tab_type,
            'artist': self.artist,
            'title': self.title,
            'rating': self.rating,
            'votes': self.votes,
            'url': self.url,
            'version': self.version
        }

class TabScraper:
    """Modern tab scraper with improved error handling and dataset integration."""
    
    def __init__(self, output_dir: str = "data/scraped_tabs", headless: bool = True):
        """
        Initialize the tab scraper.
        
        Args:
            output_dir: Directory to save scraped tabs
            headless: Whether to run browser in headless mode
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        
        # Create subdirectories for different tab types
        self.tab_dirs = {
            'Chords': self.output_dir / 'chords',
            'Tabs': self.output_dir / 'tabs', 
            'Pro': self.output_dir / 'pro',
            'Power': self.output_dir / 'power',
            'Bass Tabs': self.output_dir / 'bass',
            'Ukulele Chords': self.output_dir / 'ukulele'
        }
        
        for tab_dir in self.tab_dirs.values():
            tab_dir.mkdir(parents=True, exist_ok=True)
        
        # Search URL pattern
        self.search_url = "https://www.ultimate-guitar.com/search.php?page={}&search_type=title&value={}"
        
        # Regex patterns for parsing search results
        self.results_pattern = r'"results":(\[.*?\]),"pagination"'
        self.results_count_pattern = r'"tabs","results_count":([0-9]+?),"results"'
        
        # Download timeout
        self.download_timeout = 15
        
        # Rate limiting
        self.request_delay = 1  # seconds between requests
        self.last_request_time = 0
        
    def search_tabs(self, query: str, tab_types: List[str] = None, max_pages: int = 5) -> List[TabResult]:
        """
        Search for tabs on Ultimate Guitar.
        
        Args:
            query: Search query string
            tab_types: List of tab types to search for (e.g., ['Tabs', 'Chords'])
            max_pages: Maximum number of pages to search
            
        Returns:
            List of TabResult objects
        """
        if tab_types is None:
            tab_types = ['Tabs', 'Chords']
            
        # URL encode the search query
        search_string = "%20".join(query.split())
        results = []
        
        logger.info(f"Searching for '{query}' with types: {tab_types}")
        
        page = 1
        total_results = 0
        
        while page <= max_pages:
            try:
                # Rate limiting
                self._rate_limit()
                
                # Make request with proper headers to avoid blocking
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                
                url = self.search_url.format(page, search_string)
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                # Parse response
                response_body = response.content.decode()
                
                # Try multiple patterns for extracting results
                results_json = None
                
                # Pattern 1: Original JSON results pattern
                results_match = re.search(self.results_pattern, response_body)
                if results_match:
                    try:
                        results_json = json.loads(results_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                # Pattern 2: Look for window.UGAPP or similar JavaScript variables
                if not results_json:
                    js_patterns = [
                        r'window\.UGAPP\s*=\s*({.*?});',
                        r'window\.store\s*=\s*({.*?});',
                        r'data-content="({.*?})"',
                    ]
                    
                    for pattern in js_patterns:
                        matches = re.findall(pattern, response_body, re.DOTALL)
                        for match in matches:
                            try:
                                data = json.loads(match)
                                if 'results' in data:
                                    results_json = data['results']
                                    break
                                elif 'tabs' in data and 'results' in data['tabs']:
                                    results_json = data['tabs']['results']
                                    break
                            except json.JSONDecodeError:
                                continue
                        if results_json:
                            break
                
                # Pattern 3: Fallback to HTML parsing with BeautifulSoup
                if not results_json:
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response_body, 'html.parser')
                        
                        # Look for tab links in the HTML
                        tab_links = soup.find_all('a', href=re.compile(r'/tab/'))
                        results_json = []
                        
                        for link in tab_links[:20]:  # Limit to first 20 results per page
                            href = link.get('href', '')
                            text = link.get_text(strip=True)
                            
                            # Try to extract artist and song name
                            if '/tab/' in href:
                                # URL format: /tab/artist/song_name_123456
                                parts = href.split('/')
                                if len(parts) >= 4:
                                    artist = parts[2].replace('-', ' ').replace('_', ' ')
                                    song_parts = parts[3].split('_')
                                    song = ' '.join(song_parts[:-1]).replace('-', ' ')
                                    
                                    # Determine tab type from URL or text
                                    tab_type = 'Tabs'  # Default
                                    if 'chord' in href.lower() or 'chord' in text.lower():
                                        tab_type = 'Chords'
                                    elif 'bass' in href.lower():
                                        tab_type = 'Bass Tabs'
                                    
                                    results_json.append({
                                        'type': tab_type,
                                        'artist_name': artist.title(),
                                        'song_name': song.title(),
                                        'rating': 4.0,  # Default rating
                                        'votes': 100,   # Default votes
                                        'tab_url': f"https://www.ultimate-guitar.com{href}",
                                        'version': '1'   # Default version
                                    })
                    
                    except ImportError:
                        logger.warning("BeautifulSoup not available for HTML parsing fallback")
                    except Exception as e:
                        logger.debug(f"HTML parsing fallback failed: {e}")
                
                if not results_json:
                    logger.warning(f"No results found on page {page} using any method")
                    break
                
                # Get total count on first page
                if page == 1 and isinstance(results_json, list):
                    total_results = len(results_json) * max_pages  # Estimate
                    logger.info(f"Found approximately {total_results} total results")
                
                page_results = 0
                for item in results_json:
                    try:
                        # Filter by desired types
                        item_type = item.get("type", "Tabs")
                        if item_type in tab_types:
                            result = TabResult(
                                tab_type=item_type,
                                artist=item.get("artist_name", "Unknown"),
                                title=item.get("song_name", "Unknown"),
                                rating=float(item.get("rating", 4.0)),
                                votes=int(item.get("votes", 100)),
                                url=item.get("tab_url", ""),
                                version=str(item.get("version", "1"))
                            )
                            results.append(result)
                            page_results += 1
                    except (KeyError, ValueError) as e:
                        logger.debug(f"Skipping result due to error: {e}")
                        continue
                
                logger.info(f"Page {page}: found {page_results} relevant results")
                
                if page_results == 0:
                    break
                    
                page += 1
                
            except requests.RequestException as e:
                logger.error(f"Request failed on page {page}: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error on page {page}: {e}")
                break
        
        logger.info(f"Search complete: found {len(results)} tabs")
        return results
    
    def download_tab(self, tab_result: TabResult, save_screenshot: bool = True) -> Optional[str]:
        """
        Download a single tab.
        
        Args:
            tab_result: TabResult object to download
            save_screenshot: Whether to save a screenshot of the tab
            
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            # Determine if this is a file download or text tab
            is_file = tab_result.tab_type in ["Pro", "Power"]
            
            if is_file:
                return self._download_file(tab_result)
            else:
                return self._download_text_tab(tab_result, save_screenshot)
                
        except Exception as e:
            logger.error(f"Failed to download tab {tab_result.title} by {tab_result.artist}: {e}")
            return None
    
    def _download_text_tab(self, tab_result: TabResult, save_screenshot: bool = True) -> Optional[str]:
        """Download a text-based tab (Tabs, Chords, etc.)."""
        driver = None
        try:
            # Setup Firefox driver
            driver = self._create_driver()
            driver.get(tab_result.url)
            
            # Wait for page to load and handle popups
            self._handle_popups(driver)
            
            # Find the tab content
            tab_element = None
            try:
                # Try different selectors for the tab content
                selectors = [
                    "pre[class*='js-tab-content']",
                    "pre",
                    "[data-name='tab-content']",
                    ".js-tab-content"
                ]
                
                for selector in selectors:
                    try:
                        tab_element = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        break
                    except TimeoutException:
                        continue
                        
                if not tab_element:
                    logger.warning(f"Could not find tab content for {tab_result.title}")
                    return None
                    
            except Exception as e:
                logger.warning(f"Error finding tab content: {e}")
                return None
            
            # Extract tab text
            tab_text = tab_element.text
            
            # Create filename
            safe_artist = self._sanitize_filename(tab_result.artist)
            safe_title = self._sanitize_filename(tab_result.title)
            filename = f"{safe_artist} - {safe_title} (Ver {tab_result.version})"
            
            # Save text tab
            tab_dir = self.tab_dirs[tab_result.tab_type]
            tab_file = tab_dir / f"{filename}.tab"
            
            with open(tab_file, 'w', encoding='utf-8') as f:
                f.write(tab_text)
            
            # Save screenshot if requested
            if save_screenshot:
                screenshot_file = tab_dir / f"{filename}.png"
                tab_element.screenshot(str(screenshot_file))
            
            # Save metadata
            metadata = {
                'artist': tab_result.artist,
                'title': tab_result.title,
                'version': tab_result.version,
                'rating': tab_result.rating,
                'votes': tab_result.votes,
                'url': tab_result.url,
                'type': tab_result.tab_type,
                'downloaded_at': time.time()
            }
            
            metadata_file = tab_dir / f"{filename}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Downloaded tab: {tab_result.artist} - {tab_result.title}")
            return str(tab_file)
            
        except Exception as e:
            logger.error(f"Error downloading text tab: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def _download_file(self, tab_result: TabResult) -> Optional[str]:
        """Download a binary file tab (Guitar Pro, PowerTab)."""
        driver = None
        try:
            # Create download directory
            tab_dir = self.tab_dirs[tab_result.tab_type]
            
            # Setup Firefox with download preferences
            driver = self._create_driver(download_dir=str(tab_dir))
            driver.get(tab_result.url)
            
            # Handle popups
            self._handle_popups(driver)
            
            # Find and click download button
            download_selectors = [
                '//button/span[text()="DOWNLOAD Guitar Pro TAB"]',
                '//button/span[text()="DOWNLOAD Power TAB"]',
                '//button[contains(text(), "DOWNLOAD")]',
                '.js-download-button'
            ]
            
            download_button = None
            for selector in download_selectors:
                try:
                    if selector.startswith('//'):
                        download_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        download_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                    break
                except TimeoutException:
                    continue
            
            if not download_button:
                logger.warning(f"Could not find download button for {tab_result.title}")
                return None
            
            # Count files before download
            files_before = list(tab_dir.iterdir())
            
            # Click download button
            driver.execute_script("arguments[0].click();", download_button)
            
            # Wait for download to complete
            timeout = self.download_timeout
            while timeout > 0:
                time.sleep(0.5)
                files_after = list(tab_dir.iterdir())
                if len(files_after) > len(files_before):
                    # File was downloaded
                    new_files = [f for f in files_after if f not in files_before]
                    if new_files:
                        downloaded_file = new_files[0]
                        logger.info(f"Downloaded file: {downloaded_file.name}")
                        return str(downloaded_file)
                timeout -= 0.5
            
            logger.warning(f"Download timeout for {tab_result.title}")
            return None
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def _create_driver(self, download_dir: str = None) -> webdriver.Firefox:
        """Create a configured Firefox WebDriver."""
        options = Options()
        if self.headless:
            options.headless = True
        
        # Create Firefox profile if download directory is specified
        profile = None
        if download_dir:
            profile = FirefoxProfile()
            profile.set_preference("browser.download.folderList", 2)
            profile.set_preference("browser.download.manager.showWhenStarting", False)
            profile.set_preference("browser.download.dir", download_dir)
            profile.set_preference("browser.helperApps.neverAsk.saveToDisk", 
                                 "application/octet-stream,application/gp5,application/gpx")
        
        # Use webdriver-manager to handle geckodriver automatically
        try:
            from webdriver_manager.firefox import GeckoDriverManager
            driver_path = GeckoDriverManager().install()
            logger.debug(f"Using geckodriver at: {driver_path}")
        except ImportError:
            logger.warning("webdriver-manager not available, using system geckodriver")
            driver_path = None
        except Exception as e:
            logger.warning(f"Failed to auto-install geckodriver: {e}")
            driver_path = None
        
        # Create driver
        if driver_path:
            driver = webdriver.Firefox(
                service=webdriver.firefox.service.Service(driver_path),
                options=options,
                firefox_profile=profile
            )
        else:
            # Fallback to system geckodriver
            driver = webdriver.Firefox(
                options=options,
                firefox_profile=profile
            )
        
        return driver
    
    def _handle_popups(self, driver):
        """Handle common popups on Ultimate Guitar."""
        # Wait a moment for page to load
        time.sleep(2)
        
        # Handle privacy policy popup
        try:
            popup_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//button[text()="Got it, thanks!"]'))
            )
            popup_btn.click()
        except TimeoutException:
            pass
        
        # Handle official tab ad
        try:
            popup_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '.ai-ah button'))
            )
            popup_btn.click()
        except TimeoutException:
            pass
        
        # Hide autoscroll tool
        try:
            autoscroll = driver.find_element(By.XPATH, 
                '//span[text()="Autoscroll"]/parent::button/parent::div/parent::section')
            driver.execute_script("arguments[0].setAttribute('style', 'display: none')", autoscroll)
        except NoSuchElementException:
            pass
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a string for use as a filename."""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'[\n\r\t]', ' ', filename)
        filename = re.sub(r'\s+', ' ', filename).strip()
        return filename[:100]  # Limit length
    
    def _rate_limit(self):
        """Implement rate limiting for requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def batch_download(self, results: List[TabResult], max_downloads: int = None) -> List[str]:
        """
        Download multiple tabs in batch.
        
        Args:
            results: List of TabResult objects to download
            max_downloads: Maximum number of downloads (None for all)
            
        Returns:
            List of successfully downloaded file paths
        """
        if max_downloads:
            results = results[:max_downloads]
        
        downloaded_files = []
        total = len(results)
        
        logger.info(f"Starting batch download of {total} tabs")
        
        for i, result in enumerate(results, 1):
            logger.info(f"Downloading {i}/{total}: {result.artist} - {result.title}")
            
            file_path = self.download_tab(result)
            if file_path:
                downloaded_files.append(file_path)
            
            # Rate limiting between downloads
            if i < total:
                time.sleep(self.request_delay)
        
        logger.info(f"Batch download complete: {len(downloaded_files)}/{total} successful")
        return downloaded_files
    
    def search_and_download(self, query: str, tab_types: List[str] = None, 
                          max_pages: int = 3, max_downloads: int = 50) -> Dict:
        """
        Search for tabs and download them in one operation.
        
        Args:
            query: Search query
            tab_types: Types of tabs to search for
            max_pages: Maximum pages to search
            max_downloads: Maximum number of tabs to download
            
        Returns:
            Dictionary with search and download statistics
        """
        # Search for tabs
        results = self.search_tabs(query, tab_types, max_pages)
        
        if not results:
            return {
                'query': query,
                'search_results': 0,
                'downloaded': 0,
                'files': []
            }
        
        # Download tabs
        downloaded_files = self.batch_download(results, max_downloads)
        
        return {
            'query': query,
            'search_results': len(results),
            'downloaded': len(downloaded_files),
            'files': downloaded_files
        }


def create_dataset_from_queries(queries: List[str], output_dir: str = "data/scraped_dataset", 
                               tab_types: List[str] = None, max_per_query: int = 20) -> Dict:
    """
    Create a dataset by scraping tabs for multiple queries.
    
    Args:
        queries: List of search queries (e.g., artist names, song names)
        output_dir: Directory to save the dataset
        tab_types: Types of tabs to scrape
        max_per_query: Maximum downloads per query
        
    Returns:
        Dictionary with dataset statistics
    """
   