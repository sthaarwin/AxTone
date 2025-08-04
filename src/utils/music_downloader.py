"""
Music downloader module for pairing audio with scraped guitar tabs.

This module provides functionality to download audio from various sources
(primarily YouTube) to create complete tab+audio datasets.
"""

import os
import re
import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import subprocess
import requests

logger = logging.getLogger(__name__)

class MusicDownloader:
    """Downloads music to pair with guitar tabs."""
    
    def __init__(self, output_dir: str = "data/downloaded_music"):
        """
        Initialize the music downloader.
        
        Args:
            output_dir: Directory to save downloaded music
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if yt-dlp is available
        self.has_ytdlp = self._check_ytdlp()
        
        # Rate limiting for YouTube
        self.request_delay = 2  # seconds between downloads
        self.last_request_time = 0
    
    def _check_ytdlp(self) -> bool:
        """Check if yt-dlp is installed."""
        try:
            result = subprocess.run(['yt-dlp', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Found yt-dlp version: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            logger.warning("yt-dlp not found. Music downloading will be limited.")
        return False
    
    def search_youtube(self, artist: str, song: str, max_results: int = 5) -> List[Dict]:
        """
        Search YouTube for a song.
        
        Args:
            artist: Artist name
            song: Song title
            max_results: Maximum number of results to return
            
        Returns:
            List of video results with metadata
        """
        if not self.has_ytdlp:
            logger.error("yt-dlp not available for YouTube search")
            return []
        
        # Create search query
        query = f"{artist} {song}".replace(' ', '+')
        
        try:
            # Use yt-dlp to search YouTube
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                f'ytsearch{max_results}:{query}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"YouTube search failed: {result.stderr}")
                return []
            
            # Parse results
            results = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        video_info = json.loads(line)
                        results.append({
                            'id': video_info.get('id'),
                            'title': video_info.get('title'),
                            'duration': video_info.get('duration'),
                            'uploader': video_info.get('uploader'),
                            'view_count': video_info.get('view_count', 0),
                            'url': video_info.get('webpage_url'),
                            'description': video_info.get('description', '')
                        })
                    except json.JSONDecodeError:
                        continue
            
            # Filter and sort results to find best matches
            filtered_results = self._filter_music_results(results, artist, song)
            return filtered_results
            
        except subprocess.TimeoutExpired:
            logger.error("YouTube search timed out")
            return []
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
            return []
    
    def _filter_music_results(self, results: List[Dict], artist: str, song: str) -> List[Dict]:
        """Filter YouTube results to find actual music videos."""
        filtered = []
        
        # Keywords that suggest it's actual music
        good_keywords = ['official', 'music', 'video', 'audio', 'album', 'hd', 'hq']
        # Keywords that suggest it's not what we want
        bad_keywords = ['cover', 'tutorial', 'lesson', 'how to play', 'tab', 'guitar only', 
                       'instrumental', 'karaoke', 'live', 'concert', 'reaction']
        
        for result in results:
            title = result.get('title', '').lower()
            description = result.get('description', '').lower()
            uploader = result.get('uploader', '').lower()
            
            # Score the result
            score = 0
            
            # Check if artist and song are in title
            if artist.lower() in title:
                score += 3
            if song.lower() in title:
                score += 3
            
            # Boost official channels
            if 'official' in uploader or 'records' in uploader or 'music' in uploader:
                score += 2
            
            # Check for good keywords
            for keyword in good_keywords:
                if keyword in title:
                    score += 1
            
            # Penalize bad keywords
            for keyword in bad_keywords:
                if keyword in title:
                    score -= 2
            
            # Prefer reasonable durations (2-8 minutes for most songs)
            duration = result.get('duration', 0)
            if duration and 120 <= duration <= 480:  # 2-8 minutes
                score += 1
            elif duration and duration > 600:  # Very long videos
                score -= 1
            
            # Add score to result
            result['score'] = score
            
            # Only include results with positive scores
            if score > 0:
                filtered.append(result)
        
        # Sort by score (highest first)
        filtered.sort(key=lambda x: x['score'], reverse=True)
        
        return filtered
    
    def download_audio(self, video_url: str, artist: str, song: str, 
                      format_preference: str = 'mp3') -> Optional[str]:
        """
        Download audio from a video URL.
        
        Args:
            video_url: URL of the video to download
            artist: Artist name (for file naming)
            song: Song title (for file naming)
            format_preference: Preferred audio format
            
        Returns:
            Path to downloaded file, or None if failed
        """
        if not self.has_ytdlp:
            logger.error("yt-dlp not available for audio download")
            return None
        
        # Rate limiting
        self._rate_limit()
        
        # Create safe filename
        safe_artist = self._sanitize_filename(artist)
        safe_song = self._sanitize_filename(song)
        filename = f"{safe_artist} - {safe_song}"
        
        # Create artist directory
        artist_dir = self.output_dir / safe_artist
        artist_dir.mkdir(exist_ok=True)
        
        try:
            # Configure yt-dlp options
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', format_preference,
                '--audio-quality', '192K',
                '--output', str(artist_dir / f"{filename}.%(ext)s"),
                '--no-playlist',
                '--write-info-json',
                video_url
            ]
            
            logger.info(f"Downloading: {artist} - {song}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Find the downloaded file
                downloaded_files = list(artist_dir.glob(f"{filename}.*"))
                audio_files = [f for f in downloaded_files if f.suffix in ['.mp3', '.wav', '.m4a', '.ogg']]
                
                if audio_files:
                    audio_file = audio_files[0]
                    logger.info(f"Successfully downloaded: {audio_file.name}")
                    return str(audio_file)
                else:
                    logger.warning(f"No audio file found after download")
                    return None
            else:
                logger.error(f"Download failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Download timed out for {artist} - {song}")
            return None
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            return None
    
    def download_best_match(self, artist: str, song: str) -> Optional[str]:
        """
        Search for and download the best matching audio for a song.
        
        Args:
            artist: Artist name
            song: Song title
            
        Returns:
            Path to downloaded file, or None if failed
        """
        # Search YouTube
        results = self.search_youtube(artist, song, max_results=3)
        
        if not results:
            logger.warning(f"No YouTube results found for {artist} - {song}")
            return None
        
        # Try to download the best match
        for result in results:
            logger.info(f"Trying to download: {result['title']}")
            audio_path = self.download_audio(result['url'], artist, song)
            if audio_path:
                return audio_path
            
        logger.error(f"Failed to download any version of {artist} - {song}")
        return None
    
    def batch_download_for_tabs(self, tab_metadata: List[Dict]) -> Dict:
        """
        Download audio for a batch of tabs.
        
        Args:
            tab_metadata: List of tab metadata dicts with 'artist' and 'title' fields
            
        Returns:
            Dictionary with download statistics
        """
        total = len(tab_metadata)
        successful = 0
        failed = 0
        downloaded_files = []
        
        logger.info(f"Starting batch audio download for {total} tabs")
        
        for i, tab in enumerate(tab_metadata, 1):
            artist = tab.get('artist', 'Unknown')
            song = tab.get('title', 'Unknown')
            
            logger.info(f"[{i}/{total}] Downloading audio for: {artist} - {song}")
            
            audio_path = self.download_best_match(artist, song)
            if audio_path:
                successful += 1
                downloaded_files.append({
                    'artist': artist,
                    'song': song,
                    'audio_path': audio_path,
                    'tab_metadata': tab
                })
            else:
                failed += 1
            
            # Rate limiting between downloads
            if i < total:
                time.sleep(self.request_delay)
        
        stats = {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'downloaded_files': downloaded_files
        }
        
        logger.info(f"Batch download complete: {successful}/{total} successful ({stats['success_rate']:.1%})")
        return stats
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a string for use as a filename."""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'[\n\r\t]', ' ', filename)
        filename = re.sub(r'\s+', ' ', filename).strip()
        return filename[:100]  # Limit length
    
    def _rate_limit(self):
        """Implement rate limiting for downloads."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()


def install_ytdlp():
    """Install yt-dlp if not available."""
    try:
        import subprocess
        subprocess.run(['pip', 'install', 'yt-dlp'], check=True)
        logger.info("Successfully installed yt-dlp")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install yt-dlp: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    downloader = MusicDownloader()
    
    # Test download
    audio_path = downloader.download_best_match("Led Zeppelin", "Stairway to Heaven")
    if audio_path:
        print(f"Downloaded: {audio_path}")
    else:
        print("Download failed")