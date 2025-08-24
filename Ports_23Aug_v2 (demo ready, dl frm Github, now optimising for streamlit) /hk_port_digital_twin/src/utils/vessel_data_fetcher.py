# Comments for context:
# This module implements the real-time vessel data pipeline for the Hong Kong Port Digital Twin project.
# It downloads vessel arrival and departure XML files from the Hong Kong Marine Department's open data portal
# every 20 minutes and on application startup. The module replaces synthetic test data with live vessel information.
# 
# Problem: The current dashboard uses static synthetic data from a single XML file (Arrived_in_last_36_hours.xml)
# Solution: Automated pipeline that fetches 4 real-time XML files from the government data portal
# Approach: Modular design with robust error handling, atomic file operations, and comprehensive logging

import os
import time
import shutil
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VesselDataFetcher:
    """
    Handles downloading and managing vessel data XML files from Hong Kong Marine Department.
    
    This class provides functionality to:
    - Download multiple XML files from the government data portal
    - Validate XML content before replacing existing files
    - Create timestamped backups of previous data
    - Track update times and file metadata
    - Handle errors gracefully without breaking the application
    """
    
    def __init__(self):
        """
        Initialize the vessel data fetcher with configuration from environment variables.
        """
        # Configuration from environment variables
        self.base_url = os.getenv('HK_VESSEL_DATA_BASE_URL', 'https://data.gov.hk/en-data/dataset/hk-md-mardep-vessel-arrivals-and-departures')
        # Set data directory to the raw_data folder as requested
        self.data_directory = Path('/Users/Bhavesh/Documents/GitHub/Ports/Ports/raw_data')
        self.fetch_interval = int(os.getenv('VESSEL_DATA_FETCH_INTERVAL', '20'))
        self.pipeline_enabled = os.getenv('VESSEL_DATA_PIPELINE_ENABLED', 'true').lower() == 'true'
        
        # Create necessary directories
        self.backup_directory = self.data_directory / 'vessel_data' / 'backups'
        self.logs_directory = self.data_directory / 'vessel_data' / 'logs'
        self.temp_directory = self.data_directory / 'vessel_data' / 'temp'
        
        # Create directories if they don't exist
        for directory in [self.backup_directory, self.logs_directory, self.temp_directory]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # HTTP session configuration
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HK-Port-Digital-Twin/1.0 (Educational Project)'
        })
        
        # XML file names that we expect to download
        # Note: These are placeholder names - actual file names will be discovered from the portal
        self.expected_files = [
            'Arrived_in_last_36_hours.xml',
            'Departed_in_last_36_hours.xml',
            'Expected_arrivals.xml',
            'Expected_departures.xml'
        ]
        
        self.logger.info(f"VesselDataFetcher initialized. Pipeline enabled: {self.pipeline_enabled}")
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup dedicated logging for the vessel data pipeline.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger('vessel_data_pipeline')
        logger.setLevel(logging.INFO)
        
        # Create file handler for pipeline logs
        log_file = self.logs_directory / f'vessel_pipeline_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger (avoid duplicate handlers)
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger
    
    def fetch_xml_files(self) -> Dict[str, bool]:
        """
        Download all vessel data XML files from the government portal.
        
        This method implements the core downloading logic with retry mechanisms,
        timeout handling, and validation.
        
        Returns:
            Dict[str, bool]: Dictionary mapping file names to download success status
        """
        if not self.pipeline_enabled:
            self.logger.info("Vessel data pipeline is disabled. Skipping fetch.")
            return {}
        
        self.logger.info("Starting vessel data fetch operation")
        results = {}
        
        try:
            # First, discover available XML files from the portal
            available_files = self._discover_xml_files()
            
            if not available_files:
                self.logger.warning("No XML files discovered from the portal")
                return results
            
            # Download each discovered file
            for file_info in available_files:
                file_name = file_info['name']
                file_url = file_info['url']
                
                success = self._download_single_file(file_name, file_url)
                results[file_name] = success
                
                # Add delay between downloads to be respectful to the server
                if len(available_files) > 1:
                    time.sleep(2)
            
            self.logger.info(f"Fetch operation completed. Results: {results}")
            
        except Exception as e:
            self.logger.error(f"Error during fetch operation: {str(e)}")
            
        return results
    
    def _discover_xml_files(self) -> List[Dict[str, str]]:
        """
        Discover available XML files from the government data portal.
        
        This method uses known Hong Kong Marine Department data sources to construct
        download URLs for vessel data XML files.
        
        Returns:
            List[Dict[str, str]]: List of dictionaries with 'name' and 'url' keys
        """
        self.logger.info("Discovering XML files from Hong Kong Marine Department portal")
        
        # Hong Kong Marine Department vessel data URLs via government API
        # These are the official API endpoints for real-time vessel data
        xml_files = [
            {
                'name': 'Arrived_in_last_36_hours.xml',
                'url': 'https://res.data.gov.hk/api/get-download-file?name=https%3A%2F%2Fwww.mardep.gov.hk%2Fe_files%2Fen%2Fopendata%2FRP05005i.XML'
            },
            {
                'name': 'Departed_in_last_36_hours.xml', 
                'url': 'https://res.data.gov.hk/api/get-download-file?name=https%3A%2F%2Fwww.mardep.gov.hk%2Fe_files%2Fen%2Fopendata%2FRP04005i.XML'
            },
            {
                'name': 'Expected_arrivals.xml',
                'url': 'https://res.data.gov.hk/api/get-download-file?name=https%3A%2F%2Fwww.mardep.gov.hk%2Fe_files%2Fen%2Fopendata%2FRP06005i.XML'
            },
            {
                'name': 'Expected_departures.xml',
                'url': 'https://res.data.gov.hk/api/get-download-file?name=https%3A%2F%2Fwww.mardep.gov.hk%2Fe_files%2Fen%2Fopendata%2FRP05505i.XML'
            }
        ]
        
        # Verify URLs are accessible (basic connectivity check)
        accessible_files = []
        for file_info in xml_files:
            try:
                # Quick HEAD request to check if URL is accessible
                response = self.session.head(file_info['url'], timeout=10)
                if response.status_code == 200:
                    accessible_files.append(file_info)
                    self.logger.info(f"Verified access to {file_info['name']}")
                else:
                    self.logger.warning(f"URL not accessible for {file_info['name']}: HTTP {response.status_code}")
            except Exception as e:
                self.logger.warning(f"Could not verify access to {file_info['name']}: {str(e)}")
                # Still include the file - download attempt will handle the error
                accessible_files.append(file_info)
        
        self.logger.info(f"Discovered {len(accessible_files)} XML files for download")
        return accessible_files
    
    def _download_single_file(self, file_name: str, file_url: str) -> bool:
        """
        Download a single XML file with retry logic and validation.
        
        Args:
            file_name (str): Name of the file to download
            file_url (str): URL to download the file from
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading {file_name} (attempt {attempt + 1}/{max_retries})")
                
                # Download to temporary file first
                temp_file = self.temp_directory / f"temp_{file_name}"
                
                response = self.session.get(file_url, timeout=30)
                response.raise_for_status()
                
                # Write to temporary file
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                
                # Validate the downloaded content
                if self.validate_xml_content(temp_file):
                    # Create backup of existing file if it exists
                    target_file = self.data_directory / file_name
                    if target_file.exists():
                        self.backup_existing_files([file_name])
                    
                    # Move temp file to final location (atomic operation)
                    shutil.move(str(temp_file), str(target_file))
                    
                    # Update file timestamps
                    self.update_file_timestamps(file_name)
                    
                    self.logger.info(f"Successfully downloaded and validated {file_name}")
                    return True
                else:
                    self.logger.error(f"Downloaded file {file_name} failed validation")
                    temp_file.unlink(missing_ok=True)  # Clean up invalid file
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Network error downloading {file_name} (attempt {attempt + 1}): {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error downloading {file_name} (attempt {attempt + 1}): {str(e)}")
            
            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        self.logger.error(f"Failed to download {file_name} after {max_retries} attempts")
        return False
    
    def validate_xml_content(self, file_path: Path) -> bool:
        """
        Validate that the downloaded file contains valid XML with expected structure.
        
        Args:
            file_path (Path): Path to the XML file to validate
        
        Returns:
            bool: True if the XML is valid and has expected structure, False otherwise
        """
        try:
            # Check file size (should not be empty or too small)
            if file_path.stat().st_size < 100:  # Less than 100 bytes is suspicious
                self.logger.warning(f"File {file_path.name} is too small ({file_path.stat().st_size} bytes)")
                return False
            
            # Parse XML to ensure it's valid
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Basic validation - check if it looks like vessel data
            # Look for common vessel data elements
            vessel_indicators = ['vessel', 'ship', 'arrival', 'departure', 'call_sign', 'vessel_name']
            xml_content = ET.tostring(root, encoding='unicode').lower()
            
            found_indicators = sum(1 for indicator in vessel_indicators if indicator in xml_content)
            
            if found_indicators < 2:  # Should have at least 2 vessel-related terms
                self.logger.warning(f"File {file_path.name} doesn't appear to contain vessel data")
                return False
            
            self.logger.info(f"File {file_path.name} passed validation checks")
            return True
            
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error in {file_path.name}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Validation error for {file_path.name}: {str(e)}")
            return False
    
    def backup_existing_files(self, file_names: List[str]) -> bool:
        """
        Create timestamped backups of existing files before overwriting.
        
        Args:
            file_names (List[str]): List of file names to backup
        
        Returns:
            bool: True if all backups were successful, False otherwise
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        success_count = 0
        
        for file_name in file_names:
            try:
                source_file = self.data_directory / file_name
                if source_file.exists():
                    backup_name = f"{file_name.replace('.xml', '')}_{timestamp}.xml"
                    backup_file = self.backup_directory / backup_name
                    
                    shutil.copy2(source_file, backup_file)
                    self.logger.info(f"Created backup: {backup_name}")
                    success_count += 1
                else:
                    self.logger.info(f"No existing file to backup: {file_name}")
                    success_count += 1  # Not an error if file doesn't exist
                    
            except Exception as e:
                self.logger.error(f"Failed to backup {file_name}: {str(e)}")
        
        return success_count == len(file_names)
    
    def update_file_timestamps(self, file_name: str) -> None:
        """
        Update metadata tracking for downloaded files.
        
        Args:
            file_name (str): Name of the file to track
        """
        try:
            metadata_file = self.logs_directory / 'file_metadata.log'
            timestamp = datetime.now().isoformat()
            
            with open(metadata_file, 'a') as f:
                f.write(f"{timestamp},{file_name},downloaded\n")
                
            self.logger.info(f"Updated metadata for {file_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to update metadata for {file_name}: {str(e)}")
    
    def get_last_update_time(self, file_name: str) -> Optional[datetime]:
        """
        Get the last update time for a specific file.
        
        Args:
            file_name (str): Name of the file to check
        
        Returns:
            Optional[datetime]: Last update time or None if not found
        """
        try:
            file_path = self.data_directory / file_name
            if file_path.exists():
                timestamp = file_path.stat().st_mtime
                return datetime.fromtimestamp(timestamp)
        except Exception as e:
            self.logger.error(f"Error getting last update time for {file_name}: {str(e)}")
        
        return None
    
    def cleanup_old_backups(self, days_to_keep: int = 7) -> None:
        """
        Clean up backup files older than specified days.
        
        Args:
            days_to_keep (int): Number of days of backups to retain
        """
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
            cleaned_count = 0
            
            for backup_file in self.backup_directory.glob('*.xml'):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    cleaned_count += 1
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old backup files")
                
        except Exception as e:
            self.logger.error(f"Error during backup cleanup: {str(e)}")
    
    def get_pipeline_status(self) -> Dict[str, any]:
        """
        Get current status of the vessel data pipeline.
        
        Returns:
            Dict[str, any]: Status information including last update times and file counts
        """
        status = {
            'enabled': self.pipeline_enabled,
            'last_fetch_attempt': None,
            'files_status': {},
            'backup_count': 0,
            'errors_today': 0
        }
        
        try:
            # Check status of expected files
            for file_name in self.expected_files:
                last_update = self.get_last_update_time(file_name)
                file_path = self.data_directory / file_name
                
                status['files_status'][file_name] = {
                    'exists': file_path.exists(),
                    'last_update': last_update.isoformat() if last_update else None,
                    'size_bytes': file_path.stat().st_size if file_path.exists() else 0
                }
            
            # Count backup files
            status['backup_count'] = len(list(self.backup_directory.glob('*.xml')))
            
            # Count today's errors from log file
            today = datetime.now().strftime("%Y%m%d")
            log_file = self.logs_directory / f'vessel_pipeline_{today}.log'
            if log_file.exists():
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    status['errors_today'] = log_content.count('ERROR')
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline status: {str(e)}")
        
        return status