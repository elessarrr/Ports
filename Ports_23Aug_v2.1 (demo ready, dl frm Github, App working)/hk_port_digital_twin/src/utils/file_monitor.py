# Comments for context:
# This module implements a file monitoring system for automatic data updates.
# It watches for new PDF/Excel releases from government sources and triggers
# data processing when new files are detected. This enables real-time updates
# to the port simulation without manual intervention.

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass
import hashlib
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class FileMonitorConfig:
    """Configuration for file monitoring."""
    watch_directory: str
    file_patterns: List[str]  # e.g., ['*.xml', '*.csv', '*.xlsx']
    callback_function: Callable[[str], None]
    check_interval: int = 30  # seconds
    file_stable_time: int = 5  # seconds to wait before processing
    max_file_age: int = 86400  # seconds (24 hours)

@dataclass
class FileChangeEvent:
    """Represents a file change event."""
    file_path: str
    event_type: str  # 'created', 'modified', 'deleted'
    timestamp: datetime
    file_size: int
    file_hash: Optional[str] = None

class PortDataFileHandler(FileSystemEventHandler):
    """Custom file system event handler for port data files."""
    
    def __init__(self, monitor_instance):
        super().__init__()
        self.monitor = monitor_instance
        self.logger = logging.getLogger(__name__)
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self.logger.info(f"New file detected: {event.src_path}")
            self.monitor._handle_file_event(event.src_path, 'created')
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self.logger.info(f"File modified: {event.src_path}")
            self.monitor._handle_file_event(event.src_path, 'modified')
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            self.logger.info(f"File deleted: {event.src_path}")
            self.monitor._handle_file_event(event.src_path, 'deleted')

class FileMonitor:
    """File monitoring system for automatic data updates.
    
    This class monitors specified directories for changes to data files
    and triggers appropriate processing callbacks when changes are detected.
    """
    
    def __init__(self, config: FileMonitorConfig):
        self.config = config
        self.observer = Observer()
        self.event_handler = PortDataFileHandler(self)
        self.is_monitoring = False
        self.file_states = {}  # Track file states for change detection
        self.pending_files = {}  # Files waiting for stability
        self.processed_files = set()  # Track processed files to avoid duplicates
        
        # Create monitoring directory if it doesn't exist
        Path(self.config.watch_directory).mkdir(parents=True, exist_ok=True)
        
        # Load previous file states
        self._load_file_states()
        
        logger.info(f"File monitor initialized for directory: {self.config.watch_directory}")
    
    def start_monitoring(self):
        """Start the file monitoring service."""
        try:
            self.observer.schedule(
                self.event_handler,
                self.config.watch_directory,
                recursive=True
            )
            self.observer.start()
            self.is_monitoring = True
            
            # Start background thread for processing pending files
            self.processing_thread = threading.Thread(target=self._process_pending_files, daemon=True)
            self.processing_thread.start()
            
            logger.info("File monitoring started")
            
            # Perform initial scan of existing files
            self._initial_scan()
            
        except Exception as e:
            logger.error(f"Error starting file monitor: {e}")
            raise
    
    def stop_monitoring(self):
        """Stop the file monitoring service."""
        try:
            if self.is_monitoring:
                self.observer.stop()
                self.observer.join()
                self.is_monitoring = False
                
                # Save current file states
                self._save_file_states()
                
                logger.info("File monitoring stopped")
                
        except Exception as e:
            logger.error(f"Error stopping file monitor: {e}")
    
    def _handle_file_event(self, file_path: str, event_type: str):
        """Handle a file system event."""
        try:
            # Check if file matches our patterns
            if not self._matches_patterns(file_path):
                return
            
            # Skip if file is too old (avoid processing old files on startup)
            if event_type == 'created' and self._is_file_too_old(file_path):
                logger.debug(f"Skipping old file: {file_path}")
                return
            
            # Add to pending files for stability check
            self.pending_files[file_path] = {
                'event_type': event_type,
                'timestamp': datetime.now(),
                'attempts': 0
            }
            
            logger.debug(f"File {file_path} added to pending queue ({event_type})")
            
        except Exception as e:
            logger.error(f"Error handling file event for {file_path}: {e}")
    
    def _process_pending_files(self):
        """Background thread to process pending files after they stabilize."""
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                files_to_process = []
                
                # Check which files are ready for processing
                for file_path, info in list(self.pending_files.items()):
                    time_since_event = (current_time - info['timestamp']).total_seconds()
                    
                    if time_since_event >= self.config.file_stable_time:
                        if self._is_file_stable(file_path):
                            files_to_process.append(file_path)
                        else:
                            # File still changing, reset timestamp
                            info['timestamp'] = current_time
                            info['attempts'] += 1
                            
                            # Give up after too many attempts
                            if info['attempts'] > 10:
                                logger.warning(f"Giving up on unstable file: {file_path}")
                                del self.pending_files[file_path]
                
                # Process stable files
                for file_path in files_to_process:
                    self._process_file(file_path)
                    del self.pending_files[file_path]
                
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Error in pending files processing: {e}")
                time.sleep(self.config.check_interval)
    
    def _process_file(self, file_path: str):
        """Process a file that has been detected as changed."""
        try:
            # Skip if already processed recently
            file_key = self._get_file_key(file_path)
            if file_key in self.processed_files:
                logger.debug(f"File already processed recently: {file_path}")
                return
            
            # Check if file actually changed
            if not self._has_file_changed(file_path):
                logger.debug(f"File content unchanged: {file_path}")
                return
            
            logger.info(f"Processing file: {file_path}")
            
            # Call the configured callback function
            self.config.callback_function(file_path)
            
            # Update file state
            self._update_file_state(file_path)
            
            # Mark as processed
            self.processed_files.add(file_key)
            
            # Clean up old processed files (keep only recent ones)
            self._cleanup_processed_files()
            
            logger.info(f"Successfully processed file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    def _matches_patterns(self, file_path: str) -> bool:
        """Check if file matches configured patterns."""
        file_name = os.path.basename(file_path).lower()
        
        for pattern in self.config.file_patterns:
            pattern = pattern.lower().replace('*', '')
            if pattern in file_name or file_name.endswith(pattern):
                return True
        
        return False
    
    def _is_file_too_old(self, file_path: str) -> bool:
        """Check if file is too old to process."""
        try:
            file_mtime = os.path.getmtime(file_path)
            file_age = time.time() - file_mtime
            return file_age > self.config.max_file_age
        except OSError:
            return True
    
    def _is_file_stable(self, file_path: str) -> bool:
        """Check if file is stable (not being written to)."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False
            
            # Get current file stats
            stat1 = os.stat(file_path)
            time.sleep(1)  # Wait a bit
            stat2 = os.stat(file_path)
            
            # File is stable if size and modification time haven't changed
            return (stat1.st_size == stat2.st_size and 
                   stat1.st_mtime == stat2.st_mtime)
            
        except OSError:
            return False
    
    def _has_file_changed(self, file_path: str) -> bool:
        """Check if file content has actually changed."""
        try:
            current_hash = self._calculate_file_hash(file_path)
            previous_hash = self.file_states.get(file_path, {}).get('hash')
            
            return current_hash != previous_hash
            
        except Exception as e:
            logger.error(f"Error checking file change for {file_path}: {e}")
            return True  # Assume changed if we can't determine
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return str(time.time())  # Use timestamp as fallback
    
    def _update_file_state(self, file_path: str):
        """Update stored state for a file."""
        try:
            stat = os.stat(file_path)
            file_hash = self._calculate_file_hash(file_path)
            
            self.file_states[file_path] = {
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'hash': file_hash,
                'last_processed': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating file state for {file_path}: {e}")
    
    def _get_file_key(self, file_path: str) -> str:
        """Generate a unique key for a file."""
        try:
            stat = os.stat(file_path)
            return f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        except OSError:
            return f"{file_path}_{time.time()}"
    
    def _cleanup_processed_files(self):
        """Remove old entries from processed files set."""
        # Keep only recent entries (last 1000)
        if len(self.processed_files) > 1000:
            # Convert to list, sort, and keep recent ones
            # This is a simple cleanup - in production you might want more sophisticated logic
            self.processed_files = set(list(self.processed_files)[-500:])
    
    def _initial_scan(self):
        """Perform initial scan of existing files."""
        try:
            logger.info("Performing initial file scan...")
            
            watch_path = Path(self.config.watch_directory)
            if not watch_path.exists():
                logger.warning(f"Watch directory does not exist: {watch_path}")
                return
            
            # Scan for existing files
            for file_path in watch_path.rglob('*'):
                if file_path.is_file() and self._matches_patterns(str(file_path)):
                    # Check if this is a new file or changed file
                    if self._has_file_changed(str(file_path)):
                        logger.info(f"Found new/changed file during initial scan: {file_path}")
                        self._handle_file_event(str(file_path), 'modified')
            
            logger.info("Initial file scan completed")
            
        except Exception as e:
            logger.error(f"Error during initial file scan: {e}")
    
    def _save_file_states(self):
        """Save file states to disk for persistence."""
        try:
            state_file = Path(self.config.watch_directory) / '.file_monitor_state.json'
            with open(state_file, 'w') as f:
                json.dump(self.file_states, f, indent=2)
            logger.debug("File states saved")
        except Exception as e:
            logger.error(f"Error saving file states: {e}")
    
    def _load_file_states(self):
        """Load file states from disk."""
        try:
            state_file = Path(self.config.watch_directory) / '.file_monitor_state.json'
            if state_file.exists():
                with open(state_file, 'r') as f:
                    self.file_states = json.load(f)
                logger.debug("File states loaded")
            else:
                self.file_states = {}
        except Exception as e:
            logger.error(f"Error loading file states: {e}")
            self.file_states = {}
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'is_monitoring': self.is_monitoring,
            'watch_directory': self.config.watch_directory,
            'file_patterns': self.config.file_patterns,
            'pending_files': len(self.pending_files),
            'tracked_files': len(self.file_states),
            'processed_files_count': len(self.processed_files)
        }

class PortDataFileMonitor:
    """Specialized file monitor for Hong Kong port data files.
    
    This class provides a high-level interface for monitoring port-specific
    data files and integrating with the port simulation system.
    """
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = Path(data_directory)
        self.monitors = {}
        self.callbacks = {}
        
        # Ensure data directory exists
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Port data file monitor initialized for: {self.data_directory}")
    
    def setup_vessel_monitoring(self, callback: Callable[[str], None]):
        """Set up monitoring for vessel arrival XML files."""
        config = FileMonitorConfig(
            watch_directory=str(self.data_directory / "vessel_arrivals"),
            file_patterns=['*.xml'],
            callback_function=callback,
            check_interval=30,
            file_stable_time=5
        )
        
        monitor = FileMonitor(config)
        self.monitors['vessel_arrivals'] = monitor
        self.callbacks['vessel_arrivals'] = callback
        
        logger.info("Vessel arrival monitoring configured")
    
    def setup_cargo_monitoring(self, callback: Callable[[str], None]):
        """Set up monitoring for cargo statistics files."""
        config = FileMonitorConfig(
            watch_directory=str(self.data_directory / "cargo_statistics"),
            file_patterns=['*.csv', '*.xlsx', '*.xls'],
            callback_function=callback,
            check_interval=60,  # Less frequent for cargo data
            file_stable_time=10
        )
        
        monitor = FileMonitor(config)
        self.monitors['cargo_statistics'] = monitor
        self.callbacks['cargo_statistics'] = callback
        
        logger.info("Cargo statistics monitoring configured")
    
    def setup_berth_monitoring(self, callback: Callable[[str], None]):
        """Set up monitoring for berth occupancy files."""
        config = FileMonitorConfig(
            watch_directory=str(self.data_directory / "berth_data"),
            file_patterns=['*.csv', '*.json'],
            callback_function=callback,
            check_interval=30,
            file_stable_time=5
        )
        
        monitor = FileMonitor(config)
        self.monitors['berth_data'] = monitor
        self.callbacks['berth_data'] = callback
        
        logger.info("Berth data monitoring configured")
    
    def start_all_monitoring(self):
        """Start all configured monitors."""
        for name, monitor in self.monitors.items():
            try:
                monitor.start_monitoring()
                logger.info(f"Started monitoring for {name}")
            except Exception as e:
                logger.error(f"Failed to start monitoring for {name}: {e}")
    
    def stop_all_monitoring(self):
        """Stop all monitors."""
        for name, monitor in self.monitors.items():
            try:
                monitor.stop_monitoring()
                logger.info(f"Stopped monitoring for {name}")
            except Exception as e:
                logger.error(f"Failed to stop monitoring for {name}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all monitors."""
        status = {
            'data_directory': str(self.data_directory),
            'monitors': {}
        }
        
        for name, monitor in self.monitors.items():
            status['monitors'][name] = monitor.get_monitoring_status()
        
        return status

# Utility functions for integration
def create_default_port_monitor() -> PortDataFileMonitor:
    """Create a default port data file monitor with standard callbacks."""
    monitor = PortDataFileMonitor()
    
    # Define default callbacks
    def vessel_callback(file_path: str):
        logger.info(f"New vessel data detected: {file_path}")
        # Here you would trigger vessel data reloading
        # Example: reload_vessel_data(file_path)
    
    def cargo_callback(file_path: str):
        logger.info(f"New cargo data detected: {file_path}")
        # Here you would trigger cargo data reloading
        # Example: reload_cargo_data(file_path)
    
    def berth_callback(file_path: str):
        logger.info(f"New berth data detected: {file_path}")
        # Here you would trigger berth data reloading
        # Example: reload_berth_data(file_path)
    
    # Set up monitoring
    monitor.setup_vessel_monitoring(vessel_callback)
    monitor.setup_cargo_monitoring(cargo_callback)
    monitor.setup_berth_monitoring(berth_callback)
    
    return monitor