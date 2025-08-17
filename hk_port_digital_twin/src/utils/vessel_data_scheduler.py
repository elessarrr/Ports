# Comments for context:
# This module implements the background scheduling system for the vessel data pipeline.
# It runs the vessel data fetcher at regular intervals (every 20 minutes) and on application startup.
# The scheduler runs in a separate thread to avoid blocking the main Streamlit application.
# 
# Problem: Need automated, non-blocking execution of the vessel data pipeline
# Solution: Background thread with configurable scheduling and graceful shutdown
# Approach: Thread-safe scheduler with proper error handling and lifecycle management

import os
import time
import threading
import schedule
import logging
from datetime import datetime
from typing import Optional, Callable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VesselDataScheduler:
    """
    Manages background scheduling for the vessel data pipeline.
    
    This class provides functionality to:
    - Schedule periodic execution of the vessel data fetcher
    - Run initial fetch on application startup
    - Manage scheduler lifecycle (start/stop)
    - Handle errors gracefully without crashing the main application
    - Provide status information about scheduled tasks
    """
    
    def __init__(self, fetcher_callback: Callable[[], dict]):
        """
        Initialize the scheduler with a callback function for data fetching.
        
        Args:
            fetcher_callback (Callable[[], dict]): Function to call for fetching data
        """
        # Configuration from environment variables
        self.fetch_interval = int(os.getenv('VESSEL_DATA_FETCH_INTERVAL', '20'))
        self.pipeline_enabled = os.getenv('VESSEL_DATA_PIPELINE_ENABLED', 'true').lower() == 'true'
        
        # Callback function for data fetching
        self.fetcher_callback = fetcher_callback
        
        # Threading and scheduling
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_running = False
        
        # Status tracking
        self.last_execution_time: Optional[datetime] = None
        self.last_execution_result: Optional[dict] = None
        self.execution_count = 0
        self.error_count = 0
        
        # Setup logging
        self.logger = logging.getLogger('vessel_data_scheduler')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.info(f"VesselDataScheduler initialized. Interval: {self.fetch_interval} minutes, Enabled: {self.pipeline_enabled}")
    
    def start(self, run_immediately: bool = True) -> bool:
        """
        Start the background scheduler.
        
        Args:
            run_immediately (bool): Whether to run the fetcher immediately on startup
        
        Returns:
            bool: True if scheduler started successfully, False otherwise
        """
        if not self.pipeline_enabled:
            self.logger.info("Vessel data pipeline is disabled. Scheduler not started.")
            return False
        
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return True
        
        try:
            # Clear any existing scheduled jobs
            schedule.clear()
            
            # Schedule the periodic job
            schedule.every(self.fetch_interval).minutes.do(self._execute_fetch_job)
            
            # Reset stop event
            self.stop_event.clear()
            
            # Start the scheduler thread
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            self.is_running = True
            self.logger.info(f"Scheduler started successfully. Next execution in {self.fetch_interval} minutes.")
            
            # Run immediately if requested
            if run_immediately:
                self.logger.info("Running initial data fetch...")
                self._execute_fetch_job()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {str(e)}")
            return False
    
    def stop(self, timeout: float = 10.0) -> bool:
        """
        Stop the background scheduler gracefully.
        
        Args:
            timeout (float): Maximum time to wait for scheduler to stop
        
        Returns:
            bool: True if scheduler stopped successfully, False otherwise
        """
        if not self.is_running:
            self.logger.info("Scheduler is not running")
            return True
        
        try:
            self.logger.info("Stopping scheduler...")
            
            # Signal the scheduler thread to stop
            self.stop_event.set()
            
            # Wait for the thread to finish
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=timeout)
                
                if self.scheduler_thread.is_alive():
                    self.logger.warning(f"Scheduler thread did not stop within {timeout} seconds")
                    return False
            
            # Clear scheduled jobs
            schedule.clear()
            
            self.is_running = False
            self.scheduler_thread = None
            
            self.logger.info("Scheduler stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping scheduler: {str(e)}")
            return False
    
    def _run_scheduler(self) -> None:
        """
        Main scheduler loop that runs in a background thread.
        
        This method continuously checks for scheduled jobs and executes them
        until the stop event is set.
        """
        self.logger.info("Scheduler thread started")
        
        while not self.stop_event.is_set():
            try:
                # Run pending scheduled jobs
                schedule.run_pending()
                
                # Sleep for a short interval to avoid busy waiting
                # Check stop event every second
                self.stop_event.wait(timeout=1.0)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
                # Continue running even if there's an error
                time.sleep(5)  # Wait a bit longer after an error
        
        self.logger.info("Scheduler thread stopped")
    
    def _execute_fetch_job(self) -> None:
        """
        Execute the vessel data fetch job.
        
        This method is called by the scheduler and handles the actual
        execution of the data fetching callback with error handling.
        """
        execution_start = datetime.now()
        
        try:
            self.logger.info(f"Executing scheduled vessel data fetch (execution #{self.execution_count + 1})")
            
            # Call the fetcher callback
            result = self.fetcher_callback()
            
            # Update status tracking
            self.last_execution_time = execution_start
            self.last_execution_result = result
            self.execution_count += 1
            
            # Log results
            if result:
                success_count = sum(1 for success in result.values() if success)
                total_count = len(result)
                self.logger.info(f"Fetch job completed. Success: {success_count}/{total_count} files")
                
                if success_count < total_count:
                    self.error_count += 1
                    self.logger.warning(f"Some files failed to download: {result}")
            else:
                self.logger.info("Fetch job completed (no files to process)")
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error executing fetch job: {str(e)}")
            
            # Update status even on error
            self.last_execution_time = execution_start
            self.last_execution_result = {'error': str(e)}
            self.execution_count += 1
    
    def force_execution(self) -> dict:
        """
        Force immediate execution of the fetch job outside of the schedule.
        
        Returns:
            dict: Result of the fetch operation
        """
        self.logger.info("Force executing vessel data fetch")
        
        try:
            result = self.fetcher_callback()
            self.logger.info(f"Force execution completed: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error in force execution: {str(e)}")
            return {'error': str(e)}
    
    def get_next_execution_time(self) -> Optional[datetime]:
        """
        Get the next scheduled execution time.
        
        Returns:
            Optional[datetime]: Next execution time or None if not scheduled
        """
        try:
            jobs = schedule.get_jobs()
            if jobs:
                next_run = min(job.next_run for job in jobs)
                return next_run
        except Exception as e:
            self.logger.error(f"Error getting next execution time: {str(e)}")
        
        return None
    
    def get_scheduler_status(self) -> dict:
        """
        Get current status of the scheduler.
        
        Returns:
            dict: Status information including execution history and next run time
        """
        next_execution = self.get_next_execution_time()
        
        status = {
            'enabled': self.pipeline_enabled,
            'running': self.is_running,
            'fetch_interval_minutes': self.fetch_interval,
            'execution_count': self.execution_count,
            'error_count': self.error_count,
            'last_execution_time': self.last_execution_time.isoformat() if self.last_execution_time else None,
            'last_execution_result': self.last_execution_result,
            'next_execution_time': next_execution.isoformat() if next_execution else None,
            'thread_alive': self.scheduler_thread.is_alive() if self.scheduler_thread else False
        }
        
        return status
    
    def update_interval(self, new_interval: int) -> bool:
        """
        Update the fetch interval and restart the scheduler if running.
        
        Args:
            new_interval (int): New interval in minutes
        
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            old_interval = self.fetch_interval
            self.fetch_interval = new_interval
            
            if self.is_running:
                self.logger.info(f"Updating fetch interval from {old_interval} to {new_interval} minutes")
                
                # Restart scheduler with new interval
                was_running = self.is_running
                self.stop()
                
                if was_running:
                    return self.start(run_immediately=False)
            
            self.logger.info(f"Fetch interval updated to {new_interval} minutes")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating interval: {str(e)}")
            return False
    
    def __del__(self):
        """
        Cleanup when the scheduler object is destroyed.
        """
        if self.is_running:
            self.stop()