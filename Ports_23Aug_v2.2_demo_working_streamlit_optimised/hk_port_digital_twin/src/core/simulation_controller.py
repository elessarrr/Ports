"""Simulation Controller for Hong Kong Port Digital Twin

This module provides high-level control over simulation execution,
including start/stop/pause/reset functionality.
"""

from enum import Enum
import threading
import time
from typing import Optional, Callable, Tuple
import logging

from .port_simulation import PortSimulation
from ..utils.metrics_collector import MetricsCollector


class SimulationState(Enum):
    """Enumeration of possible simulation states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class SimulationController:
    """High-level simulation control with threading support"""
    
    def __init__(self, port_simulation: PortSimulation, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize simulation controller
        
        Args:
            port_simulation: The port simulation instance to control
            metrics_collector: Optional metrics collector for tracking performance
        """
        self.simulation = port_simulation
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.state = SimulationState.STOPPED
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        # Simulation parameters
        self.duration: float = 0.0
        self.current_time: float = 0.0
        self.time_step: float = 1.0  # Default 1 hour time steps
        
        # Callbacks for state changes
        self.on_state_change: Optional[Callable[[SimulationState], None]] = None
        self.on_progress_update: Optional[Callable[[float, float], None]] = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def start(self, duration: float, time_step: float = 1.0, threaded: bool = True) -> bool:
        """Start simulation
        
        Args:
            duration: Total simulation duration in hours
            time_step: Time step size in hours (default: 1.0)
            threaded: Whether to run in separate thread (default: True)
            
        Returns:
            bool: True if simulation started successfully, False otherwise
        """
        if self.state in [SimulationState.RUNNING, SimulationState.COMPLETED]:
            self.logger.warning(f"Cannot start simulation in {self.state} state")
            return False
            
        if duration <= 0:
            self.logger.error("Duration must be positive")
            return False
            
        self.duration = duration
        self.time_step = time_step
        self.current_time = 0.0
        
        # Reset events
        self.stop_event.clear()
        self.pause_event.clear()
        
        # Initialize metrics collection
        self.metrics_collector.start_collection(0.0)
        
        if threaded:
            self.thread = threading.Thread(target=self._run_simulation, daemon=True)
            self.thread.start()
        else:
            self._run_simulation()
            
        return True
    
    def stop(self) -> bool:
        """Stop the running simulation
        
        Returns:
            bool: True if simulation was stopped, False if not running
        """
        if self.state not in [SimulationState.RUNNING, SimulationState.PAUSED]:
            return False
            
        self.stop_event.set()
        
        # Wait for thread to finish if it exists
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            
        self._set_state(SimulationState.STOPPED)
        return True
    
    def pause(self) -> bool:
        """Pause the running simulation
        
        Returns:
            bool: True if simulation was paused, False if not running
        """
        if self.state != SimulationState.RUNNING:
            return False
            
        self.pause_event.set()
        self._set_state(SimulationState.PAUSED)
        return True
    
    def resume(self) -> bool:
        """Resume a paused simulation
        
        Returns:
            bool: True if simulation was resumed, False if not paused
        """
        if self.state != SimulationState.PAUSED:
            return False
            
        self.pause_event.clear()
        self._set_state(SimulationState.RUNNING)
        return True
    
    def reset(self) -> bool:
        """Reset simulation to initial state
        
        Returns:
            bool: True if reset successful, False otherwise
        """
        # Stop if running
        if self.state in [SimulationState.RUNNING, SimulationState.PAUSED]:
            self.stop()
            
        # Reset simulation state
        self.current_time = 0.0
        self.duration = 0.0
        
        # Reset simulation components
        try:
            self.simulation.reset_simulation()
            self.metrics_collector.reset_metrics()
            self._set_state(SimulationState.STOPPED)
            return True
        except Exception as e:
            self.logger.error(f"Error during reset: {e}")
            self._set_state(SimulationState.ERROR)
            return False
    
    def get_progress(self) -> Tuple[float, float]:
        """Get current simulation progress
        
        Returns:
            tuple: (current_time, duration)
        """
        return self.current_time, self.duration
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage
        
        Returns:
            float: Progress percentage (0.0 to 100.0)
        """
        if self.duration <= 0:
            return 0.0
        return min(100.0, (self.current_time / self.duration) * 100.0)
    
    def is_running(self) -> bool:
        """Check if simulation is currently running
        
        Returns:
            bool: True if running, False otherwise
        """
        return self.state == SimulationState.RUNNING
    
    def is_paused(self) -> bool:
        """Check if simulation is currently paused
        
        Returns:
            bool: True if paused, False otherwise
        """
        return self.state == SimulationState.PAUSED
    
    def is_completed(self) -> bool:
        """Check if simulation has completed
        
        Returns:
            bool: True if completed, False otherwise
        """
        return self.state == SimulationState.COMPLETED
    
    def get_state(self) -> SimulationState:
        """Get current simulation state
        
        Returns:
            SimulationState: Current state
        """
        return self.state
    
    def _run_simulation(self):
        """Internal method to run the simulation loop"""
        try:
            self._set_state(SimulationState.RUNNING)
            
            while self.current_time < self.duration:
                # Check for stop signal
                if self.stop_event.is_set():
                    break
                    
                # Check for pause signal
                if self.pause_event.is_set():
                    self.pause_event.wait()  # Wait until resumed
                    
                # Check for stop signal again after pause
                if self.stop_event.is_set():
                    break
                
                # Run simulation step
                self._run_simulation_step()
                
                # Update progress
                self.current_time += self.time_step
                
                # Notify progress update
                if self.on_progress_update:
                    self.on_progress_update(self.current_time, self.duration)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
            
            # End metrics collection
            self.metrics_collector.end_collection(self.current_time)
            
            # Set final state
            if self.stop_event.is_set():
                self._set_state(SimulationState.STOPPED)
            else:
                self._set_state(SimulationState.COMPLETED)
                
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            self._set_state(SimulationState.ERROR)
    
    def _run_simulation_step(self):
        """Run a single simulation time step"""
        # Advance SimPy environment by time_step
        try:
            self.simulation.env.run(until=self.simulation.env.now + self.time_step)
        except Exception as e:
            self.logger.warning(f"Simulation step error: {e}")
        
        # Update berth utilization metrics
        for berth_id, berth in self.simulation.berth_manager.berths.items():
            utilization = 1.0 if berth.is_occupied else 0.0
            self.metrics_collector.record_berth_utilization(
                berth_id, 
                utilization
            )
        
        # Record basic metrics (queue length would need to be tracked separately)
        # For now, we'll use ships_processed as a proxy for activity
        self.metrics_collector.record_queue_length(
            max(0, self.simulation.total_ships_generated - self.simulation.ships_processed)
        )
    
    def _set_state(self, new_state: SimulationState):
        """Set simulation state and notify callback
        
        Args:
            new_state: New simulation state
        """
        old_state = self.state
        self.state = new_state
        
        self.logger.info(f"Simulation state changed: {old_state.value} -> {new_state.value}")
        
        # Notify callback
        if self.on_state_change:
            self.on_state_change(new_state)
    
    def get_metrics_summary(self) -> dict:
        """Get current metrics summary
        
        Returns:
            dict: Metrics summary
        """
        return self.metrics_collector.get_performance_summary()
    
    def export_metrics_to_dataframe(self) -> dict:
        """Export metrics to pandas DataFrames
        
        Returns:
            dict: Dictionary of DataFrames with metrics data
        """
        return self.metrics_collector.export_to_dataframe()
    
    def get_metrics(self) -> dict:
        """Get performance summary from metrics collector
        
        Returns:
            dict: Dictionary containing all calculated KPIs
        """
        return self.metrics_collector.get_performance_summary()