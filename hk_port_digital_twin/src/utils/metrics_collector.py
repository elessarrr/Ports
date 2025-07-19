"""Metrics Collection for Hong Kong Port Digital Twin

This module tracks and calculates key performance indicators
for port operations analysis.

The MetricsCollector provides a centralized way to gather simulation data
and calculate meaningful KPIs for port performance evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
from datetime import datetime
import statistics


@dataclass
class SimulationMetrics:
    """Container for all simulation metrics
    
    This dataclass holds all the raw data collected during simulation
    for later analysis and reporting.
    """
    ship_waiting_times: List[float] = field(default_factory=list)
    berth_utilization: Dict[int, float] = field(default_factory=dict)
    container_throughput: List[int] = field(default_factory=list)
    queue_lengths: List[int] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    ship_arrivals: List[Tuple[str, float]] = field(default_factory=list)  # (ship_id, arrival_time)
    ship_departures: List[Tuple[str, float]] = field(default_factory=list)  # (ship_id, departure_time)
    berth_assignments: List[Tuple[str, int, float]] = field(default_factory=list)  # (ship_id, berth_id, time)
    

class MetricsCollector:
    """Collects and analyzes simulation metrics
    
    This class provides methods to record various simulation events
    and calculate performance indicators from the collected data.
    """
    
    def __init__(self):
        """Initialize metrics collector with empty metrics"""
        self.metrics = SimulationMetrics()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def start_collection(self, simulation_time: float):
        """Mark the start of metrics collection
        
        Args:
            simulation_time: Current simulation time
        """
        self.start_time = simulation_time
        
    def end_collection(self, simulation_time: float):
        """Mark the end of metrics collection
        
        Args:
            simulation_time: Current simulation time
        """
        self.end_time = simulation_time
        
    def record_ship_waiting_time(self, ship_id: str, waiting_time: float):
        """Record how long a ship waited for berth
        
        Args:
            ship_id: Unique identifier for the ship
            waiting_time: Time spent waiting in hours
        """
        if waiting_time >= 0:
            self.metrics.ship_waiting_times.append(waiting_time)
            
    def record_berth_utilization(self, berth_id: int, utilization_rate: float):
        """Record berth utilization rate
        
        Args:
            berth_id: Unique identifier for the berth
            utilization_rate: Utilization rate as percentage (0-100)
        """
        if 0 <= utilization_rate <= 100:
            self.metrics.berth_utilization[berth_id] = utilization_rate
            
    def record_container_throughput(self, container_count: int):
        """Record container processing throughput
        
        Args:
            container_count: Number of containers processed
        """
        if container_count >= 0:
            self.metrics.container_throughput.append(container_count)
            
    def record_queue_length(self, queue_length: int):
        """Record current queue length
        
        Args:
            queue_length: Number of ships waiting in queue
        """
        if queue_length >= 0:
            self.metrics.queue_lengths.append(queue_length)
            
    def record_processing_time(self, processing_time: float):
        """Record ship processing time
        
        Args:
            processing_time: Time spent processing ship in hours
        """
        if processing_time >= 0:
            self.metrics.processing_times.append(processing_time)
            
    def record_ship_arrival(self, ship_id: str, arrival_time: float):
        """Record ship arrival event
        
        Args:
            ship_id: Unique identifier for the ship
            arrival_time: Simulation time when ship arrived
        """
        self.metrics.ship_arrivals.append((ship_id, arrival_time))
        
    def record_ship_departure(self, ship_id: str, departure_time: float):
        """Record ship departure event
        
        Args:
            ship_id: Unique identifier for the ship
            departure_time: Simulation time when ship departed
        """
        self.metrics.ship_departures.append((ship_id, departure_time))
        
    def record_berth_assignment(self, ship_id: str, berth_id: int, assignment_time: float):
        """Record berth assignment event
        
        Args:
            ship_id: Unique identifier for the ship
            berth_id: Unique identifier for the berth
            assignment_time: Simulation time when berth was assigned
        """
        self.metrics.berth_assignments.append((ship_id, berth_id, assignment_time))
        
    def calculate_average_waiting_time(self) -> float:
        """Calculate average ship waiting time
        
        Returns:
            Average waiting time in hours, 0 if no data
        """
        if not self.metrics.ship_waiting_times:
            return 0.0
        return statistics.mean(self.metrics.ship_waiting_times)
        
    def calculate_max_waiting_time(self) -> float:
        """Calculate maximum ship waiting time
        
        Returns:
            Maximum waiting time in hours, 0 if no data
        """
        if not self.metrics.ship_waiting_times:
            return 0.0
        return max(self.metrics.ship_waiting_times)
        
    def calculate_average_berth_utilization(self) -> float:
        """Calculate average berth utilization across all berths
        
        Returns:
            Average utilization rate as percentage (0-100), 0 if no data
        """
        if not self.metrics.berth_utilization:
            return 0.0
        return statistics.mean(self.metrics.berth_utilization.values())
        
    def calculate_total_container_throughput(self) -> int:
        """Calculate total containers processed
        
        Returns:
            Total number of containers processed
        """
        return sum(self.metrics.container_throughput)
        
    def calculate_average_queue_length(self) -> float:
        """Calculate average queue length
        
        Returns:
            Average number of ships in queue, 0 if no data
        """
        if not self.metrics.queue_lengths:
            return 0.0
        return statistics.mean(self.metrics.queue_lengths)
        
    def calculate_max_queue_length(self) -> int:
        """Calculate maximum queue length observed
        
        Returns:
            Maximum number of ships in queue, 0 if no data
        """
        if not self.metrics.queue_lengths:
            return 0
        return max(self.metrics.queue_lengths)
        
    def calculate_average_processing_time(self) -> float:
        """Calculate average ship processing time
        
        Returns:
            Average processing time in hours, 0 if no data
        """
        if not self.metrics.processing_times:
            return 0.0
        return statistics.mean(self.metrics.processing_times)
        
    def calculate_ship_arrival_rate(self) -> float:
        """Calculate ship arrival rate per hour
        
        Returns:
            Ships per hour, 0 if no data or insufficient time
        """
        if not self.metrics.ship_arrivals or self.start_time is None or self.end_time is None:
            return 0.0
            
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
            
        return len(self.metrics.ship_arrivals) / duration
        
    def calculate_ship_departure_rate(self) -> float:
        """Calculate ship departure rate per hour
        
        Returns:
            Ships per hour, 0 if no data or insufficient time
        """
        if not self.metrics.ship_departures or self.start_time is None or self.end_time is None:
            return 0.0
            
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
            
        return len(self.metrics.ship_departures) / duration
        
    def get_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary
        
        Returns:
            Dictionary containing all calculated KPIs
        """
        return {
            'waiting_times': {
                'average': self.calculate_average_waiting_time(),
                'maximum': self.calculate_max_waiting_time(),
                'count': len(self.metrics.ship_waiting_times)
            },
            'berth_utilization': {
                'average': self.calculate_average_berth_utilization(),
                'by_berth': dict(self.metrics.berth_utilization)
            },
            'throughput': {
                'total_containers': self.calculate_total_container_throughput(),
                'average_processing_time': self.calculate_average_processing_time()
            },
            'queue_performance': {
                'average_length': self.calculate_average_queue_length(),
                'maximum_length': self.calculate_max_queue_length()
            },
            'ship_flow': {
                'arrival_rate': self.calculate_ship_arrival_rate(),
                'departure_rate': self.calculate_ship_departure_rate(),
                'total_arrivals': len(self.metrics.ship_arrivals),
                'total_departures': len(self.metrics.ship_departures)
            },
            'simulation_info': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'duration': (self.end_time - self.start_time) if self.start_time is not None and self.end_time is not None else 0
            }
        }
        
    def export_to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Export metrics to pandas DataFrames for analysis
        
        Returns:
            Dictionary of DataFrames containing different metric types
        """
        dataframes = {}
        
        # Ship events DataFrame
        if self.metrics.ship_arrivals or self.metrics.ship_departures:
            ship_events = []
            
            for ship_id, time in self.metrics.ship_arrivals:
                ship_events.append({'ship_id': ship_id, 'event': 'arrival', 'time': time})
                
            for ship_id, time in self.metrics.ship_departures:
                ship_events.append({'ship_id': ship_id, 'event': 'departure', 'time': time})
                
            dataframes['ship_events'] = pd.DataFrame(ship_events)
            
        # Berth assignments DataFrame
        if self.metrics.berth_assignments:
            berth_data = []
            for ship_id, berth_id, time in self.metrics.berth_assignments:
                berth_data.append({'ship_id': ship_id, 'berth_id': berth_id, 'assignment_time': time})
            dataframes['berth_assignments'] = pd.DataFrame(berth_data)
            
        # Time series data
        if self.metrics.queue_lengths:
            dataframes['queue_lengths'] = pd.DataFrame({
                'queue_length': self.metrics.queue_lengths
            })
            
        if self.metrics.ship_waiting_times:
            dataframes['waiting_times'] = pd.DataFrame({
                'waiting_time': self.metrics.ship_waiting_times
            })
            
        return dataframes
        
    def reset_metrics(self):
        """Reset all collected metrics to start fresh"""
        self.metrics = SimulationMetrics()
        self.start_time = None
        self.end_time = None