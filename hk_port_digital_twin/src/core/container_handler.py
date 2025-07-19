"""Container Handling System for Hong Kong Port Digital Twin

Comments for context:
This module simulates container loading/unloading operations at port berths.
It calculates processing times based on ship type, container count, and available cranes.
The system uses SimPy for discrete event simulation of container operations.

Key concepts:
- Processing rates vary by ship type (container vs bulk)
- Crane count affects processing speed (more cranes = faster processing)
- Processing includes both unloading and loading operations
- Times are calculated in simulation time units (hours)
"""

import simpy
from typing import Dict, List, Optional
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import SHIP_TYPES


class ContainerHandler:
    """Handles container operations at berths
    
    This class manages the simulation of container loading/unloading operations.
    It calculates processing times and simulates the actual container handling
    process using SimPy timeouts.
    """
    
    def __init__(self, env: simpy.Environment):
        """Initialize the container handler
        
        Args:
            env: SimPy environment for simulation timing
        """
        self.env = env
        # Load processing rates from configuration
        self.processing_rates = {
            ship_type: config['processing_rate'] 
            for ship_type, config in SHIP_TYPES.items()
        }
        # Track processing history for metrics
        self.processing_history = []
        
    def calculate_processing_time(self, ship_type: str, containers_to_unload: int, 
                                containers_to_load: int, crane_count: int) -> float:
        """Calculate time needed to process containers
        
        Args:
            ship_type: Type of ship ('container', 'bulk', etc.)
            containers_to_unload: Number of containers to unload
            containers_to_load: Number of containers to load
            crane_count: Number of cranes available at the berth
            
        Returns:
            Processing time in simulation time units (hours)
            
        Raises:
            ValueError: If ship_type is not recognized or crane_count is invalid
        """
        if ship_type not in self.processing_rates:
            raise ValueError(f"Unknown ship type: {ship_type}")
            
        if crane_count <= 0:
            raise ValueError(f"Invalid crane count: {crane_count}")
            
        # Get base processing rate for this ship type
        base_rate = self.processing_rates[ship_type]
        
        # Calculate total containers to process
        total_containers = containers_to_unload + containers_to_load
        
        # Calculate base processing time (assuming 1 crane)
        base_time = total_containers / base_rate if base_rate > 0 else 0
        
        # Adjust for number of cranes (more cranes = faster processing)
        # Use diminishing returns: efficiency decreases with more cranes
        crane_efficiency = min(crane_count, 4) * 0.8 + max(0, crane_count - 4) * 0.3
        processing_time = base_time / crane_efficiency if crane_efficiency > 0 else base_time
        
        return max(0.1, processing_time)  # Minimum 0.1 hours (6 minutes)
        
    def process_ship(self, ship, berth):
        """Simulate container loading/unloading process
        
        This is a SimPy process that simulates the time required to
        load and unload containers from a ship at a berth.
        
        Args:
            ship: Ship object with container information
            berth: Berth object with crane information
            
        Yields:
            SimPy timeout event for the processing duration
        """
        # Calculate processing time
        processing_time = self.calculate_processing_time(
            ship.ship_type,
            ship.containers_to_unload,
            ship.containers_to_load,
            berth.crane_count
        )
        
        # Record start of processing
        start_time = self.env.now
        
        # Log processing start
        print(f"Time {self.env.now:.1f}: Starting container processing for ship {ship.ship_id} "
              f"at berth {berth.berth_id} (estimated {processing_time:.1f} hours)")
        
        # Simulate the processing time
        yield self.env.timeout(processing_time)
        
        # Record completion
        end_time = self.env.now
        
        # Store processing record for metrics
        processing_record = {
            'ship_id': ship.ship_id,
            'ship_type': ship.ship_type,
            'berth_id': berth.berth_id,
            'containers_unloaded': ship.containers_to_unload,
            'containers_loaded': ship.containers_to_load,
            'crane_count': berth.crane_count,
            'start_time': start_time,
            'end_time': end_time,
            'processing_time': processing_time,
            'actual_time': end_time - start_time
        }
        self.processing_history.append(processing_record)
        
        # Log processing completion
        print(f"Time {self.env.now:.1f}: Completed container processing for ship {ship.ship_id} "
              f"at berth {berth.berth_id}")
              
    def get_processing_statistics(self) -> Dict:
        """Get statistics about container processing operations
        
        Returns:
            Dictionary containing processing statistics
        """
        if not self.processing_history:
            return {
                'total_operations': 0,
                'average_processing_time': 0,
                'total_containers_processed': 0,
                'average_crane_utilization': 0
            }
            
        total_ops = len(self.processing_history)
        avg_time = sum(record['processing_time'] for record in self.processing_history) / total_ops
        total_containers = sum(
            record['containers_unloaded'] + record['containers_loaded'] 
            for record in self.processing_history
        )
        avg_cranes = sum(record['crane_count'] for record in self.processing_history) / total_ops
        
        return {
            'total_operations': total_ops,
            'average_processing_time': round(avg_time, 2),
            'total_containers_processed': total_containers,
            'average_crane_utilization': round(avg_cranes, 2)
        }
        
    def reset_statistics(self):
        """Reset processing history and statistics"""
        self.processing_history = []