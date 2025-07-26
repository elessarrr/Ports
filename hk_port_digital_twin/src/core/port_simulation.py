"""Main Port Simulation Engine for Hong Kong Port Digital Twin

This module orchestrates the entire port simulation using SimPy.
It coordinates ships, berths, and container handling in discrete events.

Key concepts:
- Uses SimPy environment for discrete event simulation
- Coordinates ship arrivals, berth allocation, and container processing
- Generates realistic ship arrival patterns
- Tracks simulation metrics and performance
"""

import simpy
import random
import sys
import os
from typing import Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import SIMULATION_CONFIG, SHIP_TYPES, BERTH_CONFIGS
from src.core.ship_manager import ShipManager, Ship
from src.core.berth_manager import BerthManager
from src.core.container_handler import ContainerHandler


class PortSimulation:
    """Main simulation controller that orchestrates all port operations
    
    This class manages the entire port simulation, coordinating ship arrivals,
    berth allocation, and container processing operations.
    """
    
    def __init__(self, config: Dict):
        """Initialize the port simulation
        
        Args:
            config: Configuration dictionary containing simulation parameters
        """
        self.env = simpy.Environment()
        self.config = config
        
        # Initialize all managers
        self.ship_manager = ShipManager(self.env)
        # Use berth config from parameter if provided, otherwise use settings
        berth_config = config.get('berths', BERTH_CONFIGS)
        self.berth_manager = BerthManager(self.env, berth_config)
        self.container_handler = ContainerHandler(self.env)
        
        # Simulation state
        self.running = False
        self.ships_processed = 0
        self.total_ships_generated = 0
        
        # Metrics tracking
        self.metrics = {
            'ships_arrived': 0,
            'ships_processed': 0,
            'total_waiting_time': 0,
            'simulation_start_time': 0,
            'simulation_end_time': 0
        }
        
    def run_simulation(self, duration: float) -> Dict:
        """Run simulation for specified duration
        
        Args:
            duration: Simulation duration in time units (hours)
            
        Returns:
            Dictionary containing simulation results and metrics
        """
        print(f"Starting port simulation for {duration} hours...")
        
        self.running = True
        self.metrics['simulation_start_time'] = self.env.now
        
        # Start ship arrival process
        self.env.process(self.ship_arrival_process())
        
        try:
            # Run simulation
            self.env.run(until=duration)
            
        except Exception as e:
            print(f"Simulation error: {e}")
            raise
        finally:
            self.running = False
            self.metrics['simulation_end_time'] = self.env.now
            
        print(f"Simulation completed at time {self.env.now:.1f}")
        return self._generate_final_report()
        
    def ship_arrival_process(self):
        """Generate ship arrivals over time
        
        This process continuously generates new ships arriving at the port
        based on configured arrival rates and patterns.
        """
        ship_id_counter = 1
        
        while self.running:
            try:
                # Calculate next arrival time (exponential distribution for realistic arrivals)
                arrival_interval = random.expovariate(1.0 / SIMULATION_CONFIG['ship_arrival_rate'])
                
                # Wait for next arrival
                yield self.env.timeout(arrival_interval)
                
                if not self.running:
                    break
                    
                # Generate new ship
                ship = self._generate_random_ship(f"SHIP_{ship_id_counter:03d}")
                ship_id_counter += 1
                self.total_ships_generated += 1
                self.metrics['ships_arrived'] += 1
                
                print(f"Time {self.env.now:.1f}: Ship {ship.ship_id} arrived at port")
                
                # Start ship processing
                self.env.process(self._process_ship(ship))
                
            except Exception as e:
                print(f"Error in ship arrival process: {e}")
                break
                
    def _process_ship(self, ship: Ship):
        """Process a single ship through the port system
        
        Args:
            ship: Ship object to process
        """
        arrival_time = self.env.now
        
        try:
            # Request berth allocation
            print(f"Time {self.env.now:.1f}: Ship {ship.ship_id} requesting berth...")
            
            # Find available berth
            berth_id = None
            while berth_id is None:
                # Estimate ship size based on containers (rough approximation)
                ship_size_teu = (ship.containers_to_unload + ship.containers_to_load) * 20  # Rough TEU estimate
                berth_id = self.berth_manager.find_available_berth(ship.ship_type, ship_size_teu)
                
                if berth_id is None:
                    # Wait a bit and try again
                    yield self.env.timeout(0.1)  # Wait 6 minutes
            
            # Allocate the berth
            allocation_success = self.berth_manager.allocate_berth(berth_id, ship.ship_id)
            if not allocation_success:
                print(f"Failed to allocate berth {berth_id} to ship {ship.ship_id}")
                return
                
            berth = self.berth_manager.get_berth(berth_id)
            
            waiting_time = self.env.now - arrival_time
            self.metrics['total_waiting_time'] += waiting_time
            
            print(f"Time {self.env.now:.1f}: Ship {ship.ship_id} allocated to berth {berth.berth_id} "
                  f"(waited {waiting_time:.1f} hours)")
            
            # Process containers
            yield from self.container_handler.process_ship(ship, berth)
            
            # Release berth
            self.berth_manager.release_berth(berth_id)
            
            self.ships_processed += 1
            self.metrics['ships_processed'] += 1
            
            print(f"Time {self.env.now:.1f}: Ship {ship.ship_id} departed from berth {berth.berth_id}")
            
        except Exception as e:
            print(f"Error processing ship {ship.ship_id}: {e}")
            
    def _generate_random_ship(self, ship_id: str) -> Ship:
        """Generate a random ship with realistic characteristics
        
        Uses probability-based ship type selection and realistic size distributions
        based on Hong Kong Port's actual vessel characteristics.
        
        Args:
            ship_id: Unique identifier for the ship
            
        Returns:
            Ship object with random but realistic properties
        """
        # Select ship type based on arrival probabilities
        ship_types = list(SHIP_TYPES.keys())
        probabilities = [SHIP_TYPES[ship_type]['arrival_probability'] for ship_type in ship_types]
        ship_type = random.choices(ship_types, weights=probabilities)[0]
        
        ship_config = SHIP_TYPES[ship_type]
        
        # Generate realistic size from typical sizes
        if 'typical_sizes' in ship_config:
            size_teu = random.choice(ship_config['typical_sizes'])
        else:
            size_teu = random.randint(ship_config['min_size'], ship_config['max_size'])
        
        # Generate container counts based on ship type and size
        if ship_type == 'container':
            # Container ships: higher container counts, proportional to size
            base_containers = size_teu // 50  # Rough containers per TEU capacity
            containers_to_unload = random.randint(int(base_containers * 0.3), int(base_containers * 0.8))
            containers_to_load = random.randint(int(base_containers * 0.2), int(base_containers * 0.7))
        elif ship_type == 'bulk':
            # Bulk carriers: fewer containers, more bulk cargo
            containers_to_unload = random.randint(10, 100)
            containers_to_load = random.randint(5, 80)
        else:  # mixed
            # Mixed cargo: moderate container counts
            base_containers = size_teu // 80  # Lower container density for mixed cargo
            containers_to_unload = random.randint(int(base_containers * 0.2), int(base_containers * 0.6))
            containers_to_load = random.randint(int(base_containers * 0.1), int(base_containers * 0.5))
            
        return Ship(
            ship_id=ship_id,
            name=f"Vessel_{ship_id}",
            ship_type=ship_type,
            size_teu=size_teu,
            containers_to_unload=containers_to_unload,
            containers_to_load=containers_to_load,
            arrival_time=self.env.now
        )
        
    def _generate_final_report(self) -> Dict:
        """Generate comprehensive simulation report
        
        Returns:
            Dictionary containing all simulation metrics and statistics
        """
        simulation_duration = self.metrics['simulation_end_time'] - self.metrics['simulation_start_time']
        
        # Calculate average waiting time
        avg_waiting_time = (
            self.metrics['total_waiting_time'] / self.metrics['ships_processed']
            if self.metrics['ships_processed'] > 0 else 0
        )
        
        # Get component statistics
        berth_stats = self.berth_manager.get_berth_statistics()
        container_stats = self.container_handler.get_processing_statistics()
        
        report = {
            'simulation_summary': {
                'duration': round(simulation_duration, 2),
                'ships_arrived': self.metrics['ships_arrived'],
                'ships_processed': self.metrics['ships_processed'],
                'average_waiting_time': round(avg_waiting_time, 2),
                'throughput_rate': round(self.metrics['ships_processed'] / simulation_duration, 2) if simulation_duration > 0 else 0
            },
            'berth_statistics': berth_stats,
            'container_statistics': container_stats,
            'performance_metrics': {
                'berth_utilization': self._calculate_berth_utilization(),
                'queue_efficiency': self._calculate_queue_efficiency(),
                'processing_efficiency': self._calculate_processing_efficiency()
            }
        }
        
        return report
        
    def _calculate_berth_utilization(self) -> float:
        """Calculate overall berth utilization percentage"""
        berth_stats = self.berth_manager.get_berth_statistics()
        if not berth_stats or 'total_berths' not in berth_stats:
            return 0.0
            
        total_berths = berth_stats['total_berths']
        occupied_berths = berth_stats.get('occupied_berths', 0)
        
        return round((occupied_berths / total_berths) * 100, 2) if total_berths > 0 else 0.0
        
    def _calculate_queue_efficiency(self) -> float:
        """Calculate queue processing efficiency"""
        if self.metrics['ships_arrived'] == 0:
            return 100.0
            
        processed_ratio = self.metrics['ships_processed'] / self.metrics['ships_arrived']
        return round(processed_ratio * 100, 2)
        
    def _calculate_processing_efficiency(self) -> float:
        """Calculate container processing efficiency"""
        container_stats = self.container_handler.get_processing_statistics()
        
        if container_stats['total_operations'] == 0:
            return 0.0
            
        # Efficiency based on processing time vs theoretical minimum
        avg_time = container_stats['average_processing_time']
        theoretical_min = 0.5  # Theoretical minimum processing time
        
        efficiency = (theoretical_min / avg_time) * 100 if avg_time > 0 else 0
        return round(min(efficiency, 100), 2)  # Cap at 100%
        
    def get_current_status(self) -> Dict:
        """Get current simulation status and metrics
        
        Returns:
            Dictionary containing current simulation state
        """
        return {
            'current_time': round(self.env.now, 2),
            'running': self.running,
            'ships_in_system': self.metrics['ships_arrived'] - self.metrics['ships_processed'],
            'ships_processed': self.metrics['ships_processed'],
            'berth_status': self.berth_manager.get_berth_statistics(),
            'queue_length': max(0, self.metrics['ships_arrived'] - self.metrics['ships_processed'])
        }
        
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.env = simpy.Environment()
        self.running = False
        self.ships_processed = 0
        self.total_ships_generated = 0
        
        # Reset all managers using same config logic as constructor
        self.ship_manager = ShipManager(self.env)
        berth_config = self.config.get('berths', BERTH_CONFIGS)
        self.berth_manager = BerthManager(self.env, berth_config)
        self.container_handler = ContainerHandler(self.env)
        
        # Reset metrics
        self.metrics = {
            'ships_arrived': 0,
            'ships_processed': 0,
            'total_waiting_time': 0,
            'simulation_start_time': 0,
            'simulation_end_time': 0
        }
        
        print("Simulation reset to initial state")