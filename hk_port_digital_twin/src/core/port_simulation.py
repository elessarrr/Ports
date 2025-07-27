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
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import SIMULATION_CONFIG, SHIP_TYPES, BERTH_CONFIGS
from src.core.ship_manager import ShipManager, Ship
from src.core.berth_manager import BerthManager
from src.core.container_handler import ContainerHandler

# AI Optimization imports
from src.ai.optimization import (
    BerthAllocationOptimizer, ResourceAllocationOptimizer,
    Ship as AIShip, Berth as AIBerth
)
from src.ai.decision_support import DecisionSupportEngine


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
        
        # Initialize AI optimization components
        self.ai_optimization_enabled = config.get('ai_optimization', True)
        self.berth_optimizer = BerthAllocationOptimizer()
        self.resource_optimizer = ResourceAllocationOptimizer()
        self.decision_engine = DecisionSupportEngine()
        
        # Ship queue for batch optimization
        self.pending_ships = []
        self.optimization_interval = config.get('optimization_interval', 1.0)  # hours
        
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
            'simulation_end_time': 0,
            'ai_optimizations_performed': 0,
            'optimization_time_saved': 0
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
        
        # Start AI optimization process if enabled
        if self.ai_optimization_enabled:
            self.env.process(self.ai_optimization_process())
        
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
                
                # Add ship to pending queue for AI optimization or process immediately
                if self.ai_optimization_enabled:
                    self.pending_ships.append(ship)
                    print(f"Time {self.env.now:.1f}: Ship {ship.ship_id} added to optimization queue")
                else:
                    # Start ship processing with traditional allocation
                    self.env.process(self._process_ship_traditional(ship))
                
            except Exception as e:
                print(f"Error in ship arrival process: {e}")
                break
                
    def ai_optimization_process(self):
        """Periodic AI optimization process for berth allocation
        
        This process runs at regular intervals to optimize berth allocation
        for all pending ships using AI algorithms.
        """
        while self.running:
            try:
                # Wait for optimization interval
                yield self.env.timeout(self.optimization_interval)
                
                if not self.running or not self.pending_ships:
                    continue
                    
                print(f"Time {self.env.now:.1f}: Running AI optimization for {len(self.pending_ships)} ships")
                
                # Convert ships and berths to AI format
                ai_ships = self._convert_ships_to_ai_format(self.pending_ships)
                ai_berths = self._convert_berths_to_ai_format()
                
                # Run AI optimization
                optimization_start = self.env.now
                optimization_result = self.resource_optimizer.optimize_port_operations(
                    ai_ships, ai_berths, 
                    available_cranes=self.container_handler.get_available_cranes(),
                    current_time=datetime.now()
                )
                
                # Process optimized allocations
                processed_ships = []
                for ship in self.pending_ships:
                    if ship.ship_id in optimization_result['berth_allocation'].ship_berth_assignments:
                        # Start AI-optimized processing
                        self.env.process(self._process_ship_ai_optimized(ship, optimization_result))
                        processed_ships.append(ship)
                
                # Remove processed ships from pending queue
                for ship in processed_ships:
                    self.pending_ships.remove(ship)
                
                self.metrics['ai_optimizations_performed'] += 1
                optimization_time = self.env.now - optimization_start
                self.metrics['optimization_time_saved'] += optimization_result.get('efficiency_metrics', {}).get('total_time_saved', 0)
                
                print(f"Time {self.env.now:.1f}: AI optimization completed in {optimization_time:.2f}h, processed {len(processed_ships)} ships")
                
            except Exception as e:
                print(f"Error in AI optimization process: {e}")
                
    def _process_ship_traditional(self, ship: Ship):
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
                
    def _process_ship_ai_optimized(self, ship: Ship, optimization_result: Dict):
        """Process a ship using AI-optimized berth allocation
        
        Args:
            ship: Ship object to process
            optimization_result: Result from AI optimization containing berth assignments
        """
        arrival_time = self.env.now
        
        try:
            # Get AI-assigned berth
            berth_assignments = optimization_result['berth_allocation'].ship_berth_assignments
            assigned_berth_id = berth_assignments.get(ship.ship_id)
            
            if not assigned_berth_id:
                print(f"No berth assigned for ship {ship.ship_id}, falling back to traditional allocation")
                yield from self._process_ship_traditional(ship)
                return
                
            print(f"Time {self.env.now:.1f}: Ship {ship.ship_id} assigned to berth {assigned_berth_id} via AI optimization")
            
            # Wait for berth availability if needed (AI should minimize this)
            while not self.berth_manager.is_berth_available(assigned_berth_id):
                yield self.env.timeout(0.1)  # Wait 6 minutes
                
            # Allocate the AI-assigned berth
            allocation_success = self.berth_manager.allocate_berth(assigned_berth_id, ship.ship_id)
            if not allocation_success:
                print(f"Failed to allocate AI-assigned berth {assigned_berth_id} to ship {ship.ship_id}")
                yield from self._process_ship_traditional(ship)
                return
                
            berth = self.berth_manager.get_berth(assigned_berth_id)
            
            waiting_time = self.env.now - arrival_time
            self.metrics['total_waiting_time'] += waiting_time
            
            print(f"Time {self.env.now:.1f}: Ship {ship.ship_id} allocated to AI-optimized berth {berth.berth_id} "
                  f"(waited {waiting_time:.1f} hours)")
            
            # Use AI-optimized processing time if available
            crane_allocation = optimization_result.get('crane_allocation', {})
            allocated_cranes = crane_allocation.get(ship.ship_id, 0)
            
            if allocated_cranes > 0:
                # Process with AI-optimized crane allocation
                yield from self.container_handler.process_ship_with_cranes(ship, berth, allocated_cranes)
            else:
                # Process with standard container handling
                yield from self.container_handler.process_ship(ship, berth)
            
            # Release berth
            self.berth_manager.release_berth(assigned_berth_id)
            
            self.ships_processed += 1
            self.metrics['ships_processed'] += 1
            
            print(f"Time {self.env.now:.1f}: Ship {ship.ship_id} departed from AI-optimized berth {berth.berth_id}")
            
        except Exception as e:
            print(f"Error processing AI-optimized ship {ship.ship_id}: {e}")
            # Fallback to traditional processing
            yield from self._process_ship_traditional(ship)
            
    def _convert_ships_to_ai_format(self, ships: List[Ship]) -> List[AIShip]:
        """Convert simulation ships to AI optimization format
        
        Args:
            ships: List of simulation Ship objects
            
        Returns:
            List of AI Ship objects
        """
        ai_ships = []
        for ship in ships:
            ai_ship = AIShip(
                id=ship.ship_id,
                arrival_time=datetime.now(),  # Use current time as arrival
                ship_type=ship.ship_type,
                size_teu=ship.size_teu,
                priority=1,  # Default priority
                containers_to_load=getattr(ship, 'containers_to_load', 0),
                containers_to_unload=getattr(ship, 'containers_to_unload', 0)
            )
            ai_ships.append(ai_ship)
        return ai_ships
        
    def _convert_berths_to_ai_format(self) -> List[AIBerth]:
        """Convert simulation berths to AI optimization format
        
        Returns:
            List of AI Berth objects
        """
        ai_berths = []
        berth_stats = self.berth_manager.get_berth_statistics()
        
        # Get all berths from berth manager
        for berth_id in range(1, berth_stats.get('total_berths', 0) + 1):
            berth = self.berth_manager.get_berth(f"BERTH_{berth_id:03d}")
            if berth:
                ai_berth = AIBerth(
                    id=berth.berth_id,
                    capacity_teu=getattr(berth, 'capacity_teu', 5000),
                    crane_count=getattr(berth, 'crane_count', 2),
                    suitable_ship_types=getattr(berth, 'suitable_ship_types', ['container', 'bulk', 'mixed']),
                    is_available=self.berth_manager.is_berth_available(berth.berth_id)
                )
                ai_berths.append(ai_berth)
        return ai_berths
            
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