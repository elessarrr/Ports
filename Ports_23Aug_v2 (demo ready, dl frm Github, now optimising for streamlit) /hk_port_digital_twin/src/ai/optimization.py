# Comments for context:
# This module implements AI-powered optimization algorithms for Hong Kong Port Digital Twin.
# The goal is to minimize ship waiting times, optimize berth allocation, and improve
# container handling efficiency through intelligent scheduling algorithms.
# 
# Approach: Starting with simple heuristic-based optimization that can be enhanced
# with more sophisticated algorithms (genetic algorithms, simulated annealing) later.

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class Ship:
    """Represents a ship in the optimization system"""
    id: str
    arrival_time: datetime
    ship_type: str
    size: float  # TEU capacity or tonnage
    priority: int = 1  # 1=normal, 2=high, 3=urgent
    estimated_service_time: float = 0.0  # hours
    containers_to_load: int = 0
    containers_to_unload: int = 0

@dataclass
class Berth:
    """Represents a berth in the optimization system"""
    id: str
    capacity: float  # max ship size it can handle
    crane_count: int
    suitable_ship_types: List[str]
    is_available: bool = True
    current_ship: Optional[str] = None
    available_from: Optional[datetime] = None

@dataclass
class OptimizationResult:
    """Results from berth allocation optimization"""
    ship_berth_assignments: Dict[str, str]  # ship_id -> berth_id
    total_waiting_time: float
    average_waiting_time: float
    berth_utilization: Dict[str, float]
    optimization_score: float
    schedule: List[Dict]  # detailed schedule with timestamps

class BerthAllocationOptimizer:
    """Optimizes berth allocation to minimize waiting times and maximize efficiency"""
    
    def __init__(self):
        self.ships: List[Ship] = []
        self.berths: List[Berth] = []
        
    def add_ship(self, ship: Ship) -> None:
        """Add a ship to the optimization queue"""
        self.ships.append(ship)
        
    def add_berth(self, berth: Berth) -> None:
        """Add a berth to the available berths"""
        self.berths.append(berth)
        
    def estimate_service_time(self, ship: Ship, berth: Berth) -> float:
        """Estimate service time based on ship size and berth capabilities"""
        # Base service time calculation
        base_time = 2.0  # minimum 2 hours
        
        # Container handling time (assuming 30 TEU per hour per crane)
        container_time = 0
        if ship.containers_to_load > 0 or ship.containers_to_unload > 0:
            total_containers = ship.containers_to_load + ship.containers_to_unload
            containers_per_hour_per_crane = 30
            container_time = total_containers / (berth.crane_count * containers_per_hour_per_crane)
        
        # Ship size factor
        size_factor = max(1.0, ship.size / 1000)  # larger ships take longer
        
        # Ship type factor
        type_factors = {
            'container': 1.0,
            'bulk': 1.5,
            'tanker': 1.3,
            'general': 1.2,
            'passenger': 0.8
        }
        type_factor = type_factors.get(ship.ship_type.lower(), 1.0)
        
        estimated_time = base_time + container_time * size_factor * type_factor
        return max(estimated_time, 0.5)  # minimum 30 minutes
    
    def is_berth_suitable(self, ship: Ship, berth: Berth) -> bool:
        """Check if a berth is suitable for a ship"""
        # Check capacity
        if ship.size > berth.capacity:
            return False
            
        # Check ship type compatibility
        if berth.suitable_ship_types and ship.ship_type not in berth.suitable_ship_types:
            return False
            
        return True
    
    def calculate_waiting_time(self, ship: Ship, berth: Berth, current_time: datetime) -> float:
        """Calculate waiting time for a ship at a specific berth"""
        if berth.is_available:
            # Berth is available now
            if current_time >= ship.arrival_time:
                return 0.0
            else:
                return (ship.arrival_time - current_time).total_seconds() / 3600
        else:
            # Berth is occupied, calculate when it will be free
            if berth.available_from:
                available_time = max(berth.available_from, ship.arrival_time)
                return (available_time - ship.arrival_time).total_seconds() / 3600
            else:
                # Assume 4 hours if no specific time available
                return 4.0
    
    def optimize_berth_allocation(self, current_time: datetime = None) -> OptimizationResult:
        """Main optimization algorithm using First Fit Decreasing with priority"""
        if current_time is None:
            current_time = datetime.now()
            
        logger.info(f"Starting berth allocation optimization for {len(self.ships)} ships and {len(self.berths)} berths")
        
        # Sort ships by priority (high to low) then by arrival time
        sorted_ships = sorted(self.ships, key=lambda s: (-s.priority, s.arrival_time))
        
        # Initialize result tracking
        assignments = {}
        total_waiting_time = 0.0
        schedule = []
        berth_schedules = {berth.id: [] for berth in self.berths}
        
        # Track berth availability
        berth_availability = {}
        for berth in self.berths:
            if berth.is_available:
                berth_availability[berth.id] = current_time
            else:
                berth_availability[berth.id] = berth.available_from or current_time + timedelta(hours=4)
        
        # Assign each ship to the best available berth
        for ship in sorted_ships:
            best_berth = None
            min_waiting_time = float('inf')
            best_start_time = None
            
            # Find the best berth for this ship
            for berth in self.berths:
                if not self.is_berth_suitable(ship, berth):
                    continue
                    
                # Calculate when this ship can start service at this berth
                berth_free_time = berth_availability[berth.id]
                start_time = max(ship.arrival_time, berth_free_time)
                waiting_time = (start_time - ship.arrival_time).total_seconds() / 3600
                
                # Prefer berths with less waiting time
                if waiting_time < min_waiting_time:
                    min_waiting_time = waiting_time
                    best_berth = berth
                    best_start_time = start_time
            
            # Assign ship to best berth
            if best_berth:
                assignments[ship.id] = best_berth.id
                service_time = self.estimate_service_time(ship, best_berth)
                end_time = best_start_time + timedelta(hours=service_time)
                
                # Update berth availability
                berth_availability[best_berth.id] = end_time
                
                # Track waiting time
                total_waiting_time += min_waiting_time
                
                # Add to schedule
                schedule_entry = {
                    'ship_id': ship.id,
                    'berth_id': best_berth.id,
                    'arrival_time': ship.arrival_time,
                    'start_time': best_start_time,
                    'end_time': end_time,
                    'waiting_time': min_waiting_time,
                    'service_time': service_time,
                    'ship_type': ship.ship_type,
                    'priority': ship.priority
                }
                schedule.append(schedule_entry)
                berth_schedules[best_berth.id].append(schedule_entry)
                
                logger.debug(f"Assigned ship {ship.id} to berth {best_berth.id}, waiting time: {min_waiting_time:.2f}h")
            else:
                logger.warning(f"Could not assign ship {ship.id} to any berth")
        
        # Calculate berth utilization
        berth_utilization = {}
        simulation_duration = 24.0  # 24 hours
        
        for berth in self.berths:
            total_service_time = sum(
                entry['service_time'] for entry in berth_schedules[berth.id]
            )
            utilization = min(total_service_time / simulation_duration, 1.0)
            berth_utilization[berth.id] = utilization
        
        # Calculate optimization score (lower is better)
        avg_waiting_time = total_waiting_time / len(self.ships) if self.ships else 0
        avg_utilization = sum(berth_utilization.values()) / len(self.berths) if self.berths else 0
        
        # Score combines waiting time (weight 0.7) and utilization efficiency (weight 0.3)
        optimization_score = (avg_waiting_time * 0.7) + ((1 - avg_utilization) * 10 * 0.3)
        
        result = OptimizationResult(
            ship_berth_assignments=assignments,
            total_waiting_time=total_waiting_time,
            average_waiting_time=avg_waiting_time,
            berth_utilization=berth_utilization,
            optimization_score=optimization_score,
            schedule=schedule
        )
        
        logger.info(f"Optimization complete. Avg waiting time: {avg_waiting_time:.2f}h, Avg utilization: {avg_utilization:.2%}")
        return result
    
    def clear(self) -> None:
        """Clear all ships and berths for new optimization"""
        self.ships.clear()
        self.berths.clear()

class ContainerHandlingScheduler:
    """Optimizes container handling operations and crane scheduling"""
    
    def __init__(self):
        self.crane_efficiency = 30  # containers per hour per crane
        
    def optimize_crane_allocation(self, ships: List[Ship], available_cranes: int) -> Dict[str, int]:
        """Allocate cranes to ships to minimize total handling time"""
        if not ships or available_cranes <= 0:
            return {}
            
        # Calculate total container workload for each ship
        ship_workloads = []
        for ship in ships:
            total_containers = ship.containers_to_load + ship.containers_to_unload
            ship_workloads.append((ship.id, total_containers, ship.priority))
        
        # Sort by priority then by workload (descending)
        ship_workloads.sort(key=lambda x: (-x[2], -x[1]))
        
        # Allocate cranes proportionally to workload, respecting priority
        crane_allocation = {}
        total_workload = sum(workload[1] for workload in ship_workloads)
        
        if total_workload == 0:
            return {ship.id: 0 for ship in ships}
        
        remaining_cranes = available_cranes
        
        for ship_id, workload, priority in ship_workloads:
            if remaining_cranes <= 0:
                crane_allocation[ship_id] = 0
                continue
                
            # Priority ships get at least 1 crane if available
            if priority > 1 and remaining_cranes > 0:
                min_cranes = 1
            else:
                min_cranes = 0
                
            # Calculate proportional allocation
            proportion = workload / total_workload
            allocated_cranes = max(min_cranes, int(available_cranes * proportion))
            allocated_cranes = min(allocated_cranes, remaining_cranes)
            
            crane_allocation[ship_id] = allocated_cranes
            remaining_cranes -= allocated_cranes
        
        # Distribute any remaining cranes to ships with highest workload
        while remaining_cranes > 0:
            # Find ship with highest workload that can use more cranes
            best_ship = None
            max_workload = 0
            
            for ship_id, workload, _ in ship_workloads:
                if workload > max_workload and crane_allocation[ship_id] < 4:  # max 4 cranes per ship
                    max_workload = workload
                    best_ship = ship_id
            
            if best_ship:
                crane_allocation[best_ship] += 1
                remaining_cranes -= 1
            else:
                break
        
        logger.info(f"Crane allocation: {crane_allocation}")
        return crane_allocation
    
    def estimate_handling_time(self, ship: Ship, allocated_cranes: int) -> float:
        """Estimate container handling time given crane allocation"""
        if allocated_cranes <= 0:
            return float('inf')  # Cannot handle without cranes
            
        total_containers = ship.containers_to_load + ship.containers_to_unload
        if total_containers <= 0:
            return 0.5  # minimum handling time
            
        handling_time = total_containers / (allocated_cranes * self.crane_efficiency)
        return max(handling_time, 0.5)  # minimum 30 minutes

class ResourceAllocationOptimizer:
    """Optimizes overall port resource allocation including berths, cranes, and personnel"""
    
    def __init__(self):
        self.berth_optimizer = BerthAllocationOptimizer()
        self.container_scheduler = ContainerHandlingScheduler()
        
    def optimize_port_operations(self, ships: List[Ship], berths: List[Berth], 
                               available_cranes: int, current_time: datetime = None) -> Dict:
        """Comprehensive port optimization combining berth allocation and crane scheduling"""
        
        # Clear and populate optimizers
        self.berth_optimizer.clear()
        for ship in ships:
            self.berth_optimizer.add_ship(ship)
        for berth in berths:
            self.berth_optimizer.add_berth(berth)
        
        # Optimize berth allocation
        berth_result = self.berth_optimizer.optimize_berth_allocation(current_time)
        
        # Optimize crane allocation for assigned ships
        assigned_ships = [ship for ship in ships if ship.id in berth_result.ship_berth_assignments]
        crane_allocation = self.container_scheduler.optimize_crane_allocation(assigned_ships, available_cranes)
        
        # Calculate updated handling times with crane allocation
        updated_schedule = []
        for entry in berth_result.schedule:
            ship = next(s for s in ships if s.id == entry['ship_id'])
            allocated_cranes = crane_allocation.get(ship.id, 0)
            
            if allocated_cranes > 0:
                new_service_time = self.container_scheduler.estimate_handling_time(ship, allocated_cranes)
                entry['allocated_cranes'] = allocated_cranes
                entry['optimized_service_time'] = new_service_time
                entry['time_saved'] = entry['service_time'] - new_service_time
            else:
                entry['allocated_cranes'] = 0
                entry['optimized_service_time'] = entry['service_time']
                entry['time_saved'] = 0
                
            updated_schedule.append(entry)
        
        # Calculate overall efficiency metrics
        total_time_saved = sum(entry['time_saved'] for entry in updated_schedule)
        crane_utilization = sum(crane_allocation.values()) / available_cranes if available_cranes > 0 else 0
        
        return {
            'berth_optimization': berth_result,
            'crane_allocation': crane_allocation,
            'optimized_schedule': updated_schedule,
            'efficiency_metrics': {
                'total_time_saved': total_time_saved,
                'crane_utilization': crane_utilization,
                'average_waiting_time': berth_result.average_waiting_time,
                'berth_utilization': berth_result.berth_utilization
            }
        }

def create_sample_optimization_scenario() -> Dict:
    """Create a sample optimization scenario for testing and demonstration"""
    
    # Sample ships
    ships = [
        Ship("SHIP001", datetime.now(), "container", 2000, 2, 0, 150, 100),
        Ship("SHIP002", datetime.now() + timedelta(hours=1), "bulk", 5000, 1, 0, 0, 0),
        Ship("SHIP003", datetime.now() + timedelta(hours=2), "container", 1500, 3, 0, 200, 50),
        Ship("SHIP004", datetime.now() + timedelta(hours=3), "tanker", 3000, 1, 0, 0, 0),
        Ship("SHIP005", datetime.now() + timedelta(hours=4), "container", 1800, 1, 0, 120, 80)
    ]
    
    # Sample berths
    berths = [
        Berth("BERTH_A", 6000, 3, ["container", "general"], True),
        Berth("BERTH_B", 8000, 2, ["bulk", "tanker"], True),
        Berth("BERTH_C", 4000, 4, ["container"], True),
        Berth("BERTH_D", 7000, 2, ["tanker", "bulk", "general"], True)
    ]
    
    # Run optimization
    optimizer = ResourceAllocationOptimizer()
    result = optimizer.optimize_port_operations(ships, berths, available_cranes=10)
    
    return result

if __name__ == "__main__":
    # Demo the optimization system
    logging.basicConfig(level=logging.INFO)
    
    print("Hong Kong Port Digital Twin - AI Optimization Demo")
    print("=" * 50)
    
    result = create_sample_optimization_scenario()
    
    print(f"\nOptimization Results:")
    print(f"Average Waiting Time: {result['efficiency_metrics']['average_waiting_time']:.2f} hours")
    print(f"Total Time Saved: {result['efficiency_metrics']['total_time_saved']:.2f} hours")
    print(f"Crane Utilization: {result['efficiency_metrics']['crane_utilization']:.1%}")
    
    print(f"\nBerth Utilization:")
    for berth_id, utilization in result['efficiency_metrics']['berth_utilization'].items():
        print(f"  {berth_id}: {utilization:.1%}")
    
    print(f"\nCrane Allocation:")
    for ship_id, cranes in result['crane_allocation'].items():
        print(f"  {ship_id}: {cranes} cranes")