"""Peak Season Capacity Optimization Module

This module implements AI-driven optimization algorithms specifically designed for
peak season operations at Hong Kong Port. It extends the existing optimization
framework with advanced algorithms for handling high-volume traffic periods.

Key Features:
- Dynamic berth allocation with priority-based scheduling
- AI-driven capacity optimization using genetic algorithms
- Real-time adjustment based on queue length and waiting times
- Integration with strategic simulation scenarios
- Business metrics calculation for executive reporting

Approach: Uses a hybrid optimization strategy combining heuristic algorithms
with machine learning-based predictions to maximize throughput during peak seasons.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from copy import deepcopy
import random
from enum import Enum

# Import existing optimization modules
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir / 'ai'))
sys.path.insert(0, str(parent_dir))

try:
    from optimization import BerthAllocationOptimizer, Ship, Berth, OptimizationResult
    from strategic_simulations import StrategicScenarioParameters, StrategicBusinessMetrics
except ImportError as e:
    logging.warning(f"Import warning: {e}")
    # Define minimal classes for standalone operation
    @dataclass
    class Ship:
        id: str
        arrival_time: datetime
        ship_type: str
        size: float
        priority: int = 1
        estimated_service_time: float = 0.0
        containers_to_load: int = 0
        containers_to_unload: int = 0
    
    @dataclass
    class Berth:
        id: str
        capacity: float
        crane_count: int
        suitable_ship_types: List[str]
        is_available: bool = True
        current_ship: Optional[str] = None
        available_from: Optional[datetime] = None

logger = logging.getLogger(__name__)

class PeakSeasonStrategy(Enum):
    """Peak season optimization strategies"""
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_WAITING_TIME = "minimize_waiting_time"
    BALANCE_EFFICIENCY = "balance_efficiency"
    PRIORITY_BASED = "priority_based"
    DYNAMIC_ALLOCATION = "dynamic_allocation"

@dataclass
class PeakSeasonMetrics:
    """Metrics specific to peak season operations"""
    total_ships_processed: int
    average_waiting_time: float
    peak_queue_length: int
    berth_utilization_rate: float
    container_throughput: int
    revenue_per_hour: float
    customer_satisfaction_score: float
    operational_efficiency: float
    cost_per_container: float
    time_to_clear_queue: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting"""
        return asdict(self)

@dataclass
class OptimizationConfiguration:
    """Configuration for peak season optimization"""
    strategy: PeakSeasonStrategy
    max_iterations: int = 1000
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    convergence_threshold: float = 0.001
    priority_weight: float = 2.0
    efficiency_weight: float = 1.5
    cost_weight: float = 1.0
    enable_dynamic_adjustment: bool = True
    queue_threshold: int = 10
    waiting_time_threshold: float = 4.0  # hours

class PeakSeasonOptimizer:
    """Advanced optimizer for peak season capacity management"""
    
    def __init__(self, config: OptimizationConfiguration = None):
        """Initialize the peak season optimizer
        
        Args:
            config: Optimization configuration parameters
        """
        self.config = config or OptimizationConfiguration(
            strategy=PeakSeasonStrategy.DYNAMIC_ALLOCATION
        )
        self.ships: List[Ship] = []
        self.berths: List[Berth] = []
        self.current_time = datetime.now()
        self.optimization_history: List[Dict] = []
        
        logger.info(f"Initialized PeakSeasonOptimizer with strategy: {self.config.strategy.value}")
    
    def add_ships(self, ships: List[Ship]) -> None:
        """Add multiple ships to the optimization queue"""
        self.ships.extend(ships)
        logger.info(f"Added {len(ships)} ships to optimization queue")
    
    def add_berths(self, berths: List[Berth]) -> None:
        """Add multiple berths to the available berths"""
        self.berths.extend(berths)
        logger.info(f"Added {len(berths)} berths to optimization")
    
    def calculate_priority_score(self, ship: Ship) -> float:
        """Calculate dynamic priority score for a ship during peak season
        
        Args:
            ship: Ship to calculate priority for
            
        Returns:
            Priority score (higher = more priority)
        """
        base_priority = ship.priority * self.config.priority_weight
        
        # Time-based urgency (ships waiting longer get higher priority)
        waiting_time = (self.current_time - ship.arrival_time).total_seconds() / 3600
        urgency_score = min(waiting_time * 0.5, 5.0)  # Cap at 5 points
        
        # Size efficiency (larger ships get slight priority for efficiency)
        size_score = min(ship.size / 10000, 2.0)  # Normalize and cap
        
        # Container volume priority
        container_score = (ship.containers_to_load + ship.containers_to_unload) / 1000
        
        total_score = base_priority + urgency_score + size_score + container_score
        return total_score
    
    def estimate_service_time_advanced(self, ship: Ship, berth: Berth) -> float:
        """Advanced service time estimation for peak season conditions
        
        Args:
            ship: Ship to estimate service time for
            berth: Berth where ship will be serviced
            
        Returns:
            Estimated service time in hours
        """
        # Base calculation from existing optimizer
        base_time = 2.0
        
        # Container handling with peak season efficiency factors
        total_containers = ship.containers_to_load + ship.containers_to_unload
        if total_containers > 0:
            # Peak season: reduced efficiency due to congestion
            peak_efficiency_factor = 0.85  # 15% reduction during peak
            containers_per_hour_per_crane = 30 * peak_efficiency_factor
            container_time = total_containers / (berth.crane_count * containers_per_hour_per_crane)
        else:
            container_time = 0
        
        # Ship size and type factors
        size_factor = max(1.0, ship.size / 1000)
        type_factors = {
            'container': 1.0,
            'bulk': 1.5,
            'tanker': 1.3,
            'general': 1.2,
            'passenger': 0.8
        }
        type_factor = type_factors.get(ship.ship_type.lower(), 1.0)
        
        # Peak season congestion factor
        congestion_factor = 1.2  # 20% increase due to peak season congestion
        
        estimated_time = (base_time + container_time * size_factor * type_factor) * congestion_factor
        return max(estimated_time, 0.5)
    
    def genetic_algorithm_optimization(self) -> Dict[str, str]:
        """Use genetic algorithm for optimal berth allocation
        
        Returns:
            Dictionary mapping ship IDs to berth IDs
        """
        if not self.ships or not self.berths:
            return {}
        
        logger.info("Starting genetic algorithm optimization")
        
        # Initialize population
        population = self._create_initial_population()
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(self.config.max_iterations):
            # Evaluate fitness for each solution
            fitness_scores = [self._evaluate_fitness(solution) for solution in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()
            
            # Check convergence
            if generation > 10 and abs(best_fitness - np.mean(fitness_scores)) < self.config.convergence_threshold:
                logger.info(f"Converged at generation {generation}")
                break
            
            # Create next generation
            population = self._create_next_generation(population, fitness_scores)
        
        logger.info(f"Genetic algorithm completed with fitness: {best_fitness:.3f}")
        return self._solution_to_assignment(best_solution)
    
    def _create_initial_population(self) -> List[List[int]]:
        """Create initial population for genetic algorithm"""
        population = []
        
        for _ in range(self.config.population_size):
            # Each solution is a list where index=ship, value=berth_index
            solution = []
            for ship in self.ships:
                # Find suitable berths for this ship
                suitable_berths = [i for i, berth in enumerate(self.berths) 
                                 if self._is_berth_suitable(ship, berth)]
                if suitable_berths:
                    solution.append(random.choice(suitable_berths))
                else:
                    solution.append(0)  # Default to first berth
            population.append(solution)
        
        return population
    
    def _is_berth_suitable(self, ship: Ship, berth: Berth) -> bool:
        """Check if berth is suitable for ship"""
        if ship.size > berth.capacity:
            return False
        if berth.suitable_ship_types and ship.ship_type not in berth.suitable_ship_types:
            return False
        return True
    
    def _evaluate_fitness(self, solution: List[int]) -> float:
        """Evaluate fitness of a solution
        
        Args:
            solution: List mapping ship index to berth index
            
        Returns:
            Fitness score (higher is better)
        """
        total_waiting_time = 0
        total_revenue = 0
        berth_utilization = {i: 0 for i in range(len(self.berths))}
        
        # Simulate the schedule
        berth_schedules = {i: [] for i in range(len(self.berths))}
        
        for ship_idx, berth_idx in enumerate(solution):
            if ship_idx >= len(self.ships) or berth_idx >= len(self.berths):
                continue
                
            ship = self.ships[ship_idx]
            berth = self.berths[berth_idx]
            
            if not self._is_berth_suitable(ship, berth):
                total_waiting_time += 24  # Heavy penalty for unsuitable assignment
                continue
            
            # Calculate when ship can start service
            service_time = self.estimate_service_time_advanced(ship, berth)
            
            if not berth_schedules[berth_idx]:
                start_time = max(ship.arrival_time, self.current_time)
            else:
                last_end_time = berth_schedules[berth_idx][-1]['end_time']
                start_time = max(ship.arrival_time, last_end_time)
            
            end_time = start_time + timedelta(hours=service_time)
            
            # Calculate waiting time
            waiting_time = (start_time - ship.arrival_time).total_seconds() / 3600
            total_waiting_time += max(0, waiting_time)
            
            # Add to berth schedule
            berth_schedules[berth_idx].append({
                'ship_id': ship.id,
                'start_time': start_time,
                'end_time': end_time,
                'service_time': service_time
            })
            
            # Calculate revenue (simplified)
            container_revenue = (ship.containers_to_load + ship.containers_to_unload) * 50  # $50 per container
            total_revenue += container_revenue
            
            # Update berth utilization
            berth_utilization[berth_idx] += service_time
        
        # Calculate fitness components
        avg_waiting_time = total_waiting_time / len(self.ships) if self.ships else 0
        avg_berth_utilization = np.mean(list(berth_utilization.values()))
        
        # Fitness function (higher is better)
        fitness = (
            total_revenue * 0.4 -  # Maximize revenue
            avg_waiting_time * 1000 * self.config.efficiency_weight -  # Minimize waiting
            (24 - avg_berth_utilization) * 100  # Maximize utilization
        )
        
        return fitness
    
    def _create_next_generation(self, population: List[List[int]], fitness_scores: List[float]) -> List[List[int]]:
        """Create next generation using selection, crossover, and mutation"""
        next_generation = []
        
        # Keep best solutions (elitism)
        elite_count = max(1, self.config.population_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            next_generation.append(population[idx].copy())
        
        # Generate rest through crossover and mutation
        while len(next_generation) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)
            
            next_generation.extend([child1, child2])
        
        return next_generation[:self.config.population_size]
    
    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """Tournament selection for genetic algorithm"""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Single-point crossover"""
        if len(parent1) <= 1:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def _mutate(self, solution: List[int]) -> List[int]:
        """Mutate solution by randomly changing berth assignments"""
        mutated = solution.copy()
        for i in range(len(mutated)):
            if random.random() < 0.1:  # 10% chance to mutate each gene
                if i < len(self.ships):
                    ship = self.ships[i]
                    suitable_berths = [j for j, berth in enumerate(self.berths) 
                                     if self._is_berth_suitable(ship, berth)]
                    if suitable_berths:
                        mutated[i] = random.choice(suitable_berths)
        return mutated
    
    def _solution_to_assignment(self, solution: List[int]) -> Dict[str, str]:
        """Convert genetic algorithm solution to ship-berth assignment"""
        assignment = {}
        for ship_idx, berth_idx in enumerate(solution):
            if ship_idx < len(self.ships) and berth_idx < len(self.berths):
                assignment[self.ships[ship_idx].id] = self.berths[berth_idx].id
        return assignment
    
    def optimize_peak_season(self, current_time: datetime = None) -> Dict[str, Any]:
        """Main optimization method for peak season operations
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Comprehensive optimization results
        """
        if current_time:
            self.current_time = current_time
        
        logger.info(f"Starting peak season optimization for {len(self.ships)} ships")
        
        # Choose optimization strategy
        if self.config.strategy == PeakSeasonStrategy.DYNAMIC_ALLOCATION:
            assignments = self.genetic_algorithm_optimization()
        else:
            # Fallback to priority-based allocation
            assignments = self._priority_based_allocation()
        
        # Calculate comprehensive metrics
        metrics = self._calculate_peak_season_metrics(assignments)
        
        # Generate detailed schedule
        schedule = self._generate_detailed_schedule(assignments)
        
        # Store optimization history
        optimization_result = {
            'timestamp': self.current_time,
            'strategy': self.config.strategy.value,
            'assignments': assignments,
            'metrics': metrics,
            'schedule': schedule,
            'ships_count': len(self.ships),
            'berths_count': len(self.berths)
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"Peak season optimization completed. Average waiting time: {metrics.average_waiting_time:.2f}h")
        
        return optimization_result
    
    def _priority_based_allocation(self) -> Dict[str, str]:
        """Fallback priority-based allocation algorithm"""
        assignments = {}
        berth_availability = {berth.id: self.current_time for berth in self.berths}
        
        # Sort ships by priority score
        sorted_ships = sorted(self.ships, key=self.calculate_priority_score, reverse=True)
        
        for ship in sorted_ships:
            best_berth = None
            earliest_time = None
            
            for berth in self.berths:
                if self._is_berth_suitable(ship, berth):
                    available_time = berth_availability[berth.id]
                    start_time = max(ship.arrival_time, available_time)
                    
                    if earliest_time is None or start_time < earliest_time:
                        earliest_time = start_time
                        best_berth = berth
            
            if best_berth:
                assignments[ship.id] = best_berth.id
                service_time = self.estimate_service_time_advanced(ship, best_berth)
                berth_availability[best_berth.id] = earliest_time + timedelta(hours=service_time)
        
        return assignments
    
    def _calculate_peak_season_metrics(self, assignments: Dict[str, str]) -> PeakSeasonMetrics:
        """Calculate comprehensive metrics for peak season operations"""
        total_waiting_time = 0
        total_containers = 0
        total_revenue = 0
        berth_usage = {berth.id: 0 for berth in self.berths}
        
        # Simulate the assignments to calculate metrics
        berth_schedules = {berth.id: [] for berth in self.berths}
        
        for ship in self.ships:
            if ship.id not in assignments:
                continue
                
            berth_id = assignments[ship.id]
            berth = next((b for b in self.berths if b.id == berth_id), None)
            if not berth:
                continue
            
            # Calculate service details
            service_time = self.estimate_service_time_advanced(ship, berth)
            
            if not berth_schedules[berth_id]:
                start_time = max(ship.arrival_time, self.current_time)
            else:
                last_end = berth_schedules[berth_id][-1]['end_time']
                start_time = max(ship.arrival_time, last_end)
            
            end_time = start_time + timedelta(hours=service_time)
            waiting_time = max(0, (start_time - ship.arrival_time).total_seconds() / 3600)
            
            # Update metrics
            total_waiting_time += waiting_time
            total_containers += ship.containers_to_load + ship.containers_to_unload
            total_revenue += total_containers * 50  # $50 per container
            berth_usage[berth_id] += service_time
            
            berth_schedules[berth_id].append({
                'ship_id': ship.id,
                'start_time': start_time,
                'end_time': end_time,
                'waiting_time': waiting_time
            })
        
        # Calculate final metrics
        avg_waiting_time = total_waiting_time / len(self.ships) if self.ships else 0
        avg_berth_utilization = np.mean(list(berth_usage.values())) / 24 * 100  # Percentage
        
        # Calculate queue metrics
        max_queue = 0
        current_queue = 0
        for ship in sorted(self.ships, key=lambda s: s.arrival_time):
            if ship.id in assignments:
                berth_id = assignments[ship.id]
                # Simplified queue calculation
                current_queue = max(0, current_queue - 1)
                if current_queue > max_queue:
                    max_queue = current_queue
                current_queue += 1
        
        return PeakSeasonMetrics(
            total_ships_processed=len([s for s in self.ships if s.id in assignments]),
            average_waiting_time=avg_waiting_time,
            peak_queue_length=max_queue,
            berth_utilization_rate=avg_berth_utilization,
            container_throughput=total_containers,
            revenue_per_hour=total_revenue / 24 if total_revenue > 0 else 0,
            customer_satisfaction_score=max(0, 100 - avg_waiting_time * 10),  # Simplified
            operational_efficiency=min(100, avg_berth_utilization),
            cost_per_container=200 - (avg_berth_utilization * 2),  # Simplified cost model
            time_to_clear_queue=max_queue * 2  # Simplified estimate
        )
    
    def _generate_detailed_schedule(self, assignments: Dict[str, str]) -> List[Dict]:
        """Generate detailed schedule from assignments"""
        schedule = []
        berth_schedules = {berth.id: [] for berth in self.berths}
        
        for ship in sorted(self.ships, key=lambda s: s.arrival_time):
            if ship.id not in assignments:
                continue
                
            berth_id = assignments[ship.id]
            berth = next((b for b in self.berths if b.id == berth_id), None)
            if not berth:
                continue
            
            service_time = self.estimate_service_time_advanced(ship, berth)
            
            if not berth_schedules[berth_id]:
                start_time = max(ship.arrival_time, self.current_time)
            else:
                last_end = berth_schedules[berth_id][-1]['end_time']
                start_time = max(ship.arrival_time, last_end)
            
            end_time = start_time + timedelta(hours=service_time)
            waiting_time = max(0, (start_time - ship.arrival_time).total_seconds() / 3600)
            
            schedule_entry = {
                'ship_id': ship.id,
                'ship_type': ship.ship_type,
                'ship_size': ship.size,
                'berth_id': berth_id,
                'arrival_time': ship.arrival_time,
                'start_time': start_time,
                'end_time': end_time,
                'service_time': service_time,
                'waiting_time': waiting_time,
                'containers': ship.containers_to_load + ship.containers_to_unload,
                'priority': ship.priority
            }
            
            schedule.append(schedule_entry)
            berth_schedules[berth_id].append(schedule_entry)
        
        return sorted(schedule, key=lambda x: x['start_time'])
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance"""
        if not self.optimization_history:
            return {"message": "No optimization runs completed yet"}
        
        latest = self.optimization_history[-1]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'latest_strategy': latest['strategy'],
            'latest_metrics': latest['metrics'].to_dict(),
            'ships_processed': latest['ships_count'],
            'berths_utilized': latest['berths_count'],
            'optimization_timestamp': latest['timestamp'],
            'performance_trend': self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> Dict[str, float]:
        """Calculate performance trends across optimization runs"""
        if len(self.optimization_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_runs = self.optimization_history[-5:]  # Last 5 runs
        waiting_times = [run['metrics'].average_waiting_time for run in recent_runs]
        utilizations = [run['metrics'].berth_utilization_rate for run in recent_runs]
        
        return {
            'waiting_time_trend': np.mean(np.diff(waiting_times)) if len(waiting_times) > 1 else 0,
            'utilization_trend': np.mean(np.diff(utilizations)) if len(utilizations) > 1 else 0,
            'average_waiting_time': np.mean(waiting_times),
            'average_utilization': np.mean(utilizations)
        }
    
    def clear(self) -> None:
        """Clear ships and berths for new optimization run"""
        self.ships.clear()
        self.berths.clear()
        logger.info("Cleared ships and berths for new optimization")


def create_sample_peak_season_scenario() -> Dict[str, Any]:
    """Create a sample peak season scenario for testing
    
    Returns:
        Dictionary containing sample scenario data and optimization results
    """
    # Create sample ships for peak season (high volume)
    ships = []
    base_time = datetime.now()
    
    # Generate 20 ships arriving within 6 hours (peak season scenario)
    for i in range(20):
        arrival_time = base_time + timedelta(hours=random.uniform(0, 6))
        ship = Ship(
            id=f"SHIP_{i+1:03d}",
            arrival_time=arrival_time,
            ship_type=random.choice(['container', 'bulk', 'tanker', 'general']),
            size=random.uniform(5000, 25000),  # TEU
            priority=random.choice([1, 1, 1, 2, 2, 3]),  # Weighted towards normal priority
            containers_to_load=random.randint(100, 800),
            containers_to_unload=random.randint(200, 1200)
        )
        ships.append(ship)
    
    # Create sample berths
    berths = [
        Berth(id="BERTH_A1", capacity=30000, crane_count=4, suitable_ship_types=['container', 'general']),
        Berth(id="BERTH_A2", capacity=25000, crane_count=3, suitable_ship_types=['container', 'general']),
        Berth(id="BERTH_B1", capacity=40000, crane_count=2, suitable_ship_types=['bulk', 'tanker']),
        Berth(id="BERTH_B2", capacity=35000, crane_count=2, suitable_ship_types=['bulk', 'tanker']),
        Berth(id="BERTH_C1", capacity=20000, crane_count=3, suitable_ship_types=['container', 'general']),
        Berth(id="BERTH_C2", capacity=45000, crane_count=5, suitable_ship_types=['container', 'bulk', 'tanker', 'general'])
    ]
    
    # Run optimization
    config = OptimizationConfiguration(
        strategy=PeakSeasonStrategy.DYNAMIC_ALLOCATION,
        max_iterations=100,  # Reduced for demo
        population_size=20
    )
    
    optimizer = PeakSeasonOptimizer(config)
    optimizer.add_ships(ships)
    optimizer.add_berths(berths)
    
    result = optimizer.optimize_peak_season(base_time)
    summary = optimizer.get_optimization_summary()
    
    return {
        'scenario_name': 'Peak Season Capacity Test',
        'ships': [{'id': s.id, 'type': s.ship_type, 'size': s.size, 'arrival': s.arrival_time} for s in ships],
        'berths': [{'id': b.id, 'capacity': b.capacity, 'cranes': b.crane_count} for b in berths],
        'optimization_result': result,
        'performance_summary': summary
    }


if __name__ == "__main__":
    # Demo the peak season optimizer
    logging.basicConfig(level=logging.INFO)
    
    print("Hong Kong Port Digital Twin - Peak Season Capacity Optimization")
    print("=" * 70)
    
    scenario = create_sample_peak_season_scenario()
    
    print(f"\nScenario: {scenario['scenario_name']}")
    print(f"Ships: {len(scenario['ships'])}")
    print(f"Berths: {len(scenario['berths'])}")
    
    metrics = scenario['optimization_result']['metrics']
    print(f"\nOptimization Results:")
    print(f"  Average Waiting Time: {metrics.average_waiting_time:.2f} hours")
    print(f"  Berth Utilization: {metrics.berth_utilization_rate:.1f}%")
    print(f"  Container Throughput: {metrics.container_throughput:,} TEU")
    print(f"  Revenue per Hour: ${metrics.revenue_per_hour:,.2f}")
    print(f"  Customer Satisfaction: {metrics.customer_satisfaction_score:.1f}%")
    print(f"  Peak Queue Length: {metrics.peak_queue_length} ships")
    
    print(f"\nSchedule Summary:")
    schedule = scenario['optimization_result']['schedule']
    for entry in schedule[:5]:  # Show first 5 entries
        print(f"  {entry['ship_id']} -> {entry['berth_id']} | "
              f"Wait: {entry['waiting_time']:.1f}h | "
              f"Service: {entry['service_time']:.1f}h")
    
    if len(schedule) > 5:
        print(f"  ... and {len(schedule) - 5} more ships")