"""Multi-Scenario Operational Optimization Simulation

This module implements the Multi-Scenario Operational Optimization Simulation as part of
Week 6 Priority 2B: Advanced Digital Twin Simulation Framework.

Key Features:
- Extract seasonal cargo patterns from 14+ years of container throughput data
- Calculate realistic ship arrival distributions for Peak/Normal/Low seasons
- Integrate BerthAllocationOptimizer with scenario-specific parameters
- Create optimization comparison framework (optimized vs. non-optimized)
- Provide scenario selection interface with optimization toggle
- Generate side-by-side comparison visualizations

This simulation qualifies the dashboard as a true "Digital Twin" by providing
sophisticated operational modeling based on real historical data patterns.
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

class OptimizationObjective(Enum):
    """Enumeration of optimization objectives"""
    MINIMIZE_WAITING_TIME = "minimize_waiting_time"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCE_UTILIZATION = "balance_utilization"

@dataclass
class OptimizationResult:
    """Results from optimization analysis"""
    scenario_name: str
    optimization_enabled: bool
    total_waiting_time: float
    average_waiting_time: float
    berth_utilization: float
    throughput: int
    cost_efficiency: float
    optimization_improvement: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

# Import existing modules
import sys
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
project_root = parent_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'ai'))
sys.path.insert(0, str(parent_dir / 'utils'))
sys.path.insert(0, str(current_dir))

try:
    from optimization import BerthAllocationOptimizer, Ship, Berth, OptimizationResult
    from data_loader import load_port_cargo_statistics, get_time_series_data
    from scenario_parameters import (
        ScenarioParameters, ALL_SCENARIOS, get_scenario_parameters,
        PEAK_SEASON_PARAMETERS, NORMAL_OPERATIONS_PARAMETERS, LOW_SEASON_PARAMETERS
    )
    
    try:
        from scenario_manager import ScenarioManager
    except ImportError:
        ScenarioManager = None
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Python path: {sys.path[:5]}")
    raise

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ScenarioComparisonResult:
    """Results from comparing optimized vs non-optimized scenarios."""
    scenario_name: str
    scenario_description: str
    
    # Optimized results
    optimized_result: OptimizationResult
    optimized_total_waiting_time: float
    optimized_avg_waiting_time: float
    optimized_berth_utilization: Dict[str, float]
    optimized_score: float
    
    # Non-optimized (FIFO) results
    non_optimized_result: OptimizationResult
    non_optimized_total_waiting_time: float
    non_optimized_avg_waiting_time: float
    non_optimized_berth_utilization: Dict[str, float]
    non_optimized_score: float
    
    # Improvement metrics
    waiting_time_improvement: float  # Percentage improvement
    utilization_improvement: float   # Average utilization improvement
    efficiency_gain: float          # Overall efficiency gain
    
    # Scenario-specific metrics
    ships_processed: int
    simulation_duration_hours: float
    cargo_volume_processed: float
    
    # Timestamp
    generated_at: datetime

@dataclass
class SeasonalCargoPattern:
    """Seasonal cargo patterns extracted from historical data."""
    season_name: str
    months: List[int]
    avg_monthly_throughput: float
    peak_month_multiplier: float
    ship_arrival_rate: float  # Ships per day
    avg_ship_size: float     # TEU
    container_volume_ratio: float  # Containers per ship
    
class MultiScenarioOptimizer:
    """Multi-Scenario Operational Optimization Simulation Engine.
    
    This class implements sophisticated operational modeling by:
    1. Extracting seasonal patterns from historical cargo data
    2. Generating realistic ship arrival distributions
    3. Running optimization comparisons across scenarios
    4. Providing comprehensive analysis and visualization data
    """
    
    def __init__(self, use_historical_data: bool = True):
        """Initialize the multi-scenario optimizer.
        
        Args:
            use_historical_data: Whether to use real historical data for patterns
        """
        self.use_historical_data = use_historical_data
        self.seasonal_patterns: Dict[str, SeasonalCargoPattern] = {}
        self.cargo_data: Optional[pd.DataFrame] = None
        
        # Initialize scenario manager if available
        if ScenarioManager is not None:
            self.scenario_manager = ScenarioManager()
        else:
            self.scenario_manager = None
            logger.warning("ScenarioManager not available, using fallback mode")
        
        # Initialize optimizers
        self.optimized_engine = BerthAllocationOptimizer()
        self.non_optimized_engine = BerthAllocationOptimizer()
        
        # Load and analyze historical data
        if self.use_historical_data:
            self._load_historical_patterns()
        else:
            self._create_synthetic_patterns()
            
        logger.info(f"MultiScenarioOptimizer initialized with {'historical' if use_historical_data else 'synthetic'} data")
    
    def _load_historical_patterns(self) -> None:
        """Load and analyze historical cargo data to extract seasonal patterns."""
        try:
            # Load cargo statistics
            self.cargo_data = load_port_cargo_statistics()
            if self.cargo_data is not None and not self.cargo_data.empty:
                # Analyze seasonal patterns using time series data
                seasonal_analysis = self._analyze_seasonal_patterns(self.cargo_data)
                self._extract_seasonal_patterns(seasonal_analysis)
                logger.info(f"Loaded historical cargo data: {len(self.cargo_data)} records")
            else:
                logger.warning("No historical cargo data available, using synthetic patterns")
                self._create_synthetic_patterns()
        except Exception as e:
            logger.error(f"Error loading historical patterns: {e}")
            self._create_synthetic_patterns()
    
    def _analyze_seasonal_patterns(self, cargo_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns from cargo data.
        
        Args:
            cargo_data: Historical cargo data
            
        Returns:
            Dictionary with seasonal analysis results
        """
        try:
            # Extract monthly statistics
            cargo_data['month'] = pd.to_datetime(cargo_data['date']).dt.month
            monthly_stats = cargo_data.groupby('month').agg({
                'throughput': ['mean', 'std', 'count']
            }).to_dict()
            
            return {
                'monthly_stats': {
                    month: {
                        'avg_throughput': monthly_stats[('throughput', 'mean')].get(month, 0),
                        'std_throughput': monthly_stats[('throughput', 'std')].get(month, 0),
                        'count': monthly_stats[('throughput', 'count')].get(month, 0)
                    }
                    for month in range(1, 13)
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {e}")
            return {'monthly_stats': {}}
    
    def _extract_seasonal_patterns(self, seasonal_analysis: Dict[str, Any]) -> None:
        """Extract seasonal patterns from historical analysis.
        
        Args:
            seasonal_analysis: Results from _analyze_seasonal_patterns function
        """
        try:
            # Extract patterns for each scenario
            monthly_stats = seasonal_analysis.get('monthly_stats', {})
            
            # Peak season pattern (Oct-Dec, Jan)
            peak_months = [10, 11, 12, 1]
            peak_values = [monthly_stats.get(month, {}).get('avg_throughput', 0) 
                          for month in peak_months if month in monthly_stats]
            peak_throughput = np.mean(peak_values) if peak_values else 450000
            
            self.seasonal_patterns['peak'] = SeasonalCargoPattern(
                season_name="Peak Season",
                months=peak_months,
                avg_monthly_throughput=peak_throughput,
                peak_month_multiplier=1.4,
                ship_arrival_rate=self._calculate_ship_arrival_rate(peak_throughput, 1.4),
                avg_ship_size=self._estimate_avg_ship_size(peak_throughput, 'peak'),
                container_volume_ratio=1.3
            )
            
            # Normal season pattern (Mar-Aug)
            normal_months = [3, 4, 5, 6, 7, 8]
            normal_values = [monthly_stats.get(month, {}).get('avg_throughput', 0) 
                            for month in normal_months if month in monthly_stats]
            normal_throughput = np.mean(normal_values) if normal_values else 350000
            
            self.seasonal_patterns['normal'] = SeasonalCargoPattern(
                season_name="Normal Operations",
                months=normal_months,
                avg_monthly_throughput=normal_throughput,
                peak_month_multiplier=1.0,
                ship_arrival_rate=self._calculate_ship_arrival_rate(normal_throughput, 1.0),
                avg_ship_size=self._estimate_avg_ship_size(normal_throughput, 'normal'),
                container_volume_ratio=1.0
            )
            
            # Low season pattern (May-Aug with reduced activity)
            low_months = [5, 6, 7, 8]
            low_values = [monthly_stats.get(month, {}).get('avg_throughput', 0) 
                         for month in low_months if month in monthly_stats]
            low_throughput = (np.mean(low_values) if low_values else 350000) * 0.7
            
            self.seasonal_patterns['low'] = SeasonalCargoPattern(
                season_name="Low Season",
                months=low_months,
                avg_monthly_throughput=low_throughput,
                peak_month_multiplier=0.7,
                ship_arrival_rate=self._calculate_ship_arrival_rate(low_throughput, 0.7),
                avg_ship_size=self._estimate_avg_ship_size(low_throughput, 'low'),
                container_volume_ratio=0.8
            )
            
            logger.info(f"Extracted seasonal patterns for {len(self.seasonal_patterns)} scenarios")
            
        except Exception as e:
            logger.error(f"Error extracting seasonal patterns: {e}")
            self._create_synthetic_patterns()
    
    def _calculate_ship_arrival_rate(self, monthly_throughput: float, multiplier: float) -> float:
        """Calculate ship arrival rate based on monthly throughput.
        
        Args:
            monthly_throughput: Average monthly container throughput
            multiplier: Seasonal multiplier
            
        Returns:
            Ships per day arrival rate
        """
        # Estimate based on average ship capacity and monthly throughput
        avg_ship_capacity = 2500  # TEU (typical container ship)
        monthly_ships = (monthly_throughput * multiplier) / avg_ship_capacity
        daily_arrival_rate = monthly_ships / 30  # Convert to daily rate
        return max(daily_arrival_rate, 0.5)  # Minimum 0.5 ships per day
    
    def _estimate_avg_ship_size(self, monthly_throughput: float, scenario: str) -> float:
        """Estimate average ship size based on throughput and scenario.
        
        Args:
            monthly_throughput: Average monthly container throughput
            scenario: Scenario type ('peak', 'normal', 'low')
            
        Returns:
            Average ship size in TEU
        """
        base_size = 2500  # Base ship size in TEU
        
        # Adjust based on scenario
        if scenario == 'peak':
            return base_size * 1.25  # Larger ships during peak
        elif scenario == 'low':
            return base_size * 0.85  # Smaller ships during low season
        else:
            return base_size  # Normal size
    
    def _create_synthetic_patterns(self) -> None:
        """Create synthetic seasonal patterns when historical data is unavailable."""
        # Synthetic patterns based on typical port operations
        self.seasonal_patterns = {
            'peak': SeasonalCargoPattern(
                season_name="Peak Season",
                months=[10, 11, 12, 1],
                avg_monthly_throughput=450000,  # TEU
                peak_month_multiplier=1.4,
                ship_arrival_rate=6.0,  # Ships per day
                avg_ship_size=3125,     # TEU
                container_volume_ratio=1.3
            ),
            'normal': SeasonalCargoPattern(
                season_name="Normal Operations",
                months=[3, 4, 5, 6, 7, 8],
                avg_monthly_throughput=350000,  # TEU
                peak_month_multiplier=1.0,
                ship_arrival_rate=4.5,  # Ships per day
                avg_ship_size=2500,     # TEU
                container_volume_ratio=1.0
            ),
            'low': SeasonalCargoPattern(
                season_name="Low Season",
                months=[5, 6, 7, 8],
                avg_monthly_throughput=245000,  # TEU
                peak_month_multiplier=0.7,
                ship_arrival_rate=3.0,  # Ships per day
                avg_ship_size=2125,     # TEU
                container_volume_ratio=0.8
            )
        }
        logger.info("Created synthetic seasonal patterns")
    
    def generate_scenario_ships(self, scenario_name: str, simulation_hours: int = 168, 
                              current_time: datetime = None) -> List[Ship]:
        """Generate realistic ship arrivals for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario ('peak', 'normal', 'low')
            simulation_hours: Duration of simulation in hours (default: 1 week)
            current_time: Starting time for simulation
            
        Returns:
            List of Ship objects with realistic arrival patterns
        """
        if current_time is None:
            current_time = datetime.now()
        
        scenario_params = get_scenario_parameters(scenario_name)
        seasonal_pattern = self.seasonal_patterns.get(scenario_name)
        
        if not scenario_params or not seasonal_pattern:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        ships = []
        ship_id_counter = 1
        
        # Calculate arrival intervals based on scenario parameters
        base_arrival_rate = seasonal_pattern.ship_arrival_rate  # Ships per day
        adjusted_rate = base_arrival_rate * scenario_params.arrival_rate_multiplier
        avg_interval_hours = 24.0 / adjusted_rate
        
        # Generate ships over the simulation period
        current_sim_time = current_time
        end_time = current_time + timedelta(hours=simulation_hours)
        
        while current_sim_time < end_time:
            # Calculate next arrival time with some randomness
            interval_variation = np.random.exponential(avg_interval_hours)
            next_arrival = current_sim_time + timedelta(hours=interval_variation)
            
            if next_arrival >= end_time:
                break
            
            # Apply peak hour and weekend multipliers
            hour_multiplier = self._get_hour_multiplier(next_arrival, scenario_params)
            if np.random.random() > (1.0 / hour_multiplier):  # Skip some ships during off-peak
                current_sim_time = next_arrival
                continue
            
            # Determine ship type based on scenario distribution
            ship_type = self._select_ship_type(scenario_params.ship_type_distribution)
            
            # Calculate ship size and container load
            base_size = seasonal_pattern.avg_ship_size * scenario_params.average_ship_size_multiplier
            ship_size = np.random.normal(base_size, base_size * 0.2)  # 20% variation
            ship_size = max(ship_size, 1000)  # Minimum size
            
            # Calculate container load based on ship type and scenario
            container_multiplier = scenario_params.container_volume_multipliers.get(ship_type, 1.0)
            containers_to_unload = int(ship_size * 0.7 * container_multiplier)  # 70% of capacity
            containers_to_load = int(ship_size * 0.3 * container_multiplier)    # 30% of capacity
            
            # Determine priority based on ship characteristics
            priority = self._calculate_ship_priority(ship_type, ship_size, scenario_params)
            
            # Create ship object
            ship = Ship(
                id=f"{scenario_name}_{ship_type}_{ship_id_counter:03d}",
                arrival_time=next_arrival,
                ship_type=ship_type,
                size=ship_size,
                priority=priority,
                containers_to_load=containers_to_load,
                containers_to_unload=containers_to_unload
            )
            
            ships.append(ship)
            ship_id_counter += 1
            current_sim_time = next_arrival
        
        logger.info(f"Generated {len(ships)} ships for {scenario_name} scenario over {simulation_hours} hours")
        return ships
    
    def _get_hour_multiplier(self, arrival_time: datetime, scenario_params: ScenarioParameters) -> float:
        """Get hour-based arrival multiplier.
        
        Args:
            arrival_time: Ship arrival time
            scenario_params: Scenario parameters
            
        Returns:
            Multiplier for arrival probability
        """
        hour = arrival_time.hour
        is_weekend = arrival_time.weekday() >= 5
        
        # Peak hours: 8-18 (business hours)
        if 8 <= hour <= 18:
            multiplier = scenario_params.peak_hour_multiplier
        else:
            multiplier = 1.0
        
        # Weekend adjustment
        if is_weekend:
            multiplier *= scenario_params.weekend_multiplier
        
        return multiplier
    
    def _select_ship_type(self, distribution: Dict[str, float]) -> str:
        """Select ship type based on probability distribution.
        
        Args:
            distribution: Ship type probability distribution
            
        Returns:
            Selected ship type
        """
        rand_val = np.random.random()
        cumulative_prob = 0.0
        
        for ship_type, prob in distribution.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return ship_type
        
        # Fallback to container ship
        return 'container'
    
    def _calculate_ship_priority(self, ship_type: str, ship_size: float, 
                               scenario_params: ScenarioParameters) -> int:
        """Calculate ship priority based on type, size, and scenario parameters.
        
        Args:
            ship_type: Type of ship
            ship_size: Ship size in TEU
            scenario_params: Scenario parameters
            
        Returns:
            Priority level (1=normal, 2=high, 3=urgent)
        """
        base_priority = 1
        
        # Large ship bonus
        if ship_size > 4000:  # Large ship threshold
            if scenario_params.large_ship_priority_boost > 1.3:
                base_priority = 2
        
        # Container ship bonus
        if ship_type == 'container' and scenario_params.container_ship_priority_boost > 1.2:
            base_priority = max(base_priority, 2)
        
        # Random urgent priority (5% chance)
        if np.random.random() < 0.05:
            base_priority = 3
        
        return base_priority
    
    def generate_scenario_berths(self, scenario_name: str) -> List[Berth]:
        """Generate berths configured for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            List of Berth objects configured for the scenario
        """
        scenario_params = get_scenario_parameters(scenario_name)
        if not scenario_params:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        # Base berth configuration (Hong Kong Port typical setup)
        base_berths = [
            # Container terminals
            Berth(id="CT1", capacity=8000, crane_count=4, suitable_ship_types=["container", "mixed"]),
            Berth(id="CT2", capacity=8000, crane_count=4, suitable_ship_types=["container", "mixed"]),
            Berth(id="CT3", capacity=6000, crane_count=3, suitable_ship_types=["container", "mixed"]),
            Berth(id="CT4", capacity=6000, crane_count=3, suitable_ship_types=["container", "mixed"]),
            Berth(id="CT5", capacity=10000, crane_count=5, suitable_ship_types=["container"]),
            Berth(id="CT6", capacity=10000, crane_count=5, suitable_ship_types=["container"]),
            
            # Bulk cargo terminals
            Berth(id="BT1", capacity=15000, crane_count=2, suitable_ship_types=["bulk", "mixed"]),
            Berth(id="BT2", capacity=15000, crane_count=2, suitable_ship_types=["bulk", "mixed"]),
            Berth(id="BT3", capacity=12000, crane_count=2, suitable_ship_types=["bulk"]),
            
            # Multi-purpose terminals
            Berth(id="MT1", capacity=7000, crane_count=3, suitable_ship_types=["container", "bulk", "mixed"]),
            Berth(id="MT2", capacity=7000, crane_count=3, suitable_ship_types=["container", "bulk", "mixed"]),
            Berth(id="MT3", capacity=5000, crane_count=2, suitable_ship_types=["container", "bulk", "mixed"])
        ]
        
        # Apply scenario-specific adjustments
        scenario_berths = []
        for berth in base_berths:
            # Create a copy of the berth
            scenario_berth = Berth(
                id=berth.id,
                capacity=berth.capacity,
                crane_count=berth.crane_count,
                suitable_ship_types=berth.suitable_ship_types.copy(),
                is_available=True,
                current_ship=None,
                available_from=None
            )
            
            # Apply availability factor (some berths may be under maintenance)
            if np.random.random() > scenario_params.berth_availability_factor:
                scenario_berth.is_available = False
                logger.debug(f"Berth {berth.id} unavailable due to maintenance in {scenario_name} scenario")
            
            scenario_berths.append(scenario_berth)
        
        available_berths = [b for b in scenario_berths if b.is_available]
        logger.info(f"Generated {len(available_berths)}/{len(scenario_berths)} available berths for {scenario_name} scenario")
        
        return scenario_berths
    
    def run_optimization_comparison(self, scenario_name: str, simulation_hours: int = 168,
                                  current_time: datetime = None) -> ScenarioComparisonResult:
        """Run optimization comparison for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario to simulate
            simulation_hours: Duration of simulation in hours
            current_time: Starting time for simulation
            
        Returns:
            ScenarioComparisonResult with detailed comparison metrics
        """
        if current_time is None:
            current_time = datetime.now()
        
        logger.info(f"Running optimization comparison for {scenario_name} scenario")
        
        # Generate ships and berths for the scenario
        ships = self.generate_scenario_ships(scenario_name, simulation_hours, current_time)
        berths = self.generate_scenario_berths(scenario_name)
        
        # Get scenario parameters
        scenario_params = get_scenario_parameters(scenario_name)
        
        # Run optimized simulation
        optimized_result = self._run_optimized_simulation(ships, berths, scenario_params, current_time)
        
        # Run non-optimized (FIFO) simulation
        non_optimized_result = self._run_fifo_simulation(ships, berths, scenario_params, current_time)
        
        # Calculate improvement metrics
        waiting_time_improvement = (
            (non_optimized_result.total_waiting_time - optimized_result.total_waiting_time) /
            non_optimized_result.total_waiting_time * 100
        ) if non_optimized_result.total_waiting_time > 0 else 0
        
        # Calculate average utilization improvement
        opt_avg_util = np.mean(list(optimized_result.berth_utilization.values()))
        non_opt_avg_util = np.mean(list(non_optimized_result.berth_utilization.values()))
        utilization_improvement = opt_avg_util - non_opt_avg_util
        
        # Calculate overall efficiency gain
        efficiency_gain = (
            (optimized_result.optimization_score - non_optimized_result.optimization_score) /
            non_optimized_result.optimization_score * 100
        ) if non_optimized_result.optimization_score > 0 else 0
        
        # Calculate cargo volume processed
        total_cargo = sum(ship.containers_to_load + ship.containers_to_unload for ship in ships)
        
        # Create comparison result
        comparison_result = ScenarioComparisonResult(
            scenario_name=scenario_params.scenario_name,
            scenario_description=scenario_params.scenario_description,
            
            # Optimized results
            optimized_result=optimized_result,
            optimized_total_waiting_time=optimized_result.total_waiting_time,
            optimized_avg_waiting_time=optimized_result.average_waiting_time,
            optimized_berth_utilization=optimized_result.berth_utilization,
            optimized_score=optimized_result.optimization_score,
            
            # Non-optimized results
            non_optimized_result=non_optimized_result,
            non_optimized_total_waiting_time=non_optimized_result.total_waiting_time,
            non_optimized_avg_waiting_time=non_optimized_result.average_waiting_time,
            non_optimized_berth_utilization=non_optimized_result.berth_utilization,
            non_optimized_score=non_optimized_result.optimization_score,
            
            # Improvement metrics
            waiting_time_improvement=waiting_time_improvement,
            utilization_improvement=utilization_improvement,
            efficiency_gain=efficiency_gain,
            
            # Scenario metrics
            ships_processed=len(ships),
            simulation_duration_hours=simulation_hours,
            cargo_volume_processed=total_cargo,
            
            generated_at=datetime.now()
        )
        
        logger.info(f"Optimization comparison completed for {scenario_name}: "
                   f"{waiting_time_improvement:.1f}% waiting time improvement, "
                   f"{efficiency_gain:.1f}% efficiency gain")
        
        return comparison_result
    
    def _run_optimized_simulation(self, ships: List[Ship], berths: List[Berth],
                                scenario_params: ScenarioParameters, current_time: datetime) -> OptimizationResult:
        """Run optimized berth allocation simulation.
        
        Args:
            ships: List of ships to process
            berths: List of available berths
            scenario_params: Scenario-specific parameters
            current_time: Simulation start time
            
        Returns:
            OptimizationResult from optimized allocation
        """
        # Clear and setup optimizer
        self.optimized_engine.clear()
        
        # Add berths to optimizer
        for berth in berths:
            if berth.is_available:
                self.optimized_engine.add_berth(berth)
        
        # Add ships with scenario-specific adjustments
        for ship in ships:
            # Apply scenario-specific service time adjustments
            adjusted_ship = self._apply_scenario_adjustments(ship, scenario_params)
            self.optimized_engine.add_ship(adjusted_ship)
        
        # Run optimization
        result = self.optimized_engine.optimize_berth_allocation(current_time)
        
        return result
    
    def _run_fifo_simulation(self, ships: List[Ship], berths: List[Berth],
                           scenario_params: ScenarioParameters, current_time: datetime) -> OptimizationResult:
        """Run First-In-First-Out (non-optimized) simulation.
        
        Args:
            ships: List of ships to process
            berths: List of available berths
            scenario_params: Scenario-specific parameters
            current_time: Simulation start time
            
        Returns:
            OptimizationResult from FIFO allocation
        """
        # Sort ships by arrival time (FIFO)
        sorted_ships = sorted(ships, key=lambda s: s.arrival_time)
        available_berths = [b for b in berths if b.is_available]
        
        # Simple FIFO allocation
        ship_berth_assignments = {}
        berth_schedules = {berth.id: [] for berth in available_berths}
        berth_availability = {berth.id: current_time for berth in available_berths}
        
        total_waiting_time = 0.0
        schedule = []
        
        for ship in sorted_ships:
            # Apply scenario adjustments
            adjusted_ship = self._apply_scenario_adjustments(ship, scenario_params)
            
            # Find first available berth that can handle this ship
            assigned_berth = None
            earliest_available_time = None
            
            for berth in available_berths:
                if self.optimized_engine.is_berth_suitable(adjusted_ship, berth):
                    berth_available_time = berth_availability[berth.id]
                    if earliest_available_time is None or berth_available_time < earliest_available_time:
                        earliest_available_time = berth_available_time
                        assigned_berth = berth
            
            if assigned_berth:
                # Calculate service time
                service_time = self.optimized_engine.estimate_service_time(adjusted_ship, assigned_berth)
                
                # Calculate waiting time
                start_time = max(adjusted_ship.arrival_time, earliest_available_time)
                waiting_time = (start_time - adjusted_ship.arrival_time).total_seconds() / 3600.0
                total_waiting_time += waiting_time
                
                # Update berth availability
                end_time = start_time + timedelta(hours=service_time)
                berth_availability[assigned_berth.id] = end_time
                
                # Record assignment
                ship_berth_assignments[adjusted_ship.id] = assigned_berth.id
                
                # Add to schedule
                schedule.append({
                    'ship_id': adjusted_ship.id,
                    'berth_id': assigned_berth.id,
                    'arrival_time': adjusted_ship.arrival_time,
                    'start_time': start_time,
                    'end_time': end_time,
                    'waiting_time': waiting_time,
                    'service_time': service_time
                })
        
        # Calculate berth utilization
        simulation_duration = max((s['end_time'] for s in schedule), default=current_time) - current_time
        simulation_hours = simulation_duration.total_seconds() / 3600.0
        
        berth_utilization = {}
        for berth in available_berths:
            berth_busy_time = sum(
                s['service_time'] for s in schedule if s['berth_id'] == berth.id
            )
            utilization = berth_busy_time / simulation_hours if simulation_hours > 0 else 0
            berth_utilization[berth.id] = min(utilization, 1.0)
        
        # Calculate metrics
        avg_waiting_time = total_waiting_time / len(ships) if ships else 0
        avg_utilization = np.mean(list(berth_utilization.values())) if berth_utilization else 0
        optimization_score = (1.0 - avg_waiting_time / 24.0) * avg_utilization  # Simple score
        
        return OptimizationResult(
            ship_berth_assignments=ship_berth_assignments,
            total_waiting_time=total_waiting_time,
            average_waiting_time=avg_waiting_time,
            berth_utilization=berth_utilization,
            optimization_score=optimization_score,
            schedule=schedule
        )
    
    def _apply_scenario_adjustments(self, ship: Ship, scenario_params: ScenarioParameters) -> Ship:
        """Apply scenario-specific adjustments to ship parameters.
        
        Args:
            ship: Original ship object
            scenario_params: Scenario parameters
            
        Returns:
            Ship object with scenario adjustments applied
        """
        # Create a copy of the ship
        adjusted_ship = Ship(
            id=ship.id,
            arrival_time=ship.arrival_time,
            ship_type=ship.ship_type,
            size=ship.size,
            priority=ship.priority,
            containers_to_load=ship.containers_to_load,
            containers_to_unload=ship.containers_to_unload
        )
        
        # Apply priority adjustments
        if ship.size > 4000 and scenario_params.large_ship_priority_boost > 1.0:
            adjusted_ship.priority = min(adjusted_ship.priority + 1, 3)
        
        if ship.ship_type == 'container' and scenario_params.container_ship_priority_boost > 1.0:
            adjusted_ship.priority = min(adjusted_ship.priority + 1, 3)
        
        return adjusted_ship
    
    def run_all_scenarios_comparison(self, simulation_hours: int = 168,
                                   current_time: datetime = None) -> Dict[str, ScenarioComparisonResult]:
        """Run optimization comparison for all scenarios.
        
        Args:
            simulation_hours: Duration of simulation in hours
            current_time: Starting time for simulation
            
        Returns:
            Dictionary mapping scenario names to comparison results
        """
        if current_time is None:
            current_time = datetime.now()
        
        results = {}
        
        for scenario_name in ['peak', 'normal', 'low']:
            try:
                result = self.run_optimization_comparison(scenario_name, simulation_hours, current_time)
                results[scenario_name] = result
            except Exception as e:
                logger.error(f"Error running comparison for {scenario_name} scenario: {e}")
        
        logger.info(f"Completed optimization comparison for {len(results)} scenarios")
        return results
    
    def get_seasonal_patterns_summary(self) -> Dict[str, Any]:
        """Get summary of extracted seasonal patterns.
        
        Returns:
            Dictionary with seasonal pattern information
        """
        summary = {
            'data_source': 'historical' if self.use_historical_data else 'synthetic',
            'patterns': {},
            'total_scenarios': len(self.seasonal_patterns)
        }
        
        for scenario_name, pattern in self.seasonal_patterns.items():
            summary['patterns'][scenario_name] = {
                'season_name': pattern.season_name,
                'months': pattern.months,
                'avg_monthly_throughput': pattern.avg_monthly_throughput,
                'ship_arrival_rate': pattern.ship_arrival_rate,
                'avg_ship_size': pattern.avg_ship_size,
                'container_volume_ratio': pattern.container_volume_ratio
            }
        
        return summary


def create_sample_multi_scenario_comparison() -> Dict[str, Any]:
    """Create a sample multi-scenario optimization comparison.
    
    Returns:
        Dictionary with sample comparison results
    """
    try:
        # Initialize optimizer
        print("Initializing MultiScenarioOptimizer...")
        optimizer = MultiScenarioOptimizer(use_historical_data=True)
        print(f"Optimizer initialized. Seasonal patterns: {len(optimizer.seasonal_patterns)}")
        
        # Run comparison for all scenarios (shorter simulation for demo)
        print("Running scenario comparisons...")
        results = optimizer.run_all_scenarios_comparison(simulation_hours=72)  # 3 days
        print(f"Scenario comparisons completed. Results: {type(results)}, Length: {len(results) if results else 'None'}")
        
        if not results:
            print("Warning: No results returned from scenario comparison")
            return None
        
        # Create summary
        summary = {
            'seasonal_patterns': optimizer.get_seasonal_patterns_summary(),
            'scenario_comparisons': {},
            'overall_insights': {}
        }
        print("Summary structure created")
    except Exception as e:
        print(f"Error in create_sample_multi_scenario_comparison: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Process results
    total_efficiency_gain = 0
    total_waiting_time_improvement = 0
    scenario_count = 0
    
    for scenario_name, result in results.items():
        summary['scenario_comparisons'][scenario_name] = {
            'scenario_description': result.scenario_description,
            'ships_processed': result.ships_processed,
            'cargo_volume_processed': result.cargo_volume_processed,
            'waiting_time_improvement': result.waiting_time_improvement,
            'utilization_improvement': result.utilization_improvement,
            'efficiency_gain': result.efficiency_gain,
            'optimized_avg_waiting': result.optimized_avg_waiting_time,
            'non_optimized_avg_waiting': result.non_optimized_avg_waiting_time
        }
        
        total_efficiency_gain += result.efficiency_gain
        total_waiting_time_improvement += result.waiting_time_improvement
        scenario_count += 1
    
    # Calculate overall insights
    if scenario_count > 0:
        summary['overall_insights'] = {
            'avg_efficiency_gain': total_efficiency_gain / scenario_count,
            'avg_waiting_time_improvement': total_waiting_time_improvement / scenario_count,
            'scenarios_analyzed': scenario_count,
            'optimization_effectiveness': 'High' if total_efficiency_gain / scenario_count > 15 else 'Moderate'
        }
    
    return summary


if __name__ == "__main__":
    # Demo the multi-scenario optimization
    logging.basicConfig(level=logging.INFO)
    
    print("Hong Kong Port Digital Twin - Multi-Scenario Optimization Demo")
    print("=" * 70)
    
    try:
        demo_results = create_sample_multi_scenario_comparison()
        
        print("\nSeasonal Patterns Analysis:")
        patterns = demo_results['seasonal_patterns']['patterns']
        for scenario, pattern in patterns.items():
            print(f"  {pattern['season_name']}:")
            print(f"    - Months: {pattern['months']}")
            print(f"    - Ship arrival rate: {pattern['ship_arrival_rate']:.1f} ships/day")
            print(f"    - Avg ship size: {pattern['avg_ship_size']:.0f} TEU")
            print(f"    - Monthly throughput: {pattern['avg_monthly_throughput']:,.0f} TEU")
        
        print("\nOptimization Comparison Results:")
        for scenario, comparison in demo_results['scenario_comparisons'].items():
            print(f"\n  {comparison['scenario_description']}:")
            print(f"    - Ships processed: {comparison['ships_processed']}")
            print(f"    - Cargo volume: {comparison['cargo_volume_processed']:,.0f} containers")
            print(f"    - Waiting time improvement: {comparison['waiting_time_improvement']:.1f}%")
            print(f"    - Efficiency gain: {comparison['efficiency_gain']:.1f}%")
            print(f"    - Optimized avg waiting: {comparison['optimized_avg_waiting']:.1f}h")
            print(f"    - Non-optimized avg waiting: {comparison['non_optimized_avg_waiting']:.1f}h")
        
        print("\nOverall Insights:")
        insights = demo_results['overall_insights']
        print(f"  - Average efficiency gain: {insights['avg_efficiency_gain']:.1f}%")
        print(f"  - Average waiting time improvement: {insights['avg_waiting_time_improvement']:.1f}%")
        print(f"  - Optimization effectiveness: {insights['optimization_effectiveness']}")
        print(f"  - Scenarios analyzed: {insights['scenarios_analyzed']}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed with error: {e}")
        print("This may be due to missing historical data or import issues.")