"""Comments for context:
This module provides a scenario-aware wrapper for the BerthAllocationOptimizer.
It integrates scenario-specific parameters with the existing optimization logic
without modifying the core optimizer implementation.

The ScenarioAwareBerthOptimizer acts as a decorator/wrapper around the original
BerthAllocationOptimizer, applying scenario-specific adjustments to:
- Ship arrival patterns and priorities
- Berth capacity and availability
- Processing rates and efficiency factors
- Optimization objectives and constraints

This design maintains backward compatibility while enabling scenario-based
optimization comparisons across different operational conditions.

Key features:
- Transparent wrapping of existing BerthAllocationOptimizer
- Scenario-specific parameter application
- Enhanced ship and berth management with scenario context
- Optimization result comparison across scenarios
- Integration with ScenarioManager for parameter retrieval
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from copy import deepcopy
import random

# Import the existing optimization module
try:
    from ..ai.optimization import BerthAllocationOptimizer, Ship, Berth, OptimizationResult
except ImportError:
    # Fallback for testing or standalone usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ai.optimization import BerthAllocationOptimizer, Ship, Berth, OptimizationResult

from .scenario_manager import ScenarioManager
from .scenario_parameters import ScenarioParameters

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ScenarioOptimizationResult:
    """Extended optimization result with scenario context.
    
    This class extends the basic OptimizationResult with scenario-specific
    information and comparison capabilities.
    """
    
    # Original optimization result
    base_result: OptimizationResult
    
    # Scenario context
    scenario_name: str
    scenario_parameters: Dict[str, Any]
    
    # Enhanced metrics
    scenario_adjusted_metrics: Dict[str, float]
    parameter_impacts: Dict[str, float]
    
    # Comparison data (populated when comparing scenarios)
    comparison_baseline: Optional[str] = None
    relative_performance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result_dict = {
            'scenario_name': self.scenario_name,
            'scenario_parameters': self.scenario_parameters,
            'scenario_adjusted_metrics': self.scenario_adjusted_metrics,
            'parameter_impacts': self.parameter_impacts,
            'base_result': {
                'total_waiting_time': self.base_result.total_waiting_time,
                'average_waiting_time': self.base_result.average_waiting_time,
                'berth_utilization': self.base_result.berth_utilization,
                'unallocated_ships': 0,  # Calculate from ship_berth_assignments if needed
                'allocation_efficiency': self.base_result.optimization_score
            }
        }
        
        if self.comparison_baseline:
            result_dict['comparison_baseline'] = self.comparison_baseline
            result_dict['relative_performance'] = self.relative_performance
        
        return result_dict

class ScenarioAwareBerthOptimizer:
    """Scenario-aware wrapper for BerthAllocationOptimizer.
    
    This class wraps the existing BerthAllocationOptimizer and applies
    scenario-specific parameters to modify optimization behavior without
    changing the core optimization logic.
    """
    
    def __init__(self, scenario_manager: Optional[ScenarioManager] = None):
        """Initialize the scenario-aware optimizer.
        
        Args:
            scenario_manager: ScenarioManager instance (creates default if None)
        """
        self.scenario_manager = scenario_manager or ScenarioManager()
        self.base_optimizer = BerthAllocationOptimizer()
        self.optimization_history = []
        
        logger.info("ScenarioAwareBerthOptimizer initialized")
    
    def set_scenario(self, scenario_name: str) -> bool:
        """Set the current scenario for optimization.
        
        Args:
            scenario_name: Name of the scenario to use
            
        Returns:
            True if scenario was set successfully
        """
        success = self.scenario_manager.set_scenario(scenario_name)
        if success:
            logger.info(f"Optimizer scenario set to: {scenario_name}")
        return success
    
    def get_current_scenario(self) -> Optional[str]:
        """Get the current scenario name."""
        return self.scenario_manager.get_current_scenario()
    
    def add_ship(self, ship: Ship) -> None:
        """Add a ship with scenario-specific adjustments.
        
        Args:
            ship: Ship object to add
        """
        # Apply scenario-specific adjustments to ship
        adjusted_ship = self._apply_scenario_ship_adjustments(ship)
        self.base_optimizer.add_ship(adjusted_ship)
    
    def add_berth(self, berth: Berth) -> None:
        """Add a berth with scenario-specific adjustments.
        
        Args:
            berth: Berth object to add
        """
        # Apply scenario-specific adjustments to berth
        adjusted_berth = self._apply_scenario_berth_adjustments(berth)
        self.base_optimizer.add_berth(adjusted_berth)
    
    def optimize(self) -> ScenarioOptimizationResult:
        """Perform optimization with scenario-specific parameters.
        
        Returns:
            ScenarioOptimizationResult with enhanced metrics
        """
        current_scenario = self.scenario_manager.get_current_scenario()
        if not current_scenario:
            logger.warning("No scenario set, using default optimization")
            base_result = self.base_optimizer.optimize_berth_allocation()
            return self._create_scenario_result(base_result, 'default', {})
        
        logger.info(f"Starting optimization with scenario: {current_scenario}")
        
        # Get scenario parameters
        scenario_params = self.scenario_manager.get_optimization_parameters()
        
        # Apply scenario-specific optimization settings
        self._apply_scenario_optimization_settings(scenario_params)
        
        # Perform base optimization
        base_result = self.base_optimizer.optimize_berth_allocation()
        
        # Create enhanced result
        scenario_result = self._create_scenario_result(
            base_result, current_scenario, scenario_params
        )
        
        # Add to history
        self.optimization_history.append(scenario_result)
        
        logger.info(f"Optimization completed for scenario: {current_scenario}")
        return scenario_result
    
    def optimize_with_scenario(self, ships: List[Ship], berths: List[Berth], current_time=None) -> Dict[str, Any]:
        """Perform optimization with scenario-specific parameters using provided ships and berths.
        
        Args:
            ships: List of ships to optimize
            berths: List of available berths
            current_time: Current simulation time (optional)
            
        Returns:
            Dictionary containing optimization results and berth allocation
        """
        # Clear existing ships and berths
        self.base_optimizer.ships.clear()
        self.base_optimizer.berths.clear()
        
        # Add provided ships and berths
        for ship in ships:
            self.add_ship(ship)
        for berth in berths:
            self.add_berth(berth)
        
        # Perform optimization
        result = self.optimize()
        
        # Return result in expected format
        return {
            'berth_allocation': result.base_result,
            'scenario_metrics': result.scenario_adjusted_metrics,
            'optimization_time': getattr(result, 'optimization_time', 0.0),
            'scenario_name': result.scenario_name
        }
    
    def compare_scenarios(self, scenarios: List[str]) -> Dict[str, ScenarioOptimizationResult]:
        """Compare optimization results across multiple scenarios.
        
        Args:
            scenarios: List of scenario names to compare
            
        Returns:
            Dictionary mapping scenario names to optimization results
        """
        if not scenarios:
            logger.warning("No scenarios provided for comparison")
            return {}
        
        logger.info(f"Comparing scenarios: {scenarios}")
        
        # Store current state
        original_scenario = self.get_current_scenario()
        original_ships = deepcopy(self.base_optimizer.ships)
        original_berths = deepcopy(self.base_optimizer.berths)
        
        results = {}
        baseline_result = None
        
        try:
            for scenario_name in scenarios:
                # Reset optimizer state
                self.base_optimizer.ships = deepcopy(original_ships)
                self.base_optimizer.berths = deepcopy(original_berths)
                
                # Set scenario and optimize
                if self.set_scenario(scenario_name):
                    result = self.optimize()
                    results[scenario_name] = result
                    
                    # Use first scenario as baseline for comparison
                    if baseline_result is None:
                        baseline_result = result
                else:
                    logger.error(f"Failed to set scenario: {scenario_name}")
            
            # Add relative performance metrics
            if baseline_result:
                self._add_comparison_metrics(results, baseline_result)
            
        finally:
            # Restore original state
            if original_scenario:
                self.set_scenario(original_scenario)
            self.base_optimizer.ships = original_ships
            self.base_optimizer.berths = original_berths
        
        logger.info(f"Scenario comparison completed for {len(results)} scenarios")
        return results
    
    def _apply_scenario_ship_adjustments(self, ship: Ship) -> Ship:
        """Apply scenario-specific adjustments to a ship.
        
        Args:
            ship: Original ship object
            
        Returns:
            Ship with scenario adjustments applied
        """
        params = self.scenario_manager.get_current_parameters()
        if not params:
            return ship
        
        # Create adjusted ship
        adjusted_ship = deepcopy(ship)
        
        # Adjust container volume based on ship type
        if hasattr(adjusted_ship, 'container_count') and adjusted_ship.ship_type in params.container_volume_multipliers:
            multiplier = params.container_volume_multipliers[adjusted_ship.ship_type]
            adjusted_ship.container_count = int(adjusted_ship.container_count * multiplier)
        
        # Adjust ship size
        if hasattr(adjusted_ship, 'size'):
            adjusted_ship.size = int(adjusted_ship.size * params.average_ship_size_multiplier)
        
        # Adjust priority based on ship characteristics
        base_priority = getattr(adjusted_ship, 'priority', 1.0)
        
        # Apply large ship priority boost
        if hasattr(adjusted_ship, 'size') and adjusted_ship.size > 200:  # Large ship threshold
            base_priority *= params.large_ship_priority_boost
        
        # Apply container ship priority boost
        if adjusted_ship.ship_type == 'container':
            base_priority *= params.container_ship_priority_boost
        
        adjusted_ship.priority = base_priority
        
        return adjusted_ship
    
    def _apply_scenario_berth_adjustments(self, berth: Berth) -> Berth:
        """Apply scenario-specific adjustments to a berth.
        
        Args:
            berth: Original berth object
            
        Returns:
            Berth with scenario adjustments applied
        """
        params = self.scenario_manager.get_current_parameters()
        if not params:
            return berth
        
        # Create adjusted berth
        adjusted_berth = deepcopy(berth)
        
        # Adjust crane efficiency
        if hasattr(adjusted_berth, 'crane_count'):
            # Simulate efficiency by adjusting effective crane count
            effective_cranes = adjusted_berth.crane_count * params.crane_efficiency_multiplier
            adjusted_berth.crane_count = max(1, int(effective_cranes))
        
        # Adjust berth availability (simulate maintenance)
        if hasattr(adjusted_berth, 'available'):
            # Randomly make some berths unavailable based on availability factor
            if random.random() > params.berth_availability_factor:
                adjusted_berth.available = False
                logger.debug(f"Berth {adjusted_berth.berth_id} marked unavailable due to scenario parameters")
        
        return adjusted_berth
    
    def _apply_scenario_optimization_settings(self, scenario_params: Dict[str, Any]) -> None:
        """Apply scenario-specific settings to the optimization process.
        
        Args:
            scenario_params: Dictionary of scenario parameters
        """
        # This method can be extended to modify optimization algorithm parameters
        # based on scenario requirements. For now, it serves as a placeholder
        # for future enhancements.
        
        logger.debug(f"Applied scenario optimization settings: {list(scenario_params.keys())}")
    
    def _create_scenario_result(self, base_result: OptimizationResult, 
                              scenario_name: str, scenario_params: Dict[str, Any]) -> ScenarioOptimizationResult:
        """Create a ScenarioOptimizationResult from base result.
        
        Args:
            base_result: Original optimization result
            scenario_name: Name of the scenario
            scenario_params: Scenario parameters used
            
        Returns:
            Enhanced optimization result with scenario context
        """
        # Calculate scenario-adjusted metrics
        adjusted_metrics = self._calculate_adjusted_metrics(base_result, scenario_params)
        
        # Calculate parameter impacts
        parameter_impacts = self._calculate_parameter_impacts(scenario_params)
        
        return ScenarioOptimizationResult(
            base_result=base_result,
            scenario_name=scenario_name,
            scenario_parameters=scenario_params,
            scenario_adjusted_metrics=adjusted_metrics,
            parameter_impacts=parameter_impacts
        )
    
    def _calculate_adjusted_metrics(self, result: OptimizationResult, 
                                  params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scenario-adjusted performance metrics.
        
        Args:
            result: Base optimization result
            params: Scenario parameters
            
        Returns:
            Dictionary of adjusted metrics
        """
        adjusted_metrics = {}
        
        # Adjust waiting time based on processing efficiency
        if 'processing_rate_multiplier' in params:
            adjusted_waiting_time = result.total_waiting_time / params['processing_rate_multiplier']
            adjusted_metrics['adjusted_total_waiting_time'] = adjusted_waiting_time
            adjusted_metrics['adjusted_average_waiting_time'] = (
                adjusted_waiting_time / max(1, len(result.ship_berth_assignments))
            )
        
        # Adjust berth utilization based on target utilization
        if 'target_berth_utilization' in params:
            avg_utilization = sum(result.berth_utilization.values()) / max(1, len(result.berth_utilization))
            utilization_efficiency = avg_utilization / params['target_berth_utilization']
            adjusted_metrics['utilization_efficiency'] = min(1.0, utilization_efficiency)
        
        # Calculate throughput efficiency
        if 'arrival_rate_multiplier' in params:
            throughput_factor = params['arrival_rate_multiplier']
            adjusted_metrics['throughput_efficiency'] = result.optimization_score * throughput_factor
        
        return adjusted_metrics
    
    def _calculate_parameter_impacts(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the impact of each parameter on optimization.
        
        Args:
            params: Scenario parameters
            
        Returns:
            Dictionary of parameter impact scores
        """
        impacts = {}
        
        # Calculate impact scores based on deviation from baseline (1.0)
        for param_name, value in params.items():
            if isinstance(value, (int, float)) and param_name.endswith('_multiplier'):
                # Impact is the absolute deviation from 1.0
                impacts[param_name] = abs(value - 1.0)
        
        return impacts
    
    def _add_comparison_metrics(self, results: Dict[str, ScenarioOptimizationResult], 
                              baseline: ScenarioOptimizationResult) -> None:
        """Add relative performance metrics to comparison results.
        
        Args:
            results: Dictionary of scenario results
            baseline: Baseline result for comparison
        """
        baseline_waiting_time = baseline.base_result.total_waiting_time
        baseline_utilization = baseline.base_result.berth_utilization
        baseline_efficiency = baseline.base_result.optimization_score
        
        for scenario_name, result in results.items():
            if scenario_name == baseline.scenario_name:
                continue
            
            # Calculate relative performance
            relative_performance = {}
            
            if baseline_waiting_time > 0:
                waiting_time_improvement = (
                    (baseline_waiting_time - result.base_result.total_waiting_time) / baseline_waiting_time
                )
                relative_performance['waiting_time_improvement'] = waiting_time_improvement
            
            if baseline_utilization > 0:
                utilization_improvement = (
                    (result.base_result.berth_utilization - baseline_utilization) / baseline_utilization
                )
                relative_performance['utilization_improvement'] = utilization_improvement
            
            if baseline_efficiency > 0:
                efficiency_improvement = (
                    (result.base_result.optimization_score - baseline_efficiency) / baseline_efficiency
                )
                relative_performance['efficiency_improvement'] = efficiency_improvement
            
            # Update result with comparison data
            result.comparison_baseline = baseline.scenario_name
            result.relative_performance = relative_performance
    
    def get_optimization_history(self) -> List[ScenarioOptimizationResult]:
        """Get history of optimization results.
        
        Returns:
            List of optimization results
        """
        return self.optimization_history.copy()
    
    def clear_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history.clear()
        logger.info("Optimization history cleared")
    
    def reset(self) -> None:
        """Reset the optimizer to initial state."""
        self.base_optimizer = BerthAllocationOptimizer()
        self.clear_history()
        logger.info("Optimizer reset to initial state")
    
    def __str__(self) -> str:
        """String representation of the optimizer."""
        current_scenario = self.get_current_scenario()
        return f"ScenarioAwareBerthOptimizer(scenario='{current_scenario}', history={len(self.optimization_history)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


# Utility functions for easy integration

def create_scenario_optimizer(scenario_name: str = 'normal') -> ScenarioAwareBerthOptimizer:
    """Create a scenario-aware optimizer with a specific scenario.
    
    Args:
        scenario_name: Initial scenario to set
        
    Returns:
        Configured ScenarioAwareBerthOptimizer
    """
    scenario_manager = ScenarioManager()
    scenario_manager.set_scenario(scenario_name)
    return ScenarioAwareBerthOptimizer(scenario_manager)

def quick_scenario_comparison(scenarios: List[str], ships: List[Ship], 
                            berths: List[Berth]) -> Dict[str, ScenarioOptimizationResult]:
    """Quick comparison of scenarios with provided ships and berths.
    
    Args:
        scenarios: List of scenario names to compare
        ships: List of ships to optimize
        berths: List of berths available
        
    Returns:
        Dictionary of optimization results by scenario
    """
    optimizer = ScenarioAwareBerthOptimizer()
    
    # Add ships and berths
    for ship in ships:
        optimizer.add_ship(ship)
    for berth in berths:
        optimizer.add_berth(berth)
    
    # Compare scenarios
    return optimizer.compare_scenarios(scenarios)