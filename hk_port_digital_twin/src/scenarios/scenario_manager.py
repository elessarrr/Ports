"""Comments for context:
This module provides the ScenarioManager class for managing operational scenarios
in the Hong Kong Port Digital Twin simulation. It handles scenario selection,
parameter retrieval, and integration with historical data analysis.

The ScenarioManager serves as the central interface between the simulation engine
and scenario-specific parameters. It can automatically detect the most appropriate
scenario based on historical patterns or allow manual scenario selection.

Key features:
- Automatic scenario detection based on seasonal patterns
- Manual scenario selection and switching
- Integration with data_loader.py for historical analysis
- Parameter validation and error handling
- Logging and monitoring of scenario changes

This design maintains separation of concerns while providing a clean interface
for the simulation engine to access scenario-specific parameters.
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
from enum import Enum

class ScenarioType(Enum):
    """Enumeration of scenario types"""
    PEAK_SEASON = "peak_season"
    NORMAL_OPERATIONS = "normal_operations"
    LOW_SEASON = "low_season"
    EQUIPMENT_MAINTENANCE = "equipment_maintenance"
    TYPHOON_DISRUPTION = "typhoon_disruption"
    CAPACITY_STRESS_TEST = "capacity_stress_test"
    AI_OPTIMIZATION_BENCHMARK = "ai_optimization_benchmark"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"

class ScenarioStatus(Enum):
    """Enumeration of scenario status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    COMPLETED = "completed"
    ERROR = "error"

from .scenario_parameters import (
    ScenarioParameters,
    ALL_SCENARIOS,
    SCENARIO_ALIASES,
    get_scenario_parameters,
    list_available_scenarios,
    validate_scenario_parameters
)
from .historical_extractor import HistoricalParameterExtractor

# Configure logging
logger = logging.getLogger(__name__)

class ScenarioManager:
    """Manages operational scenarios for the port simulation.
    
    This class provides a centralized interface for scenario management,
    including automatic scenario detection based on historical patterns,
    manual scenario selection, and parameter retrieval.
    """
    
    def __init__(self, default_scenario: str = 'normal', use_historical_data: bool = True):
        """Initialize the scenario manager.
        
        Args:
            default_scenario: Default scenario to use if auto-detection fails
            use_historical_data: Whether to use historical data for parameter extraction
        """
        self.default_scenario = default_scenario
        self.current_scenario = None
        self.current_parameters = None
        self.scenario_history = []  # Track scenario changes
        self.auto_detection_enabled = True
        self.use_historical_data = use_historical_data
        
        # Initialize historical parameter extractor
        self.historical_extractor = HistoricalParameterExtractor() if use_historical_data else None
        if self.historical_extractor:
            self.historical_extractor.load_historical_data()
        
        # Initialize with default scenario
        self.set_scenario(default_scenario)
        
        logger.info(f"ScenarioManager initialized with default scenario: {default_scenario}")
    
    def set_scenario(self, scenario_name: str, validate: bool = True) -> bool:
        """Set the current scenario.
        
        Args:
            scenario_name: Name or alias of the scenario to set
            validate: Whether to validate scenario parameters
            
        Returns:
            True if scenario was set successfully, False otherwise
        """
        try:
            # Try to extract parameters from historical data first
            params = None
            if self.historical_extractor and self.use_historical_data:
                try:
                    params = self.historical_extractor.extract_scenario_parameters(scenario_name)
                    if params:
                        logger.info(f"Using historical data parameters for scenario: {scenario_name}")
                except Exception as e:
                    logger.warning(f"Failed to extract historical parameters for {scenario_name}: {e}")
            
            # Fallback to predefined parameters
            if params is None:
                params = get_scenario_parameters(scenario_name)
                if params:
                    logger.info(f"Using predefined parameters for scenario: {scenario_name}")
            
            if params is None:
                logger.error(f"Unknown scenario: {scenario_name}")
                return False
            
            # Validate parameters if requested
            if validate:
                errors = validate_scenario_parameters(params)
                if errors:
                    logger.error(f"Validation errors in scenario {scenario_name}: {errors}")
                    return False
            
            # Record scenario change
            old_scenario = self.current_scenario
            self.current_scenario = scenario_name
            self.current_parameters = params
            
            # Add to history
            self.scenario_history.append({
                'timestamp': datetime.now(),
                'old_scenario': old_scenario,
                'new_scenario': scenario_name,
                'method': 'manual'
            })
            
            logger.info(f"Scenario changed from {old_scenario} to {scenario_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting scenario {scenario_name}: {e}")
            return False
    
    def get_current_scenario(self) -> Optional[str]:
        """Get the name of the current scenario.
        
        Returns:
            Current scenario name or None if not set
        """
        return self.current_scenario
    
    def get_current_parameters(self) -> Optional[ScenarioParameters]:
        """Get the current scenario parameters.
        
        Returns:
            Current ScenarioParameters object or None if not set
        """
        return self.current_parameters
    
    def get_parameters_dict(self) -> Optional[Dict[str, Any]]:
        """Get current scenario parameters as a dictionary.
        
        Returns:
            Dictionary representation of current parameters or None
        """
        if self.current_parameters:
            return asdict(self.current_parameters)
        return None
    
    def auto_detect_scenario(self, target_date: Optional[date] = None) -> str:
        """Automatically detect the most appropriate scenario based on date.
        
        This method uses the primary_months and secondary_months defined in
        scenario parameters to determine the best scenario for a given date.
        
        Args:
            target_date: Date to analyze (defaults to current date)
            
        Returns:
            Name of the detected scenario
        """
        if target_date is None:
            target_date = date.today()
        
        target_month = target_date.month
        
        # Check each scenario for primary month match
        for scenario_key, params in ALL_SCENARIOS.items():
            if target_month in params.primary_months:
                logger.info(f"Auto-detected scenario '{scenario_key}' for month {target_month} (primary match)")
                return scenario_key
        
        # Check for secondary month match
        for scenario_key, params in ALL_SCENARIOS.items():
            if target_month in params.secondary_months:
                logger.info(f"Auto-detected scenario '{scenario_key}' for month {target_month} (secondary match)")
                return scenario_key
        
        # Default to normal operations if no match found
        logger.warning(f"No scenario match for month {target_month}, defaulting to 'normal'")
        return 'normal'
    
    def auto_set_scenario(self, target_date: Optional[date] = None) -> bool:
        """Automatically detect and set the appropriate scenario.
        
        Args:
            target_date: Date to analyze (defaults to current date)
            
        Returns:
            True if scenario was set successfully, False otherwise
        """
        if not self.auto_detection_enabled:
            logger.debug("Auto-detection is disabled")
            return False
        
        try:
            detected_scenario = self.auto_detect_scenario(target_date)
            
            # Only change if different from current
            if detected_scenario != self.current_scenario:
                success = self.set_scenario(detected_scenario)
                if success:
                    # Update history to reflect auto-detection
                    if self.scenario_history:
                        self.scenario_history[-1]['method'] = 'auto_detection'
                        self.scenario_history[-1]['target_date'] = target_date
                return success
            else:
                logger.debug(f"Current scenario '{self.current_scenario}' is already optimal for the target date")
                return True
                
        except Exception as e:
            logger.error(f"Error in auto scenario detection: {e}")
            return False
    
    def enable_auto_detection(self, enabled: bool = True):
        """Enable or disable automatic scenario detection.
        
        Args:
            enabled: Whether to enable auto-detection
        """
        self.auto_detection_enabled = enabled
        logger.info(f"Auto-detection {'enabled' if enabled else 'disabled'}")
    
    def get_scenario_info(self, scenario_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed information about a scenario.
        
        Args:
            scenario_name: Name of scenario to get info for (defaults to current)
            
        Returns:
            Dictionary with scenario information or None if not found
        """
        if scenario_name is None:
            scenario_name = self.current_scenario
        
        if scenario_name is None:
            return None
        
        params = get_scenario_parameters(scenario_name)
        if params is None:
            return None
        
        return {
            'name': params.scenario_name,
            'description': params.scenario_description,
            'primary_months': params.primary_months,
            'secondary_months': params.secondary_months,
            'arrival_rate_multiplier': params.arrival_rate_multiplier,
            'target_berth_utilization': params.target_berth_utilization,
            'ship_type_distribution': params.ship_type_distribution,
            'parameters': asdict(params)
        }
    
    def list_scenarios(self) -> List[Dict[str, str]]:
        """Get list of all available scenarios with basic info.
        
        Returns:
            List of dictionaries with scenario name and description
        """
        scenarios = []
        for scenario_key in list_available_scenarios():
            params = get_scenario_parameters(scenario_key)
            if params:
                scenarios.append({
                    'key': scenario_key,
                    'name': params.scenario_name,
                    'description': params.scenario_description
                })
        return scenarios
    
    def get_scenario_history(self) -> List[Dict[str, Any]]:
        """Get history of scenario changes.
        
        Returns:
            List of scenario change records
        """
        return self.scenario_history.copy()
    
    def compare_scenarios(self, scenario1: str, scenario2: str) -> Optional[Dict[str, Any]]:
        """Compare two scenarios and return key differences.
        
        Args:
            scenario1: First scenario to compare
            scenario2: Second scenario to compare
            
        Returns:
            Dictionary with comparison results or None if scenarios not found
        """
        params1 = get_scenario_parameters(scenario1)
        params2 = get_scenario_parameters(scenario2)
        
        if params1 is None or params2 is None:
            return None
        
        # Key metrics to compare
        comparison = {
            'scenario1': {
                'name': params1.scenario_name,
                'arrival_rate': params1.arrival_rate_multiplier,
                'berth_utilization': params1.target_berth_utilization,
                'crane_efficiency': params1.crane_efficiency_multiplier,
                'ship_size': params1.average_ship_size_multiplier
            },
            'scenario2': {
                'name': params2.scenario_name,
                'arrival_rate': params2.arrival_rate_multiplier,
                'berth_utilization': params2.target_berth_utilization,
                'crane_efficiency': params2.crane_efficiency_multiplier,
                'ship_size': params2.average_ship_size_multiplier
            },
            'differences': {
                'arrival_rate_diff': params2.arrival_rate_multiplier - params1.arrival_rate_multiplier,
                'berth_utilization_diff': params2.target_berth_utilization - params1.target_berth_utilization,
                'crane_efficiency_diff': params2.crane_efficiency_multiplier - params1.crane_efficiency_multiplier,
                'ship_size_diff': params2.average_ship_size_multiplier - params1.average_ship_size_multiplier
            }
        }
        
        return comparison
    
    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get parameters specifically needed for berth allocation optimization.
        
        Returns:
            Dictionary with optimization-specific parameters
        """
        if self.current_parameters is None:
            logger.warning("No current scenario set, using default parameters")
            return {}
        
        params = self.current_parameters
        
        return {
            'arrival_rate_multiplier': params.arrival_rate_multiplier,
            'ship_type_distribution': params.ship_type_distribution,
            'container_volume_multipliers': params.container_volume_multipliers,
            'average_ship_size_multiplier': params.average_ship_size_multiplier,
            'crane_efficiency_multiplier': params.crane_efficiency_multiplier,
            'processing_rate_multiplier': params.processing_rate_multiplier,
            'target_berth_utilization': params.target_berth_utilization,
            'berth_availability_factor': params.berth_availability_factor,
            'large_ship_priority_boost': params.large_ship_priority_boost,
            'container_ship_priority_boost': params.container_ship_priority_boost
        }
    
    def reset_to_default(self) -> bool:
        """Reset to the default scenario.
        
        Returns:
            True if reset was successful, False otherwise
        """
        logger.info(f"Resetting to default scenario: {self.default_scenario}")
        return self.set_scenario(self.default_scenario)
    
    def validate_current_scenario(self) -> List[str]:
        """Validate the current scenario parameters.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        if self.current_parameters is None:
            return ["No current scenario set"]
        
        return validate_scenario_parameters(self.current_parameters)
    
    def __str__(self) -> str:
        """String representation of the scenario manager."""
        if self.current_scenario:
            return f"ScenarioManager(current='{self.current_scenario}', auto_detection={self.auto_detection_enabled})"
        else:
            return f"ScenarioManager(no_scenario_set, auto_detection={self.auto_detection_enabled})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


# Utility functions for integration with existing code

def create_scenario_manager(scenario: str = 'normal', auto_detect: bool = True) -> ScenarioManager:
    """Create and configure a scenario manager.
    
    Args:
        scenario: Initial scenario to set
        auto_detect: Whether to enable auto-detection
        
    Returns:
        Configured ScenarioManager instance
    """
    manager = ScenarioManager(default_scenario=scenario)
    manager.enable_auto_detection(auto_detect)
    return manager

def get_scenario_for_month(month: int) -> str:
    """Get the most appropriate scenario for a given month.
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Scenario name
    """
    target_date = date(2024, month, 1)  # Use arbitrary year
    manager = ScenarioManager()
    return manager.auto_detect_scenario(target_date)

def get_optimization_params_for_scenario(scenario_name: str) -> Dict[str, Any]:
    """Get optimization parameters for a specific scenario.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        Dictionary with optimization parameters
    """
    manager = ScenarioManager()
    if manager.set_scenario(scenario_name):
        return manager.get_optimization_parameters()
    else:
        logger.error(f"Failed to set scenario {scenario_name}, returning empty parameters")
        return {}