"""Scenarios package for Hong Kong Port Digital Twin.

This package provides scenario-based simulation capabilities for the port digital twin,
allowing different operational scenarios (Peak Season, Normal Operations, Low Season)
to be simulated with realistic parameters derived from historical data.

Main components:
- ScenarioParameters: Dataclass defining scenario-specific parameters
- ScenarioManager: Central manager for scenario selection and parameter retrieval
- Pre-defined scenarios based on Hong Kong Port historical patterns

Usage:
    from scenarios import ScenarioManager, get_scenario_parameters
    
    # Create a scenario manager
    manager = ScenarioManager()
    
    # Set a specific scenario
    manager.set_scenario('peak')
    
    # Get optimization parameters
    params = manager.get_optimization_parameters()
"""

from .scenario_parameters import (
    ScenarioParameters,
    PEAK_SEASON_PARAMETERS,
    NORMAL_OPERATIONS_PARAMETERS,
    LOW_SEASON_PARAMETERS,
    ALL_SCENARIOS,
    SCENARIO_ALIASES,
    get_scenario_parameters,
    list_available_scenarios,
    get_scenario_description,
    validate_scenario_parameters
)

from .scenario_manager import (
    ScenarioManager,
    create_scenario_manager,
    get_scenario_for_month,
    get_optimization_params_for_scenario
)

from .historical_extractor import HistoricalParameterExtractor
try:
    from .scenario_optimizer import ScenarioAwareBerthOptimizer
except ImportError:
    # Skip if optimization module is not available
    ScenarioAwareBerthOptimizer = None

# Package version
__version__ = '1.0.0'

# Package metadata
__author__ = 'Hong Kong Port Digital Twin Team'
__description__ = 'Scenario-based simulation parameters for port operations'

# Export main classes and functions
__all__ = [
    # Core classes
    'ScenarioParameters',
    'ScenarioManager',
    'HistoricalParameterExtractor',
    'ScenarioAwareBerthOptimizer',
    
    # Pre-defined scenarios
    'PEAK_SEASON_PARAMETERS',
    'NORMAL_OPERATIONS_PARAMETERS', 
    'LOW_SEASON_PARAMETERS',
    'ALL_SCENARIOS',
    'SCENARIO_ALIASES',
    
    # Utility functions
    'get_scenario_parameters',
    'list_available_scenarios',
    'get_scenario_description',
    'validate_scenario_parameters',
    'create_scenario_manager',
    'get_scenario_for_month',
    'get_optimization_params_for_scenario'
]

# Quick access to common scenarios
PEAK = 'peak'
NORMAL = 'normal'
LOW = 'low'

# Convenience function for quick scenario setup
def quick_setup(scenario_name: str = 'normal') -> ScenarioManager:
    """Quick setup function for creating a scenario manager.
    
    Args:
        scenario_name: Initial scenario to set
        
    Returns:
        Configured ScenarioManager instance
    """
    return create_scenario_manager(scenario_name, auto_detect=True)