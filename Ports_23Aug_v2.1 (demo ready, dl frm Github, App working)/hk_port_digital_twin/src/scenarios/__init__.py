"""Scenario Management Module for Hong Kong Port Digital Twin

This module provides comprehensive scenario management capabilities including:
- Scenario parameter definitions and validation
- Multi-scenario optimization and comparison
- Scenario execution and monitoring
- Performance benchmarking across scenarios
- Advanced scenario library with templates and collections
- Scenario saving, loading, and cataloging
"""

from .scenario_parameters import (
    ScenarioParameters,
    SeasonType,
    WeatherCondition,
    OperationalMode,
    list_available_scenarios
)

from .scenario_manager import (
    ScenarioManager,
    ScenarioType,
    ScenarioStatus
)

from .multi_scenario_optimizer import (
    MultiScenarioOptimizer,
    OptimizationObjective,
    OptimizationResult
)

from .scenario_library import (
    AdvancedScenarioLibrary,
    ScenarioTemplate,
    ScenarioInstance,
    ScenarioCollection,
    ScenarioParameter,
    ScenarioCategory,
    ScenarioComplexity,
    ScenarioStatus as LibraryScenarioStatus
)

from .scenario_optimizer import (
    ScenarioAwareBerthOptimizer
)

__all__ = [
    # Scenario Parameters
    'ScenarioParameters',
    'SeasonType',
    'WeatherCondition',
    'OperationalMode',
    'list_available_scenarios',
    
    # Scenario Manager
    'ScenarioManager',
    'ScenarioType',
    'ScenarioStatus',
    
    # Scenario Optimizer
    'ScenarioAwareBerthOptimizer',
    
    # Multi-Scenario Optimizer
    'MultiScenarioOptimizer',
    'OptimizationObjective',
    'OptimizationResult',
    
    # Advanced Scenario Library
    'AdvancedScenarioLibrary',
    'ScenarioTemplate',
    'ScenarioInstance',
    'ScenarioCollection',
    'ScenarioParameter',
    'ScenarioCategory',
    'ScenarioComplexity',
    'LibraryScenarioStatus'
]