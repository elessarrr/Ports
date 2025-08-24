"""Integration Module for Hong Kong Port Digital Twin

This module provides integration capabilities that combine all the enhanced
logistics components with the core simulation system, demonstrating the
full capabilities of the Week 7 implementation.

Components:
- EnhancedPortSimulation: Main integration class combining all systems
- Enhanced simulation configuration and orchestration
- Coordinated operations between yard, truck, maintenance, and disruption systems
- AI-powered optimization across all components
- Comprehensive performance monitoring and analytics
"""

from .enhanced_simulation import (
    EnhancedPortSimulation,
    EnhancedSimulationConfig,
    run_enhanced_simulation_demo
)

__all__ = [
    'EnhancedPortSimulation',
    'EnhancedSimulationConfig',
    'run_enhanced_simulation_demo'
]

__version__ = "1.0.0"
__author__ = "Hong Kong Port Digital Twin Team"
__description__ = "Integration module for enhanced port simulation with logistics modeling"