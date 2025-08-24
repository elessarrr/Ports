"""Hong Kong Port Digital Twin - Main Package

A comprehensive digital twin simulation system for Hong Kong Port operations,
featuring AI-powered optimization, real-time monitoring, advanced analytics,
and enhanced logistics modeling capabilities.

Main Components:
- Core simulation engine with berth, vessel, and container management
- AI-powered optimization algorithms for berth allocation and resource scheduling
- Real-time dashboard with interactive visualizations
- Advanced scenario library with templates and collections
- Enhanced logistics modeling (yard management, truck routing, equipment maintenance)
- Supply chain disruption modeling and analysis
- Performance analytics and benchmarking
- Integration with external data sources
"""

# Core simulation components
from .core import (
    PortSimulation,
    SimulationController,
    Berth,
    Vessel,
    Container,
    BerthOptimizer,
    ResourceManager
)

# AI and optimization modules
from .ai import (
    AIOptimizer,
    PredictiveAnalytics,
    ReinforcementLearningAgent,
    OptimizationObjective
)

# Analysis and monitoring
from .analysis import (
    PerformanceAnalyzer,
    BenchmarkReporter,
    RealTimeMonitor,
    MetricsCollector
)

# Enhanced logistics modeling
from .logistics import (
    ContainerYardManager,
    TruckRoutingSystem,
    EquipmentMaintenanceScheduler,
    SupplyChainDisruptionModeler
)

# Advanced scenario management
from .scenarios import (
    ScenarioManager,
    ScenarioParameters,
    MultiScenarioOptimizer,
    AdvancedScenarioLibrary,
    ScenarioTemplate,
    ScenarioCollection
)

# Dashboard and visualization
from .dashboard import (
    DashboardApp,
    create_dashboard,
    run_dashboard
)

# Integration and orchestration
from .integration import (
    EnhancedPortSimulation,
    EnhancedSimulationConfig,
    run_enhanced_simulation_demo
)

# Utilities
from .utils import (
    ConfigManager,
    DataLoader,
    Logger,
    ValidationUtils
)

__version__ = "2.1.0"
__author__ = "Hong Kong Port Digital Twin Team"
__description__ = "Advanced Digital Twin System for Hong Kong Port Operations with Enhanced Logistics"

__all__ = [
    # Core Components
    'PortSimulation',
    'SimulationController',
    'Berth',
    'Vessel', 
    'Container',
    'BerthOptimizer',
    'ResourceManager',
    
    # AI and Optimization
    'AIOptimizer',
    'PredictiveAnalytics',
    'ReinforcementLearningAgent',
    'OptimizationObjective',
    
    # Analysis and Monitoring
    'PerformanceAnalyzer',
    'BenchmarkReporter',
    'RealTimeMonitor',
    'MetricsCollector',
    
    # Enhanced Logistics
    'ContainerYardManager',
    'TruckRoutingSystem',
    'EquipmentMaintenanceScheduler',
    'SupplyChainDisruptionModeler',
    
    # Advanced Scenario Management
    'ScenarioManager',
    'ScenarioParameters',
    'MultiScenarioOptimizer',
    'AdvancedScenarioLibrary',
    'ScenarioTemplate',
    'ScenarioCollection',
    
    # Dashboard
    'DashboardApp',
    'create_dashboard',
    'run_dashboard',
    
    # Integration
    'EnhancedPortSimulation',
    'EnhancedSimulationConfig',
    'run_enhanced_simulation_demo',
    
    # Utilities
    'ConfigManager',
    'DataLoader',
    'Logger',
    'ValidationUtils'
]