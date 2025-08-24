"""Unified Simulation Framework for Hong Kong Port Digital Twin

This module consolidates all simulation types from both operational scenarios and strategic
simulations into a single, comprehensive framework. It provides a unified interface for
all simulation types while maintaining backward compatibility with existing systems.

The framework supports:
- Operational scenarios (Peak Season, Normal Operations, Low Season)
- Strategic simulations (Capacity Optimization, Maintenance Planning)
- Disruption scenarios (Typhoon, Equipment Maintenance, Supply Chain)
- Performance testing scenarios (Capacity Stress Test)

Designed for demo purposes with enhanced business intelligence and visualization capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import logging
from datetime import datetime

# Import existing components
from .scenario_parameters import ScenarioParameters
from .strategic_simulations import StrategicBusinessMetrics
from .scenario_manager import ScenarioType
from .strategic_simulations import StrategicScenarioType, BusinessMetricType

# Configure logging
logger = logging.getLogger(__name__)

class UnifiedSimulationType(Enum):
    """Unified enumeration of all simulation types for the demo.
    
    Consolidates operational scenarios, strategic simulations, and disruption scenarios
    into a single comprehensive enum for the unified simulation tab.
    """
    
    # Operational Scenarios (from ScenarioType)
    PEAK_SEASON = "peak_season"
    NORMAL_OPERATIONS = "normal_operations"
    LOW_SEASON = "low_season"
    
    # Strategic Simulations (from StrategicScenarioType)
    PEAK_SEASON_CAPACITY_OPTIMIZATION = "peak_season_capacity_optimization"
    MAINTENANCE_WINDOW_OPTIMIZATION = "maintenance_window_optimization"
    
    # Disruption Scenarios
    EQUIPMENT_MAINTENANCE = "equipment_maintenance"
    TYPHOON_DISRUPTION = "typhoon_disruption"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    
    # Performance Testing
    CAPACITY_STRESS_TEST = "capacity_stress_test"

class SimulationCategory(Enum):
    """Categories for organizing simulations in the UI."""
    OPERATIONAL = "operational"  # Day-to-day operations
    STRATEGIC = "strategic"      # Business planning and optimization
    DISRUPTION = "disruption"    # Crisis and disruption management
    PERFORMANCE = "performance"  # Stress testing and benchmarking

class SimulationComplexity(Enum):
    """Complexity levels for demo presentation."""
    BASIC = "basic"              # Simple scenarios for quick demos
    INTERMEDIATE = "intermediate" # Moderate complexity with multiple factors
    ADVANCED = "advanced"        # Complex scenarios with full business intelligence

class DemoFocusArea(Enum):
    """Focus areas for demo presentation."""
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    STRATEGIC_PLANNING = "strategic_planning"
    CRISIS_MANAGEMENT = "crisis_management"
    ROI_ANALYSIS = "roi_analysis"
    COMPETITIVE_ADVANTAGE = "competitive_advantage"

@dataclass
class UnifiedBusinessMetrics:
    """Enhanced business metrics combining operational and strategic KPIs.
    
    Designed for comprehensive demo presentation with both operational
    efficiency metrics and strategic business intelligence.
    """
    
    # Operational Metrics
    throughput_teu_per_hour: float = 0.0
    berth_utilization_percentage: float = 0.0
    average_waiting_time_hours: float = 0.0
    crane_efficiency_percentage: float = 0.0
    
    # Financial Metrics
    revenue_per_hour: float = 0.0
    operational_cost_per_hour: float = 0.0
    cost_savings_percentage: float = 0.0
    roi_percentage: float = 0.0
    
    # Strategic Metrics
    customer_satisfaction_score: float = 0.0
    competitive_advantage_index: float = 0.0
    capacity_optimization_score: float = 0.0
    sustainability_index: float = 0.0
    
    # Risk and Performance Metrics
    risk_mitigation_score: float = 0.0
    performance_reliability: float = 0.0
    scalability_factor: float = 0.0
    innovation_index: float = 0.0

@dataclass
class DemoConfiguration:
    """Configuration for demo-specific features and presentation."""
    
    # Demo presentation settings
    demo_duration_minutes: int = 15
    highlight_key_metrics: List[str] = field(default_factory=list)
    focus_areas: List[DemoFocusArea] = field(default_factory=list)
    
    # Visualization preferences
    enable_real_time_updates: bool = True
    show_comparative_analysis: bool = True
    include_roi_projections: bool = True
    display_competitive_benchmarks: bool = True
    
    # Business intelligence features
    enable_predictive_analytics: bool = True
    show_scenario_recommendations: bool = True
    include_executive_summary: bool = True
    generate_action_items: bool = True

@dataclass
class UnifiedSimulationParameters:
    """Unified parameters combining operational and strategic simulation parameters.
    
    This class extends the existing ScenarioParameters structure with strategic
    business metrics, demo configuration, and enhanced visualization capabilities
    for comprehensive demo presentation.
    """
    
    # Core identification
    simulation_type: UnifiedSimulationType
    simulation_name: str
    simulation_description: str
    
    # Categorization for UI organization
    category: SimulationCategory
    complexity: SimulationComplexity
    demo_focus_areas: List[DemoFocusArea] = field(default_factory=list)
    
    # Operational parameters (from ScenarioParameters)
    arrival_rate_multiplier: float = 1.0
    peak_hour_multiplier: float = 1.5
    weekend_multiplier: float = 0.8
    ship_type_distribution: Dict[str, float] = field(default_factory=dict)
    container_volume_multipliers: Dict[str, float] = field(default_factory=dict)
    average_ship_size_multiplier: float = 1.0
    crane_efficiency_multiplier: float = 1.0
    docking_time_multiplier: float = 1.0
    processing_rate_multiplier: float = 1.0
    target_berth_utilization: float = 0.85
    berth_availability_factor: float = 1.0
    large_ship_priority_boost: float = 0.0
    container_ship_priority_boost: float = 0.0
    primary_months: List[int] = field(default_factory=list)
    secondary_months: List[int] = field(default_factory=list)
    
    # Strategic and business parameters
    business_objective: str = ""
    executive_summary: str = ""
    planning_horizon_days: int = 30
    investment_required: float = 0.0
    expected_roi_percentage: float = 0.0
    market_demand_multiplier: float = 1.0
    competitor_efficiency_baseline: float = 0.8
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Enhanced business metrics
    business_metrics: UnifiedBusinessMetrics = field(default_factory=UnifiedBusinessMetrics)
    
    # Demo configuration
    demo_config: DemoConfiguration = field(default_factory=DemoConfiguration)
    
    # Metadata for tracking and analytics
    created_timestamp: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    def to_scenario_parameters(self) -> ScenarioParameters:
        """Convert to legacy ScenarioParameters for backward compatibility."""
        return ScenarioParameters(
            scenario_name=self.simulation_name,
            scenario_description=self.simulation_description,
            arrival_rate_multiplier=self.arrival_rate_multiplier,
            peak_hour_multiplier=self.peak_hour_multiplier,
            weekend_multiplier=self.weekend_multiplier,
            ship_type_distribution=self.ship_type_distribution,
            container_volume_multipliers=self.container_volume_multipliers,
            average_ship_size_multiplier=self.average_ship_size_multiplier,
            crane_efficiency_multiplier=self.crane_efficiency_multiplier,
            docking_time_multiplier=self.docking_time_multiplier,
            processing_rate_multiplier=self.processing_rate_multiplier,
            target_berth_utilization=self.target_berth_utilization,
            berth_availability_factor=self.berth_availability_factor,
            large_ship_priority_boost=self.large_ship_priority_boost,
            container_ship_priority_boost=self.container_ship_priority_boost,
            primary_months=self.primary_months,
            secondary_months=self.secondary_months
        )
    
    def get_demo_highlights(self) -> List[str]:
        """Get key highlights for demo presentation."""
        highlights = []
        
        if self.category == SimulationCategory.STRATEGIC:
            highlights.append(f"ROI: {self.expected_roi_percentage:.1f}%")
            highlights.append(f"Investment: ${self.investment_required:,.0f}")
        
        if self.business_metrics.throughput_teu_per_hour > 0:
            highlights.append(f"Throughput: {self.business_metrics.throughput_teu_per_hour:.0f} TEU/hr")
        
        if self.business_metrics.cost_savings_percentage > 0:
            highlights.append(f"Cost Savings: {self.business_metrics.cost_savings_percentage:.1f}%")
        
        if self.business_metrics.berth_utilization_percentage > 0:
            highlights.append(f"Berth Utilization: {self.business_metrics.berth_utilization_percentage:.1f}%")
        
        return highlights
    
    def get_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive risk assessment for the simulation."""
        return {
            "risk_factors": self.risk_factors,
            "mitigation_strategies": self.mitigation_strategies,
            "risk_score": self.business_metrics.risk_mitigation_score,
            "reliability": self.business_metrics.performance_reliability,
            "complexity_level": self.complexity.value
        }

class UnifiedSimulationController:
    """Controller for managing unified simulations.
    
    Provides a single interface for all simulation types while maintaining
    backward compatibility with existing scenario and strategic simulation systems.
    """
    
    def __init__(self):
        self.current_simulation: Optional[UnifiedSimulationParameters] = None
        self.simulation_history: List[UnifiedSimulationParameters] = []
        self.available_simulations: Dict[UnifiedSimulationType, UnifiedSimulationParameters] = {}
        
        # Initialize with default simulations
        self._initialize_default_simulations()
    
    def _initialize_default_simulations(self):
        """Initialize the controller with default simulation configurations."""
        # This will be populated with actual simulation definitions
        # in the next phase of implementation
        pass
    
    def get_available_simulations(self) -> Dict[UnifiedSimulationType, UnifiedSimulationParameters]:
        """Get all available simulations organized by type."""
        return self.available_simulations
    
    def get_simulations_by_category(self, category: SimulationCategory) -> List[UnifiedSimulationParameters]:
        """Get simulations filtered by category."""
        return [sim for sim in self.available_simulations.values() if sim.category == category]
    
    def get_simulations_by_complexity(self, complexity: SimulationComplexity) -> List[UnifiedSimulationParameters]:
        """Get simulations filtered by complexity level."""
        return [sim for sim in self.available_simulations.values() if sim.complexity == complexity]
    
    def get_demo_ready_simulations(self) -> List[UnifiedSimulationParameters]:
        """Get simulations optimized for demo presentation."""
        return [sim for sim in self.available_simulations.values() 
                if sim.demo_config.enable_real_time_updates and 
                   sim.demo_config.include_executive_summary]
    
    def select_simulation(self, simulation_type: UnifiedSimulationType) -> Optional[UnifiedSimulationParameters]:
        """Select and activate a simulation."""
        if simulation_type in self.available_simulations:
            self.current_simulation = self.available_simulations[simulation_type]
            self.simulation_history.append(self.current_simulation)
            logger.info(f"Selected simulation: {simulation_type.value}")
            return self.current_simulation
        else:
            logger.error(f"Simulation type not found: {simulation_type.value}")
            return None
    
    def get_current_simulation(self) -> Optional[UnifiedSimulationParameters]:
        """Get the currently active simulation."""
        return self.current_simulation
    
    def get_comparative_analysis(self, simulation_types: List[UnifiedSimulationType]) -> Dict[str, Any]:
        """Generate comparative analysis between multiple simulations."""
        simulations = [self.available_simulations[sim_type] for sim_type in simulation_types 
                      if sim_type in self.available_simulations]
        
        if not simulations:
            return {}
        
        comparison = {
            "simulations": [sim.simulation_name for sim in simulations],
            "metrics_comparison": {},
            "roi_comparison": [sim.expected_roi_percentage for sim in simulations],
            "complexity_comparison": [sim.complexity.value for sim in simulations],
            "investment_comparison": [sim.investment_required for sim in simulations]
        }
        
        # Add detailed metrics comparison
        metric_names = [
            "throughput_teu_per_hour", "berth_utilization_percentage", 
            "cost_savings_percentage", "customer_satisfaction_score"
        ]
        
        for metric in metric_names:
            comparison["metrics_comparison"][metric] = [
                getattr(sim.business_metrics, metric) for sim in simulations
            ]
        
        return comparison
    
    def generate_executive_summary(self, simulation_type: UnifiedSimulationType) -> Dict[str, Any]:
        """Generate executive summary for a simulation."""
        if simulation_type not in self.available_simulations:
            return {}
        
        simulation = self.available_simulations[simulation_type]
        
        return {
            "simulation_name": simulation.simulation_name,
            "business_objective": simulation.business_objective,
            "executive_summary": simulation.executive_summary,
            "key_highlights": simulation.get_demo_highlights(),
            "roi_projection": simulation.expected_roi_percentage,
            "investment_required": simulation.investment_required,
            "risk_assessment": simulation.get_risk_assessment(),
            "recommended_actions": simulation.mitigation_strategies[:3],  # Top 3 actions
            "complexity_level": simulation.complexity.value,
            "demo_duration": simulation.demo_config.demo_duration_minutes
        }