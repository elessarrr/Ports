# Comments for context:
# This module defines strategic simulation scenarios specifically designed for senior management
# decision-making and business intelligence. Unlike operational scenarios that focus on
# day-to-day operations, these strategic scenarios are designed to demonstrate business
# value, ROI, and strategic planning capabilities.
#
# The scenarios extend the existing ScenarioParameters structure but add business-focused
# metrics and strategic considerations that align with executive-level concerns such as
# capacity optimization, maintenance planning, and competitive advantage through AI.

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

from .scenario_parameters import (
    ScenarioParameters, 
    PEAK_SEASON_PARAMETERS,
    NORMAL_OPERATIONS_PARAMETERS,
    LOW_SEASON_PARAMETERS
)

class StrategicScenarioType(Enum):
    """Strategic scenario types for business intelligence and executive decision-making."""
    PEAK_SEASON_CAPACITY_OPTIMIZATION = "peak_season_capacity_optimization"
    MAINTENANCE_WINDOW_OPTIMIZATION = "maintenance_window_optimization"
    AI_VS_TRADITIONAL_COMPARISON = "ai_vs_traditional_comparison"
    CAPACITY_EXPANSION_PLANNING = "capacity_expansion_planning"
    COMPETITIVE_ADVANTAGE_ANALYSIS = "competitive_advantage_analysis"

class BusinessMetricType(Enum):
    """Business metrics tracked in strategic scenarios."""
    REVENUE_PER_HOUR = "revenue_per_hour"
    BERTH_UTILIZATION_EFFICIENCY = "berth_utilization_efficiency"
    CUSTOMER_SATISFACTION_SCORE = "customer_satisfaction_score"
    OPERATIONAL_COST_REDUCTION = "operational_cost_reduction"
    THROUGHPUT_OPTIMIZATION = "throughput_optimization"
    MAINTENANCE_COST_SAVINGS = "maintenance_cost_savings"
    AI_ROI_METRICS = "ai_roi_metrics"

@dataclass
class StrategicBusinessMetrics:
    """Business metrics for strategic scenario evaluation.
    
    Defines key performance indicators and financial metrics used to
    evaluate the business impact of strategic operational decisions.
    """
    target_revenue_per_hour: float = 0.0  # Target revenue per operational hour
    cost_per_delayed_ship: float = 0.0    # Financial impact of ship delays
    maintenance_cost_per_day: float = 0.0 # Daily cost of maintenance operations
    
    # Performance targets
    target_throughput_increase: float = 0.0      # Expected throughput improvement (%)
    customer_satisfaction_target: float = 0.0    # Target customer satisfaction score
    operational_efficiency_target: float = 0.0   # Target operational efficiency improvement
    
    # Strategic planning metrics
    capacity_utilization_target: float = 0.0     # Strategic capacity utilization goal
    competitive_advantage_score: float = 0.0     # Competitive positioning metric
    ai_optimization_benefit: float = 0.0         # Expected AI optimization benefit (%)

@dataclass
class StrategicScenarioParameters:
    """Strategic scenario parameters for business simulations.
    
    Contains both operational parameters and strategic business metrics
    for executive decision-making and ROI analysis.
    """
    
    # Base scenario parameters
    base_scenario: ScenarioParameters = None
    
    # Strategic scenario identification
    strategic_type: StrategicScenarioType = StrategicScenarioType.PEAK_SEASON_CAPACITY_OPTIMIZATION
    business_objective: str = ""
    executive_summary: str = ""
    
    # Business metrics and KPIs
    business_metrics: Optional[StrategicBusinessMetrics] = None
    
    # Strategic planning parameters
    planning_horizon_days: int = 90              # Planning horizon for this scenario
    investment_required: float = 0.0              # Required investment for optimization
    expected_roi_percentage: float = 0.0          # Expected return on investment
    
    # Competitive and market factors
    market_demand_multiplier: float = 1.0         # Market demand adjustment
    competitor_efficiency_baseline: float = 0.0   # Competitor efficiency for comparison
    
    # Risk and mitigation factors
    risk_factors: List[str] = field(default_factory=list)                 # Identified risk factors
    mitigation_strategies: List[str] = field(default_factory=list)        # Risk mitigation approaches
    
    # Convenience properties to access base scenario attributes
    @property
    def scenario_name(self) -> str:
        return self.base_scenario.scenario_name
    
    @property
    def scenario_description(self) -> str:
        return self.base_scenario.scenario_description

# Strategic Scenario Definitions

# Peak Season Capacity Optimization Scenario
# Business Objective: Demonstrate AI-driven capacity optimization during peak demand
PEAK_SEASON_CAPACITY_OPTIMIZATION = StrategicScenarioParameters(
    # Base scenario from existing parameters
    base_scenario=PEAK_SEASON_PARAMETERS,
    
    # Strategic scenario specific parameters
    strategic_type=StrategicScenarioType.PEAK_SEASON_CAPACITY_OPTIMIZATION,
    business_objective="Maximize revenue and throughput during peak season through "
                      "AI-driven capacity optimization while maintaining service quality",
    executive_summary="Demonstrates 25-30% throughput increase and 20% revenue growth "
                     "through intelligent resource allocation and predictive optimization",
    
    business_metrics=StrategicBusinessMetrics(
        target_revenue_per_hour=50000.0,       # $50K per hour target
        cost_per_delayed_ship=25000.0,         # $25K cost per delayed ship
        maintenance_cost_per_day=15000.0,      # $15K daily maintenance cost
        target_throughput_increase=28.0,       # 28% throughput increase
        customer_satisfaction_target=95.0,     # 95% satisfaction target
        operational_efficiency_target=25.0,    # 25% efficiency improvement
        capacity_utilization_target=90.0,      # 90% capacity utilization
        competitive_advantage_score=85.0,      # High competitive advantage
        ai_optimization_benefit=30.0           # 30% AI benefit
    ),
    
    planning_horizon_days=120,              # 4-month planning horizon
    investment_required=2500000.0,          # $2.5M investment
    expected_roi_percentage=180.0,          # 180% ROI over 2 years
    
    market_demand_multiplier=1.4,           # 40% higher market demand
    competitor_efficiency_baseline=75.0,    # Competitor baseline efficiency
    
    risk_factors=[
        "Equipment failure during peak operations",
        "Weather disruptions affecting ship arrivals",
        "Labor shortage during high-demand periods",
        "Unexpected surge in cargo volumes"
    ],
    mitigation_strategies=[
        "Predictive maintenance scheduling",
        "Weather-adaptive resource allocation",
        "Cross-trained workforce deployment",
        "Dynamic capacity scaling protocols"
    ]
)

# Maintenance Window Optimization Scenario
# Business Objective: Minimize revenue loss during necessary maintenance operations
MAINTENANCE_WINDOW_OPTIMIZATION = StrategicScenarioParameters(
    # Base scenario from existing parameters
    base_scenario=NORMAL_OPERATIONS_PARAMETERS,
    
    # Strategic scenario specific parameters
    strategic_type=StrategicScenarioType.MAINTENANCE_WINDOW_OPTIMIZATION,
    business_objective="Minimize revenue loss during maintenance while ensuring "
                      "equipment reliability and operational continuity",
    executive_summary="Reduces maintenance-related revenue loss by 40% through "
                     "intelligent scheduling and resource optimization",
    
    business_metrics=StrategicBusinessMetrics(
        target_revenue_per_hour=35000.0,       # $35K per hour (reduced capacity)
        cost_per_delayed_ship=30000.0,         # $30K cost per delayed ship
        maintenance_cost_per_day=25000.0,      # $25K daily maintenance cost
        target_throughput_increase=15.0,       # 15% throughput optimization
        customer_satisfaction_target=88.0,     # 88% satisfaction (maintenance impact)
        operational_efficiency_target=20.0,    # 20% efficiency improvement
        capacity_utilization_target=70.0,      # 70% capacity utilization
        competitive_advantage_score=75.0,      # Moderate competitive advantage
        ai_optimization_benefit=25.0           # 25% AI benefit
    ),
    
    planning_horizon_days=90,               # 3-month planning horizon
    investment_required=1800000.0,          # $1.8M investment
    expected_roi_percentage=150.0,          # 150% ROI over 18 months
    
    market_demand_multiplier=0.9,           # 10% reduced market demand
    competitor_efficiency_baseline=70.0,    # Competitor baseline during maintenance
    
    risk_factors=[
        "Extended maintenance duration",
        "Critical equipment failure",
        "Maintenance crew availability",
        "Customer service disruption"
    ],
    mitigation_strategies=[
        "Predictive maintenance algorithms",
        "Redundant equipment deployment",
        "Flexible workforce scheduling",
        "Proactive customer communication"
    ]
)

# Strategic scenarios dictionary for easy access
STRATEGIC_SCENARIOS = {
    'peak_capacity_optimization': PEAK_SEASON_CAPACITY_OPTIMIZATION,
    'maintenance_optimization': MAINTENANCE_WINDOW_OPTIMIZATION
}

# Strategic scenario aliases
STRATEGIC_SCENARIO_ALIASES = {
    'peak_optimization': 'peak_capacity_optimization',
    'capacity_optimization': 'peak_capacity_optimization',
    'peak_strategic': 'peak_capacity_optimization',
    'maintenance_strategic': 'maintenance_optimization',
    'maintenance_planning': 'maintenance_optimization',
    'maintenance_window': 'maintenance_optimization'
}

def get_strategic_scenario(scenario_key: str) -> Optional[StrategicScenarioParameters]:
    """Get strategic scenario parameters by key or alias.
    
    Args:
        scenario_key: Scenario key or alias
        
    Returns:
        StrategicScenarioParameters if found, None otherwise
    """
    # Check direct key first
    if scenario_key in STRATEGIC_SCENARIOS:
        return STRATEGIC_SCENARIOS[scenario_key]
    
    # Check aliases
    if scenario_key in STRATEGIC_SCENARIO_ALIASES:
        return STRATEGIC_SCENARIOS[STRATEGIC_SCENARIO_ALIASES[scenario_key]]
    
    return None

def list_strategic_scenarios() -> List[str]:
    """Get list of available strategic scenario keys.
    
    Returns:
        List of strategic scenario keys
    """
    return list(STRATEGIC_SCENARIOS.keys())

def get_business_metrics_summary(scenario: StrategicScenarioParameters) -> Dict[str, float]:
    """Extract business metrics summary from strategic scenario.
    
    Args:
        scenario: Strategic scenario parameters
        
    Returns:
        Dictionary of key business metrics
    """
    return {
        'target_revenue_per_hour': scenario.business_metrics.target_revenue_per_hour,
        'expected_roi_percentage': scenario.expected_roi_percentage,
        'throughput_increase': scenario.business_metrics.target_throughput_increase,
        'efficiency_improvement': scenario.business_metrics.operational_efficiency_target,
        'capacity_utilization': scenario.business_metrics.capacity_utilization_target,
        'competitive_advantage': scenario.business_metrics.competitive_advantage_score,
        'ai_optimization_benefit': scenario.business_metrics.ai_optimization_benefit
    }