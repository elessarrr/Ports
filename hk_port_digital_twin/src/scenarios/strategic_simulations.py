# Comments for context:
# This module defines strategic simulation scenarios specifically designed for senior management
# decision-making and business intelligence. Unlike operational scenarios that focus on
# day-to-day operations, these strategic scenarios are designed to demonstrate business
# value, ROI, and strategic planning capabilities.
#
# The scenarios extend the existing ScenarioParameters structure but add business-focused
# metrics and strategic considerations that align with executive-level concerns such as
# capacity optimization, maintenance planning, and competitive advantage through AI.

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from .scenario_parameters import ScenarioParameters

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
    """Business metrics and KPIs for strategic scenario evaluation."""
    
    # Revenue and financial metrics
    target_revenue_per_hour: float  # Target revenue per operational hour
    cost_per_delayed_ship: float    # Financial impact of ship delays
    maintenance_cost_per_day: float # Daily cost of maintenance operations
    
    # Efficiency and performance metrics
    target_throughput_increase: float      # Expected throughput improvement (%)
    customer_satisfaction_target: float    # Target customer satisfaction score
    operational_efficiency_target: float   # Target operational efficiency improvement
    
    # Strategic planning metrics
    capacity_utilization_target: float     # Strategic capacity utilization goal
    competitive_advantage_score: float     # Competitive positioning metric
    ai_optimization_benefit: float         # Expected AI optimization benefit (%)

@dataclass
class StrategicScenarioParameters(ScenarioParameters):
    """Extended scenario parameters for strategic business simulations.
    
    Inherits all operational parameters from ScenarioParameters and adds
    business-focused metrics and strategic considerations for executive
    decision-making and ROI analysis.
    """
    
    # Strategic scenario identification
    strategic_type: StrategicScenarioType
    business_objective: str
    executive_summary: str
    
    # Business metrics and KPIs
    business_metrics: StrategicBusinessMetrics
    
    # Strategic planning parameters
    planning_horizon_days: int              # Planning horizon for this scenario
    investment_required: float              # Required investment for optimization
    expected_roi_percentage: float          # Expected return on investment
    
    # Competitive and market factors
    market_demand_multiplier: float         # Market demand adjustment
    competitor_efficiency_baseline: float   # Competitor efficiency for comparison
    
    # Risk and mitigation factors
    risk_factors: List[str]                 # Identified risk factors
    mitigation_strategies: List[str]        # Risk mitigation approaches

# Strategic Scenario Definitions

# Peak Season Capacity Optimization Scenario
# Business Objective: Demonstrate AI-driven capacity optimization during peak demand
PEAK_SEASON_CAPACITY_OPTIMIZATION = StrategicScenarioParameters(
    # Base scenario parameters (inheriting from PEAK_SEASON_PARAMETERS structure)
    scenario_name="Peak Season Capacity Optimization",
    scenario_description="Strategic simulation demonstrating AI-driven capacity optimization "
                        "during peak season to maximize throughput and revenue while "
                        "maintaining service quality. Showcases competitive advantage "
                        "through intelligent resource allocation.",
    
    # Enhanced peak season operational parameters
    arrival_rate_multiplier=1.5,        # 50% increase in ship arrivals
    peak_hour_multiplier=2.3,           # Higher peak hour concentration
    weekend_multiplier=0.9,             # Extended weekend operations
    
    ship_type_distribution={
        'container': 0.85,  # Focus on high-value container ships
        'bulk': 0.12,       # Reduced bulk to prioritize containers
        'mixed': 0.03       # Minimal mixed cargo
    },
    
    container_volume_multipliers={
        'container': 1.4,   # 40% more containers per ship
        'bulk': 1.2,        # Increased bulk volumes
        'mixed': 1.1        # Slightly increased mixed cargo
    },
    average_ship_size_multiplier=1.35,  # 35% larger ships
    
    # AI-optimized operational efficiency
    crane_efficiency_multiplier=1.25,   # 25% AI-driven efficiency gain
    docking_time_multiplier=0.85,       # 15% faster AI-optimized docking
    processing_rate_multiplier=1.3,     # 30% faster AI-optimized processing
    
    # Aggressive capacity utilization
    target_berth_utilization=0.90,      # 90% utilization target
    berth_availability_factor=0.98,     # 98% berth availability
    
    # Strategic priority optimization
    large_ship_priority_boost=1.8,      # Strong preference for large ships
    container_ship_priority_boost=1.6,  # High priority for container ships
    
    primary_months=[10, 11, 12, 1],     # Peak season months
    secondary_months=[9, 2],            # Transition months
    
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
    # Base scenario parameters optimized for maintenance periods
    scenario_name="Maintenance Window Optimization",
    scenario_description="Strategic simulation demonstrating intelligent maintenance "
                        "scheduling to minimize revenue impact while ensuring "
                        "equipment reliability. Showcases AI-driven maintenance "
                        "planning and resource reallocation.",
    
    # Maintenance-adjusted operational parameters
    arrival_rate_multiplier=0.8,        # 20% reduced arrivals during maintenance
    peak_hour_multiplier=1.6,           # Compressed peak hours
    weekend_multiplier=1.2,             # Extended weekend operations
    
    ship_type_distribution={
        'container': 0.70,  # Reduced container focus during maintenance
        'bulk': 0.25,       # Increased bulk cargo (less time-sensitive)
        'mixed': 0.05       # Standard mixed cargo
    },
    
    container_volume_multipliers={
        'container': 0.9,   # Slightly reduced container volumes
        'bulk': 1.1,        # Increased bulk to compensate
        'mixed': 1.0        # Normal mixed cargo
    },
    average_ship_size_multiplier=0.95,  # Slightly smaller ships
    
    # Maintenance-constrained efficiency
    crane_efficiency_multiplier=0.85,   # 15% reduced efficiency (maintenance)
    docking_time_multiplier=1.15,       # 15% longer docking (reduced capacity)
    processing_rate_multiplier=0.9,     # 10% slower processing
    
    # Reduced capacity during maintenance
    target_berth_utilization=0.70,      # 70% utilization (maintenance impact)
    berth_availability_factor=0.80,     # 80% berth availability (20% in maintenance)
    
    # Adjusted priorities for maintenance operations
    large_ship_priority_boost=1.3,      # Moderate preference for large ships
    container_ship_priority_boost=1.2,  # Slight container priority
    
    primary_months=[3, 4, 5, 6],        # Maintenance season months
    secondary_months=[2, 7],            # Transition months
    
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