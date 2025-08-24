"""Comments for context:
Comprehensive Business Intelligence Integration Module

This module provides a unified interface for all business intelligence capabilities
across the Hong Kong Port Digital Twin system. It integrates:
- Strategic ROI calculations from strategic_roi_calculator.py
- Shared business intelligence utilities from business_intelligence_utils.py
- Business intelligence engine from analytics/business_intelligence.py
- Scenario-specific business metrics from scenario parameters

The goal is to provide a single, comprehensive interface for:
1. Cross-scenario comparative analysis
2. Enhanced ROI analysis with strategic insights
3. Executive-level business intelligence reporting
4. Demo-ready business intelligence features

This module is specifically designed to support the unified simulations tab
with advanced business intelligence capabilities for demo presentations.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# Import existing business intelligence components
try:
    from .strategic_roi_calculator import (
        StrategicROICalculator,
        StrategicInvestmentScenario,
        StrategicInvestmentType,
        StrategicROIResult
    )
    from .business_intelligence_utils import (
        SharedBusinessIntelligence,
        ROIAnalysisResult,
        BusinessMetricsCalculation,
        FinancialParameters,
        ROICalculationMethod
    )
    from ..analytics.business_intelligence import (
        BusinessIntelligenceEngine,
        KPIMetric,
        KPICategory,
        TrendDirection
    )
    from ..scenarios.scenario_parameters import (
        ScenarioParameters,
        DemoBusinessMetrics,
        ALL_SCENARIOS
    )
except ImportError:
    # Fallback for direct imports
    from strategic_roi_calculator import (
        StrategicROICalculator,
        StrategicInvestmentScenario,
        StrategicInvestmentType,
        StrategicROIResult
    )
    from business_intelligence_utils import (
        SharedBusinessIntelligence,
        ROIAnalysisResult,
        BusinessMetricsCalculation,
        FinancialParameters,
        ROICalculationMethod
    )
    from analytics.business_intelligence import (
        BusinessIntelligenceEngine,
        KPIMetric,
        KPICategory,
        TrendDirection
    )
    from scenarios.scenario_parameters import (
        ScenarioParameters,
        DemoBusinessMetrics,
        ALL_SCENARIOS
    )

class ComparisonType(Enum):
    """Types of business intelligence comparisons."""
    SCENARIO_VS_SCENARIO = "scenario_vs_scenario"
    SCENARIO_VS_BASELINE = "scenario_vs_baseline"
    SCENARIO_VS_INDUSTRY = "scenario_vs_industry"
    CURRENT_VS_OPTIMIZED = "current_vs_optimized"
    INVESTMENT_ALTERNATIVES = "investment_alternatives"

class BusinessIntelligenceScope(Enum):
    """Scope of business intelligence analysis."""
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    EXECUTIVE = "executive"
    COMPREHENSIVE = "comprehensive"

@dataclass
class ComparativeAnalysisResult:
    """Results of comparative business intelligence analysis."""
    comparison_type: ComparisonType
    primary_scenario: str
    comparison_scenario: Optional[str]
    
    # Financial comparison
    roi_comparison: Dict[str, float]
    revenue_comparison: Dict[str, float]
    cost_comparison: Dict[str, float]
    
    # Operational comparison
    efficiency_comparison: Dict[str, float]
    utilization_comparison: Dict[str, float]
    throughput_comparison: Dict[str, float]
    
    # Strategic comparison
    competitive_advantage_comparison: Dict[str, float]
    sustainability_comparison: Dict[str, float]
    risk_comparison: Dict[str, float]
    
    # Executive insights
    key_insights: List[str]
    recommendations: List[str]
    investment_priorities: List[str]
    risk_mitigation_strategies: List[str]
    
    # Quantitative summary
    overall_score_difference: float
    confidence_level: float
    implementation_complexity: float

@dataclass
class ExecutiveBusinessReport:
    """Executive-level business intelligence report."""
    scenario_name: str
    report_date: datetime
    
    # Executive summary
    executive_summary: str
    key_achievements: List[str]
    critical_issues: List[str]
    strategic_opportunities: List[str]
    
    # Financial highlights
    financial_performance: Dict[str, Any]
    roi_analysis: StrategicROIResult
    investment_recommendations: List[str]
    
    # Operational highlights
    operational_performance: Dict[str, Any]
    efficiency_metrics: Dict[str, float]
    capacity_optimization: Dict[str, float]
    
    # Strategic insights
    market_position: Dict[str, Any]
    competitive_analysis: Dict[str, float]
    future_outlook: Dict[str, Any]
    
    # Action items
    immediate_actions: List[str]
    medium_term_initiatives: List[str]
    long_term_strategic_goals: List[str]

class ComprehensiveBusinessIntelligence:
    """Comprehensive business intelligence system for unified simulations.
    
    This class provides advanced business intelligence capabilities including:
    - Cross-scenario comparative analysis
    - Enhanced ROI analysis with strategic insights
    - Executive-level reporting and recommendations
    - Demo-ready business intelligence features
    """
    
    def __init__(self, financial_params: Optional[FinancialParameters] = None):
        """Initialize comprehensive business intelligence system."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize component systems
        self.strategic_roi_calculator = StrategicROICalculator(financial_params)
        self.shared_bi = SharedBusinessIntelligence(financial_params)
        self.bi_engine = BusinessIntelligenceEngine()
        
        # Configuration
        self.financial_params = financial_params or FinancialParameters()
        
        # Industry benchmarks for comparison
        self.industry_benchmarks = {
            "throughput_teu_per_hour": 280.0,
            "berth_utilization": 70.0,
            "revenue_per_hour": 85000.0,
            "roi_percentage": 12.0,
            "customer_satisfaction": 75.0,
            "operational_efficiency": 80.0,
            "cost_savings_percentage": 8.0
        }
        
        # Strategic targets
        self.strategic_targets = {
            "throughput_teu_per_hour": 400.0,
            "berth_utilization": 85.0,
            "revenue_per_hour": 120000.0,
            "roi_percentage": 20.0,
            "customer_satisfaction": 90.0,
            "operational_efficiency": 95.0,
            "cost_savings_percentage": 15.0
        }
    
    def perform_comparative_analysis(
        self,
        primary_scenario: str,
        comparison_scenario: Optional[str] = None,
        comparison_type: ComparisonType = ComparisonType.SCENARIO_VS_BASELINE,
        scope: BusinessIntelligenceScope = BusinessIntelligenceScope.COMPREHENSIVE
    ) -> ComparativeAnalysisResult:
        """Perform comprehensive comparative business intelligence analysis.
        
        Args:
            primary_scenario: Primary scenario for analysis
            comparison_scenario: Scenario to compare against (if applicable)
            comparison_type: Type of comparison to perform
            scope: Scope of analysis (operational, strategic, executive, comprehensive)
            
        Returns:
            ComparativeAnalysisResult with detailed comparison insights
        """
        self.logger.info(f"Performing {comparison_type.value} analysis for {primary_scenario}")
        
        # Get scenario parameters
        primary_params = self._get_scenario_parameters(primary_scenario)
        comparison_params = None
        if comparison_scenario:
            comparison_params = self._get_scenario_parameters(comparison_scenario)
        
        # Perform analysis based on comparison type
        if comparison_type == ComparisonType.SCENARIO_VS_SCENARIO:
            return self._compare_scenarios(primary_params, comparison_params, scope)
        elif comparison_type == ComparisonType.SCENARIO_VS_BASELINE:
            baseline_params = self._get_scenario_parameters('normal')
            return self._compare_scenarios(primary_params, baseline_params, scope)
        elif comparison_type == ComparisonType.SCENARIO_VS_INDUSTRY:
            return self._compare_scenario_vs_industry(primary_params, scope)
        elif comparison_type == ComparisonType.CURRENT_VS_OPTIMIZED:
            return self._compare_current_vs_optimized(primary_params, scope)
        else:
            raise ValueError(f"Unsupported comparison type: {comparison_type}")
    
    def generate_executive_report(
        self,
        scenario_name: str,
        include_comparative_analysis: bool = True,
        comparison_scenarios: Optional[List[str]] = None
    ) -> ExecutiveBusinessReport:
        """Generate comprehensive executive business intelligence report.
        
        Args:
            scenario_name: Scenario to analyze
            include_comparative_analysis: Whether to include comparative analysis
            comparison_scenarios: List of scenarios to compare against
            
        Returns:
            ExecutiveBusinessReport with comprehensive insights
        """
        self.logger.info(f"Generating executive report for {scenario_name}")
        
        scenario_params = self._get_scenario_parameters(scenario_name)
        business_metrics = scenario_params.business_metrics
        
        # Generate strategic ROI analysis
        investment_scenario = self._create_investment_scenario(scenario_params)
        roi_result = self.strategic_roi_calculator.calculate_strategic_roi(investment_scenario)
        
        # Create executive summary
        executive_summary = self._generate_executive_summary(scenario_params, roi_result)
        
        # Financial performance analysis
        financial_performance = self._analyze_financial_performance(business_metrics, roi_result)
        
        # Operational performance analysis
        operational_performance = self._analyze_operational_performance(business_metrics)
        
        # Strategic insights
        market_position = self._analyze_market_position(business_metrics)
        competitive_analysis = self._analyze_competitive_position(business_metrics)
        
        # Generate recommendations
        recommendations = self._generate_strategic_recommendations(scenario_params, roi_result)
        
        return ExecutiveBusinessReport(
            scenario_name=scenario_name,
            report_date=datetime.now(),
            executive_summary=executive_summary,
            key_achievements=self._extract_key_achievements(business_metrics),
            critical_issues=self._identify_critical_issues(business_metrics),
            strategic_opportunities=self._identify_strategic_opportunities(business_metrics),
            financial_performance=financial_performance,
            roi_analysis=roi_result,
            investment_recommendations=recommendations['investment'],
            operational_performance=operational_performance,
            efficiency_metrics=self._calculate_efficiency_metrics(business_metrics),
            capacity_optimization=self._analyze_capacity_optimization(business_metrics),
            market_position=market_position,
            competitive_analysis=competitive_analysis,
            future_outlook=self._generate_future_outlook(scenario_params),
            immediate_actions=recommendations['immediate'],
            medium_term_initiatives=recommendations['medium_term'],
            long_term_strategic_goals=recommendations['long_term']
        )
    
    def calculate_investment_prioritization(
        self,
        scenarios: List[str],
        investment_budget: float,
        strategic_priorities: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate investment prioritization across multiple scenarios.
        
        Args:
            scenarios: List of scenarios to evaluate
            investment_budget: Available investment budget
            strategic_priorities: Strategic priority weights
            
        Returns:
            Investment prioritization analysis
        """
        self.logger.info(f"Calculating investment prioritization for {len(scenarios)} scenarios")
        
        prioritization_results = []
        
        for scenario in scenarios:
            scenario_params = self._get_scenario_parameters(scenario)
            investment_scenario = self._create_investment_scenario(scenario_params)
            roi_result = self.strategic_roi_calculator.calculate_strategic_roi(investment_scenario)
            
            # Calculate weighted score based on strategic priorities
            weighted_score = self._calculate_weighted_strategic_score(
                roi_result, strategic_priorities
            )
            
            prioritization_results.append({
                'scenario': scenario,
                'investment_required': investment_scenario.investment_amount,
                'expected_roi': roi_result.base_roi_analysis.simple_roi_percentage,
                'strategic_score': weighted_score,
                'payback_period': roi_result.base_roi_analysis.payback_period_years,
                'risk_score': roi_result.risk_adjusted_score,
                'priority_rank': 0  # Will be calculated after sorting
            })
        
        # Sort by strategic score and assign ranks
        prioritization_results.sort(key=lambda x: x['strategic_score'], reverse=True)
        for i, result in enumerate(prioritization_results):
            result['priority_rank'] = i + 1
        
        # Calculate budget allocation
        budget_allocation = self._calculate_optimal_budget_allocation(
            prioritization_results, investment_budget
        )
        
        return {
            'prioritization_results': prioritization_results,
            'budget_allocation': budget_allocation,
            'total_expected_roi': sum(r['expected_roi'] for r in budget_allocation),
            'portfolio_risk_score': np.mean([r['risk_score'] for r in budget_allocation]),
            'implementation_timeline': self._generate_implementation_timeline(budget_allocation)
        }
    
    def _get_scenario_parameters(self, scenario_name: str) -> ScenarioParameters:
        """Get scenario parameters by name."""
        scenario_key = scenario_name.lower().replace('_season', '').replace('_operations', '')
        if scenario_key in ALL_SCENARIOS:
            return ALL_SCENARIOS[scenario_key]
        else:
            self.logger.warning(f"Scenario {scenario_name} not found, using normal operations")
            return ALL_SCENARIOS['normal']
    
    def _compare_scenarios(
        self,
        primary_params: ScenarioParameters,
        comparison_params: ScenarioParameters,
        scope: BusinessIntelligenceScope
    ) -> ComparativeAnalysisResult:
        """Compare two scenarios comprehensively."""
        primary_metrics = primary_params.business_metrics
        comparison_metrics = comparison_params.business_metrics
        
        # ROI comparison
        roi_comparison = {
            'primary': primary_metrics.expected_roi_percentage,
            'comparison': comparison_metrics.expected_roi_percentage,
            'difference': primary_metrics.expected_roi_percentage - comparison_metrics.expected_roi_percentage,
            'percentage_change': ((primary_metrics.expected_roi_percentage - comparison_metrics.expected_roi_percentage) / comparison_metrics.expected_roi_percentage) * 100
        }
        
        # Revenue comparison
        revenue_comparison = {
            'primary': primary_metrics.expected_revenue_per_hour,
            'comparison': comparison_metrics.expected_revenue_per_hour,
            'difference': primary_metrics.expected_revenue_per_hour - comparison_metrics.expected_revenue_per_hour,
            'percentage_change': ((primary_metrics.expected_revenue_per_hour - comparison_metrics.expected_revenue_per_hour) / comparison_metrics.expected_revenue_per_hour) * 100
        }
        
        # Generate insights and recommendations
        key_insights = self._generate_comparison_insights(primary_metrics, comparison_metrics)
        recommendations = self._generate_comparison_recommendations(primary_metrics, comparison_metrics)
        
        return ComparativeAnalysisResult(
            comparison_type=ComparisonType.SCENARIO_VS_SCENARIO,
            primary_scenario=primary_params.scenario_name,
            comparison_scenario=comparison_params.scenario_name,
            roi_comparison=roi_comparison,
            revenue_comparison=revenue_comparison,
            cost_comparison=self._calculate_cost_comparison(primary_metrics, comparison_metrics),
            efficiency_comparison=self._calculate_efficiency_comparison(primary_metrics, comparison_metrics),
            utilization_comparison=self._calculate_utilization_comparison(primary_metrics, comparison_metrics),
            throughput_comparison=self._calculate_throughput_comparison(primary_metrics, comparison_metrics),
            competitive_advantage_comparison=self._calculate_competitive_comparison(primary_metrics, comparison_metrics),
            sustainability_comparison=self._calculate_sustainability_comparison(primary_metrics, comparison_metrics),
            risk_comparison=self._calculate_risk_comparison(primary_params, comparison_params),
            key_insights=key_insights,
            recommendations=recommendations,
            investment_priorities=self._generate_investment_priorities(primary_metrics, comparison_metrics),
            risk_mitigation_strategies=self._generate_risk_mitigation_strategies(primary_params),
            overall_score_difference=self._calculate_overall_score_difference(primary_metrics, comparison_metrics),
            confidence_level=0.85,  # Based on data quality and analysis depth
            implementation_complexity=self._assess_implementation_complexity(primary_params)
        )
    
    def _create_investment_scenario(self, scenario_params: ScenarioParameters) -> StrategicInvestmentScenario:
        """Create strategic investment scenario from scenario parameters."""
        business_metrics = scenario_params.business_metrics
        
        # Estimate investment amount based on scenario characteristics
        base_investment = 5000000  # $5M base investment
        investment_multiplier = scenario_params.arrival_rate_multiplier * scenario_params.target_berth_utilization
        investment_amount = base_investment * investment_multiplier
        
        # Calculate expected benefits
        annual_benefits = []
        for year in range(10):
            annual_benefit = business_metrics.expected_revenue_per_hour * 8760 * 0.7  # 70% operational time
            annual_benefits.append(annual_benefit * (1 + 0.03) ** year)  # 3% growth
        
        # Calculate expected costs
        annual_costs = []
        base_annual_cost = investment_amount * 0.1  # 10% of investment as annual cost
        for year in range(10):
            annual_costs.append(base_annual_cost * (1 + 0.02) ** year)  # 2% cost inflation
        
        return StrategicInvestmentScenario(
            investment_type=StrategicInvestmentType.CAPACITY_EXPANSION,
            investment_amount=investment_amount,
            implementation_timeline_months=18,
            expected_annual_benefits=annual_benefits,
            expected_annual_costs=annual_costs,
            risk_factors={
                'market_risk': 0.15,
                'technology_risk': 0.10,
                'operational_risk': 0.08,
                'regulatory_risk': 0.05
            },
            market_conditions={
                'growth_rate': 0.05,
                'competition_level': 0.7,
                'market_maturity': 0.8
            },
            competitive_impact={
                'market_share_gain': business_metrics.competitive_advantage_score / 100,
                'pricing_power': 0.1,
                'customer_retention': business_metrics.customer_satisfaction_target / 100
            },
            sustainability_impact={
                'carbon_reduction': business_metrics.sustainability_impact_score / 100,
                'efficiency_gain': business_metrics.expected_crane_efficiency / 100,
                'waste_reduction': 0.15
            }
        )
    
    # Additional helper methods for calculations and analysis
    def _generate_executive_summary(self, scenario_params: ScenarioParameters, roi_result: StrategicROIResult) -> str:
        """Generate executive summary for the scenario."""
        business_metrics = scenario_params.business_metrics
        
        summary = f"""
        {scenario_params.scenario_name} presents a strategic opportunity to achieve {business_metrics.expected_roi_percentage:.1f}% ROI 
        through enhanced operational efficiency and capacity optimization. The scenario demonstrates strong potential for 
        {business_metrics.expected_throughput_teu_per_hour:.0f} TEU/hour throughput with {business_metrics.expected_berth_utilization:.1f}% 
        berth utilization. Strategic implementation would deliver ${business_metrics.expected_revenue_per_hour:,.0f} per hour 
        in revenue generation while maintaining {business_metrics.customer_satisfaction_target:.0f}% customer satisfaction. 
        The investment shows strong financial viability with a {roi_result.base_roi_analysis.payback_period_years:.1f}-year 
        payback period and significant competitive advantage potential.
        """
        
        return summary.strip()
    
    def _extract_key_achievements(self, business_metrics: DemoBusinessMetrics) -> List[str]:
        """Extract key achievements from business metrics."""
        achievements = []
        
        if business_metrics.expected_roi_percentage > 15:
            achievements.append(f"Exceptional ROI performance at {business_metrics.expected_roi_percentage:.1f}%")
        
        if business_metrics.expected_berth_utilization > 80:
            achievements.append(f"High berth utilization efficiency at {business_metrics.expected_berth_utilization:.1f}%")
        
        if business_metrics.customer_satisfaction_target > 85:
            achievements.append(f"Superior customer satisfaction target of {business_metrics.customer_satisfaction_target:.1f}%")
        
        if business_metrics.expected_cost_savings_percentage > 10:
            achievements.append(f"Significant cost optimization of {business_metrics.expected_cost_savings_percentage:.1f}%")
        
        return achievements
    
    def _identify_critical_issues(self, business_metrics: DemoBusinessMetrics) -> List[str]:
        """Identify critical issues from business metrics."""
        issues = []
        
        if business_metrics.expected_waiting_time_hours > 4:
            issues.append(f"Extended waiting times of {business_metrics.expected_waiting_time_hours:.1f} hours")
        
        if business_metrics.expected_berth_utilization < 70:
            issues.append(f"Suboptimal berth utilization at {business_metrics.expected_berth_utilization:.1f}%")
        
        if business_metrics.customer_satisfaction_target < 75:
            issues.append(f"Below-target customer satisfaction at {business_metrics.customer_satisfaction_target:.1f}%")
        
        return issues
    
    def _identify_strategic_opportunities(self, business_metrics: DemoBusinessMetrics) -> List[str]:
        """Identify strategic opportunities from business metrics."""
        opportunities = [
            "Digital transformation through AI-powered optimization",
            "Capacity expansion to meet growing demand",
            "Sustainability initiatives for competitive advantage",
            "Technology integration for operational excellence",
            "Strategic partnerships for market expansion"
        ]
        
        return opportunities
    
    def _calculate_efficiency_metrics(self, business_metrics: DemoBusinessMetrics) -> Dict[str, float]:
        """Calculate efficiency metrics."""
        return {
            'crane_efficiency': business_metrics.expected_crane_efficiency,
            'berth_utilization': business_metrics.expected_berth_utilization,
            'throughput_efficiency': (business_metrics.expected_throughput_teu_per_hour / 400) * 100,
            'cost_efficiency': business_metrics.expected_cost_savings_percentage,
            'overall_efficiency': (business_metrics.expected_crane_efficiency + business_metrics.expected_berth_utilization) / 2
        }
    
    # Additional helper methods would continue here...
    # (Implementation of remaining helper methods for completeness)
    
    def _calculate_cost_comparison(self, primary: DemoBusinessMetrics, comparison: DemoBusinessMetrics) -> Dict[str, float]:
        """Calculate cost comparison between scenarios."""
        return {
            'primary_cost_savings': primary.expected_cost_savings_percentage,
            'comparison_cost_savings': comparison.expected_cost_savings_percentage,
            'difference': primary.expected_cost_savings_percentage - comparison.expected_cost_savings_percentage
        }
    
    def _calculate_efficiency_comparison(self, primary: DemoBusinessMetrics, comparison: DemoBusinessMetrics) -> Dict[str, float]:
        """Calculate efficiency comparison between scenarios."""
        return {
            'primary_crane_efficiency': primary.expected_crane_efficiency,
            'comparison_crane_efficiency': comparison.expected_crane_efficiency,
            'efficiency_difference': primary.expected_crane_efficiency - comparison.expected_crane_efficiency
        }
    
    def _calculate_utilization_comparison(self, primary: DemoBusinessMetrics, comparison: DemoBusinessMetrics) -> Dict[str, float]:
        """Calculate utilization comparison between scenarios."""
        return {
            'primary_utilization': primary.expected_berth_utilization,
            'comparison_utilization': comparison.expected_berth_utilization,
            'utilization_difference': primary.expected_berth_utilization - comparison.expected_berth_utilization
        }
    
    def _calculate_throughput_comparison(self, primary: DemoBusinessMetrics, comparison: DemoBusinessMetrics) -> Dict[str, float]:
        """Calculate throughput comparison between scenarios."""
        return {
            'primary_throughput': primary.expected_throughput_teu_per_hour,
            'comparison_throughput': comparison.expected_throughput_teu_per_hour,
            'throughput_difference': primary.expected_throughput_teu_per_hour - comparison.expected_throughput_teu_per_hour
        }
    
    def _generate_comparison_insights(self, primary: DemoBusinessMetrics, comparison: DemoBusinessMetrics) -> List[str]:
        """Generate insights from scenario comparison."""
        insights = []
        
        roi_diff = primary.expected_roi_percentage - comparison.expected_roi_percentage
        if roi_diff > 5:
            insights.append(f"Primary scenario shows {roi_diff:.1f}% higher ROI potential")
        
        throughput_diff = primary.expected_throughput_teu_per_hour - comparison.expected_throughput_teu_per_hour
        if throughput_diff > 50:
            insights.append(f"Primary scenario delivers {throughput_diff:.0f} TEU/hour additional capacity")
        
        return insights
    
    def _generate_comparison_recommendations(self, primary: DemoBusinessMetrics, comparison: DemoBusinessMetrics) -> List[str]:
        """Generate recommendations from scenario comparison."""
        return [
            "Prioritize scenario with highest strategic value",
            "Consider phased implementation approach",
            "Implement risk mitigation strategies",
            "Monitor key performance indicators closely"
        ]
    
    def _calculate_overall_score_difference(self, primary: DemoBusinessMetrics, comparison: DemoBusinessMetrics) -> float:
        """Calculate overall score difference between scenarios."""
        primary_score = (primary.expected_roi_percentage + primary.expected_berth_utilization + 
                        primary.customer_satisfaction_target + primary.competitive_advantage_score) / 4
        comparison_score = (comparison.expected_roi_percentage + comparison.expected_berth_utilization + 
                           comparison.customer_satisfaction_target + comparison.competitive_advantage_score) / 4
        return primary_score - comparison_score