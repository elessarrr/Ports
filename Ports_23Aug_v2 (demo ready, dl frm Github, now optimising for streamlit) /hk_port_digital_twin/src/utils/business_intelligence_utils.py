# Business Intelligence Utilities
# Shared utilities for ROI calculations, financial analysis, and strategic business metrics
# Used by both operational scenarios and strategic simulations for consistent BI calculations

from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class ROICalculationMethod(Enum):
    """Methods for calculating ROI."""
    SIMPLE_ROI = "simple_roi"
    NPV = "net_present_value"
    IRR = "internal_rate_of_return"
    PAYBACK_PERIOD = "payback_period"
    PROFITABILITY_INDEX = "profitability_index"

@dataclass
class FinancialParameters:
    """Standard financial parameters for calculations."""
    discount_rate: float = 0.08  # 8% discount rate
    revenue_per_teu: float = 150.0  # Revenue per TEU
    inflation_rate: float = 0.03  # 3% annual inflation
    tax_rate: float = 0.25  # 25% corporate tax rate
    risk_free_rate: float = 0.02  # 2% risk-free rate

@dataclass
class ROIAnalysisResult:
    """Comprehensive ROI analysis results."""
    simple_roi_percentage: float
    npv: float
    irr: float
    payback_period_years: float
    profitability_index: float
    total_investment: float
    total_benefit: float
    annual_cash_flows: List[float]
    break_even_year: Optional[int]
    risk_adjusted_roi: float
    confidence_score: float

@dataclass
class BusinessMetricsCalculation:
    """Business metrics calculation results."""
    revenue_per_hour: float
    cost_per_teu: float
    operational_efficiency: float
    capacity_utilization: float
    customer_satisfaction_score: float
    competitive_advantage_index: float
    sustainability_score: float
    risk_mitigation_score: float

class SharedBusinessIntelligence:
    """Shared business intelligence utilities for consistent calculations across the system."""
    
    def __init__(self, financial_params: Optional[FinancialParameters] = None):
        """Initialize with financial parameters."""
        self.financial_params = financial_params or FinancialParameters()
        self.logger = logging.getLogger(__name__)
        
        # Industry benchmarks (would typically be loaded from configuration)
        self.industry_benchmarks = {
            "revenue_per_hour": 25000.0,
            "cost_per_teu": 120.0,
            "berth_utilization": 75.0,
            "throughput_efficiency": 35.0,
            "customer_satisfaction": 8.5,
            "operational_efficiency": 85.0
        }
        
        # Strategic targets
        self.strategic_targets = {
            "revenue_per_hour": 30000.0,
            "cost_per_teu": 100.0,
            "berth_utilization": 85.0,
            "throughput_efficiency": 40.0,
            "customer_satisfaction": 9.0,
            "operational_efficiency": 90.0
        }
    
    def calculate_comprehensive_roi(
        self,
        investment_amount: float,
        annual_benefits: List[float],
        annual_costs: List[float],
        analysis_period_years: int = 10
    ) -> ROIAnalysisResult:
        """Calculate comprehensive ROI analysis with multiple methods.
        
        Args:
            investment_amount: Initial investment amount
            annual_benefits: List of annual benefits for each year
            annual_costs: List of annual costs for each year
            analysis_period_years: Analysis period in years
            
        Returns:
            ROIAnalysisResult with comprehensive financial metrics
        """
        # Ensure lists are the right length
        if len(annual_benefits) < analysis_period_years:
            # Extend with last value
            last_benefit = annual_benefits[-1] if annual_benefits else 0
            annual_benefits.extend([last_benefit] * (analysis_period_years - len(annual_benefits)))
        
        if len(annual_costs) < analysis_period_years:
            last_cost = annual_costs[-1] if annual_costs else 0
            annual_costs.extend([last_cost] * (analysis_period_years - len(annual_costs)))
        
        # Calculate annual cash flows
        annual_cash_flows = []
        for i in range(analysis_period_years):
            if i == 0:
                # Initial investment is negative cash flow
                cash_flow = annual_benefits[i] - annual_costs[i] - investment_amount
            else:
                cash_flow = annual_benefits[i] - annual_costs[i]
            annual_cash_flows.append(cash_flow)
        
        # Simple ROI
        total_benefit = sum(annual_benefits)
        total_cost = sum(annual_costs) + investment_amount
        simple_roi = ((total_benefit - total_cost) / investment_amount * 100) if investment_amount > 0 else 0
        
        # NPV calculation
        npv = self._calculate_npv(annual_cash_flows)
        
        # IRR calculation
        irr = self._calculate_irr(annual_cash_flows)
        
        # Payback period
        payback_period = self._calculate_payback_period(annual_cash_flows, investment_amount)
        
        # Profitability Index
        present_value_benefits = sum(
            benefit / (1 + self.financial_params.discount_rate) ** i
            for i, benefit in enumerate(annual_benefits)
        )
        profitability_index = present_value_benefits / investment_amount if investment_amount > 0 else 0
        
        # Break-even analysis
        break_even_year = self._find_break_even_year(annual_cash_flows, investment_amount)
        
        # Risk-adjusted ROI
        risk_adjusted_roi = self._calculate_risk_adjusted_roi(simple_roi, annual_cash_flows)
        
        # Confidence score based on cash flow stability
        confidence_score = self._calculate_confidence_score(annual_cash_flows)
        
        return ROIAnalysisResult(
            simple_roi_percentage=simple_roi,
            npv=npv,
            irr=irr,
            payback_period_years=payback_period,
            profitability_index=profitability_index,
            total_investment=investment_amount,
            total_benefit=total_benefit,
            annual_cash_flows=annual_cash_flows,
            break_even_year=break_even_year,
            risk_adjusted_roi=risk_adjusted_roi,
            confidence_score=confidence_score
        )
    
    def calculate_business_metrics(
        self,
        operational_data: Dict[str, Any],
        financial_data: Dict[str, Any],
        strategic_context: Optional[Dict[str, Any]] = None
    ) -> BusinessMetricsCalculation:
        """Calculate comprehensive business metrics from operational and financial data.
        
        Args:
            operational_data: Operational metrics (throughput, utilization, etc.)
            financial_data: Financial metrics (revenue, costs, etc.)
            strategic_context: Additional strategic context
            
        Returns:
            BusinessMetricsCalculation with calculated metrics
        """
        # Revenue per hour calculation
        total_revenue = financial_data.get("total_revenue", 0)
        operational_hours = operational_data.get("operational_hours", 24)
        revenue_per_hour = total_revenue / operational_hours if operational_hours > 0 else 0
        
        # Cost per TEU calculation
        total_costs = financial_data.get("total_costs", 0)
        total_teu = operational_data.get("total_teu_handled", 1)
        cost_per_teu = total_costs / total_teu if total_teu > 0 else 0
        
        # Operational efficiency
        actual_throughput = operational_data.get("actual_throughput", 0)
        theoretical_max = operational_data.get("theoretical_max_throughput", 1)
        operational_efficiency = (actual_throughput / theoretical_max * 100) if theoretical_max > 0 else 0
        
        # Capacity utilization
        used_capacity = operational_data.get("used_capacity", 0)
        total_capacity = operational_data.get("total_capacity", 1)
        capacity_utilization = (used_capacity / total_capacity * 100) if total_capacity > 0 else 0
        
        # Customer satisfaction (based on waiting times and service quality)
        avg_waiting_time = operational_data.get("average_waiting_time", 0)
        service_quality = operational_data.get("service_quality_score", 8.0)
        # Simple formula: base score minus penalty for waiting time
        customer_satisfaction = max(0, service_quality - (avg_waiting_time * 0.1))
        
        # Competitive advantage index
        competitive_advantage = self._calculate_competitive_advantage(
            revenue_per_hour, cost_per_teu, operational_efficiency
        )
        
        # Sustainability score
        sustainability_score = self._calculate_sustainability_score(operational_data)
        
        # Risk mitigation score
        risk_mitigation_score = self._calculate_risk_mitigation_score(
            operational_data, strategic_context
        )
        
        return BusinessMetricsCalculation(
            revenue_per_hour=revenue_per_hour,
            cost_per_teu=cost_per_teu,
            operational_efficiency=operational_efficiency,
            capacity_utilization=capacity_utilization,
            customer_satisfaction_score=customer_satisfaction,
            competitive_advantage_index=competitive_advantage,
            sustainability_score=sustainability_score,
            risk_mitigation_score=risk_mitigation_score
        )
    
    def generate_executive_summary(
        self,
        scenario_name: str,
        roi_analysis: ROIAnalysisResult,
        business_metrics: BusinessMetricsCalculation,
        operational_highlights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary for strategic reporting.
        
        Args:
            scenario_name: Name of the scenario
            roi_analysis: ROI analysis results
            business_metrics: Business metrics calculation
            operational_highlights: Key operational highlights
            
        Returns:
            Executive summary dictionary
        """
        # Key achievements
        achievements = []
        if roi_analysis.simple_roi_percentage > 15:
            achievements.append(f"Achieved {roi_analysis.simple_roi_percentage:.1f}% ROI, exceeding 15% target")
        if business_metrics.operational_efficiency > 85:
            achievements.append(f"Operational efficiency of {business_metrics.operational_efficiency:.1f}% exceeds industry standard")
        if business_metrics.customer_satisfaction_score > 8.5:
            achievements.append(f"Customer satisfaction score of {business_metrics.customer_satisfaction_score:.1f} indicates excellent service")
        
        # Critical issues
        issues = []
        if roi_analysis.payback_period_years > 5:
            issues.append(f"Payback period of {roi_analysis.payback_period_years:.1f} years may be too long")
        if business_metrics.cost_per_teu > self.industry_benchmarks["cost_per_teu"]:
            issues.append(f"Cost per TEU of ${business_metrics.cost_per_teu:.2f} exceeds industry benchmark")
        if business_metrics.capacity_utilization < 70:
            issues.append(f"Capacity utilization of {business_metrics.capacity_utilization:.1f}% indicates underutilization")
        
        # Strategic recommendations
        recommendations = []
        if roi_analysis.npv > 0:
            recommendations.append("Investment shows positive NPV - recommend proceeding with implementation")
        if business_metrics.competitive_advantage_index > 1.1:
            recommendations.append("Strong competitive position - consider expanding market share")
        if business_metrics.sustainability_score < 70:
            recommendations.append("Focus on sustainability improvements to meet environmental targets")
        
        return {
            "scenario_name": scenario_name,
            "period": datetime.now().strftime("%Y-%m-%d"),
            "key_achievements": achievements,
            "critical_issues": issues,
            "financial_highlights": {
                "roi_percentage": roi_analysis.simple_roi_percentage,
                "npv": roi_analysis.npv,
                "payback_period": roi_analysis.payback_period_years,
                "revenue_per_hour": business_metrics.revenue_per_hour
            },
            "operational_highlights": {
                "efficiency": business_metrics.operational_efficiency,
                "capacity_utilization": business_metrics.capacity_utilization,
                "customer_satisfaction": business_metrics.customer_satisfaction_score,
                "cost_per_teu": business_metrics.cost_per_teu
            },
            "strategic_recommendations": recommendations,
            "confidence_score": roi_analysis.confidence_score,
            "risk_assessment": {
                "financial_risk": 100 - roi_analysis.confidence_score,
                "operational_risk": 100 - business_metrics.risk_mitigation_score,
                "market_risk": max(0, 100 - business_metrics.competitive_advantage_index * 50)
            }
        }
    
    # Private helper methods
    
    def _calculate_npv(self, cash_flows: List[float]) -> float:
        """Calculate Net Present Value."""
        npv = 0
        for i, cash_flow in enumerate(cash_flows):
            npv += cash_flow / (1 + self.financial_params.discount_rate) ** i
        return npv
    
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method."""
        if not cash_flows or all(cf <= 0 for cf in cash_flows):
            return 0.0
        
        # Initial guess
        irr = 0.1
        tolerance = 1e-6
        max_iterations = 100
        
        for _ in range(max_iterations):
            npv = sum(cf / (1 + irr) ** i for i, cf in enumerate(cash_flows))
            npv_derivative = sum(-i * cf / (1 + irr) ** (i + 1) for i, cf in enumerate(cash_flows))
            
            if abs(npv) < tolerance:
                break
            
            if npv_derivative == 0:
                break
            
            irr = irr - npv / npv_derivative
            
            if irr < -1:  # Prevent negative IRR below -100%
                irr = -0.99
                break
        
        return irr * 100  # Return as percentage
    
    def _calculate_payback_period(self, cash_flows: List[float], initial_investment: float) -> float:
        """Calculate payback period in years."""
        cumulative_cash_flow = -initial_investment
        
        for i, cash_flow in enumerate(cash_flows):
            if i == 0:
                cumulative_cash_flow += cash_flow + initial_investment  # First year includes investment
            else:
                cumulative_cash_flow += cash_flow
            
            if cumulative_cash_flow >= 0:
                # Linear interpolation for fractional year
                if i == 0:
                    return 0
                
                previous_cumulative = cumulative_cash_flow - cash_flow
                fraction = abs(previous_cumulative) / cash_flow
                return i + fraction
        
        return len(cash_flows)  # If never breaks even
    
    def _find_break_even_year(self, cash_flows: List[float], initial_investment: float) -> Optional[int]:
        """Find the year when cumulative cash flow becomes positive."""
        cumulative = -initial_investment
        
        for i, cash_flow in enumerate(cash_flows):
            if i == 0:
                cumulative += cash_flow + initial_investment
            else:
                cumulative += cash_flow
            
            if cumulative >= 0:
                return i + 1  # Return 1-indexed year
        
        return None
    
    def _calculate_risk_adjusted_roi(self, simple_roi: float, cash_flows: List[float]) -> float:
        """Calculate risk-adjusted ROI based on cash flow volatility."""
        if len(cash_flows) < 2:
            return simple_roi
        
        # Calculate volatility (standard deviation of cash flows)
        mean_cash_flow = np.mean(cash_flows)
        volatility = np.std(cash_flows) / abs(mean_cash_flow) if mean_cash_flow != 0 else 0
        
        # Risk adjustment factor (higher volatility = higher risk = lower adjusted ROI)
        risk_factor = max(0.5, 1 - volatility * 0.5)
        
        return simple_roi * risk_factor
    
    def _calculate_confidence_score(self, cash_flows: List[float]) -> float:
        """Calculate confidence score based on cash flow stability."""
        if len(cash_flows) < 2:
            return 50.0
        
        # Factors affecting confidence:
        # 1. Positive cash flows
        positive_flows = sum(1 for cf in cash_flows if cf > 0)
        positive_ratio = positive_flows / len(cash_flows)
        
        # 2. Cash flow stability (lower coefficient of variation = higher confidence)
        mean_cf = np.mean(cash_flows)
        std_cf = np.std(cash_flows)
        cv = std_cf / abs(mean_cf) if mean_cf != 0 else 1
        stability_score = max(0, 1 - cv)
        
        # 3. Growth trend
        if len(cash_flows) > 1:
            growth_trend = (cash_flows[-1] - cash_flows[0]) / abs(cash_flows[0]) if cash_flows[0] != 0 else 0
            growth_score = min(1, max(0, growth_trend + 0.5))
        else:
            growth_score = 0.5
        
        # Combine factors
        confidence = (positive_ratio * 0.4 + stability_score * 0.4 + growth_score * 0.2) * 100
        return min(95, max(5, confidence))  # Clamp between 5% and 95%
    
    def _calculate_competitive_advantage(
        self, 
        revenue_per_hour: float, 
        cost_per_teu: float, 
        operational_efficiency: float
    ) -> float:
        """Calculate competitive advantage index."""
        # Compare against industry benchmarks
        revenue_advantage = revenue_per_hour / self.industry_benchmarks["revenue_per_hour"]
        cost_advantage = self.industry_benchmarks["cost_per_teu"] / cost_per_teu if cost_per_teu > 0 else 1
        efficiency_advantage = operational_efficiency / self.industry_benchmarks["operational_efficiency"]
        
        # Weighted average
        competitive_index = (revenue_advantage * 0.4 + cost_advantage * 0.3 + efficiency_advantage * 0.3)
        return competitive_index
    
    def _calculate_sustainability_score(self, operational_data: Dict[str, Any]) -> float:
        """Calculate sustainability score based on operational data."""
        # Factors: energy efficiency, emissions, waste reduction
        energy_efficiency = operational_data.get("energy_efficiency", 70)
        emission_reduction = operational_data.get("emission_reduction_percentage", 0)
        waste_reduction = operational_data.get("waste_reduction_percentage", 0)
        
        # Normalize and combine
        sustainability = (
            min(100, energy_efficiency) * 0.5 +
            min(100, emission_reduction) * 0.3 +
            min(100, waste_reduction) * 0.2
        )
        
        return sustainability
    
    def _calculate_risk_mitigation_score(
        self, 
        operational_data: Dict[str, Any], 
        strategic_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate risk mitigation score."""
        # Base score
        base_score = 70
        
        # Operational risk factors
        equipment_reliability = operational_data.get("equipment_reliability", 85)
        backup_systems = operational_data.get("backup_systems_available", False)
        staff_training = operational_data.get("staff_training_score", 80)
        
        # Strategic risk factors
        if strategic_context:
            market_diversification = strategic_context.get("market_diversification", 50)
            financial_reserves = strategic_context.get("financial_reserves_months", 6)
        else:
            market_diversification = 50
            financial_reserves = 6
        
        # Calculate score
        risk_score = (
            equipment_reliability * 0.3 +
            (100 if backup_systems else 50) * 0.2 +
            staff_training * 0.2 +
            market_diversification * 0.2 +
            min(100, financial_reserves * 10) * 0.1
        )
        
        return min(100, max(0, risk_score))


# Convenience functions for easy access

def calculate_simple_roi(investment: float, annual_benefit: float, years: int = 5) -> float:
    """Calculate simple ROI percentage."""
    total_benefit = annual_benefit * years
    return ((total_benefit - investment) / investment * 100) if investment > 0 else 0

def calculate_payback_period(investment: float, annual_cash_flow: float) -> float:
    """Calculate simple payback period in years."""
    return investment / annual_cash_flow if annual_cash_flow > 0 else float('inf')

def format_currency(amount: float) -> str:
    """Format currency for display."""
    if abs(amount) >= 1_000_000:
        return f"${amount/1_000_000:.1f}M"
    elif abs(amount) >= 1_000:
        return f"${amount/1_000:.1f}K"
    else:
        return f"${amount:.2f}"

def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format percentage for display."""
    return f"{value:.{decimal_places}f}%"