# Strategic ROI Calculator
# Specialized ROI calculations for strategic port scenarios
# Integrates with shared business intelligence utilities

from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .business_intelligence_utils import (
    SharedBusinessIntelligence,
    ROIAnalysisResult,
    BusinessMetricsCalculation,
    FinancialParameters
)

class StrategicInvestmentType(Enum):
    """Types of strategic investments."""
    CAPACITY_EXPANSION = "capacity_expansion"
    TECHNOLOGY_UPGRADE = "technology_upgrade"
    AUTOMATION = "automation"
    INFRASTRUCTURE = "infrastructure"
    SUSTAINABILITY = "sustainability"
    DIGITALIZATION = "digitalization"
    WORKFORCE_DEVELOPMENT = "workforce_development"
    MARKET_EXPANSION = "market_expansion"

@dataclass
class StrategicInvestmentScenario:
    """Strategic investment scenario definition."""
    investment_type: StrategicInvestmentType
    investment_amount: float
    implementation_timeline_months: int
    expected_annual_benefits: List[float]
    expected_annual_costs: List[float]
    risk_factors: Dict[str, float]
    market_conditions: Dict[str, Any]
    competitive_impact: Dict[str, float]
    sustainability_impact: Dict[str, float]
    
@dataclass
class StrategicROIResult:
    """Strategic ROI analysis results with additional strategic metrics."""
    base_roi_analysis: ROIAnalysisResult
    strategic_value_score: float
    market_positioning_impact: float
    competitive_advantage_gain: float
    sustainability_contribution: float
    risk_adjusted_score: float
    strategic_alignment_score: float
    implementation_complexity: float
    stakeholder_impact_score: float
    long_term_value_potential: float

class StrategicROICalculator:
    """Advanced ROI calculator for strategic port investments."""
    
    def __init__(self, financial_params: Optional[FinancialParameters] = None):
        """Initialize strategic ROI calculator."""
        self.bi_engine = SharedBusinessIntelligence(financial_params)
        self.logger = logging.getLogger(__name__)
        
        # Strategic weights for different factors
        self.strategic_weights = {
            "financial_performance": 0.35,
            "market_positioning": 0.20,
            "competitive_advantage": 0.15,
            "sustainability": 0.15,
            "risk_mitigation": 0.10,
            "stakeholder_value": 0.05
        }
        
        # Investment type multipliers
        self.investment_multipliers = {
            StrategicInvestmentType.CAPACITY_EXPANSION: {
                "revenue_impact": 1.3,
                "market_share_impact": 1.2,
                "risk_factor": 1.1
            },
            StrategicInvestmentType.TECHNOLOGY_UPGRADE: {
                "efficiency_impact": 1.4,
                "cost_reduction": 1.2,
                "competitive_advantage": 1.3
            },
            StrategicInvestmentType.AUTOMATION: {
                "cost_reduction": 1.5,
                "efficiency_impact": 1.6,
                "workforce_impact": 0.8
            },
            StrategicInvestmentType.SUSTAINABILITY: {
                "brand_value": 1.3,
                "regulatory_compliance": 1.4,
                "long_term_viability": 1.5
            },
            StrategicInvestmentType.DIGITALIZATION: {
                "operational_efficiency": 1.4,
                "data_value": 1.6,
                "customer_experience": 1.3
            }
        }
    
    def calculate_strategic_roi(
        self,
        scenario: StrategicInvestmentScenario,
        current_performance: Dict[str, Any],
        strategic_objectives: Dict[str, Any],
        analysis_period_years: int = 10
    ) -> StrategicROIResult:
        """Calculate comprehensive strategic ROI analysis.
        
        Args:
            scenario: Strategic investment scenario
            current_performance: Current operational and financial performance
            strategic_objectives: Strategic objectives and targets
            analysis_period_years: Analysis period in years
            
        Returns:
            StrategicROIResult with comprehensive strategic analysis
        """
        # Base ROI calculation
        base_roi = self.bi_engine.calculate_comprehensive_roi(
            scenario.investment_amount,
            scenario.expected_annual_benefits,
            scenario.expected_annual_costs,
            analysis_period_years
        )
        
        # Strategic value calculations
        strategic_value_score = self._calculate_strategic_value_score(
            scenario, current_performance, strategic_objectives
        )
        
        market_positioning_impact = self._calculate_market_positioning_impact(
            scenario, current_performance
        )
        
        competitive_advantage_gain = self._calculate_competitive_advantage_gain(
            scenario, current_performance
        )
        
        sustainability_contribution = self._calculate_sustainability_contribution(
            scenario
        )
        
        risk_adjusted_score = self._calculate_risk_adjusted_score(
            base_roi, scenario.risk_factors
        )
        
        strategic_alignment_score = self._calculate_strategic_alignment_score(
            scenario, strategic_objectives
        )
        
        implementation_complexity = self._calculate_implementation_complexity(
            scenario
        )
        
        stakeholder_impact_score = self._calculate_stakeholder_impact_score(
            scenario, current_performance
        )
        
        long_term_value_potential = self._calculate_long_term_value_potential(
            scenario, base_roi
        )
        
        return StrategicROIResult(
            base_roi_analysis=base_roi,
            strategic_value_score=strategic_value_score,
            market_positioning_impact=market_positioning_impact,
            competitive_advantage_gain=competitive_advantage_gain,
            sustainability_contribution=sustainability_contribution,
            risk_adjusted_score=risk_adjusted_score,
            strategic_alignment_score=strategic_alignment_score,
            implementation_complexity=implementation_complexity,
            stakeholder_impact_score=stakeholder_impact_score,
            long_term_value_potential=long_term_value_potential
        )
    
    def compare_strategic_scenarios(
        self,
        scenarios: List[StrategicInvestmentScenario],
        current_performance: Dict[str, Any],
        strategic_objectives: Dict[str, Any],
        budget_constraint: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compare multiple strategic investment scenarios.
        
        Args:
            scenarios: List of strategic investment scenarios
            current_performance: Current performance metrics
            strategic_objectives: Strategic objectives
            budget_constraint: Optional budget constraint
            
        Returns:
            Comparison analysis with rankings and recommendations
        """
        results = []
        
        for scenario in scenarios:
            if budget_constraint and scenario.investment_amount > budget_constraint:
                continue
                
            roi_result = self.calculate_strategic_roi(
                scenario, current_performance, strategic_objectives
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_strategic_score(roi_result)
            
            results.append({
                "scenario": scenario,
                "roi_result": roi_result,
                "overall_score": overall_score,
                "investment_efficiency": overall_score / scenario.investment_amount * 1000000  # Per million invested
            })
        
        # Sort by overall score
        results.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_strategic_recommendations(results, budget_constraint)
        
        return {
            "scenario_results": results,
            "top_recommendation": results[0] if results else None,
            "budget_optimized": self._optimize_for_budget(results, budget_constraint),
            "risk_balanced": self._find_risk_balanced_option(results),
            "strategic_recommendations": recommendations,
            "portfolio_suggestions": self._suggest_investment_portfolio(results, budget_constraint)
        }
    
    def generate_strategic_business_case(
        self,
        scenario: StrategicInvestmentScenario,
        roi_result: StrategicROIResult,
        current_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive business case for strategic investment.
        
        Args:
            scenario: Strategic investment scenario
            roi_result: Strategic ROI analysis results
            current_performance: Current performance baseline
            
        Returns:
            Comprehensive business case document
        """
        # Executive summary
        executive_summary = self.bi_engine.generate_executive_summary(
            f"Strategic Investment: {scenario.investment_type.value.replace('_', ' ').title()}",
            roi_result.base_roi_analysis,
            BusinessMetricsCalculation(
                revenue_per_hour=current_performance.get("revenue_per_hour", 0),
                cost_per_teu=current_performance.get("cost_per_teu", 0),
                operational_efficiency=current_performance.get("operational_efficiency", 0),
                capacity_utilization=current_performance.get("capacity_utilization", 0),
                customer_satisfaction_score=current_performance.get("customer_satisfaction", 0),
                competitive_advantage_index=roi_result.competitive_advantage_gain,
                sustainability_score=roi_result.sustainability_contribution,
                risk_mitigation_score=roi_result.risk_adjusted_score
            ),
            current_performance
        )
        
        # Strategic justification
        strategic_justification = {
            "strategic_alignment": {
                "score": roi_result.strategic_alignment_score,
                "description": self._get_alignment_description(roi_result.strategic_alignment_score)
            },
            "competitive_positioning": {
                "current_position": current_performance.get("market_position", "Unknown"),
                "projected_improvement": roi_result.competitive_advantage_gain,
                "market_impact": roi_result.market_positioning_impact
            },
            "sustainability_impact": {
                "environmental_benefits": scenario.sustainability_impact,
                "sustainability_score": roi_result.sustainability_contribution,
                "regulatory_compliance": self._assess_regulatory_compliance(scenario)
            }
        }
        
        # Risk analysis
        risk_analysis = {
            "implementation_risks": self._identify_implementation_risks(scenario),
            "financial_risks": self._identify_financial_risks(roi_result.base_roi_analysis),
            "market_risks": self._identify_market_risks(scenario),
            "mitigation_strategies": self._suggest_risk_mitigation_strategies(scenario),
            "overall_risk_score": 100 - roi_result.risk_adjusted_score
        }
        
        # Implementation roadmap
        implementation_roadmap = self._create_implementation_roadmap(scenario)
        
        # Success metrics
        success_metrics = self._define_success_metrics(scenario, roi_result)
        
        return {
            "executive_summary": executive_summary,
            "investment_overview": {
                "investment_type": scenario.investment_type.value,
                "total_investment": scenario.investment_amount,
                "implementation_timeline": scenario.implementation_timeline_months,
                "expected_roi": roi_result.base_roi_analysis.simple_roi_percentage,
                "payback_period": roi_result.base_roi_analysis.payback_period_years,
                "npv": roi_result.base_roi_analysis.npv
            },
            "strategic_justification": strategic_justification,
            "financial_analysis": {
                "base_roi_analysis": roi_result.base_roi_analysis,
                "strategic_value_score": roi_result.strategic_value_score,
                "long_term_value_potential": roi_result.long_term_value_potential
            },
            "risk_analysis": risk_analysis,
            "implementation_roadmap": implementation_roadmap,
            "success_metrics": success_metrics,
            "stakeholder_impact": {
                "score": roi_result.stakeholder_impact_score,
                "analysis": self._analyze_stakeholder_impact(scenario)
            }
        }
    
    # Private helper methods
    
    def _calculate_strategic_value_score(
        self,
        scenario: StrategicInvestmentScenario,
        current_performance: Dict[str, Any],
        strategic_objectives: Dict[str, Any]
    ) -> float:
        """Calculate strategic value score."""
        # Base score from investment type
        base_score = 70
        
        # Investment type specific bonuses
        multipliers = self.investment_multipliers.get(scenario.investment_type, {})
        
        # Strategic alignment bonus
        alignment_bonus = 0
        for objective, target in strategic_objectives.items():
            if objective in multipliers:
                alignment_bonus += multipliers[objective] * 10
        
        # Market conditions adjustment
        market_adjustment = scenario.market_conditions.get("growth_potential", 1.0) * 10
        
        strategic_score = base_score + alignment_bonus + market_adjustment
        return min(100, max(0, strategic_score))
    
    def _calculate_market_positioning_impact(self, scenario: StrategicInvestmentScenario, current_performance: Dict[str, Any]) -> float:
        """Calculate market positioning impact."""
        current_market_share = current_performance.get("market_share", 0.15)
        
        # Investment type impact on market positioning
        impact_factors = {
            StrategicInvestmentType.CAPACITY_EXPANSION: 1.4,
            StrategicInvestmentType.TECHNOLOGY_UPGRADE: 1.2,
            StrategicInvestmentType.AUTOMATION: 1.1,
            StrategicInvestmentType.SUSTAINABILITY: 1.3,
            StrategicInvestmentType.DIGITALIZATION: 1.25
        }
        
        impact_factor = impact_factors.get(scenario.investment_type, 1.0)
        market_growth = scenario.market_conditions.get("market_growth_rate", 0.05)
        
        positioning_impact = (impact_factor - 1) * 100 + market_growth * 20
        return min(50, max(0, positioning_impact))
    
    def _calculate_competitive_advantage_gain(self, scenario: StrategicInvestmentScenario, current_performance: Dict[str, Any]) -> float:
        """Calculate competitive advantage gain."""
        multipliers = self.investment_multipliers.get(scenario.investment_type, {})
        competitive_impact = scenario.competitive_impact
        
        # Base competitive advantage
        base_advantage = current_performance.get("competitive_index", 1.0)
        
        # Calculate improvement
        efficiency_gain = multipliers.get("efficiency_impact", 1.0) - 1
        cost_reduction = multipliers.get("cost_reduction", 1.0) - 1
        market_impact = competitive_impact.get("market_differentiation", 0)
        
        advantage_gain = (efficiency_gain + cost_reduction + market_impact) * 25
        return min(100, max(0, advantage_gain))
    
    def _calculate_sustainability_contribution(self, scenario: StrategicInvestmentScenario) -> float:
        """Calculate sustainability contribution score."""
        sustainability_impact = scenario.sustainability_impact
        
        # Key sustainability factors
        carbon_reduction = sustainability_impact.get("carbon_footprint_reduction", 0) * 30
        energy_efficiency = sustainability_impact.get("energy_efficiency_improvement", 0) * 25
        waste_reduction = sustainability_impact.get("waste_reduction", 0) * 20
        water_conservation = sustainability_impact.get("water_conservation", 0) * 15
        biodiversity_impact = sustainability_impact.get("biodiversity_protection", 0) * 10
        
        total_score = carbon_reduction + energy_efficiency + waste_reduction + water_conservation + biodiversity_impact
        return min(100, max(0, total_score))
    
    def _calculate_risk_adjusted_score(self, base_roi: ROIAnalysisResult, risk_factors: Dict[str, float]) -> float:
        """Calculate risk-adjusted score."""
        base_confidence = base_roi.confidence_score
        
        # Risk factor adjustments
        implementation_risk = risk_factors.get("implementation_complexity", 0.3) * 20
        market_risk = risk_factors.get("market_volatility", 0.2) * 15
        technology_risk = risk_factors.get("technology_obsolescence", 0.1) * 10
        regulatory_risk = risk_factors.get("regulatory_changes", 0.15) * 12
        
        total_risk_penalty = implementation_risk + market_risk + technology_risk + regulatory_risk
        risk_adjusted = base_confidence - total_risk_penalty
        
        return min(100, max(0, risk_adjusted))
    
    def _calculate_strategic_alignment_score(self, scenario: StrategicInvestmentScenario, strategic_objectives: Dict[str, Any]) -> float:
        """Calculate strategic alignment score."""
        alignment_score = 0
        total_objectives = len(strategic_objectives)
        
        if total_objectives == 0:
            return 50  # Neutral score if no objectives defined
        
        # Check alignment with each strategic objective
        investment_benefits = {
            StrategicInvestmentType.CAPACITY_EXPANSION: ["growth", "market_share", "revenue"],
            StrategicInvestmentType.TECHNOLOGY_UPGRADE: ["efficiency", "innovation", "competitiveness"],
            StrategicInvestmentType.AUTOMATION: ["cost_reduction", "efficiency", "scalability"],
            StrategicInvestmentType.SUSTAINABILITY: ["environmental", "compliance", "brand_value"],
            StrategicInvestmentType.DIGITALIZATION: ["data_driven", "customer_experience", "innovation"]
        }
        
        relevant_benefits = investment_benefits.get(scenario.investment_type, [])
        
        for objective in strategic_objectives:
            if any(benefit in objective.lower() for benefit in relevant_benefits):
                alignment_score += 100 / total_objectives
        
        return min(100, max(0, alignment_score))
    
    def _calculate_implementation_complexity(self, scenario: StrategicInvestmentScenario) -> float:
        """Calculate implementation complexity score."""
        # Base complexity by investment type
        complexity_scores = {
            StrategicInvestmentType.CAPACITY_EXPANSION: 60,
            StrategicInvestmentType.TECHNOLOGY_UPGRADE: 70,
            StrategicInvestmentType.AUTOMATION: 80,
            StrategicInvestmentType.INFRASTRUCTURE: 75,
            StrategicInvestmentType.SUSTAINABILITY: 55,
            StrategicInvestmentType.DIGITALIZATION: 85,
            StrategicInvestmentType.WORKFORCE_DEVELOPMENT: 45
        }
        
        base_complexity = complexity_scores.get(scenario.investment_type, 60)
        
        # Timeline adjustment
        timeline_factor = min(1.5, scenario.implementation_timeline_months / 12)
        timeline_adjustment = (1 - timeline_factor) * 20
        
        # Investment size adjustment
        size_factor = min(2.0, scenario.investment_amount / 10_000_000)  # $10M baseline
        size_adjustment = size_factor * 10
        
        complexity = base_complexity + timeline_adjustment + size_adjustment
        return min(100, max(0, complexity))
    
    def _calculate_stakeholder_impact_score(self, scenario: StrategicInvestmentScenario, current_performance: Dict[str, Any]) -> float:
        """Calculate stakeholder impact score."""
        # Stakeholder categories and their weights
        stakeholder_impacts = {
            "employees": 0.25,
            "customers": 0.30,
            "shareholders": 0.25,
            "community": 0.15,
            "regulators": 0.05
        }
        
        total_impact = 0
        
        # Investment type specific impacts
        if scenario.investment_type == StrategicInvestmentType.AUTOMATION:
            total_impact += 60 * stakeholder_impacts["employees"]  # Potential job impact
            total_impact += 85 * stakeholder_impacts["customers"]  # Better service
            total_impact += 90 * stakeholder_impacts["shareholders"]  # Cost savings
        elif scenario.investment_type == StrategicInvestmentType.SUSTAINABILITY:
            total_impact += 80 * stakeholder_impacts["employees"]  # Pride in work
            total_impact += 75 * stakeholder_impacts["customers"]  # Brand preference
            total_impact += 70 * stakeholder_impacts["shareholders"]  # Long-term value
            total_impact += 95 * stakeholder_impacts["community"]  # Environmental benefit
            total_impact += 90 * stakeholder_impacts["regulators"]  # Compliance
        else:
            # Default positive impact
            for weight in stakeholder_impacts.values():
                total_impact += 75 * weight
        
        return min(100, max(0, total_impact))
    
    def _calculate_long_term_value_potential(self, scenario: StrategicInvestmentScenario, base_roi: ROIAnalysisResult) -> float:
        """Calculate long-term value potential."""
        # Base value from ROI
        base_value = min(100, max(0, base_roi.simple_roi_percentage))
        
        # Investment type long-term multipliers
        long_term_multipliers = {
            StrategicInvestmentType.TECHNOLOGY_UPGRADE: 1.3,
            StrategicInvestmentType.DIGITALIZATION: 1.4,
            StrategicInvestmentType.SUSTAINABILITY: 1.5,
            StrategicInvestmentType.AUTOMATION: 1.2,
            StrategicInvestmentType.CAPACITY_EXPANSION: 1.1
        }
        
        multiplier = long_term_multipliers.get(scenario.investment_type, 1.0)
        
        # Market growth potential
        market_growth = scenario.market_conditions.get("long_term_growth_potential", 1.0)
        
        long_term_value = base_value * multiplier * market_growth
        return min(100, max(0, long_term_value))
    
    def _calculate_overall_strategic_score(self, roi_result: StrategicROIResult) -> float:
        """Calculate overall strategic score using weighted factors."""
        # Normalize ROI to 0-100 scale
        financial_score = min(100, max(0, roi_result.base_roi_analysis.simple_roi_percentage * 2))
        
        overall_score = (
            financial_score * self.strategic_weights["financial_performance"] +
            roi_result.market_positioning_impact * self.strategic_weights["market_positioning"] +
            roi_result.competitive_advantage_gain * self.strategic_weights["competitive_advantage"] +
            roi_result.sustainability_contribution * self.strategic_weights["sustainability"] +
            roi_result.risk_adjusted_score * self.strategic_weights["risk_mitigation"] +
            roi_result.stakeholder_impact_score * self.strategic_weights["stakeholder_value"]
        )
        
        return overall_score
    
    def _generate_strategic_recommendations(self, results: List[Dict], budget_constraint: Optional[float]) -> List[str]:
        """Generate strategic recommendations based on analysis results."""
        recommendations = []
        
        if not results:
            return ["No viable investment scenarios identified within constraints."]
        
        top_result = results[0]
        
        # Top recommendation
        recommendations.append(
            f"Recommend {top_result['scenario'].investment_type.value.replace('_', ' ').title()} "
            f"with overall score of {top_result['overall_score']:.1f} and "
            f"{top_result['roi_result'].base_roi_analysis.simple_roi_percentage:.1f}% ROI."
        )
        
        # Budget considerations
        if budget_constraint:
            affordable_options = [r for r in results if r['scenario'].investment_amount <= budget_constraint]
            if affordable_options:
                best_affordable = affordable_options[0]
                recommendations.append(
                    f"Within budget constraint of ${budget_constraint:,.0f}, "
                    f"{best_affordable['scenario'].investment_type.value.replace('_', ' ').title()} "
                    f"offers the best value with {best_affordable['investment_efficiency']:.1f} "
                    f"points per million invested."
                )
        
        # Risk considerations
        low_risk_options = [r for r in results if r['roi_result'].risk_adjusted_score > 70]
        if low_risk_options:
            recommendations.append(
                f"For risk-averse strategy, consider "
                f"{low_risk_options[0]['scenario'].investment_type.value.replace('_', ' ').title()} "
                f"with risk-adjusted score of {low_risk_options[0]['roi_result'].risk_adjusted_score:.1f}."
            )
        
        return recommendations
    
    def _optimize_for_budget(self, results: List[Dict], budget_constraint: Optional[float]) -> Optional[Dict]:
        """Find budget-optimized investment option."""
        if not budget_constraint:
            return None
        
        affordable_options = [r for r in results if r['scenario'].investment_amount <= budget_constraint]
        if not affordable_options:
            return None
        
        # Sort by investment efficiency (score per dollar)
        affordable_options.sort(key=lambda x: x['investment_efficiency'], reverse=True)
        return affordable_options[0]
    
    def _find_risk_balanced_option(self, results: List[Dict]) -> Optional[Dict]:
        """Find risk-balanced investment option."""
        if not results:
            return None
        
        # Find option with best balance of return and risk
        risk_balanced = max(
            results,
            key=lambda x: (x['roi_result'].base_roi_analysis.simple_roi_percentage * 
                          x['roi_result'].risk_adjusted_score / 100)
        )
        
        return risk_balanced
    
    def _suggest_investment_portfolio(self, results: List[Dict], budget_constraint: Optional[float]) -> List[Dict]:
        """Suggest portfolio of investments if budget allows."""
        if not budget_constraint or not results:
            return []
        
        # Sort by investment efficiency
        sorted_results = sorted(results, key=lambda x: x['investment_efficiency'], reverse=True)
        
        portfolio = []
        remaining_budget = budget_constraint
        
        for result in sorted_results:
            if result['scenario'].investment_amount <= remaining_budget:
                portfolio.append(result)
                remaining_budget -= result['scenario'].investment_amount
                
                if len(portfolio) >= 3:  # Limit portfolio size
                    break
        
        return portfolio
    
    def _get_alignment_description(self, score: float) -> str:
        """Get description for strategic alignment score."""
        if score >= 80:
            return "Excellent alignment with strategic objectives"
        elif score >= 60:
            return "Good alignment with strategic objectives"
        elif score >= 40:
            return "Moderate alignment with strategic objectives"
        else:
            return "Limited alignment with strategic objectives"
    
    def _assess_regulatory_compliance(self, scenario: StrategicInvestmentScenario) -> Dict[str, Any]:
        """Assess regulatory compliance impact."""
        compliance_impact = {
            "environmental_regulations": "positive" if scenario.investment_type == StrategicInvestmentType.SUSTAINABILITY else "neutral",
            "safety_regulations": "positive" if scenario.investment_type in [StrategicInvestmentType.AUTOMATION, StrategicInvestmentType.TECHNOLOGY_UPGRADE] else "neutral",
            "labor_regulations": "requires_attention" if scenario.investment_type == StrategicInvestmentType.AUTOMATION else "neutral",
            "data_protection": "positive" if scenario.investment_type == StrategicInvestmentType.DIGITALIZATION else "neutral"
        }
        
        return compliance_impact
    
    def _identify_implementation_risks(self, scenario: StrategicInvestmentScenario) -> List[str]:
        """Identify implementation risks."""
        risks = []
        
        if scenario.implementation_timeline_months < 6:
            risks.append("Aggressive timeline may lead to implementation challenges")
        
        if scenario.investment_amount > 50_000_000:
            risks.append("Large investment size increases execution complexity")
        
        if scenario.investment_type == StrategicInvestmentType.AUTOMATION:
            risks.append("Workforce resistance to automation changes")
            risks.append("Technology integration complexity")
        
        if scenario.investment_type == StrategicInvestmentType.DIGITALIZATION:
            risks.append("Cybersecurity vulnerabilities")
            risks.append("Data migration challenges")
        
        return risks
    
    def _identify_financial_risks(self, roi_analysis: ROIAnalysisResult) -> List[str]:
        """Identify financial risks."""
        risks = []
        
        if roi_analysis.payback_period_years > 7:
            risks.append(f"Long payback period of {roi_analysis.payback_period_years:.1f} years")
        
        if roi_analysis.confidence_score < 60:
            risks.append("Low confidence in financial projections")
        
        if roi_analysis.npv < 0:
            risks.append("Negative Net Present Value indicates potential value destruction")
        
        return risks
    
    def _identify_market_risks(self, scenario: StrategicInvestmentScenario) -> List[str]:
        """Identify market risks."""
        risks = []
        
        market_volatility = scenario.market_conditions.get("volatility", 0.2)
        if market_volatility > 0.3:
            risks.append("High market volatility may affect returns")
        
        competitive_pressure = scenario.competitive_impact.get("competitive_response", 0.5)
        if competitive_pressure > 0.7:
            risks.append("Strong competitive response expected")
        
        return risks
    
    def _suggest_risk_mitigation_strategies(self, scenario: StrategicInvestmentScenario) -> List[str]:
        """Suggest risk mitigation strategies."""
        strategies = []
        
        strategies.append("Implement phased rollout to reduce implementation risk")
        strategies.append("Establish clear success metrics and monitoring systems")
        strategies.append("Develop contingency plans for key risk scenarios")
        
        if scenario.investment_type == StrategicInvestmentType.AUTOMATION:
            strategies.append("Invest in workforce retraining and change management")
        
        if scenario.investment_type == StrategicInvestmentType.DIGITALIZATION:
            strategies.append("Implement robust cybersecurity measures")
            strategies.append("Ensure data backup and recovery procedures")
        
        return strategies
    
    def _create_implementation_roadmap(self, scenario: StrategicInvestmentScenario) -> Dict[str, Any]:
        """Create implementation roadmap."""
        total_months = scenario.implementation_timeline_months
        
        phases = {
            "planning_and_design": {
                "duration_months": max(1, total_months * 0.2),
                "key_activities": ["Detailed planning", "Design finalization", "Resource allocation"]
            },
            "procurement_and_setup": {
                "duration_months": max(1, total_months * 0.3),
                "key_activities": ["Vendor selection", "Equipment procurement", "Site preparation"]
            },
            "implementation": {
                "duration_months": max(1, total_months * 0.4),
                "key_activities": ["System installation", "Integration", "Testing"]
            },
            "deployment_and_optimization": {
                "duration_months": max(1, total_months * 0.1),
                "key_activities": ["Go-live", "Performance optimization", "Training completion"]
            }
        }
        
        return {
            "total_timeline_months": total_months,
            "phases": phases,
            "critical_milestones": [
                "Design approval",
                "Procurement completion",
                "System integration",
                "Go-live",
                "Performance targets achieved"
            ]
        }
    
    def _define_success_metrics(self, scenario: StrategicInvestmentScenario, roi_result: StrategicROIResult) -> Dict[str, Any]:
        """Define success metrics for the investment."""
        return {
            "financial_metrics": {
                "target_roi": roi_result.base_roi_analysis.simple_roi_percentage,
                "target_payback_period": roi_result.base_roi_analysis.payback_period_years,
                "target_npv": roi_result.base_roi_analysis.npv
            },
            "operational_metrics": {
                "efficiency_improvement": "15% increase in operational efficiency",
                "capacity_utilization": "Target 85% capacity utilization",
                "cost_reduction": "10% reduction in cost per TEU"
            },
            "strategic_metrics": {
                "market_positioning": f"Improve market position by {roi_result.market_positioning_impact:.1f} points",
                "competitive_advantage": f"Gain {roi_result.competitive_advantage_gain:.1f} points in competitive advantage",
                "sustainability": f"Achieve {roi_result.sustainability_contribution:.1f} sustainability score"
            },
            "timeline_metrics": {
                "implementation_completion": f"Complete implementation within {scenario.implementation_timeline_months} months",
                "benefit_realization": "Achieve 50% of projected benefits within first year"
            }
        }
    
    def _analyze_stakeholder_impact(self, scenario: StrategicInvestmentScenario) -> Dict[str, str]:
        """Analyze impact on different stakeholders."""
        impact_analysis = {
            "employees": "Positive - Enhanced working conditions and skill development opportunities",
            "customers": "Positive - Improved service quality and efficiency",
            "shareholders": "Positive - Strong ROI and long-term value creation",
            "community": "Positive - Economic growth and environmental benefits",
            "regulators": "Positive - Enhanced compliance and industry leadership"
        }
        
        # Customize based on investment type
        if scenario.investment_type == StrategicInvestmentType.AUTOMATION:
            impact_analysis["employees"] = "Mixed - Job displacement concerns offset by upskilling opportunities"
        
        if scenario.investment_type == StrategicInvestmentType.SUSTAINABILITY:
            impact_analysis["community"] = "Highly Positive - Significant environmental and social benefits"
            impact_analysis["regulators"] = "Highly Positive - Proactive compliance and industry leadership"
        
        return impact_analysis