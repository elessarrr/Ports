#!/usr/bin/env python3
"""
Investment Planning Simulation Module

This module implements dynamic capacity planning and investment simulation for the Hong Kong Port Digital Twin.
It models various investment scenarios and calculates ROI for capacity expansion decisions.

Author: AI Assistant
Date: 2024
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvestmentType(Enum):
    """Types of investment options"""
    NEW_BERTH = "new_berth"
    CRANE_UPGRADE = "crane_upgrade"
    AUTOMATION = "automation"
    STORAGE_EXPANSION = "storage_expansion"
    DIGITAL_INFRASTRUCTURE = "digital_infrastructure"
    ENVIRONMENTAL_UPGRADE = "environmental_upgrade"

class InvestmentPriority(Enum):
    """Investment priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DemandProjection:
    """Represents demand projections for capacity planning"""
    year: int
    vessel_arrivals_per_day: float
    average_vessel_size: float  # TEU
    peak_season_multiplier: float
    growth_rate: float  # Annual growth rate
    confidence_level: float  # 0.0 to 1.0
    
    def get_annual_throughput(self) -> float:
        """Calculate annual throughput in TEU"""
        return self.vessel_arrivals_per_day * self.average_vessel_size * 365

@dataclass
class InvestmentOption:
    """Represents an investment option with its characteristics"""
    investment_id: str
    name: str
    investment_type: InvestmentType
    priority: InvestmentPriority
    capital_cost: float  # Initial investment
    annual_operating_cost: float
    implementation_time_months: int
    capacity_increase: float  # Additional TEU capacity per year
    efficiency_improvement: float  # Percentage improvement (0.0 to 1.0)
    lifespan_years: int
    description: str = ""
    prerequisites: List[str] = field(default_factory=list)
    environmental_impact: str = "neutral"
    
    def calculate_annual_benefit(self, current_throughput: float, revenue_per_teu: float) -> float:
        """Calculate annual financial benefit"""
        # Direct capacity benefit
        capacity_benefit = min(self.capacity_increase, current_throughput * 0.2) * revenue_per_teu
        
        # Efficiency benefit
        efficiency_benefit = current_throughput * self.efficiency_improvement * revenue_per_teu * 0.1
        
        return capacity_benefit + efficiency_benefit

@dataclass
class InvestmentScenario:
    """Represents a complete investment scenario"""
    scenario_id: str
    name: str
    description: str
    investments: List[InvestmentOption]
    implementation_timeline: Dict[str, int]  # investment_id -> start_month
    total_capital_cost: float = 0.0
    total_annual_operating_cost: float = 0.0
    
    def __post_init__(self):
        self.total_capital_cost = sum(inv.capital_cost for inv in self.investments)
        self.total_annual_operating_cost = sum(inv.annual_operating_cost for inv in self.investments)

class InvestmentPlanner:
    """Main class for investment planning and ROI analysis"""
    
    def __init__(self):
        self.investment_options: Dict[str, InvestmentOption] = {}
        self.demand_projections: List[DemandProjection] = []
        self.scenarios: Dict[str, InvestmentScenario] = {}
        self.financial_parameters = {
            'discount_rate': 0.08,  # 8% discount rate
            'revenue_per_teu': 150.0,  # Revenue per TEU
            'inflation_rate': 0.03,  # 3% annual inflation
            'tax_rate': 0.25  # 25% corporate tax rate
        }
        self._initialize_investment_options()
        self._initialize_demand_projections()
    
    def _initialize_investment_options(self):
        """Initialize predefined investment options"""
        options = [
            InvestmentOption(
                investment_id="new_berth_1",
                name="New Container Berth (400m)",
                investment_type=InvestmentType.NEW_BERTH,
                priority=InvestmentPriority.HIGH,
                capital_cost=250_000_000,  # $250M
                annual_operating_cost=5_000_000,  # $5M
                implementation_time_months=36,
                capacity_increase=800_000,  # 800K TEU/year
                efficiency_improvement=0.0,
                lifespan_years=30,
                description="New 400m container berth with modern infrastructure",
                environmental_impact="moderate"
            ),
            InvestmentOption(
                investment_id="crane_upgrade_1",
                name="Automated Crane System",
                investment_type=InvestmentType.CRANE_UPGRADE,
                priority=InvestmentPriority.MEDIUM,
                capital_cost=50_000_000,  # $50M
                annual_operating_cost=2_000_000,  # $2M
                implementation_time_months=18,
                capacity_increase=200_000,  # 200K TEU/year
                efficiency_improvement=0.15,  # 15% efficiency improvement
                lifespan_years=20,
                description="Automated crane system for existing berths",
                environmental_impact="positive"
            ),
            InvestmentOption(
                investment_id="automation_1",
                name="Port Automation System",
                investment_type=InvestmentType.AUTOMATION,
                priority=InvestmentPriority.HIGH,
                capital_cost=100_000_000,  # $100M
                annual_operating_cost=8_000_000,  # $8M
                implementation_time_months=24,
                capacity_increase=300_000,  # 300K TEU/year
                efficiency_improvement=0.25,  # 25% efficiency improvement
                lifespan_years=15,
                description="Comprehensive port automation and AI systems",
                prerequisites=["crane_upgrade_1"],
                environmental_impact="positive"
            ),
            InvestmentOption(
                investment_id="storage_expansion_1",
                name="Container Storage Expansion",
                investment_type=InvestmentType.STORAGE_EXPANSION,
                priority=InvestmentPriority.MEDIUM,
                capital_cost=30_000_000,  # $30M
                annual_operating_cost=1_500_000,  # $1.5M
                implementation_time_months=12,
                capacity_increase=150_000,  # 150K TEU/year
                efficiency_improvement=0.05,  # 5% efficiency improvement
                lifespan_years=25,
                description="Additional container storage areas",
                environmental_impact="neutral"
            ),
            InvestmentOption(
                investment_id="digital_infra_1",
                name="Digital Infrastructure Upgrade",
                investment_type=InvestmentType.DIGITAL_INFRASTRUCTURE,
                priority=InvestmentPriority.MEDIUM,
                capital_cost=20_000_000,  # $20M
                annual_operating_cost=3_000_000,  # $3M
                implementation_time_months=15,
                capacity_increase=0,  # No direct capacity increase
                efficiency_improvement=0.20,  # 20% efficiency improvement
                lifespan_years=10,
                description="5G network, IoT sensors, and digital twin infrastructure",
                environmental_impact="positive"
            ),
            InvestmentOption(
                investment_id="environmental_1",
                name="Green Port Initiative",
                investment_type=InvestmentType.ENVIRONMENTAL_UPGRADE,
                priority=InvestmentPriority.LOW,
                capital_cost=40_000_000,  # $40M
                annual_operating_cost=2_000_000,  # $2M
                implementation_time_months=20,
                capacity_increase=0,  # No direct capacity increase
                efficiency_improvement=0.10,  # 10% efficiency improvement
                lifespan_years=20,
                description="Solar panels, electric equipment, emission reduction systems",
                environmental_impact="very_positive"
            )
        ]
        
        for option in options:
            self.investment_options[option.investment_id] = option
    
    def _initialize_demand_projections(self):
        """Initialize demand projections for the next 10 years"""
        base_year = datetime.now().year
        base_arrivals = 25.0  # vessels per day
        base_vessel_size = 8000.0  # TEU
        
        for year_offset in range(10):
            year = base_year + year_offset
            growth_rate = 0.05 - (year_offset * 0.002)  # Declining growth rate
            
            projection = DemandProjection(
                year=year,
                vessel_arrivals_per_day=base_arrivals * (1 + growth_rate) ** year_offset,
                average_vessel_size=base_vessel_size * (1 + 0.02) ** year_offset,  # 2% annual size increase
                peak_season_multiplier=1.3,
                growth_rate=growth_rate,
                confidence_level=max(0.6, 0.9 - year_offset * 0.05)  # Decreasing confidence
            )
            
            self.demand_projections.append(projection)
    
    def create_investment_scenario(
        self,
        scenario_id: str,
        name: str,
        description: str,
        investment_ids: List[str],
        implementation_timeline: Optional[Dict[str, int]] = None
    ) -> InvestmentScenario:
        """Create a new investment scenario"""
        
        investments = []
        for inv_id in investment_ids:
            if inv_id in self.investment_options:
                investments.append(self.investment_options[inv_id])
            else:
                logger.warning(f"Investment option {inv_id} not found")
        
        # Default timeline if not provided
        if implementation_timeline is None:
            implementation_timeline = {inv_id: i * 6 for i, inv_id in enumerate(investment_ids)}
        
        scenario = InvestmentScenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            investments=investments,
            implementation_timeline=implementation_timeline
        )
        
        self.scenarios[scenario_id] = scenario
        return scenario
    
    def calculate_roi_analysis(
        self,
        scenario: InvestmentScenario,
        analysis_period_years: int = 20
    ) -> Dict[str, Any]:
        """Calculate comprehensive ROI analysis for investment scenario"""
        
        logger.info(f"Calculating ROI for scenario: {scenario.name}")
        
        # Initialize cash flow analysis
        cash_flows = []
        cumulative_capacity = 0
        current_throughput = self.demand_projections[0].get_annual_throughput()
        
        for year in range(analysis_period_years):
            year_cash_flow = {
                'year': year,
                'capital_expenditure': 0,
                'operating_cost': 0,
                'revenue_benefit': 0,
                'net_cash_flow': 0,
                'cumulative_capacity': cumulative_capacity
            }
            
            # Calculate capital expenditures for this year
            for investment in scenario.investments:
                implementation_month = scenario.implementation_timeline.get(investment.investment_id, 0)
                implementation_year = implementation_month // 12
                
                if implementation_year == year:
                    year_cash_flow['capital_expenditure'] += investment.capital_cost
                    cumulative_capacity += investment.capacity_increase
                
                # Operating costs start after implementation
                if year >= implementation_year:
                    year_cash_flow['operating_cost'] += investment.annual_operating_cost
            
            # Calculate revenue benefits
            if year < len(self.demand_projections):
                projected_throughput = self.demand_projections[year].get_annual_throughput()
            else:
                # Extrapolate beyond projection period
                last_projection = self.demand_projections[-1]
                projected_throughput = last_projection.get_annual_throughput() * (1 + last_projection.growth_rate) ** (year - len(self.demand_projections) + 1)
            
            # Revenue benefit from capacity and efficiency improvements
            for investment in scenario.investments:
                implementation_month = scenario.implementation_timeline.get(investment.investment_id, 0)
                implementation_year = implementation_month // 12
                
                if year >= implementation_year:
                    year_cash_flow['revenue_benefit'] += investment.calculate_annual_benefit(
                        projected_throughput, self.financial_parameters['revenue_per_teu']
                    )
            
            # Apply inflation and taxes
            inflation_factor = (1 + self.financial_parameters['inflation_rate']) ** year
            year_cash_flow['operating_cost'] *= inflation_factor
            year_cash_flow['revenue_benefit'] *= inflation_factor
            
            # Calculate net cash flow (after tax)
            taxable_income = year_cash_flow['revenue_benefit'] - year_cash_flow['operating_cost']
            tax = max(0, taxable_income * self.financial_parameters['tax_rate'])
            
            year_cash_flow['net_cash_flow'] = (
                -year_cash_flow['capital_expenditure'] +
                year_cash_flow['revenue_benefit'] -
                year_cash_flow['operating_cost'] -
                tax
            )
            
            year_cash_flow['cumulative_capacity'] = cumulative_capacity
            cash_flows.append(year_cash_flow)
        
        # Calculate financial metrics
        financial_metrics = self._calculate_financial_metrics(cash_flows)
        
        # Calculate capacity metrics
        capacity_metrics = self._calculate_capacity_metrics(scenario, cash_flows)
        
        # Generate risk assessment
        risk_assessment = self._assess_investment_risks(scenario)
        
        return {
            'scenario': scenario,
            'cash_flows': cash_flows,
            'financial_metrics': financial_metrics,
            'capacity_metrics': capacity_metrics,
            'risk_assessment': risk_assessment,
            'recommendations': self._generate_investment_recommendations(scenario, financial_metrics, risk_assessment)
        }
    
    def _calculate_financial_metrics(self, cash_flows: List[Dict]) -> Dict[str, float]:
        """Calculate key financial metrics"""
        discount_rate = self.financial_parameters['discount_rate']
        
        # Net Present Value (NPV)
        npv = sum(
            cf['net_cash_flow'] / (1 + discount_rate) ** cf['year']
            for cf in cash_flows
        )
        
        # Internal Rate of Return (IRR) - simplified calculation
        irr = self._calculate_irr([cf['net_cash_flow'] for cf in cash_flows])
        
        # Payback Period
        payback_period = self._calculate_payback_period(cash_flows)
        
        # Return on Investment (ROI)
        total_investment = sum(cf['capital_expenditure'] for cf in cash_flows)
        total_benefit = sum(cf['revenue_benefit'] for cf in cash_flows)
        roi = (total_benefit - total_investment) / total_investment if total_investment > 0 else 0
        
        # Profitability Index
        present_value_benefits = sum(
            cf['revenue_benefit'] / (1 + discount_rate) ** cf['year']
            for cf in cash_flows
        )
        present_value_costs = sum(
            (cf['capital_expenditure'] + cf['operating_cost']) / (1 + discount_rate) ** cf['year']
            for cf in cash_flows
        )
        profitability_index = present_value_benefits / present_value_costs if present_value_costs > 0 else 0
        
        return {
            'npv': npv,
            'irr': irr,
            'payback_period': payback_period,
            'roi': roi,
            'profitability_index': profitability_index,
            'total_investment': total_investment,
            'total_benefit': total_benefit
        }
    
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method"""
        if not cash_flows or all(cf >= 0 for cf in cash_flows):
            return 0.0
        
        # Initial guess
        rate = 0.1
        
        for _ in range(100):  # Maximum iterations
            npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
            npv_derivative = sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))
            
            if abs(npv) < 1e-6:  # Convergence threshold
                return rate
            
            if npv_derivative == 0:
                break
            
            rate = rate - npv / npv_derivative
            
            if rate < -0.99:  # Prevent negative rates below -99%
                rate = -0.99
        
        return rate
    
    def _calculate_payback_period(self, cash_flows: List[Dict]) -> float:
        """Calculate payback period in years"""
        cumulative_cash_flow = 0
        
        for cf in cash_flows:
            cumulative_cash_flow += cf['net_cash_flow']
            if cumulative_cash_flow >= 0:
                return cf['year'] + 1
        
        return float('inf')  # Never pays back
    
    def _calculate_capacity_metrics(self, scenario: InvestmentScenario, cash_flows: List[Dict]) -> Dict[str, Any]:
        """Calculate capacity-related metrics"""
        final_capacity = cash_flows[-1]['cumulative_capacity'] if cash_flows else 0
        
        # Capacity utilization over time
        utilization_timeline = []
        for i, cf in enumerate(cash_flows):
            if i < len(self.demand_projections):
                demand = self.demand_projections[i].get_annual_throughput()
            else:
                # Extrapolate demand
                last_projection = self.demand_projections[-1]
                demand = last_projection.get_annual_throughput() * (1 + last_projection.growth_rate) ** (i - len(self.demand_projections) + 1)
            
            base_capacity = 2_000_000  # Assume 2M TEU base capacity
            total_capacity = base_capacity + cf['cumulative_capacity']
            utilization = min(1.0, demand / total_capacity) if total_capacity > 0 else 0
            
            utilization_timeline.append({
                'year': cf['year'],
                'demand': demand,
                'capacity': total_capacity,
                'utilization': utilization
            })
        
        return {
            'final_additional_capacity': final_capacity,
            'capacity_cost_per_teu': scenario.total_capital_cost / final_capacity if final_capacity > 0 else 0,
            'utilization_timeline': utilization_timeline,
            'average_utilization': np.mean([u['utilization'] for u in utilization_timeline]),
            'peak_utilization': max([u['utilization'] for u in utilization_timeline]) if utilization_timeline else 0
        }
    
    def _assess_investment_risks(self, scenario: InvestmentScenario) -> Dict[str, Any]:
        """Assess risks associated with investment scenario"""
        risks = {
            'implementation_risk': 'medium',
            'demand_risk': 'medium',
            'technology_risk': 'low',
            'financial_risk': 'medium',
            'regulatory_risk': 'low',
            'overall_risk_score': 0.5
        }
        
        # Implementation risk based on complexity and timeline
        total_implementation_time = max(
            inv.implementation_time_months for inv in scenario.investments
        ) if scenario.investments else 0
        
        if total_implementation_time > 36:
            risks['implementation_risk'] = 'high'
        elif total_implementation_time > 24:
            risks['implementation_risk'] = 'medium'
        else:
            risks['implementation_risk'] = 'low'
        
        # Technology risk based on investment types
        tech_investments = [
            inv for inv in scenario.investments
            if inv.investment_type in [InvestmentType.AUTOMATION, InvestmentType.DIGITAL_INFRASTRUCTURE]
        ]
        
        if len(tech_investments) > 1:
            risks['technology_risk'] = 'medium'
        elif len(tech_investments) > 0:
            risks['technology_risk'] = 'low'
        
        # Financial risk based on total investment size
        if scenario.total_capital_cost > 300_000_000:
            risks['financial_risk'] = 'high'
        elif scenario.total_capital_cost > 100_000_000:
            risks['financial_risk'] = 'medium'
        else:
            risks['financial_risk'] = 'low'
        
        # Calculate overall risk score
        risk_weights = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8
        }
        
        risk_scores = [
            risk_weights[risks['implementation_risk']],
            risk_weights[risks['demand_risk']],
            risk_weights[risks['technology_risk']],
            risk_weights[risks['financial_risk']],
            risk_weights[risks['regulatory_risk']]
        ]
        
        risks['overall_risk_score'] = np.mean(risk_scores)
        
        return risks
    
    def _generate_investment_recommendations(self, scenario: InvestmentScenario, financial_metrics: Dict, risk_assessment: Dict) -> List[str]:
        """Generate investment recommendations based on analysis"""
        recommendations = []
        
        # NPV-based recommendations
        if financial_metrics['npv'] > 50_000_000:
            recommendations.append("Strong positive NPV indicates excellent investment opportunity")
        elif financial_metrics['npv'] > 0:
            recommendations.append("Positive NPV suggests viable investment with moderate returns")
        else:
            recommendations.append("Negative NPV indicates investment may not be financially viable")
        
        # IRR-based recommendations
        if financial_metrics['irr'] > 0.15:
            recommendations.append("High IRR exceeds typical hurdle rates - recommend proceeding")
        elif financial_metrics['irr'] > 0.08:
            recommendations.append("IRR meets minimum requirements but monitor market conditions")
        else:
            recommendations.append("IRR below acceptable threshold - consider alternative investments")
        
        # Payback period recommendations
        if financial_metrics['payback_period'] < 7:
            recommendations.append("Short payback period reduces investment risk")
        elif financial_metrics['payback_period'] < 12:
            recommendations.append("Moderate payback period acceptable for infrastructure investment")
        else:
            recommendations.append("Long payback period increases investment risk")
        
        # Risk-based recommendations
        if risk_assessment['overall_risk_score'] < 0.4:
            recommendations.append("Low overall risk profile supports investment decision")
        elif risk_assessment['overall_risk_score'] < 0.6:
            recommendations.append("Moderate risk requires careful project management")
        else:
            recommendations.append("High risk profile requires additional risk mitigation strategies")
        
        return recommendations
    
    def compare_investment_scenarios(self, scenario_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple investment scenarios"""
        
        if not scenario_ids:
            return {}
        
        comparison_results = {
            'scenarios': [],
            'ranking': [],
            'summary': {}
        }
        
        # Analyze each scenario
        for scenario_id in scenario_ids:
            if scenario_id in self.scenarios:
                scenario = self.scenarios[scenario_id]
                analysis = self.calculate_roi_analysis(scenario)
                comparison_results['scenarios'].append(analysis)
        
        # Rank scenarios
        ranking_criteria = [
            ('npv', 'financial_metrics', 'desc'),
            ('irr', 'financial_metrics', 'desc'),
            ('overall_risk_score', 'risk_assessment', 'asc')
        ]
        
        scenario_scores = []
        for analysis in comparison_results['scenarios']:
            score = 0
            
            # NPV score (normalized)
            npv = analysis['financial_metrics']['npv']
            npv_score = min(1.0, max(0.0, (npv + 100_000_000) / 200_000_000))
            score += npv_score * 0.4
            
            # IRR score
            irr = analysis['financial_metrics']['irr']
            irr_score = min(1.0, max(0.0, irr / 0.2))
            score += irr_score * 0.3
            
            # Risk score (inverted)
            risk_score = 1.0 - analysis['risk_assessment']['overall_risk_score']
            score += risk_score * 0.3
            
            scenario_scores.append({
                'scenario_id': analysis['scenario'].scenario_id,
                'scenario_name': analysis['scenario'].name,
                'score': score,
                'npv': npv,
                'irr': irr,
                'risk_score': analysis['risk_assessment']['overall_risk_score']
            })
        
        # Sort by score
        scenario_scores.sort(key=lambda x: x['score'], reverse=True)
        comparison_results['ranking'] = scenario_scores
        
        # Generate summary
        if scenario_scores:
            best_scenario = scenario_scores[0]
            comparison_results['summary'] = {
                'recommended_scenario': best_scenario['scenario_id'],
                'total_scenarios_analyzed': len(scenario_scores),
                'best_npv': best_scenario['npv'],
                'best_irr': best_scenario['irr'],
                'key_insights': [
                    f"Best scenario: {best_scenario['scenario_name']}",
                    f"Highest NPV: ${best_scenario['npv']:,.0f}",
                    f"Best IRR: {best_scenario['irr']:.1%}"
                ]
            }
        
        return comparison_results
    
    def create_sample_investment_scenarios(self) -> List[str]:
        """Create sample investment scenarios for testing"""
        
        # Scenario 1: Conservative Growth
        self.create_investment_scenario(
            scenario_id="conservative_growth",
            name="Conservative Growth Strategy",
            description="Focus on efficiency improvements and moderate capacity expansion",
            investment_ids=["crane_upgrade_1", "digital_infra_1", "storage_expansion_1"],
            implementation_timeline={
                "digital_infra_1": 0,
                "crane_upgrade_1": 6,
                "storage_expansion_1": 18
            }
        )
        
        # Scenario 2: Aggressive Expansion
        self.create_investment_scenario(
            scenario_id="aggressive_expansion",
            name="Aggressive Expansion Strategy",
            description="Major capacity expansion with new berth and full automation",
            investment_ids=["new_berth_1", "automation_1", "crane_upgrade_1"],
            implementation_timeline={
                "crane_upgrade_1": 0,
                "automation_1": 12,
                "new_berth_1": 6
            }
        )
        
        # Scenario 3: Green Technology Focus
        self.create_investment_scenario(
            scenario_id="green_technology",
            name="Green Technology Strategy",
            description="Sustainable development with environmental focus",
            investment_ids=["environmental_1", "automation_1", "digital_infra_1"],
            implementation_timeline={
                "digital_infra_1": 0,
                "environmental_1": 6,
                "automation_1": 18
            }
        )
        
        return ["conservative_growth", "aggressive_expansion", "green_technology"]

def create_sample_investment_analysis():
    """Create and run sample investment analysis"""
    planner = InvestmentPlanner()
    
    # Create sample scenarios
    scenario_ids = planner.create_sample_investment_scenarios()
    
    # Run comparison analysis
    results = planner.compare_investment_scenarios(scenario_ids)
    
    print("\n=== INVESTMENT PLANNING ANALYSIS ===")
    print(f"Analyzed {len(scenario_ids)} investment scenarios")
    
    for i, analysis in enumerate(results['scenarios']):
        scenario = analysis['scenario']
        financial = analysis['financial_metrics']
        
        print(f"\nScenario {i+1}: {scenario.name}")
        print(f"  Description: {scenario.description}")
        print(f"  Total Investment: ${scenario.total_capital_cost:,.0f}")
        print(f"  NPV: ${financial['npv']:,.0f}")
        print(f"  IRR: {financial['irr']:.1%}")
        print(f"  Payback Period: {financial['payback_period']:.1f} years")
        print(f"  ROI: {financial['roi']:.1%}")
    
    print("\n=== SCENARIO RANKING ===")
    for i, ranking in enumerate(results['ranking']):
        print(f"{i+1}. {ranking['scenario_name']} (Score: {ranking['score']:.2f})")
    
    print("\n=== SUMMARY INSIGHTS ===")
    for insight in results['summary'].get('key_insights', []):
        print(f"  • {insight}")
    
    return results

if __name__ == "__main__":
    create_sample_investment_analysis()