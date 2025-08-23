import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import core simulation components
from hk_port_digital_twin.src.scenarios import ScenarioManager, list_available_scenarios
from hk_port_digital_twin.src.scenarios.scenario_parameters import (
    ALL_SCENARIOS, PEAK_SEASON_PARAMETERS, NORMAL_OPERATIONS_PARAMETERS, 
    LOW_SEASON_PARAMETERS, DemoBusinessMetrics
)
from hk_port_digital_twin.src.scenarios.unified_simulation_framework import (
    UnifiedSimulationType, UnifiedSimulationParameters, UnifiedBusinessMetrics,
    UnifiedSimulationController
)
from hk_port_digital_twin.src.utils.business_intelligence_utils import (
    SharedBusinessIntelligence, ROICalculationMethod, FinancialParameters
)
from hk_port_digital_twin.src.utils.strategic_roi_calculator import (
    StrategicROICalculator, StrategicInvestmentType, StrategicInvestmentScenario
)
from hk_port_digital_twin.src.utils.comprehensive_business_intelligence import (
    ComprehensiveBusinessIntelligence, ComparativeAnalysisResult, ExecutiveBusinessReport,
    ComparisonType, BusinessIntelligenceScope
)
from hk_port_digital_twin.src.core.simulation_controller import SimulationController
from hk_port_digital_twin.src.core.strategic_simulation_controller import StrategicSimulationController
from hk_port_digital_twin.src.dashboard.executive_dashboard import ExecutiveDashboard
from hk_port_digital_twin.src.utils.strategic_visualization import StrategicVisualization
from hk_port_digital_twin.src.utils.enhanced_visualization import (
    EnhancedVisualizationSystem,
    EnhancedVisualizationConfig,
    ViewType,
    ChartTheme,
    create_unified_operational_view,
    create_unified_strategic_view,
    create_unified_executive_view,
    create_unified_comparison_view
)


class ViewMode(Enum):
    """Available view modes for the unified simulations interface."""
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    EXECUTIVE = "executive"
    COMPARISON = "comparison"


@dataclass
class SimulationDisplayConfig:
    """Configuration for simulation display and presentation."""
    title: str
    description: str
    icon: str
    business_focus: str
    expected_roi: float
    investment_required: float
    timeline_months: int
    key_benefits: List[str]
    risk_factors: List[str]
    demo_highlights: List[str]


class UnifiedSimulationsTab:
    """Unified simulations interface that consolidates operational and strategic simulations."""
    
    def __init__(self):
        """Initialize the unified simulations tab."""
        self.scenario_manager = None
        self.simulation_controller = None
        self.strategic_controller = None
        self.executive_dashboard = None
        self.strategic_viz = None
        self.business_intelligence = SharedBusinessIntelligence()
        self.roi_calculator = StrategicROICalculator()
        self.comprehensive_bi = ComprehensiveBusinessIntelligence()
        self.unified_framework = UnifiedSimulationController()
        
        # Initialize enhanced visualization system
        self.enhanced_viz = EnhancedVisualizationSystem()
        
        # Initialize simulation display configurations
        self._initialize_simulation_configs()
    
    def _initialize_simulation_configs(self) -> None:
        """Initialize display configurations for all simulations."""
        self.simulation_configs = {
            'peak_season': SimulationDisplayConfig(
                title="Peak Season Capacity Optimization",
                description="AI-driven optimization for maximum throughput during peak demand periods",
                icon="📈",
                business_focus="Revenue Maximization & Capacity Utilization",
                expected_roi=180.0,
                investment_required=2500000.0,
                timeline_months=24,
                key_benefits=[
                    "35% increase in throughput capacity",
                    "180% ROI over 2 years",
                    "Reduced vessel waiting times by 40%",
                    "Enhanced competitive positioning"
                ],
                risk_factors=[
                    "High initial investment requirement",
                    "Technology adoption learning curve",
                    "Market demand fluctuation risk"
                ],
                demo_highlights=[
                    "Real-time AI optimization",
                    "Dynamic capacity allocation",
                    "Predictive demand management",
                    "Executive ROI dashboard"
                ]
            ),
            'normal_operations': SimulationDisplayConfig(
                title="Normal Operations Baseline",
                description="Optimized standard operations with continuous improvement focus",
                icon="⚖️",
                business_focus="Operational Excellence & Efficiency",
                expected_roi=125.0,
                investment_required=1500000.0,
                timeline_months=18,
                key_benefits=[
                    "25% improvement in operational efficiency",
                    "125% ROI over 18 months",
                    "Standardized best practices",
                    "Consistent service quality"
                ],
                risk_factors=[
                    "Moderate investment requirement",
                    "Change management challenges",
                    "Process standardization complexity"
                ],
                demo_highlights=[
                    "Baseline performance metrics",
                    "Efficiency optimization",
                    "Quality consistency",
                    "Cost-benefit analysis"
                ]
            ),
            'low_season': SimulationDisplayConfig(
                title="Low Season Optimization",
                description="Strategic maintenance and cost optimization during reduced demand",
                icon="🔧",
                business_focus="Cost Optimization & Sustainability",
                expected_roi=285.0,
                investment_required=800000.0,
                timeline_months=12,
                key_benefits=[
                    "22% cost savings through efficiency",
                    "285% ROI via operational optimization",
                    "88% sustainability improvement",
                    "Optimal maintenance scheduling"
                ],
                risk_factors=[
                    "Lower investment requirement",
                    "Seasonal demand variability",
                    "Maintenance scheduling complexity"
                ],
                demo_highlights=[
                    "Strategic maintenance optimization",
                    "Cost-efficient resource allocation",
                    "Sustainability focus",
                    "Highest ROI achievement"
                ]
            )
        }
    
    def _initialize_session_components(self) -> None:
        """Initialize simulation components in session state."""
        if 'scenario_manager' not in st.session_state:
            st.session_state.scenario_manager = ScenarioManager()
        
        if 'unified_simulation_framework' not in st.session_state:
            st.session_state.unified_simulation_framework = self.unified_framework
        
        if 'business_intelligence' not in st.session_state:
            st.session_state.business_intelligence = self.business_intelligence
        
        if 'roi_calculator' not in st.session_state:
            st.session_state.roi_calculator = self.roi_calculator
    
    def render_view_mode_selector(self) -> ViewMode:
        """Render the view mode selector and return selected mode."""
        st.subheader("🎯 Unified Port Simulations")
        st.markdown("*Comprehensive simulation platform for operational excellence and strategic planning*")
        
        # View mode selection
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            operational_selected = st.button(
                "⚙️ Operational View",
                help="Focus on operational metrics, efficiency, and day-to-day performance",
                use_container_width=True
            )
        
        with col2:
            strategic_selected = st.button(
                "📊 Strategic View",
                help="Business intelligence, ROI analysis, and strategic planning",
                use_container_width=True
            )
        
        with col3:
            executive_selected = st.button(
                "👔 Executive View",
                help="High-level business metrics and executive decision support",
                use_container_width=True
            )
        
        with col4:
            comparison_selected = st.button(
                "🔄 Comparison View",
                help="Side-by-side scenario comparison and analysis",
                use_container_width=True
            )
        
        # Determine selected view mode
        if operational_selected:
            st.session_state.view_mode = ViewMode.OPERATIONAL
        elif strategic_selected:
            st.session_state.view_mode = ViewMode.STRATEGIC
        elif executive_selected:
            st.session_state.view_mode = ViewMode.EXECUTIVE
        elif comparison_selected:
            st.session_state.view_mode = ViewMode.COMPARISON
        
        # Default to operational if no mode selected
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = ViewMode.OPERATIONAL
        
        return st.session_state.view_mode
    
    def render_scenario_selector(self) -> str:
        """Render scenario selection interface with rich descriptions."""
        st.subheader("📋 Scenario Selection")
        
        # Create scenario cards
        col1, col2, col3 = st.columns(3)
        
        scenarios = ['peak_season', 'normal_operations', 'low_season']
        scenario_names = ['Peak Season', 'Normal Operations', 'Low Season']
        
        selected_scenario = st.session_state.get('selected_scenario', 'normal_operations')
        
        for i, (scenario_key, scenario_name) in enumerate(zip(scenarios, scenario_names)):
            col = [col1, col2, col3][i]
            config = self.simulation_configs[scenario_key]
            
            with col:
                # Create scenario card
                with st.container():
                    st.markdown(f"### {config.icon} {config.title}")
                    st.markdown(f"**Focus:** {config.business_focus}")
                    st.markdown(f"**Expected ROI:** {config.expected_roi:.1f}%")
                    st.markdown(f"**Timeline:** {config.timeline_months} months")
                    
                    # Key benefits
                    with st.expander("📈 Key Benefits"):
                        for benefit in config.key_benefits:
                            st.markdown(f"• {benefit}")
                    
                    # Demo highlights
                    with st.expander("✨ Demo Highlights"):
                        for highlight in config.demo_highlights:
                            st.markdown(f"• {highlight}")
                    
                    # Selection button
                    if st.button(
                        f"Select {scenario_name}",
                        key=f"select_{scenario_key}",
                        use_container_width=True,
                        type="primary" if selected_scenario == scenario_key else "secondary"
                    ):
                        st.session_state.selected_scenario = scenario_key
                        selected_scenario = scenario_key
                        st.rerun()
        
        return selected_scenario
    
    def render_operational_view(self, scenario: str) -> None:
        """Render operational view with detailed operational metrics."""
        st.subheader("⚙️ Operational Performance View")

        # Get scenario parameters
        scenario_params = ALL_SCENARIOS.get(scenario.replace('_season', '').replace('_operations', ''), 
                                          NORMAL_OPERATIONS_PARAMETERS)

        # Display operational metrics dashboard
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🚢 Vessel Operations")
            
            # Vessel metrics
            metrics_data = {
                'Arrival Rate Multiplier': scenario_params.arrival_rate_multiplier,
                'Peak Hour Multiplier': scenario_params.peak_hour_multiplier,
                'Weekend Multiplier': scenario_params.weekend_multiplier,
                'Average Ship Size': scenario_params.average_ship_size_multiplier
            }
            
            for metric, value in metrics_data.items():
                st.metric(metric, f"{value:.2f}x")
        
        with col2:
            st.markdown("#### ⚡ Operational Efficiency")
            
            # Efficiency metrics
            efficiency_data = {
                'Crane Efficiency': f"{scenario_params.crane_efficiency_multiplier:.2f}x",
                'Processing Rate': f"{scenario_params.processing_rate_multiplier:.2f}x",
                'Target Berth Utilization': f"{scenario_params.target_berth_utilization:.1%}",
                'Berth Availability': f"{scenario_params.berth_availability_factor:.1%}"
            }
            
            for metric, value in efficiency_data.items():
                st.metric(metric, value)

        if st.button("📊 Show Detailed Operational Charts", key="show_operational_charts"):
            # Use enhanced visualization for operational view
            config = EnhancedVisualizationConfig(
                view_type=ViewType.OPERATIONAL,
                theme=ChartTheme.EXECUTIVE
            )
            
            # Generate operational data from scenario parameters
            # Create sample berth data as DataFrame (required by create_port_layout_chart)
            berth_data_list = [
                {"berth_id": i, "berth_type": "Container", "status": "Occupied" if i % 3 == 0 else "Available", 
                 "current_ship": f"Ship_{i}" if i % 3 == 0 else None, "utilization": 0.8 if i % 3 == 0 else 0.0,
                 "is_occupied": i % 3 == 0, "crane_count": 2 + (i % 3), "name": f"Berth-{i}",
                 "max_capacity_teu": 5000 + (i * 500)}
                for i in range(1, 11)
            ]
            berth_data = pd.DataFrame(berth_data_list)
            
            # Create sample queue data
            queue_data = [
                {"ship_id": f"Ship_{i}", "name": f"Ship_{i}", "ship_type": "container", 
                 "arrival_time": datetime.now() + timedelta(hours=i),
                 "estimated_processing_time": 4 + (i % 3), "priority": i % 3,
                 "size_teu": 8000 + (i * 1000), "waiting_time": i * 2}
                for i in range(1, 8)
            ]
            
            # Create utilization data from scenario parameters
            utilization_data = {i: scenario_params.target_berth_utilization * (0.8 + 0.4 * (i % 3) / 2) for i in range(1, 11)}
            
            # Create sample throughput data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            throughput_data = pd.DataFrame({
                'time': dates,
                'containers_processed': np.random.normal(1000 * scenario_params.processing_rate_multiplier, 200, len(dates)),
                'bulk_cargo_processed': np.random.normal(500 * scenario_params.processing_rate_multiplier, 100, len(dates))
            })
            
            # Create unified operational view
            operational_charts = create_unified_operational_view(
                berth_data, queue_data, utilization_data, throughput_data
            )
            
            # Display enhanced operational charts
            for chart_name, chart_fig in operational_charts.items():
                st.plotly_chart(chart_fig, use_container_width=True)
    
    def render_strategic_view(self, scenario: str) -> None:
        """Render strategic view with business intelligence and ROI analysis."""
        st.subheader("📊 Strategic Business Intelligence View")
        
        # Get scenario parameters and business metrics
        scenario_params = ALL_SCENARIOS.get(scenario.replace('_season', '').replace('_operations', ''), 
                                          NORMAL_OPERATIONS_PARAMETERS)
        
        # Use enhanced visualization for strategic view
        config = EnhancedVisualizationConfig(
            view_type=ViewType.STRATEGIC,
            theme=ChartTheme.EXECUTIVE
        )
        
        if hasattr(scenario_params, 'business_metrics') and scenario_params.business_metrics:
            business_metrics = scenario_params.business_metrics
            
            # Prepare scenario data for strategic view
            scenario_data = {
                scenario: {
                    'revenue_impact': getattr(business_metrics, 'revenue_impact', 0.0),
                    'cost_savings': getattr(business_metrics, 'cost_savings', 0.0),
                    'efficiency_gain': getattr(business_metrics, 'efficiency_gain', 0.0),
                    'market_share_impact': getattr(business_metrics, 'market_share_impact', 0.0)
                }
            }
            
            # Prepare ROI data
            roi_data = {
                'investment_amount': getattr(business_metrics, 'investment_required', 0.0),
                'annual_benefits': getattr(business_metrics, 'annual_benefits', 0.0),
                'payback_period': getattr(business_metrics, 'payback_period_months', 24),
                'roi_percentage': getattr(business_metrics, 'roi_percentage', 0.0)
            }
            
            # Create unified strategic view
            strategic_charts = create_unified_strategic_view(
                scenario_data, business_metrics.__dict__ if hasattr(business_metrics, '__dict__') else {}, roi_data
            )
            
            # Business KPIs dashboard
            st.markdown("#### 📈 Key Performance Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Throughput (TEU/hr)",
                    f"{business_metrics.expected_throughput_teu_per_hour:.0f}",
                    delta="+15% vs baseline"
                )
            
            with col2:
                st.metric(
                    "Berth Utilization",
                    f"{business_metrics.expected_berth_utilization:.1f}%",
                    delta="+8% efficiency"
                )
            
            with col3:
                st.metric(
                    "Revenue/Hour",
                    f"${business_metrics.expected_revenue_per_hour:,.0f}",
                    delta="+22% revenue"
                )
            
            with col4:
                st.metric(
                    "ROI",
                    f"{business_metrics.expected_roi_percentage:.1f}%",
                    delta="Above target"
                )
            
            # Display enhanced strategic charts
            for chart_name, chart_fig in strategic_charts.items():
                st.plotly_chart(chart_fig, use_container_width=True)
            
            # Business value proposition
            st.markdown("#### 🎯 Business Value Proposition")
            st.info(business_metrics.business_value_proposition)
            
            # Executive summary points
            st.markdown("#### 📋 Executive Summary")
            for point in business_metrics.executive_summary_points:
                st.markdown(f"• {point}")
            
            # Add comprehensive business intelligence analysis
            st.markdown("#### 🔍 Advanced Business Intelligence")
            
            # Comparative analysis with other scenarios
            try:
                comparative_analysis = self.comprehensive_bi.perform_comparative_analysis(
                    scenarios=list(ALL_SCENARIOS.keys()),
                    comparison_type=ComparisonType.ROI_ANALYSIS,
                    scope=BusinessIntelligenceScope.STRATEGIC
                )
                
                if comparative_analysis:
                    st.markdown("##### 📊 Scenario Comparison")
                    
                    # Create comparison metrics
                    comparison_data = {
                        'Scenario': [],
                        'ROI (%)': [],
                        'Revenue Impact': [],
                        'Risk Level': []
                    }
                    
                    for scenario_key in ALL_SCENARIOS.keys():
                        scenario_params = ALL_SCENARIOS[scenario_key]
                        if hasattr(scenario_params, 'business_metrics') and scenario_params.business_metrics:
                            bm = scenario_params.business_metrics
                            comparison_data['Scenario'].append(scenario_key.title())
                            comparison_data['ROI (%)'].append(bm.expected_roi_percentage)
                            comparison_data['Revenue Impact'].append(bm.expected_revenue_per_hour)
                            comparison_data['Risk Level'].append('Low' if bm.expected_roi_percentage > 15 else 'Medium')
                    
                    if comparison_data['Scenario']:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Highlight current scenario
                        current_scenario_name = scenario.replace('_season', '').replace('_operations', '').title()
                        if current_scenario_name in comparison_df['Scenario'].values:
                            st.success(f"📍 Current scenario: **{current_scenario_name}** - {comparative_analysis.summary}")
            
            except Exception as e:
                st.info("Comparative analysis will be available after running simulations.")
        
        else:
            st.warning("Business metrics not available for this scenario.")
    
    def render_executive_view(self, scenario: str) -> None:
        """Render executive view with high-level business metrics."""
        st.subheader("👔 Executive Decision Support View")
        
        config = self.simulation_configs[scenario]
        
        # Use enhanced visualization for executive view
        viz_config = EnhancedVisualizationConfig(
            view_type=ViewType.EXECUTIVE,
            theme=ChartTheme.EXECUTIVE
        )
        
        # Executive summary card
        with st.container():
            st.markdown(f"### {config.icon} {config.title}")
            st.markdown(f"**Business Focus:** {config.business_focus}")
            st.markdown(config.description)
        
        # Investment overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Investment Required",
                f"${config.investment_required:,.0f}",
                help="Total investment required for implementation"
            )
        
        with col2:
            st.metric(
                "Expected ROI",
                f"{config.expected_roi:.1f}%",
                help="Return on investment over project timeline"
            )
        
        with col3:
            st.metric(
                "Implementation Timeline",
                f"{config.timeline_months} months",
                help="Expected time to full implementation"
            )
        
        # Create strategic investment scenario
        investment_scenario = StrategicInvestmentScenario(
            investment_type=StrategicInvestmentType.CAPACITY_EXPANSION,
            investment_amount=config.investment_required,
            timeline_months=config.timeline_months,
            expected_annual_benefits=config.investment_required * (config.expected_roi / 100) / (config.timeline_months / 12),
            implementation_costs=config.investment_required * 0.1,
            operational_cost_changes=-config.investment_required * 0.05,
            risk_factors=config.risk_factors,
            strategic_objectives=["Operational Excellence", "Market Leadership", "Sustainability"]
        )
        
        # Calculate strategic ROI
        roi_result = self.roi_calculator.calculate_strategic_roi(investment_scenario)
        
        # Prepare executive metrics
        executive_metrics = {
            'investment_amount': config.investment_required,
            'expected_roi': config.expected_roi,
            'timeline_months': config.timeline_months,
            'annual_benefits': roi_result.annual_benefits if hasattr(roi_result, 'annual_benefits') else 0.0,
            'payback_period': roi_result.payback_period_months if hasattr(roi_result, 'payback_period_months') else config.timeline_months,
            'risk_score': len(config.risk_factors) / 10.0,  # Simple risk scoring
            'business_focus': config.business_focus,
            'key_benefits': config.key_benefits,
            'demo_highlights': config.demo_highlights
        }
        
        # Create unified executive view
        executive_chart = create_unified_executive_view(executive_metrics)
        
        # Display enhanced executive chart
        st.plotly_chart(executive_chart, use_container_width=True)
        
        # Risk assessment
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Key Benefits")
            for benefit in config.key_benefits:
                st.markdown(f"• {benefit}")
        
        with col2:
            st.markdown("#### ⚠️ Risk Factors")
            for risk in config.risk_factors:
                st.markdown(f"• {risk}")
        
        # Display strategic metrics
        st.markdown("#### 💼 Strategic Business Case")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Strategic Value Score",
                f"{roi_result.strategic_value_score:.1f}/100",
                help="Overall strategic value assessment"
            )
        
        with col2:
            st.metric(
                "Market Positioning Impact",
                f"{roi_result.market_positioning_impact:.1f}/100",
                help="Impact on competitive market position"
            )
        
        with col3:
            st.metric(
                "Sustainability Contribution",
                f"{roi_result.sustainability_contribution:.1f}/100",
                help="Environmental and sustainability benefits"
            )
        
        # Add comprehensive executive business report
        st.markdown("#### 📊 Executive Business Intelligence Report")
        
        try:
            # Generate comprehensive executive report
            executive_report = self.comprehensive_bi.generate_executive_report(
                scenario_name=scenario,
                investment_scenario=investment_scenario,
                roi_result=roi_result,
                scope=BusinessIntelligenceScope.EXECUTIVE
            )
            
            if executive_report:
                # Executive summary
                st.markdown("##### 📋 Executive Summary")
                st.info(executive_report.executive_summary)
                
                # Key findings
                if executive_report.key_findings:
                    st.markdown("##### 🔍 Key Findings")
                    for finding in executive_report.key_findings:
                        st.markdown(f"• {finding}")
                
                # Strategic recommendations
                if executive_report.strategic_recommendations:
                    st.markdown("##### 🎯 Strategic Recommendations")
                    for recommendation in executive_report.strategic_recommendations:
                        st.markdown(f"• {recommendation}")
                
                # Risk assessment
                if executive_report.risk_assessment:
                    st.markdown("##### ⚠️ Risk Assessment")
                    st.warning(executive_report.risk_assessment)
                
                # Implementation roadmap
                if executive_report.implementation_roadmap:
                    st.markdown("##### 🛣️ Implementation Roadmap")
                    for phase in executive_report.implementation_roadmap:
                        st.markdown(f"• {phase}")
        
        except Exception as e:
            st.info("Executive business intelligence report will be available after running simulations.")
    
    def render_comparison_view(self) -> None:
        """Render comparison view with side-by-side scenario analysis."""
        st.subheader("🔄 Scenario Comparison Analysis")
        
        # Scenario selection for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            scenario1 = st.selectbox(
                "Select First Scenario",
                options=['peak_season', 'normal_operations', 'low_season'],
                format_func=lambda x: self.simulation_configs[x].title,
                key="comparison_scenario1"
            )
        
        with col2:
            scenario2 = st.selectbox(
                "Select Second Scenario",
                options=['peak_season', 'normal_operations', 'low_season'],
                format_func=lambda x: self.simulation_configs[x].title,
                key="comparison_scenario2"
            )
        
        if scenario1 != scenario2:
            # Comparison metrics
            config1 = self.simulation_configs[scenario1]
            config2 = self.simulation_configs[scenario2]
            
            # Use enhanced visualization for comparison view
            viz_config = EnhancedVisualizationConfig(
                view_type=ViewType.COMPARISON,
                theme=ChartTheme.EXECUTIVE
            )
            
            # Prepare scenario data for comparison
            scenario_1_data = {
                'title': config1.title,
                'investment_required': config1.investment_required,
                'expected_roi': config1.expected_roi,
                'timeline_months': config1.timeline_months,
                'business_focus': config1.business_focus,
                'risk_score': len(config1.risk_factors) / 10.0
            }
            
            scenario_2_data = {
                'title': config2.title,
                'investment_required': config2.investment_required,
                'expected_roi': config2.expected_roi,
                'timeline_months': config2.timeline_months,
                'business_focus': config2.business_focus,
                'risk_score': len(config2.risk_factors) / 10.0
            }
            
            comparison_metrics = ['investment_required', 'expected_roi', 'timeline_months', 'risk_score']
            
            # Create unified comparison view
            comparison_chart = create_unified_comparison_view(
                scenario_1_data,
                scenario_2_data,
                comparison_metrics
            )
            
            # Display enhanced comparison chart
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Detailed comparison table
            comparison_data = {
                'Metric': ['Expected ROI (%)', 'Investment Required ($)', 'Timeline (months)', 'Business Focus'],
                config1.title: [
                    f"{config1.expected_roi:.1f}%",
                    f"${config1.investment_required:,.0f}",
                    f"{config1.timeline_months}",
                    config1.business_focus
                ],
                config2.title: [
                    f"{config2.expected_roi:.1f}%",
                    f"${config2.investment_required:,.0f}",
                    f"{config2.timeline_months}",
                    config2.business_focus
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Add comprehensive business intelligence comparison
            st.markdown("#### 🔍 Advanced Comparative Analysis")
            
            try:
                # Perform comprehensive comparative analysis
                comparative_analysis = self.comprehensive_bi.perform_comparative_analysis(
                    scenarios=[scenario1, scenario2],
                    comparison_type=ComparisonType.COMPREHENSIVE,
                    scope=BusinessIntelligenceScope.STRATEGIC
                )
                
                if comparative_analysis:
                    # Analysis summary
                    st.markdown("##### 📊 Analysis Summary")
                    st.info(comparative_analysis.summary)
                    
                    # Key differences
                    if comparative_analysis.key_differences:
                        st.markdown("##### 🔄 Key Differences")
                        for difference in comparative_analysis.key_differences:
                            st.markdown(f"• {difference}")
                    
                    # Recommendations
                    if comparative_analysis.recommendations:
                        st.markdown("##### 🎯 Strategic Recommendations")
                        for recommendation in comparative_analysis.recommendations:
                            st.markdown(f"• {recommendation}")
                    
                    # Risk comparison
                    if comparative_analysis.risk_comparison:
                        st.markdown("##### ⚠️ Risk Comparison")
                        st.warning(comparative_analysis.risk_comparison)
                    
                    # Investment comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"##### 💰 {config1.title} Investment Profile")
                        st.metric("ROI", f"{config1.expected_roi:.1f}%")
                        st.metric("Investment", f"${config1.investment_required:,.0f}")
                        st.metric("Timeline", f"{config1.timeline_months} months")
                    
                    with col2:
                        st.markdown(f"##### 💰 {config2.title} Investment Profile")
                        st.metric("ROI", f"{config2.expected_roi:.1f}%")
                        st.metric("Investment", f"${config2.investment_required:,.0f}")
                        st.metric("Timeline", f"{config2.timeline_months} months")
                    
                    # Decision support
                    if comparative_analysis.decision_support:
                        st.markdown("##### 🎯 Decision Support")
                        st.success(comparative_analysis.decision_support)
            
            except Exception as e:
                st.info("Advanced comparative analysis will be available after running simulations.")
        
        else:
            st.warning("Please select different scenarios for comparison.")
    
    def render(self) -> None:
        """Main render method for the unified simulations tab."""
        # Initialize session components
        self._initialize_session_components()
        
        # Render view mode selector
        view_mode = self.render_view_mode_selector()
        
        st.divider()
        
        # Render scenario selector (except for comparison view)
        if view_mode != ViewMode.COMPARISON:
            selected_scenario = self.render_scenario_selector()
            st.divider()
        
        # Render appropriate view based on selected mode
        if view_mode == ViewMode.OPERATIONAL:
            self.render_operational_view(selected_scenario)
        elif view_mode == ViewMode.STRATEGIC:
            self.render_strategic_view(selected_scenario)
        elif view_mode == ViewMode.EXECUTIVE:
            self.render_executive_view(selected_scenario)
        elif view_mode == ViewMode.COMPARISON:
            self.render_comparison_view()
        
        # Add export functionality
        st.divider()
        st.subheader("📤 Export & Presentation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Dashboard", use_container_width=True):
                st.success("Dashboard export functionality would be implemented here.")
        
        with col2:
            if st.button("📈 Generate Report", use_container_width=True):
                st.success("Report generation functionality would be implemented here.")
        
        with col3:
            if st.button("🎯 Demo Mode", use_container_width=True):
                st.success("Demo mode functionality would be implemented here.")


def render_unified_simulations_tab() -> None:
    """Convenience function to render the unified simulations tab."""
    if 'unified_simulations_tab' not in st.session_state:
        st.session_state.unified_simulations_tab = UnifiedSimulationsTab()
    
    st.session_state.unified_simulations_tab.render()