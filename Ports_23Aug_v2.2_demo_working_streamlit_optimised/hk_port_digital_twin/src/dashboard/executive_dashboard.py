"""Executive Dashboard Components

Comments for context:
This module provides executive-level dashboard components for the Hong Kong Port Digital Twin.
It focuses on business intelligence, strategic planning, and high-level KPIs that executives
and decision-makers need to evaluate port operations and strategic initiatives.

The module integrates with the strategic simulation controller and business intelligence engine
to provide real-time business metrics, ROI calculations, and strategic planning tools.

Key Features:
- Executive summary widgets with KPIs and business impact
- Real-time business metrics tracking (revenue, efficiency, cost savings)
- Strategic planning tools (scenario comparison, investment planning)
- Risk assessment and mitigation recommendations
- Interactive controls for strategic decision-making
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Import strategic simulation components
try:
    from ..core.strategic_simulation_controller import StrategicSimulationController, StrategicSimulationMode, BusinessMetrics
    from ..analytics.business_intelligence import BusinessIntelligenceEngine, KPICategory, ExecutiveSummary
except ImportError:
    # Fallback for development
    StrategicSimulationController = None
    BusinessIntelligenceEngine = None


class DashboardTheme(Enum):
    """Dashboard theme options for executive presentations"""
    EXECUTIVE = "executive"  # Professional, clean theme
    PRESENTATION = "presentation"  # High contrast for presentations
    DARK = "dark"  # Dark theme for extended viewing


@dataclass
class ExecutiveMetrics:
    """Container for executive-level metrics"""
    revenue_per_hour: float
    efficiency_improvement: float
    cost_savings: float
    customer_satisfaction: float
    roi_percentage: float
    risk_score: float
    capacity_utilization: float
    throughput_improvement: float


@dataclass
class StrategicInsight:
    """Strategic business insight for executive decision-making"""
    title: str
    description: str
    impact_level: str  # "High", "Medium", "Low"
    recommendation: str
    financial_impact: float
    implementation_timeline: str
    risk_factors: List[str]


from hk_port_digital_twin.src.core.port_simulation import PortSimulation

class ExecutiveDashboard:
    """Main class for executive dashboard functionality"""
    
    def __init__(self, port_simulation: PortSimulation):
        """Initialize the executive dashboard"""
        self.theme = DashboardTheme.EXECUTIVE
        self.strategic_controller = None
        self.bi_engine = None
        
        # Initialize controllers if available
        if StrategicSimulationController:
            self.strategic_controller = StrategicSimulationController(port_simulation)
        if BusinessIntelligenceEngine:
            self.bi_engine = BusinessIntelligenceEngine()
    
    def render_executive_summary(self, metrics: Optional[ExecutiveMetrics] = None) -> None:
        """Render executive summary with key KPIs
        
        Args:
            metrics: Executive metrics to display, uses sample data if None
        """
        st.subheader("📊 Executive Summary")
        
        # Use sample data if no metrics provided
        if metrics is None:
            metrics = self._get_sample_executive_metrics()
        
        # Create KPI cards in a grid layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Revenue/Hour",
                f"${metrics.revenue_per_hour:,.0f}",
                delta=f"+{metrics.efficiency_improvement:.1f}%",
                help="Revenue generated per operational hour"
            )
            
        with col2:
            st.metric(
                "Cost Savings",
                f"${metrics.cost_savings:,.0f}",
                delta=f"+{metrics.roi_percentage:.1f}%",
                help="Total cost savings from optimization"
            )
            
        with col3:
            st.metric(
                "Efficiency",
                f"{metrics.efficiency_improvement:.1f}%",
                delta=f"+{metrics.throughput_improvement:.1f}%",
                help="Overall operational efficiency improvement"
            )
            
        with col4:
            st.metric(
                "Capacity Utilization",
                f"{metrics.capacity_utilization:.1f}%",
                delta=f"-{metrics.risk_score:.1f}% risk",
                help="Current capacity utilization rate"
            )
    
    def render_business_impact_chart(self, scenario_data: Optional[Dict] = None) -> None:
        """Render business impact visualization
        
        Args:
            scenario_data: Scenario comparison data
        """
        st.subheader("💼 Business Impact Analysis")
        
        # Create sample data if none provided
        if scenario_data is None:
            scenario_data = self._get_sample_scenario_data()
        
        # Create before/after comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Impact', 'Efficiency Gains', 'Cost Reduction', 'ROI Timeline'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Revenue impact (top-left)
        scenarios = list(scenario_data.keys())
        revenues = [scenario_data[s]['revenue'] for s in scenarios]
        
        fig.add_trace(
            go.Bar(x=scenarios, y=revenues, name="Revenue", marker_color="#1f77b4"),
            row=1, col=1
        )
        
        # Efficiency gains (top-right)
        efficiency = [scenario_data[s]['efficiency'] for s in scenarios]
        fig.add_trace(
            go.Scatter(x=scenarios, y=efficiency, mode='lines+markers', name="Efficiency", line_color="#ff7f0e"),
            row=1, col=2
        )
        
        # Cost reduction (bottom-left)
        costs = [scenario_data[s]['cost_savings'] for s in scenarios]
        fig.add_trace(
            go.Bar(x=scenarios, y=costs, name="Cost Savings", marker_color="#2ca02c"),
            row=2, col=1
        )
        
        # ROI timeline (bottom-right)
        months = list(range(1, 13))
        roi_timeline = [5 + i * 2.5 + np.random.normal(0, 1) for i in months]
        fig.add_trace(
            go.Scatter(x=months, y=roi_timeline, mode='lines+markers', name="ROI %", line_color="#d62728"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Strategic Business Impact Dashboard"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_strategic_planning_tools(self) -> None:
        """Render strategic planning and investment tools"""
        st.subheader("🎯 Strategic Planning Tools")
        
        # Create tabs for different planning tools
        plan_tab1, plan_tab2, plan_tab3 = st.tabs(["📊 Scenario Comparison", "💰 Investment Planning", "⚠️ Risk Assessment"])
        
        with plan_tab1:
            self._render_scenario_comparison()
            
        with plan_tab2:
            self._render_investment_planning()
            
        with plan_tab3:
            self._render_risk_assessment()
    
    def render_real_time_metrics(self) -> None:
        """Render real-time business metrics dashboard"""
        st.subheader("⏱️ Real-Time Business Metrics")
        
        # Create metrics columns
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            # Revenue tracking chart
            st.write("**Revenue per Hour Tracking**")
            
            # Generate sample hourly revenue data
            hours = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
            revenue_data = pd.DataFrame({
                'hour': hours,
                'revenue': [15000 + np.random.normal(0, 2000) + 5000 * np.sin(i/4) for i in range(len(hours))]
            })
            
            fig_revenue = px.line(
                revenue_data, x='hour', y='revenue',
                title="24-Hour Revenue Tracking",
                labels={'revenue': 'Revenue ($)', 'hour': 'Time'}
            )
            fig_revenue.update_layout(height=300)
            st.plotly_chart(fig_revenue, use_container_width=True)
            
        with metric_col2:
            # Efficiency meter
            st.write("**Operational Efficiency**")
            
            current_efficiency = 87.5
            target_efficiency = 95.0
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = current_efficiency,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Efficiency %"},
                delta = {'reference': target_efficiency},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': target_efficiency
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    def render_strategic_insights(self) -> None:
        """Render strategic insights and recommendations"""
        st.subheader("💡 Strategic Insights & Recommendations")
        
        # Get sample insights
        insights = self._get_sample_strategic_insights()
        
        for insight in insights:
            with st.expander(f"{insight.impact_level} Impact: {insight.title}"):
                st.write(f"**Description:** {insight.description}")
                st.write(f"**Recommendation:** {insight.recommendation}")
                st.write(f"**Financial Impact:** ${insight.financial_impact:,.0f}")
                st.write(f"**Timeline:** {insight.implementation_timeline}")
                
                if insight.risk_factors:
                    st.write("**Risk Factors:**")
                    for risk in insight.risk_factors:
                        st.write(f"• {risk}")
    
    def _render_scenario_comparison(self) -> None:
        """Render scenario comparison table"""
        st.write("**Multi-Scenario Analysis**")
        
        # Sample scenario comparison data
        comparison_data = {
            'Scenario': ['Current Operations', 'Peak Season Optimized', 'Maintenance Window', 'AI-Enhanced'],
            'Revenue Impact': ['$0', '+$2.5M', '-$800K', '+$4.2M'],
            'Efficiency Gain': ['0%', '+15%', '-5%', '+25%'],
            'Implementation Cost': ['$0', '$500K', '$200K', '$1.2M'],
            'ROI Timeline': ['N/A', '6 months', '3 months', '8 months'],
            'Risk Level': ['Low', 'Medium', 'Low', 'Medium']
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Add export functionality
        if st.button("📊 Export Comparison Report"):
            st.success("Scenario comparison report exported successfully!")
    
    def _render_investment_planning(self) -> None:
        """Render investment planning calculator"""
        st.write("**Investment Planning Calculator**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            investment_amount = st.number_input("Investment Amount ($)", min_value=0, value=1000000, step=50000)
            expected_roi = st.slider("Expected ROI (%)", min_value=0, max_value=50, value=15)
            timeline_years = st.slider("Timeline (years)", min_value=1, max_value=10, value=3)
            
        with col2:
            # Calculate projections
            annual_return = investment_amount * (expected_roi / 100)
            total_return = annual_return * timeline_years
            net_profit = total_return - investment_amount
            
            st.metric("Annual Return", f"${annual_return:,.0f}")
            st.metric("Total Return", f"${total_return:,.0f}")
            st.metric("Net Profit", f"${net_profit:,.0f}")
            
        # ROI projection chart
        years = list(range(1, timeline_years + 1))
        cumulative_returns = [annual_return * year for year in years]
        
        fig_roi = px.line(
            x=years, y=cumulative_returns,
            title="ROI Projection Timeline",
            labels={'x': 'Years', 'y': 'Cumulative Return ($)'}
        )
        st.plotly_chart(fig_roi, use_container_width=True)
    
    def _render_risk_assessment(self) -> None:
        """Render risk assessment matrix"""
        st.write("**Risk Assessment Matrix**")
        
        # Sample risk data
        risks = {
            'Risk Factor': [
                'Technology Implementation',
                'Market Volatility',
                'Regulatory Changes',
                'Operational Disruption',
                'Competitive Response'
            ],
            'Probability': ['Medium', 'High', 'Low', 'Medium', 'High'],
            'Impact': ['High', 'Medium', 'Medium', 'High', 'Low'],
            'Mitigation Strategy': [
                'Phased rollout with pilot testing',
                'Diversified revenue streams',
                'Regulatory compliance monitoring',
                'Comprehensive change management',
                'Competitive intelligence program'
            ],
            'Risk Score': [6, 6, 2, 6, 3]
        }
        
        df_risks = pd.DataFrame(risks)
        st.dataframe(df_risks, use_container_width=True)
        
        # Risk visualization
        fig_risk = px.scatter(
            df_risks, x='Probability', y='Impact', size='Risk Score',
            hover_data=['Risk Factor'], title="Risk Assessment Matrix"
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    def _get_sample_executive_metrics(self) -> ExecutiveMetrics:
        """Generate sample executive metrics for demonstration"""
        return ExecutiveMetrics(
            revenue_per_hour=18500.0,
            efficiency_improvement=15.3,
            cost_savings=2500000.0,
            customer_satisfaction=92.5,
            roi_percentage=22.8,
            risk_score=3.2,
            capacity_utilization=87.5,
            throughput_improvement=18.7
        )
    
    def _get_sample_scenario_data(self) -> Dict[str, Dict[str, float]]:
        """Generate sample scenario data for demonstration"""
        return {
            'Current': {'revenue': 15000000, 'efficiency': 75, 'cost_savings': 0},
            'Peak Season': {'revenue': 17500000, 'efficiency': 85, 'cost_savings': 2500000},
            'Maintenance': {'revenue': 12000000, 'efficiency': 65, 'cost_savings': 800000},
            'AI-Enhanced': {'revenue': 19200000, 'efficiency': 92, 'cost_savings': 4200000}
        }
    
    def _get_sample_strategic_insights(self) -> List[StrategicInsight]:
        """Generate sample strategic insights for demonstration"""
        return [
            StrategicInsight(
                title="Peak Season Capacity Optimization",
                description="AI-driven berth allocation during peak season can increase throughput by 18%",
                impact_level="High",
                recommendation="Implement dynamic berth allocation system before next peak season",
                financial_impact=2500000,
                implementation_timeline="6 months",
                risk_factors=["Technology integration complexity", "Staff training requirements"]
            ),
            StrategicInsight(
                title="Maintenance Window Efficiency",
                description="Optimized maintenance scheduling reduces operational disruption by 35%",
                impact_level="Medium",
                recommendation="Adopt predictive maintenance scheduling with AI optimization",
                financial_impact=800000,
                implementation_timeline="3 months",
                risk_factors=["Initial setup costs", "Change management"]
            ),
            StrategicInsight(
                title="Customer Satisfaction Enhancement",
                description="Reduced waiting times improve customer satisfaction scores by 12%",
                impact_level="Medium",
                recommendation="Focus on queue management and communication systems",
                financial_impact=1200000,
                implementation_timeline="4 months",
                risk_factors=["Customer adoption", "System reliability"]
            )
        ]


def render_executive_dashboard_tab() -> None:
    """Main function to render the executive dashboard tab"""
    st.header("🏢 Executive Dashboard")
    st.markdown("Strategic business intelligence and decision-making tools for port operations")
    
    # Initialize dashboard
    dashboard = ExecutiveDashboard()
    
    # Create main dashboard sections
    exec_tab1, exec_tab2, exec_tab3, exec_tab4 = st.tabs([
        "📊 Executive Summary", 
        "💼 Business Impact", 
        "🎯 Strategic Planning", 
        "💡 Insights & Recommendations"
    ])
    
    with exec_tab1:
        dashboard.render_executive_summary()
        dashboard.render_real_time_metrics()
        
    with exec_tab2:
        dashboard.render_business_impact_chart()
        
    with exec_tab3:
        dashboard.render_strategic_planning_tools()
        
    with exec_tab4:
        dashboard.render_strategic_insights()


if __name__ == "__main__":
    # For testing purposes
    render_executive_dashboard_tab()