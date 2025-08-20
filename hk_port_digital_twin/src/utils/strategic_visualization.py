"""Strategic Visualization Components

Comments for context:
This module provides business-focused visualizations for the Hong Kong Port Digital Twin's
strategic simulations and executive dashboard. It focuses on creating charts and interactive
components that help executives and decision-makers understand business impact, ROI, and
strategic insights from port operations optimization.

The module complements the executive dashboard by providing specialized visualization functions
for strategic scenarios, business intelligence, and executive reporting.

Key Features:
- Executive-level charts (revenue impact, ROI projections, efficiency trends)
- Interactive controls for scenario parameters and real-time simulation
- Business intelligence displays (executive summaries, key findings, recommendations)
- Export capabilities for presentations and reports
- Risk assessment matrices and cost-benefit analysis visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import io
import base64


class ChartTheme(Enum):
    """Chart theme options for different presentation contexts"""
    EXECUTIVE = "plotly_white"  # Clean, professional theme
    PRESENTATION = "plotly"     # High contrast for presentations
    DARK = "plotly_dark"        # Dark theme for extended viewing


class ExportFormat(Enum):
    """Export format options for charts and reports"""
    PNG = "png"
    PDF = "pdf"
    HTML = "html"
    SVG = "svg"


@dataclass
class VisualizationConfig:
    """Configuration for strategic visualizations"""
    theme: ChartTheme = ChartTheme.EXECUTIVE
    height: int = 400
    width: Optional[int] = None
    show_legend: bool = True
    interactive: bool = True
    export_enabled: bool = True


class StrategicVisualization:
    """Main class for strategic visualization components"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize strategic visualization with configuration
        
        Args:
            config: Visualization configuration, uses defaults if None
        """
        self.config = config or VisualizationConfig()
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'neutral': '#8c564b'
        }
    
    def create_revenue_impact_chart(self, 
                                  scenario_data: Dict[str, Dict[str, float]], 
                                  comparison_type: str = "before_after") -> go.Figure:
        """Create revenue impact comparison chart
        
        Args:
            scenario_data: Dictionary with scenario names and their revenue data
            comparison_type: Type of comparison ('before_after', 'multi_scenario')
            
        Returns:
            Plotly figure object
        """
        if comparison_type == "before_after":
            return self._create_before_after_chart(scenario_data, 'revenue', 'Revenue Impact ($)')
        else:
            return self._create_multi_scenario_chart(scenario_data, 'revenue', 'Revenue Comparison ($)')
    
    def create_roi_timeline_chart(self, 
                                investment_amount: float,
                                monthly_returns: List[float],
                                timeline_months: int = 24) -> go.Figure:
        """Create ROI timeline projection chart
        
        Args:
            investment_amount: Initial investment amount
            monthly_returns: List of monthly return amounts
            timeline_months: Number of months to project
            
        Returns:
            Plotly figure object
        """
        months = list(range(1, timeline_months + 1))
        cumulative_returns = np.cumsum(monthly_returns[:timeline_months])
        roi_percentages = [(cum_return / investment_amount) * 100 for cum_return in cumulative_returns]
        
        fig = go.Figure()
        
        # Add cumulative returns line
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_returns,
            mode='lines+markers',
            name='Cumulative Returns',
            line=dict(color=self.color_palette['primary'], width=3),
            yaxis='y'
        ))
        
        # Add ROI percentage line
        fig.add_trace(go.Scatter(
            x=months,
            y=roi_percentages,
            mode='lines+markers',
            name='ROI %',
            line=dict(color=self.color_palette['success'], width=3),
            yaxis='y2'
        ))
        
        # Add break-even line
        fig.add_hline(
            y=0, line_dash="dash", line_color="red",
            annotation_text="Break-even", annotation_position="bottom right"
        )
        
        # Update layout with dual y-axes
        fig.update_layout(
            title="ROI Timeline Projection",
            xaxis_title="Months",
            yaxis=dict(title="Cumulative Returns ($)", side="left"),
            yaxis2=dict(title="ROI (%)", side="right", overlaying="y"),
            height=self.config.height,
            template=self.config.theme.value,
            showlegend=self.config.show_legend
        )
        
        return fig
    
    def create_efficiency_improvement_chart(self, 
                                          baseline_efficiency: float,
                                          scenario_efficiencies: Dict[str, float]) -> go.Figure:
        """Create efficiency improvement trend chart
        
        Args:
            baseline_efficiency: Baseline efficiency percentage
            scenario_efficiencies: Dictionary of scenario names and efficiency values
            
        Returns:
            Plotly figure object
        """
        scenarios = list(scenario_efficiencies.keys())
        efficiencies = list(scenario_efficiencies.values())
        improvements = [(eff - baseline_efficiency) for eff in efficiencies]
        
        # Create bar chart with color coding based on improvement
        colors = [self.color_palette['success'] if imp > 0 else self.color_palette['warning'] for imp in improvements]
        
        fig = go.Figure(data=[
            go.Bar(
                x=scenarios,
                y=improvements,
                marker_color=colors,
                text=[f"+{imp:.1f}%" if imp > 0 else f"{imp:.1f}%" for imp in improvements],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Efficiency Improvement vs Baseline ({baseline_efficiency:.1f}%)",
            xaxis_title="Scenarios",
            yaxis_title="Efficiency Improvement (%)",
            height=self.config.height,
            template=self.config.theme.value,
            showlegend=False
        )
        
        # Add baseline reference line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Baseline")
        
        return fig
    
    def create_cost_benefit_analysis_chart(self, 
                                         costs: Dict[str, float],
                                         benefits: Dict[str, float]) -> go.Figure:
        """Create cost-benefit analysis visualization
        
        Args:
            costs: Dictionary of cost categories and amounts
            benefits: Dictionary of benefit categories and amounts
            
        Returns:
            Plotly figure object
        """
        categories = list(set(costs.keys()) | set(benefits.keys()))
        cost_values = [costs.get(cat, 0) for cat in categories]
        benefit_values = [benefits.get(cat, 0) for cat in categories]
        net_values = [benefit_values[i] - cost_values[i] for i in range(len(categories))]
        
        fig = go.Figure()
        
        # Add cost bars (negative values)
        fig.add_trace(go.Bar(
            x=categories,
            y=[-cost for cost in cost_values],
            name='Costs',
            marker_color=self.color_palette['warning'],
            text=[f"${cost:,.0f}" for cost in cost_values],
            textposition='auto'
        ))
        
        # Add benefit bars (positive values)
        fig.add_trace(go.Bar(
            x=categories,
            y=benefit_values,
            name='Benefits',
            marker_color=self.color_palette['success'],
            text=[f"${benefit:,.0f}" for benefit in benefit_values],
            textposition='auto'
        ))
        
        # Add net benefit line
        fig.add_trace(go.Scatter(
            x=categories,
            y=net_values,
            mode='lines+markers',
            name='Net Benefit',
            line=dict(color=self.color_palette['primary'], width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Cost-Benefit Analysis",
            xaxis_title="Categories",
            yaxis_title="Amount ($)",
            height=self.config.height,
            template=self.config.theme.value,
            showlegend=self.config.show_legend,
            barmode='relative'
        )
        
        # Add break-even line
        fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Break-even")
        
        return fig
    
    def create_risk_assessment_matrix(self, 
                                    risk_data: List[Dict[str, Any]]) -> go.Figure:
        """Create risk assessment matrix visualization
        
        Args:
            risk_data: List of risk dictionaries with 'name', 'probability', 'impact', 'score'
            
        Returns:
            Plotly figure object
        """
        # Convert probability and impact to numeric values
        prob_map = {'Low': 1, 'Medium': 2, 'High': 3}
        impact_map = {'Low': 1, 'Medium': 2, 'High': 3}
        
        x_values = [prob_map.get(risk['probability'], 2) for risk in risk_data]
        y_values = [impact_map.get(risk['impact'], 2) for risk in risk_data]
        risk_scores = [risk.get('score', x_values[i] * y_values[i]) for i, risk in enumerate(risk_data)]
        risk_names = [risk['name'] for risk in risk_data]
        
        # Create scatter plot with risk scores as bubble sizes
        fig = go.Figure(data=go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers+text',
            marker=dict(
                size=[score * 10 for score in risk_scores],
                color=risk_scores,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            text=risk_names,
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>Probability: %{x}<br>Impact: %{y}<br>Score: %{marker.color}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Risk Assessment Matrix",
            xaxis=dict(
                title="Probability",
                tickmode='array',
                tickvals=[1, 2, 3],
                ticktext=['Low', 'Medium', 'High']
            ),
            yaxis=dict(
                title="Impact",
                tickmode='array',
                tickvals=[1, 2, 3],
                ticktext=['Low', 'Medium', 'High']
            ),
            height=self.config.height,
            template=self.config.theme.value,
            showlegend=False
        )
        
        return fig
    
    def create_executive_summary_cards(self, 
                                     metrics: Dict[str, Any]) -> None:
        """Create executive summary cards with key metrics
        
        Args:
            metrics: Dictionary of metric names and values
        """
        # Calculate number of columns based on metrics count
        num_metrics = len(metrics)
        cols = st.columns(min(num_metrics, 4))
        
        for i, (metric_name, metric_data) in enumerate(metrics.items()):
            col_idx = i % 4
            
            with cols[col_idx]:
                if isinstance(metric_data, dict):
                    value = metric_data.get('value', 'N/A')
                    delta = metric_data.get('delta', None)
                    help_text = metric_data.get('help', None)
                else:
                    value = metric_data
                    delta = None
                    help_text = None
                
                st.metric(
                    label=metric_name,
                    value=value,
                    delta=delta,
                    help=help_text
                )
    
    def create_interactive_scenario_controls(self, 
                                           scenario_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create interactive controls for scenario parameters
        
        Args:
            scenario_params: Dictionary of parameter names and their configurations
            
        Returns:
            Dictionary of selected parameter values
        """
        st.subheader("🎛️ Scenario Parameters")
        
        selected_params = {}
        
        # Create columns for parameter controls
        param_cols = st.columns(2)
        
        for i, (param_name, param_config) in enumerate(scenario_params.items()):
            col_idx = i % 2
            
            with param_cols[col_idx]:
                param_type = param_config.get('type', 'slider')
                
                if param_type == 'slider':
                    selected_params[param_name] = st.slider(
                        param_config.get('label', param_name),
                        min_value=param_config.get('min', 0),
                        max_value=param_config.get('max', 100),
                        value=param_config.get('default', 50),
                        help=param_config.get('help', None)
                    )
                elif param_type == 'selectbox':
                    selected_params[param_name] = st.selectbox(
                        param_config.get('label', param_name),
                        options=param_config.get('options', []),
                        index=param_config.get('default_index', 0),
                        help=param_config.get('help', None)
                    )
                elif param_type == 'number_input':
                    selected_params[param_name] = st.number_input(
                        param_config.get('label', param_name),
                        min_value=param_config.get('min', 0),
                        max_value=param_config.get('max', None),
                        value=param_config.get('default', 0),
                        help=param_config.get('help', None)
                    )
        
        return selected_params
    
    def create_simulation_control_panel(self) -> Dict[str, bool]:
        """Create simulation control panel with start/stop/reset buttons
        
        Returns:
            Dictionary of control states
        """
        st.subheader("🎮 Simulation Controls")
        
        control_cols = st.columns(4)
        controls = {}
        
        with control_cols[0]:
            controls['start'] = st.button("▶️ Start", type="primary")
            
        with control_cols[1]:
            controls['pause'] = st.button("⏸️ Pause")
            
        with control_cols[2]:
            controls['stop'] = st.button("⏹️ Stop")
            
        with control_cols[3]:
            controls['reset'] = st.button("🔄 Reset")
        
        return controls
    
    def create_export_controls(self, 
                             chart_fig: go.Figure, 
                             data: Optional[pd.DataFrame] = None) -> None:
        """Create export controls for charts and data
        
        Args:
            chart_fig: Plotly figure to export
            data: Optional dataframe to export
        """
        if not self.config.export_enabled:
            return
            
        st.subheader("📤 Export Options")
        
        export_cols = st.columns(3)
        
        with export_cols[0]:
            if st.button("📊 Export Chart (PNG)"):
                img_bytes = chart_fig.to_image(format="png")
                st.download_button(
                    label="Download Chart",
                    data=img_bytes,
                    file_name=f"strategic_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
        
        with export_cols[1]:
            if data is not None and st.button("📋 Export Data (CSV)"):
                csv_buffer = io.StringIO()
                data.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Data",
                    data=csv_buffer.getvalue(),
                    file_name=f"strategic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with export_cols[2]:
            if st.button("📄 Export Report (HTML)"):
                html_content = self._generate_html_report(chart_fig, data)
                st.download_button(
                    label="Download Report",
                    data=html_content,
                    file_name=f"strategic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
    
    def _create_before_after_chart(self, 
                                 scenario_data: Dict[str, Dict[str, float]], 
                                 metric_key: str, 
                                 title: str) -> go.Figure:
        """Create before/after comparison chart"""
        scenarios = list(scenario_data.keys())
        values = [scenario_data[s][metric_key] for s in scenarios]
        
        colors = [self.color_palette['neutral'] if 'current' in s.lower() or 'baseline' in s.lower() 
                 else self.color_palette['primary'] for s in scenarios]
        
        fig = go.Figure(data=[
            go.Bar(x=scenarios, y=values, marker_color=colors)
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Scenarios",
            yaxis_title="Value",
            height=self.config.height,
            template=self.config.theme.value,
            showlegend=False
        )
        
        return fig
    
    def _create_multi_scenario_chart(self, 
                                   scenario_data: Dict[str, Dict[str, float]], 
                                   metric_key: str, 
                                   title: str) -> go.Figure:
        """Create multi-scenario comparison chart"""
        scenarios = list(scenario_data.keys())
        values = [scenario_data[s][metric_key] for s in scenarios]
        
        fig = go.Figure(data=[
            go.Bar(
                x=scenarios, 
                y=values, 
                marker_color=self.color_palette['primary'],
                text=[f"${val:,.0f}" for val in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Scenarios",
            yaxis_title="Value",
            height=self.config.height,
            template=self.config.theme.value,
            showlegend=False
        )
        
        return fig
    
    def _generate_html_report(self, 
                            chart_fig: go.Figure, 
                            data: Optional[pd.DataFrame] = None) -> str:
        """Generate HTML report with chart and data"""
        chart_html = chart_fig.to_html(include_plotlyjs='cdn')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategic Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #1f77b4; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>Strategic Analysis Report</h1>
            <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Visualization</h2>
            {chart_html}
            
        """
        
        if data is not None:
            data_html = data.to_html(classes='table table-striped')
            html_content += f"""
            <h2>Data</h2>
            {data_html}
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content


# Convenience functions for easy import and use
def create_strategic_revenue_chart(scenario_data: Dict[str, Dict[str, float]]) -> go.Figure:
    """Convenience function to create revenue impact chart"""
    viz = StrategicVisualization()
    return viz.create_revenue_impact_chart(scenario_data)


def create_strategic_roi_chart(investment: float, returns: List[float]) -> go.Figure:
    """Convenience function to create ROI timeline chart"""
    viz = StrategicVisualization()
    return viz.create_roi_timeline_chart(investment, returns)


def create_strategic_efficiency_chart(baseline: float, scenarios: Dict[str, float]) -> go.Figure:
    """Convenience function to create efficiency improvement chart"""
    viz = StrategicVisualization()
    return viz.create_efficiency_improvement_chart(baseline, scenarios)


def render_strategic_controls(scenario_params: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    """Convenience function to render scenario controls and simulation panel"""
    viz = StrategicVisualization()
    
    # Create two columns for controls
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        selected_params = viz.create_interactive_scenario_controls(scenario_params)
    
    with control_col2:
        simulation_controls = viz.create_simulation_control_panel()
    
    return selected_params, simulation_controls


if __name__ == "__main__":
    # For testing purposes
    st.title("Strategic Visualization Components Test")
    
    # Sample data for testing
    sample_scenario_data = {
        'Current': {'revenue': 15000000, 'efficiency': 75},
        'Peak Season': {'revenue': 17500000, 'efficiency': 85},
        'AI-Enhanced': {'revenue': 19200000, 'efficiency': 92}
    }
    
    viz = StrategicVisualization()
    
    # Test revenue chart
    fig = viz.create_revenue_impact_chart(sample_scenario_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Test ROI chart
    monthly_returns = [50000 + np.random.normal(0, 10000) for _ in range(24)]
    roi_fig = viz.create_roi_timeline_chart(1000000, monthly_returns)
    st.plotly_chart(roi_fig, use_container_width=True)