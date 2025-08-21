"""Enhanced Visualization System

Comments for context:
This module provides a unified visualization system that integrates both operational
and strategic visualizations for the Hong Kong Port Digital Twin. It combines the
operational charts (berth utilization, ship queues, throughput) with strategic
business intelligence charts (ROI, revenue impact, efficiency trends).

The module serves as the central visualization engine for the unified simulations tab,
providing a consistent interface for creating both operational and strategic charts
with unified styling and interactive capabilities.

Key Features:
- Unified chart creation interface for operational and strategic views
- Consistent styling and theming across all chart types
- Interactive controls and real-time updates
- Export capabilities for presentations and reports
- Responsive design for different screen sizes
- Integration with scenario parameters and business metrics
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

# Import existing visualization components
try:
    from ..utils.visualization import (
        create_port_layout_chart,
        create_ship_queue_chart,
        create_berth_utilization_chart,
        create_throughput_timeline
    )
except ImportError:
    # Fallback for direct imports
    from visualization import (
        create_port_layout_chart,
        create_ship_queue_chart,
        create_berth_utilization_chart,
        create_throughput_timeline
    )

try:
    from ..utils.strategic_visualization import StrategicVisualization, ChartTheme, VisualizationConfig
except ImportError:
    from strategic_visualization import StrategicVisualization, ChartTheme, VisualizationConfig


class ViewType(Enum):
    """Types of visualization views"""
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    EXECUTIVE = "executive"
    COMPARISON = "comparison"


@dataclass
class EnhancedVisualizationConfig:
    """Configuration for enhanced visualization system"""
    view_type: ViewType = ViewType.OPERATIONAL
    theme: ChartTheme = ChartTheme.EXECUTIVE
    height: int = 400
    width: Optional[int] = None
    show_legend: bool = True
    interactive: bool = True
    export_enabled: bool = True
    responsive: bool = True


class EnhancedVisualizationSystem:
    """Unified visualization system for operational and strategic charts"""
    
    def __init__(self, config: Optional[EnhancedVisualizationConfig] = None):
        """Initialize enhanced visualization system
        
        Args:
            config: Visualization configuration, uses defaults if None
        """
        self.config = config or EnhancedVisualizationConfig()
        self.strategic_viz = StrategicVisualization()
        
        # Unified color palette for consistency
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'neutral': '#8c564b',
            'berth_high': '#d62728',
            'berth_medium': '#ff7f0e',
            'berth_low': '#2ca02c',
            'container': '#87ceeb',
            'bulk': '#90ee90',
            'mixed': '#ffffe0'
        }
    
    def create_operational_dashboard(self, 
                                   berth_data: List[Dict],
                                   queue_data: List[Dict],
                                   utilization_data: Dict[int, float],
                                   throughput_data: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create comprehensive operational dashboard
        
        Args:
            berth_data: List of berth information dictionaries
            queue_data: List of ship queue data dictionaries
            utilization_data: Dictionary mapping berth_id to utilization percentage
            throughput_data: DataFrame with throughput timeline data
            
        Returns:
            Dictionary of chart names to Plotly figures
        """
        charts = {}
        
        # Port layout chart
        charts['port_layout'] = create_port_layout_chart(berth_data)
        
        # Ship queue chart
        charts['ship_queue'] = create_ship_queue_chart(queue_data)
        
        # Berth utilization chart
        charts['berth_utilization'] = create_berth_utilization_chart(utilization_data)
        
        # Throughput timeline chart
        charts['throughput_timeline'] = create_throughput_timeline(throughput_data)
        
        # Apply unified styling
        for chart_name, fig in charts.items():
            self._apply_unified_styling(fig, chart_name)
        
        return charts
    
    def create_strategic_dashboard(self, 
                                 scenario_data: Dict[str, Dict[str, float]],
                                 business_metrics: Dict[str, Any],
                                 roi_data: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create comprehensive strategic dashboard
        
        Args:
            scenario_data: Dictionary with scenario comparison data
            business_metrics: Business metrics and KPIs
            roi_data: ROI analysis data
            
        Returns:
            Dictionary of chart names to Plotly figures
        """
        charts = {}
        
        # Revenue impact chart
        charts['revenue_impact'] = self.strategic_viz.create_revenue_impact_chart(
            scenario_data, "multi_scenario"
        )
        
        # ROI timeline chart
        if 'investment_amount' in roi_data and 'monthly_returns' in roi_data:
            charts['roi_timeline'] = self.strategic_viz.create_roi_timeline_chart(
                roi_data['investment_amount'],
                roi_data['monthly_returns'],
                roi_data.get('timeline_months', 24)
            )
        
        # Efficiency improvement chart
        if 'baseline_efficiency' in business_metrics and 'scenario_efficiencies' in business_metrics:
            charts['efficiency_improvement'] = self.strategic_viz.create_efficiency_improvement_chart(
                business_metrics['baseline_efficiency'],
                business_metrics['scenario_efficiencies']
            )
        
        # Business KPI summary chart
        charts['kpi_summary'] = self._create_kpi_summary_chart(business_metrics)
        
        # Apply unified styling
        for chart_name, fig in charts.items():
            self._apply_unified_styling(fig, chart_name)
        
        return charts
    
    def create_executive_summary_chart(self, 
                                     executive_metrics: Dict[str, Any]) -> go.Figure:
        """Create executive summary visualization
        
        Args:
            executive_metrics: Executive-level metrics and KPIs
            
        Returns:
            Plotly figure with executive summary
        """
        # Create a comprehensive executive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Revenue vs Investment',
                'Operational Efficiency',
                'Risk vs Return',
                'Strategic Value Score'
            ),
            specs=[[{"secondary_y": True}, {"type": "indicator"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Revenue vs Investment (top-left)
        scenarios = list(executive_metrics.get('scenarios', {}).keys())
        revenues = [executive_metrics['scenarios'][s].get('revenue', 0) for s in scenarios]
        investments = [executive_metrics['scenarios'][s].get('investment', 0) for s in scenarios]
        
        fig.add_trace(
            go.Bar(x=scenarios, y=revenues, name="Revenue", marker_color=self.color_palette['primary']),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=scenarios, y=investments, mode='lines+markers', name="Investment", 
                      line_color=self.color_palette['warning']),
            row=1, col=1, secondary_y=True
        )
        
        # Operational Efficiency Gauge (top-right)
        efficiency_score = executive_metrics.get('efficiency_score', 75)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=efficiency_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Efficiency %"},
                delta={'reference': 70},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['success']},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=2
        )
        
        # Risk vs Return Scatter (bottom-left)
        risk_scores = [executive_metrics['scenarios'][s].get('risk_score', 0) for s in scenarios]
        return_scores = [executive_metrics['scenarios'][s].get('return_score', 0) for s in scenarios]
        
        fig.add_trace(
            go.Scatter(
                x=risk_scores, y=return_scores, mode='markers+text',
                text=scenarios, textposition="top center",
                marker=dict(size=15, color=self.color_palette['info']),
                name="Risk vs Return"
            ),
            row=2, col=1
        )
        
        # Strategic Value Score Gauge (bottom-right)
        strategic_score = executive_metrics.get('strategic_value_score', 80)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=strategic_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Strategic Value"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['primary']},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "gray"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Executive Strategic Overview",
            showlegend=True,
            template=self.config.theme.value
        )
        
        return fig
    
    def create_comparison_chart(self, 
                              scenario_1_data: Dict[str, Any],
                              scenario_2_data: Dict[str, Any],
                              comparison_metrics: List[str]) -> go.Figure:
        """Create side-by-side scenario comparison chart
        
        Args:
            scenario_1_data: Data for first scenario
            scenario_2_data: Data for second scenario
            comparison_metrics: List of metrics to compare
            
        Returns:
            Plotly figure with comparison visualization
        """
        fig = go.Figure()
        
        # Extract values for comparison
        scenario_1_values = [scenario_1_data.get(metric, 0) for metric in comparison_metrics]
        scenario_2_values = [scenario_2_data.get(metric, 0) for metric in comparison_metrics]
        
        # Create grouped bar chart
        fig.add_trace(go.Bar(
            name=scenario_1_data.get('name', 'Scenario 1'),
            x=comparison_metrics,
            y=scenario_1_values,
            marker_color=self.color_palette['primary']
        ))
        
        fig.add_trace(go.Bar(
            name=scenario_2_data.get('name', 'Scenario 2'),
            x=comparison_metrics,
            y=scenario_2_values,
            marker_color=self.color_palette['secondary']
        ))
        
        fig.update_layout(
            title="Scenario Comparison Analysis",
            xaxis_title="Metrics",
            yaxis_title="Values",
            barmode='group',
            height=self.config.height,
            template=self.config.theme.value,
            showlegend=self.config.show_legend
        )
        
        return fig
    
    def _create_kpi_summary_chart(self, business_metrics: Dict[str, Any]) -> go.Figure:
        """Create KPI summary chart
        
        Args:
            business_metrics: Business metrics data
            
        Returns:
            Plotly figure with KPI summary
        """
        # Extract KPI data
        kpis = {
            'Throughput': business_metrics.get('throughput', 0),
            'Efficiency': business_metrics.get('efficiency', 0),
            'Revenue': business_metrics.get('revenue', 0),
            'Cost Savings': business_metrics.get('cost_savings', 0),
            'ROI': business_metrics.get('roi', 0)
        }
        
        # Normalize values for radar chart
        max_values = {
            'Throughput': 1000,
            'Efficiency': 100,
            'Revenue': 1000000,
            'Cost Savings': 500000,
            'ROI': 50
        }
        
        normalized_values = []
        labels = []
        
        for kpi, value in kpis.items():
            if value > 0:
                normalized_value = min((value / max_values[kpi]) * 100, 100)
                normalized_values.append(normalized_value)
                labels.append(kpi)
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=labels,
            fill='toself',
            name='Current Performance',
            line_color=self.color_palette['primary']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Business KPI Summary",
            height=self.config.height,
            template=self.config.theme.value,
            showlegend=False
        )
        
        return fig
    
    def _apply_unified_styling(self, fig: go.Figure, chart_type: str) -> None:
        """Apply unified styling to charts
        
        Args:
            fig: Plotly figure to style
            chart_type: Type of chart for specific styling
        """
        # Apply theme
        fig.update_layout(template=self.config.theme.value)
        
        # Apply responsive sizing
        if self.config.responsive:
            fig.update_layout(
                autosize=True,
                margin=dict(l=20, r=20, t=40, b=20)
            )
        
        # Apply height
        if self.config.height:
            fig.update_layout(height=self.config.height)
        
        # Apply width if specified
        if self.config.width:
            fig.update_layout(width=self.config.width)
        
        # Chart-specific styling
        if chart_type == 'berth_utilization':
            fig.update_traces(marker_line_width=1, marker_line_color="white")
        elif chart_type == 'ship_queue':
            fig.update_layout(yaxis=dict(autorange="reversed"))
        elif chart_type in ['revenue_impact', 'roi_timeline']:
            fig.update_layout(hovermode='x unified')
    
    def export_chart(self, fig: go.Figure, filename: str, format: str = "png") -> bytes:
        """Export chart to specified format
        
        Args:
            fig: Plotly figure to export
            filename: Name for exported file
            format: Export format (png, pdf, html, svg)
            
        Returns:
            Exported chart as bytes
        """
        if format.lower() == "png":
            return fig.to_image(format="png")
        elif format.lower() == "pdf":
            return fig.to_image(format="pdf")
        elif format.lower() == "html":
            return fig.to_html().encode()
        elif format.lower() == "svg":
            return fig.to_image(format="svg")
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Convenience functions for easy integration
def create_unified_operational_view(berth_data: List[Dict],
                                  queue_data: List[Dict],
                                  utilization_data: Dict[int, float],
                                  throughput_data: pd.DataFrame) -> Dict[str, go.Figure]:
    """Convenience function to create operational view"""
    viz_system = EnhancedVisualizationSystem(
        EnhancedVisualizationConfig(view_type=ViewType.OPERATIONAL)
    )
    return viz_system.create_operational_dashboard(
        berth_data, queue_data, utilization_data, throughput_data
    )


def create_unified_strategic_view(scenario_data: Dict[str, Dict[str, float]],
                                business_metrics: Dict[str, Any],
                                roi_data: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Convenience function to create strategic view"""
    viz_system = EnhancedVisualizationSystem(
        EnhancedVisualizationConfig(view_type=ViewType.STRATEGIC)
    )
    return viz_system.create_strategic_dashboard(
        scenario_data, business_metrics, roi_data
    )


def create_unified_executive_view(executive_metrics: Dict[str, Any]) -> go.Figure:
    """Convenience function to create executive view"""
    viz_system = EnhancedVisualizationSystem(
        EnhancedVisualizationConfig(view_type=ViewType.EXECUTIVE)
    )
    return viz_system.create_executive_summary_chart(executive_metrics)


def create_unified_comparison_view(scenario_1_data: Dict[str, Any],
                                 scenario_2_data: Dict[str, Any],
                                 comparison_metrics: List[str]) -> go.Figure:
    """Convenience function to create comparison view"""
    viz_system = EnhancedVisualizationSystem(
        EnhancedVisualizationConfig(view_type=ViewType.COMPARISON)
    )
    return viz_system.create_comparison_chart(
        scenario_1_data, scenario_2_data, comparison_metrics
    )