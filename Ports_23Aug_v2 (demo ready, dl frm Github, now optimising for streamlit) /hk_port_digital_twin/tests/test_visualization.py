"""Tests for visualization utilities module

This module tests all visualization functions to ensure they create
valid Plotly figures with correct data representation.
"""

import pytest
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.visualization import (
    create_port_layout_chart,
    create_ship_queue_chart,
    create_berth_utilization_chart,
    create_throughput_timeline,
    create_waiting_time_distribution,
    create_kpi_summary_chart
)


class TestPortLayoutChart:
    """Test port layout visualization"""
    
    def test_create_port_layout_chart_basic(self):
        """Test basic port layout chart creation"""
        berths_data = pd.DataFrame({
            'berth_id': [1, 2, 3],
            'name': ['Berth A1', 'Berth A2', 'Berth B1'],
            'max_capacity_teu': [20000, 20000, 30000],
            'crane_count': [4, 4, 2],
            'berth_type': ['container', 'container', 'bulk'],
            'is_occupied': [True, False, True]
        })
        
        fig = create_port_layout_chart(berths_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # One trace per berth
        assert "Hong Kong Port - Berth Layout" in fig.layout.title.text
    
    def test_create_port_layout_chart_empty(self):
        """Test port layout chart with empty data"""
        berths_data = pd.DataFrame({
            'berth_id': [],
            'name': [],
            'max_capacity_teu': [],
            'crane_count': [],
            'berth_type': [],
            'is_occupied': []
        })
        
        fig = create_port_layout_chart(berths_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_create_port_layout_chart_different_types(self):
        """Test port layout chart with different berth types"""
        berths_data = pd.DataFrame({
            'berth_id': [1, 2, 3],
            'name': ['Container', 'Bulk', 'Mixed'],
            'max_capacity_teu': [20000, 30000, 25000],
            'crane_count': [4, 2, 3],
            'berth_type': ['container', 'bulk', 'mixed'],
            'is_occupied': [False, False, False]
        })
        
        fig = create_port_layout_chart(berths_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3
        
        # Check that different berth types have different colors
        colors = [trace.marker.color for trace in fig.data]
        assert 'blue' in colors  # container
        assert 'green' in colors  # bulk
        assert 'orange' in colors  # mixed


class TestShipQueueChart:
    """Test ship queue visualization"""
    
    def test_create_ship_queue_chart_basic(self):
        """Test basic ship queue chart creation"""
        queue_data = [
            {
                'ship_id': '1',
                'name': 'Ship A',
                'ship_type': 'container',
                'size_teu': 15000,
                'waiting_time': 2.5
            },
            {
                'ship_id': '2',
                'name': 'Ship B',
                'ship_type': 'bulk',
                'size_teu': 25000,
                'waiting_time': 1.0
            }
        ]
        
        fig = create_ship_queue_chart(queue_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # One bar chart
        assert "Ship Waiting Queue" in fig.layout.title.text
    
    def test_create_ship_queue_chart_empty(self):
        """Test ship queue chart with empty queue"""
        queue_data = []
        
        fig = create_ship_queue_chart(queue_data)
        
        assert isinstance(fig, go.Figure)
        assert "No ships currently in queue" in str(fig.layout.annotations[0].text)
    
    def test_create_ship_queue_chart_single_ship(self):
        """Test ship queue chart with single ship"""
        queue_data = [
            {
                'ship_id': '1',
                'name': 'Single Ship',
                'ship_type': 'container',
                'size_teu': 10000,
                'waiting_time': 3.0
            }
        ]
        
        fig = create_ship_queue_chart(queue_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert len(fig.data[0].y) == 1  # One ship


class TestBerthUtilizationChart:
    """Test berth utilization visualization"""
    
    def test_create_berth_utilization_chart_basic(self):
        """Test basic berth utilization chart creation"""
        utilization_data = {
            1: 85.5,
            2: 45.2,
            3: 92.1,
            4: 67.8
        }
        
        fig = create_berth_utilization_chart(utilization_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # One bar chart
        assert "Berth Utilization" in fig.layout.title.text
        assert len(fig.data[0].x) == 4  # Four berths
    
    def test_create_berth_utilization_chart_empty(self):
        """Test berth utilization chart with empty data"""
        utilization_data = {}
        
        fig = create_berth_utilization_chart(utilization_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 0  # No berths
    
    def test_create_berth_utilization_chart_color_coding(self):
        """Test berth utilization chart color coding"""
        utilization_data = {
            1: 90.0,  # Should be red (>80%)
            2: 70.0,  # Should be orange (60-80%)
            3: 50.0   # Should be green (<60%)
        }
        
        fig = create_berth_utilization_chart(utilization_data)
        
        assert isinstance(fig, go.Figure)
        colors = fig.data[0].marker.color
        assert 'red' in colors
        assert 'orange' in colors
        assert 'green' in colors


class TestThroughputTimeline:
    """Test throughput timeline visualization"""
    
    def test_create_throughput_timeline_basic(self):
        """Test basic throughput timeline creation"""
        throughput_data = pd.DataFrame({
            'time': [0, 1, 2, 3, 4],
            'containers_processed': [0, 50, 120, 200, 280]
        })
        
        fig = create_throughput_timeline(throughput_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # One line chart
        assert "Container Throughput Over Time" in fig.layout.title.text
        assert len(fig.data[0].x) == 5  # Five time points
    
    def test_create_throughput_timeline_empty(self):
        """Test throughput timeline with empty data"""
        throughput_data = pd.DataFrame({
            'time': [],
            'containers_processed': []
        })
        
        fig = create_throughput_timeline(throughput_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 0  # No data points


class TestWaitingTimeDistribution:
    """Test waiting time distribution visualization"""
    
    def test_create_waiting_time_distribution_basic(self):
        """Test basic waiting time distribution creation"""
        waiting_times = [1.0, 2.5, 1.8, 3.2, 0.5, 4.1, 2.0, 1.5, 2.8, 3.5]
        
        fig = create_waiting_time_distribution(waiting_times)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # One histogram
        assert "Ship Waiting Time Distribution" in fig.layout.title.text
    
    def test_create_waiting_time_distribution_empty(self):
        """Test waiting time distribution with empty data"""
        waiting_times = []
        
        fig = create_waiting_time_distribution(waiting_times)
        
        assert isinstance(fig, go.Figure)
        assert "No waiting time data available" in str(fig.layout.annotations[0].text)
    
    def test_create_waiting_time_distribution_single_value(self):
        """Test waiting time distribution with single value"""
        waiting_times = [2.5]
        
        fig = create_waiting_time_distribution(waiting_times)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1


class TestKPISummaryChart:
    """Test KPI summary visualization"""
    
    def test_create_kpi_summary_chart_basic(self):
        """Test basic KPI summary chart creation"""
        kpi_data = {
            'average_waiting_time': 2.5,
            'average_berth_utilization': 0.75,
            'total_ships_processed': 25,
            'total_containers_processed': 1500,
            'average_queue_length': 3.2
        }
        
        fig = create_kpi_summary_chart(kpi_data)
        
        assert isinstance(fig, go.Figure)
        assert "Key Performance Indicators" in fig.layout.title.text
        assert len(fig.data) >= 1  # At least one trace (gauge)
    
    def test_create_kpi_summary_chart_empty(self):
        """Test KPI summary chart with empty data"""
        kpi_data = {}
        
        fig = create_kpi_summary_chart(kpi_data)
        
        assert isinstance(fig, go.Figure)
        assert "Key Performance Indicators" in fig.layout.title.text
    
    def test_create_kpi_summary_chart_partial_data(self):
        """Test KPI summary chart with partial data"""
        kpi_data = {
            'average_waiting_time': 1.8,
            'total_ships_processed': 15
        }
        
        fig = create_kpi_summary_chart(kpi_data)
        
        assert isinstance(fig, go.Figure)
        assert "Key Performance Indicators" in fig.layout.title.text


class TestVisualizationIntegration:
    """Test integration scenarios for visualization functions"""
    
    def test_all_functions_return_plotly_figures(self):
        """Test that all visualization functions return valid Plotly figures"""
        # Test data
        berths_data = pd.DataFrame({
            'berth_id': [1, 2],
            'name': ['Berth A', 'Berth B'],
            'max_capacity_teu': [20000, 25000],
            'crane_count': [4, 3],
            'berth_type': ['container', 'bulk'],
            'is_occupied': [True, False]
        })
        
        queue_data = [{
            'ship_id': '1',
            'name': 'Test Ship',
            'ship_type': 'container',
            'size_teu': 15000,
            'waiting_time': 2.0
        }]
        
        utilization_data = {1: 75.0, 2: 60.0}
        
        throughput_data = pd.DataFrame({
            'time': [0, 1, 2],
            'containers_processed': [0, 50, 100]
        })
        
        waiting_times = [1.0, 2.0, 3.0]
        
        kpi_data = {
            'average_waiting_time': 2.0,
            'average_berth_utilization': 0.67,
            'total_ships_processed': 10
        }
        
        # Test all functions
        functions_and_data = [
            (create_port_layout_chart, berths_data),
            (create_ship_queue_chart, queue_data),
            (create_berth_utilization_chart, utilization_data),
            (create_throughput_timeline, throughput_data),
            (create_waiting_time_distribution, waiting_times),
            (create_kpi_summary_chart, kpi_data)
        ]
        
        for func, data in functions_and_data:
            fig = func(data)
            assert isinstance(fig, go.Figure), f"{func.__name__} did not return a Plotly Figure"
            assert hasattr(fig, 'layout'), f"{func.__name__} figure missing layout"
            assert hasattr(fig, 'data'), f"{func.__name__} figure missing data"
    
    def test_figure_titles_are_set(self):
        """Test that all figures have appropriate titles"""
        # Create minimal test data
        berths_data = pd.DataFrame({
            'berth_id': [1],
            'name': ['Test'],
            'max_capacity_teu': [20000],
            'crane_count': [4],
            'berth_type': ['container'],
            'is_occupied': [False]
        })
        
        expected_titles = [
            (create_port_layout_chart(berths_data), "Hong Kong Port - Berth Layout"),
            (create_ship_queue_chart([]), "Ship Waiting Queue"),
            (create_berth_utilization_chart({1: 50.0}), "Berth Utilization"),
            (create_throughput_timeline(pd.DataFrame({'time': [0], 'containers_processed': [0]})), "Container Throughput Over Time"),
            (create_waiting_time_distribution([]), "Ship Waiting Time Distribution"),
            (create_kpi_summary_chart({}), "Key Performance Indicators")
        ]
        
        for fig, expected_title in expected_titles:
            assert expected_title in fig.layout.title.text, f"Figure missing expected title: {expected_title}"


if __name__ == "__main__":
    pytest.main([__file__])