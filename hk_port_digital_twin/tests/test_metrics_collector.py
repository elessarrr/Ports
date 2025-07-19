"""Tests for Metrics Collector Module

This module tests the MetricsCollector class and SimulationMetrics dataclass
to ensure proper data collection and KPI calculation functionality.
"""

import pytest
import pandas as pd
from src.utils.metrics_collector import MetricsCollector, SimulationMetrics


class TestSimulationMetrics:
    """Test SimulationMetrics dataclass"""
    
    def test_simulation_metrics_initialization(self):
        """Test that SimulationMetrics initializes with empty collections"""
        metrics = SimulationMetrics()
        
        assert metrics.ship_waiting_times == []
        assert metrics.berth_utilization == {}
        assert metrics.container_throughput == []
        assert metrics.queue_lengths == []
        assert metrics.processing_times == []
        assert metrics.ship_arrivals == []
        assert metrics.ship_departures == []
        assert metrics.berth_assignments == []
        
    def test_simulation_metrics_with_data(self):
        """Test SimulationMetrics with initial data"""
        metrics = SimulationMetrics(
            ship_waiting_times=[1.5, 2.0, 0.5],
            berth_utilization={1: 85.0, 2: 92.5},
            container_throughput=[100, 150, 200]
        )
        
        assert len(metrics.ship_waiting_times) == 3
        assert len(metrics.berth_utilization) == 2
        assert len(metrics.container_throughput) == 3
        assert metrics.berth_utilization[1] == 85.0


class TestMetricsCollector:
    """Test MetricsCollector class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.collector = MetricsCollector()
        
    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization"""
        assert isinstance(self.collector.metrics, SimulationMetrics)
        assert self.collector.start_time is None
        assert self.collector.end_time is None
        
    def test_start_end_collection(self):
        """Test start and end collection timing"""
        self.collector.start_collection(0.0)
        assert self.collector.start_time == 0.0
        
        self.collector.end_collection(24.0)
        assert self.collector.end_time == 24.0
        
    def test_record_ship_waiting_time(self):
        """Test recording ship waiting times"""
        self.collector.record_ship_waiting_time("SHIP_001", 2.5)
        self.collector.record_ship_waiting_time("SHIP_002", 1.0)
        
        assert len(self.collector.metrics.ship_waiting_times) == 2
        assert 2.5 in self.collector.metrics.ship_waiting_times
        assert 1.0 in self.collector.metrics.ship_waiting_times
        
    def test_record_ship_waiting_time_negative(self):
        """Test that negative waiting times are not recorded"""
        self.collector.record_ship_waiting_time("SHIP_001", -1.0)
        
        assert len(self.collector.metrics.ship_waiting_times) == 0
        
    def test_record_berth_utilization(self):
        """Test recording berth utilization"""
        self.collector.record_berth_utilization(1, 85.5)
        self.collector.record_berth_utilization(2, 92.0)
        
        assert len(self.collector.metrics.berth_utilization) == 2
        assert self.collector.metrics.berth_utilization[1] == 85.5
        assert self.collector.metrics.berth_utilization[2] == 92.0
        
    def test_record_berth_utilization_invalid(self):
        """Test that invalid utilization rates are not recorded"""
        self.collector.record_berth_utilization(1, -10.0)  # Negative
        self.collector.record_berth_utilization(2, 150.0)  # Over 100%
        
        assert len(self.collector.metrics.berth_utilization) == 0
        
    def test_record_container_throughput(self):
        """Test recording container throughput"""
        self.collector.record_container_throughput(100)
        self.collector.record_container_throughput(150)
        self.collector.record_container_throughput(200)
        
        assert len(self.collector.metrics.container_throughput) == 3
        assert sum(self.collector.metrics.container_throughput) == 450
        
    def test_record_container_throughput_negative(self):
        """Test that negative throughput is not recorded"""
        self.collector.record_container_throughput(-50)
        
        assert len(self.collector.metrics.container_throughput) == 0
        
    def test_record_queue_length(self):
        """Test recording queue lengths"""
        self.collector.record_queue_length(5)
        self.collector.record_queue_length(3)
        self.collector.record_queue_length(0)
        
        assert len(self.collector.metrics.queue_lengths) == 3
        assert max(self.collector.metrics.queue_lengths) == 5
        assert min(self.collector.metrics.queue_lengths) == 0
        
    def test_record_processing_time(self):
        """Test recording processing times"""
        self.collector.record_processing_time(4.5)
        self.collector.record_processing_time(3.2)
        
        assert len(self.collector.metrics.processing_times) == 2
        assert 4.5 in self.collector.metrics.processing_times
        assert 3.2 in self.collector.metrics.processing_times
        
    def test_record_ship_events(self):
        """Test recording ship arrival and departure events"""
        self.collector.record_ship_arrival("SHIP_001", 1.0)
        self.collector.record_ship_arrival("SHIP_002", 2.5)
        self.collector.record_ship_departure("SHIP_001", 5.0)
        
        assert len(self.collector.metrics.ship_arrivals) == 2
        assert len(self.collector.metrics.ship_departures) == 1
        assert ("SHIP_001", 1.0) in self.collector.metrics.ship_arrivals
        assert ("SHIP_001", 5.0) in self.collector.metrics.ship_departures
        
    def test_record_berth_assignment(self):
        """Test recording berth assignments"""
        self.collector.record_berth_assignment("SHIP_001", 1, 2.0)
        self.collector.record_berth_assignment("SHIP_002", 2, 3.5)
        
        assert len(self.collector.metrics.berth_assignments) == 2
        assert ("SHIP_001", 1, 2.0) in self.collector.metrics.berth_assignments
        assert ("SHIP_002", 2, 3.5) in self.collector.metrics.berth_assignments
        
    def test_calculate_average_waiting_time(self):
        """Test average waiting time calculation"""
        # Test with no data
        assert self.collector.calculate_average_waiting_time() == 0.0
        
        # Test with data
        self.collector.record_ship_waiting_time("SHIP_001", 2.0)
        self.collector.record_ship_waiting_time("SHIP_002", 4.0)
        self.collector.record_ship_waiting_time("SHIP_003", 6.0)
        
        assert self.collector.calculate_average_waiting_time() == 4.0
        
    def test_calculate_max_waiting_time(self):
        """Test maximum waiting time calculation"""
        # Test with no data
        assert self.collector.calculate_max_waiting_time() == 0.0
        
        # Test with data
        self.collector.record_ship_waiting_time("SHIP_001", 2.0)
        self.collector.record_ship_waiting_time("SHIP_002", 8.5)
        self.collector.record_ship_waiting_time("SHIP_003", 3.0)
        
        assert self.collector.calculate_max_waiting_time() == 8.5
        
    def test_calculate_average_berth_utilization(self):
        """Test average berth utilization calculation"""
        # Test with no data
        assert self.collector.calculate_average_berth_utilization() == 0.0
        
        # Test with data
        self.collector.record_berth_utilization(1, 80.0)
        self.collector.record_berth_utilization(2, 90.0)
        self.collector.record_berth_utilization(3, 70.0)
        
        assert self.collector.calculate_average_berth_utilization() == 80.0
        
    def test_calculate_total_container_throughput(self):
        """Test total container throughput calculation"""
        # Test with no data
        assert self.collector.calculate_total_container_throughput() == 0
        
        # Test with data
        self.collector.record_container_throughput(100)
        self.collector.record_container_throughput(250)
        self.collector.record_container_throughput(150)
        
        assert self.collector.calculate_total_container_throughput() == 500
        
    def test_calculate_average_queue_length(self):
        """Test average queue length calculation"""
        # Test with no data
        assert self.collector.calculate_average_queue_length() == 0.0
        
        # Test with data
        self.collector.record_queue_length(2)
        self.collector.record_queue_length(4)
        self.collector.record_queue_length(6)
        self.collector.record_queue_length(0)
        
        assert self.collector.calculate_average_queue_length() == 3.0
        
    def test_calculate_max_queue_length(self):
        """Test maximum queue length calculation"""
        # Test with no data
        assert self.collector.calculate_max_queue_length() == 0
        
        # Test with data
        self.collector.record_queue_length(2)
        self.collector.record_queue_length(7)
        self.collector.record_queue_length(3)
        
        assert self.collector.calculate_max_queue_length() == 7
        
    def test_calculate_average_processing_time(self):
        """Test average processing time calculation"""
        # Test with no data
        assert self.collector.calculate_average_processing_time() == 0.0
        
        # Test with data
        self.collector.record_processing_time(3.0)
        self.collector.record_processing_time(5.0)
        self.collector.record_processing_time(4.0)
        
        assert self.collector.calculate_average_processing_time() == 4.0
        
    def test_calculate_ship_rates(self):
        """Test ship arrival and departure rate calculations"""
        # Test with no data
        assert self.collector.calculate_ship_arrival_rate() == 0.0
        assert self.collector.calculate_ship_departure_rate() == 0.0
        
        # Set up time period
        self.collector.start_collection(0.0)
        self.collector.end_collection(10.0)  # 10 hour period
        
        # Add ship events
        self.collector.record_ship_arrival("SHIP_001", 1.0)
        self.collector.record_ship_arrival("SHIP_002", 3.0)
        self.collector.record_ship_arrival("SHIP_003", 7.0)
        
        self.collector.record_ship_departure("SHIP_001", 5.0)
        self.collector.record_ship_departure("SHIP_002", 8.0)
        
        # 3 arrivals in 10 hours = 0.3 ships/hour
        assert self.collector.calculate_ship_arrival_rate() == 0.3
        # 2 departures in 10 hours = 0.2 ships/hour
        assert self.collector.calculate_ship_departure_rate() == 0.2
        
    def test_calculate_ship_rates_zero_duration(self):
        """Test ship rate calculations with zero duration"""
        self.collector.start_collection(5.0)
        self.collector.end_collection(5.0)  # Same time
        
        self.collector.record_ship_arrival("SHIP_001", 5.0)
        
        assert self.collector.calculate_ship_arrival_rate() == 0.0
        
    def test_get_performance_summary(self):
        """Test comprehensive performance summary generation"""
        # Add sample data
        self.collector.start_collection(0.0)
        self.collector.end_collection(24.0)
        
        self.collector.record_ship_waiting_time("SHIP_001", 2.0)
        self.collector.record_ship_waiting_time("SHIP_002", 4.0)
        
        self.collector.record_berth_utilization(1, 85.0)
        self.collector.record_berth_utilization(2, 75.0)
        
        self.collector.record_container_throughput(100)
        self.collector.record_container_throughput(150)
        
        self.collector.record_queue_length(3)
        self.collector.record_queue_length(5)
        
        self.collector.record_processing_time(4.0)
        
        self.collector.record_ship_arrival("SHIP_001", 1.0)
        self.collector.record_ship_departure("SHIP_001", 5.0)
        
        summary = self.collector.get_performance_summary()
        
        # Check structure
        assert 'waiting_times' in summary
        assert 'berth_utilization' in summary
        assert 'throughput' in summary
        assert 'queue_performance' in summary
        assert 'ship_flow' in summary
        assert 'simulation_info' in summary
        
        # Check values
        assert summary['waiting_times']['average'] == 3.0
        assert summary['waiting_times']['maximum'] == 4.0
        assert summary['waiting_times']['count'] == 2
        
        assert summary['berth_utilization']['average'] == 80.0
        assert summary['berth_utilization']['by_berth'][1] == 85.0
        
        assert summary['throughput']['total_containers'] == 250
        assert summary['throughput']['average_processing_time'] == 4.0
        
        assert summary['queue_performance']['average_length'] == 4.0
        assert summary['queue_performance']['maximum_length'] == 5
        
        assert summary['ship_flow']['total_arrivals'] == 1
        assert summary['ship_flow']['total_departures'] == 1
        
        assert summary['simulation_info']['duration'] == 24.0
        
    def test_export_to_dataframe(self):
        """Test exporting metrics to pandas DataFrames"""
        # Add sample data
        self.collector.record_ship_arrival("SHIP_001", 1.0)
        self.collector.record_ship_departure("SHIP_001", 5.0)
        self.collector.record_berth_assignment("SHIP_001", 1, 2.0)
        self.collector.record_queue_length(3)
        self.collector.record_ship_waiting_time("SHIP_001", 1.0)
        
        dataframes = self.collector.export_to_dataframe()
        
        # Check that DataFrames are created
        assert 'ship_events' in dataframes
        assert 'berth_assignments' in dataframes
        assert 'queue_lengths' in dataframes
        assert 'waiting_times' in dataframes
        
        # Check DataFrame contents
        ship_events_df = dataframes['ship_events']
        assert len(ship_events_df) == 2  # 1 arrival + 1 departure
        assert 'ship_id' in ship_events_df.columns
        assert 'event' in ship_events_df.columns
        assert 'time' in ship_events_df.columns
        
        berth_df = dataframes['berth_assignments']
        assert len(berth_df) == 1
        assert berth_df.iloc[0]['ship_id'] == 'SHIP_001'
        assert berth_df.iloc[0]['berth_id'] == 1
        
    def test_export_to_dataframe_empty(self):
        """Test exporting empty metrics to DataFrames"""
        dataframes = self.collector.export_to_dataframe()
        
        # Should return empty dictionary when no data
        assert len(dataframes) == 0
        
    def test_reset_metrics(self):
        """Test resetting all metrics"""
        # Add some data
        self.collector.start_collection(0.0)
        self.collector.record_ship_waiting_time("SHIP_001", 2.0)
        self.collector.record_berth_utilization(1, 85.0)
        
        # Verify data exists
        assert len(self.collector.metrics.ship_waiting_times) == 1
        assert len(self.collector.metrics.berth_utilization) == 1
        assert self.collector.start_time == 0.0
        
        # Reset
        self.collector.reset_metrics()
        
        # Verify everything is cleared
        assert len(self.collector.metrics.ship_waiting_times) == 0
        assert len(self.collector.metrics.berth_utilization) == 0
        assert self.collector.start_time is None
        assert self.collector.end_time is None
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with zero values
        self.collector.record_ship_waiting_time("SHIP_001", 0.0)
        self.collector.record_berth_utilization(1, 0.0)
        self.collector.record_berth_utilization(2, 100.0)
        self.collector.record_container_throughput(0)
        self.collector.record_queue_length(0)
        
        assert len(self.collector.metrics.ship_waiting_times) == 1
        assert len(self.collector.metrics.berth_utilization) == 2
        assert len(self.collector.metrics.container_throughput) == 1
        assert len(self.collector.metrics.queue_lengths) == 1
        
        # Verify calculations work with zero values
        assert self.collector.calculate_average_waiting_time() == 0.0
        assert self.collector.calculate_average_berth_utilization() == 50.0
        assert self.collector.calculate_total_container_throughput() == 0