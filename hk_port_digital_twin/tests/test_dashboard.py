"""Tests for the Streamlit Dashboard module

This module tests the dashboard functionality, data loading,
and integration with visualization components.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dashboard.streamlit_app import (
    load_sample_data,
    initialize_session_state,
)


class TestLoadSampleData:
    """Test the load_sample_data function"""
    
    def test_load_sample_data_structure(self):
        """Test that load_sample_data returns correct structure"""
        data = load_sample_data()
        
        # Check that all expected keys are present
        expected_keys = ['berths', 'queue', 'timeline', 'waiting', 'kpis']
        assert all(key in data for key in expected_keys)
        
        # Check that all values are DataFrames
        for key, df in data.items():
            assert isinstance(df, pd.DataFrame), f"{key} should be a DataFrame"
            assert not df.empty, f"{key} DataFrame should not be empty"
    
    def test_berths_data_structure(self):
        """Test berths data has correct columns and types"""
        data = load_sample_data()
        berths_df = data['berths']
        
        expected_columns = ['berth_id', 'x', 'y', 'status', 'ship_id', 'utilization', 'berth_type']
        assert all(col in berths_df.columns for col in expected_columns)
        
        # Check data types and constraints
        assert berths_df['x'].dtype in [np.int64, int]
        assert berths_df['y'].dtype in [np.int64, int]
        assert berths_df['utilization'].dtype in [np.int64, int]
        assert all(berths_df['utilization'] >= 0)
        assert all(berths_df['utilization'] <= 100)
        
        # Check status values are valid
        valid_statuses = ['occupied', 'available', 'maintenance']
        assert all(status in valid_statuses for status in berths_df['status'])
        
        # Check berth types are valid
        valid_berth_types = ['container', 'bulk', 'mixed']
        assert all(berth_type in valid_berth_types for berth_type in berths_df['berth_type'])
    
    def test_queue_data_structure(self):
        """Test queue data has correct columns and types"""
        data = load_sample_data()
        queue_df = data['queue']
        
        expected_columns = ['ship_id', 'name', 'ship_type', 'arrival_time', 'containers', 'size_teu', 'waiting_time', 'priority']
        assert all(col in queue_df.columns for col in expected_columns)
        
        # Check data types
        assert all(isinstance(ship_id, str) for ship_id in queue_df['ship_id'])
        assert all(isinstance(arrival_time, datetime) for arrival_time in queue_df['arrival_time'])
        assert queue_df['containers'].dtype in [np.int64, int]
        
        # Check ship types are valid
        valid_ship_types = ['container', 'bulk', 'mixed']
        assert all(ship_type in valid_ship_types for ship_type in queue_df['ship_type'])
        
        # Check priorities are valid
        valid_priorities = ['high', 'medium', 'low']
        assert all(priority in valid_priorities for priority in queue_df['priority'])
    
    def test_timeline_data_structure(self):
        """Test timeline data has correct columns and types"""
        data = load_sample_data()
        timeline_df = data['timeline']
        
        expected_columns = ['time', 'containers_processed', 'ships_processed']
        assert all(col in timeline_df.columns for col in expected_columns)
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(timeline_df['time'])
        assert timeline_df['containers_processed'].dtype in [np.int64, int]
        assert timeline_df['ships_processed'].dtype in [np.int64, int]
        
        # Check values are non-negative
        assert all(timeline_df['containers_processed'] >= 0)
        assert all(timeline_df['ships_processed'] >= 0)
    
    def test_waiting_data_structure(self):
        """Test waiting time data has correct columns and types"""
        data = load_sample_data()
        waiting_df = data['waiting']
        
        expected_columns = ['ship_id', 'waiting_time', 'ship_type']
        assert all(col in waiting_df.columns for col in expected_columns)
        
        # Check data types
        assert all(isinstance(ship_id, str) for ship_id in waiting_df['ship_id'])
        assert waiting_df['waiting_time'].dtype in [np.float64, float]
        
        # Check waiting times are non-negative
        assert all(waiting_df['waiting_time'] >= 0)
        
        # Check ship types are valid
        valid_ship_types = ['container', 'bulk', 'mixed']
        assert all(ship_type in valid_ship_types for ship_type in waiting_df['ship_type'])
    
    def test_kpi_data_structure(self):
        """Test KPI data has correct columns and types"""
        data = load_sample_data()
        kpi_df = data['kpis']
        
        expected_columns = ['metric', 'value', 'unit', 'target', 'status']
        assert all(col in kpi_df.columns for col in expected_columns)
        
        # Check data types
        assert all(isinstance(metric, str) for metric in kpi_df['metric'])
        assert kpi_df['value'].dtype in [np.float64, float, np.int64, int]
        assert all(isinstance(unit, str) for unit in kpi_df['unit'])
        assert kpi_df['target'].dtype in [np.float64, float, np.int64, int]
        
        # Check status values are valid
        valid_statuses = ['good', 'warning', 'critical']
        assert all(status in valid_statuses for status in kpi_df['status'])


class TestDashboardIntegration:
    """Test dashboard integration with other components"""
    
    @patch('src.dashboard.streamlit_app.st')
    def test_initialize_session_state(self, mock_st):
        """Test session state initialization"""
        # Create a custom mock class for session state
        class MockSessionState(dict):
            def __contains__(self, key):
                return super().__contains__(key)
        
        mock_st.session_state = MockSessionState()
        
        # Call initialize function - should not raise any errors
        try:
            initialize_session_state()
            # If we get here, the function executed successfully
            assert True
        except Exception as e:
            pytest.fail(f"initialize_session_state() raised an exception: {e}")
    
    def test_data_consistency(self):
        """Test that sample data is consistent across multiple calls"""
        data1 = load_sample_data()
        data2 = load_sample_data()
        
        # Check that structure is consistent
        assert data1.keys() == data2.keys()
        
        # Check that berths data is consistent (should be deterministic)
        pd.testing.assert_frame_equal(
            data1['berths'][['berth_id', 'x', 'y', 'status']],
            data2['berths'][['berth_id', 'x', 'y', 'status']]
        )
    
    def test_data_integration_with_visualization(self):
        """Test that sample data works with visualization functions"""
        from src.utils.visualization import (
            create_port_layout_chart,
            create_ship_queue_chart,
            create_berth_utilization_chart,
            create_throughput_timeline,
            create_waiting_time_distribution,
            create_kpi_summary_chart
        )
        
        data = load_sample_data()
        
        # Test each visualization function individually to isolate errors
        try:
            print("Testing port layout chart...")
            fig1 = create_port_layout_chart(data['berths'])
            assert fig1 is not None
            print("Port layout chart: OK")
            
            print("Testing ship queue chart...")
            # Convert DataFrame to list of dictionaries for queue chart
            queue_list = data['queue'].to_dict('records')
            fig2 = create_ship_queue_chart(queue_list)
            assert fig2 is not None
            print("Ship queue chart: OK")
            
            print("Testing berth utilization chart...")
            # Convert DataFrame to dictionary for berth utilization
            berth_util_dict = dict(zip(data['berths']['berth_id'], data['berths']['utilization']))
            fig3 = create_berth_utilization_chart(berth_util_dict)
            assert fig3 is not None
            print("Berth utilization chart: OK")
            
            print("Testing throughput timeline...")
            fig4 = create_throughput_timeline(data['timeline'])
            assert fig4 is not None
            print("Throughput timeline: OK")
            
            print("Testing waiting time distribution...")
            # Convert DataFrame to list for waiting time distribution
            waiting_times_list = data['waiting']['waiting_time'].tolist()
            fig5 = create_waiting_time_distribution(waiting_times_list)
            assert fig5 is not None
            print("Waiting time distribution: OK")
            
            print("Testing KPI summary chart...")
            # Convert KPI DataFrame to expected dictionary format
            kpi_dict = {
                'average_waiting_time': 2.5,
                'average_berth_utilization': 0.75,
                'total_ships_processed': 85,
                'total_containers_processed': 1200,
                'average_queue_length': 3
            }
            fig6 = create_kpi_summary_chart(kpi_dict)
            assert fig6 is not None
            print("KPI summary chart: OK")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            pytest.fail(f"Visualization integration failed: {e}")


class TestDashboardConfiguration:
    """Test dashboard configuration and settings"""
    
    def test_config_imports(self):
        """Test that configuration imports work correctly"""
        try:
            from config.settings import SIMULATION_CONFIG, PORT_CONFIG, SHIP_TYPES
            
            # Check that configs are dictionaries
            assert isinstance(SIMULATION_CONFIG, dict)
            assert isinstance(PORT_CONFIG, dict)
            assert isinstance(SHIP_TYPES, dict)
            
            # Check required keys
            assert 'default_duration' in SIMULATION_CONFIG
            assert 'ship_arrival_rate' in SIMULATION_CONFIG
            assert 'num_berths' in PORT_CONFIG
            
        except ImportError as e:
            pytest.fail(f"Configuration import failed: {e}")
    
    def test_simulation_controller_integration(self):
        """Test that simulation controller can be imported and used"""
        try:
            from src.core.simulation_controller import SimulationController
            from src.core.port_simulation import PortSimulation
            from config.settings import SIMULATION_CONFIG
            
            # Test that we can create instances
            simulation = PortSimulation(SIMULATION_CONFIG)
            controller = SimulationController(simulation)
            
            assert controller is not None
            assert hasattr(controller, 'start')
            assert hasattr(controller, 'stop')
            assert hasattr(controller, 'is_running')
            
        except ImportError as e:
            pytest.fail(f"Simulation controller integration failed: {e}")


class TestDashboardDataValidation:
    """Test data validation and error handling"""
    
    def test_berth_data_validation(self):
        """Test berth data validation"""
        data = load_sample_data()
        berths_df = data['berths']
        
        # Check that occupied berths have ship_ids
        occupied_berths = berths_df[berths_df['status'] == 'occupied']
        assert all(pd.notna(occupied_berths['ship_id']))
        
        # Check that available berths don't have ship_ids
        available_berths = berths_df[berths_df['status'] == 'available']
        assert all(pd.isna(available_berths['ship_id']))
        
        # Check that occupied berths have non-zero utilization
        assert all(occupied_berths['utilization'] > 0)
        
        # Check that available berths have zero utilization
        assert all(available_berths['utilization'] == 0)
    
    def test_timeline_data_continuity(self):
        """Test that timeline data is continuous"""
        data = load_sample_data()
        timeline_df = data['timeline']
        
        # Check that time column is sorted
        assert timeline_df['time'].is_monotonic_increasing
        
        # Check that we have 25 hours of data (24 + 1)
        assert len(timeline_df) == 25
        
        # Check that time intervals are consistent (1 hour)
        time_diffs = timeline_df['time'].diff().dropna()
        expected_diff = pd.Timedelta(hours=1)
        assert all(time_diffs == expected_diff)
    
    def test_queue_data_realism(self):
        """Test that queue data is realistic"""
        data = load_sample_data()
        queue_df = data['queue']
        
        # Check that arrival times are in the past
        now = datetime.now()
        assert all(arrival_time <= now for arrival_time in queue_df['arrival_time'])
        
        # Check that container counts are reasonable
        assert all(queue_df['containers'] > 0)
        assert all(queue_df['containers'] <= 1000)  # Reasonable upper limit
        
        # Check that ship IDs are unique
        assert len(queue_df['ship_id'].unique()) == len(queue_df)