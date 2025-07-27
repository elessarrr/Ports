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
    get_real_berth_data,
)


class TestLoadSampleData:
    """Test the load_sample_data function"""
    
    def test_load_sample_data_structure(self):
        """Test that load_sample_data returns correct structure"""
        data = load_sample_data()
        
        # Check that all expected keys are present
        expected_keys = ['berths', 'queue', 'timeline', 'waiting', 'kpis', 'vessel_queue_analysis']
        assert all(key in data for key in expected_keys)
        
        # Check that DataFrame values are DataFrames
        dataframe_keys = ['berths', 'queue', 'timeline', 'waiting', 'kpis']
        for key in dataframe_keys:
            assert isinstance(data[key], pd.DataFrame), f"{key} should be a DataFrame"
            assert not data[key].empty, f"{key} DataFrame should not be empty"
        
        # Check that vessel_queue_analysis is a dictionary (not a DataFrame)
        assert isinstance(data['vessel_queue_analysis'], dict), "vessel_queue_analysis should be a dictionary"
    
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
        
        # Check for real data columns (from container throughput)
        if 'seaborne_teus' in timeline_df.columns:
            # Real data structure
            expected_columns = ['time', 'seaborne_teus', 'river_teus', 'total_teus']
            assert all(col in timeline_df.columns for col in expected_columns)
            
            # Check data types
            assert pd.api.types.is_datetime64_any_dtype(timeline_df['time'])
            assert timeline_df['seaborne_teus'].dtype in [np.float64, float, np.int64, int]
            assert timeline_df['river_teus'].dtype in [np.float64, float, np.int64, int]
            assert timeline_df['total_teus'].dtype in [np.float64, float, np.int64, int]
            
            # Check values are non-negative (excluding NaN values)
            assert all(timeline_df['seaborne_teus'].dropna() >= 0)
            assert all(timeline_df['river_teus'].dropna() >= 0)
            assert all(timeline_df['total_teus'].dropna() >= 0)
        else:
            # Fallback data structure
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
    
    def test_data_export_functionality(self):
        """Test data export functionality"""
        import json
        data = load_sample_data()
        
        # Test CSV export for each data type
        berth_csv = data['berths'].to_csv(index=False)
        assert isinstance(berth_csv, str)
        assert len(berth_csv) > 0
        assert 'berth_id' in berth_csv
        
        queue_csv = data['queue'].to_csv(index=False)
        assert isinstance(queue_csv, str)
        assert len(queue_csv) > 0
        assert 'ship_id' in queue_csv
        
        timeline_csv = data['timeline'].to_csv(index=False)
        assert isinstance(timeline_csv, str)
        assert len(timeline_csv) > 0
        assert 'time' in timeline_csv
        
        # Test JSON export
        export_data = {
            'berths': data['berths'].to_dict('records'),
            'queue': data['queue'].to_dict('records'),
            'timeline': data['timeline'].to_dict('records'),
            'waiting': data['waiting'].to_dict('records'),
            'kpis': data['kpis'].to_dict('records'),
            'export_timestamp': datetime.now().isoformat()
        }
        json_data = json.dumps(export_data, indent=2, default=str)
        assert isinstance(json_data, str)
        assert len(json_data) > 0
        
        # Verify JSON can be parsed back
        parsed_data = json.loads(json_data)
        assert 'berths' in parsed_data
        assert 'queue' in parsed_data
        assert 'timeline' in parsed_data
        assert 'export_timestamp' in parsed_data
    
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
        
        # Check that we have reasonable amount of data (real data has 53 rows)
        assert len(timeline_df) >= 25  # At least 25 data points
        
        # For real data, check monthly intervals; for sample data, check hourly
        if len(timeline_df) > 30:  # Likely real data with monthly intervals
            # Real data has monthly intervals, so we just check it's sorted
            assert timeline_df['time'].is_monotonic_increasing
        else:
            # Sample data should have hourly intervals
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


class TestRealTimeBerthData:
    """Test real-time berth data functionality"""
    
    def test_get_real_berth_data_structure(self):
        """Test that get_real_berth_data returns correct structure"""
        berth_data, berth_metrics = get_real_berth_data()
        
        # Check that berth_data is a DataFrame
        assert isinstance(berth_data, pd.DataFrame)
        assert not berth_data.empty
        
        # Check that berth_metrics is a dictionary
        assert isinstance(berth_metrics, dict)
        
        # Check required columns in berth_data
        expected_columns = ['berth_id', 'name', 'status', 'berth_type', 'crane_count', 
                          'max_capacity_teu', 'is_occupied', 'utilization']
        assert all(col in berth_data.columns for col in expected_columns)
        
        # Check required keys in berth_metrics
        expected_keys = ['total_berths', 'occupied_berths', 'available_berths', 
                        'utilization_rate', 'berth_types']
        assert all(key in berth_metrics for key in expected_keys)
    
    def test_berth_data_consistency(self):
        """Test that berth data is internally consistent"""
        berth_data, berth_metrics = get_real_berth_data()
        
        # Check that metrics match the data
        total_berths = len(berth_data)
        occupied_berths = len(berth_data[berth_data['status'] == 'occupied'])
        available_berths = len(berth_data[berth_data['status'] == 'available'])
        
        assert berth_metrics['total_berths'] == total_berths
        assert berth_metrics['occupied_berths'] == occupied_berths
        assert berth_metrics['available_berths'] == available_berths
        
        # Check utilization rate calculation
        expected_utilization = (occupied_berths / total_berths * 100) if total_berths > 0 else 0
        assert abs(berth_metrics['utilization_rate'] - expected_utilization) < 0.1
    
    def test_berth_data_types(self):
        """Test that berth data has correct data types"""
        berth_data, berth_metrics = get_real_berth_data()
        
        # Check DataFrame column types
        assert berth_data['berth_id'].dtype == object  # string
        assert berth_data['name'].dtype == object  # string
        assert berth_data['status'].dtype == object  # string
        assert berth_data['berth_type'].dtype == object  # string
        assert berth_data['crane_count'].dtype in [np.int64, int]
        assert berth_data['max_capacity_teu'].dtype in [np.int64, int]
        assert berth_data['is_occupied'].dtype == bool
        assert berth_data['utilization'].dtype in [np.int64, int, np.float64, float]
        
        # Check metrics types
        assert isinstance(berth_metrics['total_berths'], int)
        assert isinstance(berth_metrics['occupied_berths'], int)
        assert isinstance(berth_metrics['available_berths'], int)
        assert isinstance(berth_metrics['utilization_rate'], (int, float))
        assert isinstance(berth_metrics['berth_types'], dict)
    
    def test_berth_status_values(self):
        """Test that berth status values are valid"""
        berth_data, berth_metrics = get_real_berth_data()
        
        # Check valid status values
        valid_statuses = ['occupied', 'available', 'maintenance']
        assert all(status in valid_statuses for status in berth_data['status'])
        
        # Check valid berth types
        valid_berth_types = ['container', 'bulk', 'mixed']
        assert all(berth_type in valid_berth_types for berth_type in berth_data['berth_type'])
        
        # Check that is_occupied matches status
        for _, row in berth_data.iterrows():
            if row['status'] == 'occupied':
                assert row['is_occupied'] == True
                assert row['utilization'] > 0
            elif row['status'] == 'available':
                assert row['is_occupied'] == False
                assert row['utilization'] == 0
    
    def test_berth_capacity_values(self):
        """Test that berth capacity values are reasonable"""
        berth_data, berth_metrics = get_real_berth_data()
        
        # Check crane count is positive
        assert all(berth_data['crane_count'] > 0)
        assert all(berth_data['crane_count'] <= 10)  # Reasonable upper limit
        
        # Check max capacity is positive
        assert all(berth_data['max_capacity_teu'] > 0)
        assert all(berth_data['max_capacity_teu'] <= 20000)  # Reasonable upper limit
        
        # Check utilization is within valid range
        assert all(berth_data['utilization'] >= 0)
        assert all(berth_data['utilization'] <= 100)
    
    def test_berth_types_distribution(self):
        """Test that berth types are properly distributed"""
        berth_data, berth_metrics = get_real_berth_data()
        
        # Check that berth_types in metrics matches actual data
        actual_types = berth_data['berth_type'].value_counts().to_dict()
        
        for berth_type, count in berth_metrics['berth_types'].items():
            assert berth_type in actual_types
            assert actual_types[berth_type] == count
    
    def test_fallback_to_sample_data(self):
        """Test that function falls back to sample data when BerthManager fails"""
        # This test ensures the function is robust and doesn't crash
        # even if there are issues with the BerthManager
        
        with patch('src.dashboard.streamlit_app.BerthManager') as mock_berth_manager:
            # Make BerthManager raise an exception
            mock_berth_manager.side_effect = Exception("Simulated BerthManager failure")
            
            # Function should still return valid data (fallback)
            berth_data, berth_metrics = get_real_berth_data()
            
            assert isinstance(berth_data, pd.DataFrame)
            assert isinstance(berth_metrics, dict)
            assert not berth_data.empty
            assert len(berth_metrics) > 0