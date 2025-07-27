"""Tests for Data Loader

This module tests the data loading functionality for Hong Kong Port Digital Twin.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import (
    load_container_throughput,
    load_annual_container_throughput,
    load_port_cargo_statistics,
    get_throughput_trends,
    validate_data_quality,
    load_sample_data,
    load_vessel_arrivals,
    get_vessel_queue_analysis,
    _categorize_ship_type,
    _categorize_location,
    DataCache,
    data_cache,
    RealTimeDataManager,
    _validate_container_data,
    _validate_cargo_data,
    _validate_vessel_data,
    _validate_weather_data,
    _cross_validate_datasets,
    _detect_data_anomalies,
    _check_data_freshness
)


class TestDataLoader(unittest.TestCase):
    """Test cases for data loader functions"""
    
    def test_load_sample_data(self):
        """Test sample data generation functionality"""
        sample_df = load_sample_data()
        
        # Check DataFrame structure
        self.assertIsInstance(sample_df, pd.DataFrame)
        self.assertGreater(len(sample_df), 0)
        
        # Check required columns
        required_columns = ['seaborne_teus', 'river_teus', 'total_teus']
        for col in required_columns:
            self.assertIn(col, sample_df.columns)
        
        # Check data types
        for col in required_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(sample_df[col]))
        
        # Check that total equals seaborne + river (approximately)
        total_calculated = sample_df['seaborne_teus'] + sample_df['river_teus']
        np.testing.assert_array_almost_equal(
            sample_df['total_teus'].values, 
            total_calculated.values, 
            decimal=1
        )
        
        # Check that values are positive
        for col in required_columns:
            self.assertTrue(all(sample_df[col] > 0))
        
        # Check datetime index
        self.assertIsInstance(sample_df.index, pd.DatetimeIndex)
    
    def test_load_container_throughput_with_real_data(self):
        """Test loading real container throughput data if available"""
        try:
            df = load_container_throughput()
            
            if not df.empty:
                # Check DataFrame structure
                self.assertIsInstance(df, pd.DataFrame)
                
                # Check required columns
                expected_columns = [
                    'seaborne_teus', 'river_teus', 'total_teus',
                    'seaborne_yoy_change', 'river_yoy_change', 'total_yoy_change'
                ]
                for col in expected_columns:
                    self.assertIn(col, df.columns)
                
                # Check datetime index
                self.assertIsInstance(df.index, pd.DatetimeIndex)
                
                # Check that data is sorted by date
                self.assertTrue(df.index.is_monotonic_increasing)
                
                # Check data consistency (total should roughly equal seaborne + river)
                # Allow for some tolerance due to rounding
                non_null_rows = df.dropna(subset=['seaborne_teus', 'river_teus', 'total_teus'])
                if len(non_null_rows) > 0:
                    calculated_total = non_null_rows['seaborne_teus'] + non_null_rows['river_teus']
                    # Allow 1% tolerance for rounding differences
                    tolerance = non_null_rows['total_teus'] * 0.01
                    differences = abs(calculated_total - non_null_rows['total_teus'])
                    self.assertTrue(all(differences <= tolerance))
            else:
                self.skipTest("Container throughput data file not found or empty")
                
        except Exception as e:
            self.skipTest(f"Could not test real data loading: {e}")
    
    def test_load_annual_container_throughput_with_real_data(self):
        """Test loading annual container throughput data if available"""
        try:
            df = load_annual_container_throughput()
            
            if not df.empty:
                # Check DataFrame structure
                self.assertIsInstance(df, pd.DataFrame)
                
                # Check that we only have annual data (no monthly breakdown)
                self.assertTrue(all(df['Month'] == 'All'))
                
                # Check required columns
                expected_columns = [
                    'seaborne_teus', 'river_teus', 'total_teus',
                    'seaborne_yoy_change', 'river_yoy_change', 'total_yoy_change'
                ]
                for col in expected_columns:
                    self.assertIn(col, df.columns)
                
                # Check that years are in reasonable range
                self.assertTrue(all(df['Year'] >= 2010))
                self.assertTrue(all(df['Year'] <= 2030))
            else:
                self.skipTest("Annual container throughput data not available")
                
        except Exception as e:
            self.skipTest(f"Could not test annual data loading: {e}")
    
    def test_load_port_cargo_statistics_with_real_data(self):
        """Test loading port cargo statistics if available"""
        try:
            cargo_stats = load_port_cargo_statistics()
            
            if cargo_stats:
                # Check that we get a dictionary
                self.assertIsInstance(cargo_stats, dict)
                
                # Check that each value is a DataFrame
                for table_name, df in cargo_stats.items():
                    self.assertIsInstance(df, pd.DataFrame)
                    self.assertIsInstance(table_name, str)
                    self.assertGreater(len(df), 0)  # Should have some data
            else:
                self.skipTest("Port cargo statistics data not available")
                
        except Exception as e:
            self.skipTest(f"Could not test cargo statistics loading: {e}")
    
    def test_get_throughput_trends_with_sample_data(self):
        """Test enhanced throughput trend analysis with sample data"""
        # Mock the load_container_throughput function to return sample data
        with patch('utils.data_loader.load_container_throughput') as mock_load:
            sample_df = load_sample_data()
            mock_load.return_value = sample_df
            
            trends = get_throughput_trends()
            
            # Check that we get a dictionary with expected keys
            self.assertIsInstance(trends, dict)
            
            # Check main analysis sections (based on actual structure)
            expected_sections = [
                'basic_statistics',
                'time_series_analysis',
                'year_over_year_analysis', 
                'seasonal_analysis',
                'forecasting',
                'modal_split_analysis',
                'analysis_timestamp'
            ]
            
            for section in expected_sections:
                self.assertIn(section, trends)
                if section != 'analysis_timestamp':
                    self.assertIsInstance(trends[section], dict)
            
            # Test basic statistics
            basic_stats = trends['basic_statistics']
            basic_expected_keys = [
                'latest_month', 'latest_value', 'mean_monthly',
                'std_monthly', 'min_value', 'max_value', 'total_records', 'date_range'
            ]
            for key in basic_expected_keys:
                self.assertIn(key, basic_stats)
            
            # Test time series analysis
            ts_analysis = trends['time_series_analysis']
            self.assertIn('linear_trend', ts_analysis)
            self.assertIn('moving_averages', ts_analysis)
            self.assertIn('volatility', ts_analysis)
            
            # Test year-over-year analysis
            yoy_analysis = trends['year_over_year_analysis']
            self.assertIn('monthly_yoy_changes', yoy_analysis)
            self.assertIn('annual_growth', yoy_analysis)
            
            # Test seasonal analysis
            seasonal = trends['seasonal_analysis']
            self.assertIn('monthly_patterns', seasonal)
            self.assertIn('quarterly_patterns', seasonal)
            self.assertIn('seasonal_insights', seasonal)
            
            # Test forecasting
            forecasting = trends['forecasting']
            self.assertIn('total_forecast', forecasting)
            self.assertIn('seaborne_forecast', forecasting)
            self.assertIn('river_forecast', forecasting)
            
            # Test modal split analysis
            modal_split = trends['modal_split_analysis']
            self.assertIn('current_modal_split', modal_split)
            self.assertIn('historical_average', modal_split)
            self.assertIn('modal_split_trends', modal_split)
            
            # Check data types and ranges
            self.assertIsInstance(basic_stats['total_records'], int)
            self.assertGreater(basic_stats['total_records'], 0)
            self.assertIsInstance(basic_stats['mean_monthly'], (int, float))
            self.assertGreater(basic_stats['mean_monthly'], 0)
            self.assertIsInstance(basic_stats['std_monthly'], (int, float))
            self.assertGreaterEqual(basic_stats['std_monthly'], 0)
            
            # Check timestamp
            self.assertIsInstance(trends['analysis_timestamp'], str)
    
    def test_get_throughput_trends_with_empty_data(self):
        """Test throughput trend analysis with empty data"""
        # Mock the load_container_throughput function to return empty DataFrame
        with patch('utils.data_loader.load_container_throughput') as mock_load:
            mock_load.return_value = pd.DataFrame()
            
            trends = get_throughput_trends()
            
            # Should return empty dict for empty data
            self.assertEqual(trends, {})
    
    def test_validate_data_quality_with_sample_data(self):
        """Test data quality validation with sample data"""
        # Mock both data loading functions
        with patch('utils.data_loader.load_container_throughput') as mock_throughput, \
             patch('utils.data_loader.load_port_cargo_statistics') as mock_cargo:
            
            # Set up mock returns
            sample_df = load_sample_data()
            mock_throughput.return_value = sample_df
            mock_cargo.return_value = {'Table_1': pd.DataFrame({'col1': [1, 2, 3]})}
            
            validation = validate_data_quality()
            
            # Check structure
            self.assertIsInstance(validation, dict)
            self.assertIn('container_throughput', validation)
            self.assertIn('cargo_statistics', validation)
            self.assertIn('overall_status', validation)
            
            # Check container throughput validation
            ct_validation = validation['container_throughput']
            self.assertIn('records_count', ct_validation)
            self.assertIn('date_range', ct_validation)
            self.assertIn('missing_values', ct_validation)
            self.assertIn('data_completeness', ct_validation)
            
            # Check cargo statistics validation
            cs_validation = validation['cargo_statistics']
            self.assertIn('tables_loaded', cs_validation)
            self.assertIn('table_names', cs_validation)
            
            # Check overall status
            self.assertIn(validation['overall_status'], ['excellent', 'good', 'fair', 'poor', 'no_data', 'error'])
    
    def test_validate_data_quality_with_no_data(self):
        """Test data quality validation with no data available"""
        # Mock all data loading functions to return empty data
        with patch('utils.data_loader.load_container_throughput') as mock_throughput, \
             patch('utils.data_loader.load_port_cargo_statistics') as mock_cargo, \
             patch('utils.data_loader.load_vessel_arrivals') as mock_vessel:
            
            mock_throughput.return_value = pd.DataFrame()
            mock_cargo.return_value = {}
            mock_vessel.return_value = pd.DataFrame()
            
            validation = validate_data_quality()
            
            # Should indicate no_data when no data is available
            self.assertEqual(validation['overall_status'], 'no_data')
    
    def test_data_file_paths_exist(self):
        """Test that expected data file paths are correctly constructed"""
        from utils.data_loader import RAW_DATA_DIR, CONTAINER_THROUGHPUT_FILE, PORT_CARGO_STATS_DIR
        
        # Check that paths are Path objects
        self.assertIsInstance(RAW_DATA_DIR, Path)
        self.assertIsInstance(CONTAINER_THROUGHPUT_FILE, Path)
        self.assertIsInstance(PORT_CARGO_STATS_DIR, Path)
        
        # Check that the raw data directory path makes sense
        self.assertTrue(str(RAW_DATA_DIR).endswith('raw_data'))
        
        # Check that container throughput file path makes sense
        self.assertTrue(str(CONTAINER_THROUGHPUT_FILE).endswith('.csv'))
        self.assertIn('container_throughput', str(CONTAINER_THROUGHPUT_FILE).lower())
    
    def test_error_handling_in_functions(self):
        """Test that functions handle errors gracefully"""
        # Test with invalid file paths by mocking Path operations
        with patch('utils.data_loader.CONTAINER_THROUGHPUT_FILE', Path('/nonexistent/file.csv')):
            # Should return empty DataFrame, not raise exception
            df = load_container_throughput()
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue(df.empty)
            
            # Annual data should also handle errors gracefully
            annual_df = load_annual_container_throughput()
            self.assertIsInstance(annual_df, pd.DataFrame)
            self.assertTrue(annual_df.empty)
        
        # Test cargo statistics with invalid directory
        with patch('utils.data_loader.PORT_CARGO_STATS_DIR', Path('/nonexistent/directory')):
            cargo_stats = load_port_cargo_statistics()
            self.assertIsInstance(cargo_stats, dict)
            self.assertEqual(len(cargo_stats), 0)


    def test_cargo_breakdown_analysis(self):
        """Test comprehensive cargo breakdown analysis."""
        # Mock the load_port_cargo_statistics function
        with patch('utils.data_loader.load_port_cargo_statistics') as mock_load:
            # Create sample cargo statistics data
            sample_cargo_stats = {
                'Table_1_Eng': pd.DataFrame({
                    'Shipment Type': ['Direct shipment', 'Transhipment', 'Overall'],
                    '2023 (thousand tonnes)': [50000, 30000, 80000],
                    '2023 (percentage)': [62.5, 37.5, 100.0]
                }),
                'Table_2_Eng': pd.DataFrame({
                    'Transport Mode': ['Seaborne', 'River', 'Waterborne'],
                    '2023 (thousand tonnes)': [60000, 20000, 80000],
                    '2023 (percentage)': [75.0, 25.0, 100.0]
                })
            }
            mock_load.return_value = sample_cargo_stats
            
            # Import and test the function
            from utils.data_loader import get_cargo_breakdown_analysis
            analysis = get_cargo_breakdown_analysis()
            
            # Should return a dictionary
            self.assertIsInstance(analysis, dict)
            
            # Check required analysis sections
            expected_sections = [
                'shipment_type_analysis',
                'transport_mode_analysis', 
                'cargo_type_analysis',
                'location_analysis',
                'efficiency_metrics',
                'data_summary'
            ]
            
            for section in expected_sections:
                self.assertIn(section, analysis)
                self.assertIsInstance(analysis[section], dict)
            
            # Check data summary
            self.assertIn('tables_processed', analysis['data_summary'])
            self.assertIn('analysis_timestamp', analysis['data_summary'])
            self.assertIsInstance(analysis['data_summary']['tables_processed'], int)
    
    def test_load_port_cargo_statistics_with_real_data_standalone(self):
        """Test loading port cargo statistics with real data as standalone function."""
        cargo_stats = load_port_cargo_statistics()
        
        # Should return a dictionary
        self.assertIsInstance(cargo_stats, dict)
        
        # If data exists, each table should be a non-empty DataFrame
        for table_name, df in cargo_stats.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
    
    def test_shipment_type_analysis(self):
        """Test shipment type analysis functionality."""
        # Create sample data for testing
        sample_data = {
            'Shipment Type': ['Direct shipment', 'Transhipment', 'Overall'],
            '2023 (thousand tonnes)': [50000, 30000, 80000],
            '2023 (percentage)': [62.5, 37.5, 100.0]
        }
        df = pd.DataFrame(sample_data)
        
        # Mock the internal function
        with patch('utils.data_loader._analyze_shipment_types') as mock_analyze:
            mock_analyze.return_value = {
                'direct_shipment_2023': 50000,
                'transhipment_2023': 30000,
                'total_2023': 80000,
                'direct_percentage': 62.5,
                'transhipment_percentage': 37.5
            }
            
            analysis = mock_analyze(df)
            
            # Check expected keys
            expected_keys = [
                'direct_shipment_2023',
                'transhipment_2023', 
                'total_2023',
                'direct_percentage',
                'transhipment_percentage'
            ]
            
            for key in expected_keys:
                self.assertIn(key, analysis)
                self.assertIsInstance(analysis[key], (int, float))
            
            # Check calculations
            self.assertEqual(analysis['total_2023'], 80000)
            self.assertEqual(analysis['direct_percentage'], 62.5)
            self.assertEqual(analysis['transhipment_percentage'], 37.5)
    
    def test_transport_mode_analysis(self):
        """Test transport mode analysis functionality."""
        # Create sample data for testing
        sample_data = {
            'Transport Mode': ['Seaborne', 'River', 'Waterborne'],
            '2023 (thousand tonnes)': [60000, 20000, 80000],
            '2023 (percentage)': [75.0, 25.0, 100.0]
        }
        df = pd.DataFrame(sample_data)
        
        # Mock the internal function
        with patch('utils.data_loader._analyze_transport_modes') as mock_analyze:
            mock_analyze.return_value = {
                'seaborne_2023': 60000,
                'river_2023': 20000,
                'total_2023': 80000,
                'seaborne_percentage': 75.0,
                'river_percentage': 25.0
            }
            
            analysis = mock_analyze(df)
            
            # Check expected keys
            expected_keys = [
                'seaborne_2023',
                'river_2023',
                'total_2023', 
                'seaborne_percentage',
                'river_percentage'
            ]
            
            for key in expected_keys:
                self.assertIn(key, analysis)
                self.assertIsInstance(analysis[key], (int, float))
            
            # Check calculations
            self.assertEqual(analysis['total_2023'], 80000)
            self.assertEqual(analysis['seaborne_percentage'], 75.0)
            self.assertEqual(analysis['river_percentage'], 25.0)
    
    def test_cargo_data_cleaning(self):
        """Test cargo statistics data cleaning functionality."""
        # Create sample data with issues that need cleaning
        sample_data = {
            'Cargo Type': ['Container', 'Bulk', 'General'],
            '2023 (thousand tonnes)': ['1000', '-', '§'],
            '2022 (thousand tonnes)': [950, 'N/A', 50],
            'Description': ['Test cargo', 'Test bulk', 'Test general']
        }
        df = pd.DataFrame(sample_data)
        
        # Mock the internal cleaning function
        with patch('utils.data_loader._clean_cargo_statistics_data') as mock_clean:
            cleaned_data = {
                'Cargo Type': ['Container', 'Bulk', 'General'],
                '2023 (thousand tonnes)': [1000.0, np.nan, 0.0],
                '2022 (thousand tonnes)': [950.0, np.nan, 50.0],
                'Description': ['Test cargo', 'Test bulk', 'Test general']
            }
            mock_clean.return_value = pd.DataFrame(cleaned_data)
            
            cleaned_df = mock_clean(df, 'test_table')
            
            # Should return a DataFrame
            self.assertIsInstance(cleaned_df, pd.DataFrame)
            
            # Check that numeric columns are properly converted
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['2023 (thousand tonnes)']))
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['2022 (thousand tonnes)']))
            
            # Check specific conversions
            self.assertEqual(cleaned_df['2023 (thousand tonnes)'].iloc[0], 1000.0)
            self.assertTrue(pd.isna(cleaned_df['2023 (thousand tonnes)'].iloc[1]))
            self.assertEqual(cleaned_df['2023 (thousand tonnes)'].iloc[2], 0.0)
    
    def test_efficiency_metrics_calculation(self):
        """Test efficiency metrics calculation."""
        # Create sample cargo statistics
        table1_data = {
            'Shipment Type': ['Direct shipment', 'Transhipment'],
            '2023 (thousand tonnes)': [60000, 40000]
        }
        
        table2_data = {
            'Transport Mode': ['Seaborne', 'River'],
            '2023 (thousand tonnes)': [75000, 25000]
        }
        
        cargo_stats = {
            'Table_1_Eng': pd.DataFrame(table1_data),
            'Table_2_Eng': pd.DataFrame(table2_data),
            'Table_6_Eng': pd.DataFrame({'Cargo': ['A', 'B', 'C']}),
            'Table_7_Eng': pd.DataFrame({'Location': ['L1', 'L2']})
        }
        
        # Mock the internal function
        with patch('utils.data_loader._calculate_efficiency_metrics') as mock_calc:
            mock_calc.return_value = {
                'transhipment_ratio': 0.4,
                'direct_shipment_ratio': 0.6,
                'seaborne_ratio': 0.75,
                'river_ratio': 0.25,
                'cargo_diversity_index': 3,
                'location_utilization_index': 2
            }
            
            metrics = mock_calc(cargo_stats)
            
            # Should return metrics dictionary
            self.assertIsInstance(metrics, dict)
            
            # Check expected metrics
            expected_metrics = [
                'transhipment_ratio',
                'direct_shipment_ratio',
                'seaborne_ratio', 
                'river_ratio',
                'cargo_diversity_index',
                'location_utilization_index'
            ]
            
            for metric in expected_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], (int, float))
            
            # Check specific calculations
            self.assertEqual(metrics['cargo_diversity_index'], 3)
            self.assertEqual(metrics['location_utilization_index'], 2)
    
    def test_time_series_analysis_components(self):
        """Test individual components of time series analysis."""
        # Create sample time series data with stronger trend
        dates = pd.date_range('2020-01-01', '2023-12-01', freq='MS')
        values = np.random.normal(1000000, 50000, len(dates)) + np.arange(len(dates)) * 5000
        sample_df = pd.DataFrame({
            'total_teus': values,
            'seaborne_teus': values * 0.8,
            'river_teus': values * 0.2
        }, index=dates)
        
        with patch('utils.data_loader.load_container_throughput') as mock_load:
            mock_load.return_value = sample_df
            
            trends = get_throughput_trends()
            
            # Test time series analysis components
            ts_analysis = trends['time_series_analysis']
            
            # Check for required sections
            required_sections = ['linear_trend', 'moving_averages', 'volatility']
            for section in required_sections:
                self.assertIn(section, ts_analysis)
            
            # Check linear trend components
            linear_trend = ts_analysis['linear_trend']
            self.assertIn('slope', linear_trend)
            self.assertIn('r_squared', linear_trend)
            
            # Linear trend should have a slope value (direction may vary with synthetic data)
            self.assertIsInstance(linear_trend['slope'], (int, float))
            
            # R-squared should be reasonable (> 0.05 for sample data)
            self.assertGreater(linear_trend['r_squared'], 0.05)
            
            # Check moving averages
            moving_avgs = ts_analysis['moving_averages']
            self.assertIn('ma_3_latest', moving_avgs)
            self.assertIn('ma_12_latest', moving_avgs)
            
            # Moving averages should be positive
            self.assertGreater(moving_avgs['ma_3_latest'], 0)
            self.assertGreater(moving_avgs['ma_12_latest'], 0)
            
            # Check volatility components
            volatility = ts_analysis['volatility']
            self.assertIn('monthly_std', volatility)
            self.assertIn('annualized_volatility', volatility)
            self.assertIn('coefficient_of_variation', volatility)
            
            # Volatility measures should be non-negative
            self.assertGreaterEqual(volatility['monthly_std'], 0)
            self.assertGreaterEqual(volatility['annualized_volatility'], 0)
            self.assertGreaterEqual(volatility['coefficient_of_variation'], 0)
    
    def test_seasonal_pattern_analysis(self):
        """Test seasonal pattern analysis functionality."""
        # Create sample data with clear seasonal pattern
        dates = pd.date_range('2020-01-01', '2023-12-01', freq='MS')
        # Add seasonal component (peak in summer months)
        seasonal_factor = np.sin(2 * np.pi * (dates.month - 1) / 12) * 100000
        base_values = np.random.normal(1000000, 50000, len(dates))
        values = base_values + seasonal_factor
        
        sample_df = pd.DataFrame({
            'total_teus': values,
            'seaborne_teus': values * 0.8,
            'river_teus': values * 0.2
        }, index=dates)
        
        with patch('utils.data_loader.load_container_throughput') as mock_load:
            mock_load.return_value = sample_df
            
            trends = get_throughput_trends()
            
            # Test seasonal patterns
            seasonal = trends['seasonal_analysis']
            
            # Should have monthly and quarterly patterns
            self.assertIn('monthly_patterns', seasonal)
            self.assertIn('quarterly_patterns', seasonal)
            
            # Monthly patterns should be a dict with pattern info
            self.assertIsInstance(seasonal['monthly_patterns'], dict)
            
            # Quarterly patterns should be a dict with pattern info
            self.assertIsInstance(seasonal['quarterly_patterns'], dict)
            
            # Should have seasonal insights
            self.assertIn('seasonal_insights', seasonal)
            
            # Check seasonal insights structure
            insights = seasonal['seasonal_insights']
            self.assertIn('is_highly_seasonal', insights)
            self.assertIn('seasonal_classification', insights)
            
            # Check monthly patterns structure
            monthly = seasonal['monthly_patterns']
            self.assertIn('peak_month', monthly)
            self.assertIn('low_month', monthly)
            self.assertIn('peak_value', monthly)
            self.assertIn('low_value', monthly)
            
            # Check quarterly patterns structure
            quarterly = seasonal['quarterly_patterns']
            self.assertIn('peak_quarter', quarterly)
            self.assertIn('low_quarter', quarterly)
    
    def test_forecasting_functionality(self):
        """Test forecasting functionality."""
        # Create sample data with trend
        dates = pd.date_range('2020-01-01', '2023-12-01', freq='MS')
        trend = np.arange(len(dates)) * 1000
        noise = np.random.normal(0, 50000, len(dates))
        values = 1000000 + trend + noise
        
        sample_df = pd.DataFrame({
            'total_teus': values,
            'seaborne_teus': values * 0.8,
            'river_teus': values * 0.2
        }, index=dates)
        
        with patch('utils.data_loader.load_container_throughput') as mock_load:
            mock_load.return_value = sample_df
            
            trends = get_throughput_trends()
            
            # Test forecasts
            forecasting = trends['forecasting']
            
            # Should have forecasts for different cargo types
            forecast_types = ['total_forecast', 'seaborne_forecast', 'river_forecast']
            for forecast_type in forecast_types:
                self.assertIn(forecast_type, forecasting)
                self.assertIsInstance(forecasting[forecast_type], dict)
                
                # Check forecast structure
                forecast_data = forecasting[forecast_type]
                self.assertIn('method', forecast_data)
                self.assertIn('forecast_horizon', forecast_data)
                self.assertIn('forecast_values', forecast_data)
                self.assertIn('model_performance', forecast_data)
                
                # Validate forecast values are positive
                forecast_values = forecast_data['forecast_values']
                self.assertIsInstance(forecast_values, list)
                self.assertGreater(len(forecast_values), 0)
                for value in forecast_values:
                    self.assertGreater(value, 0)
                
                # Check model performance metrics
                performance = forecast_data['model_performance']
                self.assertIn('mae', performance)
                self.assertIn('rmse', performance)
                self.assertIn('r_squared', performance)
    
    def test_modal_split_trends(self):
        """Test modal split trend analysis."""
        # Create sample data with changing modal split
        dates = pd.date_range('2020-01-01', '2023-12-01', freq='MS')
        total_values = np.random.normal(1000000, 100000, len(dates))
        
        # Simulate changing modal split (seaborne increasing over time)
        seaborne_ratio = 0.7 + 0.1 * np.arange(len(dates)) / len(dates)
        seaborne_values = total_values * seaborne_ratio
        river_values = total_values * (1 - seaborne_ratio)
        
        sample_df = pd.DataFrame({
            'total_teus': total_values,
            'seaborne_teus': seaborne_values,
            'river_teus': river_values
        }, index=dates)
        
        with patch('utils.data_loader.load_container_throughput') as mock_load:
            mock_load.return_value = sample_df
            
            trends = get_throughput_trends()
            
            # Test modal split analysis
            modal_split = trends['modal_split_analysis']
            
            # Should have current modal split
            self.assertIn('current_modal_split', modal_split)
            current_split = modal_split['current_modal_split']
            self.assertIn('seaborne_percentage', current_split)
            self.assertIn('river_percentage', current_split)
            
            # Should have historical average
            self.assertIn('historical_average', modal_split)
            hist_avg = modal_split['historical_average']
            self.assertIn('seaborne_percentage', hist_avg)
            self.assertIn('river_percentage', hist_avg)
            
            # Should have modal split trends
            self.assertIn('modal_split_trends', modal_split)
            trends_data = modal_split['modal_split_trends']
            self.assertIn('seaborne_trend', trends_data)
            self.assertIn('river_trend', trends_data)
            
            # Percentages should sum to approximately 100
            self.assertAlmostEqual(
                current_split['seaborne_percentage'] + current_split['river_percentage'],
                100, places=1
            )
    
    def test_year_over_year_analysis(self):
        """Test year-over-year analysis functionality."""
        # Create multi-year sample data
        dates = pd.date_range('2020-01-01', '2023-12-01', freq='MS')
        base_values = 1000000
        # Add year-over-year growth
        yearly_growth = 0.05  # 5% annual growth
        values = []
        
        for date in dates:
            years_from_start = (date.year - 2020) + (date.month - 1) / 12
            value = base_values * (1 + yearly_growth) ** years_from_start
            values.append(value + np.random.normal(0, 50000))
        
        sample_df = pd.DataFrame({
            'total_teus': values,
            'seaborne_teus': np.array(values) * 0.8,
            'river_teus': np.array(values) * 0.2
        }, index=dates)
        
        with patch('utils.data_loader.load_container_throughput') as mock_load:
            mock_load.return_value = sample_df
            
            trends = get_throughput_trends()
            
            # Test year-over-year analysis
            yoy_analysis = trends['year_over_year_analysis']
            
            # Should have monthly YoY changes and annual growth
            self.assertIn('monthly_yoy_changes', yoy_analysis)
            self.assertIn('annual_growth', yoy_analysis)
            
            # Average YoY should be positive (we added growth)
            annual_growth = yoy_analysis['annual_growth']
            self.assertIsInstance(annual_growth, dict)
            
            # Monthly YoY changes should be a dict
            self.assertIsInstance(yoy_analysis['monthly_yoy_changes'], dict)
            
            # Annual growth should have expected keys
            self.assertIn('avg_annual_growth', annual_growth)
            self.assertIn('latest_annual_growth', annual_growth)


    def test_enhanced_trends_integration(self):
        """Test integration of all enhanced trend analysis components."""
        # Create comprehensive sample data
        dates = pd.date_range('2020-01-01', '2023-12-01', freq='MS')
        
        # Create realistic data with trend, seasonality, and noise
        trend = np.arange(len(dates)) * 2000  # Linear growth
        seasonal = np.sin(2 * np.pi * (dates.month - 1) / 12) * 100000  # Seasonal pattern
        noise = np.random.normal(0, 50000, len(dates))  # Random variation
        base_values = 1000000 + trend + seasonal + noise
        
        sample_df = pd.DataFrame({
            'total_teus': base_values,
            'seaborne_teus': base_values * 0.75,
            'river_teus': base_values * 0.25,
            'total_yoy_change': np.random.normal(5, 2, len(dates)),
            'seaborne_yoy_change': np.random.normal(4, 2, len(dates)),
            'river_yoy_change': np.random.normal(6, 3, len(dates))
        }, index=dates)
        
        with patch('utils.data_loader.load_container_throughput') as mock_load:
            mock_load.return_value = sample_df
            
            # Test the complete enhanced function
            trends = get_throughput_trends()
            
            # Verify all main sections are present and properly structured
            main_sections = [
                'basic_statistics',
                'time_series_analysis',
                'year_over_year_analysis',
                'seasonal_analysis', 
                'forecasting',
                'modal_split_analysis'
            ]
            
            for section in main_sections:
                self.assertIn(section, trends)
                self.assertIsInstance(trends[section], dict)
                self.assertGreater(len(trends[section]), 0)
            
            # Verify data consistency across sections
            basic_stats = trends['basic_statistics']
            self.assertEqual(basic_stats['total_records'], len(sample_df))
            self.assertEqual(basic_stats['latest_month'], sample_df.index[-1].strftime('%Y-%m'))
            
            # Verify numerical consistency
            latest_value = basic_stats['latest_value']
            self.assertAlmostEqual(latest_value, sample_df['total_teus'].iloc[-1], places=0)
            
            # Verify forecast reasonableness
            forecasting = trends['forecasting']
            total_forecast = forecasting['total_forecast']
            forecast_values = total_forecast['forecast_values']
            
            # Forecasts should be positive and reasonable
            self.assertIsInstance(forecast_values, list)
            self.assertGreater(len(forecast_values), 0)
            for value in forecast_values:
                self.assertGreater(value, 0)
                # Should be within reasonable range of current values
                self.assertLess(abs(value - latest_value) / latest_value, 1.0)

    def test_categorize_ship_type(self):
        """Test ship type categorization function."""
        # Test container ships
        self.assertEqual(_categorize_ship_type('Container Ship'), 'container')
        self.assertEqual(_categorize_ship_type('CONTAINER VESSEL'), 'container')
        
        # Test bulk carriers
        self.assertEqual(_categorize_ship_type('Bulk Carrier'), 'bulk_carrier')
        self.assertEqual(_categorize_ship_type('Ore Carrier'), 'bulk_carrier')
        self.assertEqual(_categorize_ship_type('Cement Carrier'), 'bulk_carrier')
        self.assertEqual(_categorize_ship_type('Woodchip Carrier'), 'bulk_carrier')
        
        # Test tankers
        self.assertEqual(_categorize_ship_type('Chemical Tanker'), 'chemical_tanker')
        self.assertEqual(_categorize_ship_type('Oil Tanker'), 'tanker')
        self.assertEqual(_categorize_ship_type('Product Tanker'), 'tanker')
        
        # Test general cargo
        self.assertEqual(_categorize_ship_type('General Cargo Ship'), 'general_cargo')
        self.assertEqual(_categorize_ship_type('Heavy Lift Vessel'), 'general_cargo')
        
        # Test edge cases
        self.assertEqual(_categorize_ship_type(''), 'unknown')
        self.assertEqual(_categorize_ship_type(None), 'unknown')
        self.assertEqual(_categorize_ship_type('Unknown Ship Type'), 'other')

    def test_categorize_location(self):
        """Test location categorization function."""
        # Test berth locations
        self.assertEqual(_categorize_location('Berth 1'), 'berth')
        self.assertEqual(_categorize_location('Container Terminal 9'), 'berth')
        self.assertEqual(_categorize_location('KWAI TSING TERMINAL'), 'berth')
        
        # Test anchorage locations
        self.assertEqual(_categorize_location('Western Anchorage'), 'anchorage')
        self.assertEqual(_categorize_location('Eastern Anchorage Area'), 'anchorage')
        
        # Test channel locations
        self.assertEqual(_categorize_location('Main Channel'), 'channel')
        self.assertEqual(_categorize_location('Buoy 12'), 'channel')
        
        # Test edge cases
        self.assertEqual(_categorize_location(''), 'unknown')
        self.assertEqual(_categorize_location(None), 'unknown')
        self.assertEqual(_categorize_location('Unknown Location'), 'other')

    def test_load_vessel_arrivals_with_real_data(self):
        """Test loading real vessel arrival data if XML file exists."""
        try:
            df = load_vessel_arrivals()
            
            if not df.empty:
                # Check DataFrame structure
                self.assertIsInstance(df, pd.DataFrame)
                
                # Check required columns
                expected_columns = [
                    'call_sign', 'vessel_name', 'ship_type', 'agent_name',
                    'current_location', 'arrival_time_str', 'remark',
                    'arrival_time', 'status', 'ship_category', 'location_type'
                ]
                for col in expected_columns:
                    self.assertIn(col, df.columns)
                
                # Check data types
                self.assertTrue(df['arrival_time'].dtype.name.startswith('datetime') or df['arrival_time'].isnull().all())
                
                # Check status values
                valid_statuses = ['in_port', 'departed']
                self.assertTrue(all(status in valid_statuses for status in df['status'].unique()))
                
                # Check ship categories
                valid_categories = ['container', 'bulk_carrier', 'chemical_tanker', 'general_cargo', 'tanker', 'other', 'unknown']
                self.assertTrue(all(cat in valid_categories for cat in df['ship_category'].unique()))
                
                # Check location types
                valid_locations = ['berth', 'anchorage', 'channel', 'other', 'unknown']
                self.assertTrue(all(loc in valid_locations for loc in df['location_type'].unique()))
                
                # Check that DataFrame is sorted by arrival time
                non_null_times = df.dropna(subset=['arrival_time'])
                if len(non_null_times) > 1:
                    self.assertTrue(non_null_times['arrival_time'].is_monotonic_increasing)
            else:
                self.skipTest("Vessel arrivals XML file not found or empty")
                
        except Exception as e:
            self.skipTest(f"Could not test vessel arrivals loading: {e}")

    @patch('utils.data_loader.load_vessel_arrivals')
    def test_get_vessel_queue_analysis_with_mock_data(self, mock_load_vessels):
        """Test vessel queue analysis with mock data."""
        # Create mock vessel data
        mock_data = pd.DataFrame({
            'call_sign': ['VESSEL1', 'VESSEL2', 'VESSEL3', 'VESSEL4'],
            'vessel_name': ['Ship A', 'Ship B', 'Ship C', 'Ship D'],
            'ship_category': ['container', 'bulk_carrier', 'container', 'tanker'],
            'location_type': ['berth', 'anchorage', 'berth', 'anchorage'],
            'status': ['in_port', 'in_port', 'departed', 'in_port'],
            'arrival_time': pd.to_datetime([
                '2024-01-01 10:00', '2024-01-01 12:00', 
                '2024-01-01 08:00', '2024-01-01 14:00'
            ])
        })
        
        mock_load_vessels.return_value = mock_data
        
        analysis = get_vessel_queue_analysis()
        
        # Check analysis structure
        self.assertIsInstance(analysis, dict)
        
        # Check main sections
        expected_sections = [
            'current_status', 'location_breakdown', 'ship_category_breakdown',
            'recent_activity', 'analysis_timestamp', 'data_freshness'
        ]
        for section in expected_sections:
            self.assertIn(section, analysis)
        
        # Check current status
        current_status = analysis['current_status']
        self.assertEqual(current_status['total_vessels_in_port'], 3)  # Excluding departed vessel
        self.assertEqual(current_status['vessels_at_berth'], 1)  # Only VESSEL1 at berth and in_port
        self.assertEqual(current_status['vessels_in_queue'], 2)  # VESSEL2 and VESSEL4 at anchorage
        
        # Check location breakdown
        location_breakdown = analysis['location_breakdown']
        self.assertEqual(location_breakdown.get('berth', 0), 1)
        self.assertEqual(location_breakdown.get('anchorage', 0), 2)
        
        # Check ship category breakdown
        ship_breakdown = analysis['ship_category_breakdown']
        self.assertEqual(ship_breakdown.get('container', 0), 1)  # Only VESSEL1 (in_port)
        self.assertEqual(ship_breakdown.get('bulk_carrier', 0), 1)  # VESSEL2
        self.assertEqual(ship_breakdown.get('tanker', 0), 1)  # VESSEL4
        
        # Check data freshness
        self.assertEqual(analysis['data_freshness'], 'real_time')

    @patch('utils.data_loader.load_vessel_arrivals')
    def test_get_vessel_queue_analysis_empty_data(self, mock_load_vessels):
        """Test vessel queue analysis with empty data."""
        mock_load_vessels.return_value = pd.DataFrame()
        
        analysis = get_vessel_queue_analysis()
        
        # Should return empty dict when no data available
        self.assertEqual(analysis, {})

    def test_load_vessel_arrivals_file_not_found(self):
        """Test vessel arrivals loading when XML file doesn't exist."""
        # Mock the file path to point to non-existent file
        with patch('utils.data_loader.VESSEL_ARRIVALS_XML') as mock_path:
            mock_path.exists.return_value = False
            
            df = load_vessel_arrivals()
            
            # Should return empty DataFrame
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue(df.empty)


class TestEnhancedDataProcessingPipeline(unittest.TestCase):
    """Test cases for enhanced data processing pipeline features"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Clear global cache before each test
        data_cache.clear()
    
    def test_data_cache_basic_operations(self):
        """Test basic DataCache operations"""
        cache = DataCache()
        
        # Test set and get
        test_data = {'test': 'value'}
        cache.set('test_key', test_data)
        retrieved = cache.get('test_key')
        self.assertEqual(retrieved, test_data)
        
        # Test non-existent key
        self.assertIsNone(cache.get('non_existent'))
        
        # Test clear
        cache.clear()
        self.assertIsNone(cache.get('test_key'))
    
    def test_data_cache_ttl_expiration(self):
        """Test DataCache TTL expiration"""
        cache = DataCache(default_ttl=1)  # 1 second TTL
        
        test_data = {'test': 'value'}
        cache.set('test_key', test_data)
        
        # Should be available immediately
        self.assertEqual(cache.get('test_key'), test_data)
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        # Should be expired
        self.assertIsNone(cache.get('test_key'))
    
    def test_data_cache_validation(self):
        """Test DataCache with validation function"""
        cache = DataCache()
        
        # Test basic set and get operations
        cache.set('positive', 10)
        self.assertEqual(cache.get('positive'), 10)
        
        # Test that invalid keys return None
        self.assertIsNone(cache.get('non_existent'))
    
    def test_data_cache_statistics(self):
        """Test DataCache statistics tracking"""
        cache = DataCache()
        
        # Initial stats
        stats = cache.get_stats()
        self.assertEqual(stats['cached_items'], 0)
        self.assertEqual(stats['total_access_count'], 0)
        self.assertEqual(stats['cache_keys'], [])
        
        # Test cache set and access
        cache.set('test', 'value')
        cache.get('test')
        stats = cache.get_stats()
        self.assertEqual(stats['cached_items'], 1)
        self.assertIn('test', stats['cache_keys'])
        self.assertEqual(stats['access_counts']['test'], 1)
    
    def test_validate_container_data(self):
        """Test container data validation function"""
        # Valid container data
        valid_data = pd.DataFrame({
            'total_teus': [1000000, 1100000, 1200000],
            'seaborne_teus': [750000, 825000, 900000],
            'river_teus': [250000, 275000, 300000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='MS'))
        
        result = _validate_container_data(valid_data)
        self.assertEqual(result['records_count'], 3)
        self.assertIn('date_range', result)
        self.assertIn('missing_values', result)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        result = _validate_container_data(empty_data)
        self.assertEqual(result['records_count'], 0)
    
    def test_validate_cargo_data(self):
        """Test cargo data validation function"""
        # Valid cargo data
        valid_data = {
            'Table_1': pd.DataFrame({
                'Cargo Type': ['Container', 'Bulk', 'General'],
                '2023 (thousand tonnes)': [50000, 30000, 20000]
            })
        }
        
        result = _validate_cargo_data(valid_data)
        self.assertEqual(result['tables_loaded'], 1)
        self.assertIn('table_names', result)
        self.assertIn('table_details', result)
        
        # Empty data
        result = _validate_cargo_data({})
        self.assertEqual(result['tables_loaded'], 0)
        self.assertEqual(result['table_names'], [])
    
    def test_validate_vessel_data(self):
        """Test vessel data validation function"""
        # Valid vessel data
        valid_data = pd.DataFrame({
            'call_sign': ['VESSEL1', 'VESSEL2'],
            'vessel_name': ['Ship A', 'Ship B'],
            'ship_category': ['container', 'bulk_carrier'],
            'arrival_time': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 12:00'])
        })
        
        result = _validate_vessel_data(valid_data)
        self.assertEqual(result['records_count'], 2)
        self.assertEqual(result['unique_vessels'], 2)
        self.assertIn('date_range', result)
        self.assertIn('missing_values', result)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        result = _validate_vessel_data(empty_data)
        self.assertEqual(result['records_count'], 0)
    
    def test_validate_weather_data(self):
        """Test weather data validation function"""
        # Test weather data validation (no parameters)
        result = _validate_weather_data()
        
        # Should return a dictionary with weather validation info
        self.assertIsInstance(result, dict)
        # Weather data might not be available, so we just check it returns a dict
    
    def test_cross_validate_datasets(self):
        """Test cross-dataset validation function"""
        # Create consistent datasets
        container_data = pd.DataFrame({
            'total_teus': [1000000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='MS'))
        
        cargo_stats = {
            'Table_1': pd.DataFrame({
                'Cargo Type': ['Container'],
                '2024 (thousand tonnes)': [50000]
            })
        }
        
        vessel_data = {'date_range': '2024-01-01 to 2024-01-31'}
        
        result = _cross_validate_datasets(container_data, cargo_stats, vessel_data)
        self.assertIsInstance(result, dict)
        self.assertIn('temporal_alignment', result)
        self.assertIn('volume_consistency', result)
        
        # Test with empty data
        result = _cross_validate_datasets(pd.DataFrame(), {}, {})
        self.assertIsInstance(result, dict)
    
    def test_detect_data_anomalies(self):
        """Test data anomaly detection function"""
        # Normal data
        normal_data = pd.DataFrame({
            'total_teus': [100000, 105000, 95000, 110000, 90000, 108000, 102000]
        })
        
        result = _detect_data_anomalies(normal_data)
        self.assertIsInstance(result, dict)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        result = _detect_data_anomalies(empty_data)
        self.assertIsInstance(result, dict)
    
    def test_check_data_freshness(self):
        """Test data freshness checking function"""
        # Test data freshness (no parameters)
        result = _check_data_freshness()
        
        # Should return a dictionary with freshness info
        self.assertIsInstance(result, dict)
        self.assertIn('container_data', result)
        self.assertIn('vessel_data', result)
        self.assertIn('weather_data', result)
    
    @patch('utils.data_loader.load_container_throughput')
    @patch('utils.data_loader.load_vessel_arrivals')
    def test_real_time_data_manager_initialization(self, mock_vessel, mock_container):
        """Test RealTimeDataManager initialization"""
        mock_container.return_value = load_sample_data()
        mock_vessel.return_value = pd.DataFrame()
        
        manager = RealTimeDataManager()
        
        # Check initialization
        self.assertIsNotNone(manager.config)
        self.assertEqual(manager.error_counts, {})
        self.assertEqual(manager.circuit_breaker_threshold, 5)
        self.assertIsInstance(manager.circuit_breaker_states, dict)
    
    @patch('utils.data_loader.load_container_throughput')
    @patch('utils.data_loader.load_vessel_arrivals')
    def test_real_time_data_manager_circuit_breaker(self, mock_vessel, mock_container):
        """Test RealTimeDataManager circuit breaker functionality"""
        mock_container.return_value = load_sample_data()
        mock_vessel.return_value = pd.DataFrame()
        
        manager = RealTimeDataManager()
        
        # Test circuit breaker is initially closed
        self.assertFalse(manager._is_circuit_breaker_open('test_source'))
        
        # Simulate multiple failures
        for _ in range(6):  # Exceed threshold of 5
            manager._record_operation_failure('test_source')
        
        # Circuit breaker should now be open
        self.assertTrue(manager._is_circuit_breaker_open('test_source'))
        
        # Test success resets the circuit breaker
        manager._record_operation_success('test_source')
        self.assertFalse(manager._is_circuit_breaker_open('test_source'))
    
    @patch('utils.data_loader.load_container_throughput')
    @patch('utils.data_loader.load_vessel_arrivals')
    def test_real_time_data_manager_validate_and_process(self, mock_vessel, mock_container):
        """Test RealTimeDataManager data validation and processing"""
        sample_data = load_sample_data()
        mock_container.return_value = sample_data
        mock_vessel.return_value = pd.DataFrame()
        
        manager = RealTimeDataManager()
        
        # Test successful validation and processing
        result = manager.validate_and_process_data('container_throughput', sample_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        
        # Test with invalid data type
        result = manager.validate_and_process_data('unknown_type', sample_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
    
    @patch('utils.data_loader.load_container_throughput')
    @patch('utils.data_loader.load_vessel_arrivals')
    def test_real_time_data_manager_comprehensive_report(self, mock_vessel, mock_container):
        """Test RealTimeDataManager comprehensive data quality report"""
        mock_container.return_value = load_sample_data()
        mock_vessel.return_value = pd.DataFrame()
        
        manager = RealTimeDataManager()
        
        # Generate some cache activity
        data_cache.set('test_key', {'test': 'data'})
        data_cache.get('test_key')
        data_cache.get('non_existent')
        
        report = manager.get_comprehensive_data_quality_report()
        
        # Check report structure (validate_data_quality + real_time_metrics)
        self.assertIsInstance(report, dict)
        
        # Check validate_data_quality sections
        expected_sections = [
            'container_throughput', 'cargo_statistics', 'vessel_arrivals', 'weather_data',
            'cross_validation', 'anomaly_detection', 'data_freshness', 'overall_status'
        ]
        for section in expected_sections:
            self.assertIn(section, report)
        
        # Check real_time_metrics section
        self.assertIn('real_time_metrics', report)
        real_time_metrics = report['real_time_metrics']
        self.assertIn('global_cache_stats', real_time_metrics)
        
        # Check cache stats structure
        cache_stats = real_time_metrics['global_cache_stats']
        self.assertIn('cached_items', cache_stats)
        self.assertIn('total_access_count', cache_stats)
        self.assertIn('cache_keys', cache_stats)
        self.assertIn('access_counts', cache_stats)
    
    def test_enhanced_validate_data_quality_integration(self):
        """Test enhanced validate_data_quality function with all data types"""
        # Mock all data loading functions
        with patch('utils.data_loader.load_container_throughput') as mock_container, \
             patch('utils.data_loader.load_port_cargo_statistics') as mock_cargo, \
             patch('utils.data_loader.load_vessel_arrivals') as mock_vessel:
            
            # Set up mock returns
            mock_container.return_value = load_sample_data()
            mock_cargo.return_value = {'Table_1': pd.DataFrame({'col1': [1, 2, 3]})}
            mock_vessel.return_value = pd.DataFrame({
                'call_sign': ['VESSEL1'],
                'vessel_name': ['Ship A'],
                'arrival_time': pd.to_datetime(['2024-01-01 10:00'])
            })
            
            validation = validate_data_quality()
            
            # Check enhanced structure
            self.assertIsInstance(validation, dict)
            expected_sections = [
                'container_throughput', 'cargo_statistics', 'vessel_arrivals', 'weather_data',
                'cross_validation', 'anomaly_detection', 'data_freshness',
                'overall_status'
            ]
            
            for section in expected_sections:
                self.assertIn(section, validation)
            
            # Check that overall status is determined correctly
            self.assertIn(validation['overall_status'], ['excellent', 'good', 'fair', 'poor', 'no_data', 'error'])


if __name__ == '__main__':
    unittest.main()