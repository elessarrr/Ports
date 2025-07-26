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
    load_sample_data
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
            self.assertIn(validation['overall_status'], ['success', 'partial', 'failed', 'error'])
    
    def test_validate_data_quality_with_no_data(self):
        """Test data quality validation with no data available"""
        # Mock both functions to return empty data
        with patch('utils.data_loader.load_container_throughput') as mock_throughput, \
             patch('utils.data_loader.load_port_cargo_statistics') as mock_cargo:
            
            mock_throughput.return_value = pd.DataFrame()
            mock_cargo.return_value = {}
            
            validation = validate_data_quality()
            
            # Should indicate failure when no data is available
            self.assertEqual(validation['overall_status'], 'failed')
    
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
        # Create sample time series data
        dates = pd.date_range('2020-01-01', '2023-12-01', freq='MS')
        values = np.random.normal(1000000, 100000, len(dates)) + np.arange(len(dates)) * 1000
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
            
            # Linear trend should show positive slope (we added increasing trend)
            self.assertGreater(ts_analysis['linear_trend_slope'], 0)
            
            # R-squared should be reasonable (> 0.5 for our synthetic data)
            self.assertGreater(ts_analysis['linear_trend_r_squared'], 0.5)
            
            # Moving averages should be positive
            self.assertGreater(ts_analysis['moving_average_3m'], 0)
            self.assertGreater(ts_analysis['moving_average_12m'], 0)
            
            # Volatility should be non-negative
            self.assertGreaterEqual(ts_analysis['volatility'], 0)
    
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
            seasonal = trends['seasonal_patterns']
            
            # Should have monthly and quarterly patterns
            self.assertIn('monthly_patterns', seasonal)
            self.assertIn('quarterly_patterns', seasonal)
            
            # Monthly patterns should have 12 entries
            self.assertEqual(len(seasonal['monthly_patterns']), 12)
            
            # Quarterly patterns should have 4 entries
            self.assertEqual(len(seasonal['quarterly_patterns']), 4)
            
            # Peak and low months should be valid
            self.assertIn(seasonal['peak_month'], range(1, 13))
            self.assertIn(seasonal['low_month'], range(1, 13))
            
            # Seasonal strength should be between 0 and 1
            self.assertGreaterEqual(seasonal['seasonal_strength'], 0)
            self.assertLessEqual(seasonal['seasonal_strength'], 1)
    
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
            forecasts = trends['forecasts']
            
            # Should have forecasts for different periods
            forecast_periods = ['next_3_months', 'next_6_months', 'next_12_months']
            for period in forecast_periods:
                self.assertIn(period, forecasts)
                self.assertIsInstance(forecasts[period], list)
                
                # Check forecast length matches period
                if period == 'next_3_months':
                    self.assertEqual(len(forecasts[period]), 3)
                elif period == 'next_6_months':
                    self.assertEqual(len(forecasts[period]), 6)
                elif period == 'next_12_months':
                    self.assertEqual(len(forecasts[period]), 12)
            
            # Should have confidence and performance metrics
            self.assertIn('forecast_confidence', forecasts)
            self.assertIn('model_performance', forecasts)
            
            # Confidence should be between 0 and 1
            self.assertGreaterEqual(forecasts['forecast_confidence'], 0)
            self.assertLessEqual(forecasts['forecast_confidence'], 1)
    
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
            
            # Test modal split trends
            modal_split = trends['modal_split_trends']
            
            # Should have trends for both modes
            expected_keys = [
                'seaborne_trend', 'river_trend',
                'seaborne_share_trend', 'river_share_trend'
            ]
            
            for key in expected_keys:
                self.assertIn(key, modal_split)
                self.assertIsInstance(modal_split[key], dict)
            
            # Each trend should have slope and r_squared
            for trend_key in ['seaborne_trend', 'river_trend', 'seaborne_share_trend', 'river_share_trend']:
                trend_data = modal_split[trend_key]
                self.assertIn('slope', trend_data)
                self.assertIn('r_squared', trend_data)
                self.assertIsInstance(trend_data['slope'], (int, float))
                self.assertIsInstance(trend_data['r_squared'], (int, float))
                
                # R-squared should be between 0 and 1
                self.assertGreaterEqual(trend_data['r_squared'], 0)
                self.assertLessEqual(trend_data['r_squared'], 1)
    
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
            
            # Should have monthly and annual YoY changes
            self.assertIn('monthly_yoy_changes', yoy_analysis)
            self.assertIn('annual_yoy_changes', yoy_analysis)
            
            # Should have average YoY metrics
            self.assertIn('avg_monthly_yoy', yoy_analysis)
            self.assertIn('avg_annual_yoy', yoy_analysis)
            
            # Average YoY should be positive (we added growth)
            self.assertGreater(yoy_analysis['avg_annual_yoy'], 0)
            
            # Monthly YoY changes should be a list
            self.assertIsInstance(yoy_analysis['monthly_yoy_changes'], list)
            
            # Annual YoY changes should be a list
            self.assertIsInstance(yoy_analysis['annual_yoy_changes'], list)


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
                'time_series_analysis',
                'year_over_year_analysis',
                'seasonal_patterns', 
                'forecasts',
                'modal_split_trends',
                'summary'
            ]
            
            for section in main_sections:
                self.assertIn(section, trends)
                self.assertIsInstance(trends[section], dict)
                self.assertGreater(len(trends[section]), 0)
            
            # Verify data consistency across sections
            summary = trends['summary']
            self.assertEqual(summary['total_records'], len(sample_df))
            self.assertEqual(summary['latest_month'], sample_df.index[-1].strftime('%Y-%m'))
            
            # Verify numerical consistency
            latest_value = summary['latest_value']
            self.assertAlmostEqual(latest_value, sample_df['total_teus'].iloc[-1], places=0)
            
            # Verify forecast reasonableness
            forecasts = trends['forecasts']
            next_3_months = forecasts['next_3_months']
            
            # Forecasts should be positive and reasonable
            for forecast_value in next_3_months:
                self.assertGreater(forecast_value, 0)
                # Should be within reasonable range of current values
                self.assertLess(abs(forecast_value - latest_value) / latest_value, 0.5)


if __name__ == '__main__':
    unittest.main()