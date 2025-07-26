"""Data loading utilities for Hong Kong Port Digital Twin.

This module provides functions to load and process various data sources including:
- Container throughput time series data
- Port cargo statistics
- Vessel arrival data
- Port berth configurations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data file paths
RAW_DATA_DIR = (Path(__file__).parent.parent.parent / ".." / "raw_data").resolve()
CONTAINER_THROUGHPUT_FILE = RAW_DATA_DIR / "Total_container_throughput_by_mode_of_transport_(EN).csv"
PORT_CARGO_STATS_DIR = RAW_DATA_DIR / "Port Cargo Statistics"
VESSEL_ARRIVALS_XML = RAW_DATA_DIR / "Arrived_in_last_36_hours.xml"

def load_container_throughput() -> pd.DataFrame:
    """Load and process container throughput time series data.
    
    Returns:
        pd.DataFrame: Processed container throughput data with datetime index
    """
    try:
        # Load the CSV file
        df = pd.read_csv(CONTAINER_THROUGHPUT_FILE)
        
        # Clean and process the data
        df = df.copy()
        
        # Handle missing values in numeric columns
        numeric_cols = ['Seaborne ( \'000 TEUs)', 'River ( \'000 TEUs)', 'Total ( \'000 TEUs)']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create proper datetime index for monthly data
        monthly_data = df[df['Month'] != 'All'].copy()
        
        # Create datetime from Year and Month
        monthly_data['Date'] = pd.to_datetime(
            monthly_data['Year'].astype(str) + '-' + monthly_data['Month'], 
            format='%Y-%b', 
            errors='coerce'
        )
        
        # Filter out rows with invalid dates
        monthly_data = monthly_data.dropna(subset=['Date'])
        
        # Set datetime as index
        monthly_data = monthly_data.set_index('Date').sort_index()
        
        # Rename columns for easier access
        monthly_data = monthly_data.rename(columns={
            'Seaborne ( \'000 TEUs)': 'seaborne_teus',
            'River ( \'000 TEUs)': 'river_teus', 
            'Total ( \'000 TEUs)': 'total_teus',
            'Seaborne (Year-on-year change %)': 'seaborne_yoy_change',
            'River (Year-on-year change %)': 'river_yoy_change',
            'Total (Year-on-year change %)': 'total_yoy_change'
        })
        
        logger.info(f"Loaded container throughput data: {len(monthly_data)} monthly records")
        return monthly_data
        
    except Exception as e:
        logger.error(f"Error loading container throughput data: {e}")
        return pd.DataFrame()

def load_annual_container_throughput() -> pd.DataFrame:
    """Load annual container throughput summary data.
    
    Returns:
        pd.DataFrame: Annual throughput data
    """
    try:
        df = pd.read_csv(CONTAINER_THROUGHPUT_FILE)
        
        # Filter for annual data (Month == 'All')
        annual_data = df[df['Month'] == 'All'].copy()
        
        # Clean numeric columns
        numeric_cols = ['Seaborne ( \'000 TEUs)', 'River ( \'000 TEUs)', 'Total ( \'000 TEUs)']
        for col in numeric_cols:
            annual_data[col] = pd.to_numeric(annual_data[col], errors='coerce')
        
        # Rename columns
        annual_data = annual_data.rename(columns={
            'Seaborne ( \'000 TEUs)': 'seaborne_teus',
            'River ( \'000 TEUs)': 'river_teus',
            'Total ( \'000 TEUs)': 'total_teus',
            'Seaborne (Year-on-year change %)': 'seaborne_yoy_change',
            'River (Year-on-year change %)': 'river_yoy_change',
            'Total (Year-on-year change %)': 'total_yoy_change'
        })
        
        logger.info(f"Loaded annual container throughput data: {len(annual_data)} years")
        return annual_data
        
    except Exception as e:
        logger.error(f"Error loading annual container throughput data: {e}")
        return pd.DataFrame()

def load_port_cargo_statistics() -> Dict[str, pd.DataFrame]:
    """Load port cargo statistics from multiple CSV files.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of cargo statistics by table
    """
    cargo_stats = {}
    
    try:
        # Get all CSV files in the Port Cargo Statistics directory
        csv_files = list(PORT_CARGO_STATS_DIR.glob("*.CSV"))
        
        for csv_file in csv_files:
            table_name = csv_file.stem.replace("Port Cargo Statistics_CSV_Eng-", "")
            
            try:
                df = pd.read_csv(csv_file)
                
                # Clean and validate the data
                df = _clean_cargo_statistics_data(df, table_name)
                
                cargo_stats[table_name] = df
                logger.info(f"Loaded {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                
            except Exception as e:
                logger.error(f"Error loading {csv_file.name}: {e}")
                continue
        
        logger.info(f"Loaded {len(cargo_stats)} cargo statistics tables")
        return cargo_stats
        
    except Exception as e:
        logger.error(f"Error loading port cargo statistics: {e}")
        return {}

def _clean_cargo_statistics_data(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Clean and validate cargo statistics data.
    
    Args:
        df: Raw DataFrame from CSV
        table_name: Name of the table for context
        
    Returns:
        pd.DataFrame: Cleaned and validated DataFrame
    """
    try:
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Convert numeric columns (those containing year data)
        numeric_columns = [col for col in cleaned_df.columns if any(year in col for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'])]
        
        for col in numeric_columns:
            # Handle special characters and convert to numeric
            if cleaned_df[col].dtype == 'object':
                # Replace common non-numeric indicators
                cleaned_df[col] = cleaned_df[col].astype(str).replace({
                    '-': np.nan,
                    '§': '0',  # Less than 0.05% indicator
                    'N/A': np.nan,
                    '': np.nan
                })
                
                # Convert to numeric, coercing errors to NaN
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Log data quality metrics
        total_cells = cleaned_df.size
        missing_cells = cleaned_df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        logger.info(f"{table_name} data quality: {completeness:.1f}% complete")
        
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error cleaning {table_name} data: {e}")
        return df

def get_cargo_breakdown_analysis() -> Dict[str, any]:
    """Analyze cargo breakdown by type, shipment mode, and location.
    
    Returns:
        Dict: Comprehensive cargo analysis including efficiency metrics
    """
    try:
        cargo_stats = load_port_cargo_statistics()
        
        if not cargo_stats:
            logger.warning("No cargo statistics data available for analysis")
            return {}
        
        analysis = {
            'shipment_type_analysis': {},
            'transport_mode_analysis': {},
            'cargo_type_analysis': {},
            'location_analysis': {},
            'efficiency_metrics': {},
            'data_summary': {
                'tables_processed': len(cargo_stats),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        # Analyze shipment types (Table 1: Direct vs Transhipment)
        if 'Table_1_Eng' in cargo_stats:
            shipment_df = cargo_stats['Table_1_Eng']
            analysis['shipment_type_analysis'] = _analyze_shipment_types(shipment_df)
        
        # Analyze transport modes (Table 2: Seaborne vs River)
        if 'Table_2_Eng' in cargo_stats:
            transport_df = cargo_stats['Table_2_Eng']
            analysis['transport_mode_analysis'] = _analyze_transport_modes(transport_df)
        
        # Analyze cargo types (Table 6)
        if 'Table_6_Eng' in cargo_stats:
            cargo_type_df = cargo_stats['Table_6_Eng']
            analysis['cargo_type_analysis'] = _analyze_cargo_types(cargo_type_df)
        
        # Analyze handling locations (Table 7)
        if 'Table_7_Eng' in cargo_stats:
            location_df = cargo_stats['Table_7_Eng']
            analysis['location_analysis'] = _analyze_handling_locations(location_df)
        
        # Calculate efficiency metrics
        analysis['efficiency_metrics'] = _calculate_efficiency_metrics(cargo_stats)
        
        logger.info("Completed comprehensive cargo breakdown analysis")
        return analysis
        
    except Exception as e:
        logger.error(f"Error in cargo breakdown analysis: {e}")
        return {}

def _analyze_shipment_types(df: pd.DataFrame) -> Dict[str, any]:
    """Analyze direct shipment vs transhipment cargo patterns."""
    try:
        # Get latest year data (2023)
        latest_cols = [col for col in df.columns if '2023' in col and 'percentage' not in col.lower()]
        
        if not latest_cols:
            return {}
        
        # Extract shipment type data
        direct_row = df[df.iloc[:, 0].str.contains('Direct', case=False, na=False)]
        tranship_row = df[df.iloc[:, 0].str.contains('Transhipment', case=False, na=False)]
        
        analysis = {}
        
        if not direct_row.empty and not tranship_row.empty:
            direct_value = direct_row[latest_cols[0]].iloc[0] if len(latest_cols) > 0 else 0
            tranship_value = tranship_row[latest_cols[0]].iloc[0] if len(latest_cols) > 0 else 0
            total_value = direct_value + tranship_value
            
            analysis = {
                'direct_shipment_2023': float(direct_value) if pd.notna(direct_value) else 0,
                'transhipment_2023': float(tranship_value) if pd.notna(tranship_value) else 0,
                'total_2023': float(total_value) if pd.notna(total_value) else 0,
                'direct_percentage': (float(direct_value) / float(total_value) * 100) if total_value > 0 else 0,
                'transhipment_percentage': (float(tranship_value) / float(total_value) * 100) if total_value > 0 else 0
            }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing shipment types: {e}")
        return {}

def _analyze_transport_modes(df: pd.DataFrame) -> Dict[str, any]:
    """Analyze seaborne vs river transport patterns."""
    try:
        # Get latest year data (2023)
        latest_cols = [col for col in df.columns if '2023' in col and 'percentage' not in col.lower()]
        
        if not latest_cols:
            return {}
        
        # Extract transport mode data
        seaborne_row = df[df.iloc[:, 0].str.contains('Seaborne', case=False, na=False)]
        river_row = df[df.iloc[:, 0].str.contains('River', case=False, na=False)]
        
        analysis = {}
        
        if not seaborne_row.empty and not river_row.empty:
            seaborne_value = seaborne_row[latest_cols[0]].iloc[0] if len(latest_cols) > 0 else 0
            river_value = river_row[latest_cols[0]].iloc[0] if len(latest_cols) > 0 else 0
            total_value = seaborne_value + river_value
            
            analysis = {
                'seaborne_2023': float(seaborne_value) if pd.notna(seaborne_value) else 0,
                'river_2023': float(river_value) if pd.notna(river_value) else 0,
                'total_2023': float(total_value) if pd.notna(total_value) else 0,
                'seaborne_percentage': (float(seaborne_value) / float(total_value) * 100) if total_value > 0 else 0,
                'river_percentage': (float(river_value) / float(total_value) * 100) if total_value > 0 else 0
            }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing transport modes: {e}")
        return {}

def _analyze_cargo_types(df: pd.DataFrame) -> Dict[str, any]:
    """Analyze different cargo types and their throughput."""
    try:
        # Get 2023 overall cargo data
        overall_cols = [col for col in df.columns if '2023' in col and 'overall' in col.lower()]
        
        if not overall_cols:
            return {}
        
        # Extract cargo types (first column typically contains cargo type names)
        cargo_types = df.iloc[:, 0].dropna().tolist()
        cargo_data = []
        
        for i, cargo_type in enumerate(cargo_types):
            if i < len(df) and pd.notna(df.iloc[i][overall_cols[0]]):
                cargo_data.append({
                    'cargo_type': cargo_type,
                    'throughput_2023': float(df.iloc[i][overall_cols[0]])
                })
        
        # Sort by throughput
        cargo_data.sort(key=lambda x: x['throughput_2023'], reverse=True)
        
        analysis = {
            'top_cargo_types': cargo_data[:5],  # Top 5 cargo types
            'total_cargo_types': len(cargo_data),
            'total_throughput': sum(item['throughput_2023'] for item in cargo_data)
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing cargo types: {e}")
        return {}

def _analyze_handling_locations(df: pd.DataFrame) -> Dict[str, any]:
    """Analyze cargo handling by different port locations."""
    try:
        # Get 2023 overall cargo data
        overall_cols = [col for col in df.columns if '2023' in col and 'overall' in col.lower()]
        
        if not overall_cols:
            return {}
        
        # Extract location data
        locations = df.iloc[:, 0].dropna().tolist()
        location_data = []
        
        for i, location in enumerate(locations):
            if i < len(df) and pd.notna(df.iloc[i][overall_cols[0]]):
                location_data.append({
                    'location': location,
                    'throughput_2023': float(df.iloc[i][overall_cols[0]])
                })
        
        # Sort by throughput
        location_data.sort(key=lambda x: x['throughput_2023'], reverse=True)
        
        analysis = {
            'top_locations': location_data[:5],  # Top 5 locations
            'total_locations': len(location_data),
            'total_throughput': sum(item['throughput_2023'] for item in location_data)
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing handling locations: {e}")
        return {}

def _calculate_efficiency_metrics(cargo_stats: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """Calculate port efficiency metrics from cargo statistics."""
    try:
        metrics = {}
        
        # Calculate transhipment ratio (efficiency indicator)
        if 'Table_1_Eng' in cargo_stats:
            shipment_df = cargo_stats['Table_1_Eng']
            tranship_analysis = _analyze_shipment_types(shipment_df)
            
            if tranship_analysis:
                metrics['transhipment_ratio'] = tranship_analysis.get('transhipment_percentage', 0)
                metrics['direct_shipment_ratio'] = tranship_analysis.get('direct_percentage', 0)
        
        # Calculate modal split efficiency
        if 'Table_2_Eng' in cargo_stats:
            transport_df = cargo_stats['Table_2_Eng']
            transport_analysis = _analyze_transport_modes(transport_df)
            
            if transport_analysis:
                metrics['seaborne_ratio'] = transport_analysis.get('seaborne_percentage', 0)
                metrics['river_ratio'] = transport_analysis.get('river_percentage', 0)
        
        # Calculate cargo diversity index (number of cargo types handled)
        if 'Table_6_Eng' in cargo_stats:
            cargo_df = cargo_stats['Table_6_Eng']
            metrics['cargo_diversity_index'] = len(cargo_df)
        
        # Calculate location utilization efficiency
        if 'Table_7_Eng' in cargo_stats:
            location_df = cargo_stats['Table_7_Eng']
            metrics['location_utilization_index'] = len(location_df)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating efficiency metrics: {e}")
        return {}

def get_throughput_trends() -> Dict[str, any]:
    """Comprehensive container throughput trend analysis with forecasting.
    
    Implements Priority 1C requirements:
    - Time series analysis for container throughput data
    - Year-over-year comparison visualizations
    - Seasonal pattern recognition for peak/off-peak periods
    - Basic forecasting models using historical trends
    
    Returns:
        Dict: Comprehensive analysis including trends, seasonality, and forecasts
    """
    monthly_data = load_container_throughput()
    
    if monthly_data.empty:
        return {}
    
    try:
        total_teus = monthly_data['total_teus'].dropna()
        seaborne_teus = monthly_data['seaborne_teus'].dropna()
        river_teus = monthly_data['river_teus'].dropna()
        
        # Basic statistics
        basic_stats = {
            'latest_month': total_teus.index[-1].strftime('%Y-%m') if len(total_teus) > 0 else None,
            'latest_value': float(total_teus.iloc[-1]) if len(total_teus) > 0 else None,
            'mean_monthly': float(total_teus.mean()),
            'std_monthly': float(total_teus.std()),
            'min_value': float(total_teus.min()),
            'max_value': float(total_teus.max()),
            'total_records': len(total_teus),
            'date_range': {
                'start': total_teus.index[0].strftime('%Y-%m'),
                'end': total_teus.index[-1].strftime('%Y-%m')
            }
        }
        
        # Time series analysis with trend detection
        time_series_analysis = _analyze_time_series_trends(total_teus)
        
        # Year-over-year comparison analysis
        yoy_analysis = _analyze_year_over_year_changes(monthly_data)
        
        # Comprehensive seasonal pattern recognition
        seasonal_analysis = _analyze_seasonal_patterns(monthly_data)
        
        # Basic forecasting models
        forecasting_results = _generate_forecasts(total_teus, seaborne_teus, river_teus)
        
        # Modal split analysis
        modal_analysis = _analyze_modal_split_trends(monthly_data)
        
        # Combine all analyses
        comprehensive_trends = {
            'basic_statistics': basic_stats,
            'time_series_analysis': time_series_analysis,
            'year_over_year_analysis': yoy_analysis,
            'seasonal_analysis': seasonal_analysis,
            'forecasting': forecasting_results,
            'modal_split_analysis': modal_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Generated comprehensive throughput trend analysis")
        return comprehensive_trends
        
    except Exception as e:
        logger.error(f"Error analyzing throughput trends: {e}")
        return {}

def _analyze_time_series_trends(data: pd.Series) -> Dict[str, any]:
    """Analyze time series trends using statistical methods."""
    try:
        # Convert index to numeric for trend analysis
        x = np.arange(len(data))
        y = data.values
        
        # Linear trend analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate trend direction and strength
        trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        trend_strength = abs(r_value)
        
        # Moving averages for smoothing
        ma_3 = data.rolling(window=3).mean()
        ma_12 = data.rolling(window=12).mean()
        
        # Volatility analysis
        returns = data.pct_change().dropna()
        volatility = returns.std() * np.sqrt(12)  # Annualized volatility
        
        return {
            'linear_trend': {
                'slope': float(slope),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'direction': trend_direction,
                'strength': float(trend_strength)
            },
            'moving_averages': {
                'ma_3_latest': float(ma_3.iloc[-1]) if not ma_3.empty else None,
                'ma_12_latest': float(ma_12.iloc[-1]) if not ma_12.empty else None
            },
            'volatility': {
                'monthly_std': float(returns.std()),
                'annualized_volatility': float(volatility),
                'coefficient_of_variation': float(data.std() / data.mean())
            }
        }
    except Exception as e:
        logger.error(f"Error in time series trend analysis: {e}")
        return {}

def _analyze_year_over_year_changes(data: pd.DataFrame) -> Dict[str, any]:
    """Analyze year-over-year changes and growth patterns."""
    try:
        # Calculate YoY changes for each metric
        yoy_changes = {}
        
        for metric in ['total_teus', 'seaborne_teus', 'river_teus']:
            if metric in data.columns:
                series = data[metric].dropna()
                yoy_change = series.pct_change(periods=12) * 100  # 12-month change
                
                yoy_changes[metric] = {
                    'latest_yoy_change': float(yoy_change.iloc[-1]) if not yoy_change.empty else None,
                    'avg_yoy_change': float(yoy_change.mean()),
                    'max_yoy_change': float(yoy_change.max()),
                    'min_yoy_change': float(yoy_change.min()),
                    'yoy_volatility': float(yoy_change.std())
                }
        
        # Annual growth analysis
        annual_data = data.groupby(data.index.year).agg({
            'total_teus': 'sum',
            'seaborne_teus': 'sum', 
            'river_teus': 'sum'
        })
        
        annual_growth = annual_data.pct_change() * 100
        
        return {
            'monthly_yoy_changes': yoy_changes,
            'annual_growth': {
                'avg_annual_growth': float(annual_growth['total_teus'].mean()) if 'total_teus' in annual_growth else None,
                'latest_annual_growth': float(annual_growth['total_teus'].iloc[-1]) if len(annual_growth) > 0 else None,
                'growth_consistency': float(1 - (annual_growth['total_teus'].std() / abs(annual_growth['total_teus'].mean()))) if 'total_teus' in annual_growth else None
            }
        }
    except Exception as e:
        logger.error(f"Error in YoY analysis: {e}")
        return {}

def _analyze_seasonal_patterns(data: pd.DataFrame) -> Dict[str, any]:
    """Comprehensive seasonal pattern recognition."""
    try:
        # Monthly seasonality
        monthly_patterns = data.groupby(data.index.month).agg({
            'total_teus': ['mean', 'std', 'count'],
            'seaborne_teus': ['mean', 'std'],
            'river_teus': ['mean', 'std']
        })
        
        # Quarterly patterns
        quarterly_patterns = data.groupby(data.index.quarter).agg({
            'total_teus': ['mean', 'std'],
            'seaborne_teus': ['mean', 'std'],
            'river_teus': ['mean', 'std']
        })
        
        # Identify peak and low periods
        monthly_avg = monthly_patterns[('total_teus', 'mean')]
        peak_month = int(monthly_avg.idxmax())
        low_month = int(monthly_avg.idxmin())
        
        quarterly_avg = quarterly_patterns[('total_teus', 'mean')]
        peak_quarter = int(quarterly_avg.idxmax())
        low_quarter = int(quarterly_avg.idxmin())
        
        # Calculate seasonality strength
        seasonal_coefficient = monthly_avg.std() / monthly_avg.mean()
        
        # Month names for readability
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
        
        return {
            'monthly_patterns': {
                'peak_month': {'number': peak_month, 'name': month_names[peak_month-1]},
                'low_month': {'number': low_month, 'name': month_names[low_month-1]},
                'peak_value': float(monthly_avg.max()),
                'low_value': float(monthly_avg.min()),
                'seasonal_range': float(monthly_avg.max() - monthly_avg.min()),
                'seasonality_strength': float(seasonal_coefficient)
            },
            'quarterly_patterns': {
                'peak_quarter': {'number': peak_quarter, 'name': quarter_names[peak_quarter-1]},
                'low_quarter': {'number': low_quarter, 'name': quarter_names[low_quarter-1]},
                'peak_value': float(quarterly_avg.max()),
                'low_value': float(quarterly_avg.min())
            },
            'seasonal_insights': {
                'is_highly_seasonal': seasonal_coefficient > 0.1,
                'seasonal_classification': 'High' if seasonal_coefficient > 0.15 else 'Moderate' if seasonal_coefficient > 0.05 else 'Low'
            }
        }
    except Exception as e:
        logger.error(f"Error in seasonal analysis: {e}")
        return {}

def _generate_forecasts(total_teus: pd.Series, seaborne_teus: pd.Series, river_teus: pd.Series) -> Dict[str, any]:
    """Generate basic forecasting models using historical trends."""
    try:
        forecasts = {}
        
        for name, series in [('total', total_teus), ('seaborne', seaborne_teus), ('river', river_teus)]:
            if len(series) < 12:  # Need at least 12 months for meaningful forecast
                continue
                
            # Prepare data for forecasting
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            
            # Linear regression forecast
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate 6-month forecast
            future_X = np.arange(len(series), len(series) + 6).reshape(-1, 1)
            forecast_values = model.predict(future_X)
            
            # Calculate model performance on historical data
            historical_pred = model.predict(X)
            mae = mean_absolute_error(y, historical_pred)
            rmse = np.sqrt(mean_squared_error(y, historical_pred))
            
            # Seasonal adjustment (simple)
            seasonal_factors = _calculate_seasonal_factors(series)
            adjusted_forecast = []
            
            for i, base_forecast in enumerate(forecast_values):
                month_ahead = (series.index[-1].month + i) % 12 + 1
                seasonal_factor = seasonal_factors.get(month_ahead, 1.0)
                adjusted_forecast.append(base_forecast * seasonal_factor)
            
            forecasts[f'{name}_forecast'] = {
                'method': 'linear_regression_with_seasonal_adjustment',
                'forecast_horizon': '6_months',
                'forecast_values': [float(x) for x in adjusted_forecast],
                'model_performance': {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r_squared': float(model.score(X, y))
                },
                'confidence_level': 'basic',
                'notes': 'Linear trend with seasonal adjustment'
            }
        
        return forecasts
        
    except Exception as e:
        logger.error(f"Error generating forecasts: {e}")
        return {}

def _calculate_seasonal_factors(series: pd.Series) -> Dict[int, float]:
    """Calculate seasonal adjustment factors by month."""
    try:
        # Group by month and calculate average
        monthly_avg = series.groupby(series.index.month).mean()
        overall_avg = series.mean()
        
        # Calculate seasonal factors (ratio to overall average)
        seasonal_factors = {}
        for month in range(1, 13):
            if month in monthly_avg.index:
                seasonal_factors[month] = monthly_avg[month] / overall_avg
            else:
                seasonal_factors[month] = 1.0
        
        return seasonal_factors
    except Exception as e:
        logger.error(f"Error calculating seasonal factors: {e}")
        return {i: 1.0 for i in range(1, 13)}

def _analyze_modal_split_trends(data: pd.DataFrame) -> Dict[str, any]:
    """Analyze trends in seaborne vs river transport modes."""
    try:
        # Calculate modal split percentages
        data_copy = data.copy()
        data_copy['seaborne_pct'] = (data_copy['seaborne_teus'] / data_copy['total_teus']) * 100
        data_copy['river_pct'] = (data_copy['river_teus'] / data_copy['total_teus']) * 100
        
        # Trend analysis for modal split
        seaborne_trend = _analyze_time_series_trends(data_copy['seaborne_pct'].dropna())
        river_trend = _analyze_time_series_trends(data_copy['river_pct'].dropna())
        
        return {
            'current_modal_split': {
                'seaborne_percentage': float(data_copy['seaborne_pct'].iloc[-1]) if not data_copy['seaborne_pct'].empty else None,
                'river_percentage': float(data_copy['river_pct'].iloc[-1]) if not data_copy['river_pct'].empty else None
            },
            'historical_average': {
                'seaborne_percentage': float(data_copy['seaborne_pct'].mean()),
                'river_percentage': float(data_copy['river_pct'].mean())
            },
            'modal_split_trends': {
                'seaborne_trend': seaborne_trend.get('linear_trend', {}),
                'river_trend': river_trend.get('linear_trend', {})
            }
        }
    except Exception as e:
        logger.error(f"Error in modal split analysis: {e}")
        return {}

def validate_data_quality() -> Dict[str, any]:
    """Validate data quality across all loaded datasets.
    
    Returns:
        Dict: Data quality metrics and validation results
    """
    validation_results = {
        'container_throughput': {},
        'cargo_statistics': {},
        'overall_status': 'unknown'
    }
    
    try:
        # Validate container throughput data
        monthly_data = load_container_throughput()
        if not monthly_data.empty:
            validation_results['container_throughput'] = {
                'records_count': len(monthly_data),
                'date_range': f"{monthly_data.index.min()} to {monthly_data.index.max()}",
                'missing_values': monthly_data.isnull().sum().to_dict(),
                'data_completeness': (1 - monthly_data.isnull().sum().sum() / monthly_data.size) * 100
            }
        
        # Validate cargo statistics
        cargo_stats = load_port_cargo_statistics()
        if cargo_stats:
            validation_results['cargo_statistics'] = {
                'tables_loaded': len(cargo_stats),
                'table_names': list(cargo_stats.keys())
            }
        
        # Overall status
        if monthly_data.empty and not cargo_stats:
            validation_results['overall_status'] = 'failed'
        elif monthly_data.empty or not cargo_stats:
            validation_results['overall_status'] = 'partial'
        else:
            validation_results['overall_status'] = 'success'
        
        logger.info(f"Data validation completed: {validation_results['overall_status']}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Error during data validation: {e}")
        validation_results['overall_status'] = 'error'
        return validation_results

# Sample data loading function for fallback
def load_sample_data() -> pd.DataFrame:
    """Load sample data for development/testing when real data is unavailable.
    
    Returns:
        pd.DataFrame: Sample container throughput data
    """
    # Generate sample monthly data for the last 3 years
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='MS')
    
    # Create realistic sample data with seasonal patterns
    np.random.seed(42)  # For reproducible results
    base_throughput = 1200  # Base monthly TEUs in thousands
    
    sample_data = []
    for date in dates:
        # Add seasonal variation (higher in Q4, lower in Q1)
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * date.month / 12)
        
        # Add some random variation
        random_factor = 1.0 + np.random.normal(0, 0.05)
        
        total_teus = base_throughput * seasonal_factor * random_factor
        seaborne_teus = total_teus * 0.7  # Roughly 70% seaborne
        river_teus = total_teus * 0.3     # Roughly 30% river
        
        sample_data.append({
            'Date': date,
            'seaborne_teus': round(seaborne_teus, 1),
            'river_teus': round(river_teus, 1),
            'total_teus': round(total_teus, 1)
        })
    
    df = pd.DataFrame(sample_data)
    df = df.set_index('Date')
    
    logger.info(f"Generated sample data: {len(df)} monthly records")
    return df