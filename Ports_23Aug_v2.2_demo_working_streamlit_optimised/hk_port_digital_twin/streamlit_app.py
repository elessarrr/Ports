"""Data loading utilities for Hong Kong Port Digital Twin.

This module provides functions to load and process various data sources including:
- Container throughput time series data
- Port cargo statistics
- Vessel arrival data
- Port berth configurations
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xml.etree.ElementTree as ET
import warnings
import threading
import time
from dataclasses import dataclass
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our new modules
try:
    # Weather integration temporarily disabled for removal
    # from .weather_integration import HKObservatoryIntegration, get_weather_impact_for_simulation
    from .file_monitor import PortDataFileMonitor, create_default_port_monitor
    from .vessel_data_fetcher import VesselDataFetcher
    from .vessel_data_scheduler import VesselDataScheduler
    
    # Set weather integration to None (disabled)
    HKObservatoryIntegration = None
    get_weather_impact_for_simulation = None
    logger.info("Weather integration disabled - feature removal in progress")
except ImportError:
    # Fallback for when modules are not available
    logger.warning("Some modules not available (file monitoring, vessel data pipeline)")
    HKObservatoryIntegration = None
    get_weather_impact_for_simulation = None
    PortDataFileMonitor = None
    create_default_port_monitor = None
    VesselDataFetcher = None
    VesselDataScheduler = None

# Data file paths
RAW_DATA_DIR = (Path(__file__).parent.parent.parent / ".." / "raw_data").resolve()
CONTAINER_THROUGHPUT_FILE = RAW_DATA_DIR / "Total_container_throughput_by_mode_of_transport_(EN).csv"
PORT_CARGO_STATS_DIR = RAW_DATA_DIR / "Port Cargo Statistics"
VESSEL_ARRIVALS_XML = (Path(__file__).parent.parent.parent / ".." / "raw_data" / "Arrived_in_last_36_hours.xml").resolve()

# Vessel data pipeline configuration
VESSEL_DATA_DIR = (Path(__file__).parent.parent.parent / ".." / "raw_data").resolve()
VESSEL_XML_FILES = [
    'Arrived_in_last_36_hours.xml',
    'Departed_in_last_36_hours.xml', 
    'Expected_arrivals.xml',
    'Expected_departures.xml'
]

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

def load_port_cargo_statistics(focus_tables: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load port cargo statistics from multiple CSV files.
    
    Args:
        focus_tables: Optional list of specific table names to load (e.g., ['Table_1_Eng', 'Table_2_Eng'])
                     If None, loads all available tables
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of cargo statistics by table
    """
    cargo_stats = {}
    
    try:
        # Get all CSV files in the Port Cargo Statistics directory
        csv_files = list(PORT_CARGO_STATS_DIR.glob("*.CSV"))
        
        for csv_file in csv_files:
            table_name = csv_file.stem.replace("Port Cargo Statistics_CSV_Eng-", "")
            
            # Skip if focus_tables is specified and this table is not in the list
            if focus_tables and table_name not in focus_tables:
                continue
            
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

def load_focused_cargo_statistics() -> Dict[str, pd.DataFrame]:
    """Load only Tables 1 & 2 for focused time series analysis.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing only Table_1_Eng and Table_2_Eng
    """
    return load_port_cargo_statistics(focus_tables=['Table_1_Eng', 'Table_2_Eng'])

def get_time_series_data(cargo_stats: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Extract and format time series data from Tables 1 & 2.
    
    Args:
        cargo_stats: Dictionary containing cargo statistics DataFrames
        
    Returns:
        Dict[str, pd.DataFrame]: Time series data formatted for analysis and visualization
    """
    time_series = {}
    
    try:
        # Process Table 1 - Shipment Types (2014-2023)
        if 'Table_1_Eng' in cargo_stats:
            df1 = cargo_stats['Table_1_Eng']
            
            # Extract throughput columns (years 2014-2023)
            throughput_cols = [col for col in df1.columns if 'Port_cargo_throughput_' in col and not 'percentage' in col and not 'rate_of_change' in col]
            
            # Create time series DataFrame
            years = [int(col.split('_')[-1]) for col in throughput_cols]
            
            shipment_ts = pd.DataFrame(index=years)
            for _, row in df1.iterrows():
                shipment_type = row.iloc[0]  # First column is shipment type
                values = [row[col] for col in throughput_cols]
                shipment_ts[shipment_type] = values
            
            shipment_ts.index.name = 'Year'
            time_series['shipment_types'] = shipment_ts
            
        # Process Table 2 - Transport Modes (2014, 2019-2023)
        if 'Table_2_Eng' in cargo_stats:
            df2 = cargo_stats['Table_2_Eng']
            
            # Extract throughput columns
            throughput_cols = [col for col in df2.columns if 'Port_cargo_throughput_' in col and not 'percentage' in col and not 'rate_of_change' in col]
            
            # Create time series DataFrame
            years = [int(col.split('_')[-1]) for col in throughput_cols]
            
            transport_ts = pd.DataFrame(index=years)
            for _, row in df2.iterrows():
                transport_mode = row.iloc[0]  # First column is transport mode
                values = [row[col] for col in throughput_cols]
                transport_ts[transport_mode] = values
            
            transport_ts.index.name = 'Year'
            time_series['transport_modes'] = transport_ts
            
        logger.info(f"Generated time series data for {len(time_series)} categories")
        return time_series
        
    except Exception as e:
        logger.error(f"Error generating time series data: {e}")
        return {}

def forecast_cargo_throughput(time_series_data: Dict[str, pd.DataFrame], forecast_years: int = 3) -> Dict[str, Dict]:
    """Generate forecasts for cargo throughput using linear regression.
    
    Args:
        time_series_data: Time series data from get_time_series_data()
        forecast_years: Number of years to forecast ahead
        
    Returns:
        Dict: Forecasts and model metrics for each category
    """
    forecasts = {}
    
    try:
        for category, df in time_series_data.items():
            category_forecasts = {}
            
            for column in df.columns:
                # Get non-null values
                series = df[column].dropna()
                
                if len(series) < 3:  # Need at least 3 points for meaningful forecast
                    continue
                    
                # Prepare data for linear regression
                X = np.array(series.index).reshape(-1, 1)
                y = series.values
                
                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Generate forecasts
                last_year = series.index.max()
                future_years = np.array(range(last_year + 1, last_year + forecast_years + 1)).reshape(-1, 1)
                predictions = model.predict(future_years)
                
                # Calculate model metrics
                y_pred = model.predict(X)
                mae = mean_absolute_error(y, y_pred)
                mse = mean_squared_error(y, y_pred)