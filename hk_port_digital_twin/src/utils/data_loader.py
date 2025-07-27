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
    from .weather_integration import HKObservatoryIntegration, get_weather_impact_for_simulation
    from .file_monitor import PortDataFileMonitor, create_default_port_monitor
except ImportError:
    # Fallback for when modules are not available
    logger.warning("Weather integration and file monitoring modules not available")
    HKObservatoryIntegration = None
    get_weather_impact_for_simulation = None
    PortDataFileMonitor = None
    create_default_port_monitor = None

# Data file paths
RAW_DATA_DIR = (Path(__file__).parent.parent.parent / ".." / "raw_data").resolve()
CONTAINER_THROUGHPUT_FILE = RAW_DATA_DIR / "Total_container_throughput_by_mode_of_transport_(EN).csv"
PORT_CARGO_STATS_DIR = RAW_DATA_DIR / "Port Cargo Statistics"
VESSEL_ARRIVALS_XML = (Path(__file__).parent.parent.parent / ".." / "Arrived_in_last_36_hours.xml").resolve()

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

def load_vessel_arrivals() -> pd.DataFrame:
    """Load and process real-time vessel arrival data from XML.
    
    Returns:
        pd.DataFrame: Processed vessel arrival data with structured information
    """
    try:
        # Check if XML file exists
        if not VESSEL_ARRIVALS_XML.exists():
            logger.warning(f"Vessel arrivals XML file not found: {VESSEL_ARRIVALS_XML}")
            return pd.DataFrame()
        
        # Read and clean XML content (skip comment lines)
        with open(VESSEL_ARRIVALS_XML, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Filter out comment lines and keep only XML content
        xml_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('This XML file') and not line.startswith('associated with it'):
                # Escape unescaped ampersands for proper XML parsing
                line = line.replace(' & ', ' &amp; ')
                xml_lines.append(line)
        
        # Join the cleaned lines
        content = '\n'.join(xml_lines)
        
        # Parse cleaned XML content
        root = ET.fromstring(content)
        
        # Extract vessel data
        vessels = []
        for vessel_element in root.findall('G_SQL1'):
            vessel_data = {}
            
            # Extract all vessel information
            call_sign = vessel_element.find('CALL_SIGN')
            vessel_name = vessel_element.find('VESSEL_NAME')
            ship_type = vessel_element.find('SHIP_TYPE')
            agent_name = vessel_element.find('AGENT_NAME')
            current_location = vessel_element.find('CURRENT_LOCATION')
            arrival_time = vessel_element.find('ARRIVAL_TIME')
            remark = vessel_element.find('REMARK')
            
            # Safely extract text content
            vessel_data['call_sign'] = call_sign.text if call_sign is not None else None
            vessel_data['vessel_name'] = vessel_name.text if vessel_name is not None else None
            vessel_data['ship_type'] = ship_type.text if ship_type is not None else None
            vessel_data['agent_name'] = agent_name.text if agent_name is not None else None
            vessel_data['current_location'] = current_location.text if current_location is not None else None
            vessel_data['arrival_time_str'] = arrival_time.text if arrival_time is not None else None
            vessel_data['remark'] = remark.text if remark is not None else None
            
            # Parse arrival time
            if vessel_data['arrival_time_str']:
                try:
                    vessel_data['arrival_time'] = pd.to_datetime(
                        vessel_data['arrival_time_str'], 
                        format='%d-%b-%Y %H:%M'
                    )
                except ValueError:
                    logger.warning(f"Could not parse arrival time: {vessel_data['arrival_time_str']}")
                    vessel_data['arrival_time'] = None
            else:
                vessel_data['arrival_time'] = None
            
            # Determine vessel status
            vessel_data['status'] = 'departed' if vessel_data.get('remark') == 'Departed' else 'in_port'
            
            # Categorize ship type
            vessel_data['ship_category'] = _categorize_ship_type(vessel_data['ship_type'])
            
            # Determine if at berth or anchorage
            vessel_data['location_type'] = _categorize_location(vessel_data['current_location'])
            
            vessels.append(vessel_data)
        
        # Create DataFrame
        df = pd.DataFrame(vessels)
        
        # Sort by arrival time
        if not df.empty and 'arrival_time' in df.columns:
            df = df.sort_values('arrival_time', na_position='last')
        
        logger.info(f"Loaded vessel arrivals data: {len(df)} vessels")
        return df
        
    except Exception as e:
        logger.error(f"Error loading vessel arrivals data: {e}")
        return pd.DataFrame()

def _categorize_ship_type(ship_type: str) -> str:
    """Categorize ship type into standard categories.
    
    Args:
        ship_type: Raw ship type string from XML
        
    Returns:
        str: Standardized ship category
    """
    if not ship_type:
        return 'unknown'
    
    ship_type_lower = ship_type.lower()
    
    if 'container' in ship_type_lower:
        return 'container'
    elif any(term in ship_type_lower for term in ['bulk', 'ore', 'cement', 'woodchip']):
        return 'bulk_carrier'
    elif 'chemical' in ship_type_lower:
        return 'chemical_tanker'
    elif any(term in ship_type_lower for term in ['general', 'cargo', 'heavy lift']):
        return 'general_cargo'
    elif 'tanker' in ship_type_lower:
        return 'tanker'
    else:
        return 'other'

def _categorize_location(location: str) -> str:
    """Categorize vessel location type.
    
    Args:
        location: Current location string from XML
        
    Returns:
        str: Location category (berth, anchorage, channel, etc.)
    """
    if not location:
        return 'unknown'
    
    location_lower = location.lower()
    
    if 'berth' in location_lower or 'terminal' in location_lower:
        return 'berth'
    elif 'anchorage' in location_lower:
        return 'anchorage'
    elif 'channel' in location_lower or 'buoy' in location_lower:
        return 'channel'
    else:
        return 'other'

def get_vessel_queue_analysis() -> Dict[str, any]:
    """Analyze current vessel queue and berth occupancy.
    
    Returns:
        Dict: Analysis of current port operations including queue metrics
    """
    try:
        vessels_df = load_vessel_arrivals()
        
        if vessels_df.empty:
            logger.warning("No vessel data available for queue analysis")
            return {}
        
        # Filter for vessels currently in port (not departed)
        active_vessels = vessels_df[vessels_df['status'] != 'departed'].copy()
        
        # Analyze by location type
        location_analysis = active_vessels.groupby('location_type').size().to_dict()
        
        # Analyze by ship category
        ship_category_analysis = active_vessels.groupby('ship_category').size().to_dict()
        
        # Count vessels at berths vs anchorage (queue)
        berth_count = location_analysis.get('berth', 0)
        queue_count = location_analysis.get('anchorage', 0)
        
        # Calculate queue metrics
        total_active = len(active_vessels)
        berth_utilization = (berth_count / (berth_count + queue_count)) * 100 if (berth_count + queue_count) > 0 else 0
        
        # Analyze arrival patterns (last 24 hours)
        now = datetime.now()
        recent_arrivals = vessels_df[
            (vessels_df['arrival_time'].notna()) & 
            (vessels_df['arrival_time'] >= now - pd.Timedelta(hours=24))
        ]
        
        analysis = {
            'current_status': {
                'total_vessels_in_port': total_active,
                'vessels_at_berth': berth_count,
                'vessels_in_queue': queue_count,
                'berth_utilization_pct': round(berth_utilization, 1)
            },
            'location_breakdown': location_analysis,
            'ship_category_breakdown': ship_category_analysis,
            'recent_activity': {
                'arrivals_last_24h': len(recent_arrivals),
                'departures_last_24h': len(vessels_df[vessels_df['status'] == 'departed'])
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'data_freshness': 'real_time'
        }
        
        logger.info(f"Vessel queue analysis completed: {total_active} active vessels")
        return analysis
        
    except Exception as e:
        logger.error(f"Error in vessel queue analysis: {e}")
        return {}

@dataclass
class RealTimeDataConfig:
    """Configuration for real-time data processing."""
    enable_weather_integration: bool = True
    enable_file_monitoring: bool = True
    vessel_update_interval: int = 300  # seconds (5 minutes)
    weather_update_interval: int = 1800  # seconds (30 minutes)
    auto_reload_on_file_change: bool = True
    cache_duration: int = 3600  # seconds (1 hour)

class RealTimeDataManager:
    """Enhanced data manager with real-time capabilities.
    
    This class provides real-time data processing capabilities including:
    - Automatic vessel data updates
    - Weather condition integration
    - File monitoring for automatic data reloading
    - Cached data management
    """
    
    def __init__(self, config: Optional[RealTimeDataConfig] = None):
        self.config = config or RealTimeDataConfig()
        self.weather_integration = None
        self.file_monitor = None
        self.is_running = False
        self.data_cache = {}
        self.last_updates = {}
        self.update_callbacks = {}
        
        # Enhanced error handling with circuit breaker pattern
        self.error_counts = {}
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_reset_time = 300  # 5 minutes
        self.circuit_breaker_states = {}  # 'open', 'closed', 'half_open'
        
        # Initialize components
        self._initialize_weather_integration()
        self._initialize_file_monitoring()
        
        logger.info("Real-time data manager initialized with enhanced error handling")
    
    def _initialize_weather_integration(self):
        """Initialize weather data integration."""
        if self.config.enable_weather_integration and HKObservatoryIntegration:
            try:
                self.weather_integration = HKObservatoryIntegration()
                logger.info("Weather integration enabled")
            except Exception as e:
                logger.error(f"Failed to initialize weather integration: {e}")
                self.weather_integration = None
        else:
            logger.info("Weather integration disabled or not available")
    
    def _initialize_file_monitoring(self):
        """Initialize file monitoring system."""
        if self.config.enable_file_monitoring and PortDataFileMonitor:
            try:
                self.file_monitor = create_default_port_monitor()
                
                # Set up custom callbacks for data reloading
                self.file_monitor.setup_vessel_monitoring(self._on_vessel_file_change)
                self.file_monitor.setup_cargo_monitoring(self._on_cargo_file_change)
                self.file_monitor.setup_berth_monitoring(self._on_berth_file_change)
                
                logger.info("File monitoring enabled")
            except Exception as e:
                logger.error(f"Failed to initialize file monitoring: {e}")
                self.file_monitor = None
        else:
            logger.info("File monitoring disabled or not available")
    
    def start_real_time_updates(self):
        """Start real-time data update processes."""
        if self.is_running:
            logger.warning("Real-time updates already running")
            return
        
        self.is_running = True
        
        # Start file monitoring
        if self.file_monitor:
            self.file_monitor.start_all_monitoring()
        
        # Start background update threads
        self.vessel_update_thread = threading.Thread(target=self._vessel_update_loop, daemon=True)
        self.weather_update_thread = threading.Thread(target=self._weather_update_loop, daemon=True)
        
        self.vessel_update_thread.start()
        self.weather_update_thread.start()
        
        logger.info("Real-time data updates started")
    
    def stop_real_time_updates(self):
        """Stop real-time data update processes."""
        self.is_running = False
        
        # Stop file monitoring
        if self.file_monitor:
            self.file_monitor.stop_all_monitoring()
        
        logger.info("Real-time data updates stopped")
    
    def _vessel_update_loop(self):
        """Background loop for vessel data updates."""
        while self.is_running:
            try:
                self._update_vessel_data()
                time.sleep(self.config.vessel_update_interval)
            except Exception as e:
                logger.error(f"Error in vessel update loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _weather_update_loop(self):
        """Background loop for weather data updates."""
        while self.is_running:
            try:
                self._update_weather_data()
                time.sleep(self.config.weather_update_interval)
            except Exception as e:
                logger.error(f"Error in weather update loop: {e}")
                time.sleep(300)  # Wait before retrying
    
    def _update_vessel_data(self):
        """Update vessel arrival data with validation and enhanced caching."""
        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open('vessel_update'):
                logger.warning("Skipping vessel data update - circuit breaker open")
                return
            
            vessel_data = load_vessel_arrivals()
            if not vessel_data.empty:
                # Validate and process data
                validation_result = self.validate_and_process_data('vessel_arrivals', vessel_data)
                
                if validation_result['status'] == 'success' and validation_result['validation_result'].get('valid', False):
                    # Store in both local and global cache
                    self.data_cache['vessel_arrivals'] = vessel_data
                    self.last_updates['vessel_arrivals'] = datetime.now()
                    data_cache.set('vessel_arrivals', vessel_data)
                    
                    # Cross-reference with historical patterns
                    try:
                        historical_analysis = self._cross_reference_vessel_data(vessel_data)
                        if historical_analysis:
                            data_cache.set('vessel_historical_analysis', historical_analysis)
                    except Exception as e:
                        logger.warning(f"Error in historical cross-reference: {e}")
                    
                    # Trigger callbacks
                    if 'vessel_arrivals' in self.update_callbacks:
                        for callback in self.update_callbacks['vessel_arrivals']:
                            try:
                                callback(vessel_data)
                            except Exception as e:
                                logger.error(f"Error in vessel update callback: {e}")
                    
                    self._record_operation_success('vessel_update')
                    logger.debug("Vessel data updated and validated successfully")
                else:
                    self._record_operation_failure('vessel_update')
                    logger.warning(f"Vessel data validation failed: {validation_result}")
            else:
                logger.warning("No vessel data available for update")
                
        except Exception as e:
            self._record_operation_failure('vessel_update')
            logger.error(f"Error updating vessel data: {e}")
    
    def _cross_reference_vessel_data(self, current_data: pd.DataFrame) -> Dict[str, any]:
        """Cross-reference current vessel data with historical patterns."""
        try:
            # Get historical container throughput for comparison
            historical_data = data_cache.get('container_throughput')
            if historical_data is None:
                historical_data = load_container_throughput()
                if not historical_data.empty:
                    data_cache.set('container_throughput', historical_data)
            
            if historical_data is None or historical_data.empty:
                return None
            
            analysis = {
                'current_vessel_count': len(current_data),
                'timestamp': datetime.now().isoformat(),
                'patterns': {}
            }
            
            # Analyze vessel types vs historical throughput trends
            if 'ship_type' in current_data.columns:
                vessel_types = current_data['ship_type'].value_counts().to_dict()
                analysis['patterns']['vessel_type_distribution'] = vessel_types
            
            # Compare with recent throughput trends
            if not historical_data.empty:
                recent_throughput = historical_data.tail(6)  # Last 6 months
                avg_monthly_teus = recent_throughput['total_teus'].mean()
                
                # Estimate expected vessel count based on throughput
                # Rough estimate: 1 vessel per 1000 TEUs per month
                expected_vessels_per_day = (avg_monthly_teus * 1000) / (30 * 1000)  # Convert to daily
                
                analysis['patterns']['throughput_correlation'] = {
                    'avg_monthly_teus': float(avg_monthly_teus),
                    'estimated_daily_vessels': float(expected_vessels_per_day),
                    'current_vessels': len(current_data),
                    'variance_from_expected': len(current_data) - expected_vessels_per_day
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in vessel data cross-reference: {e}")
            return None
    
    def _update_weather_data(self):
        """Update weather condition data."""
        if not self.weather_integration:
            return
        
        try:
            weather_data = self.weather_integration.get_current_weather()
            if weather_data:
                self.data_cache['weather_conditions'] = weather_data
                self.last_updates['weather_conditions'] = datetime.now()
                
                # Trigger callbacks
                if 'weather_conditions' in self.update_callbacks:
                    for callback in self.update_callbacks['weather_conditions']:
                        try:
                            callback(weather_data)
                        except Exception as e:
                            logger.error(f"Error in weather update callback: {e}")
                
                logger.debug("Weather data updated")
        except Exception as e:
            logger.error(f"Error updating weather data: {e}")
    
    def _on_vessel_file_change(self, file_path: str):
        """Handle vessel file changes."""
        logger.info(f"Vessel file changed: {file_path}")
        if self.config.auto_reload_on_file_change:
            self._update_vessel_data()
    
    def _on_cargo_file_change(self, file_path: str):
        """Handle cargo file changes."""
        logger.info(f"Cargo file changed: {file_path}")
        if self.config.auto_reload_on_file_change:
            # Reload cargo statistics
            try:
                cargo_data = load_port_cargo_statistics()
                self.data_cache['cargo_statistics'] = cargo_data
                self.last_updates['cargo_statistics'] = datetime.now()
            except Exception as e:
                logger.error(f"Error reloading cargo data: {e}")
    
    def _on_berth_file_change(self, file_path: str):
        """Handle berth file changes."""
        logger.info(f"Berth file changed: {file_path}")
        if self.config.auto_reload_on_file_change:
            # Custom berth data reloading logic would go here
            logger.info("Berth data reload triggered")
    
    def register_update_callback(self, data_type: str, callback: Callable):
        """Register a callback for data updates.
        
        Args:
            data_type: Type of data ('vessel_arrivals', 'weather_conditions', etc.)
            callback: Function to call when data is updated
        """
        if data_type not in self.update_callbacks:
            self.update_callbacks[data_type] = []
        self.update_callbacks[data_type].append(callback)
        logger.info(f"Registered callback for {data_type} updates")
    
    def get_cached_data(self, data_type: str, max_age_seconds: Optional[int] = None) -> Optional[any]:
        """Get cached data if available and fresh.
        
        Args:
            data_type: Type of data to retrieve
            max_age_seconds: Maximum age of data in seconds (uses config default if None)
            
        Returns:
            Cached data if available and fresh, None otherwise
        """
        # First check global cache
        global_cached = data_cache.get(data_type, max_age_seconds)
        if global_cached is not None:
            return global_cached
        
        # Fallback to local cache
        if data_type not in self.data_cache:
            return None
        
        # Check data freshness
        max_age = max_age_seconds or self.config.cache_duration
        last_update = self.last_updates.get(data_type)
        
        if last_update and (datetime.now() - last_update).total_seconds() <= max_age:
            return self.data_cache[data_type]
        
        return None
    
    def _is_circuit_breaker_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for a specific operation."""
        if operation not in self.circuit_breaker_states:
            self.circuit_breaker_states[operation] = 'closed'
            return False
        
        state = self.circuit_breaker_states[operation]
        if state == 'open':
            # Check if we should try half-open
            last_error_time = getattr(self, f'_last_error_time_{operation}', 0)
            if time.time() - last_error_time > self.circuit_breaker_reset_time:
                self.circuit_breaker_states[operation] = 'half_open'
                return False
            return True
        
        return False
    
    def _record_operation_success(self, operation: str):
        """Record successful operation and reset circuit breaker."""
        self.error_counts[operation] = 0
        self.circuit_breaker_states[operation] = 'closed'
    
    def _record_operation_failure(self, operation: str):
        """Record operation failure and update circuit breaker state."""
        self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
        setattr(self, f'_last_error_time_{operation}', time.time())
        
        if self.error_counts[operation] >= self.circuit_breaker_threshold:
            self.circuit_breaker_states[operation] = 'open'
            logger.warning(f"Circuit breaker opened for {operation} after {self.error_counts[operation]} failures")
    
    def validate_and_process_data(self, data_type: str, data: any) -> Dict[str, any]:
        """Validate and process data with enhanced error handling.
        
        Args:
            data_type: Type of data being processed
            data: Data to validate and process
            
        Returns:
            Dict containing validation results and processed data
        """
        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open(f'validate_{data_type}'):
                return {
                    'status': 'circuit_breaker_open',
                    'message': f'Circuit breaker open for {data_type} validation',
                    'data': data
                }
            
            # Perform validation based on data type
            if data_type == 'vessel_arrivals':
                validation_result = _validate_vessel_data(data)
            elif data_type == 'container_throughput':
                validation_result = _validate_container_data(data)
            elif data_type == 'weather_conditions':
                validation_result = _validate_weather_data(data)
            else:
                validation_result = {'status': 'unknown_data_type', 'valid': False}
            
            # Cache validated data if successful
            if validation_result.get('valid', False):
                cache_key = f'validated_{data_type}'
                data_cache.set(cache_key, data)
                self.data_cache[cache_key] = data
                self.last_updates[cache_key] = datetime.now()
                self._record_operation_success(f'validate_{data_type}')
            else:
                self._record_operation_failure(f'validate_{data_type}')
            
            return {
                'status': 'success',
                'validation_result': validation_result,
                'data': data
            }
            
        except Exception as e:
            self._record_operation_failure(f'validate_{data_type}')
            logger.error(f"Error validating {data_type} data: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'data': data
            }
    
    def get_comprehensive_data_quality_report(self) -> Dict[str, any]:
        """Get comprehensive data quality report for all data sources."""
        try:
            # Use the enhanced validate_data_quality function
            quality_report = validate_data_quality()
            
            # Add real-time manager specific metrics
            quality_report['real_time_metrics'] = {
                'circuit_breaker_states': self.circuit_breaker_states.copy(),
                'error_counts': self.error_counts.copy(),
                'cached_data_types': list(self.data_cache.keys()),
                'last_updates': {k: v.isoformat() for k, v in self.last_updates.items()},
                'global_cache_stats': data_cache.get_stats()
            }
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_real_time_vessel_data(self) -> pd.DataFrame:
        """Get real-time vessel data with weather impact assessment.
        
        Returns:
            Enhanced vessel data with weather impact information
        """
        # Get cached vessel data or load fresh
        vessel_data = self.get_cached_data('vessel_arrivals')
        if vessel_data is None:
            vessel_data = load_vessel_arrivals()
            if not vessel_data.empty:
                self.data_cache['vessel_arrivals'] = vessel_data
                self.last_updates['vessel_arrivals'] = datetime.now()
        
        if vessel_data is None or vessel_data.empty:
            return pd.DataFrame()
        
        # Enhance with weather impact if available
        weather_data = self.get_cached_data('weather_conditions')
        if weather_data:
            vessel_data = vessel_data.copy()
            vessel_data['weather_impact_score'] = weather_data.impact_score
            vessel_data['operational_status'] = weather_data.operational_status
            vessel_data['weather_description'] = weather_data.weather_description
        
        return vessel_data
    
    def get_enhanced_queue_analysis(self) -> Dict[str, any]:
        """Get enhanced vessel queue analysis with weather conditions.
        
        Returns:
            Comprehensive analysis including weather impact
        """
        # Get base queue analysis
        analysis = get_vessel_queue_analysis()
        
        # Enhance with weather data
        weather_data = self.get_cached_data('weather_conditions')
        if weather_data:
            analysis['weather_conditions'] = {
                'current_conditions': weather_data.weather_description,
                'operational_impact': weather_data.operational_status,
                'impact_score': weather_data.impact_score,
                'wind_speed_kmh': weather_data.wind_speed,
                'visibility_km': weather_data.visibility,
                'timestamp': weather_data.timestamp.isoformat()
            }
            
            # Adjust operational metrics based on weather
            if weather_data.operational_status == 'suspended':
                analysis['current_status']['operational_note'] = 'Operations suspended due to weather'
            elif weather_data.operational_status == 'restricted':
                analysis['current_status']['operational_note'] = 'Operations restricted due to weather'
            else:
                analysis['current_status']['operational_note'] = 'Normal operations'
        
        # Add real-time data freshness info
        analysis['data_freshness'] = {
            'vessel_data_age': self._get_data_age('vessel_arrivals'),
            'weather_data_age': self._get_data_age('weather_conditions'),
            'last_update_check': datetime.now().isoformat()
        }
        
        return analysis
    
    def _get_data_age(self, data_type: str) -> Optional[str]:
        """Get age of cached data in human-readable format."""
        last_update = self.last_updates.get(data_type)
        if last_update:
            age_seconds = (datetime.now() - last_update).total_seconds()
            if age_seconds < 60:
                return f"{int(age_seconds)} seconds ago"
            elif age_seconds < 3600:
                return f"{int(age_seconds/60)} minutes ago"
            else:
                return f"{int(age_seconds/3600)} hours ago"
        return None
    
    def get_status(self) -> Dict[str, any]:
        """Get status of real-time data manager."""
        status = {
            'is_running': self.is_running,
            'config': {
                'weather_integration_enabled': self.config.enable_weather_integration,
                'file_monitoring_enabled': self.config.enable_file_monitoring,
                'vessel_update_interval': self.config.vessel_update_interval,
                'weather_update_interval': self.config.weather_update_interval
            },
            'components': {
                'weather_integration': self.weather_integration is not None,
                'file_monitor': self.file_monitor is not None
            },
            'cached_data': list(self.data_cache.keys()),
            'last_updates': {k: v.isoformat() for k, v in self.last_updates.items()},
            'registered_callbacks': {k: len(v) for k, v in self.update_callbacks.items()}
        }
        
        # Add file monitor status if available
        if self.file_monitor:
            status['file_monitor_status'] = self.file_monitor.get_status()
        
        return status

# Global instance for easy access
_real_time_manager = None

def get_real_time_manager(config: Optional[RealTimeDataConfig] = None) -> RealTimeDataManager:
    """Get or create the global real-time data manager instance.
    
    Args:
        config: Configuration for the manager (only used on first call)
        
    Returns:
        RealTimeDataManager instance
    """
    global _real_time_manager
    if _real_time_manager is None:
        _real_time_manager = RealTimeDataManager(config)
    return _real_time_manager

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
    """Enhanced data quality validation across all loaded datasets.
    
    Returns:
        Dict: Comprehensive data quality metrics and validation results
    """
    validation_results = {
        'container_throughput': {},
        'cargo_statistics': {},
        'vessel_arrivals': {},
        'weather_data': {},
        'overall_status': 'unknown',
        'data_freshness': {},
        'anomaly_detection': {},
        'cross_validation': {}
    }
    
    try:
        # Validate container throughput data
        monthly_data = load_container_throughput()
        if not monthly_data.empty:
            validation_results['container_throughput'] = _validate_container_data(monthly_data)
        
        # Validate cargo statistics
        cargo_stats = load_port_cargo_statistics()
        if cargo_stats:
            validation_results['cargo_statistics'] = _validate_cargo_data(cargo_stats)
        
        # Validate vessel arrivals data
        try:
            vessel_data = load_vessel_arrivals()
            if not vessel_data.empty:
                validation_results['vessel_arrivals'] = _validate_vessel_data(vessel_data)
        except Exception as e:
            logger.warning(f"Could not validate vessel data: {e}")
            validation_results['vessel_arrivals'] = {'status': 'unavailable', 'error': str(e)}
        
        # Validate weather data if available
        validation_results['weather_data'] = _validate_weather_data()
        
        # Cross-reference validation
        validation_results['cross_validation'] = _cross_validate_datasets(
            monthly_data, cargo_stats, validation_results.get('vessel_arrivals', {})
        )
        
        # Anomaly detection
        validation_results['anomaly_detection'] = _detect_data_anomalies(monthly_data)
        
        # Data freshness check
        validation_results['data_freshness'] = _check_data_freshness()
        
        # Overall status determination
        validation_results['overall_status'] = _determine_overall_status(validation_results)
        
        logger.info(f"Enhanced data validation completed: {validation_results['overall_status']}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Error during enhanced data validation: {e}")
        validation_results['overall_status'] = 'error'
        validation_results['error_details'] = str(e)
        return validation_results

def _validate_container_data(data: pd.DataFrame) -> Dict[str, any]:
    """Validate container throughput data quality."""
    try:
        validation = {
            'records_count': len(data),
            'date_range': f"{data.index.min()} to {data.index.max()}",
            'missing_values': data.isnull().sum().to_dict(),
            'data_completeness': (1 - data.isnull().sum().sum() / data.size) * 100 if data.size > 0 else 0,
            'data_consistency': {},
            'outlier_detection': {}
        }
        
        # Check data consistency (total should equal seaborne + river)
        if all(col in data.columns for col in ['seaborne_teus', 'river_teus', 'total_teus']):
            calculated_total = data['seaborne_teus'] + data['river_teus']
            diff = abs(data['total_teus'] - calculated_total)
            tolerance = data['total_teus'] * 0.01  # 1% tolerance
            consistency_issues = (diff > tolerance).sum()
            
            validation['data_consistency'] = {
                'total_calculation_errors': int(consistency_issues),
                'consistency_rate': float((len(data) - consistency_issues) / len(data) * 100)
            }
        
        # Outlier detection using IQR method
        for col in ['seaborne_teus', 'river_teus', 'total_teus']:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                validation['outlier_detection'][col] = int(outliers)
        
        return validation
    except Exception as e:
        logger.error(f"Error validating container data: {e}")
        return {'status': 'error', 'error': str(e)}

def _validate_cargo_data(cargo_stats: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """Validate cargo statistics data quality."""
    try:
        validation = {
            'tables_loaded': len(cargo_stats),
            'table_names': list(cargo_stats.keys()),
            'table_details': {}
        }
        
        for table_name, df in cargo_stats.items():
            table_validation = {
                'records': len(df),
                'columns': len(df.columns),
                'missing_data_percentage': (df.isnull().sum().sum() / df.size) * 100,
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns)
            }
            validation['table_details'][table_name] = table_validation
        
        return validation
    except Exception as e:
        logger.error(f"Error validating cargo data: {e}")
        return {'status': 'error', 'error': str(e)}

def _validate_vessel_data(vessel_data: pd.DataFrame) -> Dict[str, any]:
    """Validate vessel arrivals data quality."""
    try:
        validation = {
            'records_count': len(vessel_data),
            'unique_vessels': vessel_data['vessel_name'].nunique() if 'vessel_name' in vessel_data.columns else 0,
            'date_range': f"{vessel_data['arrival_time'].min()} to {vessel_data['arrival_time'].max()}" if 'arrival_time' in vessel_data.columns else 'unknown',
            'missing_values': vessel_data.isnull().sum().to_dict(),
            'data_completeness': (1 - vessel_data.isnull().sum().sum() / vessel_data.size) * 100 if vessel_data.size > 0 else 0
        }
        
        # Check for duplicate vessels
        if 'vessel_name' in vessel_data.columns and 'arrival_time' in vessel_data.columns:
            duplicates = vessel_data.duplicated(subset=['vessel_name', 'arrival_time']).sum()
            validation['duplicate_records'] = int(duplicates)
        
        return validation
    except Exception as e:
        logger.error(f"Error validating vessel data: {e}")
        return {'status': 'error', 'error': str(e)}

def _validate_weather_data() -> Dict[str, any]:
    """Validate weather data availability and quality."""
    try:
        # Check if weather integration is available
        if HKObservatoryIntegration:
            weather_integration = HKObservatoryIntegration()
            current_weather = weather_integration.get_current_weather()
            
            if current_weather:
                return {
                    'status': 'available',
                    'last_update': current_weather.get('timestamp', 'unknown'),
                    'data_fields': list(current_weather.keys())
                }
            else:
                return {'status': 'unavailable', 'reason': 'no_current_data'}
        else:
            return {'status': 'unavailable', 'reason': 'integration_not_available'}
    except Exception as e:
        logger.error(f"Error validating weather data: {e}")
        return {'status': 'error', 'error': str(e)}

def _cross_validate_datasets(container_data: pd.DataFrame, cargo_stats: Dict, vessel_data: Dict) -> Dict[str, any]:
    """Cross-validate data consistency across different datasets."""
    try:
        cross_validation = {
            'temporal_alignment': {},
            'volume_consistency': {},
            'pattern_correlation': {}
        }
        
        # Check temporal alignment
        if not container_data.empty:
            container_date_range = (container_data.index.min(), container_data.index.max())
            cross_validation['temporal_alignment']['container_data'] = {
                'start': container_date_range[0].strftime('%Y-%m-%d'),
                'end': container_date_range[1].strftime('%Y-%m-%d')
            }
        
        # Check vessel data temporal alignment
        if vessel_data.get('status') != 'error' and 'date_range' in vessel_data:
            cross_validation['temporal_alignment']['vessel_data'] = vessel_data['date_range']
        
        # Volume consistency checks
        if not container_data.empty and cargo_stats:
            # Basic consistency check between container and cargo data
            latest_container_volume = container_data['total_teus'].iloc[-1] if 'total_teus' in container_data.columns else None
            cross_validation['volume_consistency']['latest_container_volume'] = float(latest_container_volume) if latest_container_volume else None
        
        return cross_validation
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        return {'status': 'error', 'error': str(e)}

def _detect_data_anomalies(data: pd.DataFrame) -> Dict[str, any]:
    """Detect anomalies in container throughput data."""
    try:
        anomalies = {
            'sudden_changes': {},
            'trend_breaks': {},
            'seasonal_anomalies': {}
        }
        
        if data.empty:
            return anomalies
        
        # Detect sudden changes (month-over-month changes > 20%)
        for col in ['seaborne_teus', 'river_teus', 'total_teus']:
            if col in data.columns:
                pct_change = data[col].pct_change().abs()
                sudden_changes = pct_change[pct_change > 0.2]
                anomalies['sudden_changes'][col] = {
                    'count': len(sudden_changes),
                    'dates': sudden_changes.index.strftime('%Y-%m').tolist() if len(sudden_changes) > 0 else []
                }
        
        # Detect trend breaks using rolling statistics
        if 'total_teus' in data.columns and len(data) >= 12:
            rolling_mean = data['total_teus'].rolling(window=6).mean()
            rolling_std = data['total_teus'].rolling(window=6).std()
            z_scores = abs((data['total_teus'] - rolling_mean) / rolling_std)
            trend_breaks = z_scores[z_scores > 2.5]  # More than 2.5 standard deviations
            
            anomalies['trend_breaks'] = {
                'count': len(trend_breaks),
                'dates': trend_breaks.index.strftime('%Y-%m').tolist() if len(trend_breaks) > 0 else []
            }
        
        return anomalies
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return {'status': 'error', 'error': str(e)}

def _check_data_freshness() -> Dict[str, any]:
    """Check how fresh/recent the available data is."""
    try:
        freshness = {
            'container_data': 'unknown',
            'vessel_data': 'unknown',
            'weather_data': 'unknown'
        }
        
        # Check container data freshness
        try:
            container_data = load_container_throughput()
            if not container_data.empty:
                latest_date = container_data.index.max()
                days_old = (datetime.now() - latest_date).days
                freshness['container_data'] = {
                    'latest_date': latest_date.strftime('%Y-%m-%d'),
                    'days_old': days_old,
                    'freshness_status': 'fresh' if days_old < 60 else 'stale' if days_old < 180 else 'very_old'
                }
        except Exception:
            freshness['container_data'] = 'unavailable'
        
        # Check vessel data freshness
        try:
            vessel_data = load_vessel_arrivals()
            if not vessel_data.empty and 'arrival_time' in vessel_data.columns:
                latest_arrival = vessel_data['arrival_time'].max()
                hours_old = (datetime.now() - latest_arrival).total_seconds() / 3600
                freshness['vessel_data'] = {
                    'latest_arrival': latest_arrival.strftime('%Y-%m-%d %H:%M'),
                    'hours_old': round(hours_old, 1),
                    'freshness_status': 'real_time' if hours_old < 1 else 'recent' if hours_old < 24 else 'old'
                }
        except Exception:
            freshness['vessel_data'] = 'unavailable'
        
        return freshness
    except Exception as e:
        logger.error(f"Error checking data freshness: {e}")
        return {'status': 'error', 'error': str(e)}

def _determine_overall_status(validation_results: Dict) -> str:
    """Determine overall data quality status based on all validation results."""
    try:
        # Count successful validations
        successful_validations = 0
        total_validations = 0
        has_data = False
        
        for key in ['container_throughput', 'cargo_statistics', 'vessel_arrivals']:
            if key in validation_results and validation_results[key]:
                total_validations += 1
                validation_result = validation_results[key]
                
                # Check if this validation has actual data
                if key == 'container_throughput' and validation_result.get('records_count', 0) > 0:
                    has_data = True
                elif key == 'cargo_statistics' and validation_result.get('tables_loaded', 0) > 0:
                    has_data = True
                elif key == 'vessel_arrivals' and validation_result.get('records_count', 0) > 0:
                    has_data = True
                
                # Check if validation was successful (no error status)
                if validation_result.get('status') != 'error':
                    successful_validations += 1
        
        # If no actual data is present, return no_data
        if not has_data:
            return 'no_data'
        elif total_validations == 0:
            return 'no_data'
        elif successful_validations == total_validations:
            return 'excellent'
        elif successful_validations >= total_validations * 0.7:
            return 'good'
        elif successful_validations >= total_validations * 0.5:
            return 'fair'
        else:
            return 'poor'
    except Exception:
        return 'error'

# Enhanced caching system
class DataCache:
    """Enhanced data caching system with TTL and validation."""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
        self.access_counts = {}
        
    def get(self, key: str, ttl: Optional[int] = None) -> Optional[any]:
        """Get cached data if available and fresh."""
        if key not in self.cache:
            return None
            
        ttl = ttl or self.default_ttl
        age = time.time() - self.timestamps[key]
        
        if age > ttl:
            # Data is stale, remove from cache
            self.invalidate(key)
            return None
            
        # Update access count
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        return self.cache[key]
    
    def set(self, key: str, value: any) -> None:
        """Cache data with timestamp."""
        self.cache[key] = value
        self.timestamps[key] = time.time()
        self.access_counts[key] = 0
        
    def invalidate(self, key: str) -> None:
        """Remove data from cache."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_counts.pop(key, None)
        
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.timestamps.clear()
        self.access_counts.clear()
        
    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        return {
            'cached_items': len(self.cache),
            'total_access_count': sum(self.access_counts.values()),
            'cache_keys': list(self.cache.keys()),
            'access_counts': self.access_counts.copy()
        }

# Global cache instance
data_cache = DataCache()

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