"""Main Streamlit application for the Hong Kong Port Digital Twin.

This application provides a comprehensive dashboard for monitoring and analyzing
port operations, including cargo throughput, vessel traffic, and real-time updates.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import time
import threading
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Data Loading and Caching ---

@st.cache_data(ttl=3600)
def load_container_throughput() -> pd.DataFrame:
    """Load and process container throughput time series data."""
    file_path = Path(__file__).parent / "raw_data" / "Total_container_throughput_by_mode_of_transport_(EN).csv"
    try:
        df = pd.read_csv(file_path)
        df = df.copy()
        numeric_cols = ['Seaborne ( \'000 TEUs)', 'River ( \'000 TEUs)', 'Total ( \'000 TEUs)']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        monthly_data = df[df['Month'] != 'All'].copy()
        monthly_data['Date'] = pd.to_datetime(
            monthly_data['Year'].astype(str) + '-' + monthly_data['Month'],
            format='%Y-%b',
            errors='coerce'
        )
        monthly_data = monthly_data.dropna(subset=['Date'])
        monthly_data = monthly_data.set_index('Date').sort_index()
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

@st.cache_data(ttl=3600)
def load_port_cargo_statistics(focus_tables: list = None) -> dict:
    """Load port cargo statistics from multiple CSV files."""
    cargo_stats = {}
    stats_dir = Path(__file__).parent / "raw_data" / "Port Cargo Statistics"
    try:
        csv_files = list(stats_dir.glob("*.CSV"))
        for csv_file in csv_files:
            table_name = csv_file.stem.replace("Port Cargo Statistics_CSV_Eng-", "")
            if focus_tables and table_name not in focus_tables:
                continue
            try:
                df = pd.read_csv(csv_file)
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
    """Clean and validate cargo statistics data."""
    try:
        cleaned_df = df.copy()
        numeric_columns = [col for col in cleaned_df.columns if any(year in col for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'])]
        for col in numeric_columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].astype(str).replace({
                    '-': np.nan,
                    '§': '0',
                    'N/A': np.nan,
                    '': np.nan
                })
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        total_cells = cleaned_df.size
        missing_cells = cleaned_df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        logger.info(f"{table_name} data quality: {completeness:.1f}% complete")
        return cleaned_df
    except Exception as e:
        logger.error(f"Error cleaning {table_name} data: {e}")
        return df

@st.cache_data(ttl=3600)
def get_time_series_data(cargo_stats: dict) -> dict:
    """Extract and format time series data from Tables 1 & 2."""
    time_series = {}
    try:
        if 'Table_1_Eng' in cargo_stats:
            df1 = cargo_stats['Table_1_Eng']
            throughput_cols = [col for col in df1.columns if 'Port_cargo_throughput_' in col and not 'percentage' in col and not 'rate_of_change' in col]
            years = [int(col.split('_')[-1]) for col in throughput_cols]
            shipment_ts = pd.DataFrame(index=years)
            for _, row in df1.iterrows():
                shipment_type = row.iloc[0]
                values = [row[col] for col in throughput_cols]
                shipment_ts[shipment_type] = values
            shipment_ts.index.name = 'Year'
            time_series['shipment_types'] = shipment_ts
        if 'Table_2_Eng' in cargo_stats:
            df2 = cargo_stats['Table_2_Eng']
            throughput_cols = [col for col in df2.columns if 'Port_cargo_throughput_' in col and not 'percentage' in col and not 'rate_of_change' in col]
            years = [int(col.split('_')[-1]) for col in throughput_cols]
            transport_ts = pd.DataFrame(index=years)
            for _, row in df2.iterrows():
                transport_mode = row.iloc[0]
                values = [row[col] for col in throughput_cols]
                transport_ts[transport_mode] = values
            transport_ts.index.name = 'Year'
            time_series['transport_modes'] = transport_ts
        logger.info(f"Generated time series data for {len(time_series)} categories")
        return time_series
    except Exception as e:
        logger.error(f"Error generating time series data: {e}")
        return {}

@st.cache_data(ttl=3600)
def forecast_cargo_throughput(time_series_data: dict, forecast_years: int = 3) -> dict:
    """Generate forecasts for cargo throughput using Exponential Smoothing."""
    forecasts = {}
    try:
        for category, df in time_series_data.items():
            category_forecasts = {}
            for col in df.columns:
                series = df[col].dropna()
                if len(series) > 1:
                    try:
                        model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method='estimated').fit()
                        forecast_values = model.forecast(forecast_years)
                        category_forecasts[col] = {
                            'historical_data': series.to_dict(),
                            'forecast_years': list(range(series.index.max() + 1, series.index.max() + 1 + forecast_years)),
                            'forecast_values': forecast_values.tolist(),
                            'model_params': model.params
                        }
                    except Exception as e:
                        logger.warning(f"Could not generate forecast for {col} in {category}: {e}")
                        continue
            forecasts[category] = category_forecasts
        logger.info(f"Generated forecasts for {len(forecasts)} categories")
        return forecasts
    except Exception as e:
        logger.error(f"Error generating cargo forecasts: {e}")
        return {}

def get_enhanced_cargo_analysis(cargo_stats: dict) -> dict:
    """Provide enhanced analysis of cargo statistics."""
    analysis = {}
    try:
        # Example: Analyze trends in the first available table
        if cargo_stats:
            first_table_name = next(iter(cargo_stats))
            df = cargo_stats[first_table_name]
            analysis['trends'] = _analyze_trends(df)
            analysis['efficiency_metrics'] = _calculate_focused_efficiency_metrics(df)
        return analysis
    except Exception as e:
        logger.error(f"Error in enhanced cargo analysis: {e}")
        return {}

def _analyze_trends(df: pd.DataFrame) -> dict:
    """Analyze trends in cargo data."""
    trends = {}
    try:
        for i, row in df.iterrows():
            category = row[0]
            # Select only numeric columns for trend analysis
            numeric_cols = df.select_dtypes(include=np.number).columns
            values = pd.to_numeric(row[numeric_cols], errors='coerce').dropna()
            if len(values) > 1:
                if values.iloc[-1] > values.iloc[0]:
                    trends[category] = "Upward"
                elif values.iloc[-1] < values.iloc[0]:
                    trends[category] = "Downward"
                else:
                    trends[category] = "Stable"
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
    return trends

def _calculate_focused_efficiency_metrics(df: pd.DataFrame) -> dict:
    """Calculate focused efficiency metrics from cargo data."""
    metrics = {}
    try:
        if 'Laden' in df.iloc[:, 0].values and 'Empty' in df.iloc[:, 0].values:
            laden_volume = df[df.iloc[:, 0] == 'Laden'].select_dtypes(include=np.number).sum().sum()
            empty_volume = df[df.iloc[:, 0] == 'Empty'].select_dtypes(include=np.number).sum().sum()
            if empty_volume > 0:
                metrics['laden_to_empty_ratio'] = laden_volume / empty_volume
    except Exception as e:
        logger.error(f"Error calculating efficiency metrics: {e}")
    return metrics

@st.cache_data(ttl=3600)
def load_arriving_ships() -> pd.DataFrame:
    """Load and process arriving ships data from XML."""
    file_path = Path(__file__).parent / "raw_data" / "Arrived_in_last_36_hours.xml"
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ships_data = []
        for ship_element in root.findall('arriving_ship'):
            ship_info = {
                'name': ship_element.findtext('name'),
                'arrival_time': _parse_vessel_timestamp(ship_element.findtext('arrival_time')),
                'ship_type': _categorize_ship_type(ship_element.findtext('ship_type')),
                'location': _categorize_location(ship_element.findtext('location'))
            }
            ships_data.append(ship_info)
        df = pd.DataFrame(ships_data)
        logger.info(f"Loaded arriving ships data: {len(df)} records")
        return df
    except (ET.ParseError, FileNotFoundError) as e:
        logger.error(f"Error loading arriving ships data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_vessel_arrivals() -> pd.DataFrame:
    """Load and process vessel arrival data from CSV."""
    file_path = Path(__file__).parent / "raw_data" / "Vessel_Arrivals.csv"
    try:
        df = pd.read_csv(file_path)
        df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce')
        df['ship_type'] = df['ship_type'].apply(_categorize_ship_type)
        df['location'] = df['location'].apply(_categorize_location)
        logger.info(f"Loaded vessel arrivals data: {len(df)} records")
        return df
    except FileNotFoundError:
        logger.error(f"Vessel arrivals file not found: {file_path}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_combined_vessel_data() -> pd.DataFrame:
    """Load and combine all available vessel data."""
    arriving_ships = load_arriving_ships()
    vessel_arrivals = load_vessel_arrivals()
    # Placeholder for other vessel data sources
    # departing_ships = load_departing_ships()
    # expected_arrivals = load_expected_arrivals()

    combined_df = pd.concat([arriving_ships, vessel_arrivals], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['name', 'arrival_time'])
    combined_df = combined_df.sort_values(by='arrival_time').reset_index(drop=True)
    logger.info(f"Combined vessel data: {len(combined_df)} unique records")
    return combined_df

def _parse_vessel_timestamp(time_str: str) -> datetime:
    """Parse vessel timestamp from various string formats."""
    if not time_str: return None
    try:
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M')
    except ValueError:
        try:
            return datetime.strptime(time_str, '%d/%m/%Y %H:%M')
        except ValueError:
            logger.warning(f"Could not parse timestamp: {time_str}")
            return None

def _categorize_ship_type(ship_type: str) -> str:
    """Categorize ship type into broader categories."""
    if not ship_type: return "Unknown"
    ship_type = ship_type.lower()
    if "container" in ship_type: return "Container Ship"
    if "tanker" in ship_type: return "Tanker"
    if "cargo" in ship_type: return "General Cargo"
    if "bulk" in ship_type: return "Bulk Carrier"
    if "passenger" in ship_type: return "Passenger Ship"
    return "Other"

def _categorize_location(location: str) -> str:
    """Categorize vessel location."""
    if not location: return "Unknown"
    location = location.lower()
    if "terminal" in location: return "Terminal"
    if "anchorage" in location: return "Anchorage"
    if "buoy" in location: return "Buoy"
    return "Other"

def get_vessel_queue_analysis(vessel_data: pd.DataFrame) -> dict:
    """Perform in-depth analysis of the vessel queue."""
    if vessel_data.empty:
        return {}
    analysis = {}
    now = datetime.now()
    analysis['queue_length'] = len(vessel_data)
    vessel_data['waiting_time'] = (now - vessel_data['arrival_time']).dt.total_seconds() / 3600  # in hours
    analysis['average_waiting_time'] = vessel_data['waiting_time'].mean()
    analysis['queue_by_ship_type'] = vessel_data['ship_type'].value_counts().to_dict()
    analysis['queue_by_location'] = vessel_data['location'].value_counts().to_dict()
    analysis['priority_vessels'] = vessel_data[vessel_data['ship_type'] == 'Tanker'].to_dict('records')
    logger.info("Generated vessel queue analysis")
    return analysis

@st.cache_data(ttl=3600)
def load_all_vessel_data() -> pd.DataFrame:
    """Load and combine all vessel data from XML files."""
    vessel_data_dir = Path(__file__).parent / "raw_data"
    xml_files = [
        "Arrived_in_last_36_hours.xml",
        "Departed_in_last_36_hours.xml",
        "Expected_arrivals.xml",
        "Expected_departures.xml"
    ]
    all_vessel_data = []
    for file_name in xml_files:
        file_path = vessel_data_dir / file_name
        if file_path.exists():
            df = load_vessel_data_from_xml(file_path)
            all_vessel_data.append(df)
    
    if not all_vessel_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_vessel_data, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['name', 'timestamp']).sort_values(by='timestamp').reset_index(drop=True)
    logger.info(f"Loaded and combined data from {len(xml_files)} vessel files: {len(combined_df)} records")
    return combined_df

def load_vessel_data_from_xml(file_path: Path) -> pd.DataFrame:
    """Load vessel data from a single XML file."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        vessel_records = []
        for vessel_element in root.find_all('vessel'): # Generic vessel tag
            record = {
                'name': vessel_element.findtext('name'),
                'timestamp': _parse_vessel_timestamp(vessel_element.findtext('time')),
                'ship_type': _categorize_ship_type(vessel_element.findtext('type')),
                'location': _categorize_location(vessel_element.findtext('location')),
                'status': file_path.stem  # e.g., 'Arrived_in_last_36_hours'
            }
            vessel_records.append(record)
        return pd.DataFrame(vessel_records)
    except (ET.ParseError, FileNotFoundError) as e:
        logger.error(f"Error processing vessel XML {file_path}: {e}")
        return pd.DataFrame()

def get_comprehensive_vessel_analysis(vessel_df: pd.DataFrame) -> dict:
    """Provide a comprehensive analysis of vessel data."""
    if vessel_df.empty:
        return {}
    
    analysis = {}
    now = datetime.now()

    # Vessel counts by status
    analysis['vessel_counts'] = vessel_df['status'].value_counts().to_dict()

    # Turnaround time for departed vessels
    departed = vessel_df[vessel_df['status'] == 'Departed_in_last_36_hours']
    arrived = vessel_df[vessel_df['status'] == 'Arrived_in_last_36_hours']
    if not departed.empty and not arrived.empty:
        merged_df = pd.merge(arrived, departed, on='name', suffixes=['_arrival', '_departure'])
        merged_df['turnaround_hours'] = (merged_df['timestamp_departure'] - merged_df['timestamp_arrival']).dt.total_seconds() / 3600
        analysis['average_turnaround_time'] = merged_df['turnaround_hours'].mean()

    # Congestion analysis
    analysis['congestion_hotspots'] = vessel_df['location'].value_counts().nlargest(5).to_dict()

    logger.info("Generated comprehensive vessel analysis")
    return analysis

# --- Real-Time Data Management ---

@dataclass
class RealTimeDataConfig:
    vessel_data_dir: Path
    weather_api_key: str
    weather_city: str
    update_interval: int = 60

class RealTimeDataManager:
    def __init__(self, config: RealTimeDataConfig):
        self.config = config
        self.vessel_data = pd.DataFrame()
        self.weather_data = {}
        self._stop_event = threading.Event()
        self._thread = None

    def start_real_time_updates(self):
        """Start the background thread for real-time data updates."""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._real_time_loop, daemon=True)
            self._thread.start()
            logger.info("Real-time data manager started.")

    def stop_real_time_updates(self):
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.info("Real-time data manager stopped.")

    def _real_time_loop(self):
        """The main loop for updating data in the background."""
        while not self._stop_event.is_set():
            self._update_vessel_data()
            self._update_weather_data()
            time.sleep(self.config.update_interval)

    def _update_vessel_data(self):
        """Update vessel data from the latest files."""
        try:
            latest_data = load_all_vessel_data() # Re-use the loading function
            if not latest_data.empty:
                self.vessel_data = latest_data
                logger.info("Updated real-time vessel data.")
        except Exception as e:
            logger.error(f"Error updating real-time vessel data: {e}")

    def _update_weather_data(self):
        """Fetch the latest weather data from an API."""
        # This is a placeholder for a real weather API call
        try:
            # In a real implementation, you would use a library like `requests`
            # to fetch data from a weather API.
            self.weather_data = {
                'temperature': np.random.uniform(15, 30),
                'wind_speed': np.random.uniform(5, 25),
                'conditions': np.random.choice(['Clear', 'Cloudy', 'Rainy'])
            }
            logger.info("Updated real-time weather data.")
        except Exception as e:
            logger.error(f"Error updating real-time weather data: {e}")

    def get_latest_data(self) -> tuple[pd.DataFrame, dict]:
        """Return the latest cached data."""
        return self.vessel_data, self.weather_data

# --- UI and Visualization ---

def display_cargo_dashboard(cargo_stats: dict, time_series: dict, forecasts: dict):
    """Display the cargo-related dashboards."""
    st.header("Cargo Throughput Analysis")

    # Display container throughput
    st.subheader("Container Throughput (Monthly)")
    container_data = load_container_throughput()
    if not container_data.empty:
        st.line_chart(container_data[['seaborne_teus', 'river_teus', 'total_teus']])

    # Display cargo statistics
    st.subheader("Port Cargo Statistics")
    if time_series:
        selected_ts = st.selectbox("Select Time Series", list(time_series.keys()))
        if selected_ts:
            st.line_chart(time_series[selected_ts])

    # Display forecasts
    st.subheader("Cargo Forecasts")
    if forecasts:
        selected_forecast = st.selectbox("Select Forecast", list(forecasts.keys()))
        if selected_forecast:
            st.write(forecasts[selected_forecast])

def display_vessel_dashboard(vessel_data: pd.DataFrame):
    """Display the vessel-related dashboards."""
    st.header("Vessel Movement Analysis")

    if vessel_data.empty:
        st.warning("No vessel data available.")
        return

    # Display vessel queue analysis
    st.subheader("Vessel Queue Analysis")
    queue_analysis = get_vessel_queue_analysis(vessel_data)
    if queue_analysis:
        col1, col2 = st.columns(2)
        col1.metric("Vessels in Queue", queue_analysis.get('queue_length', 0))
        col2.metric("Average Wait Time (hours)", f"{queue_analysis.get('average_waiting_time', 0):.2f}")
        st.bar_chart(queue_analysis.get('queue_by_ship_type', {}))

    # Display comprehensive analysis
    st.subheader("Comprehensive Vessel Analysis")
    comp_analysis = get_comprehensive_vessel_analysis(vessel_data)
    if comp_analysis:
        st.write(comp_analysis)

def display_real_time_dashboard(data_manager: RealTimeDataManager):
    """Display the real-time monitoring dashboard."""
    st.header("Real-Time Port Monitoring")
    vessel_data, weather_data = data_manager.get_latest_data()

    if weather_data:
        st.subheader("Current Weather")
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{weather_data.get('temperature', 0):.1f}°C")
        col2.metric("Wind Speed", f"{weather_data.get('wind_speed', 0):.1f} km/h")
        col3.metric("Conditions", weather_data.get('conditions', 'N/A'))

    if not vessel_data.empty:
        st.subheader("Live Vessel Positions")
        st.map(vessel_data)

# --- Main Application ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="HK Port Digital Twin", layout="wide")
    st.title("Hong Kong Port Digital Twin")

    # --- Data Loading ---
    cargo_stats = load_port_cargo_statistics()
    time_series = get_time_series_data(cargo_stats)
    forecasts = forecast_cargo_throughput(time_series)
    vessel_data = load_all_vessel_data()

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Cargo Dashboard", "Vessel Dashboard", "Real-Time Monitoring"])

    if page == "Cargo Dashboard":
        display_cargo_dashboard(cargo_stats, time_series, forecasts)
    elif page == "Vessel Dashboard":
        display_vessel_dashboard(vessel_data)
    elif page == "Real-Time Monitoring":
        # For simplicity, we instantiate the manager here.
        # In a real app, this would be managed more carefully.
        config = RealTimeDataConfig(
            vessel_data_dir=Path(__file__).parent / "raw_data",
            weather_api_key="YOUR_API_KEY", # Replace with a real key
            weather_city="Hong Kong"
        )
        data_manager = RealTimeDataManager(config)
        data_manager.start_real_time_updates()
        display_real_time_dashboard(data_manager)

        # Ensure the thread is stopped when the app closes
        st.session_state['data_manager'] = data_manager

if __name__ == "__main__":
    main()
    # Stop the real-time manager when the script is re-run or stopped
    if 'data_manager' in st.session_state:
        st.session_state['data_manager'].stop_real_time_updates()
