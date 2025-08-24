"""
Main Streamlit application for the Hong Kong Port Digital Twin.

This application provides a comprehensive dashboard for monitoring and analyzing
port operations, including container throughput, cargo statistics, vessel movements,
and simulations.

**Data Sources:**
- Container Throughput: CSV files from the Hong Kong Marine Department.
- Port Cargo Statistics: CSV files from the Hong Kong Marine Department.
- Vessel Movements: Real-time data from the MarineTraffic API and historical data
  from XML files.

**Key Features:**
- Executive Dashboard: High-level overview of port performance.
- Container Throughput Analysis: Detailed analysis of container traffic.
- Port Cargo Statistics: Insights into cargo types and trends.
- Vessel Movement Monitoring: Real-time tracking of ships in the port.
- Simulation: Tools for simulating port operations and testing scenarios.

**Usage:**
To run the application, use the following command:
`streamlit run streamlit_app.py`
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import requests
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- General Settings ---
st.set_page_config(
    page_title="Hong Kong Port Digital Twin",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Directory and File Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
LOG_DIR.mkdir(exist_ok=True)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler(),
    ],
)

# --- Simulation Parameters ---
SIMULATION_SPEEDS = {"Slow": 1, "Medium": 5, "Fast": 10}

# --- Class Definitions ---

class ContainerHandler:
    """Handles loading and processing of container throughput data."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads container throughput data from a CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info(f"Successfully loaded data from {self.file_path}")
        except FileNotFoundError:
            logging.error(f"File not found: {self.file_path}")
            st.error(f"Error: Container data file not found at {self.file_path}")

    def get_summary(self):
        """Returns a summary of the container throughput data."""
        if self.data is not None:
            return self.data.describe()
        return None

class ShipManager:
    """Manages ship data, including loading, processing, and visualization."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.ships = None

    def load_ship_data(self):
        """Loads ship data from an XML file."""
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            ship_list = []
            for ship_elem in root.findall("ship"):
                ship_details = {
                    "name": ship_elem.find("name").text,
                    "type": ship_elem.find("type").text,
                    "imo": ship_elem.find("imo").text,
                }
                ship_list.append(ship_details)
            self.ships = pd.DataFrame(ship_list)
            logging.info(f"Successfully loaded ship data from {self.file_path}")
        except (FileNotFoundError, ET.ParseError) as e:
            logging.error(f"Error loading ship data: {e}")
            st.error("Error: Could not load ship data.")

    def display_ship_info(self):
        """Displays ship information in a Streamlit table."""
        if self.ships is not None:
            st.dataframe(self.ships)

class SimulationController:
    """Controls the simulation of port operations."""

    def __init__(self, speed="Medium"):
        self.speed = SIMULATION_SPEEDS[speed]
        self.time_step = 0

    def run_simulation(self):
        """Runs the simulation for a single time step."""
        self.time_step += 1
        # In a real simulation, this would update ship positions, etc.
        logging.info(f"Simulation running at step {self.time_step}")

    def get_status(self):
        """Returns the current status of the simulation."""
        return f"Time Step: {self.time_step}"

class ExecutiveDashboard:
    """Renders the main executive dashboard with KPIs and charts."""

    def __init__(self, container_data, ship_data):
        self.container_data = container_data
        self.ship_data = ship_data

    def display_kpis(self):
        """Displays key performance indicators."""
        if self.container_data is not None:
            total_throughput = self.container_data["Throughput (TEU)"].sum()
            st.metric("Total Container Throughput (TEU)", f"{total_throughput:,.0f}")

    def display_charts(self):
        """Displays charts for the executive dashboard."""
        if self.container_data is not None:
            fig = px.line(
                self.container_data,
                x="Year",
                y="Throughput (TEU)",
                title="Container Throughput Over Time",
            )
            st.plotly_chart(fig, use_container_width=True)

class MarineTrafficIntegration:
    """Integrates with the MarineTraffic API to get real-time vessel data."""

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.marinetraffic.com/v1/"

    def get_vessel_positions(self, bbox):
        """Retrieves vessel positions within a bounding box."""
        endpoint = "positions"
        url = f"{self.base_url}{endpoint}?api_key={self.api_key}&bbox={bbox}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from MarineTraffic API: {e}")
            return None

# --- Chart Rendering Functions ---

def render_container_throughput_chart(data):
    """Renders a chart for container throughput."""
    st.subheader("Container Throughput Analysis")
    fig = px.bar(
        data,
        x="Year",
        y="Throughput (TEU)",
        color="Type",
        title="Container Throughput by Type",
    )
    st.plotly_chart(fig, use_container_width=True)

def render_cargo_statistics_chart(data):
    """Renders a chart for port cargo statistics."""
    st.subheader("Port Cargo Statistics")
    fig = px.pie(
        data,
        names="Cargo Type",
        values="Tonnage",
        title="Cargo Tonnage Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

def render_vessel_arrival_chart(data):
    """Renders a chart for vessel arrivals."""
    st.subheader("Vessel Arrivals")
    arrivals_by_type = data["type"].value_counts().reset_index()
    arrivals_by_type.columns = ["Ship Type", "Number of Arrivals"]
    fig = px.bar(
        arrivals_by_type,
        x="Ship Type",
        y="Number of Arrivals",
        title="Vessel Arrivals by Ship Type",
    )
    st.plotly_chart(fig, use_container_width=True)

def render_vessel_departure_chart(data):
    """Renders a chart for vessel departures."""
    st.subheader("Vessel Departures")
    departures_by_type = data["type"].value_counts().reset_index()
    departures_by_type.columns = ["Ship Type", "Number of Departures"]
    fig = px.bar(
        departures_by_type,
        x="Ship Type",
        y="Number of Departures",
        title="Vessel Departures by Ship Type",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit application."""
    st.title("🚢 Hong Kong Port Digital Twin")

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Executive Dashboard",
            "Container Throughput",
            "Port Cargo Statistics",
            "Vessel Movements",
            "Simulation",
        ],
    )

    # --- Data Loading ---
    # Load data using the handlers
    container_handler = ContainerHandler(PROCESSED_DATA_DIR / "container_throughput.csv")
    container_handler.load_data()

    ship_manager = ShipManager(RAW_DATA_DIR / "ship_data.xml")
    ship_manager.load_ship_data()

    # --- Page Rendering ---
    if page == "Executive Dashboard":
        st.header("Executive Dashboard")
        dashboard = ExecutiveDashboard(container_handler.data, ship_manager.ships)
        dashboard.display_kpis()
        dashboard.display_charts()

    elif page == "Container Throughput":
        st.header("Container Throughput Analysis")
        if container_handler.data is not None:
            render_container_throughput_chart(container_handler.data)

    elif page == "Port Cargo Statistics":
        st.header("Port Cargo Statistics")
        # Placeholder for cargo statistics data
        cargo_data = pd.DataFrame(
            {
                "Cargo Type": ["Containerized", "Bulk", "Liquid"],
                "Tonnage": [150, 50, 30],
            }
        )
        render_cargo_statistics_chart(cargo_data)

    elif page == "Vessel Movements":
        st.header("Vessel Movement Monitoring")
        if ship_manager.ships is not None:
            st.subheader("Vessel Information")
            ship_manager.display_ship_info()

            # Placeholder for arrival/departure data
            render_vessel_arrival_chart(ship_manager.ships)
            render_vessel_departure_chart(ship_manager.ships)

    elif page == "Simulation":
        st.header("Port Operation Simulation")
        sim_speed = st.select_slider(
            "Simulation Speed", options=["Slow", "Medium", "Fast"]
        )
        sim_controller = SimulationController(speed=sim_speed)
        if st.button("Run Simulation Step"):
            sim_controller.run_simulation()
        st.write(sim_controller.get_status())

# --- Data Loading Functions (from data_loader.py) ---

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
        
        logging.info(f"Loaded container throughput data: {len(monthly_data)} monthly records")
        return monthly_data
        
    except Exception as e:
        logging.error(f"Error loading container throughput data: {e}")
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
        
        logging.info(f"Loaded annual container throughput data: {len(annual_data)} years")
        return annual_data
        
    except Exception as e:
        logging.error(f"Error loading annual container throughput data: {e}")
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
                logging.info(f"Loaded {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                
            except Exception as e:
                logging.error(f"Error loading {csv_file.name}: {e}")
                continue
        
        logging.info(f"Loaded {len(cargo_stats)} cargo statistics tables")
        return cargo_stats
        
    except Exception as e:
        logging.error(f"Error loading port cargo statistics: {e}")
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
            
        logging.info(f"Generated time series data for {len(time_series)} categories")
        return time_series
        
    except Exception as e:
        logging.error(f"Error generating time series data: {e}")
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
                rmse = np.sqrt(mse)
                
                # Calculate R-squared
                r2 = model.score(X, y)
                
                category_forecasts[column] = {
                    'historical_data': series.to_dict(),
                    'forecast_years': future_years.flatten().tolist(),
                    'forecast_values': predictions.tolist(),
                    'trend_slope': model.coef_[0],
                    'model_metrics': {
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2
                    }
                }
            
            forecasts[category] = category_forecasts
            
        logging.info(f"Generated forecasts for {len(forecasts)} categories")
        return forecasts
        
    except Exception as e:
        logging.error(f"Error generating forecasts: {e}")
        return {}

def get_enhanced_cargo_analysis() -> Dict[str, any]:
    """Enhanced cargo analysis focusing on Tables 1 & 2 with time series insights.
    
    Returns:
        Dict: Comprehensive analysis including trends, forecasts, and insights
    """
    try:
        # Load focused data
        cargo_stats = load_focused_cargo_statistics()
        
        if not cargo_stats:
            logging.warning("No focused cargo statistics data available")
            return {}
        
        # Generate time series data
        time_series = get_time_series_data(cargo_stats)
        
        # Generate forecasts
        forecasts = forecast_cargo_throughput(time_series, forecast_years=3)
        
        # Calculate trend analysis
        trends = _analyze_trends(time_series)
        
        # Calculate efficiency metrics
        efficiency_metrics = _calculate_focused_efficiency_metrics(cargo_stats)
        
        analysis = {
            'time_series_data': time_series,
            'forecasts': forecasts,
            'trend_analysis': trends,
            'efficiency_metrics': efficiency_metrics,
            'data_summary': {
                'tables_processed': len(cargo_stats),
                'analysis_timestamp': datetime.now().isoformat(),
                'forecast_horizon': '3 years'
            }
        }
        
        logging.info("Completed enhanced cargo analysis with forecasting")
        return analysis
        
    except Exception as e:
        logging.error(f"Error in enhanced cargo analysis: {e}")
        return {}

def _analyze_trends(time_series_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Analyze trends in time series data.
    
    Args:
        time_series_data: Time series data from get_time_series_data()
        
    Returns:
        Dict: Trend analysis for each category and metric
    """
    trends = {}
    
    try:
        for category, df in time_series_data.items():
            category_trends = {}
            
            for column in df.columns:
                series = df[column].dropna()
                
                if len(series) < 2:
                    continue
                    
                # Calculate trend metrics
                first_value = series.iloc[0]
                last_value = series.iloc[-1]
                total_change = last_value - first_value
                percent_change = (total_change / first_value) * 100 if first_value != 0 else 0
                
                # Calculate average annual growth rate
                years_span = series.index.max() - series.index.min()
                if years_span > 0:
                    cagr = ((last_value / first_value) ** (1 / years_span) - 1) * 100 if first_value > 0 else 0
                else:
                    cagr = 0
                
                # Determine trend direction
                if percent_change > 5:
                    trend_direction = 'increasing'
                elif percent_change < -5:
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
                
                category_trends[column] = {
                    'total_change': total_change,
                    'percent_change': percent_change,
                    'cagr': cagr,
                    'trend_direction': trend_direction,
                    'first_value': first_value,
                    'last_value': last_value,
                    'years_span': years_span
                }
            
            trends[category] = category_trends
            
        return trends
        
    except Exception as e:
        logging.error(f"Error analyzing trends: {e}")
        return {}

def _calculate_focused_efficiency_metrics(cargo_stats: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """Calculate efficiency metrics focused on Tables 1 & 2.
    
    Args:
        cargo_stats: Cargo statistics data
        
    Returns:
        Dict: Focused efficiency metrics
    """
    try:
        metrics = {}
        
        # Analyze shipment efficiency (Table 1)
        if 'Table_1_Eng' in cargo_stats:
            df1 = cargo_stats['Table_1_Eng']
            
            # Get 2023 data (latest year)
            latest_throughput_col = 'Port_cargo_throughput_2023'
            latest_percentage_col = 'Port_cargo_throughput_percentage_distribution_2023'
            
            if latest_throughput_col in df1.columns:
                # Calculate transhipment ratio
                tranship_row = df1[df1.iloc[:, 0].str.contains('Transhipment', na=False)]
                direct_row = df1[df1.iloc[:, 0].str.contains('Direct', na=False)]
                
                if not tranship_row.empty and not direct_row.empty:
                    tranship_pct = tranship_row[latest_percentage_col].iloc[0]
                    direct_pct = direct_row[latest_percentage_col].iloc[0]
                    
                    metrics['shipment_efficiency'] = {
                        'transhipment_ratio': tranship_pct,
                        'direct_shipment_ratio': direct_pct,
                        'transhipment_dominance': tranship_pct > direct_pct
                    }
        
        # Analyze transport efficiency (Table 2)
        if 'Table_2_Eng' in cargo_stats:
            df2 = cargo_stats['Table_2_Eng']
            
            # Get 2023 data
            latest_throughput_col = 'Port_cargo_throughput_2023'
            latest_percentage_col = 'Port_cargo_throughput_percentage_distribution_2023'
            
            if latest_throughput_col in df2.columns:
                # Calculate modal split
                seaborne_row = df2[df2.iloc[:, 0].str.contains('Seaborne', na=False)]
                river_row = df2[df2.iloc[:, 0].str.contains('River', na=False)]
                
                if not seaborne_row.empty and not river_row.empty:
                    seaborne_pct = seaborne_row[latest_percentage_col].iloc[0]
                    river_pct = river_row[latest_percentage_col].iloc[0]
                    
                    metrics['transport_efficiency'] = {
                        'seaborne_ratio': seaborne_pct,
                        'river_ratio': river_pct,
                        'modal_balance': abs(seaborne_pct - river_pct)
                    }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error calculating focused efficiency metrics: {e}")
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
        
        logging.info(f"{table_name} data quality: {completeness:.1f}% complete")
        
        return cleaned_df
        
    except Exception as e:
        logging.error(f"Error cleaning {table_name} data: {e}")
        return df

def load_arriving_ships() -> pd.DataFrame:
    """Load and process arriving ships data from XML file.
    
    Returns:
        pd.DataFrame: Processed arriving ships data with structured information
    """
    arriving_ships_xml = Path("/Users/Bhavesh/Documents/GitHub/Ports/Ports/raw_data/Expected_arrivals.xml")
    logging.info(f"Attempting to load arriving ships from: {arriving_ships_xml}")
    
    try:
        # Check if XML file exists
        if not arriving_ships_xml.exists():
            logging.error(f"Arriving ships XML file does not exist at {arriving_ships_xml}")
            return pd.DataFrame()
        
        if arriving_ships_xml.stat().st_size == 0:
            logging.warning(f"Arriving ships data file is empty: {arriving_ships_xml}")
            return pd.DataFrame()
        
        # Read and clean XML content (skip comment lines)
        with open(arriving_ships_xml, 'r', encoding='utf-8') as file:
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
        content = '
'.join(xml_lines)
        
        logging.info("Parsing arriving ships XML file...")
        # Parse cleaned XML content
        root = ET.fromstring(content)
        logging.info("Successfully parsed arriving ships XML file.")
        
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred in the main application: {e}")
        st.error("An unexpected error occurred. Please check the logs for details.")