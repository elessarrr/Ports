import sys
import os
import time
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
import numpy as np
import simpy

# Add the project root to the Python path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hk_port_digital_twin.src.utils.data_loader import RealTimeDataConfig, get_real_time_manager, load_container_throughput, load_vessel_arrivals, load_berth_configurations
from hk_port_digital_twin.config.settings import SIMULATION_CONFIG, get_enhanced_simulation_config
from hk_port_digital_twin.src.core.port_simulation import PortSimulation
from hk_port_digital_twin.src.core.simulation_controller import SimulationController
from hk_port_digital_twin.src.core.berth_manager import BerthManager
from hk_port_digital_twin.src.scenarios import ScenarioManager, list_available_scenarios
from hk_port_digital_twin.src.utils.visualization import create_kpi_summary_chart, create_port_layout_chart, create_ship_queue_chart, create_berth_utilization_chart, create_throughput_timeline, create_waiting_time_distribution
# Weather integration disabled for feature removal
# from hk_port_digital_twin.src.utils.weather_integration import HKObservatoryIntegration
HKObservatoryIntegration = None  # Disabled
from hk_port_digital_twin.src.utils.data_loader import load_focused_cargo_statistics, get_enhanced_cargo_analysis, get_time_series_data

try:
    from hk_port_digital_twin.src.dashboard.marine_traffic_integration import MarineTrafficIntegration
except ImportError:
    MarineTrafficIntegration = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')


def load_sample_data(use_real_throughput_data=True):
    """Load sample data"""
    queue_data = {
        'berth_id': ['Berth_A1', 'Berth_A2', 'Berth_A3', 'Berth_A4', 'Berth_B1', 'Berth_B2', 'Berth_C1', 'Berth_C2'],
        'name': ['Berth A1', 'Berth A2', 'Berth A3', 'Berth A4', 'Berth B1', 'Berth B2', 'Berth C1', 'Berth C2'],
        'x': [1, 2, 3, 4, 1, 2, 3, 4],
        'y': [1, 1, 1, 1, 2, 2, 3, 3],
        'status': ['occupied', 'available', 'occupied', 'maintenance', 'occupied', 'available', 'occupied', 'available'],
        'ship_id': ['SHIP_001', None, 'SHIP_003', None, 'SHIP_005', None, 'SHIP_007', None],
        'utilization': [85, 0, 92, 0, 78, 0, 88, 0],
        'berth_type': ['container', 'container', 'container', 'container', 'bulk', 'bulk', 'mixed', 'mixed'],
        'crane_count': [4, 3, 4, 2, 2, 2, 3, 3],
        'max_capacity_teu': [5000, 4000, 5000, 3000, 6000, 6000, 4500, 4500],
        'is_occupied': [True, False, True, False, True, False, True, False]
    }
    
    # Load real container throughput data instead of simulated data
    if use_real_throughput_data:
        try:
            timeline_data = load_container_throughput()
            # The data already has a datetime index, so we use that as 'time'
            timeline_data = timeline_data.reset_index()
            timeline_data = timeline_data.rename(columns={'Date': 'time'})
            # Convert TEUs from thousands to actual numbers for better visualization
            timeline_data['seaborne_teus'] = timeline_data['seaborne_teus'] * 1000
            timeline_data['river_teus'] = timeline_data['river_teus'] * 1000
            timeline_data['total_teus'] = timeline_data['total_teus'] * 1000
        except Exception as e:
            # Fallback to sample data if real data loading fails
            print(f"Warning: Could not load real throughput data: {e}")
            timeline_data = {
                'time': pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                     end=datetime.now(), freq='h'),
                'containers_processed': np.random.randint(10, 100, 25),
                'ships_processed': np.random.randint(1, 8, 25)
            }
            timeline_data = pd.DataFrame(timeline_data)
    else:
        # Fallback to sample data if function not available
        timeline_data = {
            'time': pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                 end=datetime.now(), freq='h'),
            'containers_processed': np.random.randint(10, 100, 25),
            'ships_processed': np.random.randint(1, 8, 25)
        }
        timeline_data = pd.DataFrame(timeline_data)
    
    # Sample ship queue data (ships waiting for berths)
    ship_queue_data = {
        'ship_id': ['SHIP_001', 'SHIP_002', 'SHIP_003'],
        'name': ['MSC Lucinda', 'COSCO Shanghai', 'Evergreen Marine'],
        'ship_type': ['container', 'container', 'bulk'],
        'arrival_time': [datetime.now() - timedelta(hours=2), datetime.now() - timedelta(hours=1), datetime.now() - timedelta(minutes=30)],
        'containers': [150, 200, 120],
        'size_teu': [8000, 12000, 6500],
        'waiting_time': [2.5, 1.8, 3.2],
        'priority': ['high', 'medium', 'low']
    }
    
    # Sample waiting time data
    waiting_data = {
        'ship_id': [f'SHIP_{i:03d}' for i in range(1, 21)],
        'waiting_time': np.random.exponential(2, 20),
        'ship_type': np.random.choice(['container', 'bulk', 'mixed'], 20)
    }
    
    # Sample KPI data
    kpi_data = {
        'metric': ['Average Waiting Time', 'Berth Utilization', 'Throughput Rate', 'Queue Length'],
        'value': [2.5, 75, 85, 3],
        'unit': ['hours', '%', 'containers/hour', 'ships'],
        'target': [2.0, 80, 90, 2],
        'status': ['warning', 'good', 'good', 'warning']
    }
    
    return {
        'berths': pd.DataFrame(queue_data),
        'queue': pd.DataFrame(ship_queue_data),
        'timeline': timeline_data,  # Now using real data
        'waiting': pd.DataFrame(waiting_data),
        'kpis': pd.DataFrame(kpi_data),
        'vessel_queue_analysis': {
            'total_vessels_waiting': len(ship_queue_data['ship_id']),
            'average_waiting_time': np.mean(ship_queue_data['waiting_time']),
            'queue_by_type': {'container': 2, 'bulk': 1},
            'priority_distribution': {'high': 1, 'medium': 1, 'low': 1}
        }
    }


def get_real_berth_data():
    """Get real-time berth data from BerthManager"""
    if BerthManager and simpy:
        try:
            # Create a simulation environment and berth manager for real data
            env = simpy.Environment()
            berth_configs = load_berth_configurations()
            berth_manager = BerthManager(env, berth_configs)
            
            # Get berth statistics
            berth_stats = berth_manager.get_berth_statistics()
            
            # Convert berth data to DataFrame format
            berths_list = []
            for berth_id, berth in berth_manager.berths.items():
                berths_list.append({
                    'berth_id': berth_id,
                    'name': berth.name,  # Use actual berth name from CSV
                    'status': 'occupied' if berth.is_occupied else 'available',
                    'ship_id': berth.current_ship.ship_id if berth.current_ship else None,
                    'berth_type': berth.berth_type,
                    'crane_count': berth.crane_count,
                    'max_capacity_teu': berth.max_capacity_teu,
                    'is_occupied': berth.is_occupied,
                    'utilization': 100 if berth.is_occupied else 0,
                    'x': hash(berth_id) % 5 + 1,  # Simple positioning
                    'y': hash(berth_id) % 3 + 1
                })
            
            berths_df = pd.DataFrame(berths_list)
            
            # Add berth statistics
            berth_metrics = {
                'total_berths': berth_stats['total_berths'],
                'occupied_berths': berth_stats['occupied_berths'],
                'available_berths': berth_stats['available_berths'],
                'utilization_rate': berth_stats['overall_utilization_rate'],
                'berth_types': berth_stats['berth_types']
            }
            
            return berths_df, berth_metrics
            
        except Exception as e:
            print(f"Warning: Could not get real berth data: {e}")
    
    # Fallback to sample data
    data = load_sample_data()
    berth_metrics = {
        'total_berths': len(data['berths']),
        'occupied_berths': len(data['berths'][data['berths']['status'] == 'occupied']),
        'available_berths': len(data['berths'][data['berths']['status'] == 'available']),
        'utilization_rate': len(data['berths'][data['berths']['status'] == 'occupied']) / len(data['berths']) * 100,
        'berth_types': data['berths']['berth_type'].value_counts().to_dict()
    }
    return data['berths'], berth_metrics


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    logging.info("Initializing session state...")
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'simulation_controller' not in st.session_state:
        st.session_state.simulation_controller = None
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    # Initialize real-time data manager
    if 'real_time_manager' not in st.session_state:
        if RealTimeDataConfig and get_real_time_manager:
            try:
                # Configure real-time data manager
                config = RealTimeDataConfig(
                    enable_weather_integration=True,
                    enable_file_monitoring=True,
                    vessel_update_interval=15,   # 15 seconds
                    weather_update_interval=300,  # 5 minutes
                    auto_reload_on_file_change=True,
                    cache_duration=60             # 1 minute
                )
                
                # Get and start the real-time manager
                manager = get_real_time_manager(config)
                manager.start_real_time_updates()
                st.session_state.real_time_manager = manager
                
            except Exception as e:
                print(f"Warning: Could not initialize real-time data manager: {e}")
                st.session_state.real_time_manager = None
        else:
            st.session_state.real_time_manager = None


def create_sidebar():
    """Create sidebar with simulation controls"""
    st.sidebar.title("🚢 Port Control Panel")
    
    # Scenario Management
    st.sidebar.subheader("📋 Scenario Management")
    
    # Initialize scenario manager if not in session state
    if 'scenario_manager' not in st.session_state:
        st.session_state.scenario_manager = ScenarioManager()
    
    scenario_manager = st.session_state.scenario_manager
    
    # Scenario selection
    available_scenarios = list_available_scenarios()
    current_scenario = scenario_manager.get_current_scenario()
    
    selected_scenario = st.sidebar.selectbox(
        "Select Scenario",
        available_scenarios,
        index=available_scenarios.index(current_scenario) if current_scenario in available_scenarios else 0,
        help="Choose operational scenario based on expected conditions"
    )
    
    # Update scenario if changed
    if selected_scenario != current_scenario:
        scenario_manager.set_scenario(selected_scenario)
        st.sidebar.success(f"Scenario changed to: {selected_scenario}")
    
    # Auto-detection toggle
    auto_detect = st.sidebar.checkbox(
        "Auto-detect scenario",
        value=scenario_manager.auto_detection_enabled,
        help="Automatically select scenario based on current date and historical patterns"
    )
    
    if auto_detect != scenario_manager.auto_detection_enabled:
        if auto_detect:
            scenario_manager.enable_auto_detection()
        else:
            scenario_manager.disable_auto_detection()
    
    # Display scenario info
    scenario_info = scenario_manager.get_scenario_info()
    if scenario_info:
        with st.sidebar.expander("📊 Scenario Details", expanded=False):
            st.write(f"**Description:** {scenario_info.get('description', 'N/A')}")
            st.write(f"**Ship Arrival Rate:** {scenario_info.get('ship_arrival_multiplier', 1.0):.1f}x")
            st.write(f"**Container Volume:** {scenario_info.get('container_volume_multiplier', 1.0):.1f}x")
            st.write(f"**Processing Efficiency:** {scenario_info.get('processing_efficiency_factor', 1.0):.1f}x")
    
    # Display historical simulation parameters
    try:
        enhanced_config = get_enhanced_simulation_config()
        if enhanced_config.get('enhanced_with_historical_data', False):
            with st.sidebar.expander("📈 Historical Data Integration", expanded=False):
                metadata = enhanced_config.get('historical_data_metadata', {})
                st.write(f"**Data Period:** {metadata.get('data_period', 'Unknown')}")
                st.write(f"**Years of Data:** {metadata.get('years_of_data', 0):.1f}")
                st.write(f"**Data Points:** {metadata.get('total_data_points', 0):,}")
                st.write(f"**Trend Direction:** {metadata.get('trend_direction', 'stable').title()}")
                
                # Show enhanced parameters
                if 'seasonal_patterns' in enhanced_config:
                    seasonal = enhanced_config['seasonal_patterns']
                    st.write(f"**Peak Multiplier:** {seasonal.get('peak_multiplier', 1.0):.2f}x")
                    st.write(f"**Low Multiplier:** {seasonal.get('low_multiplier', 1.0):.2f}x")
                
                if 'historical_ship_type_distribution' in enhanced_config:
                    ship_dist = enhanced_config['historical_ship_type_distribution']
                    st.write("**Ship Type Distribution:**")
                    for ship_type, percentage in ship_dist.items():
                        st.write(f"  - {ship_type.title()}: {percentage:.1%}")
        else:
            with st.sidebar.expander("📈 Historical Data Integration", expanded=False):
                st.write("⚠️ Historical data enhancement not available")
                st.write("Using default simulation parameters")
    except Exception as e:
        with st.sidebar.expander("📈 Historical Data Integration", expanded=False):
            st.write("⚠️ Error loading historical parameters")
            st.write(f"Error: {str(e)}")
    
    st.sidebar.divider()
    
    # Simulation parameters
    st.sidebar.subheader("⚙️ Simulation Settings")
    duration = st.sidebar.slider("Duration (hours)", 1, 168, SIMULATION_CONFIG['default_duration'])
    
    # Get scenario-adjusted arrival rate
    base_arrival_rate = float(SIMULATION_CONFIG['ship_arrival_rate'])
    scenario_params = scenario_manager.get_current_parameters()
    if scenario_params:
        adjusted_arrival_rate = base_arrival_rate * scenario_params.arrival_rate_multiplier
    else:
        adjusted_arrival_rate = base_arrival_rate
    
    arrival_rate = st.sidebar.slider(
        "Ship Arrival Rate (ships/hour)", 
        0.5, 5.0, 
        adjusted_arrival_rate,
        help=f"Base rate: {base_arrival_rate}, Scenario multiplier: {scenario_params.arrival_rate_multiplier if scenario_params else 1.0}x"
    )
    
    # Simulation controls
    st.sidebar.subheader("Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start", disabled=st.session_state.simulation_running or not PortSimulation):
            if PortSimulation and SimulationController:
                # Initialize simulation controller with historical parameters
                config = get_enhanced_simulation_config()
                config['ship_arrival_rate'] = arrival_rate
                
                simulation = PortSimulation(config)
                
                # Set scenario in simulation
                if hasattr(simulation, 'set_scenario'):
                    simulation.set_scenario(selected_scenario)
                
                st.session_state.simulation_controller = SimulationController(simulation)
                st.session_state.simulation_controller.start(duration)
                st.session_state.simulation_running = True
                st.success(f"Simulation started with scenario: {selected_scenario}!")
            else:
                st.error("Simulation components not available")
    
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.simulation_running):
            if st.session_state.simulation_controller:
                st.session_state.simulation_controller.stop()
                st.session_state.simulation_running = False
                st.success("Simulation stopped!")
    
    # Display simulation status
    if st.session_state.simulation_controller:
        st.sidebar.subheader("Status")
        controller = st.session_state.simulation_controller
        
        if controller.is_running():
            st.sidebar.success("🟢 Running")
            progress = controller.get_progress_percentage()
            st.sidebar.progress(progress / 100)
            st.sidebar.text(f"Progress: {progress:.1f}%")
        elif controller.is_completed():
            st.sidebar.info("✅ Completed")
        else:
            st.sidebar.warning("⏸️ Stopped")
    
    return duration, arrival_rate


def main():
    """Main dashboard application"""
    # Page configuration
    st.set_page_config(
        page_title="Hong Kong Port Digital Twin",
        page_icon="🚢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🏗️ Hong Kong Port Digital Twin Dashboard")
    st.markdown("Real-time visualization and control of port operations")
    
    # Sidebar controls
    duration, arrival_rate = create_sidebar()
    
    # Load sample data (in a real implementation, this would come from the simulation)
    data = load_sample_data()
    
    # Auto-refresh every 5 seconds when simulation is running
    if st.session_state.simulation_running:
        time.sleep(5)
        st.rerun()
    
    # Main dashboard layout
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["📊 Overview", "🚢 Ships & Berths", "📈 Analytics", "📦 Cargo Statistics", "🌊 Live Map", "🛳️ Live Vessels", "🏗️ Live Berths", "🎯 Scenarios", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Port Overview")
        
        # KPI Summary
        # Center the forecast chart
        col1, col2, col3 = st.columns([0.5, 2, 0.5])
        
        with col2:
            # Load real forecast data if available
            try:
                if get_enhanced_cargo_analysis is not None:
                    cargo_analysis = get_enhanced_cargo_analysis()
                    forecasts = cargo_analysis.get('forecasts', {})
                else:
                    forecasts = {}
            except Exception:
                forecasts = {}
            if forecasts:
                # Convert forecast data to expected format for create_kpi_summary_chart
                kpi_dict = {}
                forecast_categories = ['direct_shipment', 'transhipment', 'seaborne', 'river']
                
                for category in forecast_categories:
                    if category in forecasts:
                        # Get the first forecast data (assuming it's the main metric)
                        category_data = forecasts[category]
                        if category_data:
                            first_metric = list(category_data.keys())[0]
                            forecast_info = category_data[first_metric]
                            
                            # Ensure years are integers
                            hist_years = [int(year) for year in forecast_info.get('historical_data', {}).keys()]
                            hist_values = list(forecast_info.get('historical_data', {}).values())
                            forecast_years = [int(year) for year in forecast_info.get('forecast_years', [])]
                            forecast_values = forecast_info.get('forecast_values', [])
                            
                            kpi_dict[category] = {
                                'historical_years': hist_years,
                                'historical_values': hist_values,
                                'forecast_years': forecast_years,
                                'forecast_values': forecast_values
                            }
                
                # Add model performance data
                model_performance = {}
                for category in forecast_categories:
                    if category in forecasts:
                        category_data = forecasts[category]
                        if category_data:
                            first_metric = list(category_data.keys())[0]
                            metrics = category_data[first_metric].get('model_metrics', {})
                            model_performance[category] = {
                                'r2_score': metrics.get('r2', 0),
                                'mae': metrics.get('mae', 0)
                            }
                
                kpi_dict['model_performance'] = model_performance
                create_kpi_summary_chart(kpi_dict)
            else:
                # Fallback to sample data if no forecasts available
                kpi_dict = {
                    'average_waiting_time': 2.5,
                    'average_berth_utilization': 0.75,
                    'total_ships_processed': 85,
                    'total_containers_processed': 1200,
                    'average_queue_length': 3
                }
                create_kpi_summary_chart(kpi_dict)
        
        # Metrics section
        st.subheader("📊 Key Metrics")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Enhanced metrics with real vessel data
            vessel_analysis = data.get('vessel_queue_analysis', {})
            
            if vessel_analysis:
                active_vessels = vessel_analysis.get('current_status', {}).get('active_vessels', 0)
                st.metric("Live Vessels", active_vessels)
            else:
                st.metric("Active Ships", len(data['queue']))
            
            st.metric("Available Berths", len(data['berths'][data['berths']['status'] == 'available']))
            
            # Show recent arrivals if available
            if vessel_analysis and 'recent_activity' in vessel_analysis:
                arrivals_24h = vessel_analysis['recent_activity'].get('arrivals_last_24h', 0)
                st.metric("24h Arrivals", arrivals_24h)
            else:
                st.metric("Avg Waiting Time", "2.5 hrs")
            
            st.metric("Utilization Rate", "75%")
        
        # Port Layout
        st.subheader("Port Layout")
        if create_port_layout_chart is not None:
            fig_layout = create_port_layout_chart(data['berths'])
            st.plotly_chart(fig_layout, use_container_width=True, key="port_layout_chart")
        else:
            st.info("Port layout visualization not available. Please ensure visualization module is properly installed.")
    
    with tab2:
        st.subheader("Ships & Berths")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ship Queue")
            # Convert DataFrame to list of dictionaries for visualization
            queue_list = data['queue'].to_dict('records')
            if create_ship_queue_chart is not None:
                fig_queue = create_ship_queue_chart(queue_list)
                st.plotly_chart(fig_queue, use_container_width=True, key="main_ship_queue_chart")
            else:
                st.warning("Ship queue visualization not available. Please check visualization module import.")
                st.dataframe(data['queue'], use_container_width=True)
            
            # Ship queue table
            st.dataframe(data['queue'], use_container_width=True)
        
        with col2:
            st.subheader("Berth Utilization")
            # Convert DataFrame to dictionary for visualization
            berth_util_dict = dict(zip(data['berths']['berth_id'], data['berths']['utilization']))
            if create_berth_utilization_chart is not None:
                fig_berth = create_berth_utilization_chart(berth_util_dict)
                st.plotly_chart(fig_berth, use_container_width=True, key="main_berth_utilization_chart")
            else:
                st.warning("Berth utilization visualization not available. Please check visualization module import.")
                st.dataframe(data['berths'], use_container_width=True)
            
            # Berth status table
            st.dataframe(data['berths'], use_container_width=True)
    
    with tab3:
        st.subheader("Analytics")
        
        # Data Export Section
        st.subheader("📥 Data Export")
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        
        with export_col1:
            # Export berth data
            berth_csv = data['berths'].to_csv(index=False)
            st.download_button(
                label="📊 Export Berth Data",
                data=berth_csv,
                file_name=f"berth_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            # Export queue data
            queue_csv = data['queue'].to_csv(index=False)
            st.download_button(
                label="🚢 Export Queue Data",
                data=queue_csv,
                file_name=f"queue_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with export_col3:
            # Export timeline data
            timeline_csv = data['timeline'].to_csv(index=False)
            st.download_button(
                label="📈 Export Timeline Data",
                data=timeline_csv,
                file_name=f"timeline_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with export_col4:
            # Export all data as JSON
            import json
            export_data = {
                'berths': data['berths'].to_dict('records'),
                'queue': data['queue'].to_dict('records'),
                'timeline': data['timeline'].to_dict('records'),
                'waiting': data['waiting'].to_dict('records'),
                'kpis': data['kpis'].to_dict('records'),
                'export_timestamp': datetime.now().isoformat()
            }
            json_data = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="📋 Export All (JSON)",
                data=json_data,
                file_name=f"port_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Throughput Timeline")
            if create_throughput_timeline is not None:
                fig_timeline = create_throughput_timeline(data['timeline'])
                st.plotly_chart(fig_timeline, use_container_width=True, key="main_throughput_timeline_chart")
            else:
                st.warning("Throughput timeline visualization not available. Please check visualization module import.")
                st.dataframe(data['timeline'], use_container_width=True)
        
        with col2:
            st.subheader("Waiting Time Distribution")
            # Convert DataFrame to list for visualization
            waiting_times_list = data['waiting']['waiting_time'].tolist()
            if create_waiting_time_distribution is not None:
                fig_waiting = create_waiting_time_distribution(waiting_times_list)
                st.plotly_chart(fig_waiting, use_container_width=True, key="main_waiting_time_chart")
            else:
                st.warning("Waiting time distribution visualization not available. Please check visualization module import.")
                st.dataframe(data['waiting'], use_container_width=True)
    
    with tab4:
        st.subheader("📦 Port Cargo Statistics")
        st.markdown("Comprehensive analysis of Hong Kong port cargo throughput data with time series analysis and forecasting")
        
        # Load enhanced cargo analysis
        try:
            if load_focused_cargo_statistics is None or get_enhanced_cargo_analysis is None or get_time_series_data is None:
                st.warning("⚠️ Cargo statistics analysis not available")
                st.info("Please ensure the data loader module is properly installed and configured.")
                focused_data = {}
                cargo_analysis = {}
                time_series_data = {}
            else:
                with st.spinner("Loading enhanced cargo statistics..."):
                    # Load focused data (Tables 1 & 2)
                    focused_data = load_focused_cargo_statistics()
                    
                    # Get enhanced analysis with forecasting
                    cargo_analysis = get_enhanced_cargo_analysis()
                    
                    # Get time series data for visualization
                    time_series_data = get_time_series_data(focused_data)
        except Exception as e:
            st.error(f"Error loading cargo statistics: {str(e)}")
            st.info("Please ensure the Port Cargo Statistics CSV files are available in the raw_data directory.")
            focused_data = {}
            cargo_analysis = {}
            time_series_data = {}
        
        # Display data summary
        st.subheader("📊 Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tables_processed = cargo_analysis.get('data_summary', {}).get('tables_processed', 0)
            st.metric("Tables Processed", tables_processed)
        with col2:
            st.metric("Analysis Status", "✅ Complete" if cargo_analysis else "❌ Failed")
        with col3:
            analysis_sections = len([k for k in cargo_analysis.keys() if k.endswith('_analysis')])
            st.metric("Analysis Sections", analysis_sections)
        with col4:
            timestamp = cargo_analysis.get('data_summary', {}).get('analysis_timestamp', datetime.now().isoformat())
            st.metric("Analysis Date", timestamp[:10] if timestamp else datetime.now().strftime("%Y-%m-%d"))

        # Create tabs for different analysis sections
        cargo_tab1, cargo_tab2, cargo_tab3, cargo_tab4, cargo_tab5, cargo_tab6 = st.tabs([
            "📊 Shipment Types", "🚢 Transport Modes", "📈 Time Series", "🔮 Forecasting", "📦 Cargo Types", "📍 Locations"
        ])

        with cargo_tab1:
            # Center the whole tab content
            main_col1, main_col2, main_col3 = st.columns([0.1, 0.8, 0.1])
            with main_col2:
                st.subheader("Shipment Type Analysis")
                
                # Get shipment type data from time series
                shipment_ts = time_series_data.get('shipment_types', pd.DataFrame())
                
                if not shipment_ts.empty:
                    # Get latest year data (2023)
                    latest_data = shipment_ts.loc[2023] if 2023 in shipment_ts.index else shipment_ts.iloc[-1]
                    total = latest_data['Overall']
                    direct_pct = (latest_data['Direct shipment cargo'] / total) * 100
                    tranship_pct = (latest_data['Transhipment cargo'] / total) * 100

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**2023 Throughput Data**")
                        breakdown_data = {
                            'Direct Shipment': latest_data['Direct shipment cargo'],
                            'Transhipment': latest_data['Transhipment cargo']
                        }
                        breakdown_df = pd.DataFrame(list(breakdown_data.items()),
                                                    columns=['Shipment Type', 'Throughput (000 tonnes)'])
                        st.dataframe(breakdown_df, use_container_width=True)
                    
                    with col2:
                        st.write("**Percentage Distribution**")
                        st.metric("Direct Shipment", f"{direct_pct:.1f}%")
                        st.metric("Transhipment", f"{tranship_pct:.1f}%")
                        
                    # Show time series chart
                    st.write("**Historical Trends (2014-2023)**")
                    chart_data = shipment_ts[['Direct shipment cargo', 'Transhipment cargo']]
                    
                    # Create plotly chart for better control over formatting
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    
                    # Ensure years are integers
                    years = [int(year) for year in chart_data.index]
                    
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=chart_data['Direct shipment cargo'],
                        mode='lines+markers',
                        name='Direct Shipment Cargo',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=chart_data['Transhipment cargo'],
                        mode='lines+markers',
                        name='Transhipment Cargo',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Throughput (000 tonnes)",
                        height=400,
                        xaxis=dict(tickmode='linear', dtick=1),  # Force integer years
                        margin=dict(l=50, r=50, t=50, b=50)  # Center the chart
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="shipment_trends_chart")
                else:
                    st.info("No shipment type analysis data available.")
                    st.warning("Please ensure the Port Cargo Statistics CSV files are available in the raw_data directory.")

        with cargo_tab2:
            st.subheader("Transport Mode Analysis")
            
            # Get transport mode data from time series
            transport_ts = time_series_data.get('transport_modes', pd.DataFrame())
            
            if not transport_ts.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**2023 Transport Data**")
                    # Get latest year data (2023)
                    latest_data = transport_ts.loc[2023] if 2023 in transport_ts.index else transport_ts.iloc[-1]
                    
                    transport_breakdown = {
                        'Waterborne': latest_data['Waterborne'],
                        'Seaborne': latest_data['Seaborne'],
                        'River': latest_data['River']
                    }
                    transport_df = pd.DataFrame(list(transport_breakdown.items()), 
                                                columns=['Transport Mode', 'Throughput (000 tonnes)'])
                    st.dataframe(transport_df, use_container_width=True)
                    
                    # Calculate percentages
                    total = sum(transport_breakdown.values())
                    waterborne_pct = (latest_data['Waterborne'] / total) * 100
                    seaborne_pct = (latest_data['Seaborne'] / total) * 100
                    river_pct = (latest_data['River'] / total) * 100
                
                with col2:
                    st.write("**Modal Split Percentage**")
                    st.metric("Waterborne", f"{waterborne_pct:.1f}%")
                    st.metric("Seaborne", f"{seaborne_pct:.1f}%")
                    st.metric("River", f"{river_pct:.1f}%")
                    
                # Show time series chart
                st.write("**Historical Trends**")
                chart_data = transport_ts[['Waterborne', 'Seaborne', 'River']]
                
                # Create plotly chart for better control over formatting
                import plotly.graph_objects as go
                fig = go.Figure()
                
                # Ensure years are integers
                years = [int(year) for year in chart_data.index]
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=chart_data['Waterborne'],
                    mode='lines+markers',
                    name='Waterborne',
                    line=dict(color='purple')
                ))
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=chart_data['Seaborne'],
                    mode='lines+markers',
                    name='Seaborne',
                    line=dict(color='green')
                ))
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=chart_data['River'],
                    mode='lines+markers',
                    name='River',
                    line=dict(color='orange')
                ))
                
                fig.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Throughput (000 tonnes)",
                    height=400,
                    xaxis=dict(tickmode='linear', dtick=1),  # Force integer years
                    margin=dict(l=50, r=50, t=50, b=50)  # Center the chart
                )
                
                # Center the chart
                chart_col1, chart_col2, chart_col3 = st.columns([0.1, 0.8, 0.1])
                with chart_col2:
                    st.plotly_chart(fig, use_container_width=True, key="transport_trends_chart")
                
            else:
                st.info("No transport mode analysis data available")
                st.warning("Please ensure the Port Cargo Statistics CSV files are available in the raw_data directory.")

        with cargo_tab3:
            st.subheader("Time Series Analysis")
            
            if time_series_data:
                # Display time series charts
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    # Create subplots for different metrics
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            'Direct Shipment Trends', 'Transhipment Trends',
                            'Transport Mode Trends', 'River Transport Trends'
                        ),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Extract time series data
                    shipment_ts = time_series_data.get('shipment_types', pd.DataFrame())
                    transport_ts = time_series_data.get('transport_modes', pd.DataFrame())
                    
                    if not shipment_ts.empty:
                        # Ensure years are integers
                        years = [int(year) for year in shipment_ts.index.tolist()]
                        direct_values = shipment_ts['Direct shipment cargo'].tolist()
                        tranship_values = shipment_ts['Transhipment cargo'].tolist()
                        
                        # Direct shipment trend
                        fig.add_trace(
                            go.Scatter(x=years, y=direct_values, mode='lines+markers',
                                     name='Direct Shipment', line=dict(color='blue')),
                            row=1, col=1
                        )
                        
                        # Transhipment trend
                        fig.add_trace(
                            go.Scatter(x=years, y=tranship_values, mode='lines+markers',
                                     name='Transhipment', line=dict(color='red')),
                            row=1, col=2
                        )
                    
                    if not transport_ts.empty:
                        # Ensure years are integers
                        years = [int(year) for year in transport_ts.index.tolist()]
                        waterborne_values = transport_ts['Waterborne'].tolist()
                        seaborne_values = transport_ts['Seaborne'].tolist()
                        river_values = transport_ts['River'].tolist()
                        
                        # Waterborne transport trend
                        fig.add_trace(
                            go.Scatter(x=years, y=waterborne_values, mode='lines+markers',
                                     name='Waterborne', line=dict(color='purple')),
                            row=2, col=1
                        )
                        
                        # Seaborne transport trend
                        fig.add_trace(
                            go.Scatter(x=years, y=seaborne_values, mode='lines+markers',
                                     name='Seaborne', line=dict(color='green')),
                            row=2, col=1
                        )
                        
                        # River transport trend
                        fig.add_trace(
                            go.Scatter(x=years, y=river_values, mode='lines+markers',
                                     name='River', line=dict(color='orange')),
                            row=2, col=2
                        )
                    
                    fig.update_layout(
                        height=600,
                        title_text="Port Cargo Time Series Analysis (2014-2023)",
                        showlegend=False,
                        margin=dict(l=50, r=50, t=50, b=50)  # Center the chart
                    )
                    
                    fig.update_xaxes(title_text="Year", tickmode='linear', dtick=1)  # Force integer years
                    fig.update_yaxes(title_text="Throughput (000 tonnes)")
                    
                    # Center the chart
                    chart_col1, chart_col2, chart_col3 = st.columns([0.1, 0.8, 0.1])
                    with chart_col2:
                        st.plotly_chart(fig, use_container_width=True, key="time_series_chart")
                    
                    # Display trend analysis
                    trends = cargo_analysis.get('trend_analysis', {})
                    if trends:
                        st.subheader("📈 Trend Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Shipment Type Trends**")
                            shipment_trends = trends.get('shipment_types', {})
                            if shipment_trends:
                                for trend_type, trend_data in shipment_trends.items():
                                    if isinstance(trend_data, dict) and 'slope' in trend_data:
                                        direction = "📈" if trend_data['slope'] > 0 else "📉" if trend_data['slope'] < 0 else "➡️"
                                        st.write(f"{direction} {trend_type.replace('_', ' ').title()}: {trend_data['slope']:.2f}K tonnes/year")
                        
                        with col2:
                            st.write("**Transport Mode Trends**")
                            transport_trends = trends.get('transport_modes', {})
                            if transport_trends:
                                for trend_type, trend_data in transport_trends.items():
                                    if isinstance(trend_data, dict) and 'slope' in trend_data:
                                        direction = "📈" if trend_data['slope'] > 0 else "📉" if trend_data['slope'] < 0 else "➡️"
                                        st.write(f"{direction} {trend_type.replace('_', ' ').title()}: {trend_data['slope']:.2f}K tonnes/year")
                    else:
                        st.info("No trend data available")

        with cargo_tab4:
            st.subheader("Forecasting Analysis")
            
            forecasts = cargo_analysis.get('forecasts', {})
            if forecasts:
                # Display forecast charts
                import plotly.graph_objects as go
                
                st.write("**2024-2026 Cargo Throughput Forecasts**")
                
                # Create forecast visualization
                fig = go.Figure()
                
                # Historical data and forecasts for different categories
                forecast_categories = ['direct_shipment', 'transhipment', 'seaborne', 'river']
                colors = ['blue', 'red', 'green', 'orange']
                
                for i, category in enumerate(forecast_categories):
                    if category in forecasts:
                        forecast_data = forecasts[category]
                        
                        # Historical years (2014-2023) - ensure integers
                        hist_years = [int(year) for year in forecast_data.get('historical_years', [])]
                        hist_values = forecast_data.get('historical_values', [])
                        
                        # Forecast years (2024-2026) - ensure integers
                        forecast_years = [int(year) for year in forecast_data.get('forecast_years', [])]
                        forecast_values = forecast_data.get('forecast_values', [])
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=hist_years,
                            y=hist_values,
                            mode='lines+markers',
                            name=f'{category.replace("_", " ").title()} (Historical)',
                            line=dict(color=colors[i])
                        ))
                        
                        # Add forecast data
                        fig.add_trace(go.Scatter(
                            x=forecast_years,
                            y=forecast_values,
                            mode='lines+markers',
                            name=f'{category.replace("_", " ").title()} (Forecast)',
                            line=dict(color=colors[i], dash='dash')
                        ))
                    
                    fig.update_layout(
                        title="Port Cargo Throughput: Historical Data & Forecasts",
                        xaxis_title="Year",
                        yaxis_title="Throughput (000 tonnes)",
                        height=500,
                        xaxis=dict(tickmode='linear', dtick=1),  # Force integer years
                        margin=dict(l=50, r=50, t=50, b=50)  # Center the chart
                    )
                    
                    # Center the chart
                    chart_col1, chart_col2, chart_col3 = st.columns([0.1, 0.8, 0.1])
                    with chart_col2:
                        st.plotly_chart(fig, use_container_width=True, key=f"forecast_chart_{category}")
                    
                    # Display forecast metrics
                    st.subheader("🎯 Forecast Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**2024 Forecasts**")
                        for category in forecast_categories:
                            if category in forecasts and 'forecast_values' in forecasts[category]:
                                forecast_2024 = forecasts[category]['forecast_values'][0] if forecasts[category]['forecast_values'] else 0
                                st.metric(f"{category.replace('_', ' ').title()}", f"{forecast_2024:,.0f}K tonnes")
                    
                    with col2:
                        st.write("**2025 Forecasts**")
                        for category in forecast_categories:
                            if category in forecasts and 'forecast_values' in forecasts[category]:
                                forecast_2025 = forecasts[category]['forecast_values'][1] if len(forecasts[category]['forecast_values']) > 1 else 0
                                st.metric(f"{category.replace('_', ' ').title()}", f"{forecast_2025:,.0f}K tonnes")
                    
                    with col3:
                        st.write("**2026 Forecasts**")
                        for category in forecast_categories:
                            if category in forecasts and 'forecast_values' in forecasts[category]:
                                forecast_2026 = forecasts[category]['forecast_values'][2] if len(forecasts[category]['forecast_values']) > 2 else 0
                                st.metric(f"{category.replace('_', ' ').title()}", f"{forecast_2026:,.0f}K tonnes")
                    
                    # Model performance metrics
                    st.subheader("📊 Model Performance")
                    model_metrics = cargo_analysis.get('model_performance', {})
                    if model_metrics:
                        perf_col1, perf_col2 = st.columns(2)
                        
                        with perf_col1:
                            st.write("**R² Scores (Model Accuracy)**")
                            for category in forecast_categories:
                                if category in model_metrics and 'r2_score' in model_metrics[category]:
                                    r2_score = model_metrics[category]['r2_score']
                                    st.write(f"{category.replace('_', ' ').title()}: {r2_score:.3f}")
                        
                        with perf_col2:
                            st.write("**Mean Absolute Error**")
                            for category in forecast_categories:
                                if category in model_metrics and 'mae' in model_metrics[category]:
                                    mae = model_metrics[category]['mae']
                                    st.write(f"{category.replace('_', ' ').title()}: {mae:.1f}K tonnes")
                else:
                    st.info("No forecasting data available")

        with cargo_tab5:
            st.subheader("Cargo Type Analysis")
            cargo_type_data = cargo_analysis.get('cargo_type_analysis', {})

            if cargo_type_data:
                st.write("**Top Cargo Types (2023)**")
                if 'top_cargo_types' in cargo_type_data and cargo_type_data['top_cargo_types']:
                    cargo_df = pd.DataFrame(cargo_type_data['top_cargo_types'])
                    cargo_df.columns = ['Cargo Type', 'Throughput (000 tonnes)']
                    
                    # Center the dataframe
                    df_col1, df_col2, df_col3 = st.columns([0.1, 0.8, 0.1])
                    with df_col2:
                        st.dataframe(cargo_df, use_container_width=True)
                    with df_col2:
                        st.dataframe(cargo_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'total_cargo_types' in cargo_type_data:
                        st.metric("Total Cargo Types", cargo_type_data['total_cargo_types'])
                with col2:
                    if 'total_throughput' in cargo_type_data:
                        st.metric("Total Throughput", f"{cargo_type_data['total_throughput']:,.0f}K tonnes")
            else:
                st.info("No cargo type analysis data available")

        with cargo_tab6:
            st.subheader("Handling Location Analysis")
            location_data = cargo_analysis.get('location_analysis', {})

            if location_data:
                st.write("**Top Handling Locations (2023)**")
                if 'top_locations' in location_data and location_data['top_locations']:
                    location_df = pd.DataFrame(location_data['top_locations'])
                    location_df.columns = ['Handling Location', 'Throughput (000 tonnes)']
                    
                    # Center the dataframe
                    df_col1, df_col2, df_col3 = st.columns([0.1, 0.8, 0.1])
                    with df_col2:
                        st.dataframe(location_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'total_locations' in location_data:
                        st.metric("Total Locations", location_data['total_locations'])
                with col2:
                    if 'total_throughput' in location_data:
                        st.metric("Total Throughput", f"{location_data['total_throughput']:,.0f}K tonnes")
            else:
                st.info("No location analysis data available")

            # Efficiency Metrics Section
            st.subheader("📈 Port Efficiency Metrics")
            efficiency_data = cargo_analysis.get('efficiency_metrics', {})
            
            if efficiency_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'transhipment_ratio' in efficiency_data:
                        st.metric("Transhipment Ratio", f"{efficiency_data['transhipment_ratio']:.1f}%")
                with col2:
                    if 'seaborne_ratio' in efficiency_data:
                        st.metric("Seaborne Ratio", f"{efficiency_data['seaborne_ratio']:.1f}%")
                with col3:
                    if 'cargo_diversity_index' in efficiency_data:
                        st.metric("Cargo Diversity Index", f"{efficiency_data['cargo_diversity_index']}")
            else:
                st.info("No efficiency metrics available")
            
            # Data Summary Information
            st.subheader("📋 Analysis Summary")
            data_summary = cargo_analysis.get('data_summary', {})
            
            if data_summary:
                quality_col1, quality_col2 = st.columns(2)
                
                with quality_col1:
                    st.write("**Analysis Information**")
                    st.write(f"Tables Processed: {data_summary.get('tables_processed', 0)}")
                    if 'analysis_timestamp' in data_summary:
                        timestamp = data_summary['analysis_timestamp'][:19] if data_summary['analysis_timestamp'] else 'Unknown'
                        st.write(f"Analysis Time: {timestamp}")
                
                with quality_col2:
                    st.write("**Available Analysis Sections**")
                    analysis_sections = [k.replace('_analysis', '').replace('_', ' ').title() for k in cargo_analysis.keys() if k.endswith('_analysis')]
                    for section in analysis_sections:
                        st.write(f"✅ {section}")
            else:
                st.info("No analysis summary available")

    with tab5:
        st.subheader("🌊 Live Maritime Traffic")
        st.markdown("Real-time vessel tracking around Hong Kong waters")
        
        # Initialize MarineTraffic integration
        if MarineTrafficIntegration is not None:
            marine_traffic = MarineTrafficIntegration()
        else:
            st.warning("⚠️ MarineTraffic integration not available")
            st.info("The marine traffic visualization module could not be loaded.")
            marine_traffic = None
        
        # Display integration options
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Map Options")
            
            # Map type selection
            map_type = st.selectbox(
                "Map Type",
                ["Satellite", "Terrain", "Basic"],
                index=0
            )
            
            # Zoom level
            zoom_level = st.slider("Zoom Level", 8, 15, 11)
            
            # Show vessel types
            show_cargo = st.checkbox("Cargo Ships", True)
            show_tanker = st.checkbox("Tankers", True)
            show_passenger = st.checkbox("Passenger Ships", True)
            
            # Refresh button
            if st.button("🔄 Refresh Map"):
                st.rerun()
            
            # Information box
            st.info(
                "💡 **Note**: This is a live map integration with MarineTraffic. "
                "Vessel data is updated in real-time and shows actual ships "
                "in Hong Kong waters."
            )
            
            # API status (if available)
            if marine_traffic is not None:
                if marine_traffic.api_key:
                    st.success("✅ API Connected")
                    
                    # Show some sample API data
                    st.subheader("Live Data Sample")
                    try:
                        sample_data = marine_traffic.get_vessel_data_api()
                        if sample_data and 'data' in sample_data:
                            vessels = sample_data['data'][:3]  # Show first 3 vessels
                            for vessel in vessels:
                                st.text(f"🚢 {vessel.get('SHIPNAME', 'Unknown')}")
                                st.text(f"   Type: {vessel.get('TYPE_NAME', 'N/A')}")
                                st.text(f"   Speed: {vessel.get('SPEED', 'N/A')} knots")
                                st.text("---")
                    except Exception as e:
                        st.warning(f"API Error: {str(e)}")
                else:
                    st.warning("⚠️ API Key Required")
                    st.text("Set MARINETRAFFIC_API_KEY in .env for live data")
            else:
                st.warning("⚠️ MarineTraffic integration not available")
                st.text("Module could not be loaded")
        
        with col1:
            # Display the embedded map
            if marine_traffic is not None:
                marine_traffic.render_live_map_iframe(height=600)
            else:
                st.error("❌ Marine Traffic Map Unavailable")
                st.info("The marine traffic integration module could not be loaded. Please check the module dependencies.")
            
            # Additional information
            st.markdown(
                "**Live Maritime Traffic around Hong Kong**\n\n"
                "This map shows real-time vessel positions, including:"
            )
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.markdown("🚢 **Container Ships**\nCargo vessels carrying containers")
            with col_info2:
                st.markdown("🛢️ **Tankers**\nOil and chemical tankers")
            with col_info3:
                st.markdown("🚢 **Other Vessels**\nPassenger ships, tugs, etc.")
    
    with tab6:
        st.subheader("🛳️ Live Vessel Arrivals")
        st.markdown("Real-time vessel arrivals data from Hong Kong Marine Department")
        
        # Load real vessel data
        try:
            with st.spinner("Loading live vessel data..."):
                vessel_analysis = data.get('vessel_queue_analysis', {})
                real_vessels = load_vessel_arrivals()
            
            if vessel_analysis:
                # Display vessel queue analysis summary
                st.subheader("📊 Current Vessel Status")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    active_vessels = vessel_analysis.get('current_status', {}).get('active_vessels', 0)
                    st.metric("Active Vessels", active_vessels)
                with col2:
                    total_loaded = vessel_analysis.get('current_status', {}).get('total_vessels_loaded', 0)
                    st.metric("Total Loaded", total_loaded)
                with col3:
                    analysis_time = vessel_analysis.get('timestamps', {}).get('analysis_time', '')
                    if analysis_time:
                        time_str = analysis_time.split('T')[1][:5] if 'T' in analysis_time else analysis_time[:5]
                        st.metric("Analysis Time", time_str)
                    else:
                        st.metric("Analysis Time", "N/A")
                
                # Location breakdown
                st.subheader("📍 Vessel Locations")
                location_breakdown = vessel_analysis.get('location_breakdown', {})
                if location_breakdown:
                    location_col1, location_col2 = st.columns(2)
                    with location_col1:
                        st.write("**Current Distribution**")
                        location_df = pd.DataFrame([
                            {'Location Type': loc_type.replace('_', ' ').title(), 'Count': count}
                            for loc_type, count in location_breakdown.items()
                        ])
                        st.dataframe(location_df, use_container_width=True)
                    
                    with location_col2:
                        # Create a simple pie chart for location distribution
                        if len(location_breakdown) > 0:
                            import plotly.express as px
                            fig = px.pie(
                                values=list(location_breakdown.values()),
                                names=[name.replace('_', ' ').title() for name in location_breakdown.keys()],
                                title="Vessel Location Distribution"
                            )
                            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))  # Center the chart
                            
                            # Center the chart
                            chart_col1, chart_col2, chart_col3 = st.columns([0.1, 0.8, 0.1])
                            with chart_col2:
                                st.plotly_chart(fig, use_container_width=True, key="vessel_location_chart")
                
                # Ship category breakdown
                st.subheader("🚢 Ship Categories")
                ship_breakdown = vessel_analysis.get('ship_category_breakdown', {})
                if ship_breakdown:
                    ship_col1, ship_col2 = st.columns(2)
                    with ship_col1:
                        st.write("**Ship Types**")
                        ship_df = pd.DataFrame([
                            {'Ship Category': cat.replace('_', ' ').title(), 'Count': count}
                            for cat, count in ship_breakdown.items()
                        ])
                        st.dataframe(ship_df, use_container_width=True)
                    
                    with ship_col2:
                        # Create a bar chart for ship categories
                        if len(ship_breakdown) > 0:
                            import plotly.express as px
                            fig = px.bar(
                                x=list(ship_breakdown.keys()),
                                y=list(ship_breakdown.values()),
                                title="Ship Category Distribution",
                                labels={'x': 'Ship Category', 'y': 'Number of Vessels'}
                            )
                            fig.update_xaxes(tickangle=45)
                            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))  # Center the chart
                            
                            # Center the chart
                            chart_col1, chart_col2, chart_col3 = st.columns([0.1, 0.8, 0.1])
                            with chart_col2:
                                st.plotly_chart(fig, use_container_width=True, key="ship_category_chart")
                
                # Recent activity
                st.subheader("🕐 Recent Activity")
                recent_activity = vessel_analysis.get('recent_activity', {})
                if recent_activity:
                    activity_col1, activity_col2 = st.columns(2)
                    with activity_col1:
                        st.write("**Activity Summary**")
                        arrivals_24h = recent_activity.get('arrivals_last_24h', 0)
                        arrivals_12h = recent_activity.get('arrivals_last_12h', 0)
                        arrivals_6h = recent_activity.get('arrivals_last_6h', 0)
                        
                        st.metric("Last 24 Hours", f"{arrivals_24h} arrivals")
                        st.metric("Last 12 Hours", f"{arrivals_12h} arrivals")
                        st.metric("Last 6 Hours", f"{arrivals_6h} arrivals")
                    
                    with activity_col2:
                        # Activity trend chart
                        activity_data = {
                            'Time Period': ['Last 6h', 'Last 12h', 'Last 24h'],
                            'Arrivals': [arrivals_6h, arrivals_12h, arrivals_24h]
                        }
                        activity_df = pd.DataFrame(activity_data)
                        
                        import plotly.express as px
                        fig = px.line(
                            activity_df, 
                            x='Time Period', 
                            y='Arrivals',
                            title="Arrival Activity Trend",
                            markers=True
                        )
                        fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))  # Center the chart
                        
                        # Center the chart
                        chart_col1, chart_col2, chart_col3 = st.columns([0.1, 0.8, 0.1])
                        with chart_col2:
                            st.plotly_chart(fig, use_container_width=True, key="activity_trend_chart")
                
                # Raw vessel data table
                st.subheader("📋 Detailed Vessel Data")
                if not real_vessels.empty:
                    # Add filters
                    filter_col1, filter_col2, filter_col3 = st.columns(3)
                    
                    with filter_col1:
                        status_filter = st.selectbox(
                            "Filter by Status",
                            ['All'] + list(real_vessels['status'].unique())
                        )
                    
                    with filter_col2:
                        location_filter = st.selectbox(
                            "Filter by Location Type",
                            ['All'] + list(real_vessels['location_type'].unique())
                        )
                    
                    with filter_col3:
                        category_filter = st.selectbox(
                            "Filter by Ship Category",
                            ['All'] + list(real_vessels['ship_category'].unique())
                        )
                    
                    # Apply filters
                    filtered_vessels = real_vessels.copy()
                    if status_filter != 'All':
                        filtered_vessels = filtered_vessels[filtered_vessels['status'] == status_filter]
                    if location_filter != 'All':
                        filtered_vessels = filtered_vessels[filtered_vessels['location_type'] == location_filter]
                    if category_filter != 'All':
                        filtered_vessels = filtered_vessels[filtered_vessels['ship_category'] == category_filter]
                    
                    # Display filtered data
                    st.write(f"**Showing {len(filtered_vessels)} of {len(real_vessels)} vessels**")
                    
                    # Select columns to display
                    display_columns = ['vessel_name', 'call_sign', 'ship_category', 'location_type', 'status', 'arrival_time']
                    available_columns = [col for col in display_columns if col in filtered_vessels.columns]
                    
                    if available_columns:
                        st.dataframe(
                            filtered_vessels[available_columns].sort_values('arrival_time', ascending=False),
                            use_container_width=True
                        )
                    else:
                        st.dataframe(filtered_vessels, use_container_width=True)
                    
                    # Export vessel data
                    vessel_csv = filtered_vessels.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Vessel Data",
                        data=vessel_csv,
                        file_name=f"vessel_arrivals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No vessel data available")
            else:
                st.info("No vessel analysis data available")
                
        except Exception as e:
            st.error(f"Error loading vessel data: {str(e)}")
            st.info("Please ensure the vessel arrivals XML file is available and properly formatted.")
    
    with tab7:
        st.subheader("🏗️ Live Berth Status")
        st.markdown("Real-time berth occupancy and availability")
        
        # Load berth data
        berth_data, berth_metrics = get_real_berth_data()
        
        if not berth_data.empty:
            # Berth status overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                occupied = len(berth_data[berth_data['status'] == 'occupied'])
                st.metric("Occupied Berths", occupied)
            
            with col2:
                available = len(berth_data[berth_data['status'] == 'available'])
                st.metric("Available Berths", available)
            
            with col3:
                maintenance = len(berth_data[berth_data['status'] == 'maintenance'])
                st.metric("Under Maintenance", maintenance)
            
            with col4:
                total_berths = len(berth_data)
                utilization = (occupied / total_berths * 100) if total_berths > 0 else 0
                st.metric("Utilization Rate", f"{utilization:.1f}%")
            
            # Berth details table
            st.subheader("📋 Berth Details")
            st.dataframe(berth_data, use_container_width=True)
        else:
            st.info("No berth data available")
    
    with tab8:
        st.subheader("🎯 Scenario Analysis & Comparison")
        st.markdown("Compare different operational scenarios and their impact on port performance")
        
        # Scenario comparison interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Scenario Selection")
            
            # Get available scenarios
            available_scenarios = list_available_scenarios()
            
            # Multi-select for scenario comparison
            selected_scenarios = st.multiselect(
                "Select scenarios to compare:",
                available_scenarios,
                default=available_scenarios[:2] if len(available_scenarios) >= 2 else available_scenarios
            )
            
            if st.button("🔄 Run Scenario Comparison"):
                if len(selected_scenarios) >= 2:
                    with st.spinner("Running scenario comparison..."):
                        # Initialize scenario manager
                        scenario_manager = ScenarioManager()
                        
                        # Store comparison results
                        comparison_results = {}
                        
                        for scenario in selected_scenarios:
                            # Get scenario parameters
                            from hk_port_digital_twin.src.scenarios.scenario_parameters import get_scenario_parameters
                            params = get_scenario_parameters(scenario)
                            if params:
                                comparison_results[scenario] = {
                                    'ship_arrival_rate': params.arrival_rate_multiplier,
                                    'container_volume_multiplier': params.average_ship_size_multiplier,
                                    'processing_efficiency': params.processing_rate_multiplier,
                                    'berth_utilization': params.target_berth_utilization
                                }
                            else:
                                comparison_results[scenario] = {
                                    'ship_arrival_rate': 1.0,
                                    'container_volume_multiplier': 1.0,
                                    'processing_efficiency': 1.0,
                                    'berth_utilization': 0.8
                                }
                        
                        st.session_state.scenario_comparison = comparison_results
                        st.success("Scenario comparison completed!")
                else:
                    st.warning("Please select at least 2 scenarios for comparison")
        
        with col2:
            st.subheader("📈 Comparison Results")
            
            if hasattr(st.session_state, 'scenario_comparison') and st.session_state.scenario_comparison:
                comparison_data = st.session_state.scenario_comparison
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame(comparison_data).T
                comparison_df.index.name = 'Scenario'
                
                # Display comparison table
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization of key metrics
                st.subheader("📊 Visual Comparison")
                
                # Ship arrival rate comparison
                fig_col1, fig_col2 = st.columns(2)
                
                with fig_col1:
                    import plotly.express as px
                    
                    # Ship arrival rate chart
                    arrival_data = {
                        'Scenario': list(comparison_data.keys()),
                        'Ship Arrival Rate': [data['ship_arrival_rate'] for data in comparison_data.values()]
                    }
                    fig1 = px.bar(
                        arrival_data,
                        x='Scenario',
                        y='Ship Arrival Rate',
                        title='Ship Arrival Rate by Scenario',
                        color='Ship Arrival Rate',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with fig_col2:
                    # Processing efficiency chart
                    efficiency_data = {
                        'Scenario': list(comparison_data.keys()),
                        'Processing Efficiency': [data['processing_efficiency'] for data in comparison_data.values()]
                    }
                    fig2 = px.bar(
                        efficiency_data,
                        x='Scenario',
                        y='Processing Efficiency',
                        title='Processing Efficiency by Scenario',
                        color='Processing Efficiency',
                        color_continuous_scale='plasma'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Container volume comparison
                st.subheader("📦 Container Volume Impact")
                volume_data = {
                    'Scenario': list(comparison_data.keys()),
                    'Container Volume Multiplier': [data['container_volume_multiplier'] for data in comparison_data.values()]
                }
                fig3 = px.line(
                    volume_data,
                    x='Scenario',
                    y='Container Volume Multiplier',
                    title='Container Volume Multiplier by Scenario',
                    markers=True
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Export comparison results
                comparison_csv = comparison_df.to_csv()
                st.download_button(
                    label="📥 Export Comparison Results",
                    data=comparison_csv,
                    file_name=f"scenario_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Run a scenario comparison to see results here")
                
                # Show available scenarios info
                st.subheader("📋 Available Scenarios")
                scenario_manager = ScenarioManager()
                
                for scenario in available_scenarios:
                    with st.expander(f"📄 {scenario}"):
                        scenario_info = scenario_manager.get_scenario_info(scenario)
                        st.write(f"**Description:** {scenario_info.get('description', 'No description available')}")
                        
                        from hk_port_digital_twin.src.scenarios.scenario_parameters import get_scenario_parameters
                        from dataclasses import asdict
                        params = get_scenario_parameters(scenario)
                        params_dict = asdict(params)
                        st.write("**Parameters:**")
                        for key, value in params_dict.items():
                            if key != 'description':
                                st.write(f"- {key.replace('_', ' ').title()}: {value}")
        
        # Multi-Scenario Optimization Section
        st.markdown("---")
        st.subheader("🚀 Multi-Scenario Optimization Analysis")
        st.markdown("Compare optimized vs. non-optimized operations across different scenarios")
        
        # Multi-scenario optimization interface
        opt_col1, opt_col2 = st.columns([1, 2])
        
        with opt_col1:
            st.subheader("⚙️ Optimization Settings")
            
            # Simulation duration
            simulation_hours = st.slider(
                "Simulation Duration (hours)",
                min_value=24,
                max_value=168,
                value=72,
                step=24,
                help="Duration for the optimization comparison simulation"
            )
            
            # Use historical data toggle
            use_historical = st.checkbox(
                "Use Historical Data",
                value=True,
                help="Use real historical patterns for seasonal analysis"
            )
            
            if st.button("🔄 Run Multi-Scenario Optimization"):
                with st.spinner("Running multi-scenario optimization analysis..."):
                    try:
                        # Import the multi-scenario optimizer
                        from hk_port_digital_twin.src.scenarios.multi_scenario_optimizer import create_sample_multi_scenario_comparison
                        
                        # Run the optimization comparison
                        optimization_results = create_sample_multi_scenario_comparison(
                            simulation_hours=simulation_hours,
                            use_historical_data=use_historical
                        )
                        
                        if optimization_results:
                            st.session_state.multi_scenario_optimization = optimization_results
                            st.success("Multi-scenario optimization completed!")
                        else:
                            st.error("Failed to run multi-scenario optimization")
                            
                    except Exception as e:
                        st.error(f"Error running optimization: {str(e)}")
                        logging.error(f"Multi-scenario optimization error: {e}")
        
        with opt_col2:
            st.subheader("📊 Optimization Results")
            
            if hasattr(st.session_state, 'multi_scenario_optimization') and st.session_state.multi_scenario_optimization:
                opt_results = st.session_state.multi_scenario_optimization
                
                # Display seasonal patterns if available
                if 'seasonal_patterns' in opt_results:
                    st.subheader("🌊 Seasonal Patterns Analysis")
                    seasonal_data = opt_results['seasonal_patterns']
                    
                    # Create seasonal patterns dataframe
                    seasonal_df = pd.DataFrame({
                        'Season': ['Peak', 'Normal', 'Low'],
                        'Throughput (TEU)': [
                            seasonal_data.get('peak_throughput', 0),
                            seasonal_data.get('normal_throughput', 0),
                            seasonal_data.get('low_throughput', 0)
                        ],
                        'Ship Arrivals': [
                            seasonal_data.get('peak_arrivals', 0),
                            seasonal_data.get('normal_arrivals', 0),
                            seasonal_data.get('low_arrivals', 0)
                        ]
                    })
                    
                    st.dataframe(seasonal_df, use_container_width=True)
                
                # Display scenario comparisons
                if 'scenario_comparisons' in opt_results:
                    st.subheader("⚡ Optimization Impact by Scenario")
                    
                    scenario_comparisons = opt_results['scenario_comparisons']
                    
                    # Create comparison metrics
                    comparison_metrics = []
                    for scenario_name, comparison in scenario_comparisons.items():
                        comparison_metrics.append({
                            'Scenario': scenario_name.replace('_', ' ').title(),
                            'Waiting Time Improvement (%)': f"{comparison.get('waiting_time_improvement', 0):.1f}%",
                            'Berth Utilization Improvement (%)': f"{comparison.get('berth_utilization_improvement', 0):.1f}%",
                            'Overall Efficiency Improvement (%)': f"{comparison.get('overall_efficiency_improvement', 0):.1f}%"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_metrics)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualization of optimization improvements
                    st.subheader("📈 Optimization Impact Visualization")
                    
                    # Create improvement charts
                    import plotly.express as px
                    
                    # Waiting time improvements
                    waiting_improvements = {
                        'Scenario': [comp['Scenario'] for comp in comparison_metrics],
                        'Waiting Time Improvement (%)': [float(comp['Waiting Time Improvement (%)'].replace('%', '')) for comp in comparison_metrics]
                    }
                    
                    fig_waiting = px.bar(
                        waiting_improvements,
                        x='Scenario',
                        y='Waiting Time Improvement (%)',
                        title='Waiting Time Reduction by Scenario',
                        color='Waiting Time Improvement (%)',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_waiting, use_container_width=True)
                    
                    # Overall efficiency improvements
                    efficiency_improvements = {
                        'Scenario': [comp['Scenario'] for comp in comparison_metrics],
                        'Overall Efficiency Improvement (%)': [float(comp['Overall Efficiency Improvement (%)'].replace('%', '')) for comp in comparison_metrics]
                    }
                    
                    fig_efficiency = px.bar(
                        efficiency_improvements,
                        x='Scenario',
                        y='Overall Efficiency Improvement (%)',
                        title='Overall Efficiency Improvement by Scenario',
                        color='Overall Efficiency Improvement (%)',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_efficiency, use_container_width=True)
                
                # Display overall insights
                if 'overall_insights' in opt_results:
                    st.subheader("💡 Key Insights")
                    insights = opt_results['overall_insights']
                    
                    # Display insights in an organized manner
                    insight_col1, insight_col2 = st.columns(2)
                    
                    with insight_col1:
                        st.metric(
                            "Best Performing Scenario",
                            insights.get('best_scenario', 'N/A'),
                            help="Scenario with highest optimization impact"
                        )
                        
                        st.metric(
                            "Average Waiting Time Reduction",
                            f"{insights.get('avg_waiting_improvement', 0):.1f}%",
                            help="Average improvement across all scenarios"
                        )
                    
                    with insight_col2:
                        st.metric(
                            "Peak Season Benefit",
                            f"{insights.get('peak_season_benefit', 0):.1f}%",
                            help="Additional benefit during peak operations"
                        )
                        
                        st.metric(
                            "Overall ROI Indicator",
                            insights.get('roi_category', 'N/A'),
                            help="Return on investment category for optimization"
                        )
                    
                    # Additional insights text
                    if 'summary' in insights:
                        st.info(insights['summary'])
                
                # Export optimization results
                if 'scenario_comparisons' in opt_results:
                    export_data = []
                    for scenario_name, comparison in opt_results['scenario_comparisons'].items():
                        export_data.append({
                            'Scenario': scenario_name,
                            'Waiting_Time_Improvement': comparison.get('waiting_time_improvement', 0),
                            'Berth_Utilization_Improvement': comparison.get('berth_utilization_improvement', 0),
                            'Overall_Efficiency_Improvement': comparison.get('overall_efficiency_improvement', 0)
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    export_csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Export Optimization Results",
                        data=export_csv,
                        file_name=f"multi_scenario_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Run a multi-scenario optimization to see detailed results here")
                
                # Show optimization benefits preview
                st.markdown("""
                **Multi-Scenario Optimization Benefits:**
                - 🎯 **Optimized Berth Allocation**: AI-powered assignment reduces waiting times
                - 📊 **Seasonal Analysis**: Compare performance across peak, normal, and low seasons
                - ⚡ **Efficiency Gains**: Quantify improvements in throughput and utilization
                - 💰 **ROI Analysis**: Understand the business impact of optimization
                - 🔄 **Real-time Adaptation**: Dynamic allocation based on current conditions
                """)
        
        st.markdown("---")
        st.subheader("⚠️ Predictive Disruption Impact Simulation")
        st.markdown("Analyze the impact of potential disruptions on port operations")
        
        # Disruption simulation interface
        disrupt_col1, disrupt_col2 = st.columns([1, 2])
        
        with disrupt_col1:
            st.subheader("🌪️ Disruption Settings")
            
            # Disruption type selection
            disruption_type = st.selectbox(
                "Disruption Type",
                ["Equipment Failure", "Typhoon Impact", "Labor Shortage", "Cyber Attack", "Supply Chain Disruption"],
                help="Select the type of disruption to simulate"
            )
            
            # Severity level
            severity_level = st.selectbox(
                "Severity Level",
                ["Low", "Medium", "High", "Critical"],
                index=1,
                help="Impact severity of the disruption"
            )
            
            # Duration
            disruption_duration = st.slider(
                "Disruption Duration (hours)",
                min_value=1,
                max_value=72,
                value=12,
                help="How long the disruption lasts"
            )
            
            # Recovery time
            recovery_time = st.slider(
                "Recovery Time (hours)",
                min_value=1,
                max_value=48,
                value=6,
                help="Time to fully recover from disruption"
            )
            
            if st.button("🔥 Simulate Disruption Impact"):
                with st.spinner("Simulating disruption impact..."):
                    try:
                        # Import the disruption simulator
                        from hk_port_digital_twin.src.scenarios.disruption_simulator import DisruptionSimulator, DisruptionType, DisruptionSeverity
                        
                        # Create disruption simulator
                        simulator = DisruptionSimulator()
                        
                        # Map UI selections to enum values
                        type_mapping = {
                            "Equipment Failure": DisruptionType.EQUIPMENT_FAILURE,
                            "Typhoon Impact": DisruptionType.TYPHOON,
                            "Labor Shortage": DisruptionType.LABOR_SHORTAGE,
                            "Cyber Attack": DisruptionType.CYBER_ATTACK,
                            "Supply Chain Disruption": DisruptionType.SUPPLY_CHAIN
                        }
                        
                        severity_mapping = {
                            "Low": DisruptionSeverity.LOW,
                            "Medium": DisruptionSeverity.MEDIUM,
                            "High": DisruptionSeverity.HIGH,
                            "Critical": DisruptionSeverity.CRITICAL
                        }
                        
                        # Run disruption simulation
                        disruption_results = simulator.simulate_disruption_impact(
                            disruption_type=type_mapping[disruption_type],
                            severity=severity_mapping[severity_level],
                            duration_hours=disruption_duration,
                            recovery_hours=recovery_time
                        )
                        
                        if disruption_results:
                            st.session_state.disruption_simulation = disruption_results
                            st.success("Disruption impact simulation completed!")
                        else:
                            st.error("Failed to run disruption simulation")
                            
                    except Exception as e:
                        st.error(f"Error running disruption simulation: {str(e)}")
                        logging.error(f"Disruption simulation error: {e}")
        
        with disrupt_col2:
            st.subheader("📊 Disruption Impact Results")
            
            if hasattr(st.session_state, 'disruption_simulation') and st.session_state.disruption_simulation:
                disrupt_results = st.session_state.disruption_simulation
                
                # Display impact metrics
                st.subheader("💥 Impact Assessment")
                
                impact_col1, impact_col2, impact_col3 = st.columns(3)
                
                with impact_col1:
                    st.metric(
                        "Throughput Reduction",
                        f"{disrupt_results.get('throughput_impact', 0):.1f}%",
                        help="Percentage reduction in container throughput"
                    )
                
                with impact_col2:
                    st.metric(
                        "Waiting Time Increase",
                        f"{disrupt_results.get('waiting_time_impact', 0):.1f}%",
                        help="Percentage increase in vessel waiting times"
                    )
                
                with impact_col3:
                    st.metric(
                        "Revenue Loss",
                        f"${disrupt_results.get('revenue_loss', 0):,.0f}",
                        help="Estimated revenue loss during disruption"
                    )
                
                # Recovery timeline
                if 'recovery_timeline' in disrupt_results:
                    st.subheader("🔄 Recovery Timeline")
                    
                    recovery_data = disrupt_results['recovery_timeline']
                    recovery_df = pd.DataFrame({
                        'Hour': list(range(len(recovery_data))),
                        'Operational Capacity (%)': recovery_data
                    })
                    
                    import plotly.express as px
                    fig_recovery = px.line(
                        recovery_df,
                        x='Hour',
                        y='Operational Capacity (%)',
                        title='Port Operational Capacity Recovery',
                        markers=True
                    )
                    fig_recovery.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="Full Capacity")
                    st.plotly_chart(fig_recovery, use_container_width=True)
                
                # Recommendations
                if 'recommendations' in disrupt_results:
                    st.subheader("💡 Recovery Recommendations")
                    recommendations = disrupt_results['recommendations']
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"**{i}.** {rec}")
                
                # Export disruption results
                export_disruption_data = {
                    'Disruption_Type': disruption_type,
                    'Severity': severity_level,
                    'Duration_Hours': disruption_duration,
                    'Recovery_Hours': recovery_time,
                    'Throughput_Impact_Percent': disrupt_results.get('throughput_impact', 0),
                    'Waiting_Time_Impact_Percent': disrupt_results.get('waiting_time_impact', 0),
                    'Revenue_Loss_USD': disrupt_results.get('revenue_loss', 0)
                }
                
                export_disruption_df = pd.DataFrame([export_disruption_data])
                export_disruption_csv = export_disruption_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Export Disruption Analysis",
                    data=export_disruption_csv,
                    file_name=f"disruption_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Run a disruption simulation to see impact analysis here")
                
                # Show disruption simulation benefits preview
                st.markdown("""
                **Disruption Impact Simulation Benefits:**
                - ⚠️ **Risk Assessment**: Quantify potential impacts of various disruptions
                - 🔄 **Recovery Planning**: Optimize recovery strategies and timelines
                - 💰 **Financial Impact**: Estimate revenue losses and mitigation costs
                - 📊 **Scenario Planning**: Prepare for different disruption scenarios
                - 🛡️ **Resilience Building**: Identify vulnerabilities and strengthen operations
                """)
        
        st.markdown("---")
        st.subheader("💰 Dynamic Capacity Planning & Investment Simulation")
        st.markdown("Analyze investment opportunities and capacity expansion scenarios")
        
        # Investment planning interface
        invest_col1, invest_col2 = st.columns([1, 2])
        
        with invest_col1:
            st.subheader("🏗️ Investment Options")
            
            # Investment type selection
            investment_type = st.selectbox(
                "Investment Type",
                ["New Berth Construction", "Crane Upgrade", "Automation System", "IT Infrastructure", "Storage Expansion"],
                help="Select the type of investment to analyze"
            )
            
            # Investment amount
            investment_amount = st.number_input(
                "Investment Amount (USD Million)",
                min_value=1.0,
                max_value=500.0,
                value=50.0,
                step=5.0,
                help="Total investment cost in millions USD"
            )
            
            # Expected demand growth
            demand_growth = st.slider(
                "Expected Annual Demand Growth (%)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Expected annual growth in container throughput"
            )
            
            # Analysis period
            analysis_period = st.slider(
                "Analysis Period (years)",
                min_value=5,
                max_value=20,
                value=10,
                help="Time horizon for ROI analysis"
            )
            
            if st.button("📈 Analyze Investment ROI"):
                with st.spinner("Analyzing investment scenarios..."):
                    try:
                        # Import the investment planner
                        from hk_port_digital_twin.src.scenarios.investment_planner import InvestmentPlanner, InvestmentType
                        
                        # Create investment planner
                        planner = InvestmentPlanner()
                        
                        # Map UI selections to enum values
                        investment_type_mapping = {
                            "New Berth Construction": InvestmentType.NEW_BERTH,
                            "Crane Upgrade": InvestmentType.CRANE_UPGRADE,
                            "Automation System": InvestmentType.AUTOMATION,
                            "IT Infrastructure": InvestmentType.IT_INFRASTRUCTURE,
                            "Storage Expansion": InvestmentType.STORAGE_EXPANSION
                        }
                        
                        # Run investment analysis
                        investment_results = planner.analyze_investment_scenario(
                            investment_type=investment_type_mapping[investment_type],
                            investment_amount=investment_amount * 1_000_000,  # Convert to USD
                            demand_growth_rate=demand_growth / 100,  # Convert to decimal
                            analysis_years=analysis_period
                        )
                        
                        if investment_results:
                            st.session_state.investment_analysis = investment_results
                            st.success("Investment analysis completed!")
                        else:
                            st.error("Failed to run investment analysis")
                            
                    except Exception as e:
                        st.error(f"Error running investment analysis: {str(e)}")
                        logging.error(f"Investment analysis error: {e}")
        
        with invest_col2:
            st.subheader("📊 Investment Analysis Results")
            
            if hasattr(st.session_state, 'investment_analysis') and st.session_state.investment_analysis:
                invest_results = st.session_state.investment_analysis
                
                # Display ROI metrics
                st.subheader("💹 Financial Metrics")
                
                roi_col1, roi_col2, roi_col3 = st.columns(3)
                
                with roi_col1:
                    st.metric(
                        "ROI",
                        f"{invest_results.get('roi_percentage', 0):.1f}%",
                        help="Return on Investment percentage"
                    )
                
                with roi_col2:
                    st.metric(
                        "NPV",
                        f"${invest_results.get('npv', 0):,.0f}",
                        help="Net Present Value in USD"
                    )
                
                with roi_col3:
                    st.metric(
                        "Payback Period",
                        f"{invest_results.get('payback_years', 0):.1f} years",
                        help="Time to recover initial investment"
                    )
                
                # Capacity impact
                if 'capacity_impact' in invest_results:
                    st.subheader("📈 Capacity Impact")
                    
                    capacity_col1, capacity_col2 = st.columns(2)
                    
                    with capacity_col1:
                        st.metric(
                            "Throughput Increase",
                            f"{invest_results['capacity_impact'].get('throughput_increase', 0):.1f}%",
                            help="Expected increase in container throughput"
                        )
                    
                    with capacity_col2:
                        st.metric(
                            "Efficiency Improvement",
                            f"{invest_results['capacity_impact'].get('efficiency_improvement', 0):.1f}%",
                            help="Expected improvement in operational efficiency"
                        )
                
                # Financial projections
                if 'financial_projections' in invest_results:
                    st.subheader("💰 Financial Projections")
                    
                    projections = invest_results['financial_projections']
                    projections_df = pd.DataFrame({
                        'Year': list(range(1, len(projections) + 1)),
                        'Revenue (USD Million)': [p / 1_000_000 for p in projections],
                        'Cumulative Revenue (USD Million)': [sum(projections[:i+1]) / 1_000_000 for i in range(len(projections))]
                    })
                    
                    import plotly.express as px
                    fig_projections = px.line(
                        projections_df,
                        x='Year',
                        y=['Revenue (USD Million)', 'Cumulative Revenue (USD Million)'],
                        title='Revenue Projections Over Time',
                        markers=True
                    )
                    st.plotly_chart(fig_projections, use_container_width=True)
                
                # Investment recommendations
                if 'recommendations' in invest_results:
                    st.subheader("💡 Investment Recommendations")
                    recommendations = invest_results['recommendations']
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"**{i}.** {rec}")
                
                # Risk assessment
                if 'risk_assessment' in invest_results:
                    st.subheader("⚠️ Risk Assessment")
                    risk_data = invest_results['risk_assessment']
                    
                    risk_df = pd.DataFrame({
                        'Risk Factor': list(risk_data.keys()),
                        'Risk Level': list(risk_data.values())
                    })
                    
                    st.dataframe(risk_df, use_container_width=True)
                
                # Export investment results
                export_investment_data = {
                    'Investment_Type': investment_type,
                    'Investment_Amount_USD_Million': investment_amount,
                    'Demand_Growth_Percent': demand_growth,
                    'Analysis_Period_Years': analysis_period,
                    'ROI_Percentage': invest_results.get('roi_percentage', 0),
                    'NPV_USD': invest_results.get('npv', 0),
                    'Payback_Years': invest_results.get('payback_years', 0)
                }
                
                export_investment_df = pd.DataFrame([export_investment_data])
                export_investment_csv = export_investment_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Export Investment Analysis",
                    data=export_investment_csv,
                    file_name=f"investment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Run an investment analysis to see detailed results here")
                
                # Show investment analysis benefits preview
                st.markdown("""
                **Investment Analysis Benefits:**
                - 💹 **ROI Calculation**: Quantify return on investment for different options
                - 📈 **Capacity Planning**: Optimize expansion based on demand projections
                - 💰 **Financial Modeling**: Detailed cash flow and NPV analysis
                - ⚖️ **Risk Assessment**: Evaluate investment risks and mitigation strategies
                - 🎯 **Strategic Planning**: Make data-driven investment decisions
                """)
    
    with tab9:
        st.subheader("⚙️ Settings")
        st.markdown("Configure simulation parameters and system settings")
        
        # Simulation settings
        st.subheader("🔧 Simulation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance Settings**")
            auto_refresh = st.checkbox("Auto-refresh dashboard", value=True)
            refresh_interval = st.slider("Refresh interval (seconds)", 1, 30, 5)
            
            st.write("**Data Settings**")
            use_real_data = st.checkbox("Use real-time data", value=True)
            cache_duration = st.slider("Cache duration (minutes)", 1, 60, 10)
        
        with col2:
            st.write("**Display Settings**")
            show_debug_info = st.checkbox("Show debug information", value=False)
            chart_theme = st.selectbox("Chart theme", ["plotly", "plotly_white", "plotly_dark"], index=0)
            
            st.write("**Export Settings**")
            default_export_format = st.selectbox("Default export format", ["CSV", "Excel", "JSON"], index=0)
        
        # System information
        st.subheader("📊 System Information")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Dashboard Version", "1.0.0")
            st.metric("Streamlit Version", st.__version__)
        
        with info_col2:
            st.metric("Active Sessions", "1")
            st.metric("Uptime", "Running")
        
        with info_col3:
            st.metric("Data Sources", "3")
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
        
        # Reset options
        st.subheader("🔄 Reset Options")
        
        reset_col1, reset_col2, reset_col3 = st.columns(3)
        
        with reset_col1:
            if st.button("🗑️ Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared successfully!")
        
        with reset_col2:
            if st.button("🔄 Reset Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Session reset successfully!")
                st.rerun()
        
        with reset_col3:
            if st.button("📊 Reset Simulation"):
                if hasattr(st.session_state, 'simulation_controller'):
                    st.session_state.simulation_controller.reset()
                st.success("Simulation reset successfully!")


if __name__ == "__main__":
    main()