import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Try to import weather integration
try:
    from utils.weather_integration import HKObservatoryIntegration
except ImportError:
    HKObservatoryIntegration = None

# Try to import data loader functions
try:
    from utils.data_loader import (
        load_container_throughput,
        RealTimeDataConfig,
        get_real_time_manager,
        get_vessel_queue_analysis,
        load_focused_cargo_statistics,
        get_enhanced_cargo_analysis,
        get_time_series_data
    )
except ImportError as e:
    print(f"Import error for data_loader: {e}")
    load_container_throughput = None
    RealTimeDataConfig = None
    get_real_time_manager = None
    get_vessel_queue_analysis = None
    load_focused_cargo_statistics = None
    get_enhanced_cargo_analysis = None
    get_time_series_data = None

# Try to import simulation components
try:
    from simulation.port_simulation import PortSimulation
    from simulation.simulation_controller import SimulationController
    from simulation.berth_manager import BerthManager
    import simpy
except ImportError as e:
    print(f"Import error for simulation: {e}")
    PortSimulation = None
    SimulationController = None
    BerthManager = None
    simpy = None

# Try to import visualization functions
try:
    from utils.visualization import (
        create_port_layout_chart,
        create_ship_queue_chart,
        create_berth_utilization_chart,
        create_throughput_timeline,
        create_waiting_time_distribution
    )
except ImportError as e:
    print(f"Import error for visualization: {e}")
    create_port_layout_chart = None
    create_ship_queue_chart = None
    create_berth_utilization_chart = None
    create_throughput_timeline = None
    create_waiting_time_distribution = None

# Try to import MarineTraffic integration
try:
    from marine_traffic_integration import MarineTrafficIntegration
except ImportError:
    MarineTrafficIntegration = None

PORT_CONFIG = {
    'total_berths': 10,
    'berth_types': ['container', 'container', 'container', 'container', 'bulk', 'bulk', 'mixed', 'mixed'],
    'crane_count': [4, 3, 4, 2, 2, 2, 3, 3],
    'max_capacity_teu': [5000, 4000, 5000, 3000, 6000, 6000, 4500, 4500],
    'is_occupied': [True, False, True, False, True, False, True, False]
}


SHIP_TYPES = {
    'container': [f'SHIP_{i:03d}' for i in range(1, 21)],
    'bulk': [f'SHIP_{i:03d}' for i in range(1, 21)],
    'mixed': [f'SHIP_{i:03d}' for i in range(1, 21)]
}


SIMULATION_CONFIG = {
    'default_duration': 24,
    'ship_arrival_rate': 0.5
}


def create_kpi_summary_chart(kpi_dict):
    """Create KPI summary chart"""
    fig = go.Figure()
    
    # Add historical data and forecasts for different categories
    forecast_categories = ['direct_shipment', 'transhipment', 'seaborne', 'river']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, category in enumerate(forecast_categories):
        if category in kpi_dict:
            forecast_data = kpi_dict[category]
            
            # Historical years (2014-2023)
            hist_years = forecast_data.get('historical_years', [])
            hist_values = forecast_data.get('historical_values', [])
            
            # Forecast years (2024-2026)
            forecast_years = forecast_data.get('forecast_years', [])
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
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True, key="kpi_summary_chart")
    
    # Display forecast metrics
    st.subheader("🎯 Forecast Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**2024 Forecasts**")
        for category in forecast_categories:
            if category in kpi_dict and 'forecast_values' in kpi_dict[category]:
                forecast_2024 = kpi_dict[category]['forecast_values'][0] if kpi_dict[category]['forecast_values'] else 0
                st.metric(f"{category.replace('_', ' ').title()}", f"{forecast_2024:,.0f}K tonnes")
    
    with col2:
        st.write("**2025 Forecasts**")
        for category in forecast_categories:
            if category in kpi_dict and 'forecast_values' in kpi_dict[category]:
                forecast_2025 = kpi_dict[category]['forecast_values'][1] if len(kpi_dict[category]['forecast_values']) > 1 else 0
                st.metric(f"{category.replace('_', ' ').title()}", f"{forecast_2025:,.0f}K tonnes")
    
    with col3:
        st.write("**2026 Forecasts**")
        for category in forecast_categories:
            if category in kpi_dict and 'forecast_values' in kpi_dict[category]:
                forecast_2026 = kpi_dict[category]['forecast_values'][2] if len(kpi_dict[category]['forecast_values']) > 2 else 0
                st.metric(f"{category.replace('_', ' ').title()}", f"{forecast_2026:,.0f}K tonnes")
    
    # Model performance metrics
    st.subheader("📊 Model Performance")
    model_metrics = kpi_dict.get('model_performance', {})
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














def load_sample_data():
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
    if load_container_throughput:
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
        'size_teu': [8000, 12000, 6500],
        'waiting_time': [2.5, 1.8, 3.2]
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
        'kpis': pd.DataFrame(kpi_data)
    }


def get_real_berth_data():
    """Get real-time berth data from BerthManager"""
    if BerthManager and simpy:
        try:
            # Create a simulation environment and berth manager for real data
            env = simpy.Environment()
            berth_manager = BerthManager(env, PORT_CONFIG)
            
            # Get berth statistics
            berth_stats = berth_manager.get_berth_statistics()
            
            # Convert berth data to DataFrame format
            berths_list = []
            for berth_id, berth in berth_manager.berths.items():
                berths_list.append({
                    'berth_id': berth_id,
                    'name': f"Berth {berth_id}",
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
                'utilization_rate': berth_stats['utilization_rate'],
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
                    vessel_update_interval=300,   # 5 minutes
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
    
    # Simulation parameters
    st.sidebar.subheader("Simulation Settings")
    duration = st.sidebar.slider("Duration (hours)", 1, 168, SIMULATION_CONFIG['default_duration'])
    arrival_rate = st.sidebar.slider("Ship Arrival Rate (ships/hour)", 0.5, 5.0, float(SIMULATION_CONFIG['ship_arrival_rate']))
    
    # Simulation controls
    st.sidebar.subheader("Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start", disabled=st.session_state.simulation_running or not PortSimulation):
            if PortSimulation and SimulationController:
                # Initialize simulation controller
                config = SIMULATION_CONFIG.copy()
                config['ship_arrival_rate'] = arrival_rate
                
                simulation = PortSimulation(config)
                st.session_state.simulation_controller = SimulationController(simulation)
                st.session_state.simulation_controller.start(duration)
                st.session_state.simulation_running = True
                st.success("Simulation started!")
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["📊 Overview", "🚢 Ships & Berths", "📈 Analytics", "📦 Cargo Statistics", "🌊 Live Map", "🛳️ Live Vessels", "🌤️ Weather", "🏗️ Live Berths", "⚙️ Settings"])
    
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
        
        # Weather Summary Section
        st.subheader("🌤️ Current Weather Conditions")
        
        # Initialize weather integration for overview
        weather_col1, weather_col2, weather_col3 = st.columns(3)
        
        if HKObservatoryIntegration:
            try:
                weather_integration = HKObservatoryIntegration()
                current_weather = weather_integration.get_current_weather()
                
                if current_weather:
                    with weather_col1:
                        temp = current_weather.get('temperature', 'N/A')
                        st.metric("🌡️ Temperature", f"{temp}°C" if temp != 'N/A' else temp)
                        
                        wind_speed = current_weather.get('wind_speed', 'N/A')
                        st.metric("💨 Wind Speed", f"{wind_speed} km/h" if wind_speed != 'N/A' else wind_speed)
                    
                    with weather_col2:
                        humidity = current_weather.get('humidity', 'N/A')
                        st.metric("💧 Humidity", f"{humidity}%" if humidity != 'N/A' else humidity)
                        
                        visibility = current_weather.get('visibility', 'N/A')
                        st.metric("👁️ Visibility", f"{visibility} km" if visibility != 'N/A' else visibility)
                    
                    with weather_col3:
                        # Weather impact assessment
                        try:
                            from src.utils.weather_integration import get_weather_impact_for_simulation
                            impact_data = get_weather_impact_for_simulation(current_weather)
                            delay_factor = impact_data.get('delay_factor', 1.0)
                            
                            if delay_factor <= 1.1:
                                impact_status = "🟢 Normal"
                            elif delay_factor <= 1.3:
                                impact_status = "🟡 Caution"
                            else:
                                impact_status = "🔴 High Risk"
                            
                            st.metric("⚠️ Weather Impact", f"{delay_factor:.2f}x", help=impact_status)
                            
                            # Show current conditions
                            description = current_weather.get('description', 'No data')
                            st.info(f"**Conditions:** {description}")
                            
                        except ImportError:
                            st.metric("⚠️ Weather Impact", "N/A")
                            st.info("**Conditions:** Weather analysis unavailable")
                
                else:
                    with weather_col1:
                        st.metric("🌡️ Temperature", "N/A")
                        st.metric("💨 Wind Speed", "N/A")
                    with weather_col2:
                        st.metric("💧 Humidity", "N/A")
                        st.metric("👁️ Visibility", "N/A")
                    with weather_col3:
                        st.metric("⚠️ Weather Impact", "N/A")
                        st.warning("Weather data unavailable")
                        
            except Exception as e:
                with weather_col1:
                    st.metric("🌡️ Temperature", "N/A")
                    st.metric("💨 Wind Speed", "N/A")
                with weather_col2:
                    st.metric("💧 Humidity", "N/A")
                    st.metric("👁️ Visibility", "N/A")
                with weather_col3:
                    st.metric("⚠️ Weather Impact", "N/A")
                    st.warning(f"Weather service error: {str(e)}")
        else:
            with weather_col1:
                st.metric("🌡️ Temperature", "N/A")
                st.metric("💨 Wind Speed", "N/A")
            with weather_col2:
                st.metric("💧 Humidity", "N/A")
                st.metric("👁️ Visibility", "N/A")
            with weather_col3:
                st.metric("⚠️ Weather Impact", "N/A")
                st.info("Weather integration not available")
        
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
                st.subheader("Shipment Type Analysis")
                
                # Get shipment type data from time series
                shipment_ts = time_series_data.get('shipment_types', pd.DataFrame())
                
                if not shipment_ts.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**2023 Throughput Data**")
                        # Get latest year data (2023)
                        latest_data = shipment_ts.loc[2023] if 2023 in shipment_ts.index else shipment_ts.iloc[-1]
                        
                        breakdown_data = {
                            'Direct Shipment': latest_data['Direct shipment cargo'],
                            'Transhipment': latest_data['Transhipment cargo']
                        }
                        breakdown_df = pd.DataFrame(list(breakdown_data.items()),
                                                  columns=['Shipment Type', 'Throughput (000 tonnes)'])
                        st.dataframe(breakdown_df, use_container_width=True)
                        
                        # Calculate percentages
                        total = latest_data['Overall']
                        direct_pct = (latest_data['Direct shipment cargo'] / total) * 100
                        tranship_pct = (latest_data['Transhipment cargo'] / total) * 100
                    
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
                    
                    # Center the chart
                    chart_col1, chart_col2, chart_col3 = st.columns([0.1, 0.8, 0.1])
                    with chart_col2:
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
                    st.info("No time series data available")
            
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
                        st.plotly_chart(fig, use_container_width=True, key="forecast_chart")
                    
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
                     if 'top_cargo_types' in cargo_type_data:
                         st.write("**Top Cargo Types (2023)**")
                         cargo_types_list = cargo_type_data['top_cargo_types']
                         if cargo_types_list:
                             cargo_df = pd.DataFrame(cargo_types_list)
                             cargo_df.columns = ['Cargo Type', 'Throughput (000 tonnes)']
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
                     if 'top_locations' in location_data:
                         st.write("**Top Handling Locations (2023)**")
                         locations_list = location_data['top_locations']
                         if locations_list:
                             location_df = pd.DataFrame(locations_list)
                             location_df.columns = ['Handling Location', 'Throughput (000 tonnes)']
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
                
                col1, col2, col3, col4 = st.columns(4)
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
                with col4:
                    data_time = vessel_analysis.get('timestamps', {}).get('data_time', '')
                    if data_time:
                        date_str = data_time.split('T')[0] if 'T' in data_time else data_time[:10]
                        st.metric("Data Date", date_str)
                    else:
                        st.metric("Data Date", "N/A")
                
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
        st.subheader("🌤️ Weather Conditions & Impact")
        st.markdown("Real-time weather data and its impact on port operations")
        
        # Initialize weather integration
        weather_integration = None
        if HKObservatoryIntegration:
            try:
                weather_integration = HKObservatoryIntegration()
            except Exception as e:
                st.warning(f"Weather service unavailable: {str(e)}")
        
        if weather_integration:
            try:
                # Get current weather
                current_weather = weather_integration.get_current_weather()
                
                if current_weather:
                    # Weather overview
                    st.subheader("🌡️ Current Conditions")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        temp = current_weather.get('temperature', 'N/A')
                        st.metric("Temperature", f"{temp}°C" if temp != 'N/A' else temp)
                    with col2:
                        humidity = current_weather.get('humidity', 'N/A')
                        st.metric("Humidity", f"{humidity}%" if humidity != 'N/A' else humidity)
                    with col3:
                        wind_speed = current_weather.get('wind_speed', 'N/A')
                        st.metric("Wind Speed", f"{wind_speed} km/h" if wind_speed != 'N/A' else wind_speed)
                    with col4:
                        visibility = current_weather.get('visibility', 'N/A')
                        st.metric("Visibility", f"{visibility} km" if visibility != 'N/A' else visibility)
                    
                    # Weather description
                    description = current_weather.get('description', 'No description available')
                    st.info(f"**Current Conditions:** {description}")
                    
                    # Operational impact assessment
                    st.subheader("⚠️ Operational Impact Assessment")
                    
                    try:
                        from src.utils.weather_integration import get_weather_impact_for_simulation
                        impact_data = get_weather_impact_for_simulation(current_weather)
                        
                        impact_col1, impact_col2 = st.columns(2)
                        
                        with impact_col1:
                            st.write("**Impact Factors**")
                            
                            # Wind impact
                            wind_impact = impact_data.get('wind_impact', 1.0)
                            wind_status = "🟢 Normal" if wind_impact <= 1.1 else "🟡 Caution" if wind_impact <= 1.3 else "🔴 High Risk"
                            st.metric("Wind Impact", f"{wind_impact:.2f}x", help=wind_status)
                            
                            # Visibility impact
                            visibility_impact = impact_data.get('visibility_impact', 1.0)
                            vis_status = "🟢 Clear" if visibility_impact <= 1.1 else "🟡 Reduced" if visibility_impact <= 1.3 else "🔴 Poor"
                            st.metric("Visibility Impact", f"{visibility_impact:.2f}x", help=vis_status)
                            
                            # Overall delay factor
                            delay_factor = impact_data.get('delay_factor', 1.0)
                            delay_status = "🟢 Minimal" if delay_factor <= 1.1 else "🟡 Moderate" if delay_factor <= 1.3 else "🔴 Significant"
                            st.metric("Expected Delays", f"{delay_factor:.2f}x", help=delay_status)
                        
                        with impact_col2:
                            st.write("**Operational Recommendations**")
                            
                            recommendations = []
                            if wind_impact > 1.3:
                                recommendations.append("🌪️ High wind conditions - Consider suspending crane operations")
                            elif wind_impact > 1.1:
                                recommendations.append("💨 Moderate winds - Reduce crane operating speeds")
                            
                            if visibility_impact > 1.3:
                                recommendations.append("🌫️ Poor visibility - Implement enhanced navigation protocols")
                            elif visibility_impact > 1.1:
                                recommendations.append("👁️ Reduced visibility - Increase safety monitoring")
                            
                            if delay_factor > 1.2:
                                recommendations.append("⏰ Expect significant delays - Notify stakeholders")

                            if recommendations:
                                for rec in recommendations:
                                    st.warning(rec)
                            else:
                                st.success("✅ All systems normal. No specific recommendations.")

                    except ImportError:
                        st.error("Could not import weather impact module. Please check the installation.")
                    except Exception as e:
                        st.error(f"An error occurred during impact assessment: {str(e)}")

            except Exception as e:
                st.error(f"Failed to fetch or process weather data: {str(e)}")
        else:
            st.info("Weather integration is not configured.")


if __name__ == "__main__":
    main()