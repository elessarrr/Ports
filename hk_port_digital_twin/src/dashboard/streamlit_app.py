"""Streamlit Dashboard for Hong Kong Port Digital Twin

This module provides an interactive web dashboard for visualizing port simulation
results and real-time metrics using Streamlit and the visualization utilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import time

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.visualization import (
    create_port_layout_chart,
    create_ship_queue_chart,
    create_berth_utilization_chart,
    create_throughput_timeline,
    create_waiting_time_distribution,
    create_kpi_summary_chart
)
from src.core.port_simulation import PortSimulation
from src.core.simulation_controller import SimulationController
from src.core.berth_manager import BerthManager
from src.dashboard.marine_traffic_integration import MarineTrafficIntegration
from config.settings import SIMULATION_CONFIG, PORT_CONFIG, SHIP_TYPES
import simpy


# Add import for data_loader and real-time features
from src.utils.data_loader import (
    load_container_throughput, 
    get_cargo_breakdown_analysis, 
    load_vessel_arrivals, 
    get_vessel_queue_analysis,
    get_real_time_manager,
    RealTimeDataConfig
)

# Import weather integration
try:
    from src.utils.weather_integration import HKObservatoryIntegration
except ImportError:
    HKObservatoryIntegration = None


def _map_ship_category_to_type(ship_category: str) -> str:
    """Map detailed ship categories to simplified types for dashboard"""
    if not ship_category:
        return 'mixed'
    
    category_lower = ship_category.lower()
    
    if 'container' in category_lower:
        return 'container'
    elif any(term in category_lower for term in ['bulk', 'ore', 'cement', 'woodchip']):
        return 'bulk'
    else:
        return 'mixed'


def load_sample_data():
    """Load sample data for demonstration purposes with real vessel arrivals and container throughput data"""
    # Load real vessel arrivals data
    try:
        real_vessels = load_vessel_arrivals()
        vessel_queue_analysis = get_vessel_queue_analysis()
        
        # Use real vessel data if available, otherwise fall back to sample data
        if not real_vessels.empty:
            # Filter vessels currently in port for queue data
            active_vessels = real_vessels[real_vessels['status'] != 'departed'].copy()
            
            # Create queue data from real vessels at anchorage
            queue_vessels = active_vessels[active_vessels['location_type'] == 'anchorage'].copy()
            
            if not queue_vessels.empty:
                # Calculate waiting time based on arrival time
                now = datetime.now()
                queue_vessels['waiting_time'] = queue_vessels['arrival_time'].apply(
                    lambda x: (now - x).total_seconds() / 3600 if pd.notna(x) else 0
                )
                
                # Map ship categories to expected types
                mapped_ship_types = [_map_ship_category_to_type(cat) for cat in queue_vessels['ship_category']]
                
                # Prepare queue data
                queue_data = {
                    'ship_id': queue_vessels['call_sign'].fillna('Unknown').tolist(),
                    'name': queue_vessels['vessel_name'].fillna('Unknown Vessel').tolist(),
                    'ship_type': mapped_ship_types,
                    'arrival_time': queue_vessels['arrival_time'].tolist(),
                    'containers': [np.random.randint(50, 300) for _ in range(len(queue_vessels))],  # Estimated
                    'size_teu': [np.random.randint(2000, 8000) for _ in range(len(queue_vessels))],  # Estimated
                    'waiting_time': queue_vessels['waiting_time'].tolist(),
                    'priority': ['high' if wt > 4 else 'medium' if wt > 2 else 'low' for wt in queue_vessels['waiting_time']]
                }
            else:
                # No vessels in queue, use empty data
                queue_data = {
                    'ship_id': [], 'name': [], 'ship_type': [], 'arrival_time': [],
                    'containers': [], 'size_teu': [], 'waiting_time': [], 'priority': []
                }
        else:
            # Fallback to sample queue data
            queue_data = {
                'ship_id': ['SHIP_008', 'SHIP_009', 'SHIP_010'],
                'name': ['SHIP_008', 'SHIP_009', 'SHIP_010'],
                'ship_type': ['container', 'bulk', 'container'],
                'arrival_time': [datetime.now() - timedelta(hours=2), 
                                datetime.now() - timedelta(hours=1),
                                datetime.now() - timedelta(minutes=30)],
                'containers': [150, 80, 200],
                'size_teu': [3000, 5000, 4000],
                'waiting_time': [2.0, 1.0, 0.5],
                'priority': ['high', 'medium', 'low']
            }
    except Exception as e:
        print(f"Warning: Could not load real vessel data: {e}")
        # Fallback to sample queue data
        queue_data = {
            'ship_id': ['SHIP_008', 'SHIP_009', 'SHIP_010'],
            'name': ['SHIP_008', 'SHIP_009', 'SHIP_010'],
            'ship_type': ['container', 'bulk', 'container'],
            'arrival_time': [datetime.now() - timedelta(hours=2), 
                            datetime.now() - timedelta(hours=1),
                            datetime.now() - timedelta(minutes=30)],
            'containers': [150, 80, 200],
            'size_teu': [3000, 5000, 4000],
            'waiting_time': [2.0, 1.0, 0.5],
            'priority': ['high', 'medium', 'low']
        }
        vessel_queue_analysis = {}
    
    # Sample berth data (enhanced with real vessel data if available)
    berths_data = {
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
                                 end=datetime.now(), freq='H'),
            'containers_processed': np.random.randint(10, 100, 25),
            'ships_processed': np.random.randint(1, 8, 25)
        }
        timeline_data = pd.DataFrame(timeline_data)
    
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
        'berths': pd.DataFrame(berths_data),
        'queue': pd.DataFrame(queue_data),
        'timeline': timeline_data,  # Now using real data
        'waiting': pd.DataFrame(waiting_data),
        'kpis': pd.DataFrame(kpi_data),
        'vessel_queue_analysis': vessel_queue_analysis
    }


def get_real_berth_data():
    """Get real-time berth data from BerthManager"""
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
        if st.button("▶️ Start", disabled=st.session_state.simulation_running):
            # Initialize simulation controller
            config = SIMULATION_CONFIG.copy()
            config['ship_arrival_rate'] = arrival_rate
            
            simulation = PortSimulation(config)
            st.session_state.simulation_controller = SimulationController(simulation)
            st.session_state.simulation_controller.start(duration)
            st.session_state.simulation_running = True
            st.success("Simulation started!")
    
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
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Convert KPI DataFrame to expected dictionary format
            kpi_dict = {
                'average_waiting_time': 2.5,
                'average_berth_utilization': 0.75,
                'total_ships_processed': 85,
                'total_containers_processed': 1200,
                'average_queue_length': 3
            }
            fig_kpi = create_kpi_summary_chart(kpi_dict)
            st.plotly_chart(fig_kpi, use_container_width=True)
        
        with col2:
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
        fig_layout = create_port_layout_chart(data['berths'])
        st.plotly_chart(fig_layout, use_container_width=True)
    
    with tab2:
        st.subheader("Ships & Berths")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ship Queue")
            # Convert DataFrame to list of dictionaries for visualization
            queue_list = data['queue'].to_dict('records')
            fig_queue = create_ship_queue_chart(queue_list)
            st.plotly_chart(fig_queue, use_container_width=True)
            
            # Ship queue table
            st.dataframe(data['queue'], use_container_width=True)
        
        with col2:
            st.subheader("Berth Utilization")
            # Convert DataFrame to dictionary for visualization
            berth_util_dict = dict(zip(data['berths']['berth_id'], data['berths']['utilization']))
            fig_berth = create_berth_utilization_chart(berth_util_dict)
            st.plotly_chart(fig_berth, use_container_width=True)
            
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
            fig_timeline = create_throughput_timeline(data['timeline'])
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            st.subheader("Waiting Time Distribution")
            # Convert DataFrame to list for visualization
            waiting_times_list = data['waiting']['waiting_time'].tolist()
            fig_waiting = create_waiting_time_distribution(waiting_times_list)
            st.plotly_chart(fig_waiting, use_container_width=True)
    
    with tab4:
        st.subheader("📦 Port Cargo Statistics")
        st.markdown("Comprehensive analysis of Hong Kong port cargo throughput data")
        
        # Load cargo breakdown analysis
        try:
            with st.spinner("Loading cargo statistics..."):
                cargo_analysis = get_cargo_breakdown_analysis()
            
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
            cargo_tab1, cargo_tab2, cargo_tab3, cargo_tab4 = st.tabs(["📊 Shipment Types", "🚢 Transport Modes", "📦 Cargo Types", "📍 Locations"])
            
            with cargo_tab1:
                st.subheader("Shipment Type Analysis")
                shipment_data = cargo_analysis.get('shipment_type_analysis', {})
                
                if shipment_data:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**2023 Throughput Data**")
                        if 'direct_shipment_2023' in shipment_data and 'transhipment_2023' in shipment_data:
                            breakdown_data = {
                                'Direct Shipment': shipment_data['direct_shipment_2023'],
                                'Transhipment': shipment_data['transhipment_2023']
                            }
                            breakdown_df = pd.DataFrame(list(breakdown_data.items()), 
                                                      columns=['Shipment Type', 'Throughput (000 tonnes)'])
                            st.dataframe(breakdown_df, use_container_width=True)
                    
                    with col2:
                        st.write("**Percentage Distribution**")
                        if 'direct_percentage' in shipment_data and 'transhipment_percentage' in shipment_data:
                            st.metric("Direct Shipment", f"{shipment_data['direct_percentage']:.1f}%")
                            st.metric("Transhipment", f"{shipment_data['transhipment_percentage']:.1f}%")
                else:
                    st.info("No shipment type analysis data available")
            
            with cargo_tab2:
                st.subheader("Transport Mode Analysis")
                transport_data = cargo_analysis.get('transport_mode_analysis', {})
                
                if transport_data:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**2023 Transport Data**")
                        if 'seaborne_2023' in transport_data and 'river_2023' in transport_data:
                            transport_breakdown = {
                                'Seaborne': transport_data['seaborne_2023'],
                                'River': transport_data['river_2023']
                            }
                            transport_df = pd.DataFrame(list(transport_breakdown.items()), 
                                                      columns=['Transport Mode', 'Throughput (000 tonnes)'])
                            st.dataframe(transport_df, use_container_width=True)
                    
                    with col2:
                        st.write("**Modal Split Percentage**")
                        if 'seaborne_percentage' in transport_data and 'river_percentage' in transport_data:
                            st.metric("Seaborne", f"{transport_data['seaborne_percentage']:.1f}%")
                            st.metric("River", f"{transport_data['river_percentage']:.1f}%")
                else:
                    st.info("No transport mode analysis data available")
            
            with cargo_tab3:
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
            
            with cargo_tab4:
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
                
        except Exception as e:
            st.error(f"Error loading cargo statistics: {str(e)}")
            st.info("Please ensure the Port Cargo Statistics CSV files are available in the raw_data directory.")
    
    with tab5:
        st.subheader("🌊 Live Maritime Traffic")
        st.markdown("Real-time vessel tracking around Hong Kong waters")
        
        # Initialize MarineTraffic integration
        marine_traffic = MarineTrafficIntegration()
        
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
        
        with col1:
            # Display the embedded map
            marine_traffic.render_live_map_iframe(height=600)
            
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
                            st.plotly_chart(fig, use_container_width=True)
                
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
                            st.plotly_chart(fig, use_container_width=True)
                
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
                        st.plotly_chart(fig, use_container_width=True)
                
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
                            elif delay_factor > 1.1:
                                recommendations.append("⏱️ Minor delays expected - Monitor closely")
                            
                            if not recommendations:
                                recommendations.append("✅ Normal operations - No special precautions needed")
                            
                            for rec in recommendations:
                                st.write(f"• {rec}")
                    
                    except ImportError:
                        st.info("Weather impact analysis module not available")
                    
                    # Weather warnings
                    st.subheader("⚠️ Active Weather Warnings")
                    try:
                        warnings = weather_integration.get_weather_warnings()
                        if warnings:
                            for warning in warnings:
                                warning_type = warning.get('type', 'General')
                                warning_msg = warning.get('message', 'No details available')
                                if warning.get('severity') == 'high':
                                    st.error(f"🚨 **{warning_type}**: {warning_msg}")
                                elif warning.get('severity') == 'medium':
                                    st.warning(f"⚠️ **{warning_type}**: {warning_msg}")
                                else:
                                    st.info(f"ℹ️ **{warning_type}**: {warning_msg}")
                        else:
                            st.success("✅ No active weather warnings")
                    except Exception as e:
                        st.info("Weather warnings service temporarily unavailable")
                    
                    # Weather forecast
                    st.subheader("📅 Weather Forecast")
                    try:
                        forecast = weather_integration.get_forecast()
                        if forecast and len(forecast) > 0:
                            # Display forecast as a simple table
                            forecast_data = []
                            for item in forecast[:5]:  # Show next 5 periods
                                forecast_data.append({
                                    'Time': item.get('time', 'N/A'),
                                    'Temperature': f"{item.get('temperature', 'N/A')}°C",
                                    'Conditions': item.get('description', 'N/A'),
                                    'Wind': f"{item.get('wind_speed', 'N/A')} km/h"
                                })
                            
                            forecast_df = pd.DataFrame(forecast_data)
                            st.dataframe(forecast_df, use_container_width=True)
                        else:
                            st.info("Forecast data not available")
                    except Exception as e:
                        st.info("Weather forecast service temporarily unavailable")
                    
                    # Data source and update info
                    st.markdown("---")
                    st.caption("Data source: Hong Kong Observatory | Updates every 30 minutes")
                    
                else:
                    st.warning("Unable to retrieve current weather data")
                    
            except Exception as e:
                st.error(f"Error retrieving weather data: {str(e)}")
                st.info("Weather service may be temporarily unavailable. Please try again later.")
        else:
            st.warning("Weather integration service is not available")
            st.info("To enable weather features, ensure the weather integration module is properly configured.")
            
            # Show sample weather impact data
            st.subheader("📊 Sample Weather Impact Scenarios")
            
            sample_scenarios = [
                {"Condition": "Clear Weather", "Wind (km/h)": "< 20", "Visibility (km)": "> 10", "Delay Factor": "1.0x", "Status": "🟢 Normal"},
                {"Condition": "Light Rain", "Wind (km/h)": "20-30", "Visibility (km)": "5-10", "Delay Factor": "1.1x", "Status": "🟡 Caution"},
                {"Condition": "Heavy Rain", "Wind (km/h)": "30-50", "Visibility (km)": "2-5", "Delay Factor": "1.3x", "Status": "🟡 Caution"},
                {"Condition": "Typhoon Warning", "Wind (km/h)": "> 50", "Visibility (km)": "< 2", "Delay Factor": "2.0x+", "Status": "🔴 High Risk"}
            ]
            
            scenario_df = pd.DataFrame(sample_scenarios)
            st.dataframe(scenario_df, use_container_width=True)
    
    with tab8:
        st.subheader("🏗️ Real-Time Berth Occupancy Monitoring")
        st.markdown("Live monitoring of berth allocation, occupancy patterns, and operational efficiency")
        
        # Real-time berth status overview
        st.subheader("📊 Current Berth Status")
        
        # Get real-time berth data from BerthManager
        berth_data, berth_metrics = get_real_berth_data()
        
        # Extract metrics from real berth data
        total_berths = berth_metrics['total_berths']
        occupied_berths = berth_metrics['occupied_berths']
        available_berths = berth_metrics['available_berths']
        utilization_rate = berth_metrics['utilization_rate']
        
        # Display key metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Berths", total_berths)
        with metric_col2:
            st.metric("Occupied Berths", occupied_berths, delta=f"{occupied_berths - (total_berths - occupied_berths)}")
        with metric_col3:
            st.metric("Available Berths", available_berths)
        with metric_col4:
            st.metric("Utilization Rate", f"{utilization_rate:.1f}%", delta=f"{utilization_rate - 75:.1f}%")
        
        # Real-time berth occupancy visualization
        st.subheader("🔄 Live Berth Occupancy Grid")
        
        # Create a grid layout for berth visualization
        berth_cols = st.columns(4)
        
        for idx, (_, berth) in enumerate(berth_data.iterrows()):
            col_idx = idx % 4
            with berth_cols[col_idx]:
                # Determine berth status color and icon
                if berth['status'] == 'occupied':
                    status_color = "🔴"
                    status_text = "OCCUPIED"
                    ship_info = f"Ship: {berth.get('current_ship', 'Unknown')}"
                elif berth['status'] == 'available':
                    status_color = "🟢"
                    status_text = "AVAILABLE"
                    ship_info = "Ready for allocation"
                else:
                    status_color = "🟡"
                    status_text = "MAINTENANCE"
                    ship_info = "Under maintenance"
                
                # Create berth card
                st.markdown(f"""
                <div style="
                    border: 2px solid {'#ff4444' if berth['status'] == 'occupied' else '#44ff44' if berth['status'] == 'available' else '#ffaa44'};
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px 0;
                    background-color: {'#ffe6e6' if berth['status'] == 'occupied' else '#e6ffe6' if berth['status'] == 'available' else '#fff3e6'};
                ">
                    <h4>{status_color} {berth['berth_id']}</h4>
                    <p><strong>Status:</strong> {status_text}</p>
                    <p><strong>Type:</strong> {berth['berth_type'].title()}</p>
                    <p><strong>Capacity:</strong> {berth['max_capacity_teu']} TEU</p>
                    <p><strong>Utilization:</strong> {berth['utilization']:.1%}</p>
                    <p><em>{ship_info}</em></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Berth allocation timeline
        st.subheader("⏰ Recent Berth Allocation Activity")
        
        # Simulate recent allocation events (in real implementation, this would come from berth manager history)
        import random
        from datetime import timedelta
        
        recent_events = []
        for i in range(5):
            event_time = datetime.now() - timedelta(hours=random.randint(1, 24))
            event_type = random.choice(['allocation', 'release'])
            berth_id = random.choice(berth_data['berth_id'].tolist())
            ship_id = f"SHIP_{random.randint(1000, 9999)}"
            
            recent_events.append({
                'Timestamp': event_time.strftime('%Y-%m-%d %H:%M'),
                'Event': event_type.title(),
                'Berth': berth_id,
                'Ship': ship_id,
                'Duration': f"{random.randint(2, 48)} hours" if event_type == 'release' else 'Ongoing'
            })
        
        events_df = pd.DataFrame(recent_events)
        st.dataframe(events_df, use_container_width=True)
        
        # Berth performance analytics
        st.subheader("📈 Berth Performance Analytics")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.write("**Berth Utilization by Type**")
            
            # Use real berth type utilization data
            util_data = []
            for berth_type, type_info in berth_metrics['berth_types'].items():
                # Calculate utilization for this berth type
                type_berths = berth_data[berth_data['berth_type'] == berth_type]
                if len(type_berths) > 0:
                    util = type_berths['utilization'].mean() / 100  # Convert to decimal
                else:
                    util = 0
                
                util_data.append({
                    'Berth Type': berth_type.title(),
                    'Count': type_info,
                    'Average Utilization': f"{util:.1%}",
                    'Status': '🟢 Optimal' if util < 0.8 else '🟡 High' if util < 0.9 else '🔴 Critical'
                })
            
            util_df = pd.DataFrame(util_data)
            st.dataframe(util_df, use_container_width=True)
        
        with perf_col2:
            st.write("**Arrival Pattern Analysis**")
            
            # Simulate arrival patterns (in real implementation, this would analyze vessel data)
            pattern_data = [
                {'Time Period': 'Morning (06-12)', 'Avg Arrivals': '3.2/hour', 'Peak Load': '🟡 Moderate'},
                {'Time Period': 'Afternoon (12-18)', 'Avg Arrivals': '4.1/hour', 'Peak Load': '🔴 High'},
                {'Time Period': 'Evening (18-24)', 'Avg Arrivals': '2.8/hour', 'Peak Load': '🟢 Low'},
                {'Time Period': 'Night (00-06)', 'Avg Arrivals': '1.5/hour', 'Peak Load': '🟢 Low'}
            ]
            
            pattern_df = pd.DataFrame(pattern_data)
            st.dataframe(pattern_df, use_container_width=True)
        
        # Operational recommendations
        st.subheader("💡 Operational Recommendations")
        
        recommendations = []
        
        if utilization_rate > 90:
            recommendations.append("🔴 **High Utilization Alert**: Consider implementing queue management protocols")
        elif utilization_rate > 80:
            recommendations.append("🟡 **Moderate Load**: Monitor for potential bottlenecks")
        else:
            recommendations.append("🟢 **Normal Operations**: Current berth allocation is optimal")
        
        if available_berths < 2:
            recommendations.append("⚠️ **Low Availability**: Prepare contingency berth allocation plans")
        
        # Check for berth type imbalances
        container_berths = len(berth_data[berth_data['berth_type'] == 'container'])
        bulk_berths = len(berth_data[berth_data['berth_type'] == 'bulk'])
        
        if container_berths > 0 and bulk_berths > 0:
            container_util = berth_data[berth_data['berth_type'] == 'container']['utilization'].mean()
            bulk_util = berth_data[berth_data['berth_type'] == 'bulk']['utilization'].mean()
            
            if abs(container_util - bulk_util) > 0.3:
                recommendations.append("⚖️ **Load Imbalance**: Consider redistributing vessel types across berth categories")
        
        if not recommendations:
            recommendations.append("✅ **All Systems Optimal**: No immediate action required")
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
        
        # Auto-refresh indicator
        st.markdown("---")
        refresh_col1, refresh_col2 = st.columns([3, 1])
        
        with refresh_col1:
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: {'ON' if st.session_state.simulation_running else 'OFF'}")
        
        with refresh_col2:
            if st.button("🔄 Refresh Now", key="berth_refresh"):
                st.rerun()
    
    with tab9:
        st.subheader("Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Port Configuration")
            st.json(PORT_CONFIG)
        
        with col2:
            st.subheader("Ship Types")
            st.json(SHIP_TYPES)
        
        st.subheader("Current Simulation Config")
        current_config = SIMULATION_CONFIG.copy()
        current_config['ship_arrival_rate'] = arrival_rate
        st.json(current_config)
    
    # Footer
    st.markdown("---")
    st.markdown("*Last updated: {}*".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


if __name__ == "__main__":
    main()