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
from src.dashboard.marine_traffic_integration import MarineTrafficIntegration
from config.settings import SIMULATION_CONFIG, PORT_CONFIG, SHIP_TYPES


# Add import for data_loader
from src.utils.data_loader import load_container_throughput, get_cargo_breakdown_analysis


def load_sample_data():
    """Load sample data for demonstration purposes with real container throughput data"""
    # Sample berth data
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
    
    # Sample ship queue data
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
        'kpis': pd.DataFrame(kpi_data)
    }


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'simulation_controller' not in st.session_state:
        st.session_state.simulation_controller = None
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()


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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🚢 Ships & Berths", "📈 Analytics", "📦 Cargo Statistics", "🌊 Live Map", "⚙️ Settings"])
    
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
            st.metric("Active Ships", len(data['queue']))
            st.metric("Available Berths", len(data['berths'][data['berths']['status'] == 'available']))
            st.metric("Avg Waiting Time", "2.5 hrs")
            st.metric("Utilization Rate", "75%")
        
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