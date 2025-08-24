"""
This is the main entry point for the Streamlit application.
It is designed to be run from the root of the `hk_port_digital_twin` directory.
This script initializes the Streamlit interface and orchestrates the various
dashboard components.
"""

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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import RealTimeDataConfig, get_real_time_manager, load_container_throughput, load_vessel_arrivals, load_berth_configurations, initialize_vessel_data_pipeline, load_all_vessel_data, get_comprehensive_vessel_analysis, load_combined_vessel_data
from config.settings import SIMULATION_CONFIG, get_enhanced_simulation_config
from src.core.port_simulation import PortSimulation
from src.core.simulation_controller import SimulationController
from src.core.berth_manager import BerthManager
from src.scenarios import ScenarioManager, list_available_scenarios
from src.utils.visualization import create_kpi_summary_chart, create_port_layout_chart, create_ship_queue_chart, create_berth_utilization_chart, create_throughput_timeline, create_waiting_time_distribution
# Weather integration disabled for feature removal
# from src.utils.weather_integration import HKObservatoryIntegration
HKObservatoryIntegration = None  # Disabled
from src.utils.data_loader import load_focused_cargo_statistics, get_enhanced_cargo_analysis, get_time_series_data
from src.dashboard.scenario_tab_consolidation import ConsolidatedScenariosTab
from src.dashboard.vessel_charts import render_vessel_analytics_dashboard
from src.dashboard.tabs.cargo_statistics_tab import render_cargo_statistics_tab
from src.dashboard.tabs.live_vessels_tab import render_live_vessels_tab
from src.dashboard.tabs.ships_berths_tab import render_ships_berths_tab
from src.dashboard.tabs.scenario_analysis_tab import render_scenario_analysis_tab
from src.dashboard.executive_dashboard import ExecutiveDashboard
from src.utils.strategic_visualization import StrategicVisualization, render_strategic_controls
from src.core.strategic_simulation_controller import StrategicSimulationController
# from src.dashboard.unified_simulations_tab import UnifiedSimulationsTab  # Commented out - tab hidden

try:
    from src.dashboard.marine_traffic_integration import MarineTrafficIntegration
except ImportError:
    MarineTrafficIntegration = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')


@st.cache_data
def load_sample_data(scenario='normal', use_real_throughput_data=True):
    """Load sample data based on scenario"""
    # Define scenario-based parameters with distinct, non-overlapping ranges
    scenario_params = {
        'peak': {
            'queue_multiplier': 2,
            'utilization_range': (85, 100),  # High utilization range
            'occupied_berths_range': (6, 8),  # High occupancy
            'waiting_time_multiplier': 1.5
        },
        'low': {
            'queue_multiplier': 0.5,
            'utilization_range': (25, 45),  # Low utilization range
            'occupied_berths_range': (1, 3),  # Low occupancy
            'waiting_time_multiplier': 0.7
        },
        'normal': {
            'queue_multiplier': 1,
            'utilization_range': (60, 80),  # Medium utilization range
            'occupied_berths_range': (4, 5),  # Medium occupancy
            'waiting_time_multiplier': 1
        }
    }
    
    params = scenario_params.get(scenario, scenario_params['normal'])
    
    # Randomly determine the number of occupied berths within the defined range
    num_berths = 8
    num_occupied = np.random.randint(params['occupied_berths_range'][0], params['occupied_berths_range'][1] + 1)
    
    # Ensure we don't exceed total berths and always have at least one maintenance berth
    num_occupied = min(num_occupied, num_berths - 1)  # Reserve at least 1 berth for maintenance
    num_available = num_berths - num_occupied - 1  # 1 berth for maintenance
    
    # Create a list of statuses with exact length matching num_berths
    statuses = ['occupied'] * num_occupied + ['available'] * num_available + ['maintenance']
    np.random.shuffle(statuses)
    
    # Generate random utilization for occupied berths
    utilization_values = []
    for status in statuses:
        if status == 'occupied':
            utilization_values.append(np.random.randint(params['utilization_range'][0], params['utilization_range'][1] + 1))
        else:
            utilization_values.append(0)

    berth_data = {
        'berth_id': [f'Berth_{chr(65+i//4)}{i%4+1}' for i in range(num_berths)],
        'name': [f'Berth {chr(65+i//4)}{i%4+1}' for i in range(num_berths)],
        'x': [1, 2, 3, 4, 1, 2, 3, 4],
        'y': [1, 1, 1, 1, 2, 2, 3, 3],
        'status': statuses,
        'ship_id': [f'SHIP_{i:03d}' if statuses[i] == 'occupied' else None for i in range(num_berths)],
        'utilization': utilization_values,
        'berth_type': ['container', 'container', 'container', 'container', 'bulk', 'bulk', 'mixed', 'mixed'],
        'crane_count': [4, 3, 4, 2, 2, 2, 3, 3],
        'max_capacity_teu': [5000, 4000, 5000, 3000, 6000, 6000, 4500, 4500],
        'is_occupied': [status == 'occupied' for status in statuses]
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
    num_ships_in_queue = int(3 * params['queue_multiplier'])
    ship_queue_data = {
        'ship_id': [f'SHIP_{i:03d}' for i in range(1, num_ships_in_queue + 1)],
        'name': [f'Ship {i}' for i in range(1, num_ships_in_queue + 1)],
        'ship_type': np.random.choice(['container', 'bulk'], num_ships_in_queue) if num_ships_in_queue > 0 else [],
        'arrival_time': [datetime.now() - timedelta(hours=i) for i in range(num_ships_in_queue, 0, -1)],
        'containers': np.random.randint(100, 300, num_ships_in_queue) if num_ships_in_queue > 0 else [],
        'size_teu': np.random.randint(5000, 15000, num_ships_in_queue) if num_ships_in_queue > 0 else [],
        'waiting_time': np.random.uniform(1.0, 5.0, num_ships_in_queue) * params['waiting_time_multiplier'] if num_ships_in_queue > 0 else [],
        'priority': np.random.choice(['high', 'medium', 'low'], num_ships_in_queue) if num_ships_in_queue > 0 else []
    }
    
    # Sample waiting time data
    waiting_data = {
        'ship_id': [f'SHIP_{i:03d}' for i in range(1, 21)],
        'waiting_time': np.random.exponential(2, 20) * params['waiting_time_multiplier'],
        'ship_type': np.random.choice(['container', 'bulk', 'mixed'], 20)
    }

    # Sample KPI data
    kpi_data = {
        'metric': ['Average Waiting Time', 'Berth Utilization', 'Throughput Rate', 'Queue Length'],
        'value': [2.5 * params['waiting_time_multiplier'], np.mean(params['utilization_range']), 85, 3 * params['queue_multiplier']],
        'unit': ['hours', '%', 'containers/hour', 'ships'],
        'target': [2.0, 80, 90, 2],
        'status': ['warning', 'good', 'good', 'warning']
    }

    return {
        'berths': pd.DataFrame(berth_data),
        'queue': pd.DataFrame(ship_queue_data),
        'timeline': timeline_data,
        'waiting': pd.DataFrame(waiting_data),
        'kpis': pd.DataFrame(kpi_.pd),
        'vessel_queue_analysis': {
            'total_vessels_waiting': len(ship_queue_data['ship_id']),
            'average_waiting_time': np.mean(ship_queue_data['waiting_time']) if num_ships_in_queue > 0 else 0,
            'queue_by_type': pd.DataFrame(ship_queue_data)['ship_type'].value_counts().to_dict() if num_ships_in_queue > 0 else {},
            'priority_distribution': pd.DataFrame(ship_queue_data)['priority'].value_counts().to_dict() if num_ships_in_queue > 0 else {}
        }
    }


@st.cache_data
def get_real_berth_data(berth_config):
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
                    'name': berth.name,
                    'status': 'occupied' if berth.is_occupied else 'available',
                    'ship_id': berth.current_ship_id,
                    'utilization': berth.get_utilization(),
                    'berth_type': berth.berth_type,
                    'crane_count': berth.crane_count,
                    'max_capacity_teu': berth.max_capacity_teu,
                    'is_occupied': berth.is_occupied
                })
            
            return pd.DataFrame(berths_list)
        except Exception as e:
            st.error(f"Error getting real berth data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def main():
    """Main function to run the Streamlit dashboard"""
    st.set_page_config(page_title="HK Port Digital Twin", layout="wide")

    # Load data
    # data = load_sample_data()
    
    # Sidebar for scenario selection
    st.sidebar.title("Simulation Controls")
    
    # Add a refresh button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Scenario selection
    # scenario = st.sidebar.selectbox("Select Scenario", ["normal", "peak", "low"], index=0)
    
    # Load data based on selected scenario
    # data = load_sample_data(scenario)

    # Initialize session state for tab management
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Executive Dashboard"

    # Custom radio buttons for tab selection
    tabs = ["Executive Dashboard", "Live Vessels", "Vessel Analytics", "Cargo Statistics", "Ships & Berths", "Consolidated Scenarios", "Strategic Simulations"]
    
    # Display radio buttons and get the selected tab
    selected_tab = st.sidebar.radio("Navigate", tabs, index=tabs.index(st.session_state.active_tab))

    # Update the active tab in session state if it has changed
    if selected_tab != st.session_state.active_tab:
        st.session_state.active_tab = selected_tab
        st.rerun()

    # Display the selected tab content
    if st.session_state.active_tab == "Executive Dashboard":
        st.header("Executive Dashboard")
        # Initialize and render the Executive Dashboard
        executive_dashboard = ExecutiveDashboard()
        executive_dashboard.render()

    elif st.session_state.active_tab == "Live Vessels":
        st.header("Live Vessel & Port Status")
        render_live_vessels_tab()

    elif st.session_state.active_tab == "Vessel Analytics":
        st.header("Vessel Analytics Dashboard")
        render_vessel_analytics_dashboard()

    elif st.session_state.active_tab == "Cargo Statistics":
        st.header("Cargo Statistics Dashboard")
        render_cargo_statistics_tab()

    elif st.session_state.active_tab == "Ships & Berths":
        st.header("Ships & Berths Overview")
        render_ships_berths_tab()

    elif st.session_state.active_tab == "Consolidated Scenarios":
        st.header("Consolidated Scenario Analysis")
        # Initialize and render the Consolidated Scenarios Tab
        consolidated_scenarios_tab = ConsolidatedScenariosTab()
        consolidated_scenarios_tab.render()

    elif st.session_state.active_tab == "Strategic Simulations":
        st.header("Strategic Investment & Policy Simulations")
        
        # Initialize the strategic simulation controller
        if 'strategic_sim_controller' not in st.session_state:
            st.session_state.strategic_sim_controller = StrategicSimulationController()
        
        # Render strategic controls and get the simulation parameters
        simulation_params = render_strategic_controls()
        
        # Run simulation and display results if the button is clicked
        if simulation_params:
            # This will only be true when the "Run Strategic Simulation" button is clicked
            controller = st.session_state.strategic_sim_controller
            
            # Run the simulation with the specified parameters
            controller.run_simulation(
                investment_scenario=simulation_params['investment_scenario'],
                policy_scenario=simulation_params['policy_scenario'],
                simulation_years=simulation_params['simulation_years']
            )
            
            # Get the results from the controller
            results = controller.get_results()
            
            # Display the results using the visualization class
            if results:
                st.subheader("Simulation Results")
                viz = StrategicVisualization(results)
                
                # Create two columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(viz.plot_kpi_trends(), use_container_width=True)
                    st.plotly_chart(viz.plot_roi_and_costs(), use_container_width=True)
                
                with col2:
                    st.plotly_chart(viz.plot_throughput_and_capacity(), use_container_width=True)
                    st.plotly_chart(viz.plot_berth_utilization_and_waiting_time(), use_container_width=True)
                    
                # Display detailed results in a table
                st.write(viz.get_results_dataframe())


if __name__ == "__main__":
    main()