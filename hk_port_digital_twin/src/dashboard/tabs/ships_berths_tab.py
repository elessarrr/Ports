import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_ships_berths_tab():
    """
    Renders the ships and berths tab.
    """
    st.subheader("🚢 Ships & Berths")
    st.markdown("Real-time analysis of ships and berths including queue management, berth utilization, and vessel tracking.")

    # Create tabs for different operational views
    ops_tab1, ops_tab2, ops_tab3 = st.tabs(["🚢 Ship Queue", "🏗️ Berth Utilization", "📊 Live Operations"])

    with ops_tab1:
        st.subheader("🚢 Ship Queue Analysis")

        # Get simulation data if available
        simulation_data = getattr(st.session_state, 'simulation_data', None)

        if simulation_data and hasattr(simulation_data, 'ship_queue'):
            # Real simulation data
            queue_data = simulation_data.ship_queue

            # Queue metrics
            queue_col1, queue_col2, queue_col3, queue_col4 = st.columns(4)
            with queue_col1:
                st.metric("Ships in Queue", len(queue_data))
            with queue_col2:
                avg_wait = sum(ship.get('waiting_time', 0) for ship in queue_data) / max(len(queue_data), 1)
                st.metric("Avg Wait Time", f"{avg_wait:.1f} hrs")
            with queue_col3:
                priority_ships = sum(1 for ship in queue_data if ship.get('priority', 'normal') == 'high')
                st.metric("Priority Ships", priority_ships)
            with queue_col4:
                long_wait_ships = sum(1 for ship in queue_data if ship.get('waiting_time', 0) > 24)
                st.metric("Long Wait (>24h)", long_wait_ships)

            # Queue visualization
            if queue_data:
                queue_df = pd.DataFrame(queue_data)
                fig_queue = px.bar(
                    queue_df,
                    x='ship_id',
                    y='waiting_time',
                    color='ship_type',
                    title='Ship Queue - Waiting Times',
                    labels={'waiting_time': 'Waiting Time (hours)', 'ship_id': 'Ship ID'}
                )
                st.plotly_chart(fig_queue, use_container_width=True)

                # Queue details table
                st.subheader("📋 Queue Details")
                st.dataframe(queue_df, use_container_width=True)
            else:
                st.info("Ship queue is currently empty.")
        else:
            # Sample data for ship queue
            st.info("Displaying sample ship queue data. Run a simulation for live data.")
            queue_data = {
                'ship_id': [f'SHIP-{i:03d}' for i in range(5)],
                'ship_type': ['Container', 'Tanker', 'Bulker', 'Container', 'Cargo'],
                'waiting_time': [12, 8, 20, 5, 15],
                'priority': ['normal', 'high', 'normal', 'normal', 'low']
            }
            queue_df = pd.DataFrame(queue_data)

            # Sample queue visualization
            import plotly.express as px
            fig_queue = px.bar(
                queue_df,
                x='ship_id',
                y='waiting_time',
                color='ship_type',
                title='Ship Queue - Waiting Times (Sample Data)',
                labels={'waiting_time': 'Waiting Time (hours)', 'ship_id': 'Ship ID'}
            )
            st.plotly_chart(fig_queue, use_container_width=True)

            # Sample queue table
            st.subheader("📋 Queue Details")
            st.dataframe(queue_df, use_container_width=True)

    with ops_tab2:
        st.subheader("🏗️ Berth Utilization Analysis")
        st.info("Berth utilization analysis will be displayed here when simulation is running.")

        # Sample berth data
        berth_data = {
            'Berth ID': ['B001', 'B002', 'B003', 'B004', 'B005'],
            'Status': ['Occupied', 'Available', 'Occupied', 'Maintenance', 'Occupied'],
            'Current Ship': ['SHIP-001', '-', 'SHIP-003', '-', 'SHIP-005'],
            'Utilization %': [85, 0, 92, 0, 78]
        }
        berth_df = pd.DataFrame(berth_data)
        st.dataframe(berth_df, use_container_width=True)

    with ops_tab3:
        st.subheader("📊 Live Operations")
        st.info("Live operations dashboard will be displayed here when simulation is running.")