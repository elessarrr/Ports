import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import numpy as np

def load_vessel_arrivals():
    """
    This is a placeholder function to load vessel arrival data.
    In a real application, this would connect to a database or API.
    """
    return pd.DataFrame()

def render_live_vessels_tab():
    """
    Renders the live vessel arrivals tab.
    """
    st.subheader("🚢 Live Vessel Arrivals")
    st.markdown("Real-time vessel arrival data and analytics for Hong Kong port")

    try:
        from hk_port_digital_twin.src.utils.data_loader import load_combined_vessel_data
        vessel_data = load_combined_vessel_data()
        if vessel_data is None or vessel_data.empty:
            st.warning("⚠️ No vessel data available")
            st.info("Please ensure vessel data files are available in the data directory.")
            vessel_data = pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading combined vessel data: {str(e)}")
        vessel_data = pd.DataFrame()

    if not vessel_data.empty:
        # Current vessel status
        st.subheader("📊 Current Vessel Status")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_vessels = len(vessel_data)
            st.metric("Total Vessels", total_vessels)

        with col2:
            arriving_vessels = len(vessel_data[vessel_data.get('status', '') == 'arriving']) if 'status' in vessel_data.columns else 0
            st.metric("Arriving", arriving_vessels)

        with col3:
            in_port_vessels = len(vessel_data[vessel_data.get('status', '') == 'in_port']) if 'status' in vessel_data.columns else 0
            st.metric("In Port", in_port_vessels)

        with col4:
            departed_vessels = len(vessel_data[vessel_data.get('status', '') == 'departed']) if 'status' in vessel_data.columns else 0
            st.metric("Departed", departed_vessels)

        # Vessel locations
        st.subheader("📍 Vessel Locations")
        location_column = 'current_location' if 'current_location' in vessel_data.columns else 'location'
        if location_column in vessel_data.columns:
            location_counts = vessel_data[location_column].value_counts().head(10)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Location Distribution**")
                location_df = pd.DataFrame({
                    'Location': location_counts.index,
                    'Vessel Count': location_counts.values
                })
                st.dataframe(location_df, use_container_width=True)

            with col2:
                st.write("**Location Distribution Chart**")
                fig = px.pie(values=location_counts.values, names=location_counts.index,
                             title="Vessels by Location")
                st.plotly_chart(fig, use_container_width=True, key="location_pie_chart")

        # Ship categories
        st.subheader("🚢 Ship Categories")
        ship_type_column = 'ship_category' if 'ship_category' in vessel_data.columns else 'ship_type'
        if ship_type_column in vessel_data.columns:
            category_counts = vessel_data[ship_type_column].value_counts().head(10)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Category Distribution**")
                category_df = pd.DataFrame({
                    'Ship Type': category_counts.index,
                    'Count': category_counts.values
                })
                st.dataframe(category_df, use_container_width=True)

            with col2:
                st.write("**Category Distribution Chart**")
                fig = px.bar(x=category_counts.index, y=category_counts.values,
                             title="Vessels by Ship Type",
                             labels={'x': 'Ship Type', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True, key="category_bar_chart")

        # Recent activity
        st.subheader("⏰ Recent Activity")

        # Simulate recent arrivals data
        current_time = datetime.now()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            arrivals_6h = max(0, len(vessel_data) // 4)
            st.metric("Arrivals (6h)", arrivals_6h)

        with col2:
            arrivals_12h = max(0, len(vessel_data) // 3)
            st.metric("Arrivals (12h)", arrivals_12h)

        with col3:
            arrivals_24h = max(0, len(vessel_data) // 2)
            st.metric("Arrivals (24h)", arrivals_24h)

        with col4:
            st.write("**Activity Trend**")
            # Create a simple trend chart
            import numpy as np
            import plotly.graph_objects as go

            hours = list(range(24))
            arrivals = [max(0, int(len(vessel_data) * (0.3 + 0.7 * abs(np.sin(h/4))))) for h in hours]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=arrivals, mode='lines+markers', name='Arrivals'))
            fig.update_layout(
                title="24h Arrival Trend",
                xaxis_title="Hour",
                yaxis_title="Arrivals",
                height=200,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True, key="activity_trend_chart")

        # Detailed vessel table
        st.subheader("📋 Arriving and Departing Vessels - Detailed Information")

        # Add filters
        col1, col2, col3 = st.columns(3)

        with col1:
            if ship_type_column in vessel_data.columns:
                ship_types = ['All'] + list(vessel_data[ship_type_column].unique())
                selected_type = st.selectbox("Filter by Ship Type", ship_types)
            else:
                selected_type = 'All'

        with col2:
            if location_column in vessel_data.columns:
                locations = ['All'] + list(vessel_data[location_column].unique())
                selected_location = st.selectbox("Filter by Location", locations)
            else:
                selected_location = 'All'

        with col3:
            show_all = st.checkbox("Show All Columns", value=False)

        # Apply filters
        filtered_data = vessel_data.copy()

        if selected_type != 'All' and ship_type_column in vessel_data.columns:
            filtered_data = filtered_data[filtered_data[ship_type_column] == selected_type]

        if selected_location != 'All' and location_column in vessel_data.columns:
            filtered_data = filtered_data[filtered_data[location_column] == selected_location]

        # Display table
        if not show_all:
            # Show only key columns
            display_columns = []
            for col in ['vessel_name', 'ship_type', 'location', 'arrival_time', 'status']:
                if col in filtered_data.columns:
                    display_columns.append(col)

            if display_columns:
                st.dataframe(filtered_data[display_columns], use_container_width=True)
            else:
                st.dataframe(filtered_data, use_container_width=True)
        else:
            st.dataframe(filtered_data, use_container_width=True)

        # Export functionality
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export to CSV"):
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"vessel_arrivals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

        with col2:
            st.metric("Total Records", len(filtered_data))

    else:
        st.info("No vessel arrivals data available. Please check data sources.")

        # Show sample data structure
        st.subheader("📋 Expected Data Structure")
        sample_data = pd.DataFrame({
            'vessel_name': ['Sample Vessel 1', 'Sample Vessel 2'],
            'ship_type': ['Container Ship', 'Bulk Carrier'],
            'location': ['Kwai Tsing', 'Western Anchorage'],
            'arrival_time': [datetime.now(), datetime.now() - pd.Timedelta(hours=2)],
            'status': ['arrived', 'anchored']
        })
        st.dataframe(sample_data, use_container_width=True)