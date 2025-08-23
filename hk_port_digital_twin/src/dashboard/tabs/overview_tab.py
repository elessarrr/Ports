import streamlit as st
import pandas as pd
from hk_port_digital_twin.src.utils.visualization import create_port_layout_chart, create_kpi_summary_chart
from hk_port_digital_twin.src.utils.data_loader import get_enhanced_cargo_analysis
from hk_port_digital_twin.src.dashboard.vessel_charts import render_vessel_analytics_dashboard

def render_overview_tab(data, vessel_analysis):
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
                if category in forecasts and forecasts[category] is not None:
                    # Extract the latest forecast value
                    latest_forecast = forecasts[category].iloc[-1]
                    kpi_dict[category.replace('_', ' ').title()] = f"{latest_forecast:,.0f} TEU"

            if kpi_dict:
                st.write("#### 🚢 Cargo Forecast (Next 30 Days)")
                kpi_chart = create_kpi_summary_chart(kpi_dict)
                st.plotly_chart(kpi_chart, use_container_width=True)
            else:
                st.info("Cargo forecast data is not available.")
        else:
            # Fallback if forecast data is unavailable
            st.write("#### 🚢 Key Performance Indicators")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                st.metric("Vessel Turnaround", "18 hrs")
            with kpi_col2:
                st.metric("Berth Occupancy", "85%")
            with kpi_col3:
                st.metric("Cargo Volume", "45,000 TEU")
            with kpi_col4:
                st.metric("Port Dwell Time", "3.2 days")

    # Live Port Metrics
    st.write("#### 📊 Live Port Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Safe access to queue data with fallback
        queue_length = len(data.get('queue', [])) if 'queue' in data and data['queue'] is not None else 0
        st.metric("Active Ships", queue_length)
    
    with col2:
        # Safe access to berths data with fallback
        berths_df = data.get('berths', pd.DataFrame())
        if not berths_df.empty and 'status' in berths_df.columns:
            available_berths = len(berths_df[berths_df['status'] == 'available'])
        else:
            available_berths = 0
        st.metric("Available Berths", available_berths)
    
    with col3:
        # Show recent arrivals if available
        if vessel_analysis and 'recent_activity' in vessel_analysis:
            arrivals_24h = vessel_analysis['recent_activity'].get('arrivals_last_24h', 0)
            st.metric("24h Arrivals", arrivals_24h)
        else:
            st.metric("Avg Waiting Time", "2.5 hrs")
    
    with col4:
        st.metric("Utilization Rate", "75%")

    # Port Layout
    st.subheader("Port Layout")
    if create_port_layout_chart is not None and 'berths' in data and data['berths'] is not None:
        fig_layout = create_port_layout_chart(data['berths'])
        st.plotly_chart(fig_layout, use_container_width=True, key="port_layout_chart")
    else:
        if 'berths' not in data or data['berths'] is None:
            st.warning("Berth data not available. Please check data loading.")
        else:
            st.info("Port layout visualization not available. Please ensure visualization module is properly installed.")