import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hk_port_digital_twin.src.utils.data_loader import load_focused_cargo_statistics, get_enhanced_cargo_analysis, get_time_series_data
from datetime import datetime

def render_cargo_statistics_tab():
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
            # Load data
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
                
                breakdown_data = {
                    'Seaborne': latest_data['Seaborne'],
                    'River': latest_data['River']
                }
                breakdown_df = pd.DataFrame(list(breakdown_data.items()),
                                            columns=['Transport Mode', 'Throughput (000 tonnes)'])
                st.dataframe(breakdown_df, use_container_width=True)
            
            with col2:
                st.write("**Percentage Distribution**")
                total = latest_data['Overall']
                seaborne_pct = (latest_data['Seaborne'] / total) * 100
                river_pct = (latest_data['River'] / total) * 100
                st.metric("Seaborne", f"{seaborne_pct:.1f}%")
                st.metric("River", f"{river_pct:.1f}%")
                
            # Show time series chart
            st.write("**Historical Trends (2014-2023)**")
            chart_data = transport_ts[['Seaborne', 'River']]
            
            # Create plotly chart
            fig = go.Figure()
            
            # Ensure years are integers
            years = [int(year) for year in chart_data.index]
            
            fig.add_trace(go.Scatter(
                x=years,
                y=chart_data['Seaborne'],
                mode='lines+markers',
                name='Seaborne',
                line=dict(color='purple')
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
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True, key="transport_trends_chart")
        
        else:
            st.info("No transport mode analysis data available")
            st.warning("Please ensure the Port Cargo Statistics CSV files are available in the raw_data directory.")

    with cargo_tab3:
        st.subheader("Time Series Analysis")
        
        if time_series_data:
            # Display time series charts
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
                
                # Direct shipment trend
                fig.add_trace(
                    go.Scatter(x=years, y=shipment_ts['Direct shipment cargo'], mode='lines+markers',
                             name='Direct Shipment', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Transhipment trend
                fig.add_trace(
                    go.Scatter(x=years, y=shipment_ts['Transhipment cargo'], mode='lines+markers',
                             name='Transhipment', line=dict(color='red')),
                    row=1, col=2
                )
            
            if not transport_ts.empty:
                # Ensure years are integers
                years = [int(year) for year in transport_ts.index.tolist()]
                
                seaborne_values = transport_ts['Seaborne']
                river_values = transport_ts['River']
                
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
        else:
            st.info("No time series data available")
            st.warning("Please ensure the Port Cargo Statistics CSV files are available in the raw_data directory.")

    with cargo_tab4:
        st.subheader("Forecasting")
        
        # Get forecasting data from cargo analysis
        forecasting_data = cargo_analysis.get('forecasting_analysis', {})
        
        if forecasting_data:
            st.write("**Forecast Summary**")
            
            # Display forecast metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                forecast_period = forecasting_data.get('forecast_period', 'N/A')
                st.metric("Forecast Period", forecast_period)
            
            with col2:
                model_accuracy = forecasting_data.get('model_accuracy', 0)
                st.metric("Model Accuracy", f"{model_accuracy:.1%}" if model_accuracy else "N/A")
            
            with col3:
                trend_direction = forecasting_data.get('trend_direction', 'stable')
                st.metric("Trend Direction", trend_direction.title())
            
            # Display forecast charts if available
            forecast_charts = forecasting_data.get('charts', {})
            if forecast_charts:
                for chart_name, chart_data in forecast_charts.items():
                    st.write(f"**{chart_name.replace('_', ' ').title()}**")
                    if isinstance(chart_data, pd.DataFrame):
                        st.line_chart(chart_data)
                    else:
                        st.warning(f"Could not display chart for {chart_name}")
            else:
                st.info("No forecast charts available.")
        else:
            st.info("No forecasting analysis available")
            st.warning("Please ensure the cargo analysis module is properly configured.")

    with cargo_tab5:
        st.subheader("Cargo Types")
        
        # Get cargo types data from analysis
        cargo_types_data = cargo_analysis.get('cargo_types_analysis', {})
        
        if cargo_types_data:
            # Display breakdown
            breakdown = cargo_types_data.get('breakdown', {})
            if breakdown:
                st.write("**Cargo Type Distribution**")
                
                # Create DataFrame for display
                cargo_df = pd.DataFrame(list(breakdown.items()), 
                                         columns=['Cargo Type', 'Volume (000 tonnes)'])
                
                st.dataframe(cargo_df, use_container_width=True)
            
            # Display trends
            trends = cargo_types_data.get('trends', {})
            if trends:
                st.write("**Cargo Type Trends**")
                for cargo_type, trend_data in trends.items():
                    if isinstance(trend_data, pd.DataFrame):
                        st.write(f"**{cargo_type.title()}**")
                        st.line_chart(trend_data)
        else:
            st.info("No cargo types analysis available")
            st.warning("Please ensure the cargo analysis module is properly configured.")

    with cargo_tab6:
        st.subheader("Locations")
        
        # Get location data from analysis
        locations_data = cargo_analysis.get('locations_analysis', {})
        
        if locations_data:
            # Display location breakdown
            breakdown = locations_data.get('breakdown', {})
            if breakdown:
                st.write("**Handling Location Distribution**")
                
                # Create DataFrame for display
                location_df = pd.DataFrame(list(breakdown.items()), 
                                         columns=['Location', 'Volume (000 tonnes)'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(location_df, use_container_width=True)
                
                with col2:
                    # Create bar chart
                    fig = go.Figure(data=[go.Bar(
                        x=location_df['Location'],
                        y=location_df['Volume (000 tonnes)']
                    )])
                    fig.update_layout(
                        xaxis_title="Location",
                        yaxis_title="Volume (000 tonnes)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No location analysis available")
            st.warning("Please ensure the cargo analysis module is properly configured.")