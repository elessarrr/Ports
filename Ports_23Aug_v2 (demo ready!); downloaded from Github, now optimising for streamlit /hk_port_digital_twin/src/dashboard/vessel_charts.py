# Comments for context:
# This module contains vessel analytics chart functions that were migrated from the 'Live Vessels' tab
# to be used in the new 'Vessel Analytics' sub-tab within the 'Cargo Statistics' tab.
# The migration follows a modular design pattern to improve code organization and maintainability.
# Each function renders a specific vessel-related chart using Plotly Express for consistency.

import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from utils.data_loader import load_vessel_arrivals

def render_vessel_location_distribution(vessel_analysis: Dict[str, Any]) -> None:
    """
    Renders a pie chart showing the distribution of vessels by location.
    
    This chart displays the percentage breakdown of vessels across different
    locations/terminals in the port, helping users understand vessel concentration.
    
    Args:
        vessel_analysis: Dictionary containing vessel analysis data with 'location_distribution' key
    
    Returns:
        None (renders chart directly to Streamlit)
    """
    try:
        # Extract location distribution data
        location_data = vessel_analysis.get('location_distribution', {})
        
        if not location_data:
            st.warning("No vessel location data available")
            return
        
        # Convert to DataFrame for Plotly
        location_df = pd.DataFrame([
            {'Location': location, 'Count': count}
            for location, count in location_data.items()
        ])
        
        if location_df.empty:
            st.warning("No vessel location data to display")
            return
        
        # Create pie chart using Plotly Express
        fig = px.pie(
            location_df,
            values='Count',
            names='Location',
            title='Vessel Location Distribution'
        )
        
        # Update layout for better appearance
        fig.update_layout(
            height=400,
            showlegend=True,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering vessel location distribution: {str(e)}")

def render_ship_category_distribution(vessel_analysis: Dict[str, Any]) -> None:
    """
    Renders a bar chart showing the distribution of vessels by ship category/type.
    
    This chart displays the count of different ship types (e.g., Container, Bulk, Tanker)
    currently in the port, helping users understand the vessel mix.
    
    Args:
        vessel_analysis: Dictionary containing vessel analysis data with 'category_distribution' key
    
    Returns:
        None (renders chart directly to Streamlit)
    """
    try:
        # Extract category distribution data
        category_data = vessel_analysis.get('category_distribution', {})
        
        if not category_data:
            st.warning("No ship category data available")
            return
        
        # Convert to DataFrame for Plotly
        category_df = pd.DataFrame([
            {'Category': category, 'Count': count}
            for category, count in category_data.items()
        ])
        
        if category_df.empty:
            st.warning("No ship category data to display")
            return
        
        # Create bar chart using Plotly Express
        fig = px.bar(
            category_df,
            x='Category',
            y='Count',
            title='Ship Category Distribution',
            color='Category'
        )
        
        # Update layout for better appearance
        fig.update_layout(
            height=400,
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            xaxis_title='Ship Category',
            yaxis_title='Number of Vessels'
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering ship category distribution: {str(e)}")

def render_arrival_activity_trend(vessel_analysis: Dict[str, Any]) -> None:
    """
    Renders a line chart showing the trend of vessel arrival activity over time.
    
    This chart displays the number of vessel arrivals over recent time periods,
    helping users identify patterns and trends in port activity.
    
    Args:
        vessel_analysis: Dictionary containing vessel analysis data with 'activity_trend' key
    
    Returns:
        None (renders chart directly to Streamlit)
    """
    try:
        # Extract activity trend data
        activity_data = vessel_analysis.get('activity_trend', [])
        
        # Log data status for debugging
        st.info(f"📊 Data Status: Found {len(activity_data) if hasattr(activity_data, '__len__') else 0} activity trend data points")
        
        if not activity_data:
            st.warning("No vessel activity trend data available")
            st.info("💡 This could happen if: 1) No vessels have timestamps within the last 7 days, 2) Timestamp format issues, or 3) Data loading problems")
            
            # Show available keys for debugging
            available_keys = list(vessel_analysis.keys()) if isinstance(vessel_analysis, dict) else []
            st.write(f"Available data keys: {available_keys}")
            return
        
        # Data successfully loaded
        
        # Convert to DataFrame for Plotly
        # Handle both list of dicts and dict formats
        if isinstance(activity_data, list):
            # List of dictionaries format: [{'time': datetime, 'arrivals': count}, ...]
            activity_df = pd.DataFrame([
                {'Date': item.get('time'), 'Arrivals': item.get('arrivals', 0)}
                for item in activity_data
                if item.get('time') is not None
            ])
        elif isinstance(activity_data, dict):
            # Dictionary format: {date: count, ...}
            activity_df = pd.DataFrame([
                {'Date': date, 'Arrivals': count}
                for date, count in activity_data.items()
            ])
        else:
            st.warning("Invalid activity trend data format")
            return
        
        if activity_df.empty:
            st.warning("No vessel activity data to display")
            return
        
        # Ensure Date column is properly formatted
        activity_df['Date'] = pd.to_datetime(activity_df['Date'], errors='coerce')
        activity_df = activity_df.dropna(subset=['Date']).sort_values('Date')
        
        if activity_df.empty:
            st.warning("No valid activity data to display")
            return
        
        # Create line chart using Plotly Express
        fig = px.line(
            activity_df,
            x='Date',
            y='Arrivals',
            title='Arrival Activity Trend',
            markers=True
        )
        
        # Update layout for better appearance
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            xaxis_title='Date',
            yaxis_title='Number of Arrivals'
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering arrival activity trend: {str(e)}")

def _extract_latest_timestamp(vessel_analysis: Dict[str, Any]) -> str:
    """
    Extract the latest timestamp from vessel analysis data across all XML files.
    
    Args:
        vessel_analysis: Dictionary containing vessel analysis data
    
    Returns:
        str: Formatted latest timestamp or empty string if no timestamps found
    """
    try:
        latest_timestamp = None
        
        # Check if we have file_breakdown with timestamps
        if isinstance(vessel_analysis, dict) and 'file_breakdown' in vessel_analysis:
            for file_name, file_data in vessel_analysis['file_breakdown'].items():
                # Check both latest and earliest timestamps to find the most recent
                for ts_key in ['latest_timestamp', 'earliest_timestamp']:
                    timestamp_str = file_data.get(ts_key)
                    if timestamp_str:
                        try:
                            # Parse the ISO timestamp
                            ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if latest_timestamp is None or ts > latest_timestamp:
                                latest_timestamp = ts
                        except (ValueError, TypeError):
                            continue
        
        # Fallback: check for analysis_timestamp
        if latest_timestamp is None and isinstance(vessel_analysis, dict) and 'analysis_timestamp' in vessel_analysis:
            analysis_ts = vessel_analysis.get('analysis_timestamp')
            if analysis_ts:
                try:
                    latest_timestamp = datetime.fromisoformat(analysis_ts.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    pass
        
        # Fallback: check for timestamp column in DataFrame format
        elif isinstance(vessel_analysis, pd.DataFrame) and 'timestamp' in vessel_analysis.columns:
            timestamps = vessel_analysis['timestamp'].dropna()
            if not timestamps.empty:
                latest_timestamp = timestamps.max()
        
        # Format the timestamp for display
        if latest_timestamp:
            return latest_timestamp.strftime('%d-%b-%Y %H:%M')
        else:
            return ""
            
    except Exception as e:
        st.error(f"Error extracting timestamp: {str(e)}")
        return ""

def render_arriving_ships_list() -> None:
    """
    Renders a list of ships currently arriving at the port.
    
    This function displays detailed information about vessels that are arriving,
    including vessel name, ship type, agent, location, and arrival time.
    
    Returns:
        None (renders list directly to Streamlit)
    """
    try:
        # Load combined vessel data to get arriving ships
        from utils.data_loader import load_combined_vessel_data
        vessel_data = load_combined_vessel_data()
        
        if vessel_data.empty:
            st.info("No vessel data available")
            return
        
        # Filter for vessels that are currently arriving
        arriving_ships = vessel_data[vessel_data['status'] == 'arriving'].copy()
        
        if arriving_ships.empty:
            st.info("No ships currently arriving")
            return
        
        # Sort by arrival time (most recent first)
        arriving_ships = arriving_ships.sort_values('arrival_time', ascending=False, na_position='last')
        
        # Display the count
        st.write(f"**{len(arriving_ships)} ships currently arriving:**")
        
        # Prepare data for display
        display_data = arriving_ships[[
            'vessel_name', 'ship_type', 'agent_name', 'current_location', 'arrival_time'
        ]].copy()
        
        # Format arrival time for better display
        display_data['arrival_time'] = display_data['arrival_time'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Rename columns for better display
        display_data.columns = ['Vessel Name', 'Ship Type', 'Agent', 'Current Location', 'Arrival Time']
        
        # Display as a table
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )
        
    except Exception as e:
        st.error(f"Error loading arriving ships data: {str(e)}")


def render_vessel_analytics_dashboard(vessel_analysis) -> None:
    """
    Renders the complete vessel analytics dashboard with all three charts.
    
    This is the main function that orchestrates the display of all vessel analytics
    charts in a structured layout. It's designed to be called from the main
    Streamlit app within the 'Vessel Analytics' sub-tab.
    
    Args:
        vessel_analysis: DataFrame containing vessel data or Dictionary containing vessel analysis data
    
    Returns:
        None (renders dashboard directly to Streamlit)
    """
    try:
        # Check if vessel analysis data is available
        if vessel_analysis is None:
            st.warning("No vessel analysis data available")
            return
        
        # Handle DataFrame input by converting to expected dictionary format
        if isinstance(vessel_analysis, pd.DataFrame):
            if vessel_analysis.empty:
                st.warning("No vessel analysis data available")
                return
            
            # Convert DataFrame to expected dictionary format
            processed_data = {
                'location_distribution': {},
                'category_distribution': {},
                'activity_trend': []
            }
            
            # Process location distribution
            if 'location_type' in vessel_analysis.columns:
                processed_data['location_distribution'] = vessel_analysis['location_type'].value_counts().to_dict()
            elif 'current_location' in vessel_analysis.columns:
                processed_data['location_distribution'] = vessel_analysis['current_location'].value_counts().to_dict()
            elif 'location' in vessel_analysis.columns:
                processed_data['location_distribution'] = vessel_analysis['location'].value_counts().to_dict()
            
            # Process category distribution
            if 'ship_category' in vessel_analysis.columns:
                processed_data['category_distribution'] = vessel_analysis['ship_category'].value_counts().to_dict()
            elif 'ship_type' in vessel_analysis.columns:
                processed_data['category_distribution'] = vessel_analysis['ship_type'].value_counts().to_dict()
            
            # Process activity trend
            if 'arrival_time' in vessel_analysis.columns:
                # Convert arrival_time to datetime if it's not already
                vessel_analysis['arrival_time'] = pd.to_datetime(vessel_analysis['arrival_time'], errors='coerce')
                
                # Group by hour for activity trend
                vessel_analysis['hour'] = vessel_analysis['arrival_time'].dt.floor('H')
                hourly_counts = vessel_analysis.groupby('hour').size().reset_index(name='count')
                
                processed_data['activity_trend'] = [
                    {'time': row['hour'], 'arrivals': row['count']}
                    for _, row in hourly_counts.iterrows()
                    if pd.notna(row['hour'])
                ]
            
            vessel_analysis = processed_data
        
        # Handle dictionary input from get_comprehensive_vessel_analysis()
        elif isinstance(vessel_analysis, dict):
            if not any(vessel_analysis.values()):
                st.warning("No vessel analysis data available")
                return
            
            # Store original data for timestamp extraction
            original_vessel_analysis = vessel_analysis.copy()
            
            # Map the keys from get_comprehensive_vessel_analysis() to expected format
            processed_data = {
                'location_distribution': vessel_analysis.get('location_type_breakdown', {}),
                'category_distribution': vessel_analysis.get('ship_category_breakdown', {}),
                'activity_trend': vessel_analysis.get('activity_trend', []),
                # Preserve original data for timestamp extraction
                'file_breakdown': vessel_analysis.get('file_breakdown', {}),
                'analysis_timestamp': vessel_analysis.get('analysis_timestamp')
            }
            
            # Data processed successfully
            
            vessel_analysis = processed_data
        else:
            st.warning("Invalid vessel analysis data format")
            return
        
        # Create a centered layout
        main_col1, main_col2, main_col3 = st.columns([0.1, 0.8, 0.1])
        
        with main_col2:
            #st.subheader("🚢 Vessel Analytics Dashboard")
            #st.write("Real-time analysis of vessel distribution and activity patterns")
            
            # Create two columns for the first row of charts
            col1, col2 = st.columns(2)
            
            with col1:
                #st.write("**Vessel Locations**")
                render_vessel_location_distribution(vessel_analysis)
            
            with col2:
                #st.write("**Ship Categories**")
                render_ship_category_distribution(vessel_analysis)
            
            # Full width for the activity trend chart
            st.write("**Recent Activity**")
            render_arrival_activity_trend(vessel_analysis)
            
            # Add some spacing
            st.write("")
            
            # Full width for the arriving ships list
            st.write("**Ships Currently Arriving**")
            render_arriving_ships_list()
            
    except Exception as e:
        st.error(f"Error rendering vessel analytics dashboard: {str(e)}")