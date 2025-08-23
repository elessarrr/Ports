import streamlit as st
from datetime import datetime, timedelta
from hk_port_digital_twin.src.utils.data_loader import get_comprehensive_vessel_analysis
from hk_port_digital_twin.src.dashboard.vessel_charts import render_vessel_analytics_dashboard, _extract_latest_timestamp

def render_vessel_analytics_tab():
    st.subheader("🚢 Vessel Analytics Dashboard")
    st.markdown("Real-time analysis of vessel distribution and activity patterns")
    
    try:
        # Load comprehensive vessel analysis data (includes timestamps from all XML files)
        vessel_analysis = get_comprehensive_vessel_analysis()
        
        if vessel_analysis and vessel_analysis.get('data_summary', {}).get('total_vessels', 0) > 0:
            # Display data summary
            data_summary = vessel_analysis.get('data_summary', {})
            st.write(f"**Data Summary:** {data_summary.get('total_vessels', 0)} vessels loaded from {data_summary.get('files_processed', 0)} files")
            
            # Show data sources
            data_sources = data_summary.get('data_sources', [])
            if data_sources:
                st.write(f"**Data Sources:** {', '.join([src.replace('.xml', '') for src in data_sources])}")
            
            # Location breakdown
            location_breakdown = vessel_analysis.get('location_type_breakdown', {})
            if location_breakdown:
                st.write(f"**Locations:** {len(location_breakdown)} unique location types")
            
            # Ship category breakdown
            category_breakdown = vessel_analysis.get('ship_category_breakdown', {})
            if category_breakdown:
                st.write(f"**Ship Categories:** {len(category_breakdown)} different types")
            
            # Recent activity
            recent_activity = vessel_analysis.get('recent_activity', {})
            if recent_activity:
                st.write(f"**Recent Activity:** {recent_activity.get('vessels_last_24h', 0)} vessels in last 24 hours")
            
            # Display timestamp if available
            latest_timestamp = _extract_latest_timestamp(vessel_analysis)
            if latest_timestamp:
                st.caption(f"📅 Last updated at: {latest_timestamp}")
            else:
                st.caption("📅 Last updated at: Not available")
            
            # Render the vessel analytics dashboard with comprehensive analysis data
            render_vessel_analytics_dashboard(vessel_analysis)
            
        else:
            st.warning("No vessel data available for analytics.")
            st.info("Please ensure vessel data files are properly loaded from the raw_data directory.")
            
    except Exception as e:
        st.error(f"Error loading vessel analytics: {str(e)}")
        st.info("Using sample data for demonstration purposes.")
        
        # Fallback to sample data with proper structure
        sample_vessel_analysis = {
            'data_summary': {
                'total_vessels': 3,
                'files_processed': 1,
                'data_sources': ['sample_data']
            },
            'location_type_breakdown': {'berth': 2, 'anchorage': 1},
            'ship_category_breakdown': {'container': 2, 'bulk_carrier': 1},
            'file_breakdown': {
                'sample_data': {
                    'earliest_timestamp': (datetime.now() - timedelta(hours=8)).isoformat(),
                    'latest_timestamp': (datetime.now() - timedelta(hours=2)).isoformat()
                }
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
        render_vessel_analytics_dashboard(sample_vessel_analysis)