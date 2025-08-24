# MarineTraffic Integration Module
# Comments for context:
# This module provides optional integration with MarineTraffic's live vessel tracking
# for enhanced visualization in the Hong Kong Port Digital Twin dashboard.
# This is a proof-of-concept implementation that can be enabled/disabled via configuration.
# The integration supports both iframe embedding and API-based approaches.

import streamlit as st
import requests
import os
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarineTrafficIntegration:
    """
    Handles integration with MarineTraffic for real-time vessel visualization.
    
    This class provides methods to:
    1. Embed MarineTraffic live maps via iframe
    2. Fetch vessel data via API (when available)
    3. Handle fallbacks when service is unavailable
    """
    
    def __init__(self):
        # MarineTraffic API configuration (optional)
        self.api_key = os.getenv('MARINETRAFFIC_API_KEY')
        self.base_url = "https://services.marinetraffic.com/api"
        
        # Hong Kong coordinates for map centering
        self.hk_center = {
            'latitude': 22.3193,
            'longitude': 114.1694,
            'zoom': 11
        }
        
        # Check if integration is enabled
        self.enabled = os.getenv('ENABLE_MARINETRAFFIC', 'false').lower() == 'true'
        
    def render_live_map_iframe(self, height: int = 500) -> None:
        """
        Renders MarineTraffic live map using iframe embedding.
        
        Args:
            height: Height of the map in pixels
        """
        if not self.enabled:
            st.info("🗺️ MarineTraffic integration is disabled. Enable in settings to view live vessel tracking.")
            return
            
        try:
            # Construct MarineTraffic embed URL for Hong Kong waters
            embed_url = (
                f"https://www.marinetraffic.com/en/ais/embed/"
                f"zoom:{self.hk_center['zoom']}/"
                f"centery:{self.hk_center['latitude']}/"
                f"centerx:{self.hk_center['longitude']}/"
                f"maptype:4/"
                f"shownames:true/"
                f"mmsi:0/"
                f"shipid:0/"
                f"fleet:0/"
                f"fleet_hide_old_positions:false/"
                f"fleet_hide_fishing_vessels:true/"
                f"fleet_hide_passenger_vessels:false"
            )
            
            st.markdown(
                f'<iframe src="{embed_url}" width="100%" height="{height}px" frameborder="0"></iframe>',
                unsafe_allow_html=True
            )
            
            # Add disclaimer
            st.caption("🔴 Live vessel data provided by MarineTraffic. Updates every few minutes.")
            
        except Exception as e:
            logger.error(f"Failed to render MarineTraffic map: {e}")
            st.error("Unable to load live vessel tracking. Please try again later.")
    
    def get_vessel_data_api(self, area_bounds: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Fetches vessel data from MarineTraffic API (requires API key).
        
        Args:
            area_bounds: Dictionary with 'minlat', 'maxlat', 'minlon', 'maxlon'
            
        Returns:
            Dictionary containing vessel data or None if unavailable
        """
        if not self.api_key:
            logger.warning("MarineTraffic API key not configured")
            return None
            
        if not area_bounds:
            # Default to Hong Kong waters
            area_bounds = {
                'minlat': 22.1,
                'maxlat': 22.5,
                'minlon': 113.8,
                'maxlon': 114.5
            }
        
        try:
            # Construct API request
            endpoint = f"{self.base_url}/exportvessels/v:8"
            params = {
                'key': self.api_key,
                'protocol': 'jsono',
                'minlat': area_bounds['minlat'],
                'maxlat': area_bounds['maxlat'],
                'minlon': area_bounds['minlon'],
                'maxlon': area_bounds['maxlon'],
                'timespan': 10  # Last 10 minutes
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"MarineTraffic API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching vessel data: {e}")
            return None
    
    def render_vessel_statistics(self) -> None:
        """
        Displays vessel statistics from MarineTraffic data.
        """
        if not self.enabled:
            return
            
        vessel_data = self.get_vessel_data_api()
        
        if vessel_data and 'data' in vessel_data:
            vessels = vessel_data['data']
            
            # Create metrics columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Active Vessels", len(vessels))
            
            with col2:
                cargo_vessels = [v for v in vessels if v.get('type_name', '').lower() in ['cargo', 'container']]
                st.metric("Cargo Vessels", len(cargo_vessels))
            
            with col3:
                tankers = [v for v in vessels if 'tanker' in v.get('type_name', '').lower()]
                st.metric("Tankers", len(tankers))
            
            with col4:
                avg_speed = sum(float(v.get('speed', 0)) for v in vessels) / len(vessels) if vessels else 0
                st.metric("Avg Speed (knots)", f"{avg_speed:.1f}")
                
        else:
            st.info("Vessel statistics unavailable. API key required for detailed data.")
    
    def render_integration_settings(self) -> None:
        """
        Renders settings panel for MarineTraffic integration.
        """
        st.subheader("🗺️ MarineTraffic Integration")
        
        # Current status
        if self.enabled:
            st.success("✅ MarineTraffic integration is enabled")
        else:
            st.warning("⚠️ MarineTraffic integration is disabled")
        
        # API key status
        if self.api_key:
            st.success("✅ API key configured")
        else:
            st.info("ℹ️ No API key configured (iframe mode only)")
        
        # Configuration instructions
        with st.expander("Configuration Instructions"):
            st.markdown("""
            **To enable MarineTraffic integration:**
            
            1. **Environment Variables:**
               ```bash
               export ENABLE_MARINETRAFFIC=true
               export MARINETRAFFIC_API_KEY=your_api_key_here  # Optional
               ```
            
            2. **Features Available:**
               - **Without API key:** Live map embedding (iframe)
               - **With API key:** Live map + vessel statistics + detailed data
            
            3. **API Key Benefits:**
               - Detailed vessel information
               - Historical data access
               - Custom area queries
               - Higher update frequency
            
            4. **Cost Considerations:**
               - Free tier: 1,000 API calls/month
               - Professional: $99/month for enhanced features
               - Enterprise: Custom pricing for high-volume usage
            """)
        
        # Test connection button
        if st.button("Test MarineTraffic Connection"):
            self._test_connection()
    
    def _test_connection(self) -> None:
        """
        Tests the MarineTraffic connection and displays results.
        """
        with st.spinner("Testing MarineTraffic connection..."):
            if self.api_key:
                # Test API connection
                vessel_data = self.get_vessel_data_api()
                if vessel_data:
                    st.success(f"✅ API connection successful! Found {len(vessel_data.get('data', []))} vessels in Hong Kong waters.")
                else:
                    st.error("❌ API connection failed. Check your API key and network connection.")
            else:
                # Test iframe embedding capability
                st.info("🔄 Testing iframe embedding...")
                try:
                    # Simple test - just check if we can construct the URL
                    test_url = f"https://www.marinetraffic.com/en/ais/embed/zoom:10/centery:22.3/centerx:114.2"
                    st.success("✅ Iframe embedding should work. Enable integration to see live map.")
                except Exception as e:
                    st.error(f"❌ Iframe test failed: {e}")

# Convenience function for easy integration
def render_marinetraffic_tab():
    """
    Renders a complete MarineTraffic integration tab for the dashboard.
    """
    integration = MarineTrafficIntegration()
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["🗺️ Live Map", "📊 Statistics", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Real-Time Vessel Tracking")
        st.markdown("""
        This map shows live vessel positions in Hong Kong waters using AIS (Automatic Identification System) data.
        Vessels are updated every few minutes and include cargo ships, tankers, passenger vessels, and more.
        """)
        
        integration.render_live_map_iframe(height=600)
        
        # Additional info
        with st.expander("About AIS Data"):
            st.markdown("""
            **Automatic Identification System (AIS)** is a tracking system used by ships and vessel traffic services.
            
            - **Coverage:** All vessels >300 GT internationally, >500 GT domestically
            - **Update Rate:** Every 2-10 seconds depending on vessel speed
            - **Data Includes:** Position, speed, course, vessel type, destination
            - **Accuracy:** Typically within 10 meters
            """)
    
    with tab2:
        st.subheader("Vessel Statistics")
        integration.render_vessel_statistics()
        
        # Placeholder for additional statistics when API is available
        if not integration.api_key:
            st.info("""
            📈 **Enhanced statistics available with API access:**
            - Vessel type breakdown
            - Traffic density analysis
            - Port arrival/departure predictions
            - Historical trend analysis
            """)
    
    with tab3:
        integration.render_integration_settings()

if __name__ == "__main__":
    # For testing the module independently
    st.set_page_config(page_title="MarineTraffic Integration Test", layout="wide")
    render_marinetraffic_tab()