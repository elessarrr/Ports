#!/usr/bin/env python3
"""
Streamlit Community Cloud Entry Point for Hong Kong Port Digital Twin Dashboard

This file serves as the main entry point for Streamlit Community Cloud deployment.
Streamlit Community Cloud expects the main app file to be named 'streamlit_app.py' 
at the project root level.

This file imports and runs the actual dashboard from the nested project structure.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path for proper imports
project_root = Path(__file__).parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the main dashboard
try:
    # Import the main dashboard module
    import hk_port_digital_twin.src.dashboard.streamlit_app as dashboard_app
    
    # Check if there's a main function, otherwise the module will run automatically
    if hasattr(dashboard_app, 'main') and __name__ == "__main__":
        dashboard_app.main()
    # If no main function, the streamlit app should run automatically when imported
        
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to import dashboard: {e}")
    st.info("Please ensure all dependencies are installed and the project structure is correct.")
except Exception as e:
    import streamlit as st
    st.error(f"Error running dashboard: {e}")
    st.info("Please check the application logs for more details.")
