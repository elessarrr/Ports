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

# Add the hk_port_digital_twin directory to Python path
project_root = Path(__file__).parent
hk_port_path = project_root / "hk_port_digital_twin"

if str(hk_port_path) not in sys.path:
    sys.path.insert(0, str(hk_port_path))

# Change working directory to hk_port_digital_twin for proper relative imports
os.chdir(str(hk_port_path))

# Import and run the main dashboard
try:
    # Import the main dashboard module
    import src.dashboard.streamlit_app as dashboard_app
    
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
