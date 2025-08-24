#!/usr/bin/env python3
"""
Streamlit Community Cloud Entry Point for Hong Kong Port Digital Twin Dashboard

This file serves as the main entry point for Streamlit Community Cloud deployment.
Streamlit Community Cloud expects the main app file to be named 'streamlit_app.py' 
at the project root level.

This simplified entry point directly executes the dashboard file to avoid
path resolution issues in cloud environments.
"""

import sys
import os
from pathlib import Path

# Simple and robust entry point for cloud deployment
try:
    # Get the path to the actual dashboard file
    dashboard_path = Path(__file__).resolve().parent / "hk_port_digital_twin" / "src" / "dashboard" / "streamlit_app.py"
    
    # Verify the dashboard file exists
    if not dashboard_path.exists():
        raise FileNotFoundError(f"Dashboard file not found at: {dashboard_path}")
    
    # Execute the dashboard file directly
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        dashboard_code = f.read()
    
    # Set up the execution environment with correct project root
    exec_globals = globals().copy()
    exec_globals['__file__'] = str(dashboard_path)
    
    # Execute the dashboard code with the correct context
    exec(dashboard_code, exec_globals)
        
except FileNotFoundError as e:
    import streamlit as st
    st.error(f"Dashboard file not found: {e}")
    st.info("Please ensure the project structure is correct.")
    st.info(f"Current working directory: {os.getcwd()}")
    st.info(f"Looking for dashboard at: {Path(__file__).resolve().parent / 'hk_port_digital_twin' / 'src' / 'dashboard' / 'streamlit_app.py'}")
except Exception as e:
    import streamlit as st
    st.error(f"Error running dashboard: {e}")
    st.info("Please check the application logs for more details.")
    st.info(f"Current working directory: {os.getcwd()}")
    st.info(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
