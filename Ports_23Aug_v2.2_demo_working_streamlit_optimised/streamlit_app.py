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

# Ensure robust path handling for both local and cloud deployment
try:
    # Get the absolute path of the current file's directory
    current_dir = Path(__file__).resolve().parent
    
    # Add current directory to Python path if not already present
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Import the main dashboard module using absolute import
    from hk_port_digital_twin.src.dashboard.streamlit_app import main
    
    # Run the main function
    if __name__ == "__main__":
        main()
    else:
        # If imported as module, run main automatically
        main()
        
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to import dashboard module: {e}")
    st.info("Please ensure all dependencies are installed and the project structure is correct.")
    st.info(f"Current working directory: {os.getcwd()}")
    st.info(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
except Exception as e:
    import streamlit as st
    st.error(f"Error running dashboard: {e}")
    st.info("Please check the application logs for more details.")
    st.info(f"Current working directory: {os.getcwd()}")
