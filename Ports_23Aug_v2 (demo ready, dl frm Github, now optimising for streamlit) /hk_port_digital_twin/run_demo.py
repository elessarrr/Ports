#!/usr/bin/env python3
"""Demo launcher for Hong Kong Port Digital Twin Dashboard

This script launches the Streamlit dashboard for the port simulation.
Run this file to start the interactive web interface.

Usage:
    python run_demo.py
    
Or with custom port:
    streamlit run src/dashboard/streamlit_app.py --server.port 8502
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['streamlit', 'plotly', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True


def main():
    """Main function to launch the dashboard"""
    print("🚢 Hong Kong Port Digital Twin Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the correct directory
    current_dir = Path.cwd()
    expected_files = ['src/dashboard/streamlit_app.py', 'config/settings.py', 'requirements.txt']
    
    missing_files = []
    for file_path in expected_files:
        if not (current_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please run this script from the project root directory.")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Launch Streamlit dashboard
    dashboard_path = current_dir / 'src' / 'dashboard' / 'streamlit_app.py'
    
    print(f"🚀 Launching dashboard from: {dashboard_path}")
    print("📱 The dashboard will open in your default web browser")
    print("🔗 Default URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        cmd = [sys.executable, '-m', 'streamlit', 'run', str(dashboard_path)]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it using: pip install streamlit")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)