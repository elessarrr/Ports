#!/usr/bin/env python3
"""
Debug script to test data loader imports
This script helps diagnose import issues with the data loader module.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Testing data loader imports...")
print(f"Python path: {sys.path[:3]}")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {Path(__file__).parent}")
print(f"Src path: {src_path}")
print(f"Src path exists: {src_path.exists()}")

try:
    print("\n1. Testing basic import...")
    from utils.data_loader import load_port_cargo_statistics
    print("✅ Basic import successful")
    
    print("\n2. Testing focused cargo statistics import...")
    from utils.data_loader import load_focused_cargo_statistics
    print("✅ load_focused_cargo_statistics import successful")
    
    print("\n3. Testing enhanced cargo analysis import...")
    from utils.data_loader import get_enhanced_cargo_analysis
    print("✅ get_enhanced_cargo_analysis import successful")
    
    print("\n4. Testing time series data import...")
    from utils.data_loader import get_time_series_data
    print("✅ get_time_series_data import successful")
    
    print("\n5. Testing actual function calls...")
    
    # Test loading focused cargo statistics
    print("Testing load_focused_cargo_statistics()...")
    focused_data = load_focused_cargo_statistics()
    print(f"✅ Loaded {len(focused_data)} tables: {list(focused_data.keys())}")
    
    # Test enhanced cargo analysis
    print("\nTesting get_enhanced_cargo_analysis()...")
    analysis = get_enhanced_cargo_analysis()
    print(f"✅ Analysis keys: {list(analysis.keys())}")
    
    # Test time series data
    print("\nTesting get_time_series_data()...")
    time_series = get_time_series_data(focused_data)
    print(f"✅ Time series keys: {list(time_series.keys())}")
    
    print("\n🎉 All imports and function calls successful!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Error type: {type(e)}")
    
    # Check if utils module exists
    utils_path = src_path / "utils"
    print(f"\nUtils path: {utils_path}")
    print(f"Utils path exists: {utils_path.exists()}")
    
    if utils_path.exists():
        print(f"Utils directory contents: {list(utils_path.iterdir())}")
        
        data_loader_path = utils_path / "data_loader.py"
        print(f"Data loader path: {data_loader_path}")
        print(f"Data loader exists: {data_loader_path.exists()}")
        
except Exception as e:
    print(f"❌ Other Error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()