#!/usr/bin/env python3
"""
Debug script to check what data is being loaded in the UI
"""

import sys
import os
sys.path.append('hk_port_digital_twin/src')

from utils.data_loader import load_combined_vessel_data
import pandas as pd

print("Loading combined vessel data for UI debugging...")
df = load_combined_vessel_data()

print(f"\nTotal vessels loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")

if 'status' in df.columns:
    status_counts = df['status'].value_counts()
    print(f"\nStatus breakdown:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
else:
    print("\nNo 'status' column found!")

if 'data_source' in df.columns:
    source_counts = df['data_source'].value_counts()
    print(f"\nData source breakdown:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
else:
    print("\nNo 'data_source' column found!")

# Show first few rows with key columns
key_cols = ['vessel_name', 'status', 'data_source'] if all(col in df.columns for col in ['vessel_name', 'status', 'data_source']) else df.columns[:5]
print(f"\nFirst 5 rows (key columns):")
print(df[key_cols].head())

# Check for arriving status specifically
if 'status' in df.columns:
    arriving_vessels = df[df['status'] == 'arriving']
    print(f"\nVessels with 'arriving' status: {len(arriving_vessels)}")
    if len(arriving_vessels) > 0:
        print("Sample arriving vessels:")
        print(arriving_vessels[['vessel_name', 'status', 'data_source']].head())