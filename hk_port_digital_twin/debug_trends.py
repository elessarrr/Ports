#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from utils.data_loader import get_throughput_trends, load_sample_data
from unittest.mock import patch
import pandas as pd

# Load sample data
sample_df = load_sample_data()
print('Sample data columns:', sample_df.columns.tolist())
print('Sample data shape:', sample_df.shape)
print('Sample data index type:', type(sample_df.index))
print('Sample data date range:', sample_df.index.min(), 'to', sample_df.index.max())

# Mock and test the function
with patch('utils.data_loader.load_container_throughput') as mock_load:
    mock_load.return_value = sample_df
    trends = get_throughput_trends()
    
    print('\nTrends keys:', list(trends.keys()))
    print('\nFirst level structure:')
    for k, v in trends.items():
        print(f'  {k}: {type(v)}')
        if isinstance(v, dict):
            print(f'    Sub-keys: {list(v.keys())}')