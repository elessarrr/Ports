#!/usr/bin/env python3
"""
Comments for context:
This script provides a manual way to trigger the vessel data pipeline to download
fresh XML files from the Hong Kong Marine Department. It imports the VesselDataFetcher
class from the utils module and executes the fetch_xml_files method to download all
four XML files (Expected_arrivals.xml, Arrived_in_last_36_hours.xml, 
Departed_in_last_36_hours.xml, Expected_departures.xml).

This is useful when you need to refresh the data outside of the automated 20-minute
schedule or when testing the pipeline functionality.
"""

import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'hk_port_digital_twin', 'src'))

from utils.vessel_data_fetcher import VesselDataFetcher

def main():
    """
    Main function to manually trigger vessel data refresh.
    """
    print("=" * 60)
    print("Manual Vessel Data Refresh")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Initialize the vessel data fetcher
        print("\n1. Initializing VesselDataFetcher...")
        fetcher = VesselDataFetcher()
        
        # Download all XML files
        print("\n2. Downloading XML files from Hong Kong Marine Department...")
        results = fetcher.fetch_xml_files()
        
        # Display results
        print("\n3. Download Results:")
        print("-" * 40)
        
        if results:
            for filename, success in results.items():
                status = "✓ SUCCESS" if success else "✗ FAILED"
                print(f"  {filename}: {status}")
        else:
            print("  No results returned from fetch operation")
            
        # Summary
        successful_downloads = sum(1 for success in results.values() if success) if results else 0
        total_files = len(results) if results else 0
        
        print("\n" + "=" * 60)
        print(f"Summary: {successful_downloads}/{total_files} files downloaded successfully")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        if successful_downloads == total_files and total_files > 0:
            print("\n🎉 All files updated successfully!")
            return 0
        else:
            print("\n⚠️  Some files failed to download. Check logs for details.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during vessel data refresh: {str(e)}")
        print("Check the vessel pipeline logs for more details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)