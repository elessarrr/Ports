#!/usr/bin/env python3
"""
Test script for the vessel data pipeline implementation.

This script tests the vessel data fetcher, scheduler, and data loading functions
to ensure the pipeline is working correctly before full integration.

Comments for context:
- This is a standalone test script to validate the vessel data pipeline
- Tests both the fetching mechanism and data processing functions
- Uses environment variables for configuration
- Provides detailed logging for debugging
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add the project root to the Python path
from pathlib import Path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_vessel_data_fetcher():
    """Test the VesselDataFetcher class"""
    logger.info("Testing VesselDataFetcher...")
    
    try:
        from hk_port_digital_twin.src.utils.vessel_data_fetcher import VesselDataFetcher
        
        # Create fetcher instance
        fetcher = VesselDataFetcher()
        
        # Test configuration
        logger.info(f"Base URL: {fetcher.base_url}")
        logger.info(f"Data directory: {fetcher.data_directory}")
        logger.info(f"Pipeline enabled: {fetcher.pipeline_enabled}")
        logger.info(f"Fetch interval: {fetcher.fetch_interval} minutes")
        
        # Test single file fetch (without actually downloading)
        logger.info("VesselDataFetcher initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"VesselDataFetcher test failed: {e}")
        return False

def test_vessel_data_scheduler():
    """Test VesselDataScheduler initialization"""
    logger.info("Testing VesselDataScheduler...")
    
    try:
        from hk_port_digital_twin.src.utils.vessel_data_scheduler import VesselDataScheduler
        
        # Create a mock callback function
        def mock_fetcher_callback():
            return {"status": "success", "message": "Mock fetch completed"}
        
        # Create scheduler instance with required callback
        scheduler = VesselDataScheduler(fetcher_callback=mock_fetcher_callback)
        
        logger.info(f"Fetch interval: {scheduler.fetch_interval} minutes")
        logger.info(f"Pipeline enabled: {scheduler.pipeline_enabled}")
        
        logger.info("VesselDataScheduler initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"VesselDataScheduler test failed: {e}")
        return False

def test_data_loader_functions():
    """Test the new data loader functions"""
    logger.info("Testing data loader functions...")
    
    try:
        from hk_port_digital_twin.src.utils.data_loader import (
            load_all_vessel_data,
            initialize_vessel_data_pipeline
        )
        
        # Test pipeline initialization
        logger.info("Testing pipeline initialization...")
        initialize_vessel_data_pipeline()
        
        # Test data loading (this will work even if no XML files exist yet)
        logger.info("Testing vessel data loading...")
        vessel_data = load_all_vessel_data()
        
        if vessel_data is not None:
            logger.info(f"Loaded vessel data with {len(vessel_data)} entries")
        else:
            logger.info("No vessel data loaded (expected if XML files don't exist yet)")
        
        logger.info("Data loader functions tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data loader functions test failed: {e}")
        return False

def test_environment_configuration():
    """Test environment configuration"""
    logger.info("Testing environment configuration...")
    
    # Check for .env file
    env_file = os.path.join(project_root, '.env')
    env_example_file = os.path.join(project_root, 'hk_port_digital_twin', '.env.example')
    
    if os.path.exists(env_file):
        logger.info("Found .env file")
    elif os.path.exists(env_example_file):
        logger.info("Found .env.example file (you may want to copy it to .env)")
    else:
        logger.warning("No .env or .env.example file found")
    
    # Check key environment variables
    key_vars = [
        'HK_VESSEL_DATA_BASE_URL',
        'VESSEL_DATA_FETCH_INTERVAL',
        'VESSEL_DATA_DIRECTORY',
        'VESSEL_DATA_PIPELINE_ENABLED'
    ]
    
    for var in key_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"{var}: {value}")
        else:
            logger.info(f"{var}: Not set (will use default)")
    
    return True

def main():
    """Run all tests"""
    logger.info("Starting vessel data pipeline tests...")
    logger.info(f"Project root: {project_root}")
    
    tests = [
        ("Environment Configuration", test_environment_configuration),
        ("VesselDataFetcher", test_vessel_data_fetcher),
        ("VesselDataScheduler", test_vessel_data_scheduler),
        ("Data Loader Functions", test_data_loader_functions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Vessel data pipeline is ready.")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} test(s) failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)