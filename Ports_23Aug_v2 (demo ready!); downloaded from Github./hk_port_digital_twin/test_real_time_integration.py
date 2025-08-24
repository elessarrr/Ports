#!/usr/bin/env python3
"""Test script for real-time data integration features.

This script tests the new real-time capabilities including:
- Weather data integration
- File monitoring system
- Real-time data manager
"""

import sys
from pathlib import Path
import time
import logging

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Weather integration test removed - see planning/remove_weather_impact_plan.md

def test_file_monitoring():
    """Test file monitoring system."""
    logger.info("Testing file monitoring system...")
    
    try:
        from utils.file_monitor import create_default_port_monitor
        
        # Create file monitor
        monitor = create_default_port_monitor()
        
        # Test configuration
        status = monitor.get_status()
        logger.info(f"File monitor status: {status}")
        
        # Test setup (without actually starting monitoring)
        def test_callback(file_path):
            logger.info(f"Test callback triggered for: {file_path}")
        
        monitor.setup_vessel_monitoring(test_callback)
        monitor.setup_cargo_monitoring(test_callback)
        monitor.setup_berth_monitoring(test_callback)
        
        logger.info("File monitoring test completed successfully")
        return True
        
    except ImportError as e:
        logger.error(f"File monitoring not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing file monitoring: {e}")
        return False

def test_real_time_manager():
    """Test real-time data manager."""
    logger.info("Testing real-time data manager...")
    
    try:
        from utils.data_loader import RealTimeDataConfig, get_real_time_manager
        
        # Create configuration
        config = RealTimeDataConfig(
            enable_weather_integration=True,
            enable_file_monitoring=True,
            vessel_update_interval=60,  # 1 minute for testing
            weather_update_interval=300,  # 5 minutes for testing
            auto_reload_on_file_change=True
        )
        
        # Get manager instance
        manager = get_real_time_manager(config)
        
        # Test status
        status = manager.get_status()
        logger.info(f"Manager status: {status}")
        
        # Test data retrieval
        vessel_data = manager.get_real_time_vessel_data()
        logger.info(f"Vessel data shape: {vessel_data.shape}")
        
        # Test enhanced analysis
        analysis = manager.get_enhanced_queue_analysis()
        logger.info(f"Enhanced analysis keys: {list(analysis.keys())}")
        
        # Test callback registration
        def test_update_callback(data):
            logger.info(f"Update callback triggered with data type: {type(data)}")
        
        manager.register_update_callback('vessel_arrivals', test_update_callback)
        manager.register_update_callback('weather_conditions', test_update_callback)
        
        logger.info("Real-time manager test completed successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Real-time manager not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing real-time manager: {e}")
        return False

def test_integration():
    """Test integration between components."""
    logger.info("Testing component integration...")
    
    try:
        from utils.data_loader import get_real_time_manager, load_vessel_arrivals
        
        # Test basic data loading
        vessel_data = load_vessel_arrivals()
        logger.info(f"Loaded vessel data: {vessel_data.shape}")
        
        # Test manager integration
        manager = get_real_time_manager()
        
        # Test cached data
        cached_data = manager.get_cached_data('vessel_arrivals')
        if cached_data is not None:
            logger.info(f"Cached vessel data available: {cached_data.shape}")
        else:
            logger.info("No cached vessel data available")
        
        logger.info("Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing integration: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting real-time integration tests...")
    
    tests = [
        ("Weather Integration", test_weather_integration),
        ("File Monitoring", test_file_monitoring),
        ("Real-time Manager", test_real_time_manager),
        ("Component Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("All tests passed! Real-time integration is working correctly.")
        return 0
    else:
        logger.warning("Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)