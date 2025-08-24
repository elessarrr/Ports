"""Tests for Container Handler Module

This module tests the container handling functionality including:
- Processing time calculations
- Ship processing simulation
- Statistics tracking
- Error handling for invalid inputs
"""

import pytest
import simpy
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.container_handler import ContainerHandler
from core.ship_manager import Ship, ShipState
from core.berth_manager import Berth


class TestContainerHandler:
    """Test cases for ContainerHandler class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.env = simpy.Environment()
        self.handler = ContainerHandler(self.env)
        
        # Create test ship
        self.test_ship = Ship(
            ship_id="TEST_001",
            name="Test Container Ship",
            ship_type="container",
            size_teu=10000,
            arrival_time=0.0,
            containers_to_unload=500,
            containers_to_load=300
        )
        
        # Create test berth
        self.test_berth = Berth(
            berth_id=1,
            name="Test Berth A1",
            max_capacity_teu=20000,
            crane_count=4,
            berth_type="container"
        )
        
    def test_initialization(self):
        """Test ContainerHandler initialization"""
        assert self.handler.env == self.env
        assert 'container' in self.handler.processing_rates
        assert 'bulk' in self.handler.processing_rates
        assert self.handler.processing_rates['container'] == 120
        assert self.handler.processing_rates['bulk'] == 250
        assert self.handler.processing_history == []
        
    def test_calculate_processing_time_container_ship(self):
        """Test processing time calculation for container ship"""
        # Test with 4 cranes, 800 total containers, 100 TEU/hour rate
        time = self.handler.calculate_processing_time(
            ship_type="container",
            containers_to_unload=500,
            containers_to_load=300,
            crane_count=4
        )
        
        # Expected: 800 containers / 120 rate = 6.67 hours base
        # With 4 cranes: efficiency = 4 * 0.8 = 3.2
        # Final time: 6.67 / 3.2 = 2.08 hours
        expected_time = 800 / 120 / (4 * 0.8)
        assert abs(time - expected_time) < 0.01
        
    def test_calculate_processing_time_bulk_ship(self):
        """Test processing time calculation for bulk ship"""
        time = self.handler.calculate_processing_time(
            ship_type="bulk",
            containers_to_unload=1000,
            containers_to_load=500,
            crane_count=2
        )
        
        # Expected: 1500 containers / 250 rate = 6 hours base
        # With 2 cranes: efficiency = 2 * 0.8 = 1.6
        # Final time: 6 / 1.6 = 3.75 hours
        expected_time = 1500 / 250 / (2 * 0.8)
        assert abs(time - expected_time) < 0.01
        
    def test_calculate_processing_time_many_cranes(self):
        """Test processing time with many cranes (diminishing returns)"""
        time = self.handler.calculate_processing_time(
            ship_type="container",
            containers_to_unload=800,
            containers_to_load=0,
            crane_count=6
        )
        
        # Expected: 800 containers / 120 rate = 6.67 hours base
        # With 6 cranes: efficiency = 4 * 0.8 + 2 * 0.3 = 3.2 + 0.6 = 3.8
        # Final time: 6.67 / 3.8 ≈ 1.75 hours
        expected_time = 800 / 120 / (4 * 0.8 + 2 * 0.3)
        assert abs(time - expected_time) < 0.01
        
    def test_calculate_processing_time_zero_containers(self):
        """Test processing time with zero containers"""
        time = self.handler.calculate_processing_time(
            ship_type="container",
            containers_to_unload=0,
            containers_to_load=0,
            crane_count=2
        )
        
        # Should return minimum time of 0.1 hours
        assert time == 0.1
        
    def test_calculate_processing_time_invalid_ship_type(self):
        """Test processing time calculation with invalid ship type"""
        with pytest.raises(ValueError, match="Unknown ship type: invalid"):
            self.handler.calculate_processing_time(
                ship_type="invalid",
                containers_to_unload=100,
                containers_to_load=50,
                crane_count=2
            )
            
    def test_calculate_processing_time_invalid_crane_count(self):
        """Test processing time calculation with invalid crane count"""
        with pytest.raises(ValueError, match="Invalid crane count: 0"):
            self.handler.calculate_processing_time(
                ship_type="container",
                containers_to_unload=100,
                containers_to_load=50,
                crane_count=0
            )
            
        with pytest.raises(ValueError, match="Invalid crane count: -1"):
            self.handler.calculate_processing_time(
                ship_type="container",
                containers_to_unload=100,
                containers_to_load=50,
                crane_count=-1
            )
            
    def test_process_ship_simulation(self):
        """Test ship processing simulation"""
        def test_process():
            # Start processing
            yield self.env.process(self.handler.process_ship(self.test_ship, self.test_berth))
            
        # Run the simulation
        self.env.process(test_process())
        self.env.run()
        
        # Check that processing was recorded
        assert len(self.handler.processing_history) == 1
        record = self.handler.processing_history[0]
        
        assert record['ship_id'] == "TEST_001"
        assert record['ship_type'] == "container"
        assert record['berth_id'] == 1
        assert record['containers_unloaded'] == 500
        assert record['containers_loaded'] == 300
        assert record['crane_count'] == 4
        assert record['start_time'] == 0.0
        assert record['end_time'] > 0.0
        assert record['processing_time'] > 0.0
        
    def test_process_ship_timing(self):
        """Test that ship processing takes the calculated time"""
        def test_process():
            yield self.env.process(self.handler.process_ship(self.test_ship, self.test_berth))
            
        # Calculate expected processing time
        expected_time = self.handler.calculate_processing_time(
            self.test_ship.ship_type,
            self.test_ship.containers_to_unload,
            self.test_ship.containers_to_load,
            self.test_berth.crane_count
        )
        
        # Run simulation
        self.env.process(test_process())
        self.env.run()
        
        # Check that simulation time matches expected processing time
        assert abs(self.env.now - expected_time) < 0.01
        
    def test_get_processing_statistics_empty(self):
        """Test statistics when no processing has occurred"""
        stats = self.handler.get_processing_statistics()
        
        assert stats['total_operations'] == 0
        assert stats['average_processing_time'] == 0
        assert stats['total_containers_processed'] == 0
        assert stats['average_crane_utilization'] == 0
        
    def test_get_processing_statistics_with_data(self):
        """Test statistics after processing operations"""
        # Process the test ship
        def test_process():
            yield self.env.process(self.handler.process_ship(self.test_ship, self.test_berth))
            
        self.env.process(test_process())
        self.env.run()
        
        # Get statistics
        stats = self.handler.get_processing_statistics()
        
        assert stats['total_operations'] == 1
        assert stats['average_processing_time'] > 0
        assert stats['total_containers_processed'] == 800  # 500 + 300
        assert stats['average_crane_utilization'] == 4.0
        
    def test_multiple_ship_processing(self):
        """Test processing multiple ships"""
        # Create second ship
        ship2 = Ship(
            ship_id="TEST_002",
            name="Test Bulk Ship",
            ship_type="bulk",
            size_teu=15000,
            arrival_time=0.0,
            containers_to_unload=1000,
            containers_to_load=200
        )
        
        # Create second berth
        berth2 = Berth(
            berth_id=2,
            name="Test Berth B1",
            max_capacity_teu=30000,
            crane_count=2,
            berth_type="bulk"
        )
        
        def test_process():
            # Process both ships simultaneously
            yield self.env.all_of([
                self.env.process(self.handler.process_ship(self.test_ship, self.test_berth)),
                self.env.process(self.handler.process_ship(ship2, berth2))
            ])
            
        self.env.process(test_process())
        self.env.run()
        
        # Check that both operations were recorded
        assert len(self.handler.processing_history) == 2
        
        # Get statistics
        stats = self.handler.get_processing_statistics()
        assert stats['total_operations'] == 2
        assert stats['total_containers_processed'] == 2000  # 800 + 1200
        
    def test_reset_statistics(self):
        """Test resetting processing statistics"""
        # Process a ship first
        def test_process():
            yield self.env.process(self.handler.process_ship(self.test_ship, self.test_berth))
            
        self.env.process(test_process())
        self.env.run()
        
        # Verify data exists
        assert len(self.handler.processing_history) == 1
        
        # Reset statistics
        self.handler.reset_statistics()
        
        # Verify data is cleared
        assert len(self.handler.processing_history) == 0
        stats = self.handler.get_processing_statistics()
        assert stats['total_operations'] == 0
        
    def test_processing_time_minimum_threshold(self):
        """Test that processing time never goes below minimum threshold"""
        # Test with very few containers and many cranes
        time = self.handler.calculate_processing_time(
            ship_type="container",
            containers_to_unload=1,
            containers_to_load=0,
            crane_count=10
        )
        
        # Should be at least 0.1 hours (minimum threshold)
        assert time >= 0.1
        
    def test_crane_efficiency_calculation(self):
        """Test crane efficiency calculation with different crane counts"""
        # Test 1 crane
        time1 = self.handler.calculate_processing_time(
            "container", 100, 0, 1
        )
        
        # Test 2 cranes (should be faster)
        time2 = self.handler.calculate_processing_time(
            "container", 100, 0, 2
        )
        
        # Test 4 cranes (should be even faster)
        time4 = self.handler.calculate_processing_time(
            "container", 100, 0, 4
        )
        
        # More cranes should result in less processing time
        assert time1 > time2 > time4
        
        # Test diminishing returns with many cranes
        time8 = self.handler.calculate_processing_time(
            "container", 100, 0, 8
        )
        
        # 8 cranes should be faster than 4, but not 2x faster (diminishing returns)
        improvement_4_to_8 = time4 / time8
        improvement_2_to_4 = time2 / time4
        assert improvement_4_to_8 < improvement_2_to_4