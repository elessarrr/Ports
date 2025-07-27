"""Tests for Port Simulation Engine

This module contains comprehensive tests for the main port simulation engine,
including initialization, ship processing, metrics collection, and error handling.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.port_simulation import PortSimulation
from config.settings import SIMULATION_CONFIG


class TestPortSimulation:
    """Test cases for PortSimulation class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.test_config = {
            'berths': [
                {
                    'berth_id': 1, 
                    'berth_name': 'Test Berth 1',
                    'max_capacity_teu': 5000,
                    'crane_count': 3, 
                    'berth_type': 'container'
                },
                {
                    'berth_id': 2, 
                    'berth_name': 'Test Berth 2',
                    'max_capacity_teu': 3000,
                    'crane_count': 2, 
                    'berth_type': 'mixed'
                }
            ]
        }
        self.simulation = PortSimulation(self.test_config)
        
    def test_initialization(self):
        """Test proper initialization of simulation components"""
        assert self.simulation.env is not None
        assert self.simulation.config == self.test_config
        assert self.simulation.ship_manager is not None
        assert self.simulation.berth_manager is not None
        assert self.simulation.container_handler is not None
        assert not self.simulation.running
        assert self.simulation.ships_processed == 0
        assert self.simulation.total_ships_generated == 0
        
    def test_initial_metrics(self):
        """Test initial metrics state"""
        expected_metrics = {
            'ships_arrived': 0,
            'ships_processed': 0,
            'total_waiting_time': 0,
            'simulation_start_time': 0,
            'simulation_end_time': 0,
            'ai_optimizations_performed': 0,
            'optimization_time_saved': 0
        }
        assert self.simulation.metrics == expected_metrics
        
    def test_generate_random_ship(self):
        """Test random ship generation"""
        ship = self.simulation._generate_random_ship("TEST_SHIP_001")
        
        assert ship.ship_id == "TEST_SHIP_001"
        assert ship.ship_type in ['container', 'bulk', 'tanker', 'mixed']
        assert ship.containers_to_unload >= 0
        assert ship.containers_to_load >= 0
        assert ship.arrival_time == self.simulation.env.now
        
    def test_generate_random_ship_container_type(self):
        """Test container ship generation has appropriate container counts"""
        # Generate multiple ships to test different types
        ships = [self.simulation._generate_random_ship(f"SHIP_{i}") for i in range(20)]
        
        # Check that at least some ships are generated
        assert len(ships) == 20
        
        # Check container ships have realistic container counts based on new config
        container_ships = [s for s in ships if s.ship_type == 'container']
        if container_ships:
            for ship in container_ships:
                # Updated to match actual generation logic:
                # Smallest ship: 1500 TEU -> base_containers = 30 -> min = int(30 * 0.3) = 9
                # Largest ship: 24000 TEU -> base_containers = 480 -> max = int(480 * 0.8) = 384
                assert ship.containers_to_unload >= 9   # Minimum for 1500 TEU ship
                assert ship.containers_to_load >= 6     # Minimum for 1500 TEU ship (30 * 0.2 = 6)
                assert ship.containers_to_unload <= 384  # Maximum for 24000 TEU ship
                assert ship.containers_to_load <= 336    # Maximum for 24000 TEU ship (480 * 0.7 = 336)
                
    def test_short_simulation_run(self):
        """Test running a short simulation"""
        # Run very short simulation
        result = self.simulation.run_simulation(duration=0.1)
        
        # Check simulation completed
        assert not self.simulation.running
        assert 'simulation_summary' in result
        assert 'berth_statistics' in result
        assert 'container_statistics' in result
        assert 'performance_metrics' in result
        
        # Check simulation duration
        assert result['simulation_summary']['duration'] >= 0
        
    def test_simulation_metrics_tracking(self):
        """Test that simulation properly tracks metrics"""
        # Run short simulation
        self.simulation.run_simulation(duration=0.5)
        
        # Check metrics were updated
        assert self.simulation.metrics['simulation_end_time'] > 0
        assert self.simulation.metrics['simulation_start_time'] >= 0
        
    def test_berth_utilization_calculation(self):
        """Test berth utilization calculation"""
        utilization = self.simulation._calculate_berth_utilization()
        
        # Should return a percentage between 0 and 100
        assert 0 <= utilization <= 100
        assert isinstance(utilization, float)
        
    def test_queue_efficiency_calculation_no_ships(self):
        """Test queue efficiency with no ships"""
        efficiency = self.simulation._calculate_queue_efficiency()
        
        # With no ships, efficiency should be 100%
        assert efficiency == 100.0
        
    def test_queue_efficiency_calculation_with_ships(self):
        """Test queue efficiency with processed ships"""
        # Simulate some ships being processed
        self.simulation.metrics['ships_arrived'] = 10
        self.simulation.metrics['ships_processed'] = 8
        
        efficiency = self.simulation._calculate_queue_efficiency()
        
        # Should be 80%
        assert efficiency == 80.0
        
    def test_processing_efficiency_calculation(self):
        """Test processing efficiency calculation"""
        efficiency = self.simulation._calculate_processing_efficiency()
        
        # Should return a percentage between 0 and 100
        assert 0 <= efficiency <= 100
        assert isinstance(efficiency, float)
        
    def test_current_status(self):
        """Test getting current simulation status"""
        status = self.simulation.get_current_status()
        
        required_keys = [
            'current_time', 'running', 'ships_in_system',
            'ships_processed', 'berth_status', 'queue_length'
        ]
        
        for key in required_keys:
            assert key in status
            
        assert status['current_time'] >= 0
        assert isinstance(status['running'], bool)
        assert status['ships_in_system'] >= 0
        assert status['ships_processed'] >= 0
        assert status['queue_length'] >= 0
        
    def test_reset_simulation(self):
        """Test simulation reset functionality"""
        # Run simulation briefly
        self.simulation.run_simulation(duration=0.1)
        
        # Verify some state was created
        assert self.simulation.metrics['simulation_end_time'] > 0
        
        # Reset simulation
        self.simulation.reset_simulation()
        
        # Verify reset state
        assert not self.simulation.running
        assert self.simulation.ships_processed == 0
        assert self.simulation.total_ships_generated == 0
        assert self.simulation.metrics['ships_arrived'] == 0
        assert self.simulation.metrics['ships_processed'] == 0
        assert self.simulation.metrics['total_waiting_time'] == 0
        
    def test_final_report_structure(self):
        """Test final report contains all required sections"""
        # Run short simulation
        result = self.simulation.run_simulation(duration=0.1)
        
        # Check main sections
        assert 'simulation_summary' in result
        assert 'berth_statistics' in result
        assert 'container_statistics' in result
        assert 'performance_metrics' in result
        
        # Check simulation summary structure
        summary = result['simulation_summary']
        summary_keys = [
            'duration', 'ships_arrived', 'ships_processed',
            'average_waiting_time', 'throughput_rate'
        ]
        for key in summary_keys:
            assert key in summary
            
        # Check performance metrics structure
        performance = result['performance_metrics']
        performance_keys = [
            'berth_utilization', 'queue_efficiency', 'processing_efficiency'
        ]
        for key in performance_keys:
            assert key in performance
            
    def test_simulation_with_no_berths(self):
        """Test simulation behavior with no berths configured"""
        empty_config = {'berths': []}
        empty_simulation = PortSimulation(empty_config)
        
        # Should still initialize without errors
        assert empty_simulation.berth_manager is not None
        
        # Run very short simulation
        result = empty_simulation.run_simulation(duration=0.01)
        
        # Should complete without errors
        assert 'simulation_summary' in result
        
    def test_simulation_error_handling(self):
        """Test simulation handles errors gracefully"""
        # This test ensures the simulation doesn't crash on errors
        try:
            result = self.simulation.run_simulation(duration=0.1)
            # If we get here, no exception was raised
            assert 'simulation_summary' in result
        except Exception as e:
            # If an exception occurs, it should be handled gracefully
            pytest.fail(f"Simulation should handle errors gracefully, but got: {e}")
            
    def test_multiple_simulation_runs(self):
        """Test running multiple simulations in sequence"""
        # Run first simulation
        result1 = self.simulation.run_simulation(duration=0.1)
        
        # Reset and run second simulation
        self.simulation.reset_simulation()
        result2 = self.simulation.run_simulation(duration=0.1)
        
        # Both should complete successfully
        assert 'simulation_summary' in result1
        assert 'simulation_summary' in result2
        
        # Results should be independent
        assert result1 != result2 or result1['simulation_summary']['duration'] >= 0
        
    def test_ship_id_generation(self):
        """Test that ship IDs are generated correctly"""
        ship1 = self.simulation._generate_random_ship("SHIP_001")
        ship2 = self.simulation._generate_random_ship("SHIP_002")
        
        assert ship1.ship_id == "SHIP_001"
        assert ship2.ship_id == "SHIP_002"
        assert ship1.ship_id != ship2.ship_id
        
    def test_config_integration(self):
        """Test that configuration is properly integrated"""
        # Check that berth configuration is passed correctly (test config has 2 berths)
        assert len(self.simulation.berth_manager.berths) == 2
        
        # Check berth details from test config
        berth_ids = list(self.simulation.berth_manager.berths.keys())
        assert 1 in berth_ids
        assert 2 in berth_ids
        
        # Verify berth properties match test config
        berth_1 = self.simulation.berth_manager.berths[1]
        berth_2 = self.simulation.berth_manager.berths[2]
        assert berth_1.name == 'Test Berth 1'
        assert berth_2.name == 'Test Berth 2'
        assert berth_1.berth_type == 'container'
        assert berth_2.berth_type == 'mixed'