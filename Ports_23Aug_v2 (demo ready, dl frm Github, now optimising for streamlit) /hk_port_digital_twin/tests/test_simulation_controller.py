"""Tests for SimulationController class"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from src.core.simulation_controller import SimulationController, SimulationState
from src.core.port_simulation import PortSimulation
from src.utils.metrics_collector import MetricsCollector
from src.core.ship_manager import Ship, ShipState
from src.core.berth_manager import BerthManager
from src.core.container_handler import ContainerHandler


class TestSimulationController(unittest.TestCase):
    """Test cases for SimulationController"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test configuration
        self.config = {
            'berths': [
                {'berth_id': 1, 'berth_name': 'Berth 1', 'berth_type': 'container', 'max_capacity_teu': 50000, 'crane_count': 2},
                 {'berth_id': 2, 'berth_name': 'Berth 2', 'berth_type': 'container', 'max_capacity_teu': 60000, 'crane_count': 3}
            ]
        }
        
        self.metrics_collector = MetricsCollector()
        
        # Create port simulation
        self.port_simulation = PortSimulation(config=self.config)
        
        # Create simulation controller
        self.controller = SimulationController(
            port_simulation=self.port_simulation,
            metrics_collector=self.metrics_collector
        )
    
    def test_initialization(self):
        """Test controller initialization"""
        self.assertEqual(self.controller.state, SimulationState.STOPPED)
        self.assertEqual(self.controller.current_time, 0.0)
        self.assertEqual(self.controller.duration, 0.0)
        self.assertEqual(self.controller.time_step, 1.0)
        self.assertIsNone(self.controller.thread)
        self.assertFalse(self.controller.stop_event.is_set())
        self.assertFalse(self.controller.pause_event.is_set())
    
    def test_start_simulation_invalid_duration(self):
        """Test starting simulation with invalid duration"""
        # Test zero duration
        result = self.controller.start(0.0)
        self.assertFalse(result)
        self.assertEqual(self.controller.state, SimulationState.STOPPED)
        
        # Test negative duration
        result = self.controller.start(-5.0)
        self.assertFalse(result)
        self.assertEqual(self.controller.state, SimulationState.STOPPED)
    
    def test_start_simulation_already_running(self):
        """Test starting simulation when already running"""
        # Start first simulation
        self.controller.start(10.0, threaded=False)
        
        # Try to start again
        result = self.controller.start(5.0)
        self.assertFalse(result)
    
    def test_start_simulation_non_threaded(self):
        """Test starting simulation in non-threaded mode"""
        with patch.object(self.controller, '_run_simulation') as mock_run:
            result = self.controller.start(10.0, time_step=0.5, threaded=False)
            
            self.assertTrue(result)
            self.assertEqual(self.controller.duration, 10.0)
            self.assertEqual(self.controller.time_step, 0.5)
            mock_run.assert_called_once()
    
    def test_start_simulation_threaded(self):
        """Test starting simulation in threaded mode"""
        result = self.controller.start(1.0, threaded=True)
        
        self.assertTrue(result)
        self.assertEqual(self.controller.duration, 1.0)
        self.assertIsNotNone(self.controller.thread)
        
        # Wait for thread to start
        time.sleep(0.1)
        
        # Stop the simulation
        self.controller.stop()
    
    def test_stop_simulation(self):
        """Test stopping simulation"""
        # Test stopping when not running
        result = self.controller.stop()
        self.assertFalse(result)
        
        # Start simulation and then stop
        self.controller.start(10.0, threaded=True)
        time.sleep(0.1)  # Let it start
        
        result = self.controller.stop()
        self.assertTrue(result)
        self.assertEqual(self.controller.state, SimulationState.STOPPED)
    
    def test_pause_resume_simulation(self):
        """Test pausing and resuming simulation"""
        # Test pause when not running
        result = self.controller.pause()
        self.assertFalse(result)
        
        # Test resume when not paused
        result = self.controller.resume()
        self.assertFalse(result)
        
        # Start simulation
        self.controller.start(10.0, threaded=True)
        time.sleep(0.1)  # Let it start
        
        # Pause simulation
        result = self.controller.pause()
        self.assertTrue(result)
        self.assertEqual(self.controller.state, SimulationState.PAUSED)
        
        # Resume simulation
        result = self.controller.resume()
        self.assertTrue(result)
        self.assertEqual(self.controller.state, SimulationState.RUNNING)
        
        # Stop simulation
        self.controller.stop()
    
    def test_reset_simulation(self):
        """Test resetting simulation"""
        # Start and then reset
        self.controller.start(5.0, threaded=True)
        time.sleep(0.1)
        
        result = self.controller.reset()
        self.assertTrue(result)
        self.assertEqual(self.controller.state, SimulationState.STOPPED)
        self.assertEqual(self.controller.current_time, 0.0)
        self.assertEqual(self.controller.duration, 0.0)
    
    def test_reset_simulation_with_error(self):
        """Test reset when simulation.reset_simulation() raises exception"""
        with patch.object(self.port_simulation, 'reset_simulation', side_effect=Exception("Reset error")):
            result = self.controller.reset()
            self.assertFalse(result)
            self.assertEqual(self.controller.state, SimulationState.ERROR)
    
    def test_progress_tracking(self):
        """Test progress tracking methods"""
        # Test initial progress
        current, duration = self.controller.get_progress()
        self.assertEqual(current, 0.0)
        self.assertEqual(duration, 0.0)
        
        percentage = self.controller.get_progress_percentage()
        self.assertEqual(percentage, 0.0)
        
        # Set some progress
        self.controller.current_time = 5.0
        self.controller.duration = 10.0
        
        current, duration = self.controller.get_progress()
        self.assertEqual(current, 5.0)
        self.assertEqual(duration, 10.0)
        
        percentage = self.controller.get_progress_percentage()
        self.assertEqual(percentage, 50.0)
        
        # Test progress over 100%
        self.controller.current_time = 15.0
        percentage = self.controller.get_progress_percentage()
        self.assertEqual(percentage, 100.0)
    
    def test_state_queries(self):
        """Test state query methods"""
        # Initial state
        self.assertFalse(self.controller.is_running())
        self.assertFalse(self.controller.is_paused())
        self.assertFalse(self.controller.is_completed())
        self.assertEqual(self.controller.get_state(), SimulationState.STOPPED)
        
        # Running state
        self.controller._set_state(SimulationState.RUNNING)
        self.assertTrue(self.controller.is_running())
        self.assertFalse(self.controller.is_paused())
        self.assertFalse(self.controller.is_completed())
        
        # Paused state
        self.controller._set_state(SimulationState.PAUSED)
        self.assertFalse(self.controller.is_running())
        self.assertTrue(self.controller.is_paused())
        self.assertFalse(self.controller.is_completed())
        
        # Completed state
        self.controller._set_state(SimulationState.COMPLETED)
        self.assertFalse(self.controller.is_running())
        self.assertFalse(self.controller.is_paused())
        self.assertTrue(self.controller.is_completed())
    
    def test_state_change_callback(self):
        """Test state change callback functionality"""
        callback_calls = []
        
        def state_callback(state):
            callback_calls.append(state)
        
        self.controller.on_state_change = state_callback
        
        # Change states and verify callbacks
        self.controller._set_state(SimulationState.RUNNING)
        self.controller._set_state(SimulationState.PAUSED)
        self.controller._set_state(SimulationState.STOPPED)
        
        expected_calls = [
            SimulationState.RUNNING,
            SimulationState.PAUSED,
            SimulationState.STOPPED
        ]
        self.assertEqual(callback_calls, expected_calls)
    
    def test_progress_update_callback(self):
        """Test progress update callback functionality"""
        progress_updates = []
        
        def progress_callback(current, total):
            progress_updates.append((current, total))
        
        self.controller.on_progress_update = progress_callback
        
        # Mock the simulation step to avoid actual processing
        with patch.object(self.controller, '_run_simulation_step'):
            # Run a short simulation
            self.controller.start(2.0, time_step=1.0, threaded=False)
        
        # Should have progress updates
        self.assertGreater(len(progress_updates), 0)
        
        # Check that updates contain expected values
        for current, total in progress_updates:
            self.assertEqual(total, 2.0)
            self.assertGreaterEqual(current, 0.0)
            self.assertLessEqual(current, 2.0)
    
    def test_simulation_step_processing(self):
        """Test simulation step processing"""
        # Mock the SimPy environment run method
        with patch.object(self.port_simulation.env, 'run') as mock_run:
            
            self.controller.current_time = 5.0
            self.controller.time_step = 1.0
            
            self.controller._run_simulation_step()
            
            # Verify environment was advanced
            mock_run.assert_called_once()
    
    def test_simulation_step_with_ships(self):
        """Test simulation step with ship generation"""
        # Mock the SimPy environment and metrics collection
        with patch.object(self.port_simulation.env, 'run') as mock_run, \
             patch.object(self.controller.metrics_collector, 'record_queue_length') as mock_queue, \
             patch.object(self.controller.metrics_collector, 'record_berth_utilization') as mock_berth:
            
            self.controller.current_time = 5.0
            self.controller.time_step = 1.0
            
            self.controller._run_simulation_step()
            
            # Verify environment was advanced and metrics recorded
            mock_run.assert_called_once()
            mock_queue.assert_called_once()
            mock_berth.assert_called()
    
    def test_metrics_integration(self):
        """Test metrics collection integration"""
        # Test getting metrics summary
        summary = self.controller.get_metrics()
        self.assertIsInstance(summary, dict)
        
        # Test exporting metrics
        dataframes = self.controller.export_metrics_to_dataframe()
        self.assertIsInstance(dataframes, dict)
    
    def test_simulation_completion(self):
        """Test simulation running to completion"""
        with patch.object(self.controller, '_run_simulation_step'):
            # Run very short simulation
            self.controller.start(0.1, time_step=0.1, threaded=False)
            
            # Should be completed
            self.assertEqual(self.controller.state, SimulationState.COMPLETED)
            self.assertGreaterEqual(self.controller.current_time, 0.1)
    
    def test_simulation_error_handling(self):
        """Test simulation error handling"""
        with patch.object(self.controller, '_run_simulation_step', side_effect=Exception("Test error")):
            # Run simulation that will error
            self.controller.start(1.0, threaded=False)
            
            # Should be in error state
            self.assertEqual(self.controller.state, SimulationState.ERROR)
    
    def test_custom_metrics_collector(self):
        """Test using custom metrics collector"""
        custom_metrics = MetricsCollector()
        controller = SimulationController(
            port_simulation=self.port_simulation,
            metrics_collector=custom_metrics
        )
        
        self.assertIs(controller.metrics_collector, custom_metrics)
    
    def test_default_metrics_collector(self):
        """Test default metrics collector creation"""
        controller = SimulationController(port_simulation=self.port_simulation)
        
        self.assertIsInstance(controller.metrics_collector, MetricsCollector)


if __name__ == '__main__':
    unittest.main()