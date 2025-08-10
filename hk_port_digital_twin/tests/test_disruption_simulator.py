import unittest
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import directly from the module file to avoid package import issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scenarios'))
from disruption_simulator import (
    DisruptionSimulator, DisruptionEvent, RecoveryStrategy,
    DisruptionType, DisruptionSeverity
)

class TestDisruptionSimulator(unittest.TestCase):
    """Test cases for DisruptionSimulator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = DisruptionSimulator()
        self.start_time = datetime(2024, 1, 15, 10, 0)
        
    def test_disruption_simulator_initialization(self):
        """Test DisruptionSimulator initialization"""
        self.assertIsInstance(self.simulator.active_disruptions, list)
        self.assertIsInstance(self.simulator.recovery_strategies, dict)
        self.assertEqual(len(self.simulator.active_disruptions), 0)
        self.assertGreater(len(self.simulator.recovery_strategies), 0)
        
    def test_create_disruption_event(self):
        """Test creating a disruption event"""
        event = self.simulator.create_disruption_event(
            disruption_type=DisruptionType.EQUIPMENT_FAILURE,
            severity=DisruptionSeverity.HIGH,
            start_time=self.start_time,
            duration_hours=6.0,
            affected_berths=["Berth_1", "Berth_2"]
        )
        
        self.assertIsInstance(event, DisruptionEvent)
        self.assertEqual(event.disruption_type, DisruptionType.EQUIPMENT_FAILURE)
        self.assertEqual(event.severity, DisruptionSeverity.HIGH)
        self.assertEqual(event.start_time, self.start_time)
        self.assertEqual(event.duration_hours, 6.0)
        self.assertIn("Berth_1", event.affected_berths)
        
    def test_calculate_disruption_impact(self):
        """Test disruption impact calculation"""
        capacity_reduction, processing_increase = self.simulator._calculate_disruption_impact(
            DisruptionType.EQUIPMENT_FAILURE,
            DisruptionSeverity.HIGH
        )
        
        self.assertIsInstance(capacity_reduction, float)
        self.assertIsInstance(processing_increase, float)
        self.assertGreaterEqual(capacity_reduction, 0.0)
        self.assertLessEqual(capacity_reduction, 1.0)
        self.assertGreaterEqual(processing_increase, 0.0)
        
    def test_severity_scaling(self):
        """Test severity scaling for different disruption types"""
        # Test different severity levels
        low_impact = self.simulator._calculate_disruption_impact(
            DisruptionType.CONGESTION, DisruptionSeverity.LOW
        )
        high_impact = self.simulator._calculate_disruption_impact(
            DisruptionType.CONGESTION, DisruptionSeverity.HIGH
        )
        
        # High severity should have greater impact
        self.assertGreater(high_impact[0], low_impact[0])  # capacity reduction
        self.assertGreater(high_impact[1], low_impact[1])  # processing increase
        
    def test_simulate_disruption_impact(self):
        """Test full disruption impact simulation"""
        event = DisruptionEvent(
            event_id="test_002",
            disruption_type=DisruptionType.WEATHER,
            severity=DisruptionSeverity.CRITICAL,
            start_time=self.start_time,
            duration_hours=12.0,
            capacity_reduction=0.8,
            processing_time_increase=0.5
        )
        
        baseline_metrics = {
            'throughput': 1000.0,
            'waiting_time': 2.5,
            'berth_utilization': 0.75
        }
        
        result = self.simulator.simulate_disruption_impact(
            event, baseline_metrics, simulation_duration=24.0
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('impact_metrics', result)
        self.assertIn('recovery_timeline', result)
        self.assertIn('recommendations', result)
        
    def test_weather_impact_severity(self):
        """Test weather impact with different severity levels"""
        weather_low = self.simulator._calculate_disruption_impact(
            DisruptionType.WEATHER, DisruptionSeverity.LOW
        )
        weather_critical = self.simulator._calculate_disruption_impact(
            DisruptionType.WEATHER, DisruptionSeverity.CRITICAL
        )
        
        # Critical weather should have much higher impact
        self.assertGreater(weather_critical[0], weather_low[0])
        self.assertGreater(weather_critical[1], weather_low[1])
        
    def test_generate_recovery_recommendations(self):
        """Test recommendation generation"""
        event = DisruptionEvent(
            event_id="test_003",
            disruption_type=DisruptionType.EQUIPMENT_FAILURE,
            severity=DisruptionSeverity.HIGH,
            start_time=self.start_time,
            duration_hours=4.0,
            capacity_reduction=0.4,
            processing_time_increase=0.3
        )
        
        recommendations = self.simulator._generate_recovery_recommendations(event)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check recommendation structure
        for rec in recommendations:
            self.assertIn('strategy', rec)
            self.assertIn('effectiveness', rec)
            self.assertIn('implementation_time', rec)
            
    def test_sample_disruption_scenarios(self):
        """Test creating sample disruption scenarios"""
        scenarios = self.simulator.create_sample_disruption_scenarios()
        
        self.assertIsInstance(scenarios, list)
        self.assertGreater(len(scenarios), 0)
        
        # Check that all scenarios are DisruptionEvent instances
        for scenario in scenarios:
            self.assertIsInstance(scenario, DisruptionEvent)
            
    def test_disruption_comparison(self):
        """Test disruption scenario comparison"""
        scenarios = self.simulator.create_sample_disruption_scenarios()
        
        # Take first 2 scenarios for comparison
        comparison_result = self.simulator.run_disruption_comparison(scenarios[:2])
        
        self.assertIsInstance(comparison_result, dict)
        self.assertIn('scenarios', comparison_result)
        self.assertIn('comparison_metrics', comparison_result)
        self.assertIn('summary', comparison_result)
        
    def test_recovery_strategy_initialization(self):
        """Test that recovery strategies are properly initialized"""
        strategies = self.simulator.recovery_strategies
        
        self.assertGreater(len(strategies), 0)
        
        # Check that strategies have required attributes
        for strategy_id, strategy in strategies.items():
            self.assertIsInstance(strategy, RecoveryStrategy)
            self.assertIsInstance(strategy.effectiveness, float)
            self.assertGreaterEqual(strategy.effectiveness, 0.0)
            self.assertLessEqual(strategy.effectiveness, 1.0)

if __name__ == '__main__':
    unittest.main()