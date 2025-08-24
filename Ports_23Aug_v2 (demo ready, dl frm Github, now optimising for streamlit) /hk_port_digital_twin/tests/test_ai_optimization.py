# Test suite for AI Optimization Layer
# Tests for optimization.py, predictive_models.py, and decision_support.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import the AI modules to test
sys_path_added = False
try:
    from src.ai.optimization import (
        Ship, Berth, OptimizationResult,
        BerthAllocationOptimizer, ContainerHandlingScheduler, ResourceAllocationOptimizer
    )
    from src.ai.predictive_models import (
        ArrivalPrediction, ProcessingTimePrediction, QueueForecast,
        ShipArrivalPredictor, ProcessingTimeEstimator, QueueLengthForecaster
    )
    from src.ai.decision_support import (
        Recommendation, DecisionContext, DecisionSupportEngine,
        RecommendationType, Priority
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    sys_path_added = True
    from src.ai.optimization import (
        Ship, Berth, OptimizationResult,
        BerthAllocationOptimizer, ContainerHandlingScheduler, ResourceAllocationOptimizer
    )
    from src.ai.predictive_models import (
        ArrivalPrediction, ProcessingTimePrediction, QueueForecast,
        ShipArrivalPredictor, ProcessingTimeEstimator, QueueLengthForecaster
    )
    from src.ai.decision_support import (
        Recommendation, DecisionContext, DecisionSupportEngine,
        RecommendationType, Priority
    )

class TestOptimization:
    """Test cases for optimization.py"""
    
    def test_ship_dataclass(self):
        """Test Ship dataclass creation and attributes"""
        ship = Ship(
            id="TEST001",
            arrival_time=datetime.now(),
            ship_type="container",
            size=3000,
            priority=1,
            containers_to_load=100,
            containers_to_unload=50
        )
        
        assert ship.id == "TEST001"
        assert ship.ship_type == "container"
        assert ship.size == 3000
        assert ship.containers_to_load == 100
        assert ship.containers_to_unload == 50
        assert ship.priority == 1
        assert isinstance(ship.arrival_time, datetime)
    
    def test_berth_dataclass(self):
        """Test Berth dataclass creation and attributes"""
        berth = Berth(
            id="BERTH001",
            capacity=5000,
            crane_count=3,
            suitable_ship_types=["container"],
            is_available=True
        )
        
        assert berth.id == "BERTH001"
        assert berth.capacity == 5000
        assert berth.crane_count == 3
        assert berth.suitable_ship_types == ["container"]
        assert berth.is_available is True
    
    def test_berth_allocation_optimizer_initialization(self):
        """Test BerthAllocationOptimizer initialization"""
        optimizer = BerthAllocationOptimizer()
        
        assert optimizer.ships == []
        assert optimizer.berths == []
    
    def test_berth_allocation_optimizer_add_ship(self):
        """Test adding ships to optimizer"""
        optimizer = BerthAllocationOptimizer()
        ship = Ship("TEST001", "container", 3000, 150, datetime.now(), 1)
        
        optimizer.add_ship(ship)
        
        assert len(optimizer.ships) == 1
        assert optimizer.ships[0] == ship
    
    def test_berth_allocation_optimizer_add_berth(self):
        """Test adding berths to optimizer"""
        optimizer = BerthAllocationOptimizer()
        berth = Berth("BERTH001", "container", 5000, True, (0, 0))
        
        optimizer.add_berth(berth)
        
        assert len(optimizer.berths) == 1
        assert optimizer.berths[0] == berth
    
    def test_estimate_service_time(self):
        """Test service time estimation"""
        optimizer = BerthAllocationOptimizer()
        ship = Ship("TEST001", datetime.now(), "container", 3000, 1, containers_to_load=100, containers_to_unload=50)
        berth = Berth("BERTH001", 5000, 3, ["container"], True)
        
        service_time = optimizer.estimate_service_time(ship, berth)
        
        assert isinstance(service_time, float)
        assert service_time > 0
    
    def test_check_berth_suitability(self):
        """Test berth suitability checking"""
        optimizer = BerthAllocationOptimizer()
        ship = Ship("TEST001", datetime.now(), "container", 3000, 1)
        berth = Berth("BERTH001", 5000, 3, ["container"], True)
        
        is_suitable = optimizer.is_berth_suitable(ship, berth)
        
        assert isinstance(is_suitable, bool)
        assert is_suitable is True  # Compatible types and sufficient capacity
    
    def test_berth_suitability_incompatible_type(self):
        """Test berth suitability with incompatible ship type"""
        optimizer = BerthAllocationOptimizer()
        ship = Ship("TEST001", datetime.now(), "tanker", 3000, 1)
        berth = Berth("BERTH001", 5000, 3, ["container"], True)
        
        is_suitable = optimizer.is_berth_suitable(ship, berth)
        
        assert is_suitable is False  # Incompatible types
    
    def test_berth_suitability_insufficient_capacity(self):
        """Test berth suitability with insufficient capacity"""
        optimizer = BerthAllocationOptimizer()
        ship = Ship("TEST001", datetime.now(), "container", 6000, 1)  # Larger than berth capacity
        berth = Berth("BERTH001", 5000, 3, ["container"], True)
        
        is_suitable = optimizer.is_berth_suitable(ship, berth)
        
        assert is_suitable is False  # Insufficient capacity
    
    def test_optimize_allocation_empty_inputs(self):
        """Test optimization with empty inputs"""
        optimizer = BerthAllocationOptimizer()
        
        result = optimizer.optimize_berth_allocation()
        
        assert isinstance(result, OptimizationResult)
        assert result.ship_berth_assignments == {}
        assert result.total_waiting_time >= 0
        assert result.average_waiting_time >= 0
    
    def test_optimize_allocation_with_data(self):
        """Test optimization with actual data"""
        optimizer = BerthAllocationOptimizer()
        
        # Add ships and berths
        ship1 = Ship("SHIP001", datetime.now(), "container", 3000, 1, containers_to_load=100, containers_to_unload=50)
        ship2 = Ship("SHIP002", datetime.now(), "bulk", 4000, 2)
        berth1 = Berth("BERTH001", 5000, 3, ["container"], True)
        berth2 = Berth("BERTH002", 6000, 2, ["bulk"], True)
        
        optimizer.add_ship(ship1)
        optimizer.add_ship(ship2)
        optimizer.add_berth(berth1)
        optimizer.add_berth(berth2)
        
        result = optimizer.optimize_berth_allocation()
        
        assert isinstance(result, OptimizationResult)
        assert len(result.ship_berth_assignments) <= 2  # At most 2 allocations
        assert result.total_waiting_time >= 0
        assert result.average_waiting_time >= 0

class TestPredictiveModels:
    """Test cases for predictive_models.py"""
    
    def test_arrival_prediction_dataclass(self):
        """Test ArrivalPrediction dataclass"""
        prediction = ArrivalPrediction(
            predicted_time=datetime.now(),
            confidence_interval=(datetime.now(), datetime.now() + timedelta(hours=1)),
            probability=0.8,
            ship_type="container",
            factors={"base_rate": 1.0}
        )
        
        assert isinstance(prediction.predicted_time, datetime)
        assert isinstance(prediction.confidence_interval, tuple)
        assert 0 <= prediction.probability <= 1
        assert prediction.ship_type == "container"
        assert isinstance(prediction.factors, dict)
    
    def test_ship_arrival_predictor_initialization(self):
        """Test ShipArrivalPredictor initialization"""
        predictor = ShipArrivalPredictor()
        
        assert predictor.historical_data.empty
        assert isinstance(predictor.seasonal_patterns, dict)
        assert isinstance(predictor.ship_type_patterns, dict)
        assert predictor.is_trained is False
    
    def test_load_historical_data(self):
        """Test loading historical vessel data"""
        predictor = ShipArrivalPredictor()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'arrival_time': pd.date_range(start='2024-01-01', periods=10, freq='6H'),
            'ship_type': ['container'] * 5 + ['bulk'] * 5,
            'size': np.random.uniform(1000, 5000, 10)
        })
        
        predictor.load_historical_data(sample_data)
        
        assert not predictor.historical_data.empty
        assert len(predictor.historical_data) == 10
        assert 'hour' in predictor.historical_data.columns
        assert 'day_of_week' in predictor.historical_data.columns
    
    def test_analyze_seasonal_patterns(self):
        """Test seasonal pattern analysis"""
        predictor = ShipArrivalPredictor()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'arrival_time': pd.date_range(start='2024-01-01', periods=100, freq='6H'),
            'ship_type': np.random.choice(['container', 'bulk'], 100),
            'size': np.random.uniform(1000, 5000, 100)
        })
        
        predictor.load_historical_data(sample_data)
        predictor.analyze_seasonal_patterns()
        
        assert 'hourly' in predictor.seasonal_patterns
        assert 'daily' in predictor.seasonal_patterns
        assert 'monthly' in predictor.seasonal_patterns
        assert len(predictor.ship_type_patterns) > 0
    
    def test_train_arrival_model(self):
        """Test arrival model training"""
        predictor = ShipArrivalPredictor()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'arrival_time': pd.date_range(start='2024-01-01', periods=50, freq='6H'),
            'ship_type': np.random.choice(['container', 'bulk'], 50),
            'size': np.random.uniform(1000, 5000, 50)
        })
        
        predictor.load_historical_data(sample_data)
        predictor.train_arrival_model()
        
        assert predictor.is_trained is True
    
    def test_predict_next_arrival_untrained(self):
        """Test arrival prediction without training"""
        predictor = ShipArrivalPredictor()
        
        prediction = predictor.predict_next_arrival("container")
        
        assert isinstance(prediction, ArrivalPrediction)
        assert prediction.ship_type == "container"
        assert prediction.probability > 0
    
    def test_predict_next_arrival_trained(self):
        """Test arrival prediction with trained model"""
        predictor = ShipArrivalPredictor()
        
        # Create and load sample data
        sample_data = pd.DataFrame({
            'arrival_time': pd.date_range(start='2024-01-01', periods=50, freq='6H'),
            'ship_type': np.random.choice(['container', 'bulk'], 50),
            'size': np.random.uniform(1000, 5000, 50)
        })
        
        predictor.load_historical_data(sample_data)
        predictor.train_arrival_model()
        
        prediction = predictor.predict_next_arrival("container")
        
        assert isinstance(prediction, ArrivalPrediction)
        assert prediction.ship_type == "container"
        assert prediction.probability > 0
        assert 'base_rate' in prediction.factors
    
    def test_processing_time_estimator_initialization(self):
        """Test ProcessingTimeEstimator initialization"""
        estimator = ProcessingTimeEstimator()
        
        assert isinstance(estimator.historical_processing_times, dict)
        assert isinstance(estimator.type_factors, dict)
        assert estimator.is_trained is False
    
    def test_estimate_processing_time_default(self):
        """Test processing time estimation with default values"""
        estimator = ProcessingTimeEstimator()
        
        prediction = estimator.estimate_processing_time("container", 3000, 150)
        
        assert isinstance(prediction, ProcessingTimePrediction)
        assert prediction.estimated_hours > 0
        assert len(prediction.confidence_interval) == 2
        assert prediction.uncertainty >= 0
    
    def test_queue_length_forecaster_initialization(self):
        """Test QueueLengthForecaster initialization"""
        forecaster = QueueLengthForecaster()
        
        assert forecaster.historical_queue_data.empty
        assert isinstance(forecaster.arrival_predictor, ShipArrivalPredictor)
        assert isinstance(forecaster.processing_estimator, ProcessingTimeEstimator)
        assert forecaster.current_queue == []
    
    def test_update_current_queue(self):
        """Test updating current queue state"""
        forecaster = QueueLengthForecaster()
        
        ships = [
            {'ship_id': 'SHIP001', 'type': 'container'},
            {'ship_id': 'SHIP002', 'type': 'bulk'}
        ]
        
        forecaster.update_current_queue(ships)
        
        assert len(forecaster.current_queue) == 2
        assert forecaster.current_queue == ships
    
    def test_forecast_queue_length(self):
        """Test queue length forecasting"""
        forecaster = QueueLengthForecaster()
        
        # Set up current queue
        ships = [
            {'ship_id': 'SHIP001', 'type': 'container'},
            {'ship_id': 'SHIP002', 'type': 'bulk'}
        ]
        forecaster.update_current_queue(ships)
        
        forecasts = forecaster.forecast_queue_length(6)  # 6 hours ahead
        
        assert len(forecasts) == 6
        for forecast in forecasts:
            assert isinstance(forecast, QueueForecast)
            assert forecast.predicted_queue_length >= 0
            assert forecast.predicted_waiting_time >= 0
            assert 0 <= forecast.confidence <= 1
            assert forecast.trend in ['increasing', 'decreasing', 'stable']

class TestDecisionSupport:
    """Test cases for decision_support.py"""
    
    def test_recommendation_dataclass(self):
        """Test Recommendation dataclass"""
        recommendation = Recommendation(
            id="TEST001",
            type=RecommendationType.BERTH_ALLOCATION,
            priority=Priority.HIGH,
            title="Test Recommendation",
            description="Test description",
            rationale="Test rationale",
            expected_impact={"efficiency": 0.1},
            implementation_steps=["Step 1", "Step 2"],
            estimated_cost=1000.0,
            estimated_savings=5000.0,
            timeline="1 week",
            confidence=0.8,
            created_at=datetime.now()
        )
        
        assert recommendation.id == "TEST001"
        assert recommendation.type == RecommendationType.BERTH_ALLOCATION
        assert recommendation.priority == Priority.HIGH
        assert recommendation.estimated_cost == 1000.0
        assert recommendation.estimated_savings == 5000.0
    
    def test_decision_context_dataclass(self):
        """Test DecisionContext dataclass"""
        context = DecisionContext(
            current_time=datetime.now(),
            port_status={"operational": True},
            active_ships=[],
            available_berths=[],
            resource_utilization={"berths": 0.8},
            weather_conditions={"wind_speed": 10},
            operational_constraints=[],
            performance_metrics={"efficiency": 0.85}
        )
        
        assert isinstance(context.current_time, datetime)
        assert isinstance(context.port_status, dict)
        assert isinstance(context.active_ships, list)
        assert isinstance(context.resource_utilization, dict)
    
    def test_decision_support_engine_initialization(self):
        """Test DecisionSupportEngine initialization"""
        engine = DecisionSupportEngine()
        
        assert isinstance(engine.berth_optimizer, BerthAllocationOptimizer)
        assert isinstance(engine.arrival_predictor, ShipArrivalPredictor)
        assert isinstance(engine.active_recommendations, list)
        assert isinstance(engine.recommendation_rules, dict)
        assert isinstance(engine.thresholds, dict)
    
    def test_analyze_situation_empty_context(self):
        """Test situation analysis with minimal context"""
        engine = DecisionSupportEngine()
        
        context = DecisionContext(
            current_time=datetime.now(),
            port_status={},
            active_ships=[],
            available_berths=[],
            resource_utilization={},
            weather_conditions={},
            operational_constraints=[],
            performance_metrics={}
        )
        
        recommendations = engine.analyze_situation(context)
        
        assert isinstance(recommendations, list)
        # Should return empty list or minimal recommendations for empty context
    
    def test_analyze_situation_with_issues(self):
        """Test situation analysis with problematic context"""
        engine = DecisionSupportEngine()
        
        context = DecisionContext(
            current_time=datetime.now(),
            port_status={"operational": True},
            active_ships=[
                {'id': 'SHIP001', 'waiting_time': 10.0},  # Critical waiting time
                {'id': 'SHIP002', 'waiting_time': 2.0},
                {'id': 'SHIP003', 'waiting_time': 3.0},
                {'id': 'SHIP004', 'waiting_time': 1.0},
                {'id': 'SHIP005', 'waiting_time': 2.5},
                {'id': 'SHIP006', 'waiting_time': 4.0}  # Long queue
            ],
            available_berths=[
                {'id': 'BERTH001', 'occupied': True},
                {'id': 'BERTH002', 'occupied': True}
            ],
            resource_utilization={
                'berths': 0.95,  # High utilization
                'cranes': 0.95,  # High utilization
                'trucks': 0.6,
                'workers': 0.7
            },
            weather_conditions={'impact_score': 0.8},  # Severe weather
            operational_constraints=[],
            performance_metrics={
                'berth_utilization': 0.7,  # Below threshold
                'average_waiting_time': 5.0,  # Above threshold
                'customer_satisfaction': 0.8  # Below threshold
            }
        )
        
        recommendations = engine.analyze_situation(context)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0  # Should generate recommendations for issues
        
        # Check for critical recommendations
        critical_recs = [r for r in recommendations if r.priority == Priority.CRITICAL]
        assert len(critical_recs) > 0  # Should have critical recommendations
    
    def test_get_recommendation_summary_empty(self):
        """Test recommendation summary with no recommendations"""
        engine = DecisionSupportEngine()
        
        summary = engine.get_recommendation_summary()
        
        assert summary['total_recommendations'] == 0
        assert summary['total_estimated_savings'] == 0.0
        assert summary['total_estimated_cost'] == 0.0
    
    def test_get_recommendation_summary_with_data(self):
        """Test recommendation summary with recommendations"""
        engine = DecisionSupportEngine()
        
        # Add some test recommendations
        rec1 = Recommendation(
            id="TEST001",
            type=RecommendationType.BERTH_ALLOCATION,
            priority=Priority.HIGH,
            title="Test 1",
            description="Test",
            rationale="Test",
            expected_impact={},
            implementation_steps=[],
            estimated_cost=1000.0,
            estimated_savings=5000.0,
            timeline="1 week",
            confidence=0.8,
            created_at=datetime.now()
        )
        
        rec2 = Recommendation(
            id="TEST002",
            type=RecommendationType.RESOURCE_OPTIMIZATION,
            priority=Priority.MEDIUM,
            title="Test 2",
            description="Test",
            rationale="Test",
            expected_impact={},
            implementation_steps=[],
            estimated_cost=2000.0,
            estimated_savings=8000.0,
            timeline="2 weeks",
            confidence=0.7,
            created_at=datetime.now()
        )
        
        engine.active_recommendations = [rec1, rec2]
        
        summary = engine.get_recommendation_summary()
        
        assert summary['total_recommendations'] == 2
        assert summary['total_estimated_savings'] == 13000.0
        assert summary['total_estimated_cost'] == 3000.0
        assert summary['net_benefit'] == 10000.0
        assert summary['by_priority']['high'] == 1
        assert summary['by_priority']['medium'] == 1

class TestIntegration:
    """Integration tests for AI components working together"""
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow"""
        # Create optimizer
        optimizer = BerthAllocationOptimizer()
        
        # Add ships and berths
        ships = [
            Ship("SHIP001", datetime.now(), "container", 3000, 1, containers_to_load=100, containers_to_unload=50),
            Ship("SHIP002", datetime.now(), "bulk", 4000, 2)
        ]
        
        berths = [
            Berth("BERTH001", 5000, 3, ["container"], True),
            Berth("BERTH002", 6000, 2, ["bulk"], True)
        ]
        
        for ship in ships:
            optimizer.add_ship(ship)
        for berth in berths:
            optimizer.add_berth(berth)
        
        # Run optimization
        result = optimizer.optimize_berth_allocation()
        
        # Verify results
        assert isinstance(result, OptimizationResult)
        assert len(result.ship_berth_assignments) <= len(ships)
        assert result.total_waiting_time >= 0
    
    def test_end_to_end_prediction_workflow(self):
        """Test complete prediction workflow"""
        # Create predictor
        predictor = ShipArrivalPredictor()
        
        # Create sample historical data
        sample_data = pd.DataFrame({
            'arrival_time': pd.date_range(start='2024-01-01', periods=30, freq='8H'),
            'ship_type': np.random.choice(['container', 'bulk', 'tanker'], 30),
            'size': np.random.uniform(1000, 6000, 30)
        })
        
        # Train model
        predictor.load_historical_data(sample_data)
        predictor.train_arrival_model()
        
        # Make predictions
        prediction = predictor.predict_next_arrival("container")
        
        # Verify prediction
        assert isinstance(prediction, ArrivalPrediction)
        assert prediction.ship_type == "container"
        assert prediction.probability > 0
    
    def test_end_to_end_decision_support_workflow(self):
        """Test complete decision support workflow"""
        # Create decision support engine
        engine = DecisionSupportEngine()
        
        # Create realistic context with issues
        context = DecisionContext(
            current_time=datetime.now(),
            port_status={"operational": True, "capacity": 0.9},
            active_ships=[
                {'id': f'SHIP{i:03d}', 'waiting_time': np.random.uniform(1, 6), 'type': 'container'}
                for i in range(1, 8)  # 7 ships in queue
            ],
            available_berths=[
                {'id': f'BERTH{i:03d}', 'occupied': i <= 2} for i in range(1, 4)
            ],
            resource_utilization={
                'berths': 0.9,
                'cranes': 0.95,
                'trucks': 0.8,
                'workers': 0.85
            },
            weather_conditions={'impact_score': 0.3},
            operational_constraints=['Limited night operations'],
            performance_metrics={
                'berth_utilization': 0.75,
                'average_waiting_time': 4.5,
                'throughput_efficiency': 0.8,
                'customer_satisfaction': 0.85
            }
        )
        
        # Generate recommendations
        recommendations = engine.analyze_situation(context)
        
        # Verify recommendations
        assert isinstance(recommendations, list)
        if len(recommendations) > 0:
            for rec in recommendations:
                assert isinstance(rec, Recommendation)
                assert rec.estimated_cost >= 0
                assert rec.estimated_savings >= 0
                assert 0 <= rec.confidence <= 1
        
        # Get summary
        summary = engine.get_recommendation_summary()
        assert isinstance(summary, dict)
        assert 'total_recommendations' in summary

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])