"""Tests for Performance Benchmarking Framework

This module tests the comprehensive performance benchmarking capabilities
for the Hong Kong Port Digital Twin simulation system.
"""

import pytest
import json
import tempfile
from datetime import datetime
from pathlib import Path

from src.analysis.performance_benchmarking import (
    PerformanceBenchmarking,
    BenchmarkMetric,
    BenchmarkCategory,
    PerformanceLevel,
    BenchmarkReport,
    create_benchmark_analysis,
    get_performance_summary
)

class TestBenchmarkMetric:
    """Test BenchmarkMetric functionality"""
    
    def test_metric_creation(self):
        """Test creating a benchmark metric"""
        metric = BenchmarkMetric(
            metric_id="test_metric",
            name="Test Metric",
            category=BenchmarkCategory.THROUGHPUT,
            unit="units/hour",
            description="A test metric",
            historical_baseline=50.0,
            industry_average=60.0,
            world_class_target=80.0
        )
        
        assert metric.metric_id == "test_metric"
        assert metric.category == BenchmarkCategory.THROUGHPUT
        assert metric.current_value is None
        assert metric.performance_level is None
    
    def test_performance_level_calculation(self):
        """Test performance level calculation"""
        metric = BenchmarkMetric(
            metric_id="test_metric",
            name="Test Metric",
            category=BenchmarkCategory.THROUGHPUT,
            unit="units/hour",
            description="A test metric",
            historical_baseline=50.0,
            industry_average=60.0,
            world_class_target=80.0
        )
        
        # Test excellent performance
        metric.current_value = 78.0
        assert metric.calculate_performance_level() == PerformanceLevel.EXCELLENT
        
        # Test good performance
        metric.current_value = 65.0
        assert metric.calculate_performance_level() == PerformanceLevel.GOOD
        
        # Test poor performance
        metric.current_value = 30.0
        assert metric.calculate_performance_level() == PerformanceLevel.POOR
    
    def test_improvement_potential_calculation(self):
        """Test improvement potential calculation"""
        metric = BenchmarkMetric(
            metric_id="test_metric",
            name="Test Metric",
            category=BenchmarkCategory.THROUGHPUT,
            unit="units/hour",
            description="A test metric",
            historical_baseline=50.0,
            industry_average=60.0,
            world_class_target=80.0
        )
        
        metric.current_value = 60.0
        potential = metric.calculate_improvement_potential()
        expected = ((80.0 - 60.0) / 60.0) * 100  # 33.33%
        assert abs(potential - expected) < 0.01

class TestPerformanceBenchmarking:
    """Test PerformanceBenchmarking class"""
    
    @pytest.fixture
    def temp_benchmarks_dir(self):
        """Create temporary directory for benchmarks"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def benchmarking(self, temp_benchmarks_dir):
        """Create PerformanceBenchmarking instance with temp directory"""
        return PerformanceBenchmarking(temp_benchmarks_dir)
    
    def test_initialization(self, benchmarking):
        """Test benchmarking system initialization"""
        assert len(benchmarking.benchmark_metrics) > 0
        assert "container_throughput_teu_hour" in benchmarking.benchmark_metrics
        assert "berth_utilization_rate" in benchmarking.benchmark_metrics
        assert len(benchmarking.historical_data) > 0
    
    def test_metric_update(self, benchmarking):
        """Test updating metric values"""
        metric_id = "container_throughput_teu_hour"
        test_value = 65.0
        
        benchmarking.update_metric_value(metric_id, test_value)
        
        metric = benchmarking.benchmark_metrics[metric_id]
        assert metric.current_value == test_value
        assert metric.performance_level is not None
        assert metric.improvement_potential is not None
    
    def test_simulation_results_analysis(self, benchmarking):
        """Test analyzing simulation results"""
        # Create sample simulation results
        simulation_results = {
            'simulation_summary': {
                'duration': 24.0,
                'ships_arrived': 12,
                'ships_processed': 11,
                'average_waiting_time': 2.5,
                'throughput_rate': 0.46
            },
            'berth_statistics': {
                'total_berths': 5,
                'average_utilization': 78.5
            },
            'container_statistics': {
                'total_operations': 1100,
                'average_processing_time': 1.2
            },
            'performance_metrics': {
                'berth_utilization': 78.5,
                'queue_efficiency': 91.7,
                'processing_efficiency': 85.2
            }
        }
        
        report = benchmarking.analyze_simulation_results(simulation_results)
        
        assert isinstance(report, BenchmarkReport)
        assert report.overall_score > 0
        assert len(report.metrics) > 0
        assert len(report.category_scores) > 0
        assert len(report.recommendations) >= 0
        assert report.historical_comparison is not None
    
    def test_category_scores_calculation(self, benchmarking):
        """Test category scores calculation"""
        # Update some metrics
        benchmarking.update_metric_value("container_throughput_teu_hour", 60.0)
        benchmarking.update_metric_value("berth_utilization_rate", 75.0)
        
        category_scores = benchmarking._calculate_category_scores()
        
        assert BenchmarkCategory.THROUGHPUT in category_scores
        assert BenchmarkCategory.UTILIZATION in category_scores
        assert all(0 <= score <= 100 for score in category_scores.values())
    
    def test_overall_score_calculation(self, benchmarking):
        """Test overall score calculation"""
        category_scores = {
            BenchmarkCategory.THROUGHPUT: 80.0,
            BenchmarkCategory.EFFICIENCY: 75.0,
            BenchmarkCategory.UTILIZATION: 70.0
        }
        
        overall_score = benchmarking._calculate_overall_score(category_scores)
        
        assert 0 <= overall_score <= 100
        assert isinstance(overall_score, float)
    
    def test_recommendations_generation(self, benchmarking):
        """Test recommendations generation"""
        # Set some poor performance metrics
        benchmarking.update_metric_value("container_throughput_teu_hour", 30.0)  # Poor
        benchmarking.update_metric_value("berth_utilization_rate", 50.0)  # Poor
        
        recommendations = benchmarking._generate_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5  # Should limit recommendations
    
    def test_historical_comparison(self, benchmarking):
        """Test historical comparison creation"""
        # Update some metrics
        benchmarking.update_metric_value("container_throughput_teu_hour", 55.0)
        benchmarking.update_metric_value("berth_utilization_rate", 72.0)
        
        comparison = benchmarking._create_historical_comparison()
        
        assert 'trends' in comparison
        assert 'improvements' in comparison
        assert 'deteriorations' in comparison
        assert isinstance(comparison['trends'], dict)
    
    def test_performance_trends(self, benchmarking):
        """Test performance trends analysis"""
        metric_id = "container_throughput_teu_hour"
        
        trends = benchmarking.get_performance_trends(metric_id)
        
        assert 'historical_values' in trends
        assert 'trend_slope' in trends
        assert 'trend_direction' in trends
        assert 'average_value' in trends
        assert trends['trend_direction'] in ['improving', 'declining', 'stable']
    
    def test_industry_comparison(self, benchmarking):
        """Test industry standards comparison"""
        # Update some metrics
        benchmarking.update_metric_value("container_throughput_teu_hour", 65.0)
        benchmarking.update_metric_value("berth_utilization_rate", 80.0)
        
        comparison = benchmarking.compare_with_industry_standards()
        
        assert 'vs_industry_average' in comparison
        assert 'vs_world_class' in comparison
        assert 'performance_gaps' in comparison
        assert 'competitive_position' in comparison
        assert comparison['competitive_position'] in [
            'world_class', 'above_average', 'average', 'below_average'
        ]
    
    def test_report_serialization(self, benchmarking):
        """Test benchmark report serialization"""
        simulation_results = {
            'simulation_summary': {'duration': 24.0, 'average_waiting_time': 2.0},
            'berth_statistics': {'average_utilization': 75.0},
            'container_statistics': {'total_operations': 1000, 'average_processing_time': 1.0},
            'performance_metrics': {'berth_utilization': 75.0}
        }
        
        report = benchmarking.analyze_simulation_results(simulation_results)
        report_dict = report.to_dict()
        
        assert isinstance(report_dict, dict)
        assert 'report_id' in report_dict
        assert 'timestamp' in report_dict
        assert 'overall_score' in report_dict
        assert 'metrics' in report_dict
        
        # Test JSON serialization
        json_str = json.dumps(report_dict)
        assert isinstance(json_str, str)
        
        # Test deserialization
        loaded_dict = json.loads(json_str)
        assert loaded_dict == report_dict

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_benchmark_analysis(self):
        """Test create_benchmark_analysis utility function"""
        simulation_results = {
            'simulation_summary': {'duration': 24.0, 'average_waiting_time': 2.0},
            'berth_statistics': {'average_utilization': 75.0},
            'container_statistics': {'total_operations': 1000, 'average_processing_time': 1.0},
            'performance_metrics': {'berth_utilization': 75.0}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report = create_benchmark_analysis(simulation_results, temp_dir)
            
            assert isinstance(report, BenchmarkReport)
            assert report.overall_score > 0
    
    def test_get_performance_summary(self):
        """Test get_performance_summary utility function"""
        # Create a sample report
        metrics = [
            BenchmarkMetric(
                metric_id="test_metric",
                name="Test Metric",
                category=BenchmarkCategory.THROUGHPUT,
                unit="units",
                description="Test",
                historical_baseline=50.0,
                industry_average=60.0,
                world_class_target=80.0,
                current_value=70.0
            )
        ]
        
        report = BenchmarkReport(
            report_id="test_report",
            timestamp=datetime.now(),
            simulation_duration=24.0,
            metrics=metrics,
            overall_score=75.0,
            category_scores={BenchmarkCategory.THROUGHPUT: 75.0},
            recommendations=["Test recommendation"],
            historical_comparison={}
        )
        
        summary = get_performance_summary(report)
        
        assert 'overall_score' in summary
        assert 'performance_level' in summary
        assert 'category_scores' in summary
        assert 'top_recommendations' in summary
        assert summary['overall_score'] == 75.0
        assert summary['performance_level'] == 'Good'

class TestIntegrationScenarios:
    """Test integration scenarios with different performance levels"""
    
    @pytest.fixture
    def benchmarking(self):
        """Create benchmarking instance for integration tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield PerformanceBenchmarking(temp_dir)
    
    def test_excellent_performance_scenario(self, benchmarking):
        """Test scenario with excellent performance"""
        simulation_results = {
            'simulation_summary': {
                'duration': 24.0,
                'average_waiting_time': 1.0,  # Excellent
                'throughput_rate': 0.8
            },
            'berth_statistics': {'average_utilization': 85.0},  # Excellent
            'container_statistics': {
                'total_operations': 1800,  # High throughput
                'average_processing_time': 0.8
            },
            'performance_metrics': {'berth_utilization': 85.0}
        }
        
        report = benchmarking.analyze_simulation_results(simulation_results)
        
        assert report.overall_score >= 80  # Should be high score
        assert len([m for m in report.metrics 
                   if m.performance_level == PerformanceLevel.EXCELLENT]) > 0
    
    def test_poor_performance_scenario(self, benchmarking):
        """Test scenario with poor performance"""
        simulation_results = {
            'simulation_summary': {
                'duration': 24.0,
                'average_waiting_time': 8.0,  # Poor
                'throughput_rate': 0.2
            },
            'berth_statistics': {'average_utilization': 45.0},  # Poor
            'container_statistics': {
                'total_operations': 500,  # Low throughput
                'average_processing_time': 3.0
            },
            'performance_metrics': {'berth_utilization': 45.0}
        }
        
        report = benchmarking.analyze_simulation_results(simulation_results)
        
        assert report.overall_score <= 60  # Should be low score
        assert len(report.recommendations) > 0  # Should have recommendations
        assert len([m for m in report.metrics 
                   if m.performance_level in [PerformanceLevel.POOR, PerformanceLevel.BELOW_AVERAGE]]) > 0

if __name__ == "__main__":
    # Run basic functionality test
    print("=== Performance Benchmarking Tests ===")
    
    # Test metric creation
    metric = BenchmarkMetric(
        metric_id="test_throughput",
        name="Test Throughput",
        category=BenchmarkCategory.THROUGHPUT,
        unit="TEU/hour",
        description="Test metric for throughput",
        historical_baseline=45.0,
        industry_average=55.0,
        world_class_target=75.0,
        current_value=60.0
    )
    
    performance_level = metric.calculate_performance_level()
    improvement_potential = metric.calculate_improvement_potential()
    
    print(f"Metric: {metric.name}")
    print(f"Current Value: {metric.current_value} {metric.unit}")
    print(f"Performance Level: {performance_level.value}")
    print(f"Improvement Potential: {improvement_potential:.1f}%")
    
    # Test benchmarking system
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmarking = PerformanceBenchmarking(temp_dir)
        
        sample_results = {
            'simulation_summary': {
                'duration': 24.0,
                'average_waiting_time': 2.5
            },
            'berth_statistics': {'average_utilization': 78.5},
            'container_statistics': {
                'total_operations': 1100,
                'average_processing_time': 1.2
            },
            'performance_metrics': {'berth_utilization': 78.5}
        }
        
        report = benchmarking.analyze_simulation_results(sample_results)
        summary = get_performance_summary(report)
        
        print(f"\nBenchmark Analysis:")
        print(f"Overall Score: {summary['overall_score']:.1f}%")
        print(f"Performance Level: {summary['performance_level']}")
        print(f"Recommendations: {len(summary['top_recommendations'])}")
        
        print("\nAll tests completed successfully!")