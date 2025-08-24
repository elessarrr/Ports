"""Performance Benchmarking Framework for Hong Kong Port Digital Twin

This module provides comprehensive performance benchmarking capabilities
for comparing simulation results against historical data and industry standards.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkCategory(Enum):
    """Categories for performance benchmarks"""
    THROUGHPUT = "throughput"
    EFFICIENCY = "efficiency"
    UTILIZATION = "utilization"
    WAITING_TIME = "waiting_time"
    TURNAROUND_TIME = "turnaround_time"
    COST_EFFECTIVENESS = "cost_effectiveness"
    ENVIRONMENTAL = "environmental"
    SAFETY = "safety"

class PerformanceLevel(Enum):
    """Performance level classifications"""
    EXCELLENT = "excellent"  # Top 10% performance
    GOOD = "good"           # Top 25% performance
    AVERAGE = "average"     # 25-75% performance
    BELOW_AVERAGE = "below_average"  # Bottom 25% performance
    POOR = "poor"           # Bottom 10% performance

@dataclass
class BenchmarkMetric:
    """Represents a single benchmark metric"""
    metric_id: str
    name: str
    category: BenchmarkCategory
    unit: str
    description: str
    historical_baseline: float
    industry_average: float
    world_class_target: float
    current_value: Optional[float] = None
    performance_level: Optional[PerformanceLevel] = None
    improvement_potential: Optional[float] = None
    
    def calculate_performance_level(self) -> PerformanceLevel:
        """Calculate performance level based on current value"""
        if self.current_value is None:
            return PerformanceLevel.AVERAGE
        
        # Define thresholds based on world class target and historical baseline
        excellent_threshold = self.world_class_target * 0.95
        good_threshold = self.industry_average * 1.1
        poor_threshold = self.historical_baseline * 0.8
        
        if self.current_value >= excellent_threshold:
            return PerformanceLevel.EXCELLENT
        elif self.current_value >= good_threshold:
            return PerformanceLevel.GOOD
        elif self.current_value >= poor_threshold:
            return PerformanceLevel.AVERAGE
        elif self.current_value >= poor_threshold * 0.8:
            return PerformanceLevel.BELOW_AVERAGE
        else:
            return PerformanceLevel.POOR
    
    def calculate_improvement_potential(self) -> float:
        """Calculate improvement potential as percentage"""
        if self.current_value is None or self.current_value == 0:
            return 0.0
        
        potential = ((self.world_class_target - self.current_value) / self.current_value) * 100
        return max(0.0, potential)

@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    report_id: str
    timestamp: datetime
    simulation_duration: float
    metrics: List[BenchmarkMetric]
    overall_score: float
    category_scores: Dict[BenchmarkCategory, float]
    recommendations: List[str]
    historical_comparison: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization"""
        # Calculate overall performance level based on score
        performance_level = (
            'Excellent' if self.overall_score >= 85 else
            'Good' if self.overall_score >= 70 else
            'Average' if self.overall_score >= 55 else
            'Below Average' if self.overall_score >= 40 else
            'Poor'
        )
        
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp.isoformat(),
            'simulation_duration': self.simulation_duration,
            'metrics': [{
                'metric_id': m.metric_id,
                'name': m.name,
                'category': m.category.value,
                'unit': m.unit,
                'current_value': m.current_value,
                'performance_level': m.performance_level.value if m.performance_level else None,
                'improvement_potential': m.improvement_potential
            } for m in self.metrics],
            'overall_score': self.overall_score,
            'performance_level': performance_level,
            'category_scores': {cat.value: score for cat, score in self.category_scores.items()},
            'recommendations': self.recommendations,
            'historical_comparison': self.historical_comparison
        }

class PerformanceBenchmarking:
    """Main class for performance benchmarking and analysis"""
    
    def __init__(self, benchmarks_dir: Optional[str] = None):
        self.benchmarks_dir = Path(benchmarks_dir) if benchmarks_dir else Path("benchmarks")
        self.benchmarks_dir.mkdir(exist_ok=True)
        
        self.benchmark_metrics: Dict[str, BenchmarkMetric] = {}
        self.historical_data: Dict[str, List[float]] = {}
        self.reports: List[BenchmarkReport] = []
        
        self._initialize_benchmark_metrics()
        self._load_historical_data()
    
    def _initialize_benchmark_metrics(self):
        """Initialize standard benchmark metrics for Hong Kong port"""
        metrics = [
            BenchmarkMetric(
                metric_id="container_throughput_teu_hour",
                name="Container Throughput (TEU/hour)",
                category=BenchmarkCategory.THROUGHPUT,
                unit="TEU/hour",
                description="Number of twenty-foot equivalent units processed per hour",
                historical_baseline=45.0,  # Based on HK port historical data
                industry_average=55.0,
                world_class_target=75.0
            ),
            BenchmarkMetric(
                metric_id="berth_utilization_rate",
                name="Berth Utilization Rate",
                category=BenchmarkCategory.UTILIZATION,
                unit="%",
                description="Percentage of time berths are occupied",
                historical_baseline=65.0,
                industry_average=75.0,
                world_class_target=85.0
            ),
            BenchmarkMetric(
                metric_id="ship_turnaround_time",
                name="Ship Turnaround Time",
                category=BenchmarkCategory.TURNAROUND_TIME,
                unit="hours",
                description="Average time from ship arrival to departure",
                historical_baseline=24.0,
                industry_average=18.0,
                world_class_target=12.0
            ),
            BenchmarkMetric(
                metric_id="queue_waiting_time",
                name="Average Queue Waiting Time",
                category=BenchmarkCategory.WAITING_TIME,
                unit="hours",
                description="Average time ships wait in queue before berth allocation",
                historical_baseline=4.5,
                industry_average=3.0,
                world_class_target=1.5
            ),
            BenchmarkMetric(
                metric_id="crane_productivity",
                name="Crane Productivity",
                category=BenchmarkCategory.EFFICIENCY,
                unit="moves/hour",
                description="Container moves per crane per hour",
                historical_baseline=25.0,
                industry_average=30.0,
                world_class_target=40.0
            ),
            BenchmarkMetric(
                metric_id="yard_utilization",
                name="Container Yard Utilization",
                category=BenchmarkCategory.UTILIZATION,
                unit="%",
                description="Percentage of container yard capacity utilized",
                historical_baseline=70.0,
                industry_average=75.0,
                world_class_target=80.0
            ),
            BenchmarkMetric(
                metric_id="energy_efficiency",
                name="Energy Efficiency",
                category=BenchmarkCategory.ENVIRONMENTAL,
                unit="kWh/TEU",
                description="Energy consumption per container handled",
                historical_baseline=15.0,
                industry_average=12.0,
                world_class_target=8.0
            ),
            BenchmarkMetric(
                metric_id="cost_per_teu",
                name="Cost per TEU",
                category=BenchmarkCategory.COST_EFFECTIVENESS,
                unit="USD/TEU",
                description="Total operational cost per container handled",
                historical_baseline=120.0,
                industry_average=100.0,
                world_class_target=80.0
            )
        ]
        
        for metric in metrics:
            self.benchmark_metrics[metric.metric_id] = metric
        
        logger.info(f"Initialized {len(self.benchmark_metrics)} benchmark metrics")
    
    def _load_historical_data(self):
        """Load historical performance data for trend analysis"""
        historical_file = self.benchmarks_dir / "historical_performance.json"
        if historical_file.exists():
            try:
                with open(historical_file, 'r') as f:
                    self.historical_data = json.load(f)
                logger.info(f"Loaded historical data for {len(self.historical_data)} metrics")
            except Exception as e:
                logger.error(f"Error loading historical data: {e}")
                self._generate_sample_historical_data()
        else:
            self._generate_sample_historical_data()
    
    def _generate_sample_historical_data(self):
        """Generate sample historical data for demonstration"""
        # Generate 12 months of sample data for each metric
        for metric_id, metric in self.benchmark_metrics.items():
            # Generate realistic historical values around the baseline
            baseline = metric.historical_baseline
            variation = baseline * 0.15  # 15% variation
            
            monthly_values = []
            for month in range(12):
                # Add seasonal variation
                seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)
                value = baseline * seasonal_factor + np.random.normal(0, variation)
                monthly_values.append(max(0, value))  # Ensure positive values
            
            self.historical_data[metric_id] = monthly_values
        
        # Save generated data
        self._save_historical_data()
        logger.info("Generated sample historical data")
    
    def _save_historical_data(self):
        """Save historical data to file"""
        historical_file = self.benchmarks_dir / "historical_performance.json"
        try:
            with open(historical_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
    
    def update_metric_value(self, metric_id: str, value: float):
        """Update current value for a benchmark metric"""
        if metric_id in self.benchmark_metrics:
            metric = self.benchmark_metrics[metric_id]
            metric.current_value = value
            metric.performance_level = metric.calculate_performance_level()
            metric.improvement_potential = metric.calculate_improvement_potential()
            logger.debug(f"Updated {metric_id}: {value} {metric.unit}")
        else:
            logger.warning(f"Unknown metric ID: {metric_id}")
    
    def analyze_simulation_results(self, simulation_results: Dict[str, Any]) -> BenchmarkReport:
        """Analyze simulation results against benchmarks"""
        logger.info("Analyzing simulation results against benchmarks")
        
        # Extract metrics from simulation results
        self._extract_metrics_from_simulation(simulation_results)
        
        # Calculate scores
        category_scores = self._calculate_category_scores()
        overall_score = self._calculate_overall_score(category_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Create historical comparison
        historical_comparison = self._create_historical_comparison()
        
        # Create report
        report = BenchmarkReport(
            report_id=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            simulation_duration=simulation_results.get('simulation_summary', {}).get('duration', 0),
            metrics=list(self.benchmark_metrics.values()),
            overall_score=overall_score,
            category_scores=category_scores,
            recommendations=recommendations,
            historical_comparison=historical_comparison
        )
        
        self.reports.append(report)
        self._save_report(report)
        
        logger.info(f"Benchmark analysis completed. Overall score: {overall_score:.1f}%")
        return report
    
    def _extract_metrics_from_simulation(self, results: Dict[str, Any]):
        """Extract benchmark metrics from simulation results"""
        sim_summary = results.get('simulation_summary', {})
        berth_stats = results.get('berth_statistics', {})
        container_stats = results.get('container_statistics', {})
        performance_metrics = results.get('performance_metrics', {})
        
        # Container throughput (TEU/hour)
        throughput = 0.0
        if sim_summary.get('duration', 0) > 0:
            containers_processed = container_stats.get('total_operations', 0)
            throughput = containers_processed / sim_summary['duration']
            self.update_metric_value('container_throughput_teu_hour', throughput)
        
        # Berth utilization
        berth_utilization = performance_metrics.get('berth_utilization', 0)
        self.update_metric_value('berth_utilization_rate', berth_utilization)
        
        # Average waiting time
        avg_waiting = sim_summary.get('average_waiting_time', 0)
        self.update_metric_value('queue_waiting_time', avg_waiting)
        
        # Crane productivity (estimated from container stats)
        if container_stats.get('average_processing_time', 0) > 0:
            crane_productivity = 1.0 / container_stats['average_processing_time'] * 60  # moves per hour
            self.update_metric_value('crane_productivity', crane_productivity)
        
        # Estimate other metrics based on available data
        # Ship turnaround time (estimated)
        estimated_turnaround = avg_waiting + container_stats.get('average_processing_time', 0) * 2
        self.update_metric_value('ship_turnaround_time', estimated_turnaround)
        
        # Yard utilization (estimated based on throughput)
        yard_util = min(85.0, throughput * 1.5)  # Simplified estimation
        self.update_metric_value('yard_utilization', yard_util)
        
        # Energy efficiency (estimated)
        energy_per_teu = 15.0 - (berth_utilization - 65) * 0.1  # Better utilization = better efficiency
        self.update_metric_value('energy_efficiency', max(8.0, energy_per_teu))
        
        # Cost per TEU (estimated)
        cost_per_teu = 120.0 - (berth_utilization - 65) * 0.5  # Better utilization = lower cost
        self.update_metric_value('cost_per_teu', max(80.0, cost_per_teu))
    
    def _calculate_category_scores(self) -> Dict[BenchmarkCategory, float]:
        """Calculate performance scores by category"""
        category_scores = {}
        
        for category in BenchmarkCategory:
            category_metrics = [m for m in self.benchmark_metrics.values() if m.category == category]
            if not category_metrics:
                continue
            
            scores = []
            for metric in category_metrics:
                if metric.current_value is not None:
                    # Calculate score as percentage of world class target
                    if metric.world_class_target > 0:
                        score = min(100, (metric.current_value / metric.world_class_target) * 100)
                    else:
                        score = 50  # Default score if no target
                    scores.append(score)
            
            if scores:
                category_scores[category] = sum(scores) / len(scores)
            else:
                category_scores[category] = 0.0
        
        return category_scores
    
    def _calculate_overall_score(self, category_scores: Dict[BenchmarkCategory, float]) -> float:
        """Calculate overall performance score"""
        if not category_scores:
            return 0.0
        
        # Weight categories by importance
        weights = {
            BenchmarkCategory.THROUGHPUT: 0.25,
            BenchmarkCategory.EFFICIENCY: 0.20,
            BenchmarkCategory.UTILIZATION: 0.20,
            BenchmarkCategory.WAITING_TIME: 0.15,
            BenchmarkCategory.TURNAROUND_TIME: 0.10,
            BenchmarkCategory.COST_EFFECTIVENESS: 0.05,
            BenchmarkCategory.ENVIRONMENTAL: 0.03,
            BenchmarkCategory.SAFETY: 0.02
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = weights.get(category, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        for metric in self.benchmark_metrics.values():
            if metric.current_value is None or metric.performance_level is None:
                continue
            
            if metric.performance_level in [PerformanceLevel.BELOW_AVERAGE, PerformanceLevel.POOR]:
                if metric.category == BenchmarkCategory.THROUGHPUT:
                    recommendations.append(f"Improve {metric.name}: Consider optimizing crane operations and reducing container handling time")
                elif metric.category == BenchmarkCategory.UTILIZATION:
                    recommendations.append(f"Increase {metric.name}: Optimize berth allocation and reduce idle time")
                elif metric.category == BenchmarkCategory.WAITING_TIME:
                    recommendations.append(f"Reduce {metric.name}: Implement predictive scheduling and improve traffic flow")
                elif metric.category == BenchmarkCategory.EFFICIENCY:
                    recommendations.append(f"Enhance {metric.name}: Invest in automation and staff training")
        
        # Add general recommendations
        if len([m for m in self.benchmark_metrics.values() 
               if m.performance_level in [PerformanceLevel.BELOW_AVERAGE, PerformanceLevel.POOR]]) > 3:
            recommendations.append("Consider comprehensive operational review and process optimization")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _create_historical_comparison(self) -> Dict[str, Any]:
        """Create comparison with historical performance"""
        comparison = {
            'trends': {},
            'improvements': {},
            'deteriorations': {}
        }
        
        for metric_id, metric in self.benchmark_metrics.items():
            if metric.current_value is None or metric_id not in self.historical_data:
                continue
            
            historical_values = self.historical_data[metric_id]
            if not historical_values:
                continue
            
            avg_historical = sum(historical_values) / len(historical_values)
            current_vs_historical = ((metric.current_value - avg_historical) / avg_historical) * 100
            
            comparison['trends'][metric_id] = {
                'current_value': metric.current_value,
                'historical_average': avg_historical,
                'change_percentage': current_vs_historical,
                'trend': 'improving' if current_vs_historical > 5 else 'declining' if current_vs_historical < -5 else 'stable'
            }
            
            if current_vs_historical > 10:
                comparison['improvements'][metric_id] = current_vs_historical
            elif current_vs_historical < -10:
                comparison['deteriorations'][metric_id] = current_vs_historical
        
        return comparison
    
    def _save_report(self, report: BenchmarkReport):
        """Save benchmark report to file"""
        report_file = self.benchmarks_dir / f"{report.report_id}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Saved benchmark report: {report.report_id}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def get_performance_trends(self, metric_id: str, months: int = 12) -> Dict[str, Any]:
        """Get performance trends for a specific metric"""
        if metric_id not in self.historical_data:
            return {}
        
        historical_values = self.historical_data[metric_id][-months:]
        if len(historical_values) < 2:
            return {}
        
        # Calculate trend
        x = list(range(len(historical_values)))
        y = historical_values
        
        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        return {
            'historical_values': historical_values,
            'trend_slope': slope,
            'trend_direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
            'average_value': sum(historical_values) / len(historical_values),
            'min_value': min(historical_values),
            'max_value': max(historical_values),
            'volatility': np.std(historical_values) if len(historical_values) > 1 else 0
        }
    
    def compare_with_industry_standards(self) -> Dict[str, Any]:
        """Compare current performance with industry standards"""
        comparison = {
            'vs_industry_average': {},
            'vs_world_class': {},
            'performance_gaps': {},
            'competitive_position': 'unknown'
        }
        
        scores = []
        for metric in self.benchmark_metrics.values():
            if metric.current_value is None:
                continue
            
            # Compare with industry average
            vs_industry = ((metric.current_value - metric.industry_average) / metric.industry_average) * 100
            comparison['vs_industry_average'][metric.metric_id] = vs_industry
            
            # Compare with world class
            vs_world_class = ((metric.current_value - metric.world_class_target) / metric.world_class_target) * 100
            comparison['vs_world_class'][metric.metric_id] = vs_world_class
            
            # Calculate performance gap
            gap = metric.world_class_target - metric.current_value
            comparison['performance_gaps'][metric.metric_id] = gap
            
            # Score for competitive position
            if metric.current_value >= metric.world_class_target * 0.95:
                scores.append(4)  # Excellent
            elif metric.current_value >= metric.industry_average:
                scores.append(3)  # Above average
            elif metric.current_value >= metric.historical_baseline:
                scores.append(2)  # Average
            else:
                scores.append(1)  # Below average
        
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score >= 3.5:
                comparison['competitive_position'] = 'world_class'
            elif avg_score >= 2.5:
                comparison['competitive_position'] = 'above_average'
            elif avg_score >= 1.5:
                comparison['competitive_position'] = 'average'
            else:
                comparison['competitive_position'] = 'below_average'
        
        return comparison

# Utility functions for easy integration

def create_benchmark_analysis(simulation_results: Dict[str, Any], 
                            benchmarks_dir: Optional[str] = None) -> BenchmarkReport:
    """Create a comprehensive benchmark analysis from simulation results"""
    benchmarking = PerformanceBenchmarking(benchmarks_dir)
    return benchmarking.analyze_simulation_results(simulation_results)

def get_performance_summary(benchmark_report: BenchmarkReport) -> Dict[str, Any]:
    """Get a summary of performance metrics for dashboard display"""
    return {
        'overall_score': benchmark_report.overall_score,
        'performance_level': (
            'Excellent' if benchmark_report.overall_score >= 85 else
            'Good' if benchmark_report.overall_score >= 70 else
            'Average' if benchmark_report.overall_score >= 55 else
            'Below Average' if benchmark_report.overall_score >= 40 else
            'Poor'
        ),
        'category_scores': benchmark_report.category_scores,
        'top_recommendations': benchmark_report.recommendations[:3],
        'metrics_count': len(benchmark_report.metrics),
        'timestamp': benchmark_report.timestamp.isoformat()
    }

if __name__ == "__main__":
    # Example usage
    print("=== Performance Benchmarking Framework Test ===")
    
    # Create sample simulation results
    sample_results = {
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
    
    # Create benchmark analysis
    benchmarking = PerformanceBenchmarking()
    report = benchmarking.analyze_simulation_results(sample_results)
    
    print(f"\nBenchmark Analysis Results:")
    print(f"Overall Score: {report.overall_score:.1f}%")
    print(f"\nCategory Scores:")
    for category, score in report.category_scores.items():
        print(f"  {category.value}: {score:.1f}%")
    
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(report.recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nReport saved as: {report.report_id}")