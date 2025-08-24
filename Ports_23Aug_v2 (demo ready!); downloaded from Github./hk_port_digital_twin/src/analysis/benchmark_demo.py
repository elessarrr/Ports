"""Performance Benchmarking Demo for Hong Kong Port Digital Twin

This module demonstrates the integrated performance benchmarking system
by running simulations and analyzing performance against industry standards.
"""

import sys
import os
from typing import Dict, List
import json
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.port_simulation import PortSimulation
from src.analysis.performance_benchmarking import PerformanceBenchmarking, create_benchmark_analysis
from config.settings import get_enhanced_simulation_config
from src.utils.data_loader import load_berth_configurations


def run_benchmark_demo(duration: float = 24.0) -> Dict:
    """Run a demonstration of the performance benchmarking system
    
    Args:
        duration: Simulation duration in hours
        
    Returns:
        Dictionary containing benchmark analysis results
    """
    print("=== Hong Kong Port Performance Benchmarking Demo ===")
    print(f"Running simulation for {duration} hours...\n")
    
    # Initialize simulation with enhanced config
    config = get_enhanced_simulation_config()
    
    # Load berth configurations from CSV file
    berth_configs = load_berth_configurations()
    config['berths'] = berth_configs
    
    print(f"Loaded {len(berth_configs)} berths from configuration file")
    
    simulation = PortSimulation(config)
    
    # Run simulation
    results = simulation.run_simulation(duration)
    
    # Extract benchmark analysis
    benchmark_analysis = results.get('benchmark_analysis', {})
    
    print("\n=== PERFORMANCE BENCHMARK RESULTS ===")
    print_benchmark_summary(benchmark_analysis)
    
    return benchmark_analysis


def print_benchmark_summary(analysis: Dict):
    """Print a formatted summary of benchmark analysis
    
    Args:
        analysis: Benchmark analysis dictionary
    """
    if not analysis:
        print("No benchmark analysis available")
        return
    
    # Print overall score
    overall_score = analysis.get('overall_score', 0)
    performance_level = analysis.get('performance_level', 'Unknown')
    print(f"Overall Performance Score: {overall_score:.1f}/100 ({performance_level})")
    print("=" * 60)
    
    # Print metric details
    metrics = analysis.get('metrics', [])
    for metric_data in metrics:
        metric_name = metric_data.get('name', metric_data.get('metric_id', 'Unknown'))
        current_value = metric_data.get('current_value', 0)
        level = metric_data.get('performance_level', 'Unknown')
        unit = metric_data.get('unit', '')
        improvement_potential = metric_data.get('improvement_potential', 0)
        
        print(f"\n{metric_name}:")
        if current_value is not None:
            print(f"  Current Value: {current_value:.2f} {unit}")
        else:
            print(f"  Current Value: N/A {unit}")
        print(f"  Performance Level: {level}")
        if improvement_potential is not None:
            print(f"  Improvement Potential: {improvement_potential:.1f}%")
        
        # Print category
        category = metric_data.get('category', 'Unknown')
        if category:
            print(f"  Category: {category}")
    
    # Print recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        print("\n=== IMPROVEMENT RECOMMENDATIONS ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Overall Performance Level: {analysis.get('performance_level', 'Unknown')}")
    print(f"Overall Score: {analysis.get('overall_score', 0):.1f}/100")


def compare_scenarios_benchmark():
    """Compare performance benchmarks across different scenarios"""
    print("\n=== SCENARIO BENCHMARK COMPARISON ===")
    
    base_config = get_enhanced_simulation_config()
    
    # Create reduced capacity config with proper berth structure
    reduced_config = base_config.copy()
    # Take only first 6 berths (reduced capacity)
    if 'berths' in reduced_config:
        reduced_config['berths'] = reduced_config['berths'][:6]
    
    scenarios = [
        {'name': 'Normal Operations', 'config': base_config},
        {'name': 'High Traffic', 'config': {**base_config, 'ship_arrival_rate': 0.3}},
        {'name': 'Reduced Capacity', 'config': reduced_config}
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\nRunning {scenario['name']} scenario...")
        simulation = PortSimulation(scenario['config'])
        result = simulation.run_simulation(12.0)  # 12 hour simulation
        
        benchmark_analysis = result.get('benchmark_analysis', {})
        overall_score = benchmark_analysis.get('overall_score', 0)
        performance_level = benchmark_analysis.get('performance_level', 'Unknown')
        
        results.append({
            'scenario': scenario['name'],
            'score': overall_score,
            'level': performance_level,
            'analysis': benchmark_analysis
        })
        
        print(f"  Score: {overall_score:.1f}/100 ({performance_level})")
    
    # Print comparison summary
    print("\n=== SCENARIO COMPARISON SUMMARY ===")
    results.sort(key=lambda x: x['score'], reverse=True)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['scenario']}: {result['score']:.1f}/100 ({result['level']})")
    
    return results


def export_benchmark_report(analysis: Dict, filename: str = None):
    """Export benchmark analysis to JSON file
    
    Args:
        analysis: Benchmark analysis dictionary
        filename: Output filename (optional)
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_report_{timestamp}.json"
    
    # Add metadata
    report = {
        'generated_at': datetime.now().isoformat(),
        'report_type': 'performance_benchmark',
        'version': '1.0',
        'analysis': analysis
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nBenchmark report exported to: {filename}")


if __name__ == "__main__":
    # Run main demo
    benchmark_results = run_benchmark_demo(24.0)
    
    # Export results
    export_benchmark_report(benchmark_results)
    
    # Run scenario comparison
    scenario_results = compare_scenarios_benchmark()
    
    print("\n=== DEMO COMPLETED ===")
    print("Performance benchmarking system successfully integrated!")