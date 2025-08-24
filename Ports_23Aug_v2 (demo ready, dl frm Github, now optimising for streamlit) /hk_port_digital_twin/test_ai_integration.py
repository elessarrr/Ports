#!/usr/bin/env python3
"""
Test script to demonstrate AI Integration in Hong Kong Port Digital Twin

This script tests the newly integrated AI optimization layer with the port simulation.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.core.port_simulation import PortSimulation
from config.settings import PORT_CONFIG, BERTH_CONFIGS, SIMULATION_CONFIG

def test_ai_integration():
    """Test AI integration with port simulation"""
    print("=== Hong Kong Port Digital Twin - AI Integration Test ===")
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Create simulation with AI optimization enabled
    config = {
        'port': PORT_CONFIG,
        'berths': BERTH_CONFIGS,
        'simulation': SIMULATION_CONFIG,
        'ai_optimization': True,
        'optimization_interval': 0.5  # Optimize every 30 minutes
    }
    simulation = PortSimulation(config)
    
    print("✓ Port simulation initialized with AI optimization enabled")
    print(f"  - AI Optimization: {simulation.ai_optimization_enabled}")
    print(f"  - Optimization Interval: {simulation.optimization_interval} hours")
    print(f"  - Berth Optimizer: {type(simulation.berth_optimizer).__name__}")
    print(f"  - Resource Optimizer: {type(simulation.resource_optimizer).__name__}")
    print(f"  - Decision Support: {type(simulation.decision_engine).__name__}")
    print()
    
    # Run simulation with AI optimization
    print("Running simulation with AI optimization...")
    result = simulation.run_simulation(duration=2.0)  # 2 hours simulation
    
    print("\n=== Simulation Results ===")
    print(f"Duration: {result['simulation_summary']['duration']:.2f} hours")
    print(f"Ships arrived: {result['simulation_summary']['ships_arrived']}")
    print(f"Ships processed: {result['simulation_summary']['ships_processed']}")
    print(f"Average waiting time: {result['simulation_summary']['average_waiting_time']:.2f} hours")
    print(f"Throughput rate: {result['simulation_summary']['throughput_rate']:.2f} ships/hour")
    
    # Check AI optimization metrics
    ai_optimizations = simulation.metrics.get('ai_optimizations_performed', 0)
    time_saved = simulation.metrics.get('optimization_time_saved', 0)
    
    print("\n=== AI Optimization Metrics ===")
    print(f"AI optimizations performed: {ai_optimizations}")
    print(f"Optimization time saved: {time_saved:.2f} hours")
    
    # Performance metrics
    print("\n=== Performance Metrics ===")
    print(f"Berth utilization: {result['performance_metrics']['berth_utilization']:.1f}%")
    print(f"Queue efficiency: {result['performance_metrics']['queue_efficiency']:.1f}%")
    print(f"Processing efficiency: {result['performance_metrics']['processing_efficiency']:.1f}%")
    
    # Container processing statistics
    container_stats = result['container_statistics']
    print("\n=== Container Processing ===")
    print(f"Total operations: {container_stats['total_operations']}")
    print(f"Total containers processed: {container_stats['total_containers_processed']}")
    print(f"Average processing time: {container_stats['average_processing_time']:.2f} hours")
    print(f"Average crane utilization: {container_stats['average_crane_utilization']:.1f} cranes")
    
    # Test traditional simulation for comparison
    print("\n" + "="*60)
    print("Running comparison simulation without AI optimization...")
    
    config_no_ai = {
        'port': PORT_CONFIG,
        'berths': BERTH_CONFIGS,
        'simulation': SIMULATION_CONFIG,
        'ai_optimization': False
    }
    traditional_simulation = PortSimulation(config_no_ai)
    traditional_result = traditional_simulation.run_simulation(duration=2.0)
    
    print("\n=== Traditional Simulation Results ===")
    print(f"Duration: {traditional_result['simulation_summary']['duration']:.2f} hours")
    print(f"Ships arrived: {traditional_result['simulation_summary']['ships_arrived']}")
    print(f"Ships processed: {traditional_result['simulation_summary']['ships_processed']}")
    print(f"Average waiting time: {traditional_result['simulation_summary']['average_waiting_time']:.2f} hours")
    print(f"Throughput rate: {traditional_result['simulation_summary']['throughput_rate']:.2f} ships/hour")
    
    # Compare results
    print("\n=== AI vs Traditional Comparison ===")
    ai_waiting = result['simulation_summary']['average_waiting_time']
    traditional_waiting = traditional_result['simulation_summary']['average_waiting_time']
    
    ai_throughput = result['simulation_summary']['throughput_rate']
    traditional_throughput = traditional_result['simulation_summary']['throughput_rate']
    
    if ai_waiting < traditional_waiting:
        improvement = ((traditional_waiting - ai_waiting) / traditional_waiting) * 100
        print(f"✓ AI optimization reduced waiting time by {improvement:.1f}%")
    else:
        print(f"⚠ AI waiting time: {ai_waiting:.2f}h vs Traditional: {traditional_waiting:.2f}h")
    
    if ai_throughput > traditional_throughput:
        improvement = ((ai_throughput - traditional_throughput) / traditional_throughput) * 100
        print(f"✓ AI optimization improved throughput by {improvement:.1f}%")
    else:
        print(f"⚠ AI throughput: {ai_throughput:.2f} vs Traditional: {traditional_throughput:.2f}")
    
    print("\n=== Test Completed Successfully ===")
    print(f"Test finished at: {datetime.now()}")
    
    return {
        'ai_result': result,
        'traditional_result': traditional_result,
        'ai_optimizations': ai_optimizations,
        'time_saved': time_saved
    }

if __name__ == "__main__":
    try:
        test_results = test_ai_integration()
        print("\n🎉 AI Integration test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)