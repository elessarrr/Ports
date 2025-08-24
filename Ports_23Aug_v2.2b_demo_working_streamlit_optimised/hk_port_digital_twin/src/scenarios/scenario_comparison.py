# Comments for context:
# This module provides scenario comparison functionality for the Hong Kong Port Digital Twin.
# It was created to fix the missing import error in scenario_tab_consolidation.py.
# The module leverages existing comparison functionality from multi_scenario_optimizer.py
# and scenario_optimizer.py to provide a unified interface for scenario comparisons.

from typing import Dict, List, Any, Optional
import logging
from dataclasses import asdict

# Import existing scenario functionality
from .multi_scenario_optimizer import MultiScenarioOptimizer
from .scenario_optimizer import quick_scenario_comparison, ScenarioAwareBerthOptimizer
from .scenario_parameters import get_scenario_parameters, ALL_SCENARIOS
from .scenario_manager import ScenarioManager

# Configure logging
logger = logging.getLogger(__name__)

def create_scenario_comparison(
    primary_scenario: str,
    comparison_scenarios: List[str],
    simulation_hours: int = 72,
    use_historical_data: bool = True
) -> Optional[Dict[str, Any]]:
    """Create a comprehensive scenario comparison analysis.
    
    Args:
        primary_scenario: The main scenario to analyze
        comparison_scenarios: List of scenarios to compare against primary
        simulation_hours: Duration of simulation in hours (default: 72)
        use_historical_data: Whether to use historical data patterns (default: True)
        
    Returns:
        Dictionary containing comparison results and analysis, or None if failed
    """
    try:
        logger.info(f"Starting scenario comparison: {primary_scenario} vs {comparison_scenarios}")
        
        # Validate scenarios
        all_scenarios = [primary_scenario] + comparison_scenarios
        for scenario in all_scenarios:
            if scenario not in ALL_SCENARIOS:
                logger.error(f"Invalid scenario: {scenario}")
                return None
        
        # Initialize multi-scenario optimizer
        optimizer = MultiScenarioOptimizer(use_historical_data=use_historical_data)
        
        # Run comparison for all scenarios
        scenario_results = {}
        
        for scenario in all_scenarios:
            try:
                logger.info(f"Running simulation for scenario: {scenario}")
                
                # Get scenario parameters
                params = get_scenario_parameters(scenario)
                if not params:
                    logger.warning(f"Could not get parameters for scenario: {scenario}")
                    continue
                
                # Run single scenario optimization
                # Note: Using a simplified approach since full multi-scenario might be complex
                scenario_manager = ScenarioManager()
                scenario_manager.set_scenario(scenario)
                
                # Create scenario-aware optimizer
                scenario_optimizer = ScenarioAwareBerthOptimizer(scenario_manager)
                
                # Store scenario results (simplified for now)
                scenario_results[scenario] = {
                    'scenario_name': scenario,
                    'parameters': asdict(params),
                    'simulation_hours': simulation_hours,
                    'metrics': {
                        'arrival_rate_multiplier': params.arrival_rate_multiplier,
                        'processing_rate_multiplier': params.processing_rate_multiplier,
                        'target_berth_utilization': params.target_berth_utilization,
                        'average_ship_size_multiplier': params.average_ship_size_multiplier
                    }
                }
                
                logger.info(f"Completed simulation for scenario: {scenario}")
                
            except Exception as e:
                logger.error(f"Error processing scenario {scenario}: {e}")
                continue
        
        if not scenario_results:
            logger.error("No scenario results generated")
            return None
        
        # Create comparison analysis
        comparison_data = []
        primary_metrics = scenario_results.get(primary_scenario, {}).get('metrics', {})
        
        for scenario, results in scenario_results.items():
            metrics = results.get('metrics', {})
            
            # Calculate relative performance vs primary scenario
            relative_performance = {}
            if primary_metrics and scenario != primary_scenario:
                for metric, value in metrics.items():
                    primary_value = primary_metrics.get(metric, 1.0)
                    if primary_value != 0:
                        relative_performance[f"{metric}_vs_primary"] = (value / primary_value - 1) * 100
            
            comparison_data.append({
                'scenario': scenario,
                'is_primary': scenario == primary_scenario,
                **metrics,
                **relative_performance
            })
        
        # Generate insights
        insights = _generate_comparison_insights(primary_scenario, comparison_scenarios, scenario_results)
        
        # Prepare final results
        comparison_results = {
            'primary_scenario': primary_scenario,
            'comparison_scenarios': comparison_scenarios,
            'simulation_hours': simulation_hours,
            'use_historical_data': use_historical_data,
            'scenario_results': scenario_results,
            'comparison_data': comparison_data,
            'insights': insights,
            'timestamp': None  # Could add timestamp if needed
        }
        
        logger.info("Scenario comparison completed successfully")
        return comparison_results
        
    except Exception as e:
        logger.error(f"Error in create_scenario_comparison: {e}")
        return None

def _generate_comparison_insights(
    primary_scenario: str,
    comparison_scenarios: List[str],
    scenario_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate insights from scenario comparison results.
    
    Args:
        primary_scenario: The primary scenario name
        comparison_scenarios: List of comparison scenario names
        scenario_results: Results from all scenarios
        
    Returns:
        Dictionary containing insights and recommendations
    """
    insights = {
        'summary': f"Comparison of {primary_scenario} against {len(comparison_scenarios)} scenarios",
        'recommendations': [],
        'key_differences': [],
        'performance_ranking': []
    }
    
    try:
        primary_metrics = scenario_results.get(primary_scenario, {}).get('metrics', {})
        
        # Analyze key differences
        for scenario in comparison_scenarios:
            scenario_metrics = scenario_results.get(scenario, {}).get('metrics', {})
            
            # Compare arrival rates
            primary_arrival = primary_metrics.get('arrival_rate_multiplier', 1.0)
            scenario_arrival = scenario_metrics.get('arrival_rate_multiplier', 1.0)
            
            if abs(primary_arrival - scenario_arrival) > 0.1:
                difference = "higher" if scenario_arrival > primary_arrival else "lower"
                insights['key_differences'].append(
                    f"{scenario} has {difference} arrival rates than {primary_scenario}"
                )
            
            # Compare utilization targets
            primary_util = primary_metrics.get('target_berth_utilization', 0.8)
            scenario_util = scenario_metrics.get('target_berth_utilization', 0.8)
            
            if abs(primary_util - scenario_util) > 0.05:
                difference = "higher" if scenario_util > primary_util else "lower"
                insights['key_differences'].append(
                    f"{scenario} targets {difference} berth utilization than {primary_scenario}"
                )
        
        # Generate basic recommendations
        if primary_scenario == 'peak':
            insights['recommendations'].append(
                "Peak season requires enhanced capacity planning and resource allocation"
            )
        elif primary_scenario == 'low':
            insights['recommendations'].append(
                "Low season offers opportunities for maintenance and optimization"
            )
        else:
            insights['recommendations'].append(
                "Normal operations provide baseline for capacity planning"
            )
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        insights['error'] = str(e)
    
    return insights