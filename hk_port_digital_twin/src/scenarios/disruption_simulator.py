#!/usr/bin/env python3
"""
Disruption Impact Simulation Module

This module implements predictive disruption impact simulation for the Hong Kong Port Digital Twin.
It models various disruption events and their cascading effects on port operations.

Author: AI Assistant
Date: 2024
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DisruptionType(Enum):
    """Types of disruption events"""
    WEATHER = "weather"
    EQUIPMENT_FAILURE = "equipment_failure"
    CONGESTION = "congestion"
    LABOR_SHORTAGE = "labor_shortage"
    CYBER_SECURITY = "cyber_security"
    SUPPLY_CHAIN = "supply_chain"

class DisruptionSeverity(Enum):
    """Severity levels for disruption events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DisruptionEvent:
    """Represents a disruption event with its characteristics"""
    event_id: str
    disruption_type: DisruptionType
    severity: DisruptionSeverity
    start_time: datetime
    duration_hours: float
    affected_berths: List[str] = field(default_factory=list)
    capacity_reduction: float = 0.0  # Percentage reduction (0.0 to 1.0)
    processing_time_increase: float = 0.0  # Percentage increase (0.0 to 1.0)
    description: str = ""
    recovery_strategies: List[str] = field(default_factory=list)
    
    @property
    def end_time(self) -> datetime:
        """Calculate end time of disruption"""
        return self.start_time + timedelta(hours=self.duration_hours)
    
    def is_active(self, current_time: datetime) -> bool:
        """Check if disruption is currently active"""
        return self.start_time <= current_time <= self.end_time

@dataclass
class RecoveryStrategy:
    """Represents a recovery strategy for disruption events"""
    strategy_id: str
    name: str
    description: str
    effectiveness: float  # 0.0 to 1.0
    implementation_time: float  # Hours to implement
    cost: float  # Implementation cost
    applicable_disruptions: List[DisruptionType]

class DisruptionSimulator:
    """Main class for simulating disruption impacts on port operations"""
    
    def __init__(self):
        self.active_disruptions: List[DisruptionEvent] = []
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.disruption_history: List[DisruptionEvent] = []
        self._initialize_recovery_strategies()
    
    def _initialize_recovery_strategies(self):
        """Initialize predefined recovery strategies"""
        strategies = [
            RecoveryStrategy(
                strategy_id="berth_reallocation",
                name="Berth Reallocation",
                description="Reallocate vessels to available berths",
                effectiveness=0.7,
                implementation_time=2.0,
                cost=5000.0,
                applicable_disruptions=[DisruptionType.EQUIPMENT_FAILURE, DisruptionType.CONGESTION]
            ),
            RecoveryStrategy(
                strategy_id="priority_queuing",
                name="Priority Queuing",
                description="Implement priority queuing for critical vessels",
                effectiveness=0.6,
                implementation_time=1.0,
                cost=2000.0,
                applicable_disruptions=[DisruptionType.CONGESTION, DisruptionType.LABOR_SHORTAGE]
            ),
            RecoveryStrategy(
                strategy_id="emergency_crew",
                name="Emergency Crew Deployment",
                description="Deploy emergency crew to maintain operations",
                effectiveness=0.8,
                implementation_time=3.0,
                cost=15000.0,
                applicable_disruptions=[DisruptionType.LABOR_SHORTAGE, DisruptionType.EQUIPMENT_FAILURE]
            ),
            RecoveryStrategy(
                strategy_id="weather_protocol",
                name="Severe Weather Protocol",
                description="Implement severe weather operational procedures",
                effectiveness=0.5,
                implementation_time=0.5,
                cost=1000.0,
                applicable_disruptions=[DisruptionType.WEATHER]
            )
        ]
        
        for strategy in strategies:
            self.recovery_strategies[strategy.strategy_id] = strategy
    
    def create_disruption_event(
        self,
        disruption_type: DisruptionType,
        severity: DisruptionSeverity,
        start_time: datetime,
        duration_hours: float,
        affected_berths: Optional[List[str]] = None
    ) -> DisruptionEvent:
        """Create a new disruption event"""
        
        event_id = f"{disruption_type.value}_{severity.value}_{start_time.strftime('%Y%m%d_%H%M')}"
        
        # Calculate impact based on disruption type and severity
        capacity_reduction, processing_increase = self._calculate_disruption_impact(
            disruption_type, severity
        )
        
        # Generate description
        description = self._generate_disruption_description(disruption_type, severity)
        
        # Select applicable recovery strategies
        recovery_strategies = [
            strategy.strategy_id for strategy in self.recovery_strategies.values()
            if disruption_type in strategy.applicable_disruptions
        ]
        
        event = DisruptionEvent(
            event_id=event_id,
            disruption_type=disruption_type,
            severity=severity,
            start_time=start_time,
            duration_hours=duration_hours,
            affected_berths=affected_berths or [],
            capacity_reduction=capacity_reduction,
            processing_time_increase=processing_increase,
            description=description,
            recovery_strategies=recovery_strategies
        )
        
        return event
    
    def _calculate_disruption_impact(
        self, 
        disruption_type: DisruptionType, 
        severity: DisruptionSeverity
    ) -> Tuple[float, float]:
        """Calculate capacity reduction and processing time increase"""
        
        # Base impact factors by disruption type
        base_impacts = {
            DisruptionType.WEATHER: (0.3, 0.2),
            DisruptionType.EQUIPMENT_FAILURE: (0.5, 0.4),
            DisruptionType.CONGESTION: (0.2, 0.3),
            DisruptionType.LABOR_SHORTAGE: (0.4, 0.5),
            DisruptionType.CYBER_SECURITY: (0.6, 0.3),
            DisruptionType.SUPPLY_CHAIN: (0.3, 0.4)
        }
        
        # Severity multipliers
        severity_multipliers = {
            DisruptionSeverity.LOW: 0.5,
            DisruptionSeverity.MEDIUM: 1.0,
            DisruptionSeverity.HIGH: 1.5,
            DisruptionSeverity.CRITICAL: 2.0
        }
        
        base_capacity, base_processing = base_impacts[disruption_type]
        multiplier = severity_multipliers[severity]
        
        capacity_reduction = min(0.9, base_capacity * multiplier)  # Cap at 90%
        processing_increase = min(2.0, base_processing * multiplier)  # Cap at 200%
        
        return capacity_reduction, processing_increase
    
    def _generate_disruption_description(
        self, 
        disruption_type: DisruptionType, 
        severity: DisruptionSeverity
    ) -> str:
        """Generate human-readable description for disruption"""
        
        descriptions = {
            DisruptionType.WEATHER: {
                DisruptionSeverity.LOW: "Light rain affecting outdoor operations",
                DisruptionSeverity.MEDIUM: "Moderate storm with wind and rain",
                DisruptionSeverity.HIGH: "Severe weather with strong winds",
                DisruptionSeverity.CRITICAL: "Typhoon conditions - emergency protocols activated"
            },
            DisruptionType.EQUIPMENT_FAILURE: {
                DisruptionSeverity.LOW: "Minor equipment malfunction",
                DisruptionSeverity.MEDIUM: "Crane operational issues",
                DisruptionSeverity.HIGH: "Major equipment failure affecting multiple berths",
                DisruptionSeverity.CRITICAL: "Critical infrastructure failure"
            },
            DisruptionType.CONGESTION: {
                DisruptionSeverity.LOW: "Slight increase in vessel traffic",
                DisruptionSeverity.MEDIUM: "Moderate port congestion",
                DisruptionSeverity.HIGH: "Heavy congestion with vessel queues",
                DisruptionSeverity.CRITICAL: "Severe congestion - port at capacity"
            },
            DisruptionType.LABOR_SHORTAGE: {
                DisruptionSeverity.LOW: "Minor staffing shortage",
                DisruptionSeverity.MEDIUM: "Reduced crew availability",
                DisruptionSeverity.HIGH: "Significant labor shortage",
                DisruptionSeverity.CRITICAL: "Critical staffing crisis"
            }
        }
        
        return descriptions.get(disruption_type, {}).get(
            severity, f"{severity.value} {disruption_type.value} disruption"
        )
    
    def simulate_disruption_impact(
        self, 
        event: DisruptionEvent,
        baseline_metrics: Dict[str, float],
        simulation_duration: float = 24.0
    ) -> Dict[str, Any]:
        """Simulate the impact of a disruption event on port operations"""
        
        logger.info(f"Simulating disruption impact: {event.description}")
        
        # Calculate time-based impact
        impact_timeline = []
        current_time = event.start_time
        end_simulation = event.start_time + timedelta(hours=simulation_duration)
        
        while current_time <= end_simulation:
            if event.is_active(current_time):
                # During disruption
                capacity_impact = 1.0 - event.capacity_reduction
                processing_impact = 1.0 + event.processing_time_increase
                
                # Calculate ripple effects
                queue_impact = self._calculate_queue_impact(event, current_time)
                waiting_time_impact = self._calculate_waiting_time_impact(event, current_time)
                
            else:
                # Recovery period
                recovery_factor = self._calculate_recovery_factor(
                    current_time, event.end_time, event.severity
                )
                capacity_impact = 1.0 - (event.capacity_reduction * recovery_factor)
                processing_impact = 1.0 + (event.processing_time_increase * recovery_factor)
                queue_impact = baseline_metrics.get('average_queue_length', 0) * recovery_factor
                waiting_time_impact = baseline_metrics.get('average_waiting_time', 0) * recovery_factor
            
            impact_timeline.append({
                'timestamp': current_time,
                'capacity_factor': capacity_impact,
                'processing_factor': processing_impact,
                'queue_length': queue_impact,
                'waiting_time': waiting_time_impact
            })
            
            current_time += timedelta(hours=1)
        
        # Calculate aggregate metrics
        total_impact = self._calculate_aggregate_impact(impact_timeline, baseline_metrics)
        
        return {
            'event': event,
            'timeline': impact_timeline,
            'aggregate_impact': total_impact,
            'recovery_recommendations': self._generate_recovery_recommendations(event)
        }
    
    def _calculate_queue_impact(self, event: DisruptionEvent, current_time: datetime) -> float:
        """Calculate impact on vessel queue length"""
        base_queue = 3.0  # Baseline queue length
        
        if event.disruption_type == DisruptionType.CONGESTION:
            return base_queue * (1.0 + event.capacity_reduction * 2)
        elif event.disruption_type == DisruptionType.EQUIPMENT_FAILURE:
            return base_queue * (1.0 + event.capacity_reduction * 1.5)
        else:
            return base_queue * (1.0 + event.capacity_reduction)
    
    def _calculate_waiting_time_impact(self, event: DisruptionEvent, current_time: datetime) -> float:
        """Calculate impact on vessel waiting times"""
        base_waiting = 4.0  # Baseline waiting time in hours
        
        # Waiting time increases more dramatically with capacity reduction
        waiting_multiplier = 1.0 + (event.capacity_reduction * 3) + (event.processing_time_increase * 2)
        
        return base_waiting * waiting_multiplier
    
    def _calculate_recovery_factor(self, current_time: datetime, disruption_end: datetime, severity: DisruptionSeverity) -> float:
        """Calculate recovery factor based on time since disruption ended"""
        if current_time <= disruption_end:
            return 1.0
        
        hours_since_end = (current_time - disruption_end).total_seconds() / 3600
        
        # Recovery time based on severity
        recovery_times = {
            DisruptionSeverity.LOW: 2.0,
            DisruptionSeverity.MEDIUM: 4.0,
            DisruptionSeverity.HIGH: 8.0,
            DisruptionSeverity.CRITICAL: 12.0
        }
        
        recovery_time = recovery_times[severity]
        recovery_factor = max(0.0, 1.0 - (hours_since_end / recovery_time))
        
        return recovery_factor
    
    def _calculate_aggregate_impact(self, timeline: List[Dict], baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate aggregate impact metrics"""
        if not timeline:
            return {}
        
        # Convert timeline to DataFrame for easier analysis
        df = pd.DataFrame(timeline)
        
        return {
            'average_capacity_reduction': 1.0 - df['capacity_factor'].mean(),
            'peak_capacity_reduction': 1.0 - df['capacity_factor'].min(),
            'average_processing_delay': df['processing_factor'].mean() - 1.0,
            'peak_processing_delay': df['processing_factor'].max() - 1.0,
            'average_queue_increase': df['queue_length'].mean() - baseline_metrics.get('average_queue_length', 0),
            'peak_queue_length': df['queue_length'].max(),
            'average_waiting_time_increase': df['waiting_time'].mean() - baseline_metrics.get('average_waiting_time', 0),
            'peak_waiting_time': df['waiting_time'].max(),
            'total_disruption_hours': len([t for t in timeline if t['capacity_factor'] < 1.0]),
            'recovery_time_hours': len([t for t in timeline if 0.95 < t['capacity_factor'] < 1.0])
        }
    
    def _generate_recovery_recommendations(self, event: DisruptionEvent) -> List[Dict[str, Any]]:
        """Generate recovery strategy recommendations"""
        recommendations = []
        
        for strategy_id in event.recovery_strategies:
            strategy = self.recovery_strategies[strategy_id]
            
            # Calculate effectiveness for this specific event
            effectiveness = self._calculate_strategy_effectiveness(strategy, event)
            
            recommendations.append({
                'strategy_id': strategy_id,
                'name': strategy.name,
                'description': strategy.description,
                'effectiveness': effectiveness,
                'implementation_time': strategy.implementation_time,
                'cost': strategy.cost,
                'priority': self._calculate_strategy_priority(strategy, event, effectiveness)
            })
        
        # Sort by priority (higher is better)
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations
    
    def _calculate_strategy_effectiveness(self, strategy: RecoveryStrategy, event: DisruptionEvent) -> float:
        """Calculate strategy effectiveness for specific event"""
        base_effectiveness = strategy.effectiveness
        
        # Adjust based on event severity
        severity_adjustments = {
            DisruptionSeverity.LOW: 1.2,
            DisruptionSeverity.MEDIUM: 1.0,
            DisruptionSeverity.HIGH: 0.8,
            DisruptionSeverity.CRITICAL: 0.6
        }
        
        adjusted_effectiveness = base_effectiveness * severity_adjustments[event.severity]
        return min(1.0, adjusted_effectiveness)
    
    def _calculate_strategy_priority(self, strategy: RecoveryStrategy, event: DisruptionEvent, effectiveness: float) -> float:
        """Calculate strategy priority score"""
        # Priority based on effectiveness, cost, and implementation time
        cost_factor = 1.0 / (1.0 + strategy.cost / 10000.0)  # Normalize cost impact
        time_factor = 1.0 / (1.0 + strategy.implementation_time / 4.0)  # Normalize time impact
        
        priority = effectiveness * 0.5 + cost_factor * 0.3 + time_factor * 0.2
        return priority
    
    def create_sample_disruption_scenarios(self) -> List[DisruptionEvent]:
        """Create sample disruption scenarios for testing"""
        base_time = datetime.now()
        
        scenarios = [
            # Weather disruptions
            self.create_disruption_event(
                DisruptionType.WEATHER,
                DisruptionSeverity.MEDIUM,
                base_time,
                6.0,
                ["Berth_1", "Berth_2"]
            ),
            
            # Equipment failure
            self.create_disruption_event(
                DisruptionType.EQUIPMENT_FAILURE,
                DisruptionSeverity.HIGH,
                base_time + timedelta(hours=12),
                8.0,
                ["Berth_3"]
            ),
            
            # Congestion
            self.create_disruption_event(
                DisruptionType.CONGESTION,
                DisruptionSeverity.HIGH,
                base_time + timedelta(hours=24),
                12.0,
                ["Berth_1", "Berth_2", "Berth_3", "Berth_4"]
            ),
            
            # Labor shortage
            self.create_disruption_event(
                DisruptionType.LABOR_SHORTAGE,
                DisruptionSeverity.MEDIUM,
                base_time + timedelta(hours=36),
                16.0,
                ["Berth_2", "Berth_4"]
            )
        ]
        
        return scenarios
    
    def run_disruption_comparison(self, scenarios: List[DisruptionEvent]) -> Dict[str, Any]:
        """Run comparison analysis across multiple disruption scenarios"""
        
        baseline_metrics = {
            'average_queue_length': 3.0,
            'average_waiting_time': 4.0,
            'berth_utilization': 0.75,
            'throughput_per_hour': 50.0
        }
        
        results = {
            'scenarios': [],
            'comparison_metrics': {},
            'summary': {}
        }
        
        for scenario in scenarios:
            impact_result = self.simulate_disruption_impact(scenario, baseline_metrics)
            results['scenarios'].append(impact_result)
        
        # Generate comparison metrics
        results['comparison_metrics'] = self._generate_comparison_metrics(results['scenarios'])
        
        # Generate summary insights
        results['summary'] = self._generate_disruption_summary(results['scenarios'])
        
        return results
    
    def _generate_comparison_metrics(self, scenario_results: List[Dict]) -> Dict[str, Any]:
        """Generate comparison metrics across scenarios"""
        if not scenario_results:
            return {}
        
        metrics = {
            'most_impactful_disruption': None,
            'longest_recovery_time': None,
            'highest_cost_impact': None,
            'average_impacts': {}
        }
        
        # Find most impactful disruption
        max_impact = 0
        for result in scenario_results:
            impact = result['aggregate_impact'].get('peak_capacity_reduction', 0)
            if impact > max_impact:
                max_impact = impact
                metrics['most_impactful_disruption'] = {
                    'event_id': result['event'].event_id,
                    'type': result['event'].disruption_type.value,
                    'impact': impact
                }
        
        # Calculate average impacts
        avg_capacity_reduction = np.mean([
            r['aggregate_impact'].get('average_capacity_reduction', 0) 
            for r in scenario_results
        ])
        avg_waiting_increase = np.mean([
            r['aggregate_impact'].get('average_waiting_time_increase', 0) 
            for r in scenario_results
        ])
        
        metrics['average_impacts'] = {
            'capacity_reduction': avg_capacity_reduction,
            'waiting_time_increase': avg_waiting_increase
        }
        
        return metrics
    
    def _generate_disruption_summary(self, scenario_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary insights from disruption analysis"""
        
        total_scenarios = len(scenario_results)
        if total_scenarios == 0:
            return {}
        
        # Count disruption types
        type_counts = {}
        for result in scenario_results:
            dtype = result['event'].disruption_type.value
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        # Calculate total impact hours
        total_impact_hours = sum([
            r['aggregate_impact'].get('total_disruption_hours', 0) 
            for r in scenario_results
        ])
        
        return {
            'total_scenarios_analyzed': total_scenarios,
            'disruption_type_distribution': type_counts,
            'total_impact_hours': total_impact_hours,
            'average_impact_duration': total_impact_hours / total_scenarios if total_scenarios > 0 else 0,
            'key_insights': [
                f"Analyzed {total_scenarios} disruption scenarios",
                f"Total operational impact: {total_impact_hours:.1f} hours",
                f"Most common disruption type: {max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'N/A'}"
            ]
        }

def create_sample_disruption_analysis():
    """Create and run sample disruption analysis"""
    simulator = DisruptionSimulator()
    
    # Create sample scenarios
    scenarios = simulator.create_sample_disruption_scenarios()
    
    # Run comparison analysis
    results = simulator.run_disruption_comparison(scenarios)
    
    print("\n=== DISRUPTION IMPACT ANALYSIS ===")
    print(f"Analyzed {len(scenarios)} disruption scenarios")
    
    for i, result in enumerate(results['scenarios']):
        event = result['event']
        impact = result['aggregate_impact']
        
        print(f"\nScenario {i+1}: {event.description}")
        print(f"  Type: {event.disruption_type.value}")
        print(f"  Severity: {event.severity.value}")
        print(f"  Duration: {event.duration_hours:.1f} hours")
        print(f"  Peak Capacity Reduction: {impact.get('peak_capacity_reduction', 0):.1%}")
        print(f"  Average Waiting Time Increase: {impact.get('average_waiting_time_increase', 0):.1f} hours")
        print(f"  Recovery Time: {impact.get('recovery_time_hours', 0):.1f} hours")
    
    print("\n=== SUMMARY INSIGHTS ===")
    for insight in results['summary'].get('key_insights', []):
        print(f"  • {insight}")
    
    return results

if __name__ == "__main__":
    create_sample_disruption_analysis()