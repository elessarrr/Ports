"""Maintenance Window Optimization Module

This module implements optimization algorithms specifically designed for maintenance
window operations at Hong Kong Port. It handles scenarios where berth capacity
is reduced due to planned maintenance, requiring intelligent resource allocation
and scheduling to minimize operational disruption.

Key Features:
- Maintenance-aware berth allocation with capacity constraints
- Risk mitigation strategies for reduced operational capacity
- Cost-benefit analysis for maintenance scheduling
- Integration with strategic simulation scenarios
- Business impact assessment and reporting

Approach: Uses constraint-based optimization with risk assessment to maintain
operational efficiency during maintenance periods while ensuring safety and
compliance requirements are met.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from copy import deepcopy
import random
from enum import Enum

# Import existing optimization modules
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'ai'))
sys.path.insert(0, parent_dir)

try:
    from optimization import BerthAllocationOptimizer, Ship, Berth, OptimizationResult
    from strategic_simulations import StrategicScenarioParameters, StrategicBusinessMetrics
except ImportError as e:
    logging.warning(f"Import warning: {e}")
    # Define minimal classes for standalone operation
    @dataclass
    class Ship:
        id: str
        arrival_time: datetime
        ship_type: str
        size: float
        priority: int = 1
        estimated_service_time: float = 0.0
        containers_to_load: int = 0
        containers_to_unload: int = 0
    
    @dataclass
    class Berth:
        id: str
        capacity: float
        crane_count: int
        suitable_ship_types: List[str]
        is_available: bool = True
        current_ship: Optional[str] = None
        available_from: Optional[datetime] = None

logger = logging.getLogger(__name__)

class MaintenanceType(Enum):
    """Types of maintenance operations"""
    ROUTINE_INSPECTION = "routine_inspection"
    CRANE_MAINTENANCE = "crane_maintenance"
    BERTH_REPAIR = "berth_repair"
    INFRASTRUCTURE_UPGRADE = "infrastructure_upgrade"
    EMERGENCY_REPAIR = "emergency_repair"
    SCHEDULED_OVERHAUL = "scheduled_overhaul"

class MaintenancePriority(Enum):
    """Priority levels for maintenance operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RiskLevel(Enum):
    """Risk levels for maintenance operations"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MaintenanceWindow:
    """Represents a planned maintenance window"""
    id: str
    berth_id: str
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    start_time: datetime
    end_time: datetime
    estimated_duration: float  # hours
    cost: float
    risk_level: RiskLevel
    required_resources: List[str]
    impact_description: str
    
    @property
    def duration_hours(self) -> float:
        """Calculate duration in hours"""
        return (self.end_time - self.start_time).total_seconds() / 3600

@dataclass
class MaintenanceMetrics:
    """Metrics specific to maintenance window operations"""
    total_ships_processed: int
    average_waiting_time: float
    maintenance_windows_completed: int
    operational_capacity_reduction: float  # percentage
    revenue_impact: float  # negative value for loss
    cost_savings_from_optimization: float
    risk_mitigation_score: float
    customer_satisfaction_impact: float
    maintenance_efficiency: float
    delayed_ships_count: int
    rescheduled_maintenance_count: int
    emergency_interventions: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting"""
        return asdict(self)

@dataclass
class MaintenanceOptimizationConfig:
    """Configuration for maintenance window optimization"""
    max_capacity_reduction: float = 0.4  # Maximum 40% capacity reduction
    min_operational_berths: int = 2  # Minimum berths that must remain operational
    emergency_buffer_hours: float = 4.0  # Emergency response buffer
    cost_weight: float = 0.3
    risk_weight: float = 0.4
    efficiency_weight: float = 0.3
    allow_maintenance_rescheduling: bool = True
    max_waiting_time_threshold: float = 8.0  # hours
    priority_ship_max_delay: float = 2.0  # hours for high priority ships

class MaintenanceWindowOptimizer:
    """Optimizer for maintenance window operations with reduced capacity"""
    
    def __init__(self, config: MaintenanceOptimizationConfig = None):
        """Initialize the maintenance window optimizer
        
        Args:
            config: Optimization configuration parameters
        """
        self.config = config or MaintenanceOptimizationConfig()
        self.ships: List[Ship] = []
        self.berths: List[Berth] = []
        self.maintenance_windows: List[MaintenanceWindow] = []
        self.current_time = datetime.now()
        self.optimization_history: List[Dict] = []
        
        logger.info("Initialized MaintenanceWindowOptimizer")
    
    def add_ships(self, ships: List[Ship]) -> None:
        """Add multiple ships to the optimization queue"""
        self.ships.extend(ships)
        logger.info(f"Added {len(ships)} ships to maintenance optimization queue")
    
    def add_berths(self, berths: List[Berth]) -> None:
        """Add multiple berths to the available berths"""
        self.berths.extend(berths)
        logger.info(f"Added {len(berths)} berths to maintenance optimization")
    
    def add_maintenance_windows(self, maintenance_windows: List[MaintenanceWindow]) -> None:
        """Add maintenance windows to the schedule
        
        Args:
            maintenance_windows: List of planned maintenance windows
        """
        self.maintenance_windows.extend(maintenance_windows)
        logger.info(f"Added {len(maintenance_windows)} maintenance windows")
    
    def get_available_berths_at_time(self, check_time: datetime) -> List[Berth]:
        """Get list of berths available at a specific time
        
        Args:
            check_time: Time to check berth availability
            
        Returns:
            List of available berths
        """
        available_berths = []
        
        for berth in self.berths:
            is_under_maintenance = False
            
            # Check if berth is under maintenance at this time
            for maintenance in self.maintenance_windows:
                if (maintenance.berth_id == berth.id and 
                    maintenance.start_time <= check_time <= maintenance.end_time):
                    is_under_maintenance = True
                    break
            
            if not is_under_maintenance:
                available_berths.append(berth)
        
        return available_berths
    
    def calculate_capacity_reduction(self, time_period: Tuple[datetime, datetime]) -> float:
        """Calculate capacity reduction during a time period
        
        Args:
            time_period: Tuple of (start_time, end_time)
            
        Returns:
            Capacity reduction as percentage (0.0 to 1.0)
        """
        start_time, end_time = time_period
        total_berth_capacity = len(self.berths)
        
        if total_berth_capacity == 0:
            return 1.0  # 100% reduction if no berths
        
        # Calculate average berths under maintenance during period
        maintenance_impact = 0
        period_hours = (end_time - start_time).total_seconds() / 3600
        
        for maintenance in self.maintenance_windows:
            # Check overlap with time period
            overlap_start = max(start_time, maintenance.start_time)
            overlap_end = min(end_time, maintenance.end_time)
            
            if overlap_start < overlap_end:
                overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
                maintenance_impact += overlap_hours / period_hours
        
        capacity_reduction = min(maintenance_impact / total_berth_capacity, 1.0)
        return capacity_reduction
    
    def assess_maintenance_risk(self, maintenance: MaintenanceWindow, 
                              ship_schedule: List[Dict]) -> Dict[str, Any]:
        """Assess risk of a maintenance window given current ship schedule
        
        Args:
            maintenance: Maintenance window to assess
            ship_schedule: Current ship schedule
            
        Returns:
            Risk assessment dictionary
        """
        risk_factors = {
            'capacity_impact': 0,
            'ship_delays': 0,
            'revenue_loss': 0,
            'operational_complexity': 0,
            'safety_risk': 0
        }
        
        # Assess capacity impact
        available_berths = self.get_available_berths_at_time(maintenance.start_time)
        capacity_reduction = 1 - (len(available_berths) - 1) / len(self.berths)
        risk_factors['capacity_impact'] = min(capacity_reduction * 100, 100)
        
        # Assess ship delays
        affected_ships = 0
        total_delay_hours = 0
        
        for schedule_entry in ship_schedule:
            if (schedule_entry['berth_id'] == maintenance.berth_id and
                schedule_entry['start_time'] <= maintenance.end_time and
                schedule_entry['end_time'] >= maintenance.start_time):
                affected_ships += 1
                # Estimate delay (simplified)
                delay = min(maintenance.duration_hours, 8)  # Cap at 8 hours
                total_delay_hours += delay
        
        risk_factors['ship_delays'] = affected_ships
        
        # Assess revenue loss (simplified calculation)
        avg_revenue_per_hour = 10000  # $10k per hour per berth
        revenue_loss = maintenance.duration_hours * avg_revenue_per_hour
        risk_factors['revenue_loss'] = revenue_loss
        
        # Operational complexity based on maintenance type
        complexity_scores = {
            MaintenanceType.ROUTINE_INSPECTION: 1,
            MaintenanceType.CRANE_MAINTENANCE: 3,
            MaintenanceType.BERTH_REPAIR: 4,
            MaintenanceType.INFRASTRUCTURE_UPGRADE: 5,
            MaintenanceType.EMERGENCY_REPAIR: 2,
            MaintenanceType.SCHEDULED_OVERHAUL: 5
        }
        risk_factors['operational_complexity'] = complexity_scores.get(maintenance.maintenance_type, 3)
        
        # Safety risk based on maintenance priority and type
        safety_scores = {
            MaintenancePriority.LOW: 1,
            MaintenancePriority.MEDIUM: 2,
            MaintenancePriority.HIGH: 4,
            MaintenancePriority.CRITICAL: 5,
            MaintenancePriority.EMERGENCY: 5
        }
        risk_factors['safety_risk'] = safety_scores.get(maintenance.priority, 3)
        
        # Calculate overall risk score
        weights = {
            'capacity_impact': 0.25,
            'ship_delays': 0.20,
            'revenue_loss': 0.15,
            'operational_complexity': 0.20,
            'safety_risk': 0.20
        }
        
        overall_risk = sum(risk_factors[factor] * weights[factor] for factor in weights)
        
        return {
            'overall_risk_score': overall_risk,
            'risk_factors': risk_factors,
            'risk_level': self._categorize_risk_level(overall_risk),
            'mitigation_recommendations': self._generate_risk_mitigation(risk_factors)
        }
    
    def _categorize_risk_level(self, risk_score: float) -> RiskLevel:
        """Categorize numerical risk score into risk level"""
        if risk_score < 20:
            return RiskLevel.MINIMAL
        elif risk_score < 40:
            return RiskLevel.LOW
        elif risk_score < 60:
            return RiskLevel.MEDIUM
        elif risk_score < 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_risk_mitigation(self, risk_factors: Dict[str, float]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        if risk_factors['capacity_impact'] > 50:
            recommendations.append("Consider rescheduling maintenance to off-peak hours")
            recommendations.append("Implement temporary capacity expansion measures")
        
        if risk_factors['ship_delays'] > 5:
            recommendations.append("Notify affected shipping lines in advance")
            recommendations.append("Prepare alternative berth assignments")
        
        if risk_factors['revenue_loss'] > 50000:
            recommendations.append("Evaluate cost-benefit of maintenance timing")
            recommendations.append("Consider phased maintenance approach")
        
        if risk_factors['operational_complexity'] > 3:
            recommendations.append("Ensure adequate skilled maintenance personnel")
            recommendations.append("Prepare contingency plans for extended maintenance")
        
        if risk_factors['safety_risk'] > 3:
            recommendations.append("Prioritize safety protocols and compliance")
            recommendations.append("Consider emergency response readiness")
        
        return recommendations
    
    def optimize_maintenance_schedule(self, current_time: datetime = None) -> Dict[str, Any]:
        """Optimize maintenance schedule considering operational constraints
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Optimized maintenance schedule and ship assignments
        """
        if current_time:
            self.current_time = current_time
        
        logger.info(f"Optimizing maintenance schedule for {len(self.maintenance_windows)} windows")
        
        # Step 1: Assess current maintenance schedule risks
        initial_ship_schedule = self._create_baseline_ship_schedule()
        maintenance_risks = []
        
        for maintenance in self.maintenance_windows:
            risk_assessment = self.assess_maintenance_risk(maintenance, initial_ship_schedule)
            maintenance_risks.append({
                'maintenance': maintenance,
                'risk_assessment': risk_assessment
            })
        
        # Step 2: Optimize maintenance timing if allowed
        if self.config.allow_maintenance_rescheduling:
            optimized_maintenance = self._optimize_maintenance_timing(maintenance_risks)
        else:
            optimized_maintenance = self.maintenance_windows.copy()
        
        # Step 3: Optimize ship assignments with maintenance constraints
        ship_assignments = self._optimize_ship_assignments_with_maintenance(optimized_maintenance)
        
        # Step 4: Calculate comprehensive metrics
        metrics = self._calculate_maintenance_metrics(ship_assignments, optimized_maintenance)
        
        # Step 5: Generate detailed schedule
        detailed_schedule = self._generate_maintenance_schedule(ship_assignments, optimized_maintenance)
        
        optimization_result = {
            'timestamp': self.current_time,
            'original_maintenance_windows': len(self.maintenance_windows),
            'optimized_maintenance_windows': len(optimized_maintenance),
            'ship_assignments': ship_assignments,
            'maintenance_schedule': optimized_maintenance,
            'metrics': metrics,
            'detailed_schedule': detailed_schedule,
            'risk_assessments': maintenance_risks,
            'capacity_impact': self._calculate_overall_capacity_impact(optimized_maintenance)
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"Maintenance optimization completed. Average waiting time: {metrics.average_waiting_time:.2f}h")
        
        return optimization_result
    
    def _create_baseline_ship_schedule(self) -> List[Dict]:
        """Create baseline ship schedule without optimization"""
        schedule = []
        berth_availability = {berth.id: self.current_time for berth in self.berths}
        
        # Simple FCFS scheduling
        for ship in sorted(self.ships, key=lambda s: s.arrival_time):
            best_berth = None
            earliest_time = None
            
            for berth in self.berths:
                if self._is_berth_suitable_for_ship(ship, berth):
                    available_time = max(berth_availability[berth.id], ship.arrival_time)
                    if earliest_time is None or available_time < earliest_time:
                        earliest_time = available_time
                        best_berth = berth
            
            if best_berth:
                service_time = self._estimate_service_time(ship, best_berth)
                end_time = earliest_time + timedelta(hours=service_time)
                
                schedule.append({
                    'ship_id': ship.id,
                    'berth_id': best_berth.id,
                    'start_time': earliest_time,
                    'end_time': end_time,
                    'service_time': service_time
                })
                
                berth_availability[best_berth.id] = end_time
        
        return schedule
    
    def _optimize_maintenance_timing(self, maintenance_risks: List[Dict]) -> List[MaintenanceWindow]:
        """Optimize maintenance timing to minimize operational impact"""
        optimized_maintenance = []
        
        # Sort maintenance by priority and risk
        sorted_maintenance = sorted(maintenance_risks, 
                                  key=lambda x: (x['maintenance'].priority.value, 
                                               x['risk_assessment']['overall_risk_score']))
        
        for maintenance_risk in sorted_maintenance:
            maintenance = maintenance_risk['maintenance']
            risk_assessment = maintenance_risk['risk_assessment']
            
            # For high-risk maintenance, try to reschedule
            if (risk_assessment['overall_risk_score'] > 60 and 
                maintenance.priority != MaintenancePriority.EMERGENCY):
                
                # Try to find better timing
                better_timing = self._find_optimal_maintenance_timing(maintenance)
                if better_timing:
                    optimized_maintenance.append(better_timing)
                else:
                    optimized_maintenance.append(maintenance)
            else:
                optimized_maintenance.append(maintenance)
        
        return optimized_maintenance
    
    def _find_optimal_maintenance_timing(self, maintenance: MaintenanceWindow) -> Optional[MaintenanceWindow]:
        """Find optimal timing for a maintenance window"""
        # Try different time slots within a reasonable range
        original_start = maintenance.start_time
        duration = maintenance.duration_hours
        
        # Search window: +/- 48 hours from original time
        search_start = original_start - timedelta(hours=48)
        search_end = original_start + timedelta(hours=48)
        
        best_timing = None
        best_score = float('inf')
        
        # Try 4-hour intervals
        current_time = search_start
        while current_time <= search_end:
            test_maintenance = MaintenanceWindow(
                id=maintenance.id,
                berth_id=maintenance.berth_id,
                maintenance_type=maintenance.maintenance_type,
                priority=maintenance.priority,
                start_time=current_time,
                end_time=current_time + timedelta(hours=duration),
                estimated_duration=duration,
                cost=maintenance.cost,
                risk_level=maintenance.risk_level,
                required_resources=maintenance.required_resources,
                impact_description=maintenance.impact_description
            )
            
            # Calculate impact score for this timing
            impact_score = self._calculate_maintenance_impact_score(test_maintenance)
            
            if impact_score < best_score:
                best_score = impact_score
                best_timing = test_maintenance
            
            current_time += timedelta(hours=4)
        
        # Only return if significantly better (>20% improvement)
        original_score = self._calculate_maintenance_impact_score(maintenance)
        if best_score < original_score * 0.8:
            return best_timing
        
        return None
    
    def _calculate_maintenance_impact_score(self, maintenance: MaintenanceWindow) -> float:
        """Calculate impact score for maintenance timing"""
        # Simplified impact calculation
        # Consider: ship traffic, operational hours, resource availability
        
        # Peak hours penalty (6 AM - 6 PM)
        hour = maintenance.start_time.hour
        if 6 <= hour <= 18:
            peak_penalty = 2.0
        else:
            peak_penalty = 1.0
        
        # Weekend bonus (less traffic)
        if maintenance.start_time.weekday() >= 5:  # Saturday or Sunday
            weekend_bonus = 0.7
        else:
            weekend_bonus = 1.0
        
        # Duration penalty
        duration_penalty = min(maintenance.duration_hours / 8, 2.0)  # Normalize to 8 hours
        
        impact_score = peak_penalty * weekend_bonus * duration_penalty
        return impact_score
    
    def _optimize_ship_assignments_with_maintenance(self, 
                                                   maintenance_windows: List[MaintenanceWindow]) -> Dict[str, str]:
        """Optimize ship assignments considering maintenance constraints"""
        assignments = {}
        
        # Create time-based berth availability considering maintenance
        berth_schedules = {berth.id: [] for berth in self.berths}
        
        # Add maintenance windows to berth schedules
        for maintenance in maintenance_windows:
            berth_schedules[maintenance.berth_id].append({
                'type': 'maintenance',
                'start_time': maintenance.start_time,
                'end_time': maintenance.end_time,
                'maintenance_id': maintenance.id
            })
        
        # Sort ships by priority and arrival time
        sorted_ships = sorted(self.ships, key=lambda s: (-s.priority, s.arrival_time))
        
        for ship in sorted_ships:
            best_assignment = self._find_best_berth_assignment(ship, berth_schedules)
            
            if best_assignment:
                berth_id, start_time, service_time = best_assignment
                assignments[ship.id] = berth_id
                
                # Update berth schedule
                end_time = start_time + timedelta(hours=service_time)
                berth_schedules[berth_id].append({
                    'type': 'ship',
                    'ship_id': ship.id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'service_time': service_time
                })
                
                # Sort schedule by start time
                berth_schedules[berth_id].sort(key=lambda x: x['start_time'])
        
        return assignments
    
    def _find_best_berth_assignment(self, ship: Ship, 
                                   berth_schedules: Dict[str, List[Dict]]) -> Optional[Tuple[str, datetime, float]]:
        """Find best berth assignment for a ship considering maintenance"""
        best_assignment = None
        min_waiting_time = float('inf')
        
        for berth in self.berths:
            if not self._is_berth_suitable_for_ship(ship, berth):
                continue
            
            service_time = self._estimate_service_time(ship, berth)
            earliest_slot = self._find_earliest_available_slot(ship, berth, berth_schedules[berth.id], service_time)
            
            if earliest_slot:
                start_time = earliest_slot
                waiting_time = max(0, (start_time - ship.arrival_time).total_seconds() / 3600)
                
                if waiting_time < min_waiting_time:
                    min_waiting_time = waiting_time
                    best_assignment = (berth.id, start_time, service_time)
        
        return best_assignment
    
    def _find_earliest_available_slot(self, ship: Ship, berth: Berth, 
                                     schedule: List[Dict], service_time: float) -> Optional[datetime]:
        """Find earliest available slot in berth schedule"""
        required_duration = timedelta(hours=service_time)
        earliest_start = max(ship.arrival_time, self.current_time)
        
        if not schedule:
            return earliest_start
        
        # Sort schedule by start time
        sorted_schedule = sorted(schedule, key=lambda x: x['start_time'])
        
        # Check if we can fit before first scheduled item
        if earliest_start + required_duration <= sorted_schedule[0]['start_time']:
            return earliest_start
        
        # Check gaps between scheduled items
        for i in range(len(sorted_schedule) - 1):
            gap_start = max(earliest_start, sorted_schedule[i]['end_time'])
            gap_end = sorted_schedule[i + 1]['start_time']
            
            if gap_start + required_duration <= gap_end:
                return gap_start
        
        # Check after last scheduled item
        last_end = sorted_schedule[-1]['end_time']
        return max(earliest_start, last_end)
    
    def _is_berth_suitable_for_ship(self, ship: Ship, berth: Berth) -> bool:
        """Check if berth is suitable for ship"""
        if ship.size > berth.capacity:
            return False
        if berth.suitable_ship_types and ship.ship_type not in berth.suitable_ship_types:
            return False
        return True
    
    def _estimate_service_time(self, ship: Ship, berth: Berth) -> float:
        """Estimate service time for ship at berth"""
        base_time = 2.0
        
        # Container handling
        total_containers = ship.containers_to_load + ship.containers_to_unload
        if total_containers > 0:
            containers_per_hour_per_crane = 25  # Reduced efficiency during maintenance periods
            container_time = total_containers / (berth.crane_count * containers_per_hour_per_crane)
        else:
            container_time = 0
        
        # Ship size and type factors
        size_factor = max(1.0, ship.size / 1000)
        type_factors = {
            'container': 1.0,
            'bulk': 1.5,
            'tanker': 1.3,
            'general': 1.2,
            'passenger': 0.8
        }
        type_factor = type_factors.get(ship.ship_type.lower(), 1.0)
        
        # Maintenance period efficiency reduction
        maintenance_factor = 1.15  # 15% increase due to reduced efficiency
        
        estimated_time = (base_time + container_time * size_factor * type_factor) * maintenance_factor
        return max(estimated_time, 0.5)
    
    def _calculate_maintenance_metrics(self, ship_assignments: Dict[str, str], 
                                     maintenance_windows: List[MaintenanceWindow]) -> MaintenanceMetrics:
        """Calculate comprehensive metrics for maintenance operations"""
        total_waiting_time = 0
        total_ships = len(self.ships)
        assigned_ships = len(ship_assignments)
        delayed_ships = 0
        total_revenue_impact = 0
        
        # Calculate ship-related metrics
        berth_schedules = {berth.id: [] for berth in self.berths}
        
        for ship in self.ships:
            if ship.id not in ship_assignments:
                delayed_ships += 1
                continue
            
            berth_id = ship_assignments[ship.id]
            berth = next((b for b in self.berths if b.id == berth_id), None)
            if not berth:
                continue
            
            service_time = self._estimate_service_time(ship, berth)
            
            # Find actual start time considering maintenance
            if not berth_schedules[berth_id]:
                start_time = max(ship.arrival_time, self.current_time)
            else:
                last_end = berth_schedules[berth_id][-1]['end_time']
                start_time = max(ship.arrival_time, last_end)
            
            # Check for maintenance conflicts
            for maintenance in maintenance_windows:
                if (maintenance.berth_id == berth_id and
                    start_time < maintenance.end_time and
                    start_time + timedelta(hours=service_time) > maintenance.start_time):
                    # Ship conflicts with maintenance, delay it
                    start_time = maintenance.end_time
                    break
            
            end_time = start_time + timedelta(hours=service_time)
            waiting_time = max(0, (start_time - ship.arrival_time).total_seconds() / 3600)
            total_waiting_time += waiting_time
            
            if waiting_time > self.config.max_waiting_time_threshold:
                delayed_ships += 1
            
            # Calculate revenue impact
            containers = ship.containers_to_load + ship.containers_to_unload
            revenue_loss = waiting_time * containers * 2  # $2 per container per hour delay
            total_revenue_impact += revenue_loss
            
            berth_schedules[berth_id].append({
                'ship_id': ship.id,
                'start_time': start_time,
                'end_time': end_time,
                'waiting_time': waiting_time
            })
        
        # Calculate maintenance-related metrics
        total_maintenance_cost = sum(m.cost for m in maintenance_windows)
        maintenance_hours = sum(m.duration_hours for m in maintenance_windows)
        
        # Calculate capacity reduction
        total_berth_hours = len(self.berths) * 24  # Assuming 24-hour period
        capacity_reduction = (maintenance_hours / total_berth_hours) * 100
        
        # Calculate efficiency metrics
        avg_waiting_time = total_waiting_time / total_ships if total_ships > 0 else 0
        customer_satisfaction = max(0, 100 - avg_waiting_time * 5)  # Simplified
        maintenance_efficiency = min(100, (assigned_ships / total_ships) * 100) if total_ships > 0 else 0
        
        # Risk mitigation score (simplified)
        completed_critical_maintenance = sum(1 for m in maintenance_windows 
                                           if m.priority in [MaintenancePriority.CRITICAL, MaintenancePriority.HIGH])
        risk_mitigation_score = min(100, completed_critical_maintenance * 20)
        
        return MaintenanceMetrics(
            total_ships_processed=assigned_ships,
            average_waiting_time=avg_waiting_time,
            maintenance_windows_completed=len(maintenance_windows),
            operational_capacity_reduction=capacity_reduction,
            revenue_impact=-total_revenue_impact,  # Negative for loss
            cost_savings_from_optimization=max(0, total_maintenance_cost * 0.1),  # 10% savings estimate
            risk_mitigation_score=risk_mitigation_score,
            customer_satisfaction_impact=customer_satisfaction,
            maintenance_efficiency=maintenance_efficiency,
            delayed_ships_count=delayed_ships,
            rescheduled_maintenance_count=0,  # Would be calculated if rescheduling occurred
            emergency_interventions=0  # Would be tracked during actual operations
        )
    
    def _generate_maintenance_schedule(self, ship_assignments: Dict[str, str], 
                                     maintenance_windows: List[MaintenanceWindow]) -> List[Dict]:
        """Generate detailed schedule combining ships and maintenance"""
        schedule = []
        
        # Add maintenance windows to schedule
        for maintenance in maintenance_windows:
            schedule.append({
                'type': 'maintenance',
                'id': maintenance.id,
                'berth_id': maintenance.berth_id,
                'maintenance_type': maintenance.maintenance_type.value,
                'priority': maintenance.priority.value,
                'start_time': maintenance.start_time,
                'end_time': maintenance.end_time,
                'duration_hours': maintenance.duration_hours,
                'cost': maintenance.cost,
                'risk_level': maintenance.risk_level.value
            })
        
        # Add ship assignments to schedule
        berth_schedules = {berth.id: [] for berth in self.berths}
        
        for ship in self.ships:
            if ship.id not in ship_assignments:
                continue
            
            berth_id = ship_assignments[ship.id]
            berth = next((b for b in self.berths if b.id == berth_id), None)
            if not berth:
                continue
            
            service_time = self._estimate_service_time(ship, berth)
            start_time = self._calculate_ship_start_time(ship, berth_id, maintenance_windows, berth_schedules)
            end_time = start_time + timedelta(hours=service_time)
            waiting_time = max(0, (start_time - ship.arrival_time).total_seconds() / 3600)
            
            schedule.append({
                'type': 'ship',
                'ship_id': ship.id,
                'ship_type': ship.ship_type,
                'ship_size': ship.size,
                'berth_id': berth_id,
                'arrival_time': ship.arrival_time,
                'start_time': start_time,
                'end_time': end_time,
                'service_time': service_time,
                'waiting_time': waiting_time,
                'containers': ship.containers_to_load + ship.containers_to_unload,
                'priority': ship.priority
            })
            
            berth_schedules[berth_id].append({
                'start_time': start_time,
                'end_time': end_time
            })
        
        return sorted(schedule, key=lambda x: x['start_time'])
    
    def _calculate_ship_start_time(self, ship: Ship, berth_id: str, 
                                 maintenance_windows: List[MaintenanceWindow],
                                 berth_schedules: Dict[str, List[Dict]]) -> datetime:
        """Calculate when a ship can start service considering maintenance"""
        earliest_start = max(ship.arrival_time, self.current_time)
        
        # Check berth schedule
        if berth_schedules[berth_id]:
            last_end = berth_schedules[berth_id][-1]['end_time']
            earliest_start = max(earliest_start, last_end)
        
        # Check maintenance conflicts
        service_time = self._estimate_service_time(ship, next(b for b in self.berths if b.id == berth_id))
        
        for maintenance in maintenance_windows:
            if maintenance.berth_id == berth_id:
                # Check if ship service would conflict with maintenance
                ship_end = earliest_start + timedelta(hours=service_time)
                
                if (earliest_start < maintenance.end_time and ship_end > maintenance.start_time):
                    # Conflict detected, move ship after maintenance
                    earliest_start = maintenance.end_time
        
        return earliest_start
    
    def _calculate_overall_capacity_impact(self, maintenance_windows: List[MaintenanceWindow]) -> Dict[str, float]:
        """Calculate overall capacity impact of maintenance schedule"""
        total_berth_hours = len(self.berths) * 24  # 24-hour period
        total_maintenance_hours = sum(m.duration_hours for m in maintenance_windows)
        
        capacity_reduction = (total_maintenance_hours / total_berth_hours) * 100
        
        # Calculate peak impact (maximum berths under maintenance simultaneously)
        time_slots = []
        for maintenance in maintenance_windows:
            time_slots.append((maintenance.start_time, 'start', maintenance.berth_id))
            time_slots.append((maintenance.end_time, 'end', maintenance.berth_id))
        
        time_slots.sort()
        
        max_simultaneous = 0
        current_maintenance = set()
        
        for time, event_type, berth_id in time_slots:
            if event_type == 'start':
                current_maintenance.add(berth_id)
            else:
                current_maintenance.discard(berth_id)
            
            max_simultaneous = max(max_simultaneous, len(current_maintenance))
        
        peak_capacity_reduction = (max_simultaneous / len(self.berths)) * 100
        
        return {
            'average_capacity_reduction': capacity_reduction,
            'peak_capacity_reduction': peak_capacity_reduction,
            'total_maintenance_hours': total_maintenance_hours,
            'max_simultaneous_maintenance': max_simultaneous
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of maintenance optimization performance"""
        if not self.optimization_history:
            return {"message": "No maintenance optimization runs completed yet"}
        
        latest = self.optimization_history[-1]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'latest_metrics': latest['metrics'].to_dict(),
            'maintenance_windows': latest['optimized_maintenance_windows'],
            'ships_processed': latest['metrics'].total_ships_processed,
            'capacity_impact': latest['capacity_impact'],
            'optimization_timestamp': latest['timestamp']
        }
    
    def clear(self) -> None:
        """Clear ships, berths, and maintenance windows for new optimization"""
        self.ships.clear()
        self.berths.clear()
        self.maintenance_windows.clear()
        logger.info("Cleared all data for new maintenance optimization")


def create_sample_maintenance_scenario() -> Dict[str, Any]:
    """Create a sample maintenance scenario for testing
    
    Returns:
        Dictionary containing sample scenario data and optimization results
    """
    # Create sample ships
    ships = []
    base_time = datetime.now()
    
    # Generate 12 ships arriving during maintenance period
    for i in range(12):
        arrival_time = base_time + timedelta(hours=random.uniform(0, 8))
        ship = Ship(
            id=f"MAINT_SHIP_{i+1:03d}",
            arrival_time=arrival_time,
            ship_type=random.choice(['container', 'bulk', 'tanker', 'general']),
            size=random.uniform(8000, 30000),
            priority=random.choice([1, 1, 2, 2, 3]),
            containers_to_load=random.randint(50, 600),
            containers_to_unload=random.randint(100, 800)
        )
        ships.append(ship)
    
    # Create sample berths
    berths = [
        Berth(id="BERTH_M1", capacity=35000, crane_count=4, suitable_ship_types=['container', 'general']),
        Berth(id="BERTH_M2", capacity=30000, crane_count=3, suitable_ship_types=['container', 'general']),
        Berth(id="BERTH_M3", capacity=45000, crane_count=2, suitable_ship_types=['bulk', 'tanker']),
        Berth(id="BERTH_M4", capacity=40000, crane_count=3, suitable_ship_types=['bulk', 'tanker', 'general'])
    ]
    
    # Create sample maintenance windows
    maintenance_windows = [
        MaintenanceWindow(
            id="MAINT_001",
            berth_id="BERTH_M1",
            maintenance_type=MaintenanceType.CRANE_MAINTENANCE,
            priority=MaintenancePriority.HIGH,
            start_time=base_time + timedelta(hours=2),
            end_time=base_time + timedelta(hours=6),
            estimated_duration=4.0,
            cost=25000,
            risk_level=RiskLevel.MEDIUM,
            required_resources=["crane_technician", "safety_inspector"],
            impact_description="Crane #2 scheduled maintenance"
        ),
        MaintenanceWindow(
            id="MAINT_002",
            berth_id="BERTH_M3",
            maintenance_type=MaintenanceType.BERTH_REPAIR,
            priority=MaintenancePriority.MEDIUM,
            start_time=base_time + timedelta(hours=4),
            end_time=base_time + timedelta(hours=10),
            estimated_duration=6.0,
            cost=45000,
            risk_level=RiskLevel.HIGH,
            required_resources=["structural_engineer", "welding_crew"],
            impact_description="Berth surface repair"
        )
    ]
    
    # Run optimization
    config = MaintenanceOptimizationConfig(
        allow_maintenance_rescheduling=True,
        max_waiting_time_threshold=6.0
    )
    
    optimizer = MaintenanceWindowOptimizer(config)
    optimizer.add_ships(ships)
    optimizer.add_berths(berths)
    optimizer.add_maintenance_windows(maintenance_windows)
    
    result = optimizer.optimize_maintenance_schedule(base_time)
    summary = optimizer.get_optimization_summary()
    
    return {
        'scenario_name': 'Maintenance Window Optimization Test',
        'ships': [{'id': s.id, 'type': s.ship_type, 'size': s.size, 'arrival': s.arrival_time} for s in ships],
        'berths': [{'id': b.id, 'capacity': b.capacity, 'cranes': b.crane_count} for b in berths],
        'maintenance_windows': [{
            'id': m.id, 'berth': m.berth_id, 'type': m.maintenance_type.value,
            'start': m.start_time, 'duration': m.duration_hours, 'cost': m.cost
        } for m in maintenance_windows],
        'optimization_result': result,
        'performance_summary': summary
    }


if __name__ == "__main__":
    # Demo the maintenance window optimizer
    logging.basicConfig(level=logging.INFO)
    
    print("Hong Kong Port Digital Twin - Maintenance Window Optimization")
    print("=" * 70)
    
    scenario = create_sample_maintenance_scenario()
    
    print(f"\nScenario: {scenario['scenario_name']}")
    print(f"Ships: {len(scenario['ships'])}")
    print(f"Berths: {len(scenario['berths'])}")
    print(f"Maintenance Windows: {len(scenario['maintenance_windows'])}")
    
    metrics = scenario['optimization_result']['metrics']
    print(f"\nOptimization Results:")
    print(f"  Ships Processed: {metrics.total_ships_processed}")
    print(f"  Average Waiting Time: {metrics.average_waiting_time:.2f} hours")
    print(f"  Capacity Reduction: {metrics.operational_capacity_reduction:.1f}%")
    print(f"  Revenue Impact: ${metrics.revenue_impact:,.2f}")
    print(f"  Maintenance Efficiency: {metrics.maintenance_efficiency:.1f}%")
    print(f"  Risk Mitigation Score: {metrics.risk_mitigation_score:.1f}")
    print(f"  Delayed Ships: {metrics.delayed_ships_count}")
    
    capacity_impact = scenario['optimization_result']['capacity_impact']
    print(f"\nCapacity Impact:")
    print(f"  Average Reduction: {capacity_impact['average_capacity_reduction']:.1f}%")
    print(f"  Peak Reduction: {capacity_impact['peak_capacity_reduction']:.1f}%")
    print(f"  Total Maintenance Hours: {capacity_impact['total_maintenance_hours']:.1f}")
    
    print(f"\nMaintenance Schedule:")
    for window in scenario['maintenance_windows']:
        print(f"  {window['id']}: {window['berth']} | {window['type']} | "
              f"{window['duration']:.1f}h | ${window['cost']:,}")