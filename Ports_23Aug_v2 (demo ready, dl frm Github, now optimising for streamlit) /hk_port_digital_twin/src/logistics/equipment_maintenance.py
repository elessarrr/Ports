"""Equipment Maintenance Scheduling System for Hong Kong Port Digital Twin

This module manages maintenance operations for port equipment including:
- Preventive maintenance scheduling for cranes, RTGs, and other equipment
- Predictive maintenance based on usage patterns and condition monitoring
- Maintenance resource allocation and crew scheduling
- Equipment downtime optimization and impact analysis
- Spare parts inventory management
- Maintenance cost tracking and optimization

The system ensures optimal equipment availability while minimizing
maintenance costs and operational disruptions.
"""

import logging
import simpy
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import numpy as np
from collections import defaultdict, deque
import heapq

# Configure logging
logger = logging.getLogger(__name__)

class EquipmentType(Enum):
    """Types of port equipment"""
    QUAY_CRANE = "quay_crane"
    RTG_CRANE = "rtg_crane"
    REACH_STACKER = "reach_stacker"
    FORKLIFT = "forklift"
    TERMINAL_TRACTOR = "terminal_tractor"
    MOBILE_CRANE = "mobile_crane"
    CONVEYOR_SYSTEM = "conveyor_system"
    GATE_SYSTEM = "gate_system"
    POWER_GENERATOR = "power_generator"

class MaintenanceType(Enum):
    """Types of maintenance operations"""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"
    INSPECTION = "inspection"
    OVERHAUL = "overhaul"

class EquipmentStatus(Enum):
    """Equipment operational status"""
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    BREAKDOWN = "breakdown"
    STANDBY = "standby"
    OUT_OF_SERVICE = "out_of_service"

class MaintenancePriority(Enum):
    """Maintenance task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    ROUTINE = 5

@dataclass
class Equipment:
    """Port equipment entity"""
    equipment_id: str
    equipment_type: EquipmentType
    location: str
    status: EquipmentStatus = EquipmentStatus.OPERATIONAL
    installation_date: float = 0.0
    last_maintenance: float = 0.0
    operating_hours: float = 0.0
    cycles_completed: int = 0
    condition_score: float = 100.0  # 0-100, higher is better
    maintenance_cost_total: float = 0.0
    downtime_total: float = 0.0
    failure_count: int = 0
    next_scheduled_maintenance: Optional[float] = None
    
    # Equipment-specific parameters
    max_operating_hours: float = 8760.0  # Hours per year
    maintenance_interval: float = 500.0  # Hours between preventive maintenance
    expected_lifetime: float = 87600.0  # 10 years in hours
    
@dataclass
class MaintenanceTask:
    """Maintenance task definition"""
    task_id: str
    equipment_id: str
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    estimated_duration: float  # hours
    required_crew_size: int
    required_skills: List[str]
    spare_parts_needed: List[str]
    scheduled_start: Optional[float] = None
    actual_start: Optional[float] = None
    actual_end: Optional[float] = None
    cost_estimate: float = 0.0
    actual_cost: float = 0.0
    description: str = ""
    
@dataclass
class MaintenanceCrew:
    """Maintenance crew resource"""
    crew_id: str
    crew_size: int
    skills: List[str]
    shift_start: float
    shift_end: float
    is_available: bool = True
    current_task: Optional[str] = None
    hourly_rate: float = 50.0  # HKD per hour
    
@dataclass
class SparePart:
    """Spare parts inventory item"""
    part_id: str
    part_name: str
    equipment_types: List[EquipmentType]
    current_stock: int
    minimum_stock: int
    unit_cost: float
    lead_time: float  # hours to procure
    supplier: str = ""

class EquipmentMaintenanceScheduler:
    """Equipment Maintenance Scheduling System
    
    Manages all aspects of equipment maintenance including scheduling,
    resource allocation, and performance optimization.
    """
    
    def __init__(self, env: simpy.Environment):
        """Initialize the maintenance scheduler
        
        Args:
            env: SimPy environment for simulation timing
        """
        self.env = env
        self.equipment = self._initialize_equipment()
        self.maintenance_crews = self._initialize_crews()
        self.spare_parts = self._initialize_spare_parts()
        
        # Task management
        self.pending_tasks = []  # Priority queue
        self.active_tasks = {}  # task_id -> MaintenanceTask
        self.completed_tasks = []
        self.emergency_tasks = deque()
        
        # Scheduling parameters
        self.maintenance_window_start = 22.0  # 10 PM
        self.maintenance_window_end = 6.0   # 6 AM
        self.max_concurrent_tasks = 5
        
        # Performance metrics
        self.metrics = {
            'equipment_availability': 0.0,
            'mean_time_between_failures': 0.0,
            'mean_time_to_repair': 0.0,
            'maintenance_cost_per_hour': 0.0,
            'preventive_maintenance_ratio': 0.0,
            'spare_parts_availability': 0.0,
            'crew_utilization': 0.0
        }
        
        # Start background processes
        self.env.process(self._equipment_monitoring_process())
        self.env.process(self._maintenance_scheduler_process())
        self.env.process(self._condition_monitoring_process())
    
    def _initialize_equipment(self) -> Dict[str, Equipment]:
        """Initialize port equipment inventory"""
        equipment = {}
        
        # Quay cranes
        for i in range(1, 13):
            equipment[f"QC_{i:02d}"] = Equipment(
                equipment_id=f"QC_{i:02d}",
                equipment_type=EquipmentType.QUAY_CRANE,
                location=f"Berth_{(i-1)//3 + 1}",
                maintenance_interval=750.0,  # 750 hours
                max_operating_hours=8000.0,
                expected_lifetime=175200.0  # 20 years
            )
        
        # RTG cranes
        for i in range(1, 25):
            equipment[f"RTG_{i:02d}"] = Equipment(
                equipment_id=f"RTG_{i:02d}",
                equipment_type=EquipmentType.RTG_CRANE,
                location=f"Yard_Block_{(i-1)//3 + 1}",
                maintenance_interval=500.0,
                max_operating_hours=6000.0,
                expected_lifetime=131400.0  # 15 years
            )
        
        # Reach stackers
        for i in range(1, 16):
            equipment[f"RS_{i:02d}"] = Equipment(
                equipment_id=f"RS_{i:02d}",
                equipment_type=EquipmentType.REACH_STACKER,
                location="Yard_General",
                maintenance_interval=300.0,
                max_operating_hours=4000.0,
                expected_lifetime=87600.0  # 10 years
            )
        
        # Terminal tractors
        for i in range(1, 31):
            equipment[f"TT_{i:02d}"] = Equipment(
                equipment_id=f"TT_{i:02d}",
                equipment_type=EquipmentType.TERMINAL_TRACTOR,
                location="Terminal_General",
                maintenance_interval=200.0,
                max_operating_hours=3000.0,
                expected_lifetime=70080.0  # 8 years
            )
        
        return equipment
    
    def _initialize_crews(self) -> Dict[str, MaintenanceCrew]:
        """Initialize maintenance crews"""
        crews = {}
        
        # Day shift crews
        crews["DAY_MECHANICAL"] = MaintenanceCrew(
            crew_id="DAY_MECHANICAL",
            crew_size=4,
            skills=["mechanical", "hydraulic", "general"],
            shift_start=8.0,
            shift_end=17.0,
            hourly_rate=60.0
        )
        
        crews["DAY_ELECTRICAL"] = MaintenanceCrew(
            crew_id="DAY_ELECTRICAL",
            crew_size=3,
            skills=["electrical", "electronics", "automation"],
            shift_start=8.0,
            shift_end=17.0,
            hourly_rate=70.0
        )
        
        # Night shift crews
        crews["NIGHT_MECHANICAL"] = MaintenanceCrew(
            crew_id="NIGHT_MECHANICAL",
            crew_size=2,
            skills=["mechanical", "general"],
            shift_start=22.0,
            shift_end=6.0,
            hourly_rate=75.0  # Night shift premium
        )
        
        crews["NIGHT_ELECTRICAL"] = MaintenanceCrew(
            crew_id="NIGHT_ELECTRICAL",
            crew_size=2,
            skills=["electrical", "emergency"],
            shift_start=22.0,
            shift_end=6.0,
            hourly_rate=85.0
        )
        
        # Emergency crew (24/7)
        crews["EMERGENCY"] = MaintenanceCrew(
            crew_id="EMERGENCY",
            crew_size=2,
            skills=["emergency", "mechanical", "electrical"],
            shift_start=0.0,
            shift_end=24.0,
            hourly_rate=100.0
        )
        
        return crews
    
    def _initialize_spare_parts(self) -> Dict[str, SparePart]:
        """Initialize spare parts inventory"""
        parts = {
            "HYDRAULIC_PUMP": SparePart(
                part_id="HYDRAULIC_PUMP",
                part_name="Hydraulic Pump Assembly",
                equipment_types=[EquipmentType.QUAY_CRANE, EquipmentType.RTG_CRANE],
                current_stock=5,
                minimum_stock=2,
                unit_cost=15000.0,
                lead_time=168.0  # 1 week
            ),
            "MOTOR_CONTROLLER": SparePart(
                part_id="MOTOR_CONTROLLER",
                part_name="Motor Controller Unit",
                equipment_types=[EquipmentType.QUAY_CRANE, EquipmentType.RTG_CRANE, EquipmentType.REACH_STACKER],
                current_stock=8,
                minimum_stock=3,
                unit_cost=8000.0,
                lead_time=72.0  # 3 days
            ),
            "BRAKE_PAD_SET": SparePart(
                part_id="BRAKE_PAD_SET",
                part_name="Brake Pad Set",
                equipment_types=[EquipmentType.TERMINAL_TRACTOR, EquipmentType.REACH_STACKER],
                current_stock=20,
                minimum_stock=8,
                unit_cost=500.0,
                lead_time=24.0  # 1 day
            ),
            "WIRE_ROPE": SparePart(
                part_id="WIRE_ROPE",
                part_name="Wire Rope Assembly",
                equipment_types=[EquipmentType.QUAY_CRANE, EquipmentType.MOBILE_CRANE],
                current_stock=3,
                minimum_stock=1,
                unit_cost=12000.0,
                lead_time=240.0  # 10 days
            )
        }
        return parts
    
    def schedule_maintenance(self, equipment_id: str, maintenance_type: MaintenanceType, 
                           priority: MaintenancePriority = MaintenancePriority.MEDIUM) -> bool:
        """Schedule maintenance for equipment
        
        Args:
            equipment_id: ID of equipment to maintain
            maintenance_type: Type of maintenance
            priority: Priority level
            
        Returns:
            True if maintenance was scheduled successfully
        """
        if equipment_id not in self.equipment:
            logger.warning(f"Equipment {equipment_id} not found")
            return False
        
        equipment = self.equipment[equipment_id]
        
        # Create maintenance task
        task = self._create_maintenance_task(equipment, maintenance_type, priority)
        
        if maintenance_type == MaintenanceType.EMERGENCY:
            self.emergency_tasks.append(task)
        else:
            # Add to priority queue
            priority_score = self._calculate_task_priority(task)
            heapq.heappush(self.pending_tasks, (priority_score, task.task_id, task))
        
        logger.info(f"Scheduled {maintenance_type.value} maintenance for {equipment_id}")
        return True
    
    def _create_maintenance_task(self, equipment: Equipment, maintenance_type: MaintenanceType, 
                                priority: MaintenancePriority) -> MaintenanceTask:
        """Create a maintenance task"""
        task_id = f"{equipment.equipment_id}_{maintenance_type.value}_{int(self.env.now)}"
        
        # Determine task parameters based on equipment type and maintenance type
        duration, crew_size, skills, parts = self._get_maintenance_requirements(
            equipment.equipment_type, maintenance_type
        )
        
        cost_estimate = self._estimate_maintenance_cost(duration, crew_size, parts)
        
        return MaintenanceTask(
            task_id=task_id,
            equipment_id=equipment.equipment_id,
            maintenance_type=maintenance_type,
            priority=priority,
            estimated_duration=duration,
            required_crew_size=crew_size,
            required_skills=skills,
            spare_parts_needed=parts,
            cost_estimate=cost_estimate,
            description=f"{maintenance_type.value} maintenance for {equipment.equipment_type.value}"
        )
    
    def _get_maintenance_requirements(self, equipment_type: EquipmentType, 
                                    maintenance_type: MaintenanceType) -> Tuple[float, int, List[str], List[str]]:
        """Get maintenance requirements for equipment and maintenance type"""
        
        # Base requirements by equipment type
        base_requirements = {
            EquipmentType.QUAY_CRANE: (8.0, 3, ["mechanical", "electrical"], ["HYDRAULIC_PUMP"]),
            EquipmentType.RTG_CRANE: (6.0, 2, ["mechanical", "electrical"], ["MOTOR_CONTROLLER"]),
            EquipmentType.REACH_STACKER: (4.0, 2, ["mechanical"], ["BRAKE_PAD_SET"]),
            EquipmentType.TERMINAL_TRACTOR: (2.0, 1, ["mechanical"], ["BRAKE_PAD_SET"]),
            EquipmentType.MOBILE_CRANE: (6.0, 2, ["mechanical"], ["WIRE_ROPE"])
        }
        
        duration, crew_size, skills, parts = base_requirements.get(
            equipment_type, (4.0, 2, ["general"], [])
        )
        
        # Adjust based on maintenance type
        if maintenance_type == MaintenanceType.PREVENTIVE:
            duration *= 1.0
        elif maintenance_type == MaintenanceType.CORRECTIVE:
            duration *= 1.5
            crew_size += 1
        elif maintenance_type == MaintenanceType.EMERGENCY:
            duration *= 2.0
            crew_size += 1
            skills.append("emergency")
        elif maintenance_type == MaintenanceType.OVERHAUL:
            duration *= 3.0
            crew_size += 2
        elif maintenance_type == MaintenanceType.INSPECTION:
            duration *= 0.3
            parts = []  # No parts needed for inspection
        
        return duration, crew_size, skills, parts
    
    def _estimate_maintenance_cost(self, duration: float, crew_size: int, parts: List[str]) -> float:
        """Estimate maintenance cost"""
        # Labor cost (average hourly rate)
        labor_cost = duration * crew_size * 65.0  # Average HKD 65/hour
        
        # Parts cost
        parts_cost = sum(self.spare_parts[part].unit_cost for part in parts 
                        if part in self.spare_parts)
        
        # Overhead (20% of labor + parts)
        overhead = (labor_cost + parts_cost) * 0.2
        
        return labor_cost + parts_cost + overhead
    
    def _calculate_task_priority(self, task: MaintenanceTask) -> float:
        """Calculate task priority score (lower = higher priority)"""
        base_priority = task.priority.value
        
        equipment = self.equipment[task.equipment_id]
        
        # Urgency based on equipment condition
        condition_factor = (100 - equipment.condition_score) / 10
        
        # Criticality based on equipment type
        criticality_factor = {
            EquipmentType.QUAY_CRANE: -5,
            EquipmentType.RTG_CRANE: -3,
            EquipmentType.REACH_STACKER: -1,
            EquipmentType.TERMINAL_TRACTOR: 0,
            EquipmentType.MOBILE_CRANE: -2
        }.get(equipment.equipment_type, 0)
        
        # Overdue maintenance penalty
        overdue_factor = 0
        if equipment.next_scheduled_maintenance and equipment.next_scheduled_maintenance < self.env.now:
            overdue_factor = -10
        
        return base_priority + condition_factor + criticality_factor + overdue_factor
    
    def _maintenance_scheduler_process(self):
        """Background process to schedule and execute maintenance tasks"""
        while True:
            # Handle emergency tasks first
            if self.emergency_tasks:
                emergency_task = self.emergency_tasks.popleft()
                self.env.process(self._execute_maintenance_task(emergency_task))
            
            # Handle regular tasks
            elif self.pending_tasks and len(self.active_tasks) < self.max_concurrent_tasks:
                # Check if we're in maintenance window for non-emergency tasks
                current_hour = self.env.now % 24
                in_maintenance_window = (
                    current_hour >= self.maintenance_window_start or 
                    current_hour <= self.maintenance_window_end
                )
                
                if in_maintenance_window:
                    priority_score, task_id, task = heapq.heappop(self.pending_tasks)
                    
                    # Check if resources are available
                    if self._check_resource_availability(task):
                        self.env.process(self._execute_maintenance_task(task))
                    else:
                        # Put task back in queue
                        heapq.heappush(self.pending_tasks, (priority_score, task_id, task))
            
            # Wait before next scheduling cycle
            yield self.env.timeout(0.25)  # Check every 15 minutes
    
    def _check_resource_availability(self, task: MaintenanceTask) -> bool:
        """Check if required resources are available for task"""
        # Check crew availability
        available_crew = None
        for crew in self.maintenance_crews.values():
            if (crew.is_available and 
                crew.crew_size >= task.required_crew_size and
                all(skill in crew.skills for skill in task.required_skills)):
                available_crew = crew
                break
        
        if not available_crew:
            return False
        
        # Check spare parts availability
        for part in task.spare_parts_needed:
            if part in self.spare_parts and self.spare_parts[part].current_stock <= 0:
                return False
        
        return True
    
    def _execute_maintenance_task(self, task: MaintenanceTask):
        """Execute a maintenance task"""
        start_time = self.env.now
        task.actual_start = start_time
        
        # Find and allocate crew
        crew = self._allocate_crew(task)
        if not crew:
            logger.warning(f"No crew available for task {task.task_id}")
            return
        
        # Allocate spare parts
        self._allocate_spare_parts(task)
        
        # Update equipment status
        equipment = self.equipment[task.equipment_id]
        equipment.status = EquipmentStatus.MAINTENANCE
        
        # Add to active tasks
        self.active_tasks[task.task_id] = task
        
        try:
            # Simulate maintenance work
            actual_duration = task.estimated_duration * random.uniform(0.8, 1.3)
            yield self.env.timeout(actual_duration)
            
            # Task completed
            task.actual_end = self.env.now
            task.actual_cost = self._calculate_actual_cost(task, crew, actual_duration)
            
            # Update equipment
            self._update_equipment_after_maintenance(equipment, task)
            
            # Record completion
            self.completed_tasks.append(task)
            
            logger.info(f"Completed maintenance task {task.task_id} in {actual_duration:.1f} hours")
            
        except Exception as e:
            logger.error(f"Maintenance task {task.task_id} failed: {e}")
        
        finally:
            # Clean up resources
            crew.is_available = True
            crew.current_task = None
            equipment.status = EquipmentStatus.OPERATIONAL
            
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    def _allocate_crew(self, task: MaintenanceTask) -> Optional[MaintenanceCrew]:
        """Allocate crew for maintenance task"""
        for crew in self.maintenance_crews.values():
            if (crew.is_available and 
                crew.crew_size >= task.required_crew_size and
                all(skill in crew.skills for skill in task.required_skills)):
                
                crew.is_available = False
                crew.current_task = task.task_id
                return crew
        
        return None
    
    def _allocate_spare_parts(self, task: MaintenanceTask):
        """Allocate spare parts for maintenance task"""
        for part in task.spare_parts_needed:
            if part in self.spare_parts:
                self.spare_parts[part].current_stock -= 1
    
    def _calculate_actual_cost(self, task: MaintenanceTask, crew: MaintenanceCrew, duration: float) -> float:
        """Calculate actual maintenance cost"""
        labor_cost = duration * crew.crew_size * crew.hourly_rate
        
        parts_cost = sum(self.spare_parts[part].unit_cost for part in task.spare_parts_needed 
                        if part in self.spare_parts)
        
        overhead = (labor_cost + parts_cost) * 0.2
        
        return labor_cost + parts_cost + overhead
    
    def _update_equipment_after_maintenance(self, equipment: Equipment, task: MaintenanceTask):
        """Update equipment state after maintenance"""
        equipment.last_maintenance = self.env.now
        equipment.maintenance_cost_total += task.actual_cost
        
        if task.actual_start and task.actual_end:
            equipment.downtime_total += task.actual_end - task.actual_start
        
        # Improve condition based on maintenance type
        if task.maintenance_type == MaintenanceType.PREVENTIVE:
            equipment.condition_score = min(100.0, equipment.condition_score + 10)
            equipment.next_scheduled_maintenance = self.env.now + equipment.maintenance_interval
        elif task.maintenance_type == MaintenanceType.CORRECTIVE:
            equipment.condition_score = min(100.0, equipment.condition_score + 15)
        elif task.maintenance_type == MaintenanceType.OVERHAUL:
            equipment.condition_score = 95.0
            equipment.next_scheduled_maintenance = self.env.now + equipment.maintenance_interval * 2
    
    def _equipment_monitoring_process(self):
        """Background process to monitor equipment and schedule preventive maintenance"""
        while True:
            for equipment in self.equipment.values():
                # Check if preventive maintenance is due
                if (equipment.next_scheduled_maintenance and 
                    equipment.next_scheduled_maintenance <= self.env.now and
                    equipment.status == EquipmentStatus.OPERATIONAL):
                    
                    self.schedule_maintenance(
                        equipment.equipment_id, 
                        MaintenanceType.PREVENTIVE,
                        MaintenancePriority.MEDIUM
                    )
                
                # Check for potential failures based on condition
                if equipment.condition_score < 30 and equipment.status == EquipmentStatus.OPERATIONAL:
                    # High risk of failure - schedule corrective maintenance
                    self.schedule_maintenance(
                        equipment.equipment_id,
                        MaintenanceType.CORRECTIVE,
                        MaintenancePriority.HIGH
                    )
            
            # Check every 4 hours
            yield self.env.timeout(4.0)
    
    def _condition_monitoring_process(self):
        """Background process to simulate equipment condition degradation"""
        while True:
            for equipment in self.equipment.values():
                if equipment.status == EquipmentStatus.OPERATIONAL:
                    # Simulate condition degradation
                    degradation_rate = self._calculate_degradation_rate(equipment)
                    equipment.condition_score = max(0.0, equipment.condition_score - degradation_rate)
                    
                    # Update operating hours
                    equipment.operating_hours += 1.0
                    
                    # Random failure chance
                    failure_probability = self._calculate_failure_probability(equipment)
                    if random.random() < failure_probability:
                        equipment.status = EquipmentStatus.BREAKDOWN
                        equipment.failure_count += 1
                        
                        # Schedule emergency maintenance
                        self.schedule_maintenance(
                            equipment.equipment_id,
                            MaintenanceType.EMERGENCY,
                            MaintenancePriority.CRITICAL
                        )
                        
                        logger.warning(f"Equipment {equipment.equipment_id} has broken down")
            
            # Update every hour
            yield self.env.timeout(1.0)
    
    def _calculate_degradation_rate(self, equipment: Equipment) -> float:
        """Calculate equipment condition degradation rate"""
        base_rate = 0.1  # Base degradation per hour
        
        # Age factor
        age_factor = equipment.operating_hours / equipment.expected_lifetime
        
        # Usage intensity factor
        usage_factor = equipment.operating_hours / equipment.max_operating_hours
        
        # Maintenance factor
        time_since_maintenance = self.env.now - equipment.last_maintenance
        maintenance_factor = min(2.0, time_since_maintenance / equipment.maintenance_interval)
        
        return base_rate * (1 + age_factor + usage_factor) * maintenance_factor
    
    def _calculate_failure_probability(self, equipment: Equipment) -> float:
        """Calculate probability of equipment failure"""
        base_probability = 0.0001  # 0.01% per hour
        
        # Condition factor
        condition_factor = (100 - equipment.condition_score) / 100
        
        # Age factor
        age_factor = equipment.operating_hours / equipment.expected_lifetime
        
        return base_probability * (1 + condition_factor * 10 + age_factor * 5)
    
    def get_maintenance_status(self) -> Dict[str, Any]:
        """Get current maintenance system status"""
        total_equipment = len(self.equipment)
        operational_equipment = sum(1 for eq in self.equipment.values() 
                                  if eq.status == EquipmentStatus.OPERATIONAL)
        
        equipment_availability = (operational_equipment / total_equipment) * 100 if total_equipment else 0
        
        return {
            'total_equipment': total_equipment,
            'operational_equipment': operational_equipment,
            'equipment_availability': equipment_availability,
            'pending_tasks': len(self.pending_tasks),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'emergency_tasks': len(self.emergency_tasks),
            'performance_metrics': self.metrics
        }
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        # Equipment availability
        total_equipment = len(self.equipment)
        operational_equipment = sum(1 for eq in self.equipment.values() 
                                  if eq.status == EquipmentStatus.OPERATIONAL)
        self.metrics['equipment_availability'] = (operational_equipment / total_equipment) * 100 if total_equipment else 0
        
        # Mean time between failures
        if self.equipment:
            total_operating_hours = sum(eq.operating_hours for eq in self.equipment.values())
            total_failures = sum(eq.failure_count for eq in self.equipment.values())
            self.metrics['mean_time_between_failures'] = total_operating_hours / total_failures if total_failures else float('inf')
        
        # Mean time to repair
        if self.completed_tasks:
            repair_times = [task.actual_end - task.actual_start for task in self.completed_tasks 
                          if task.actual_start and task.actual_end and task.maintenance_type in 
                          [MaintenanceType.CORRECTIVE, MaintenanceType.EMERGENCY]]
            self.metrics['mean_time_to_repair'] = np.mean(repair_times) if repair_times else 0
        
        # Maintenance cost per hour
        if self.env.now > 0:
            total_cost = sum(eq.maintenance_cost_total for eq in self.equipment.values())
            self.metrics['maintenance_cost_per_hour'] = total_cost / self.env.now
        
        # Preventive maintenance ratio
        if self.completed_tasks:
            preventive_tasks = sum(1 for task in self.completed_tasks 
                                 if task.maintenance_type == MaintenanceType.PREVENTIVE)
            self.metrics['preventive_maintenance_ratio'] = (preventive_tasks / len(self.completed_tasks)) * 100
        
        # Crew utilization
        busy_crews = sum(1 for crew in self.maintenance_crews.values() if not crew.is_available)
        self.metrics['crew_utilization'] = (busy_crews / len(self.maintenance_crews)) * 100 if self.maintenance_crews else 0