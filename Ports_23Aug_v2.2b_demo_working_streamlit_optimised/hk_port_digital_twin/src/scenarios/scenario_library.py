"""Advanced Scenario Library for Hong Kong Port Digital Twin

This module provides a comprehensive library of pre-configured scenarios,
scenario templates, and scenario management capabilities:
- Pre-built scenario templates for common operational situations
- Scenario saving and loading functionality
- Scenario categorization and search
- Scenario validation and parameter checking
- Scenario composition and combination
- Performance benchmarking scenarios

The library supports various scenario types including:
- Operational scenarios (peak season, maintenance, disruptions)
- Performance testing scenarios (stress tests, capacity limits)
- Training scenarios (staff training, system familiarization)
- Research scenarios (optimization studies, what-if analysis)
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
import copy
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class ScenarioCategory(Enum):
    """Categories of simulation scenarios"""
    OPERATIONAL = "operational"
    PERFORMANCE = "performance"
    TRAINING = "training"
    RESEARCH = "research"
    DISRUPTION = "disruption"
    OPTIMIZATION = "optimization"
    STRESS_TEST = "stress_test"
    BENCHMARK = "benchmark"

class ScenarioComplexity(Enum):
    """Complexity levels of scenarios"""
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4

class ScenarioStatus(Enum):
    """Status of scenario execution"""
    DRAFT = "draft"
    VALIDATED = "validated"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

@dataclass
class ScenarioParameter:
    """Individual scenario parameter definition"""
    name: str
    parameter_type: str  # "int", "float", "str", "bool", "list", "dict"
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    unit: str = ""
    required: bool = True
    
@dataclass
class ScenarioTemplate:
    """Template for creating simulation scenarios"""
    template_id: str
    name: str
    description: str
    category: ScenarioCategory
    complexity: ScenarioComplexity
    parameters: Dict[str, ScenarioParameter]
    default_duration: float = 24.0  # hours
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    author: str = "System"
    
@dataclass
class ScenarioInstance:
    """Instance of a scenario with specific parameter values"""
    instance_id: str
    template_id: str
    name: str
    description: str
    category: ScenarioCategory
    parameter_values: Dict[str, Any]
    duration: float
    status: ScenarioStatus = ScenarioStatus.DRAFT
    created_date: datetime = field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
@dataclass
class ScenarioCollection:
    """Collection of related scenarios"""
    collection_id: str
    name: str
    description: str
    scenarios: List[str]  # List of scenario instance IDs
    execution_order: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

class AdvancedScenarioLibrary:
    """Advanced Scenario Library System
    
    Provides comprehensive scenario management including templates,
    instances, collections, and advanced scenario operations.
    """
    
    def __init__(self, library_path: str = "scenarios"):
        """Initialize the scenario library
        
        Args:
            library_path: Path to store scenario files
        """
        self.library_path = Path(library_path)
        self.library_path.mkdir(exist_ok=True)
        
        # Initialize storage
        self.templates: Dict[str, ScenarioTemplate] = {}
        self.instances: Dict[str, ScenarioInstance] = {}
        self.collections: Dict[str, ScenarioCollection] = {}
        
        # Initialize built-in templates
        self._initialize_builtin_templates()
        
        # Load existing scenarios
        self._load_library()
    
    def _initialize_builtin_templates(self):
        """Initialize built-in scenario templates"""
        
        # Peak Season Operations
        peak_season_params = {
            "ship_arrival_rate": ScenarioParameter(
                name="ship_arrival_rate",
                parameter_type="float",
                default_value=2.5,
                min_value=1.0,
                max_value=5.0,
                description="Ships arriving per hour during peak season",
                unit="ships/hour",
                required=True
            ),
            "container_volume_multiplier": ScenarioParameter(
                name="container_volume_multiplier",
                parameter_type="float",
                default_value=1.8,
                min_value=1.0,
                max_value=3.0,
                description="Container volume multiplier vs normal operations",
                unit="multiplier",
                required=True
            ),
            "berth_utilization_target": ScenarioParameter(
                name="berth_utilization_target",
                parameter_type="float",
                default_value=0.95,
                min_value=0.7,
                max_value=1.0,
                description="Target berth utilization rate",
                unit="percentage",
                required=True
            ),
            "overtime_enabled": ScenarioParameter(
                name="overtime_enabled",
                parameter_type="bool",
                default_value=True,
                description="Enable overtime operations",
                required=True
            )
        }
        
        self.templates["PEAK_SEASON"] = ScenarioTemplate(
            template_id="PEAK_SEASON",
            name="Peak Season Operations",
            description="High-volume operations during peak shipping season",
            category=ScenarioCategory.OPERATIONAL,
            complexity=ScenarioComplexity.INTERMEDIATE,
            parameters=peak_season_params,
            default_duration=168.0,  # 1 week
            tags=["peak", "high-volume", "seasonal"],
            expected_outcomes=[
                "High berth utilization",
                "Increased container throughput",
                "Extended operating hours",
                "Resource optimization"
            ]
        )
        
        # Equipment Maintenance Scenario
        maintenance_params = {
            "maintenance_duration": ScenarioParameter(
                name="maintenance_duration",
                parameter_type="float",
                default_value=8.0,
                min_value=2.0,
                max_value=24.0,
                description="Duration of maintenance window",
                unit="hours",
                required=True
            ),
            "equipment_count": ScenarioParameter(
                name="equipment_count",
                parameter_type="int",
                default_value=3,
                min_value=1,
                max_value=10,
                description="Number of equipment units under maintenance",
                unit="units",
                required=True
            ),
            "maintenance_type": ScenarioParameter(
                name="maintenance_type",
                parameter_type="str",
                default_value="preventive",
                allowed_values=["preventive", "corrective", "emergency"],
                description="Type of maintenance operation",
                required=True
            ),
            "backup_equipment_available": ScenarioParameter(
                name="backup_equipment_available",
                parameter_type="bool",
                default_value=True,
                description="Availability of backup equipment",
                required=True
            )
        }
        
        self.templates["EQUIPMENT_MAINTENANCE"] = ScenarioTemplate(
            template_id="EQUIPMENT_MAINTENANCE",
            name="Equipment Maintenance Operations",
            description="Planned maintenance operations with capacity adjustments",
            category=ScenarioCategory.OPERATIONAL,
            complexity=ScenarioComplexity.BASIC,
            parameters=maintenance_params,
            default_duration=24.0,
            tags=["maintenance", "equipment", "planned"],
            expected_outcomes=[
                "Reduced operational capacity",
                "Equipment reliability improvement",
                "Optimized maintenance scheduling"
            ]
        )
        
        # Typhoon Disruption Scenario
        typhoon_params = {
            "typhoon_category": ScenarioParameter(
                name="typhoon_category",
                parameter_type="int",
                default_value=3,
                min_value=1,
                max_value=5,
                description="Typhoon category (1-5)",
                unit="category",
                required=True
            ),
            "wind_speed": ScenarioParameter(
                name="wind_speed",
                parameter_type="float",
                default_value=150.0,
                min_value=60.0,
                max_value=250.0,
                description="Maximum wind speed",
                unit="km/h",
                required=True
            ),
            "duration": ScenarioParameter(
                name="duration",
                parameter_type="float",
                default_value=48.0,
                min_value=12.0,
                max_value=120.0,
                description="Typhoon duration",
                unit="hours",
                required=True
            ),
            "advance_warning": ScenarioParameter(
                name="advance_warning",
                parameter_type="float",
                default_value=72.0,
                min_value=24.0,
                max_value=168.0,
                description="Advance warning time",
                unit="hours",
                required=True
            )
        }
        
        self.templates["TYPHOON_DISRUPTION"] = ScenarioTemplate(
            template_id="TYPHOON_DISRUPTION",
            name="Typhoon Disruption Response",
            description="Port operations during typhoon conditions",
            category=ScenarioCategory.DISRUPTION,
            complexity=ScenarioComplexity.ADVANCED,
            parameters=typhoon_params,
            default_duration=120.0,  # 5 days including recovery
            tags=["typhoon", "weather", "emergency", "disruption"],
            prerequisites=["Emergency response plan", "Weather monitoring system"],
            expected_outcomes=[
                "Safe shutdown procedures",
                "Minimal equipment damage",
                "Rapid recovery operations",
                "Supply chain continuity"
            ]
        )
        
        # Capacity Stress Test
        stress_test_params = {
            "load_multiplier": ScenarioParameter(
                name="load_multiplier",
                parameter_type="float",
                default_value=2.0,
                min_value=1.5,
                max_value=5.0,
                description="Load multiplier vs normal capacity",
                unit="multiplier",
                required=True
            ),
            "test_duration": ScenarioParameter(
                name="test_duration",
                parameter_type="float",
                default_value=12.0,
                min_value=4.0,
                max_value=48.0,
                description="Duration of stress test",
                unit="hours",
                required=True
            ),
            "failure_threshold": ScenarioParameter(
                name="failure_threshold",
                parameter_type="float",
                default_value=0.95,
                min_value=0.8,
                max_value=1.0,
                description="System failure threshold",
                unit="percentage",
                required=True
            ),
            "monitor_intervals": ScenarioParameter(
                name="monitor_intervals",
                parameter_type="float",
                default_value=0.5,
                min_value=0.1,
                max_value=2.0,
                description="Monitoring interval",
                unit="hours",
                required=True
            )
        }
        
        self.templates["CAPACITY_STRESS_TEST"] = ScenarioTemplate(
            template_id="CAPACITY_STRESS_TEST",
            name="Port Capacity Stress Test",
            description="Test port operations under extreme load conditions",
            category=ScenarioCategory.STRESS_TEST,
            complexity=ScenarioComplexity.EXPERT,
            parameters=stress_test_params,
            default_duration=24.0,
            tags=["stress-test", "capacity", "performance", "limits"],
            expected_outcomes=[
                "Capacity limits identification",
                "Bottleneck analysis",
                "Performance degradation patterns",
                "System resilience assessment"
            ]
        )
        
        # AI Optimization Benchmark
        ai_benchmark_params = {
            "optimization_algorithm": ScenarioParameter(
                name="optimization_algorithm",
                parameter_type="str",
                default_value="genetic_algorithm",
                allowed_values=["genetic_algorithm", "simulated_annealing", "particle_swarm", "reinforcement_learning"],
                description="AI optimization algorithm to test",
                required=True
            ),
            "optimization_target": ScenarioParameter(
                name="optimization_target",
                parameter_type="str",
                default_value="throughput",
                allowed_values=["throughput", "efficiency", "cost", "emissions", "multi_objective"],
                description="Primary optimization target",
                required=True
            ),
            "learning_iterations": ScenarioParameter(
                name="learning_iterations",
                parameter_type="int",
                default_value=100,
                min_value=10,
                max_value=1000,
                description="Number of learning iterations",
                unit="iterations",
                required=True
            ),
            "baseline_comparison": ScenarioParameter(
                name="baseline_comparison",
                parameter_type="bool",
                default_value=True,
                description="Compare against baseline operations",
                required=True
            )
        }
        
        self.templates["AI_OPTIMIZATION_BENCHMARK"] = ScenarioTemplate(
            template_id="AI_OPTIMIZATION_BENCHMARK",
            name="AI Optimization Performance Benchmark",
            description="Benchmark AI optimization algorithms performance",
            category=ScenarioCategory.BENCHMARK,
            complexity=ScenarioComplexity.EXPERT,
            parameters=ai_benchmark_params,
            default_duration=72.0,  # 3 days
            tags=["ai", "optimization", "benchmark", "performance"],
            prerequisites=["AI optimization system", "Baseline performance data"],
            expected_outcomes=[
                "Algorithm performance comparison",
                "Optimization effectiveness metrics",
                "Convergence analysis",
                "Computational efficiency assessment"
            ]
        )
        
        # Supply Chain Disruption Analysis
        supply_chain_params = {
            "disruption_type": ScenarioParameter(
                name="disruption_type",
                parameter_type="str",
                default_value="supplier_failure",
                allowed_values=["supplier_failure", "transport_disruption", "demand_spike", "trade_restriction"],
                description="Type of supply chain disruption",
                required=True
            ),
            "disruption_severity": ScenarioParameter(
                name="disruption_severity",
                parameter_type="float",
                default_value=0.7,
                min_value=0.1,
                max_value=1.0,
                description="Severity of disruption (0.1 = minor, 1.0 = complete)",
                unit="severity",
                required=True
            ),
            "affected_routes": ScenarioParameter(
                name="affected_routes",
                parameter_type="int",
                default_value=3,
                min_value=1,
                max_value=10,
                description="Number of affected shipping routes",
                unit="routes",
                required=True
            ),
            "recovery_strategy": ScenarioParameter(
                name="recovery_strategy",
                parameter_type="str",
                default_value="alternative_routing",
                allowed_values=["alternative_routing", "inventory_buffer", "supplier_diversification", "emergency_procurement"],
                description="Recovery strategy to implement",
                required=True
            )
        }
        
        self.templates["SUPPLY_CHAIN_DISRUPTION"] = ScenarioTemplate(
            template_id="SUPPLY_CHAIN_DISRUPTION",
            name="Supply Chain Disruption Analysis",
            description="Analyze impact and recovery from supply chain disruptions",
            category=ScenarioCategory.RESEARCH,
            complexity=ScenarioComplexity.ADVANCED,
            parameters=supply_chain_params,
            default_duration=336.0,  # 2 weeks
            tags=["supply-chain", "disruption", "analysis", "recovery"],
            expected_outcomes=[
                "Disruption impact assessment",
                "Recovery time analysis",
                "Alternative strategy evaluation",
                "Resilience improvement recommendations"
            ]
        )
    
    def create_scenario_from_template(self, template_id: str, name: str, 
                                    parameter_overrides: Optional[Dict[str, Any]] = None,
                                    description: Optional[str] = None) -> str:
        """Create a scenario instance from a template
        
        Args:
            template_id: ID of the template to use
            name: Name for the new scenario instance
            parameter_overrides: Parameter values to override defaults
            description: Custom description for the instance
            
        Returns:
            ID of the created scenario instance
        """
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        instance_id = str(uuid.uuid4())
        
        # Start with default parameter values
        parameter_values = {}
        for param_name, param_def in template.parameters.items():
            parameter_values[param_name] = param_def.default_value
        
        # Apply overrides
        if parameter_overrides:
            for param_name, value in parameter_overrides.items():
                if param_name in template.parameters:
                    # Validate parameter value
                    if self._validate_parameter_value(template.parameters[param_name], value):
                        parameter_values[param_name] = value
                    else:
                        logger.warning(f"Invalid value for parameter {param_name}: {value}")
                else:
                    logger.warning(f"Unknown parameter {param_name} for template {template_id}")
        
        # Create scenario instance
        instance = ScenarioInstance(
            instance_id=instance_id,
            template_id=template_id,
            name=name,
            description=description or template.description,
            category=template.category,
            parameter_values=parameter_values,
            duration=template.default_duration,
            tags=template.tags.copy()
        )
        
        self.instances[instance_id] = instance
        logger.info(f"Created scenario instance '{name}' (ID: {instance_id}) from template {template_id}")
        
        return instance_id
    
    def _validate_parameter_value(self, param_def: ScenarioParameter, value: Any) -> bool:
        """Validate a parameter value against its definition"""
        # Type checking
        if param_def.parameter_type == "int" and not isinstance(value, int):
            return False
        elif param_def.parameter_type == "float" and not isinstance(value, (int, float)):
            return False
        elif param_def.parameter_type == "str" and not isinstance(value, str):
            return False
        elif param_def.parameter_type == "bool" and not isinstance(value, bool):
            return False
        elif param_def.parameter_type == "list" and not isinstance(value, list):
            return False
        elif param_def.parameter_type == "dict" and not isinstance(value, dict):
            return False
        
        # Range checking
        if param_def.min_value is not None and value < param_def.min_value:
            return False
        if param_def.max_value is not None and value > param_def.max_value:
            return False
        
        # Allowed values checking
        if param_def.allowed_values is not None and value not in param_def.allowed_values:
            return False
        
        return True
    
    def save_scenario(self, instance_id: str, file_path: Optional[str] = None) -> bool:
        """Save a scenario instance to file
        
        Args:
            instance_id: ID of the scenario instance to save
            file_path: Optional custom file path
            
        Returns:
            True if saved successfully
        """
        if instance_id not in self.instances:
            logger.error(f"Scenario instance {instance_id} not found")
            return False
        
        instance = self.instances[instance_id]
        
        if file_path is None:
            file_path = self.library_path / f"scenario_{instance_id}.json"
        else:
            file_path = Path(file_path)
        
        try:
            # Convert to serializable format
            scenario_data = {
                'instance': asdict(instance),
                'template': asdict(self.templates[instance.template_id]) if instance.template_id in self.templates else None
            }
            
            # Handle datetime serialization
            scenario_data['instance']['created_date'] = instance.created_date.isoformat()
            if instance.start_time:
                scenario_data['instance']['start_time'] = instance.start_time.isoformat()
            if instance.end_time:
                scenario_data['instance']['end_time'] = instance.end_time.isoformat()
            
            if scenario_data['template']:
                scenario_data['template']['created_date'] = self.templates[instance.template_id].created_date.isoformat()
            
            with open(file_path, 'w') as f:
                json.dump(scenario_data, f, indent=2, default=str)
            
            logger.info(f"Saved scenario '{instance.name}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save scenario {instance_id}: {e}")
            return False
    
    def load_scenario(self, file_path: str) -> Optional[str]:
        """Load a scenario instance from file
        
        Args:
            file_path: Path to the scenario file
            
        Returns:
            ID of the loaded scenario instance, or None if failed
        """
        try:
            with open(file_path, 'r') as f:
                scenario_data = json.load(f)
            
            # Reconstruct instance
            instance_data = scenario_data['instance']
            
            # Handle datetime deserialization
            instance_data['created_date'] = datetime.fromisoformat(instance_data['created_date'])
            if instance_data.get('start_time'):
                instance_data['start_time'] = datetime.fromisoformat(instance_data['start_time'])
            if instance_data.get('end_time'):
                instance_data['end_time'] = datetime.fromisoformat(instance_data['end_time'])
            
            # Convert enums
            instance_data['category'] = ScenarioCategory(instance_data['category'])
            instance_data['status'] = ScenarioStatus(instance_data['status'])
            
            instance = ScenarioInstance(**instance_data)
            self.instances[instance.instance_id] = instance
            
            # Load template if included and not already present
            if scenario_data.get('template') and instance.template_id not in self.templates:
                template_data = scenario_data['template']
                template_data['created_date'] = datetime.fromisoformat(template_data['created_date'])
                template_data['category'] = ScenarioCategory(template_data['category'])
                template_data['complexity'] = ScenarioComplexity(template_data['complexity'])
                
                # Reconstruct parameters
                parameters = {}
                for param_name, param_data in template_data['parameters'].items():
                    parameters[param_name] = ScenarioParameter(**param_data)
                template_data['parameters'] = parameters
                
                template = ScenarioTemplate(**template_data)
                self.templates[template.template_id] = template
            
            logger.info(f"Loaded scenario '{instance.name}' (ID: {instance.instance_id})")
            return instance.instance_id
            
        except Exception as e:
            logger.error(f"Failed to load scenario from {file_path}: {e}")
            return None
    
    def _load_library(self):
        """Load all scenarios from the library directory"""
        if not self.library_path.exists():
            return
        
        for file_path in self.library_path.glob("scenario_*.json"):
            self.load_scenario(str(file_path))
        
        logger.info(f"Loaded {len(self.instances)} scenarios from library")
    
    def search_scenarios(self, query: str = "", category: Optional[ScenarioCategory] = None,
                        complexity: Optional[ScenarioComplexity] = None,
                        tags: Optional[List[str]] = None,
                        status: Optional[ScenarioStatus] = None) -> List[str]:
        """Search for scenarios based on criteria
        
        Args:
            query: Text search in name and description
            category: Filter by scenario category
            complexity: Filter by complexity level
            tags: Filter by tags (any match)
            status: Filter by status
            
        Returns:
            List of matching scenario instance IDs
        """
        matching_instances = []
        
        for instance_id, instance in self.instances.items():
            # Text search
            if query and query.lower() not in instance.name.lower() and query.lower() not in instance.description.lower():
                continue
            
            # Category filter
            if category and instance.category != category:
                continue
            
            # Complexity filter (check template)
            if complexity and instance.template_id in self.templates:
                if self.templates[instance.template_id].complexity != complexity:
                    continue
            
            # Tags filter
            if tags and not any(tag in instance.tags for tag in tags):
                continue
            
            # Status filter
            if status and instance.status != status:
                continue
            
            matching_instances.append(instance_id)
        
        return matching_instances
    
    def create_scenario_collection(self, name: str, description: str, 
                                 scenario_ids: List[str]) -> str:
        """Create a collection of related scenarios
        
        Args:
            name: Name of the collection
            description: Description of the collection
            scenario_ids: List of scenario instance IDs to include
            
        Returns:
            ID of the created collection
        """
        collection_id = str(uuid.uuid4())
        
        # Validate scenario IDs
        valid_ids = [sid for sid in scenario_ids if sid in self.instances]
        if len(valid_ids) != len(scenario_ids):
            logger.warning(f"Some scenario IDs not found: {set(scenario_ids) - set(valid_ids)}")
        
        collection = ScenarioCollection(
            collection_id=collection_id,
            name=name,
            description=description,
            scenarios=valid_ids,
            execution_order=valid_ids.copy()
        )
        
        self.collections[collection_id] = collection
        logger.info(f"Created scenario collection '{name}' with {len(valid_ids)} scenarios")
        
        return collection_id
    
    def clone_scenario(self, instance_id: str, new_name: str, 
                      parameter_modifications: Optional[Dict[str, Any]] = None) -> str:
        """Clone an existing scenario with optional modifications
        
        Args:
            instance_id: ID of the scenario to clone
            new_name: Name for the cloned scenario
            parameter_modifications: Parameter changes for the clone
            
        Returns:
            ID of the cloned scenario
        """
        if instance_id not in self.instances:
            raise ValueError(f"Scenario {instance_id} not found")
        
        original = self.instances[instance_id]
        clone_id = str(uuid.uuid4())
        
        # Create clone with modified parameters
        clone_parameters = original.parameter_values.copy()
        if parameter_modifications:
            clone_parameters.update(parameter_modifications)
        
        clone = ScenarioInstance(
            instance_id=clone_id,
            template_id=original.template_id,
            name=new_name,
            description=f"Clone of {original.name}: {original.description}",
            category=original.category,
            parameter_values=clone_parameters,
            duration=original.duration,
            tags=original.tags.copy() + ["cloned"]
        )
        
        self.instances[clone_id] = clone
        logger.info(f"Cloned scenario '{original.name}' as '{new_name}' (ID: {clone_id})")
        
        return clone_id
    
    def get_scenario_summary(self, instance_id: str) -> Dict[str, Any]:
        """Get a summary of a scenario instance
        
        Args:
            instance_id: ID of the scenario instance
            
        Returns:
            Dictionary containing scenario summary
        """
        if instance_id not in self.instances:
            return {}
        
        instance = self.instances[instance_id]
        template = self.templates.get(instance.template_id)
        
        summary = {
            'instance_id': instance.instance_id,
            'name': instance.name,
            'description': instance.description,
            'category': instance.category.value,
            'status': instance.status.value,
            'duration': instance.duration,
            'created_date': instance.created_date.isoformat(),
            'tags': instance.tags,
            'parameter_count': len(instance.parameter_values),
            'has_results': bool(instance.results),
            'metrics_count': len(instance.metrics)
        }
        
        if template:
            summary.update({
                'template_name': template.name,
                'complexity': template.complexity.value,
                'expected_outcomes': template.expected_outcomes
            })
        
        if instance.start_time:
            summary['start_time'] = instance.start_time.isoformat()
        if instance.end_time:
            summary['end_time'] = instance.end_time.isoformat()
            summary['actual_duration'] = (instance.end_time - instance.start_time).total_seconds() / 3600
        
        return summary
    
    def get_library_statistics(self) -> Dict[str, Any]:
        """Get statistics about the scenario library
        
        Returns:
            Dictionary containing library statistics
        """
        # Count by category
        category_counts = {}
        for instance in self.instances.values():
            category = instance.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by status
        status_counts = {}
        for instance in self.instances.values():
            status = instance.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by complexity (from templates)
        complexity_counts = {}
        for instance in self.instances.values():
            if instance.template_id in self.templates:
                complexity = self.templates[instance.template_id].complexity.value
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        # Calculate average duration
        durations = [instance.duration for instance in self.instances.values()]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'total_templates': len(self.templates),
            'total_instances': len(self.instances),
            'total_collections': len(self.collections),
            'by_category': category_counts,
            'by_status': status_counts,
            'by_complexity': complexity_counts,
            'average_duration_hours': avg_duration,
            'library_path': str(self.library_path)
        }
    
    def export_scenario_catalog(self, output_path: str) -> bool:
        """Export a catalog of all scenarios and templates
        
        Args:
            output_path: Path for the catalog file
            
        Returns:
            True if exported successfully
        """
        try:
            catalog = {
                'generated_date': datetime.now().isoformat(),
                'library_statistics': self.get_library_statistics(),
                'templates': {},
                'instances': {},
                'collections': {}
            }
            
            # Export templates
            for template_id, template in self.templates.items():
                catalog['templates'][template_id] = {
                    'name': template.name,
                    'description': template.description,
                    'category': template.category.value,
                    'complexity': template.complexity.value,
                    'default_duration': template.default_duration,
                    'tags': template.tags,
                    'parameter_count': len(template.parameters),
                    'expected_outcomes': template.expected_outcomes
                }
            
            # Export instances
            for instance_id, instance in self.instances.items():
                catalog['instances'][instance_id] = self.get_scenario_summary(instance_id)
            
            # Export collections
            for collection_id, collection in self.collections.items():
                catalog['collections'][collection_id] = {
                    'name': collection.name,
                    'description': collection.description,
                    'scenario_count': len(collection.scenarios),
                    'created_date': collection.created_date.isoformat(),
                    'tags': collection.tags
                }
            
            with open(output_path, 'w') as f:
                json.dump(catalog, f, indent=2)
            
            logger.info(f"Exported scenario catalog to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export catalog: {e}")
            return False
    
    def validate_scenario(self, instance_id: str) -> Tuple[bool, List[str]]:
        """Validate a scenario instance
        
        Args:
            instance_id: ID of the scenario to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if instance_id not in self.instances:
            return False, [f"Scenario {instance_id} not found"]
        
        instance = self.instances[instance_id]
        issues = []
        
        # Check if template exists
        if instance.template_id not in self.templates:
            issues.append(f"Template {instance.template_id} not found")
            return False, issues
        
        template = self.templates[instance.template_id]
        
        # Validate all required parameters are present
        for param_name, param_def in template.parameters.items():
            if param_def.required and param_name not in instance.parameter_values:
                issues.append(f"Required parameter '{param_name}' is missing")
        
        # Validate parameter values
        for param_name, value in instance.parameter_values.items():
            if param_name in template.parameters:
                param_def = template.parameters[param_name]
                if not self._validate_parameter_value(param_def, value):
                    issues.append(f"Invalid value for parameter '{param_name}': {value}")
            else:
                issues.append(f"Unknown parameter '{param_name}' not in template")
        
        # Check duration
        if instance.duration <= 0:
            issues.append("Duration must be positive")
        
        is_valid = len(issues) == 0
        if is_valid:
            instance.status = ScenarioStatus.VALIDATED
        
        return is_valid, issues