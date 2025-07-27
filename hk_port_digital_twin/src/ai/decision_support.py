# Comments for context:
# This module implements decision support systems for Hong Kong Port Digital Twin.
# The goal is to provide intelligent recommendations for port operations,
# resource allocation, and strategic planning based on AI analysis.
# 
# Approach: Combining optimization algorithms, predictive models, and rule-based
# systems to generate actionable insights and recommendations.

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from enum import Enum

# Import our AI modules
from .optimization import BerthAllocationOptimizer, ContainerHandlingScheduler, ResourceAllocationOptimizer
from .predictive_models import ShipArrivalPredictor, ProcessingTimeEstimator, QueueLengthForecaster

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    """Types of recommendations the system can provide"""
    BERTH_ALLOCATION = "berth_allocation"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    SCHEDULE_ADJUSTMENT = "schedule_adjustment"
    CAPACITY_PLANNING = "capacity_planning"
    EMERGENCY_RESPONSE = "emergency_response"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"

class Priority(Enum):
    """Priority levels for recommendations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Recommendation:
    """A single recommendation from the decision support system"""
    id: str
    type: RecommendationType
    priority: Priority
    title: str
    description: str
    rationale: str
    expected_impact: Dict[str, float]  # metrics and their expected improvement
    implementation_steps: List[str]
    estimated_cost: float
    estimated_savings: float
    timeline: str
    confidence: float
    created_at: datetime
    expires_at: Optional[datetime] = None
    dependencies: List[str] = None  # IDs of other recommendations
    
@dataclass
class DecisionContext:
    """Context information for decision making"""
    current_time: datetime
    port_status: Dict[str, Any]
    active_ships: List[Dict]
    available_berths: List[Dict]
    resource_utilization: Dict[str, float]
    weather_conditions: Dict[str, Any]
    operational_constraints: List[str]
    performance_metrics: Dict[str, float]

class DecisionSupportEngine:
    """Main decision support engine that generates recommendations"""
    
    def __init__(self):
        # Initialize AI components
        self.berth_optimizer = BerthAllocationOptimizer()
        self.container_scheduler = ContainerHandlingScheduler()
        self.resource_optimizer = ResourceAllocationOptimizer()
        self.arrival_predictor = ShipArrivalPredictor()
        self.processing_estimator = ProcessingTimeEstimator()
        self.queue_forecaster = QueueLengthForecaster()
        
        # Recommendation tracking
        self.active_recommendations = []
        self.recommendation_history = []
        self.performance_metrics = {}
        
        # Configuration
        self.recommendation_rules = self._initialize_rules()
        self.thresholds = self._initialize_thresholds()
        
    def _initialize_rules(self) -> Dict[str, Dict]:
        """Initialize decision rules and criteria"""
        return {
            'berth_allocation': {
                'queue_length_threshold': 5,
                'waiting_time_threshold': 4.0,  # hours
                'utilization_threshold': 0.85
            },
            'resource_optimization': {
                'crane_utilization_threshold': 0.9,
                'truck_utilization_threshold': 0.8,
                'worker_utilization_threshold': 0.85
            },
            'capacity_planning': {
                'forecast_horizon': 72,  # hours
                'capacity_buffer': 0.2,  # 20% buffer
                'peak_threshold': 1.5  # 150% of average
            },
            'emergency_response': {
                'critical_waiting_time': 8.0,  # hours
                'severe_weather_threshold': 0.7,
                'equipment_failure_impact': 0.3
            }
        }
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize performance thresholds"""
        return {
            'berth_utilization': 0.85,
            'average_waiting_time': 3.0,
            'throughput_efficiency': 0.8,
            'cost_per_container': 150.0,
            'customer_satisfaction': 0.9,
            'equipment_availability': 0.95
        }
    
    def analyze_situation(self, context: DecisionContext) -> List[Recommendation]:
        """Analyze current situation and generate recommendations"""
        recommendations = []
        
        try:
            # Analyze different aspects of port operations
            berth_recommendations = self._analyze_berth_allocation(context)
            resource_recommendations = self._analyze_resource_utilization(context)
            capacity_recommendations = self._analyze_capacity_planning(context)
            performance_recommendations = self._analyze_performance_issues(context)
            emergency_recommendations = self._analyze_emergency_situations(context)
            
            # Combine all recommendations
            all_recommendations = (
                berth_recommendations + 
                resource_recommendations + 
                capacity_recommendations + 
                performance_recommendations + 
                emergency_recommendations
            )
            
            # Prioritize and filter recommendations
            recommendations = self._prioritize_recommendations(all_recommendations, context)
            
            # Update tracking
            self.active_recommendations.extend(recommendations)
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Error in situation analysis: {e}")
            
        return recommendations
    
    def _analyze_berth_allocation(self, context: DecisionContext) -> List[Recommendation]:
        """Analyze berth allocation and generate recommendations"""
        recommendations = []
        
        try:
            # Check queue length
            queue_length = len(context.active_ships)
            if queue_length > self.recommendation_rules['berth_allocation']['queue_length_threshold']:
                recommendations.append(Recommendation(
                    id=f"berth_queue_{context.current_time.strftime('%Y%m%d_%H%M%S')}",
                    type=RecommendationType.BERTH_ALLOCATION,
                    priority=Priority.HIGH,
                    title="Optimize Berth Allocation for Long Queue",
                    description=f"Current queue has {queue_length} ships waiting. Recommend optimizing berth allocation to reduce waiting times.",
                    rationale=f"Queue length ({queue_length}) exceeds threshold ({self.recommendation_rules['berth_allocation']['queue_length_threshold']})",
                    expected_impact={
                        'waiting_time_reduction': 0.25,
                        'throughput_increase': 0.15,
                        'customer_satisfaction': 0.1
                    },
                    implementation_steps=[
                        "Run berth allocation optimization algorithm",
                        "Reassign ships to optimal berths",
                        "Update port management system",
                        "Notify ship operators of changes"
                    ],
                    estimated_cost=5000.0,
                    estimated_savings=25000.0,
                    timeline="2-4 hours",
                    confidence=0.8,
                    created_at=context.current_time
                ))
            
            # Check berth utilization
            berth_utilization = context.resource_utilization.get('berths', 0.0)
            if berth_utilization > self.recommendation_rules['berth_allocation']['utilization_threshold']:
                recommendations.append(Recommendation(
                    id=f"berth_util_{context.current_time.strftime('%Y%m%d_%H%M%S')}",
                    type=RecommendationType.CAPACITY_PLANNING,
                    priority=Priority.MEDIUM,
                    title="Consider Additional Berth Capacity",
                    description=f"Berth utilization is {berth_utilization:.1%}, approaching maximum capacity.",
                    rationale=f"High berth utilization may lead to increased waiting times and reduced flexibility",
                    expected_impact={
                        'capacity_increase': 0.2,
                        'waiting_time_reduction': 0.3,
                        'revenue_increase': 0.15
                    },
                    implementation_steps=[
                        "Analyze berth expansion feasibility",
                        "Evaluate temporary berth solutions",
                        "Consider operational hour extensions",
                        "Implement dynamic berth scheduling"
                    ],
                    estimated_cost=500000.0,
                    estimated_savings=200000.0,
                    timeline="3-6 months",
                    confidence=0.7,
                    created_at=context.current_time
                ))
                
        except Exception as e:
            logger.error(f"Error in berth allocation analysis: {e}")
            
        return recommendations
    
    def _analyze_resource_utilization(self, context: DecisionContext) -> List[Recommendation]:
        """Analyze resource utilization and generate recommendations"""
        recommendations = []
        
        try:
            # Check crane utilization
            crane_util = context.resource_utilization.get('cranes', 0.0)
            if crane_util > self.recommendation_rules['resource_optimization']['crane_utilization_threshold']:
                recommendations.append(Recommendation(
                    id=f"crane_opt_{context.current_time.strftime('%Y%m%d_%H%M%S')}",
                    type=RecommendationType.RESOURCE_OPTIMIZATION,
                    priority=Priority.HIGH,
                    title="Optimize Crane Allocation",
                    description=f"Crane utilization is {crane_util:.1%}, indicating potential bottleneck.",
                    rationale="High crane utilization may cause delays in container handling operations",
                    expected_impact={
                        'handling_efficiency': 0.2,
                        'processing_time_reduction': 0.15,
                        'cost_reduction': 0.1
                    },
                    implementation_steps=[
                        "Redistribute cranes across berths",
                        "Implement dynamic crane scheduling",
                        "Consider additional crane deployment",
                        "Optimize crane maintenance schedules"
                    ],
                    estimated_cost=10000.0,
                    estimated_savings=50000.0,
                    timeline="1-2 days",
                    confidence=0.85,
                    created_at=context.current_time
                ))
            
            # Check overall resource balance
            resource_imbalance = self._calculate_resource_imbalance(context.resource_utilization)
            if resource_imbalance > 0.3:  # 30% imbalance threshold
                recommendations.append(Recommendation(
                    id=f"resource_balance_{context.current_time.strftime('%Y%m%d_%H%M%S')}",
                    type=RecommendationType.RESOURCE_OPTIMIZATION,
                    priority=Priority.MEDIUM,
                    title="Rebalance Resource Allocation",
                    description=f"Resource utilization is imbalanced (imbalance score: {resource_imbalance:.2f})",
                    rationale="Unbalanced resource utilization leads to inefficiencies and bottlenecks",
                    expected_impact={
                        'overall_efficiency': 0.15,
                        'cost_reduction': 0.12,
                        'throughput_increase': 0.1
                    },
                    implementation_steps=[
                        "Analyze current resource allocation patterns",
                        "Identify underutilized resources",
                        "Redistribute resources to high-demand areas",
                        "Implement dynamic resource allocation system"
                    ],
                    estimated_cost=15000.0,
                    estimated_savings=75000.0,
                    timeline="1 week",
                    confidence=0.75,
                    created_at=context.current_time
                ))
                
        except Exception as e:
            logger.error(f"Error in resource utilization analysis: {e}")
            
        return recommendations
    
    def _analyze_capacity_planning(self, context: DecisionContext) -> List[Recommendation]:
        """Analyze capacity planning needs"""
        recommendations = []
        
        try:
            # Forecast future demand
            forecast_horizon = self.recommendation_rules['capacity_planning']['forecast_horizon']
            
            # Simple demand forecast based on current trends
            current_throughput = len(context.active_ships)
            
            # Check if we need capacity expansion
            if current_throughput > 0:
                # Predict peak demand
                predicted_peak = current_throughput * self.recommendation_rules['capacity_planning']['peak_threshold']
                current_capacity = len(context.available_berths) * 24  # ships per day capacity
                
                if predicted_peak > current_capacity * (1 - self.recommendation_rules['capacity_planning']['capacity_buffer']):
                    recommendations.append(Recommendation(
                        id=f"capacity_plan_{context.current_time.strftime('%Y%m%d_%H%M%S')}",
                        type=RecommendationType.CAPACITY_PLANNING,
                        priority=Priority.MEDIUM,
                        title="Plan for Capacity Expansion",
                        description=f"Predicted peak demand ({predicted_peak:.0f}) may exceed current capacity ({current_capacity:.0f})",
                        rationale="Proactive capacity planning prevents future bottlenecks and service degradation",
                        expected_impact={
                            'future_capacity': 0.25,
                            'service_reliability': 0.2,
                            'revenue_potential': 0.3
                        },
                        implementation_steps=[
                            "Conduct detailed demand analysis",
                            "Evaluate expansion options",
                            "Develop phased expansion plan",
                            "Secure funding and approvals",
                            "Begin infrastructure development"
                        ],
                        estimated_cost=2000000.0,
                        estimated_savings=500000.0,
                        timeline="6-12 months",
                        confidence=0.6,
                        created_at=context.current_time
                    ))
                    
        except Exception as e:
            logger.error(f"Error in capacity planning analysis: {e}")
            
        return recommendations
    
    def _analyze_performance_issues(self, context: DecisionContext) -> List[Recommendation]:
        """Analyze performance metrics and identify improvement opportunities"""
        recommendations = []
        
        try:
            # Check key performance indicators
            for metric, threshold in self.thresholds.items():
                current_value = context.performance_metrics.get(metric, 0.0)
                
                # Determine if performance is below threshold
                if metric in ['berth_utilization', 'throughput_efficiency', 'customer_satisfaction', 'equipment_availability']:
                    # Higher is better
                    if current_value < threshold:
                        recommendations.append(self._create_performance_recommendation(
                            metric, current_value, threshold, context, "improve"
                        ))
                else:
                    # Lower is better (e.g., waiting time, cost)
                    if current_value > threshold:
                        recommendations.append(self._create_performance_recommendation(
                            metric, current_value, threshold, context, "reduce"
                        ))
                        
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            
        return recommendations
    
    def _analyze_emergency_situations(self, context: DecisionContext) -> List[Recommendation]:
        """Analyze for emergency situations requiring immediate attention"""
        recommendations = []
        
        try:
            # Check for critical waiting times
            if context.active_ships:
                max_waiting_time = max([ship.get('waiting_time', 0) for ship in context.active_ships])
                if max_waiting_time > self.recommendation_rules['emergency_response']['critical_waiting_time']:
                    recommendations.append(Recommendation(
                        id=f"emergency_wait_{context.current_time.strftime('%Y%m%d_%H%M%S')}",
                        type=RecommendationType.EMERGENCY_RESPONSE,
                        priority=Priority.CRITICAL,
                        title="Critical Waiting Time Alert",
                        description=f"Ship has been waiting {max_waiting_time:.1f} hours, exceeding critical threshold",
                        rationale="Extended waiting times can lead to customer complaints, penalties, and reputation damage",
                        expected_impact={
                            'customer_satisfaction': 0.3,
                            'reputation_protection': 0.4,
                            'penalty_avoidance': 1.0
                        },
                        implementation_steps=[
                            "Immediately prioritize affected ship",
                            "Allocate additional resources if available",
                            "Communicate with ship operator",
                            "Investigate root cause of delay",
                            "Implement corrective measures"
                        ],
                        estimated_cost=20000.0,
                        estimated_savings=100000.0,
                        timeline="Immediate",
                        confidence=0.95,
                        created_at=context.current_time,
                        expires_at=context.current_time + timedelta(hours=2)
                    ))
            
            # Check weather conditions
            weather_impact = context.weather_conditions.get('impact_score', 0.0)
            if weather_impact > self.recommendation_rules['emergency_response']['severe_weather_threshold']:
                recommendations.append(Recommendation(
                    id=f"weather_emergency_{context.current_time.strftime('%Y%m%d_%H%M%S')}",
                    type=RecommendationType.EMERGENCY_RESPONSE,
                    priority=Priority.HIGH,
                    title="Severe Weather Response",
                    description=f"Weather conditions pose significant operational risk (impact score: {weather_impact:.2f})",
                    rationale="Severe weather requires immediate operational adjustments to ensure safety and minimize disruption",
                    expected_impact={
                        'safety_improvement': 0.8,
                        'operational_continuity': 0.6,
                        'damage_prevention': 0.9
                    },
                    implementation_steps=[
                        "Activate severe weather protocol",
                        "Secure all equipment and cargo",
                        "Adjust operational schedules",
                        "Communicate with all stakeholders",
                        "Monitor conditions continuously"
                    ],
                    estimated_cost=50000.0,
                    estimated_savings=500000.0,
                    timeline="Immediate",
                    confidence=0.9,
                    created_at=context.current_time,
                    expires_at=context.current_time + timedelta(hours=6)
                ))
                
        except Exception as e:
            logger.error(f"Error in emergency situation analysis: {e}")
            
        return recommendations
    
    def _create_performance_recommendation(self, metric: str, current: float, threshold: float, 
                                         context: DecisionContext, action: str) -> Recommendation:
        """Create a performance improvement recommendation"""
        
        metric_descriptions = {
            'berth_utilization': 'berth utilization rate',
            'average_waiting_time': 'average ship waiting time',
            'throughput_efficiency': 'container throughput efficiency',
            'cost_per_container': 'cost per container handled',
            'customer_satisfaction': 'customer satisfaction score',
            'equipment_availability': 'equipment availability rate'
        }
        
        description = metric_descriptions.get(metric, metric)
        gap = abs(current - threshold)
        gap_percent = (gap / threshold) * 100
        
        return Recommendation(
            id=f"perf_{metric}_{context.current_time.strftime('%Y%m%d_%H%M%S')}",
            type=RecommendationType.PERFORMANCE_IMPROVEMENT,
            priority=Priority.MEDIUM if gap_percent < 20 else Priority.HIGH,
            title=f"{action.title()} {description.title()}",
            description=f"Current {description} is {current:.2f}, target is {threshold:.2f} ({gap_percent:.1f}% gap)",
            rationale=f"Improving {description} will enhance overall port performance and competitiveness",
            expected_impact={
                metric: min(0.5, gap_percent / 100),
                'overall_performance': 0.1
            },
            implementation_steps=[
                f"Analyze root causes of {description} issues",
                "Develop targeted improvement plan",
                "Implement process optimizations",
                "Monitor progress and adjust as needed"
            ],
            estimated_cost=gap_percent * 1000,  # Rough estimate
            estimated_savings=gap_percent * 5000,
            timeline="2-4 weeks",
            confidence=0.7,
            created_at=context.current_time
        )
    
    def _calculate_resource_imbalance(self, utilization: Dict[str, float]) -> float:
        """Calculate resource utilization imbalance score"""
        if not utilization:
            return 0.0
            
        values = list(utilization.values())
        if len(values) < 2:
            return 0.0
            
        mean_util = np.mean(values)
        std_util = np.std(values)
        
        # Imbalance score: coefficient of variation
        return std_util / mean_util if mean_util > 0 else 0.0
    
    def _prioritize_recommendations(self, recommendations: List[Recommendation], 
                                  context: DecisionContext) -> List[Recommendation]:
        """Prioritize and filter recommendations"""
        if not recommendations:
            return []
            
        # Sort by priority and confidence
        priority_order = {Priority.CRITICAL: 4, Priority.HIGH: 3, Priority.MEDIUM: 2, Priority.LOW: 1}
        
        sorted_recommendations = sorted(
            recommendations,
            key=lambda r: (priority_order[r.priority], r.confidence, -r.estimated_cost),
            reverse=True
        )
        
        # Filter out expired recommendations
        current_time = context.current_time
        active_recommendations = [
            r for r in sorted_recommendations 
            if r.expires_at is None or r.expires_at > current_time
        ]
        
        # Limit to top recommendations to avoid overwhelming users
        return active_recommendations[:10]
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Get summary of current recommendations"""
        if not self.active_recommendations:
            return {
                'total_recommendations': 0,
                'by_priority': {},
                'by_type': {},
                'total_estimated_savings': 0.0,
                'total_estimated_cost': 0.0
            }
        
        # Count by priority
        by_priority = {}
        for priority in Priority:
            count = len([r for r in self.active_recommendations if r.priority == priority])
            by_priority[priority.value] = count
        
        # Count by type
        by_type = {}
        for rec_type in RecommendationType:
            count = len([r for r in self.active_recommendations if r.type == rec_type])
            by_type[rec_type.value] = count
        
        # Calculate totals
        total_savings = sum(r.estimated_savings for r in self.active_recommendations)
        total_cost = sum(r.estimated_cost for r in self.active_recommendations)
        
        return {
            'total_recommendations': len(self.active_recommendations),
            'by_priority': by_priority,
            'by_type': by_type,
            'total_estimated_savings': total_savings,
            'total_estimated_cost': total_cost,
            'net_benefit': total_savings - total_cost
        }

def create_sample_decision_context() -> DecisionContext:
    """Create sample decision context for testing"""
    return DecisionContext(
        current_time=datetime.now(),
        port_status={'operational': True, 'capacity': 0.8},
        active_ships=[
            {'id': 'SHIP001', 'type': 'container', 'waiting_time': 2.5, 'size': 3000},
            {'id': 'SHIP002', 'type': 'bulk', 'waiting_time': 5.2, 'size': 5000},
            {'id': 'SHIP003', 'type': 'container', 'waiting_time': 1.8, 'size': 2500},
            {'id': 'SHIP004', 'type': 'tanker', 'waiting_time': 9.1, 'size': 4000},  # Critical waiting time
        ],
        available_berths=[
            {'id': 'BERTH001', 'type': 'container', 'occupied': True},
            {'id': 'BERTH002', 'type': 'bulk', 'occupied': False},
            {'id': 'BERTH003', 'type': 'container', 'occupied': True},
        ],
        resource_utilization={
            'berths': 0.67,
            'cranes': 0.92,  # High utilization
            'trucks': 0.75,
            'workers': 0.88
        },
        weather_conditions={
            'wind_speed': 15,
            'visibility': 8,
            'impact_score': 0.3
        },
        operational_constraints=[
            'Limited crane availability during maintenance',
            'Reduced night shift capacity'
        ],
        performance_metrics={
            'berth_utilization': 0.75,  # Below threshold
            'average_waiting_time': 4.5,  # Above threshold
            'throughput_efficiency': 0.85,
            'cost_per_container': 145.0,
            'customer_satisfaction': 0.88,  # Below threshold
            'equipment_availability': 0.93  # Below threshold
        }
    )

if __name__ == "__main__":
    # Demo the decision support system
    logging.basicConfig(level=logging.INFO)
    
    print("Hong Kong Port Digital Twin - Decision Support System Demo")
    print("=" * 60)
    
    # Create decision support engine
    engine = DecisionSupportEngine()
    
    # Create sample context
    context = create_sample_decision_context()
    
    # Generate recommendations
    recommendations = engine.analyze_situation(context)
    
    print(f"\nGenerated {len(recommendations)} recommendations:")
    print("=" * 50)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.title} [{rec.priority.value.upper()}]")
        print(f"   Type: {rec.type.value}")
        print(f"   Description: {rec.description}")
        print(f"   Expected Savings: ${rec.estimated_savings:,.0f}")
        print(f"   Implementation Cost: ${rec.estimated_cost:,.0f}")
        print(f"   Timeline: {rec.timeline}")
        print(f"   Confidence: {rec.confidence:.1%}")
    
    # Show summary
    summary = engine.get_recommendation_summary()
    print(f"\nRecommendation Summary:")
    print(f"  Total Recommendations: {summary['total_recommendations']}")
    print(f"  Total Estimated Savings: ${summary['total_estimated_savings']:,.0f}")
    print(f"  Total Implementation Cost: ${summary['total_estimated_cost']:,.0f}")
    print(f"  Net Benefit: ${summary['net_benefit']:,.0f}")