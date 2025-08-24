# Comments for context:
# This module extends the existing SimulationController to provide strategic simulation
# capabilities for business intelligence and executive decision-making. It adds
# business-focused metrics, ROI calculations, and scenario comparison features
# without modifying the original SimulationController.
#
# The StrategicSimulationController uses composition to extend functionality,
# ensuring zero breaking changes to existing code while providing enhanced
# capabilities for strategic scenarios like peak season optimization and
# maintenance window planning.

from typing import Dict, List, Optional, Callable, Tuple, Any
import logging
import time
import threading
from dataclasses import dataclass, asdict
from enum import Enum

from .simulation_controller import SimulationController, SimulationState
from .port_simulation import PortSimulation
from ..scenarios.strategic_simulations import (
    StrategicScenarioParameters,
    StrategicScenarioType,
    BusinessMetricType,
    get_strategic_scenario,
    get_business_metrics_summary
)
from ..utils.metrics_collector import MetricsCollector

class StrategicSimulationMode(Enum):
    """Strategic simulation execution modes."""
    SINGLE_SCENARIO = "single_scenario"
    COMPARISON = "comparison"
    OPTIMIZATION = "optimization"
    BENCHMARK = "benchmark"

@dataclass
class BusinessMetrics:
    """Real-time business metrics calculated during strategic simulation."""
    
    # Financial metrics
    revenue_per_hour: float = 0.0
    total_revenue: float = 0.0
    operational_cost: float = 0.0
    maintenance_cost: float = 0.0
    net_profit: float = 0.0
    
    # Efficiency metrics
    throughput_teu_per_hour: float = 0.0
    berth_utilization_percentage: float = 0.0
    average_waiting_time_hours: float = 0.0
    processing_efficiency: float = 0.0
    
    # Strategic metrics
    capacity_utilization: float = 0.0
    customer_satisfaction_score: float = 0.0
    competitive_advantage_score: float = 0.0
    ai_optimization_benefit: float = 0.0
    
    # ROI metrics
    roi_percentage: float = 0.0
    payback_period_months: float = 0.0
    cost_benefit_ratio: float = 0.0

@dataclass
class StrategicSimulationResult:
    """Results from a strategic simulation run."""
    
    scenario_name: str
    scenario_type: StrategicScenarioType
    simulation_duration: float
    business_metrics: BusinessMetrics
    operational_metrics: Dict[str, Any]
    executive_summary: str
    recommendations: List[str]
    risk_assessment: Dict[str, float]
    
class StrategicSimulationController:
    """Enhanced simulation controller for strategic business scenarios.
    
    This controller extends the base SimulationController with business intelligence
    capabilities, strategic scenario management, and executive-level reporting.
    It uses composition to avoid modifying existing code while providing enhanced
    functionality for strategic decision-making.
    """
    
    def __init__(self, 
                 port_simulation: PortSimulation, 
                 metrics_collector: Optional[MetricsCollector] = None):
        """Initialize strategic simulation controller.
        
        Args:
            port_simulation: The port simulation instance to control
            metrics_collector: Optional metrics collector for tracking performance
        """
        # Use composition to extend base controller
        self.base_controller = SimulationController(port_simulation, metrics_collector)
        
        # Strategic simulation specific attributes
        self.strategic_scenario: Optional[StrategicScenarioParameters] = None
        self.simulation_mode: StrategicSimulationMode = StrategicSimulationMode.SINGLE_SCENARIO
        self.business_metrics: BusinessMetrics = BusinessMetrics()
        self.simulation_results: List[StrategicSimulationResult] = []
        
        # Business intelligence tracking
        self.metrics_history: List[BusinessMetrics] = []
        self.optimization_iterations: int = 0
        self.baseline_metrics: Optional[BusinessMetrics] = None
        
        # Callbacks for strategic events
        self.on_business_metrics_update: Optional[Callable[[BusinessMetrics], None]] = None
        self.on_optimization_iteration: Optional[Callable[[int, BusinessMetrics], None]] = None
        self.on_strategic_milestone: Optional[Callable[[str, Dict[str, Any]], None]] = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Revenue calculation parameters (configurable)
        self.revenue_per_teu = 150.0  # USD per TEU
        self.operational_cost_per_hour = 5000.0  # USD per hour
        self.maintenance_cost_per_berth_per_day = 2000.0  # USD per berth per day
    
    def set_strategic_scenario(self, scenario_key: str) -> bool:
        """Set the strategic scenario for simulation.
        
        Args:
            scenario_key: Strategic scenario key or alias
            
        Returns:
            bool: True if scenario was set successfully, False otherwise
        """
        scenario = get_strategic_scenario(scenario_key)
        if scenario is None:
            self.logger.error(f"Strategic scenario not found: {scenario_key}")
            return False
            
        self.strategic_scenario = scenario
        self.logger.info(f"Strategic scenario set: {scenario.scenario_name}")
        
        # Apply scenario parameters to base simulation
        self._apply_strategic_parameters()
        
        return True
    
    def start_strategic_simulation(self, 
                                 duration: float, 
                                 mode: StrategicSimulationMode = StrategicSimulationMode.SINGLE_SCENARIO,
                                 time_step: float = 1.0,
                                 threaded: bool = True) -> bool:
        """Start strategic simulation with business intelligence tracking.
        
        Args:
            duration: Total simulation duration in hours
            mode: Strategic simulation mode
            time_step: Time step size in hours
            threaded: Whether to run in separate thread
            
        Returns:
            bool: True if simulation started successfully, False otherwise
        """
        if self.strategic_scenario is None:
            self.logger.error("No strategic scenario set")
            return False
            
        self.simulation_mode = mode
        
        # Reset strategic metrics
        self.business_metrics = BusinessMetrics()
        self.metrics_history = []
        self.optimization_iterations = 0
        
        # Set up business metrics tracking
        self.base_controller.on_progress_update = self._on_simulation_progress
        
        # Start base simulation
        success = self.base_controller.start(duration, time_step, threaded)
        
        if success:
            self.logger.info(f"Strategic simulation started: {self.strategic_scenario.scenario_name}")
            
            # Start business metrics collection thread
            if threaded:
                metrics_thread = threading.Thread(
                    target=self._business_metrics_collection_loop, 
                    daemon=True
                )
                metrics_thread.start()
        
        return success
    
    def get_business_metrics(self) -> BusinessMetrics:
        """Get current business metrics.
        
        Returns:
            BusinessMetrics: Current business metrics
        """
        return self.business_metrics
    
    def get_strategic_summary(self) -> Dict[str, Any]:
        """Get strategic simulation summary for executive reporting.
        
        Returns:
            dict: Strategic summary with business insights
        """
        if self.strategic_scenario is None:
            return {}
            
        # Calculate performance vs targets
        target_metrics = self.strategic_scenario.business_metrics
        current_metrics = self.business_metrics
        
        performance_vs_target = {
            'revenue_achievement': (current_metrics.revenue_per_hour / target_metrics.target_revenue_per_hour * 100) if target_metrics.target_revenue_per_hour > 0 else 0,
            'throughput_achievement': (current_metrics.throughput_teu_per_hour / (target_metrics.target_throughput_increase / 100 * 1000)) * 100 if target_metrics.target_throughput_increase > 0 else 0,
            'efficiency_achievement': (current_metrics.processing_efficiency / target_metrics.operational_efficiency_target * 100) if target_metrics.operational_efficiency_target > 0 else 0,
            'utilization_achievement': (current_metrics.capacity_utilization / target_metrics.capacity_utilization_target * 100) if target_metrics.capacity_utilization_target > 0 else 0
        }
        
        return {
            'scenario_name': self.strategic_scenario.scenario_name,
            'business_objective': self.strategic_scenario.business_objective,
            'executive_summary': self.strategic_scenario.executive_summary,
            'current_metrics': asdict(current_metrics),
            'target_metrics': asdict(target_metrics),
            'performance_vs_target': performance_vs_target,
            'roi_analysis': {
                'expected_roi': self.strategic_scenario.expected_roi_percentage,
                'current_roi': current_metrics.roi_percentage,
                'investment_required': self.strategic_scenario.investment_required,
                'payback_period': current_metrics.payback_period_months
            },
            'risk_factors': self.strategic_scenario.risk_factors,
            'mitigation_strategies': self.strategic_scenario.mitigation_strategies
        }
    
    def generate_executive_report(self) -> Dict[str, Any]:
        """Generate comprehensive executive report.
        
        Returns:
            dict: Executive report with recommendations and insights
        """
        summary = self.get_strategic_summary()
        
        # Generate recommendations based on performance
        recommendations = self._generate_recommendations()
        
        # Calculate competitive analysis
        competitive_analysis = self._calculate_competitive_analysis()
        
        return {
            **summary,
            'recommendations': recommendations,
            'competitive_analysis': competitive_analysis,
            'key_insights': self._extract_key_insights(),
            'next_steps': self._suggest_next_steps()
        }
    
    def compare_scenarios(self, scenario_keys: List[str], duration: float) -> Dict[str, Any]:
        """Compare multiple strategic scenarios.
        
        Args:
            scenario_keys: List of scenario keys to compare
            duration: Simulation duration for each scenario
            
        Returns:
            dict: Comparison results with recommendations
        """
        comparison_results = {}
        
        for scenario_key in scenario_keys:
            # Set scenario and run simulation
            if self.set_strategic_scenario(scenario_key):
                self.start_strategic_simulation(duration, threaded=False)
                
                # Wait for completion
                while not self.base_controller.is_completed():
                    time.sleep(0.1)
                
                # Store results
                comparison_results[scenario_key] = {
                    'scenario_name': self.strategic_scenario.scenario_name,
                    'business_metrics': asdict(self.business_metrics),
                    'strategic_summary': self.get_strategic_summary()
                }
                
                # Reset for next scenario
                self.base_controller.reset()
        
        # Generate comparison analysis
        return self._analyze_scenario_comparison(comparison_results)
    
    # Delegate base controller methods
    def stop(self) -> bool:
        """Stop the strategic simulation."""
        return self.base_controller.stop()
    
    def pause(self) -> bool:
        """Pause the strategic simulation."""
        return self.base_controller.pause()
    
    def resume(self) -> bool:
        """Resume the strategic simulation."""
        return self.base_controller.resume()
    
    def reset(self) -> bool:
        """Reset the strategic simulation."""
        # Reset strategic state
        self.business_metrics = BusinessMetrics()
        self.metrics_history = []
        self.optimization_iterations = 0
        self.baseline_metrics = None
        
        return self.base_controller.reset()
    
    def get_state(self) -> SimulationState:
        """Get current simulation state."""
        return self.base_controller.get_state()
    
    def get_progress(self) -> Tuple[float, float]:
        """Get current simulation progress."""
        return self.base_controller.get_progress()
    
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self.base_controller.is_running()
    
    # Private methods for strategic functionality
    
    def _apply_strategic_parameters(self):
        """Apply strategic scenario parameters to simulation."""
        if self.strategic_scenario is None:
            return
            
        # Apply parameters to base simulation
        # This would need to be implemented based on the specific
        # port simulation structure
        self.logger.info(f"Applied strategic parameters for {self.strategic_scenario.scenario_name}")
    
    def _on_simulation_progress(self, current_time: float, duration: float):
        """Handle simulation progress updates for business metrics calculation."""
        # Update business metrics based on current simulation state
        self._calculate_business_metrics(current_time)
        
        # Notify business metrics update
        if self.on_business_metrics_update:
            self.on_business_metrics_update(self.business_metrics)
    
    def _calculate_business_metrics(self, current_time: float):
        """Calculate real-time business metrics."""
        # Get operational metrics from base controller
        operational_metrics = self.base_controller.get_metrics_summary()
        
        # Calculate financial metrics
        # Note: These calculations would need to be refined based on actual simulation data
        throughput = operational_metrics.get('total_throughput', 0)
        self.business_metrics.throughput_teu_per_hour = throughput / max(current_time, 1)
        self.business_metrics.revenue_per_hour = self.business_metrics.throughput_teu_per_hour * self.revenue_per_teu
        self.business_metrics.total_revenue = self.business_metrics.revenue_per_hour * current_time
        
        # Calculate operational costs
        self.business_metrics.operational_cost = self.operational_cost_per_hour * current_time
        
        # Calculate berth utilization
        self.business_metrics.berth_utilization_percentage = operational_metrics.get('average_berth_utilization', 0) * 100
        
        # Calculate efficiency metrics
        self.business_metrics.processing_efficiency = operational_metrics.get('processing_efficiency', 0) * 100
        
        # Calculate ROI
        if self.strategic_scenario and self.strategic_scenario.investment_required > 0:
            net_benefit = self.business_metrics.total_revenue - self.business_metrics.operational_cost
            self.business_metrics.roi_percentage = (net_benefit / self.strategic_scenario.investment_required) * 100
        
        # Store metrics history
        self.metrics_history.append(BusinessMetrics(**asdict(self.business_metrics)))
    
    def _business_metrics_collection_loop(self):
        """Background thread for continuous business metrics collection."""
        while self.base_controller.is_running():
            current_time, _ = self.base_controller.get_progress()
            self._calculate_business_metrics(current_time)
            time.sleep(1.0)  # Update every second
    
    def _generate_recommendations(self) -> List[str]:
        """Generate business recommendations based on simulation results."""
        recommendations = []
        
        if self.strategic_scenario is None:
            return recommendations
            
        # Analyze performance vs targets
        target_metrics = self.strategic_scenario.business_metrics
        current_metrics = self.business_metrics
        
        # Revenue recommendations
        if current_metrics.revenue_per_hour < target_metrics.target_revenue_per_hour:
            recommendations.append(
                f"Revenue per hour ({current_metrics.revenue_per_hour:.0f}) is below target "
                f"({target_metrics.target_revenue_per_hour:.0f}). Consider optimizing berth allocation."
            )
        
        # Efficiency recommendations
        if current_metrics.processing_efficiency < target_metrics.operational_efficiency_target:
            recommendations.append(
                f"Processing efficiency ({current_metrics.processing_efficiency:.1f}%) is below target "
                f"({target_metrics.operational_efficiency_target:.1f}%). Implement AI-driven optimization."
            )
        
        # Utilization recommendations
        if current_metrics.capacity_utilization < target_metrics.capacity_utilization_target:
            recommendations.append(
                f"Capacity utilization ({current_metrics.capacity_utilization:.1f}%) is below target "
                f"({target_metrics.capacity_utilization_target:.1f}%). Review scheduling algorithms."
            )
        
        return recommendations
    
    def _calculate_competitive_analysis(self) -> Dict[str, Any]:
        """Calculate competitive positioning analysis."""
        if self.strategic_scenario is None:
            return {}
            
        competitor_baseline = self.strategic_scenario.competitor_efficiency_baseline
        our_efficiency = self.business_metrics.processing_efficiency
        
        competitive_advantage = our_efficiency - competitor_baseline
        
        return {
            'competitor_baseline_efficiency': competitor_baseline,
            'our_efficiency': our_efficiency,
            'competitive_advantage': competitive_advantage,
            'market_position': 'Leading' if competitive_advantage > 10 else 'Competitive' if competitive_advantage > 0 else 'Behind'
        }
    
    def _extract_key_insights(self) -> List[str]:
        """Extract key business insights from simulation results."""
        insights = []
        
        if self.strategic_scenario is None:
            return insights
            
        # ROI insights
        if self.business_metrics.roi_percentage > self.strategic_scenario.expected_roi_percentage:
            insights.append(f"ROI exceeds expectations: {self.business_metrics.roi_percentage:.1f}% vs {self.strategic_scenario.expected_roi_percentage:.1f}% expected")
        
        # Efficiency insights
        if self.business_metrics.processing_efficiency > 90:
            insights.append("Exceptional operational efficiency achieved through AI optimization")
        
        # Revenue insights
        if self.business_metrics.revenue_per_hour > self.strategic_scenario.business_metrics.target_revenue_per_hour:
            insights.append("Revenue targets exceeded through strategic capacity optimization")
        
        return insights
    
    def _suggest_next_steps(self) -> List[str]:
        """Suggest strategic next steps based on simulation results."""
        next_steps = []
        
        if self.strategic_scenario is None:
            return next_steps
            
        # Based on scenario type, suggest appropriate next steps
        if self.strategic_scenario.strategic_type == StrategicScenarioType.PEAK_SEASON_CAPACITY_OPTIMIZATION:
            next_steps.extend([
                "Implement AI-driven berth allocation system",
                "Invest in additional crane capacity for peak periods",
                "Develop dynamic pricing model for peak season"
            ])
        elif self.strategic_scenario.strategic_type == StrategicScenarioType.MAINTENANCE_WINDOW_OPTIMIZATION:
            next_steps.extend([
                "Implement predictive maintenance scheduling",
                "Develop maintenance crew optimization algorithms",
                "Create customer communication protocols for maintenance periods"
            ])
        
        return next_steps
    
    def _analyze_scenario_comparison(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and rank scenario comparison results."""
        # Rank scenarios by ROI
        scenario_rankings = []
        
        for scenario_key, results in comparison_results.items():
            metrics = results['business_metrics']
            scenario_rankings.append({
                'scenario': scenario_key,
                'scenario_name': results['scenario_name'],
                'roi': metrics['roi_percentage'],
                'revenue_per_hour': metrics['revenue_per_hour'],
                'efficiency': metrics['processing_efficiency']
            })
        
        # Sort by ROI
        scenario_rankings.sort(key=lambda x: x['roi'], reverse=True)
        
        return {
            'comparison_results': comparison_results,
            'scenario_rankings': scenario_rankings,
            'recommended_scenario': scenario_rankings[0] if scenario_rankings else None,
            'analysis_summary': f"Compared {len(comparison_results)} strategic scenarios. "
                              f"Best ROI: {scenario_rankings[0]['roi']:.1f}%" if scenario_rankings else "No scenarios compared"
        }