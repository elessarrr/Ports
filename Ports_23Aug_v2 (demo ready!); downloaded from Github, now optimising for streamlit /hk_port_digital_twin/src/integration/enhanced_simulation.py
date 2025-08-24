"""Enhanced Port Simulation with Integrated Logistics

This module demonstrates the integration of the new logistics components
(yard management, truck routing, equipment maintenance, supply chain disruption)
with the existing port simulation system for Week 7 implementation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..core import PortSimulation, Vessel, Container
from ..logistics import (
    ContainerYardManager,
    TruckRoutingSystem,
    EquipmentMaintenanceScheduler,
    SupplyChainDisruptionModeler
)
from ..scenarios import (
    AdvancedScenarioLibrary,
    ScenarioManager,
    ScenarioParameters
)
from ..ai import AIOptimizer
from ..analysis import PerformanceAnalyzer


@dataclass
class EnhancedSimulationConfig:
    """Configuration for enhanced simulation with logistics integration."""
    
    # Simulation parameters
    simulation_duration: int = 24 * 7  # 1 week in hours
    time_step: float = 0.1  # 6 minutes
    
    # Yard configuration
    yard_blocks: int = 20
    containers_per_block: int = 500
    yard_equipment_count: int = 10
    
    # Truck routing configuration
    truck_fleet_size: int = 50
    gate_capacity: int = 8
    external_locations: int = 15
    
    # Equipment maintenance configuration
    crane_count: int = 12
    rtg_count: int = 25
    truck_count: int = 50
    maintenance_crew_size: int = 8
    
    # Supply chain configuration
    supply_chain_nodes: int = 20
    disruption_probability: float = 0.1
    
    # AI optimization
    enable_ai_optimization: bool = True
    optimization_interval: int = 60  # minutes


class EnhancedPortSimulation:
    """Enhanced port simulation integrating all logistics components."""
    
    def __init__(self, config: EnhancedSimulationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize core simulation
        self.port_simulation = PortSimulation()
        
        # Initialize logistics components
        self.yard_manager = ContainerYardManager(
            num_blocks=config.yard_blocks,
            containers_per_block=config.containers_per_block,
            equipment_count=config.yard_equipment_count
        )
        
        self.truck_routing = TruckRoutingSystem(
            fleet_size=config.truck_fleet_size,
            gate_capacity=config.gate_capacity,
            external_locations=config.external_locations
        )
        
        self.maintenance_scheduler = EquipmentMaintenanceScheduler(
            crane_count=config.crane_count,
            rtg_count=config.rtg_count,
            truck_count=config.truck_count,
            crew_size=config.maintenance_crew_size
        )
        
        self.disruption_modeler = SupplyChainDisruptionModeler(
            num_nodes=config.supply_chain_nodes
        )
        
        # Initialize scenario management
        self.scenario_library = AdvancedScenarioLibrary()
        self.scenario_manager = ScenarioManager()
        
        # Initialize AI and analytics
        if config.enable_ai_optimization:
            self.ai_optimizer = AIOptimizer()
        else:
            self.ai_optimizer = None
            
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Simulation state
        self.current_time = datetime.now()
        self.simulation_data = {
            'vessels': [],
            'containers': [],
            'yard_operations': [],
            'truck_operations': [],
            'maintenance_events': [],
            'disruptions': [],
            'performance_metrics': []
        }
        
    async def initialize_simulation(self, scenario_name: str = "peak_season"):
        """Initialize the simulation with a specific scenario."""
        
        self.logger.info(f"Initializing enhanced simulation with scenario: {scenario_name}")
        
        # Load scenario from library
        scenario_template = self.scenario_library.get_template(scenario_name)
        if scenario_template:
            scenario_instance = self.scenario_library.create_scenario_from_template(
                scenario_template.template_id
            )
            self.scenario_manager.load_scenario(scenario_instance.parameters)
        
        # Initialize all components
        await self.yard_manager.initialize()
        await self.truck_routing.initialize()
        await self.maintenance_scheduler.initialize()
        await self.disruption_modeler.initialize()
        
        # Set up initial conditions based on scenario
        await self._setup_scenario_conditions()
        
        self.logger.info("Enhanced simulation initialized successfully")
        
    async def _setup_scenario_conditions(self):
        """Set up initial conditions based on the current scenario."""
        
        scenario_params = self.scenario_manager.get_current_parameters()
        
        # Configure vessel arrivals based on scenario
        if scenario_params.season_type.value == "peak":
            vessel_arrival_rate = 2.5  # vessels per hour
        elif scenario_params.season_type.value == "normal":
            vessel_arrival_rate = 1.8
        else:  # low season
            vessel_arrival_rate = 1.2
            
        # Pre-populate yard with containers based on scenario
        initial_container_count = int(
            self.config.yard_blocks * self.config.containers_per_block * 0.7
        )
        
        for i in range(initial_container_count):
            container = Container(
                container_id=f"INIT_{i:06d}",
                size="20ft" if i % 2 == 0 else "40ft",
                weight=15000 + (i % 10) * 1000,
                destination="HKG" if i % 3 == 0 else "SHA"
            )
            await self.yard_manager.store_container(container)
            
        # Schedule initial truck jobs
        for i in range(20):
            pickup_time = self.current_time + timedelta(minutes=i * 15)
            await self.truck_routing.schedule_pickup_job(
                container_id=f"INIT_{i:06d}",
                pickup_time=pickup_time,
                destination="external_location_1"
            )
            
    async def run_simulation(self, duration_hours: Optional[int] = None):
        """Run the enhanced simulation for the specified duration."""
        
        duration = duration_hours or self.config.simulation_duration
        end_time = self.current_time + timedelta(hours=duration)
        
        self.logger.info(f"Starting enhanced simulation for {duration} hours")
        
        # Main simulation loop
        while self.current_time < end_time:
            
            # Update all components
            await self._update_simulation_step()
            
            # Advance time
            self.current_time += timedelta(hours=self.config.time_step)
            
            # Periodic AI optimization
            if (self.ai_optimizer and 
                self.current_time.minute % self.config.optimization_interval == 0):
                await self._run_ai_optimization()
                
            # Check for disruptions
            await self._check_disruptions()
            
            # Collect performance metrics
            await self._collect_metrics()
            
        self.logger.info("Enhanced simulation completed")
        
    async def _update_simulation_step(self):
        """Update all simulation components for one time step."""
        
        # Update core port simulation
        await self.port_simulation.step(self.config.time_step)
        
        # Update logistics components
        await self.yard_manager.step(self.config.time_step)
        await self.truck_routing.step(self.config.time_step)
        await self.maintenance_scheduler.step(self.config.time_step)
        await self.disruption_modeler.step(self.config.time_step)
        
        # Handle vessel arrivals and departures
        await self._handle_vessel_operations()
        
        # Coordinate between components
        await self._coordinate_operations()
        
    async def _handle_vessel_operations(self):
        """Handle vessel arrivals, loading/unloading, and departures."""
        
        # Check for new vessel arrivals
        arriving_vessels = await self.port_simulation.get_arriving_vessels()
        
        for vessel in arriving_vessels:
            # Allocate berth
            berth = await self.port_simulation.allocate_berth(vessel)
            
            if berth:
                # Plan container operations
                containers_to_unload = vessel.get_containers_for_port("HKG")
                containers_to_load = await self.yard_manager.get_containers_for_vessel(
                    vessel.vessel_id
                )
                
                # Schedule yard operations
                for container in containers_to_unload:
                    await self.yard_manager.schedule_storage(container)
                    
                for container in containers_to_load:
                    await self.yard_manager.schedule_retrieval(container.container_id)
                    
                # Schedule truck operations for import containers
                for container in containers_to_unload:
                    if container.requires_truck_delivery():
                        await self.truck_routing.schedule_delivery_job(
                            container_id=container.container_id,
                            pickup_time=self.current_time + timedelta(hours=2),
                            destination=container.destination
                        )
                        
    async def _coordinate_operations(self):
        """Coordinate operations between different components."""
        
        # Coordinate yard and truck operations
        ready_containers = await self.yard_manager.get_ready_containers()
        for container in ready_containers:
            if container.has_truck_job():
                truck_job = await self.truck_routing.get_job(container.container_id)
                if truck_job and truck_job.status == "ready":
                    await self.truck_routing.dispatch_truck(truck_job.job_id)
                    
        # Coordinate maintenance and operations
        maintenance_requests = await self.maintenance_scheduler.get_pending_requests()
        for request in maintenance_requests:
            if request.equipment_type == "yard_crane":
                await self.yard_manager.handle_equipment_maintenance(
                    request.equipment_id, request.estimated_duration
                )
            elif request.equipment_type == "truck":
                await self.truck_routing.handle_truck_maintenance(
                    request.equipment_id, request.estimated_duration
                )
                
    async def _run_ai_optimization(self):
        """Run AI optimization for current operations."""
        
        if not self.ai_optimizer:
            return
            
        # Collect current state
        current_state = {
            'yard_utilization': await self.yard_manager.get_utilization(),
            'truck_utilization': await self.truck_routing.get_utilization(),
            'berth_utilization': await self.port_simulation.get_berth_utilization(),
            'pending_jobs': await self.truck_routing.get_pending_jobs_count(),
            'maintenance_schedule': await self.maintenance_scheduler.get_schedule_summary()
        }
        
        # Run optimization
        optimization_result = await self.ai_optimizer.optimize(
            current_state=current_state,
            time_horizon=timedelta(hours=4)
        )
        
        # Apply optimization recommendations
        if optimization_result.recommendations:
            await self._apply_optimization_recommendations(optimization_result.recommendations)
            
    async def _apply_optimization_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Apply AI optimization recommendations."""
        
        for rec in recommendations:
            if rec['type'] == 'berth_reallocation':
                await self.port_simulation.reallocate_berth(
                    rec['vessel_id'], rec['new_berth_id']
                )
            elif rec['type'] == 'truck_rerouting':
                await self.truck_routing.reroute_truck(
                    rec['truck_id'], rec['new_route']
                )
            elif rec['type'] == 'maintenance_rescheduling':
                await self.maintenance_scheduler.reschedule_task(
                    rec['task_id'], rec['new_schedule']
                )
                
    async def _check_disruptions(self):
        """Check for and handle supply chain disruptions."""
        
        # Check for new disruptions
        new_disruptions = await self.disruption_modeler.check_for_disruptions(
            self.current_time
        )
        
        for disruption in new_disruptions:
            self.logger.warning(f"New disruption detected: {disruption.event_type}")
            
            # Apply disruption impacts
            impacts = await self.disruption_modeler.apply_disruption(disruption)
            
            for impact in impacts:
                if impact.area == "vessel_arrivals":
                    await self.port_simulation.apply_arrival_delay(
                        impact.severity, impact.duration
                    )
                elif impact.area == "yard_operations":
                    await self.yard_manager.apply_disruption_impact(
                        impact.severity, impact.duration
                    )
                elif impact.area == "truck_operations":
                    await self.truck_routing.apply_disruption_impact(
                        impact.severity, impact.duration
                    )
                    
    async def _collect_metrics(self):
        """Collect performance metrics from all components."""
        
        metrics = {
            'timestamp': self.current_time,
            'port_metrics': await self.port_simulation.get_metrics(),
            'yard_metrics': await self.yard_manager.get_metrics(),
            'truck_metrics': await self.truck_routing.get_metrics(),
            'maintenance_metrics': await self.maintenance_scheduler.get_metrics(),
            'disruption_metrics': await self.disruption_modeler.get_metrics()
        }
        
        self.simulation_data['performance_metrics'].append(metrics)
        
        # Update performance analyzer
        await self.performance_analyzer.update_metrics(metrics)
        
    async def get_simulation_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the simulation results."""
        
        summary = {
            'simulation_config': self.config.__dict__,
            'total_duration': self.config.simulation_duration,
            'vessels_processed': len(self.simulation_data['vessels']),
            'containers_handled': len(self.simulation_data['containers']),
            'yard_operations': len(self.simulation_data['yard_operations']),
            'truck_operations': len(self.simulation_data['truck_operations']),
            'maintenance_events': len(self.simulation_data['maintenance_events']),
            'disruptions': len(self.simulation_data['disruptions']),
            'final_metrics': await self.performance_analyzer.get_final_summary(),
            'component_performance': {
                'yard_manager': await self.yard_manager.get_performance_summary(),
                'truck_routing': await self.truck_routing.get_performance_summary(),
                'maintenance_scheduler': await self.maintenance_scheduler.get_performance_summary(),
                'disruption_modeler': await self.disruption_modeler.get_performance_summary()
            }
        }
        
        return summary
        
    async def export_results(self, output_path: str):
        """Export simulation results to files."""
        
        import json
        import os
        
        os.makedirs(output_path, exist_ok=True)
        
        # Export summary
        summary = await self.get_simulation_summary()
        with open(os.path.join(output_path, 'simulation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Export detailed metrics
        with open(os.path.join(output_path, 'performance_metrics.json'), 'w') as f:
            json.dump(self.simulation_data['performance_metrics'], f, indent=2, default=str)
            
        # Export component-specific data
        await self.yard_manager.export_data(os.path.join(output_path, 'yard_data.json'))
        await self.truck_routing.export_data(os.path.join(output_path, 'truck_data.json'))
        await self.maintenance_scheduler.export_data(os.path.join(output_path, 'maintenance_data.json'))
        await self.disruption_modeler.export_data(os.path.join(output_path, 'disruption_data.json'))
        
        self.logger.info(f"Simulation results exported to {output_path}")


# Example usage and testing functions
async def run_enhanced_simulation_demo():
    """Demonstration of the enhanced simulation capabilities."""
    
    # Create configuration
    config = EnhancedSimulationConfig(
        simulation_duration=48,  # 2 days
        yard_blocks=15,
        truck_fleet_size=30,
        enable_ai_optimization=True
    )
    
    # Create and initialize simulation
    simulation = EnhancedPortSimulation(config)
    await simulation.initialize_simulation("peak_season")
    
    # Run simulation
    await simulation.run_simulation()
    
    # Get results
    summary = await simulation.get_simulation_summary()
    print(f"Simulation completed: {summary['vessels_processed']} vessels processed")
    
    # Export results
    await simulation.export_results("./simulation_results")
    
    return summary


if __name__ == "__main__":
    # Run demo
    asyncio.run(run_enhanced_simulation_demo())