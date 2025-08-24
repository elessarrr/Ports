"""Supply Chain Disruption Modeling System for Hong Kong Port Digital Twin

This module simulates various supply chain disruptions and their impacts:
- Natural disasters (typhoons, earthquakes, floods)
- Geopolitical events (trade wars, sanctions, border closures)
- Operational disruptions (strikes, equipment failures, cyber attacks)
- Economic disruptions (fuel price spikes, currency fluctuations)
- Pandemic-related disruptions (lockdowns, capacity restrictions)
- Infrastructure failures (power outages, communication breakdowns)

The system models cascading effects, recovery strategies, and
resilience measures to help optimize supply chain robustness.
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
import json

# Configure logging
logger = logging.getLogger(__name__)

class DisruptionCategory(Enum):
    """Categories of supply chain disruptions"""
    NATURAL_DISASTER = "natural_disaster"
    GEOPOLITICAL = "geopolitical"
    OPERATIONAL = "operational"
    ECONOMIC = "economic"
    PANDEMIC = "pandemic"
    INFRASTRUCTURE = "infrastructure"
    CYBER_SECURITY = "cyber_security"

class DisruptionSeverity(Enum):
    """Severity levels of disruptions"""
    MINOR = 1
    MODERATE = 2
    MAJOR = 3
    SEVERE = 4
    CATASTROPHIC = 5

class DisruptionStatus(Enum):
    """Status of disruption events"""
    PENDING = "pending"
    ACTIVE = "active"
    RECOVERING = "recovering"
    RESOLVED = "resolved"

class ImpactArea(Enum):
    """Areas affected by disruptions"""
    PORT_OPERATIONS = "port_operations"
    SHIPPING_ROUTES = "shipping_routes"
    CARGO_HANDLING = "cargo_handling"
    TRUCK_TRANSPORT = "truck_transport"
    RAIL_TRANSPORT = "rail_transport"
    CUSTOMS_CLEARANCE = "customs_clearance"
    WAREHOUSE_STORAGE = "warehouse_storage"
    SUPPLY_CHAIN = "supply_chain"

@dataclass
class DisruptionEvent:
    """Supply chain disruption event"""
    event_id: str
    category: DisruptionCategory
    severity: DisruptionSeverity
    name: str
    description: str
    start_time: float
    duration: float  # hours
    affected_areas: List[ImpactArea]
    impact_factors: Dict[str, float]  # area -> impact multiplier
    probability: float = 0.1  # Annual probability
    status: DisruptionStatus = DisruptionStatus.PENDING
    recovery_time: float = 0.0
    cascading_events: List[str] = field(default_factory=list)
    mitigation_measures: List[str] = field(default_factory=list)
    
@dataclass
class DisruptionImpact:
    """Impact assessment of a disruption"""
    event_id: str
    area: ImpactArea
    capacity_reduction: float  # Percentage reduction
    cost_increase: float  # Percentage increase
    delay_factor: float  # Time delay multiplier
    quality_impact: float  # Service quality reduction
    start_time: float
    end_time: float
    
@dataclass
class RecoveryStrategy:
    """Recovery strategy for disruptions"""
    strategy_id: str
    applicable_categories: List[DisruptionCategory]
    applicable_severities: List[DisruptionSeverity]
    recovery_time_reduction: float  # Percentage
    cost_factor: float  # Implementation cost multiplier
    effectiveness: float  # 0-1 scale
    description: str
    prerequisites: List[str] = field(default_factory=list)
    
@dataclass
class SupplyChainNode:
    """Node in the supply chain network"""
    node_id: str
    node_type: str  # "supplier", "manufacturer", "distributor", "port", "customer"
    location: str
    capacity: float
    current_capacity: float
    reliability: float  # 0-1 scale
    connections: List[str] = field(default_factory=list)
    inventory_level: float = 0.0
    lead_time: float = 24.0  # hours
    
class SupplyChainDisruptionModeler:
    """Supply Chain Disruption Modeling System
    
    Simulates various disruption scenarios and their cascading effects
    on port operations and the broader supply chain network.
    """
    
    def __init__(self, env: simpy.Environment):
        """Initialize the disruption modeler
        
        Args:
            env: SimPy environment for simulation timing
        """
        self.env = env
        self.disruption_events = self._initialize_disruption_library()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.supply_chain_network = self._initialize_supply_chain_network()
        
        # Active disruptions and impacts
        self.active_disruptions = {}  # event_id -> DisruptionEvent
        self.active_impacts = {}  # area -> List[DisruptionImpact]
        self.disruption_history = []
        
        # Monitoring and metrics
        self.resilience_metrics = {
            'total_disruptions': 0,
            'average_recovery_time': 0.0,
            'supply_chain_reliability': 100.0,
            'economic_impact_total': 0.0,
            'mitigation_effectiveness': 0.0,
            'network_robustness': 0.0
        }
        
        # Start background processes
        self.env.process(self._disruption_generator_process())
        self.env.process(self._impact_monitoring_process())
        self.env.process(self._recovery_management_process())
    
    def _initialize_disruption_library(self) -> Dict[str, DisruptionEvent]:
        """Initialize library of potential disruption events"""
        events = {}
        
        # Natural disasters
        events["TYPHOON_SEVERE"] = DisruptionEvent(
            event_id="TYPHOON_SEVERE",
            category=DisruptionCategory.NATURAL_DISASTER,
            severity=DisruptionSeverity.SEVERE,
            name="Severe Typhoon",
            description="Category 4-5 typhoon affecting Hong Kong",
            start_time=0.0,
            duration=48.0,
            affected_areas=[ImpactArea.PORT_OPERATIONS, ImpactArea.SHIPPING_ROUTES, ImpactArea.TRUCK_TRANSPORT],
            impact_factors={
                "port_operations": 0.2,  # 80% capacity reduction
                "shipping_routes": 0.1,  # 90% capacity reduction
                "truck_transport": 0.3   # 70% capacity reduction
            },
            probability=0.15,  # 15% annual probability
            recovery_time=72.0
        )
        
        events["EARTHQUAKE_MAJOR"] = DisruptionEvent(
            event_id="EARTHQUAKE_MAJOR",
            category=DisruptionCategory.NATURAL_DISASTER,
            severity=DisruptionSeverity.MAJOR,
            name="Major Earthquake",
            description="Magnitude 6.5+ earthquake",
            start_time=0.0,
            duration=6.0,
            affected_areas=[ImpactArea.PORT_OPERATIONS, ImpactArea.INFRASTRUCTURE, ImpactArea.WAREHOUSE_STORAGE],
            impact_factors={
                "port_operations": 0.4,
                "warehouse_storage": 0.5,
                "infrastructure": 0.3
            },
            probability=0.05,
            recovery_time=168.0  # 1 week
        )
        
        # Geopolitical events
        events["TRADE_WAR"] = DisruptionEvent(
            event_id="TRADE_WAR",
            category=DisruptionCategory.GEOPOLITICAL,
            severity=DisruptionSeverity.MAJOR,
            name="Trade War Escalation",
            description="Escalation of trade tensions affecting cargo flows",
            start_time=0.0,
            duration=2160.0,  # 3 months
            affected_areas=[ImpactArea.SHIPPING_ROUTES, ImpactArea.CUSTOMS_CLEARANCE, ImpactArea.SUPPLY_CHAIN],
            impact_factors={
                "shipping_routes": 0.7,
                "customs_clearance": 0.6,
                "supply_chain": 0.8
            },
            probability=0.20,
            recovery_time=720.0  # 1 month
        )
        
        events["BORDER_CLOSURE"] = DisruptionEvent(
            event_id="BORDER_CLOSURE",
            category=DisruptionCategory.GEOPOLITICAL,
            severity=DisruptionSeverity.SEVERE,
            name="Border Closure",
            description="Temporary closure of border crossings",
            start_time=0.0,
            duration=168.0,  # 1 week
            affected_areas=[ImpactArea.TRUCK_TRANSPORT, ImpactArea.CUSTOMS_CLEARANCE, ImpactArea.SUPPLY_CHAIN],
            impact_factors={
                "truck_transport": 0.1,
                "customs_clearance": 0.0,
                "supply_chain": 0.3
            },
            probability=0.10,
            recovery_time=48.0
        )
        
        # Operational disruptions
        events["PORT_STRIKE"] = DisruptionEvent(
            event_id="PORT_STRIKE",
            category=DisruptionCategory.OPERATIONAL,
            severity=DisruptionSeverity.MAJOR,
            name="Port Workers Strike",
            description="Strike by port workers affecting operations",
            start_time=0.0,
            duration=120.0,  # 5 days
            affected_areas=[ImpactArea.PORT_OPERATIONS, ImpactArea.CARGO_HANDLING],
            impact_factors={
                "port_operations": 0.2,
                "cargo_handling": 0.1
            },
            probability=0.08,
            recovery_time=24.0
        )
        
        events["CYBER_ATTACK"] = DisruptionEvent(
            event_id="CYBER_ATTACK",
            category=DisruptionCategory.CYBER_SECURITY,
            severity=DisruptionSeverity.MAJOR,
            name="Cyber Security Attack",
            description="Cyber attack on port IT systems",
            start_time=0.0,
            duration=72.0,
            affected_areas=[ImpactArea.PORT_OPERATIONS, ImpactArea.CUSTOMS_CLEARANCE, ImpactArea.CARGO_HANDLING],
            impact_factors={
                "port_operations": 0.6,
                "customs_clearance": 0.4,
                "cargo_handling": 0.5
            },
            probability=0.12,
            recovery_time=96.0
        )
        
        # Economic disruptions
        events["FUEL_PRICE_SPIKE"] = DisruptionEvent(
            event_id="FUEL_PRICE_SPIKE",
            category=DisruptionCategory.ECONOMIC,
            severity=DisruptionSeverity.MODERATE,
            name="Fuel Price Spike",
            description="Significant increase in fuel prices",
            start_time=0.0,
            duration=720.0,  # 1 month
            affected_areas=[ImpactArea.SHIPPING_ROUTES, ImpactArea.TRUCK_TRANSPORT],
            impact_factors={
                "shipping_routes": 0.9,  # Cost increase, not capacity
                "truck_transport": 0.8
            },
            probability=0.25,
            recovery_time=240.0
        )
        
        # Pandemic disruptions
        events["PANDEMIC_LOCKDOWN"] = DisruptionEvent(
            event_id="PANDEMIC_LOCKDOWN",
            category=DisruptionCategory.PANDEMIC,
            severity=DisruptionSeverity.SEVERE,
            name="Pandemic Lockdown",
            description="COVID-19 style lockdown measures",
            start_time=0.0,
            duration=1440.0,  # 2 months
            affected_areas=[ImpactArea.PORT_OPERATIONS, ImpactArea.TRUCK_TRANSPORT, ImpactArea.SUPPLY_CHAIN],
            impact_factors={
                "port_operations": 0.5,
                "truck_transport": 0.4,
                "supply_chain": 0.6
            },
            probability=0.05,
            recovery_time=720.0
        )
        
        return events
    
    def _initialize_recovery_strategies(self) -> Dict[str, RecoveryStrategy]:
        """Initialize recovery strategies"""
        strategies = {}
        
        strategies["EMERGENCY_RESPONSE"] = RecoveryStrategy(
            strategy_id="EMERGENCY_RESPONSE",
            applicable_categories=[DisruptionCategory.NATURAL_DISASTER, DisruptionCategory.OPERATIONAL],
            applicable_severities=[DisruptionSeverity.MAJOR, DisruptionSeverity.SEVERE, DisruptionSeverity.CATASTROPHIC],
            recovery_time_reduction=0.3,
            cost_factor=2.0,
            effectiveness=0.8,
            description="Rapid emergency response and resource mobilization"
        )
        
        strategies["ALTERNATIVE_ROUTING"] = RecoveryStrategy(
            strategy_id="ALTERNATIVE_ROUTING",
            applicable_categories=[DisruptionCategory.GEOPOLITICAL, DisruptionCategory.INFRASTRUCTURE],
            applicable_severities=[DisruptionSeverity.MODERATE, DisruptionSeverity.MAJOR, DisruptionSeverity.SEVERE],
            recovery_time_reduction=0.4,
            cost_factor=1.5,
            effectiveness=0.7,
            description="Reroute cargo through alternative channels"
        )
        
        strategies["BACKUP_SYSTEMS"] = RecoveryStrategy(
            strategy_id="BACKUP_SYSTEMS",
            applicable_categories=[DisruptionCategory.CYBER_SECURITY, DisruptionCategory.INFRASTRUCTURE],
            applicable_severities=[DisruptionSeverity.MINOR, DisruptionSeverity.MODERATE, DisruptionSeverity.MAJOR],
            recovery_time_reduction=0.6,
            cost_factor=1.2,
            effectiveness=0.9,
            description="Activate backup systems and redundant infrastructure"
        )
        
        strategies["INVENTORY_BUFFER"] = RecoveryStrategy(
            strategy_id="INVENTORY_BUFFER",
            applicable_categories=[DisruptionCategory.ECONOMIC, DisruptionCategory.PANDEMIC],
            applicable_severities=[DisruptionSeverity.MODERATE, DisruptionSeverity.MAJOR],
            recovery_time_reduction=0.2,
            cost_factor=1.3,
            effectiveness=0.6,
            description="Use strategic inventory buffers to maintain supply"
        )
        
        return strategies
    
    def _initialize_supply_chain_network(self) -> Dict[str, SupplyChainNode]:
        """Initialize supply chain network nodes"""
        nodes = {}
        
        # Major suppliers
        nodes["SHENZHEN_SUPPLIER"] = SupplyChainNode(
            node_id="SHENZHEN_SUPPLIER",
            node_type="supplier",
            location="Shenzhen",
            capacity=1000.0,
            current_capacity=1000.0,
            reliability=0.95,
            connections=["HK_PORT"]
        )
        
        nodes["GUANGZHOU_SUPPLIER"] = SupplyChainNode(
            node_id="GUANGZHOU_SUPPLIER",
            node_type="supplier",
            location="Guangzhou",
            capacity=800.0,
            current_capacity=800.0,
            reliability=0.92,
            connections=["HK_PORT"]
        )
        
        # Hong Kong Port
        nodes["HK_PORT"] = SupplyChainNode(
            node_id="HK_PORT",
            node_type="port",
            location="Hong Kong",
            capacity=2000.0,
            current_capacity=2000.0,
            reliability=0.98,
            connections=["REGIONAL_DISTRIBUTOR", "GLOBAL_SHIPPING"]
        )
        
        # Distribution centers
        nodes["REGIONAL_DISTRIBUTOR"] = SupplyChainNode(
            node_id="REGIONAL_DISTRIBUTOR",
            node_type="distributor",
            location="Hong Kong",
            capacity=1500.0,
            current_capacity=1500.0,
            reliability=0.94,
            connections=["RETAIL_CUSTOMERS"]
        )
        
        # Global shipping
        nodes["GLOBAL_SHIPPING"] = SupplyChainNode(
            node_id="GLOBAL_SHIPPING",
            node_type="distributor",
            location="International",
            capacity=3000.0,
            current_capacity=3000.0,
            reliability=0.90,
            connections=["INTERNATIONAL_CUSTOMERS"]
        )
        
        # Customers
        nodes["RETAIL_CUSTOMERS"] = SupplyChainNode(
            node_id="RETAIL_CUSTOMERS",
            node_type="customer",
            location="Hong Kong",
            capacity=1000.0,
            current_capacity=1000.0,
            reliability=1.0
        )
        
        nodes["INTERNATIONAL_CUSTOMERS"] = SupplyChainNode(
            node_id="INTERNATIONAL_CUSTOMERS",
            node_type="customer",
            location="International",
            capacity=2500.0,
            current_capacity=2500.0,
            reliability=1.0
        )
        
        return nodes
    
    def trigger_disruption(self, event_id: str, custom_parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Manually trigger a disruption event
        
        Args:
            event_id: ID of the disruption event to trigger
            custom_parameters: Optional custom parameters to override defaults
            
        Returns:
            True if disruption was triggered successfully
        """
        if event_id not in self.disruption_events:
            logger.warning(f"Disruption event {event_id} not found")
            return False
        
        # Create a copy of the event
        event = self.disruption_events[event_id]
        triggered_event = DisruptionEvent(
            event_id=f"{event_id}_{int(self.env.now)}",
            category=event.category,
            severity=event.severity,
            name=event.name,
            description=event.description,
            start_time=self.env.now,
            duration=event.duration,
            affected_areas=event.affected_areas.copy(),
            impact_factors=event.impact_factors.copy(),
            probability=event.probability,
            recovery_time=event.recovery_time,
            cascading_events=event.cascading_events.copy(),
            mitigation_measures=event.mitigation_measures.copy()
        )
        
        # Apply custom parameters if provided
        if custom_parameters:
            for key, value in custom_parameters.items():
                if hasattr(triggered_event, key):
                    setattr(triggered_event, key, value)
        
        # Activate the disruption
        triggered_event.status = DisruptionStatus.ACTIVE
        self.active_disruptions[triggered_event.event_id] = triggered_event
        
        # Apply impacts
        self._apply_disruption_impacts(triggered_event)
        
        # Schedule recovery
        self.env.process(self._disruption_lifecycle(triggered_event))
        
        logger.info(f"Triggered disruption: {triggered_event.name} (ID: {triggered_event.event_id})")
        return True
    
    def _apply_disruption_impacts(self, event: DisruptionEvent):
        """Apply impacts of a disruption event"""
        for area in event.affected_areas:
            impact_factor = event.impact_factors.get(area.value, 0.5)
            
            impact = DisruptionImpact(
                event_id=event.event_id,
                area=area,
                capacity_reduction=(1.0 - impact_factor) * 100,
                cost_increase=((1.0 / impact_factor) - 1.0) * 100 if impact_factor > 0 else 100,
                delay_factor=1.0 / impact_factor if impact_factor > 0 else 5.0,
                quality_impact=(1.0 - impact_factor) * 50,
                start_time=event.start_time,
                end_time=event.start_time + event.duration + event.recovery_time
            )
            
            if area not in self.active_impacts:
                self.active_impacts[area] = []
            self.active_impacts[area].append(impact)
        
        # Apply impacts to supply chain network
        self._apply_network_impacts(event)
    
    def _apply_network_impacts(self, event: DisruptionEvent):
        """Apply disruption impacts to supply chain network"""
        for node in self.supply_chain_network.values():
            # Determine if node is affected based on location and disruption type
            is_affected = self._is_node_affected(node, event)
            
            if is_affected:
                # Calculate capacity reduction
                severity_factor = event.severity.value / 5.0
                base_reduction = 0.2 * severity_factor  # 20% reduction for severe events
                
                # Apply location-specific factors
                if node.location == "Hong Kong" and event.category in [DisruptionCategory.NATURAL_DISASTER, DisruptionCategory.PANDEMIC]:
                    base_reduction *= 1.5
                elif node.node_type == "port" and event.category == DisruptionCategory.OPERATIONAL:
                    base_reduction *= 2.0
                
                # Reduce node capacity
                reduction = min(base_reduction, 0.8)  # Max 80% reduction
                node.current_capacity = node.capacity * (1.0 - reduction)
                
                logger.debug(f"Node {node.node_id} capacity reduced by {reduction*100:.1f}% due to {event.name}")
    
    def _is_node_affected(self, node: SupplyChainNode, event: DisruptionEvent) -> bool:
        """Determine if a supply chain node is affected by a disruption"""
        # Geographic proximity
        if node.location == "Hong Kong" and event.category in [
            DisruptionCategory.NATURAL_DISASTER, 
            DisruptionCategory.PANDEMIC,
            DisruptionCategory.OPERATIONAL
        ]:
            return True
        
        # Node type specific impacts
        if node.node_type == "port" and event.category in [
            DisruptionCategory.OPERATIONAL,
            DisruptionCategory.CYBER_SECURITY,
            DisruptionCategory.NATURAL_DISASTER
        ]:
            return True
        
        # Global impacts
        if event.category in [DisruptionCategory.GEOPOLITICAL, DisruptionCategory.ECONOMIC]:
            return True
        
        return False
    
    def _disruption_lifecycle(self, event: DisruptionEvent):
        """Manage the lifecycle of a disruption event"""
        # Active phase
        yield self.env.timeout(event.duration)
        
        # Start recovery
        event.status = DisruptionStatus.RECOVERING
        logger.info(f"Disruption {event.name} entering recovery phase")
        
        # Apply recovery strategies
        recovery_time = self._apply_recovery_strategies(event)
        
        # Recovery phase
        yield self.env.timeout(recovery_time)
        
        # Resolution
        event.status = DisruptionStatus.RESOLVED
        self._resolve_disruption_impacts(event)
        
        # Move to history
        self.disruption_history.append(event)
        if event.event_id in self.active_disruptions:
            del self.active_disruptions[event.event_id]
        
        logger.info(f"Disruption {event.name} resolved after {event.duration + recovery_time:.1f} hours")
    
    def _apply_recovery_strategies(self, event: DisruptionEvent) -> float:
        """Apply recovery strategies and calculate adjusted recovery time"""
        applicable_strategies = []
        
        for strategy in self.recovery_strategies.values():
            if (event.category in strategy.applicable_categories and
                event.severity in strategy.applicable_severities):
                applicable_strategies.append(strategy)
        
        if not applicable_strategies:
            return event.recovery_time
        
        # Select best strategy (highest effectiveness)
        best_strategy = max(applicable_strategies, key=lambda s: s.effectiveness)
        
        # Calculate adjusted recovery time
        time_reduction = best_strategy.recovery_time_reduction
        adjusted_recovery_time = event.recovery_time * (1.0 - time_reduction)
        
        logger.info(f"Applied recovery strategy '{best_strategy.description}' to {event.name}")
        logger.info(f"Recovery time reduced from {event.recovery_time:.1f} to {adjusted_recovery_time:.1f} hours")
        
        return adjusted_recovery_time
    
    def _resolve_disruption_impacts(self, event: DisruptionEvent):
        """Resolve impacts when disruption ends"""
        # Remove impacts from active impacts
        for area in event.affected_areas:
            if area in self.active_impacts:
                self.active_impacts[area] = [
                    impact for impact in self.active_impacts[area]
                    if impact.event_id != event.event_id
                ]
                if not self.active_impacts[area]:
                    del self.active_impacts[area]
        
        # Restore supply chain network capacity
        for node in self.supply_chain_network.values():
            if self._is_node_affected(node, event):
                node.current_capacity = node.capacity
    
    def _disruption_generator_process(self):
        """Background process to randomly generate disruption events"""
        while True:
            # Check each potential disruption
            for event_template in self.disruption_events.values():
                # Calculate probability for this time period (1 hour)
                hourly_probability = event_template.probability / 8760  # Annual to hourly
                
                if random.random() < hourly_probability:
                    # Trigger the disruption
                    self.trigger_disruption(event_template.event_id)
            
            # Wait 1 hour before next check
            yield self.env.timeout(1.0)
    
    def _impact_monitoring_process(self):
        """Background process to monitor and update disruption impacts"""
        while True:
            # Update metrics
            self._update_resilience_metrics()
            
            # Check for cascading effects
            self._check_cascading_effects()
            
            # Monitor supply chain health
            self._monitor_supply_chain_health()
            
            # Update every 30 minutes
            yield self.env.timeout(0.5)
    
    def _recovery_management_process(self):
        """Background process to manage recovery operations"""
        while True:
            # Check recovery progress for active disruptions
            for event in self.active_disruptions.values():
                if event.status == DisruptionStatus.RECOVERING:
                    # Monitor recovery progress and adjust if needed
                    self._monitor_recovery_progress(event)
            
            # Update every hour
            yield self.env.timeout(1.0)
    
    def _check_cascading_effects(self):
        """Check for and trigger cascading disruption effects"""
        for event in self.active_disruptions.values():
            if event.cascading_events and event.status == DisruptionStatus.ACTIVE:
                # Check if conditions are met for cascading events
                for cascading_event_id in event.cascading_events:
                    if (cascading_event_id in self.disruption_events and
                        cascading_event_id not in [e.event_id.split('_')[0] for e in self.active_disruptions.values()]):
                        
                        # Probability of cascading effect
                        cascade_probability = 0.1 * (event.severity.value / 5.0)
                        
                        if random.random() < cascade_probability:
                            logger.warning(f"Cascading effect: {cascading_event_id} triggered by {event.name}")
                            self.trigger_disruption(cascading_event_id)
    
    def _monitor_recovery_progress(self, event: DisruptionEvent):
        """Monitor and potentially adjust recovery progress"""
        # Check if additional complications arise during recovery
        complication_probability = 0.05  # 5% chance per hour
        
        if random.random() < complication_probability:
            # Extend recovery time by 10-50%
            extension_factor = random.uniform(1.1, 1.5)
            event.recovery_time *= extension_factor
            
            logger.warning(f"Recovery complications for {event.name}, extended recovery time")
    
    def _monitor_supply_chain_health(self):
        """Monitor overall supply chain network health"""
        total_capacity = sum(node.capacity for node in self.supply_chain_network.values())
        current_capacity = sum(node.current_capacity for node in self.supply_chain_network.values())
        
        network_health = (current_capacity / total_capacity) * 100 if total_capacity > 0 else 0
        self.resilience_metrics['network_robustness'] = network_health
        
        # Alert if network health is critically low
        if network_health < 50:
            logger.warning(f"Supply chain network health critically low: {network_health:.1f}%")
    
    def _update_resilience_metrics(self):
        """Update resilience and performance metrics"""
        # Total disruptions
        self.resilience_metrics['total_disruptions'] = len(self.disruption_history) + len(self.active_disruptions)
        
        # Average recovery time
        if self.disruption_history:
            recovery_times = [event.recovery_time for event in self.disruption_history]
            self.resilience_metrics['average_recovery_time'] = np.mean(recovery_times)
        
        # Supply chain reliability
        active_severe_disruptions = sum(1 for event in self.active_disruptions.values() 
                                      if event.severity.value >= 3)
        reliability_reduction = active_severe_disruptions * 10  # 10% per severe disruption
        self.resilience_metrics['supply_chain_reliability'] = max(0, 100 - reliability_reduction)
    
    def get_current_impacts(self) -> Dict[str, Any]:
        """Get current disruption impacts by area"""
        current_impacts = {}
        
        for area, impacts in self.active_impacts.items():
            if impacts:
                # Aggregate impacts for the area
                total_capacity_reduction = sum(impact.capacity_reduction for impact in impacts)
                avg_delay_factor = np.mean([impact.delay_factor for impact in impacts])
                avg_cost_increase = np.mean([impact.cost_increase for impact in impacts])
                
                current_impacts[area.value] = {
                    'capacity_reduction_percent': min(100, total_capacity_reduction),
                    'delay_factor': avg_delay_factor,
                    'cost_increase_percent': avg_cost_increase,
                    'active_events': len(impacts)
                }
        
        return current_impacts
    
    def get_disruption_status(self) -> Dict[str, Any]:
        """Get current disruption system status"""
        return {
            'active_disruptions': len(self.active_disruptions),
            'total_disruptions_to_date': len(self.disruption_history) + len(self.active_disruptions),
            'affected_areas': len(self.active_impacts),
            'supply_chain_nodes': len(self.supply_chain_network),
            'resilience_metrics': self.resilience_metrics,
            'current_impacts': self.get_current_impacts()
        }
    
    def generate_disruption_report(self) -> Dict[str, Any]:
        """Generate comprehensive disruption analysis report"""
        # Categorize disruptions by type
        disruptions_by_category = defaultdict(int)
        disruptions_by_severity = defaultdict(int)
        
        all_disruptions = list(self.disruption_history) + list(self.active_disruptions.values())
        
        for event in all_disruptions:
            disruptions_by_category[event.category.value] += 1
            disruptions_by_severity[event.severity.value] += 1
        
        # Calculate economic impact
        total_economic_impact = 0.0
        for event in self.disruption_history:
            # Estimate economic impact based on duration and severity
            impact_per_hour = event.severity.value * 100000  # HKD per hour
            total_economic_impact += impact_per_hour * (event.duration + event.recovery_time)
        
        return {
            'summary': {
                'total_disruptions': len(all_disruptions),
                'active_disruptions': len(self.active_disruptions),
                'resolved_disruptions': len(self.disruption_history),
                'total_economic_impact_hkd': total_economic_impact
            },
            'by_category': dict(disruptions_by_category),
            'by_severity': dict(disruptions_by_severity),
            'resilience_metrics': self.resilience_metrics,
            'supply_chain_health': {
                'network_robustness': self.resilience_metrics['network_robustness'],
                'node_status': {
                    node_id: {
                        'capacity_utilization': (node.current_capacity / node.capacity) * 100,
                        'reliability': node.reliability * 100
                    }
                    for node_id, node in self.supply_chain_network.items()
                }
            }
        }