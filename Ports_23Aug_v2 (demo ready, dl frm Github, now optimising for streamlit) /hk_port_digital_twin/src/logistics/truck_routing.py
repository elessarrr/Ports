"""Truck Routing System for Hong Kong Port Digital Twin

This module simulates truck routing and container pickup/delivery operations:
- Route optimization for container pickup and delivery
- Traffic modeling and congestion handling
- Truck scheduling and resource allocation
- Integration with yard operations and gate systems
- Real-time route adjustments based on conditions

The system models the complex logistics of moving containers between
the port, container yards, and external destinations.
"""

import logging
import simpy
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import numpy as np
from collections import defaultdict
import heapq

# Configure logging
logger = logging.getLogger(__name__)

class TruckType(Enum):
    """Types of trucks for container transport"""
    CHASSIS_20FT = "chassis_20ft"
    CHASSIS_40FT = "chassis_40ft"
    MULTI_TRAILER = "multi_trailer"
    HEAVY_HAUL = "heavy_haul"

class RouteType(Enum):
    """Types of routes"""
    PICKUP = "pickup"  # From external location to port
    DELIVERY = "delivery"  # From port to external location
    INTERNAL = "internal"  # Within port area
    REPOSITIONING = "repositioning"  # Empty container movement

class TrafficCondition(Enum):
    """Traffic conditions affecting route times"""
    FREE_FLOW = "free_flow"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    CONGESTED = "congested"

@dataclass
class Location:
    """Geographic location for routing"""
    location_id: str
    name: str
    latitude: float
    longitude: float
    location_type: str  # "port", "warehouse", "depot", "customer"
    gate_capacity: int = 1
    processing_time: float = 15.0  # minutes

@dataclass
class TruckJob:
    """Container transport job"""
    job_id: str
    container_id: str
    route_type: RouteType
    origin: Location
    destination: Location
    truck_type: TruckType
    priority: int = 1  # 1=highest, 5=lowest
    scheduled_time: Optional[float] = None
    deadline: Optional[float] = None
    special_requirements: List[str] = field(default_factory=list)
    
@dataclass
class Truck:
    """Truck resource"""
    truck_id: str
    truck_type: TruckType
    current_location: Location
    capacity: int = 1  # Number of containers
    is_available: bool = True
    current_job: Optional[TruckJob] = None
    fuel_level: float = 100.0  # Percentage
    maintenance_due: float = 0.0  # Hours until maintenance
    
@dataclass
class Route:
    """Optimized route between locations"""
    route_id: str
    origin: Location
    destination: Location
    waypoints: List[Location] = field(default_factory=list)
    distance: float = 0.0  # kilometers
    estimated_time: float = 0.0  # hours
    traffic_factor: float = 1.0
    toll_cost: float = 0.0
    fuel_cost: float = 0.0

class TruckRoutingSystem:
    """Truck Routing and Dispatch System
    
    Manages truck fleet operations, route optimization,
    and container pickup/delivery scheduling.
    """
    
    def __init__(self, env: simpy.Environment):
        """Initialize the truck routing system
        
        Args:
            env: SimPy environment for simulation timing
        """
        self.env = env
        self.locations = self._initialize_locations()
        self.trucks = self._initialize_truck_fleet()
        self.pending_jobs = []  # Priority queue of jobs
        self.active_jobs = {}  # job_id -> TruckJob
        self.completed_jobs = []
        self.route_cache = {}  # Cache for computed routes
        self.traffic_conditions = defaultdict(lambda: TrafficCondition.MODERATE)
        
        # Performance metrics
        self.metrics = {
            'total_jobs_completed': 0,
            'average_delivery_time': 0.0,
            'truck_utilization': 0.0,
            'fuel_consumption': 0.0,
            'on_time_delivery_rate': 0.0,
            'total_distance_traveled': 0.0
        }
        
        # Start background processes
        self.env.process(self._traffic_update_process())
        self.env.process(self._job_dispatcher_process())
    
    def _initialize_locations(self) -> Dict[str, Location]:
        """Initialize key locations in Hong Kong"""
        locations = {
            # Port terminals
            'CT1': Location('CT1', 'Container Terminal 1', 22.2908, 114.1501, 'port', gate_capacity=10),
            'CT2': Location('CT2', 'Container Terminal 2', 22.2915, 114.1485, 'port', gate_capacity=12),
            'CT3': Location('CT3', 'Container Terminal 3', 22.2922, 114.1470, 'port', gate_capacity=8),
            
            # Major logistics hubs
            'KWAI_CHUNG': Location('KWAI_CHUNG', 'Kwai Chung Container Port', 22.3500, 114.1300, 'port', gate_capacity=15),
            'STONECUTTERS': Location('STONECUTTERS', 'Stonecutters Island', 22.3200, 114.1400, 'warehouse'),
            
            # Industrial areas
            'TSUEN_WAN': Location('TSUEN_WAN', 'Tsuen Wan Industrial', 22.3700, 114.1100, 'warehouse'),
            'TUEN_MUN': Location('TUEN_MUN', 'Tuen Mun Logistics Hub', 22.3900, 113.9700, 'warehouse'),
            'YUEN_LONG': Location('YUEN_LONG', 'Yuen Long Distribution Center', 22.4400, 114.0300, 'warehouse'),
            
            # Cross-border points
            'LO_WU': Location('LO_WU', 'Lo Wu Border Crossing', 22.5300, 114.1100, 'border'),
            'MAN_KAM_TO': Location('MAN_KAM_TO', 'Man Kam To Crossing', 22.5100, 114.1500, 'border'),
            
            # Customer locations
            'CENTRAL': Location('CENTRAL', 'Central Business District', 22.2800, 114.1580, 'customer'),
            'KOWLOON': Location('KOWLOON', 'Kowloon Commercial Area', 22.3200, 114.1700, 'customer'),
            'NEW_TERRITORIES': Location('NEW_TERRITORIES', 'New Territories Hub', 22.4000, 114.1000, 'customer')
        }
        return locations
    
    def _initialize_truck_fleet(self) -> Dict[str, Truck]:
        """Initialize truck fleet"""
        trucks = {}
        
        # 20ft chassis trucks
        for i in range(1, 21):
            trucks[f"TRUCK_20_{i:02d}"] = Truck(
                truck_id=f"TRUCK_20_{i:02d}",
                truck_type=TruckType.CHASSIS_20FT,
                current_location=self.locations['CT1']
            )
        
        # 40ft chassis trucks
        for i in range(1, 31):
            trucks[f"TRUCK_40_{i:02d}"] = Truck(
                truck_id=f"TRUCK_40_{i:02d}",
                truck_type=TruckType.CHASSIS_40FT,
                current_location=self.locations['CT2']
            )
        
        # Multi-trailer trucks
        for i in range(1, 11):
            trucks[f"MULTI_{i:02d}"] = Truck(
                truck_id=f"MULTI_{i:02d}",
                truck_type=TruckType.MULTI_TRAILER,
                current_location=self.locations['KWAI_CHUNG'],
                capacity=2
            )
        
        return trucks
    
    def schedule_job(self, job: TruckJob) -> bool:
        """Schedule a new truck job
        
        Args:
            job: TruckJob to schedule
            
        Returns:
            True if job was scheduled successfully
        """
        # Validate job
        if not self._validate_job(job):
            logger.warning(f"Invalid job {job.job_id}")
            return False
        
        # Add to pending jobs queue (priority queue)
        priority_score = self._calculate_job_priority(job)
        heapq.heappush(self.pending_jobs, (priority_score, job.job_id, job))
        
        logger.info(f"Scheduled job {job.job_id} with priority {priority_score}")
        return True
    
    def _validate_job(self, job: TruckJob) -> bool:
        """Validate job parameters"""
        if job.origin.location_id not in self.locations:
            return False
        if job.destination.location_id not in self.locations:
            return False
        if job.deadline and job.deadline < self.env.now:
            return False
        return True
    
    def _calculate_job_priority(self, job: TruckJob) -> float:
        """Calculate job priority score (lower = higher priority)"""
        base_priority = job.priority
        
        # Urgency factor based on deadline
        urgency_factor = 0
        if job.deadline:
            time_to_deadline = job.deadline - self.env.now
            if time_to_deadline < 2:  # Less than 2 hours
                urgency_factor = -10
            elif time_to_deadline < 6:  # Less than 6 hours
                urgency_factor = -5
        
        # Route type factor
        route_factor = {
            RouteType.PICKUP: 0,
            RouteType.DELIVERY: 1,
            RouteType.INTERNAL: -2,
            RouteType.REPOSITIONING: 3
        }.get(job.route_type, 0)
        
        return base_priority + urgency_factor + route_factor
    
    def _job_dispatcher_process(self):
        """Background process to dispatch jobs to available trucks"""
        while True:
            if self.pending_jobs:
                # Get highest priority job
                priority_score, job_id, job = heapq.heappop(self.pending_jobs)
                
                # Find best available truck
                best_truck = self._find_best_truck(job)
                
                if best_truck:
                    # Assign job to truck
                    self.env.process(self._execute_job(best_truck, job))
                    self.active_jobs[job.job_id] = job
                else:
                    # No available truck, put job back in queue
                    heapq.heappush(self.pending_jobs, (priority_score, job_id, job))
            
            # Wait before next dispatch cycle
            yield self.env.timeout(0.1)  # Check every 6 minutes
    
    def _find_best_truck(self, job: TruckJob) -> Optional[Truck]:
        """Find the best available truck for a job"""
        available_trucks = [truck for truck in self.trucks.values() 
                          if truck.is_available and truck.truck_type == job.truck_type]
        
        if not available_trucks:
            return None
        
        # Score trucks based on distance to pickup location
        best_truck = None
        best_score = float('inf')
        
        for truck in available_trucks:
            # Calculate distance to job origin
            distance = self._calculate_distance(truck.current_location, job.origin)
            
            # Consider fuel level and maintenance
            fuel_penalty = (100 - truck.fuel_level) * 0.1
            maintenance_penalty = max(0, 24 - truck.maintenance_due) * 0.5
            
            score = distance + fuel_penalty + maintenance_penalty
            
            if score < best_score:
                best_score = score
                best_truck = truck
        
        return best_truck
    
    def _execute_job(self, truck: Truck, job: TruckJob):
        """Execute a truck job"""
        start_time = self.env.now
        truck.is_available = False
        truck.current_job = job
        
        try:
            # Phase 1: Travel to pickup location
            if truck.current_location.location_id != job.origin.location_id:
                yield from self._travel_to_location(truck, job.origin)
            
            # Phase 2: Pickup/loading at origin
            yield from self._handle_location_operations(truck, job.origin, 'pickup')
            
            # Phase 3: Travel to destination
            yield from self._travel_to_location(truck, job.destination)
            
            # Phase 4: Delivery/unloading at destination
            yield from self._handle_location_operations(truck, job.destination, 'delivery')
            
            # Job completed successfully
            completion_time = self.env.now
            job_duration = completion_time - start_time
            
            # Record completion
            self.completed_jobs.append({
                'job_id': job.job_id,
                'truck_id': truck.truck_id,
                'start_time': start_time,
                'completion_time': completion_time,
                'duration': job_duration,
                'on_time': job.deadline is None or completion_time <= job.deadline
            })
            
            logger.info(f"Completed job {job.job_id} with truck {truck.truck_id} in {job_duration:.1f} hours")
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
        
        finally:
            # Clean up
            truck.is_available = True
            truck.current_job = None
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    def _travel_to_location(self, truck: Truck, destination: Location):
        """Simulate truck travel between locations"""
        if truck.current_location.location_id == destination.location_id:
            return
        
        # Get or compute route
        route = self._get_route(truck.current_location, destination)
        
        # Apply traffic conditions
        traffic_factor = self._get_traffic_factor(route)
        travel_time = route.estimated_time * traffic_factor
        
        # Simulate travel
        yield self.env.timeout(travel_time)
        
        # Update truck state
        truck.current_location = destination
        truck.fuel_level -= route.distance * 0.3  # Fuel consumption
        truck.maintenance_due -= travel_time
        
        # Update metrics
        self.metrics['total_distance_traveled'] += route.distance
        self.metrics['fuel_consumption'] += route.distance * 0.3
        
        logger.debug(f"Truck {truck.truck_id} traveled to {destination.name} in {travel_time:.1f} hours")
    
    def _handle_location_operations(self, truck: Truck, location: Location, operation: str):
        """Handle operations at a location (pickup/delivery)"""
        # Simulate gate processing and container operations
        processing_time = location.processing_time / 60.0  # Convert to hours
        
        # Add variability based on location type and traffic
        if location.location_type == 'port':
            processing_time *= random.uniform(0.8, 1.5)  # Port operations variability
        elif location.location_type == 'border':
            processing_time *= random.uniform(1.2, 2.0)  # Border crossing delays
        
        yield self.env.timeout(processing_time)
        
        logger.debug(f"Truck {truck.truck_id} completed {operation} at {location.name} in {processing_time:.1f} hours")
    
    def _get_route(self, origin: Location, destination: Location) -> Route:
        """Get or compute route between locations"""
        route_key = f"{origin.location_id}_{destination.location_id}"
        
        if route_key in self.route_cache:
            return self.route_cache[route_key]
        
        # Compute new route
        route = self._compute_route(origin, destination)
        self.route_cache[route_key] = route
        
        return route
    
    def _compute_route(self, origin: Location, destination: Location) -> Route:
        """Compute route between two locations"""
        # Simple distance calculation (Haversine formula)
        distance = self._calculate_distance(origin, destination)
        
        # Estimate travel time based on distance and road type
        base_speed = 40  # km/h average speed
        estimated_time = distance / base_speed
        
        # Adjust for location types
        if origin.location_type == 'port' or destination.location_type == 'port':
            estimated_time *= 1.2  # Port area congestion
        
        if origin.location_type == 'border' or destination.location_type == 'border':
            estimated_time *= 1.5  # Border crossing time
        
        return Route(
            route_id=f"{origin.location_id}_{destination.location_id}",
            origin=origin,
            destination=destination,
            distance=distance,
            estimated_time=estimated_time,
            fuel_cost=distance * 0.8,  # HKD per km
            toll_cost=distance * 0.2 if distance > 20 else 0
        )
    
    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate distance between two locations using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1 = np.radians(loc1.latitude), np.radians(loc1.longitude)
        lat2, lon2 = np.radians(loc2.latitude), np.radians(loc2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _get_traffic_factor(self, route: Route) -> float:
        """Get traffic factor for route"""
        condition = self.traffic_conditions[route.route_id]
        
        factors = {
            TrafficCondition.FREE_FLOW: 0.8,
            TrafficCondition.LIGHT: 0.9,
            TrafficCondition.MODERATE: 1.0,
            TrafficCondition.HEAVY: 1.3,
            TrafficCondition.CONGESTED: 1.8
        }
        
        return factors.get(condition, 1.0)
    
    def _traffic_update_process(self):
        """Background process to update traffic conditions"""
        while True:
            # Update traffic conditions based on time of day
            hour = (self.env.now % 24)
            
            # Rush hour patterns
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                # Rush hours - heavier traffic
                for route_id in self.route_cache.keys():
                    if random.random() < 0.6:
                        self.traffic_conditions[route_id] = random.choice([
                            TrafficCondition.HEAVY, TrafficCondition.CONGESTED
                        ])
            elif 22 <= hour or hour <= 6:
                # Night time - lighter traffic
                for route_id in self.route_cache.keys():
                    if random.random() < 0.7:
                        self.traffic_conditions[route_id] = random.choice([
                            TrafficCondition.FREE_FLOW, TrafficCondition.LIGHT
                        ])
            else:
                # Normal hours - moderate traffic
                for route_id in self.route_cache.keys():
                    if random.random() < 0.5:
                        self.traffic_conditions[route_id] = TrafficCondition.MODERATE
            
            # Update every hour
            yield self.env.timeout(1.0)
    
    def get_fleet_status(self) -> Dict[str, Any]:
        """Get current fleet status"""
        available_trucks = sum(1 for truck in self.trucks.values() if truck.is_available)
        active_trucks = len(self.trucks) - available_trucks
        
        truck_utilization = (active_trucks / len(self.trucks)) * 100 if self.trucks else 0
        
        return {
            'total_trucks': len(self.trucks),
            'available_trucks': available_trucks,
            'active_trucks': active_trucks,
            'truck_utilization': truck_utilization,
            'pending_jobs': len(self.pending_jobs),
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.completed_jobs),
            'performance_metrics': self.metrics
        }
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        if self.completed_jobs:
            # Average delivery time
            total_duration = sum(job['duration'] for job in self.completed_jobs)
            self.metrics['average_delivery_time'] = total_duration / len(self.completed_jobs)
            
            # On-time delivery rate
            on_time_jobs = sum(1 for job in self.completed_jobs if job['on_time'])
            self.metrics['on_time_delivery_rate'] = (on_time_jobs / len(self.completed_jobs)) * 100
            
            # Total jobs completed
            self.metrics['total_jobs_completed'] = len(self.completed_jobs)
        
        # Truck utilization
        active_trucks = sum(1 for truck in self.trucks.values() if not truck.is_available)
        self.metrics['truck_utilization'] = (active_trucks / len(self.trucks)) * 100 if self.trucks else 0