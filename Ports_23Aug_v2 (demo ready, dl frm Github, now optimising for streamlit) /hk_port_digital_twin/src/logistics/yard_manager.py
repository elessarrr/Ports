"""Container Yard Management System for Hong Kong Port Digital Twin

This module simulates container yard operations including:
- Container storage allocation and optimization
- Yard block management and utilization tracking
- Container retrieval and positioning operations
- Yard equipment coordination (RTGs, reach stackers)
- Storage time tracking and demurrage calculations

The system integrates with the main simulation to provide realistic
yard-side logistics modeling for complete port operations.
"""

import logging
import simpy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class ContainerType(Enum):
    """Container types for yard management"""
    TWENTY_FOOT = "20ft"
    FORTY_FOOT = "40ft"
    FORTY_FOOT_HC = "40ft_hc"  # High cube
    REFRIGERATED = "reefer"
    TANK = "tank"
    FLAT_RACK = "flat_rack"

class ContainerStatus(Enum):
    """Container status in yard"""
    IMPORT_FULL = "import_full"
    IMPORT_EMPTY = "import_empty"
    EXPORT_FULL = "export_full"
    EXPORT_EMPTY = "export_empty"
    TRANSSHIPMENT = "transshipment"
    STORAGE = "storage"

@dataclass
class Container:
    """Container entity in the yard"""
    container_id: str
    container_type: ContainerType
    status: ContainerStatus
    arrival_time: float
    ship_id: Optional[str] = None
    destination: Optional[str] = None
    weight: float = 0.0  # in tons
    is_hazardous: bool = False
    is_refrigerated: bool = False
    storage_days: int = 0
    yard_block: Optional[str] = None
    stack_position: Optional[Tuple[int, int, int]] = None  # (row, bay, tier)
    
@dataclass
class YardBlock:
    """Yard block configuration"""
    block_id: str
    rows: int
    bays: int
    tiers: int
    container_types: List[ContainerType]
    is_reefer_block: bool = False
    rtg_count: int = 1
    capacity: int = field(init=False)
    current_containers: Dict[Tuple[int, int, int], Container] = field(default_factory=dict)
    
    def __post_init__(self):
        self.capacity = self.rows * self.bays * self.tiers
    
    @property
    def utilization(self) -> float:
        """Calculate current utilization percentage"""
        return len(self.current_containers) / self.capacity * 100
    
    @property
    def available_positions(self) -> List[Tuple[int, int, int]]:
        """Get list of available positions"""
        all_positions = [
            (row, bay, tier)
            for row in range(self.rows)
            for bay in range(self.bays)
            for tier in range(self.tiers)
        ]
        return [pos for pos in all_positions if pos not in self.current_containers]

class ContainerYardManager:
    """Container Yard Management System
    
    Manages container storage, retrieval, and yard operations
    with realistic operational constraints and optimization.
    """
    
    def __init__(self, env: simpy.Environment):
        """Initialize the yard manager
        
        Args:
            env: SimPy environment for simulation timing
        """
        self.env = env
        self.yard_blocks = self._initialize_yard_blocks()
        self.containers = {}  # container_id -> Container
        self.rtg_resources = {}  # block_id -> simpy.Resource
        self.operations_history = []
        self.performance_metrics = {
            'total_containers_handled': 0,
            'average_storage_time': 0.0,
            'yard_utilization': 0.0,
            'retrieval_efficiency': 0.0,
            'equipment_utilization': 0.0
        }
        
        # Initialize RTG resources for each block
        for block in self.yard_blocks.values():
            self.rtg_resources[block.block_id] = simpy.Resource(env, capacity=block.rtg_count)
    
    def _initialize_yard_blocks(self) -> Dict[str, YardBlock]:
        """Initialize yard block configuration based on HK port layout"""
        blocks = {}
        
        # Container Terminal 1 blocks
        for i in range(1, 9):
            blocks[f"CT1-{i:02d}"] = YardBlock(
                block_id=f"CT1-{i:02d}",
                rows=6,
                bays=40,
                tiers=4,
                container_types=[ContainerType.TWENTY_FOOT, ContainerType.FORTY_FOOT],
                rtg_count=2
            )
        
        # Container Terminal 2 blocks
        for i in range(1, 12):
            blocks[f"CT2-{i:02d}"] = YardBlock(
                block_id=f"CT2-{i:02d}",
                rows=8,
                bays=45,
                tiers=5,
                container_types=[ContainerType.TWENTY_FOOT, ContainerType.FORTY_FOOT, ContainerType.FORTY_FOOT_HC],
                rtg_count=3
            )
        
        # Reefer blocks
        for i in range(1, 4):
            blocks[f"REEFER-{i:02d}"] = YardBlock(
                block_id=f"REEFER-{i:02d}",
                rows=4,
                bays=20,
                tiers=3,
                container_types=[ContainerType.REFRIGERATED],
                is_reefer_block=True,
                rtg_count=1
            )
        
        return blocks
    
    def store_container(self, container: Container) -> simpy.Event:
        """Store a container in the yard
        
        Args:
            container: Container to store
            
        Returns:
            SimPy event for the storage operation
        """
        return self.env.process(self._store_container_process(container))
    
    def _store_container_process(self, container: Container):
        """Process for storing a container in the yard"""
        start_time = self.env.now
        
        # Find optimal yard block
        target_block = self._find_optimal_block(container)
        if not target_block:
            logger.warning(f"No available space for container {container.container_id}")
            return
        
        # Request RTG equipment
        with self.rtg_resources[target_block.block_id].request() as rtg_request:
            yield rtg_request
            
            # Find available position in block
            available_positions = target_block.available_positions
            if not available_positions:
                logger.warning(f"Block {target_block.block_id} is full")
                return
            
            # Select optimal position (ground level first, then stack up)
            position = self._select_optimal_position(available_positions, container)
            
            # Simulate storage time (RTG movement and positioning)
            storage_time = self._calculate_storage_time(position, target_block)
            yield self.env.timeout(storage_time)
            
            # Store container
            container.yard_block = target_block.block_id
            container.stack_position = position
            target_block.current_containers[position] = container
            self.containers[container.container_id] = container
            
            # Record operation
            operation_record = {
                'operation_type': 'storage',
                'container_id': container.container_id,
                'block_id': target_block.block_id,
                'position': position,
                'start_time': start_time,
                'end_time': self.env.now,
                'operation_time': self.env.now - start_time,
                'rtg_used': True
            }
            self.operations_history.append(operation_record)
            
            logger.info(f"Stored container {container.container_id} in block {target_block.block_id} at position {position}")
    
    def retrieve_container(self, container_id: str) -> simpy.Event:
        """Retrieve a container from the yard
        
        Args:
            container_id: ID of container to retrieve
            
        Returns:
            SimPy event for the retrieval operation
        """
        return self.env.process(self._retrieve_container_process(container_id))
    
    def _retrieve_container_process(self, container_id: str):
        """Process for retrieving a container from the yard"""
        start_time = self.env.now
        
        if container_id not in self.containers:
            logger.warning(f"Container {container_id} not found in yard")
            return
        
        container = self.containers[container_id]
        block = self.yard_blocks[container.yard_block]
        
        # Request RTG equipment
        with self.rtg_resources[block.block_id].request() as rtg_request:
            yield rtg_request
            
            # Check if container is accessible (no containers stacked above)
            if not self._is_container_accessible(container, block):
                # Need to move blocking containers first
                yield from self._handle_blocking_containers(container, block)
            
            # Simulate retrieval time
            retrieval_time = self._calculate_retrieval_time(container.stack_position, block)
            yield self.env.timeout(retrieval_time)
            
            # Remove container from yard
            del block.current_containers[container.stack_position]
            del self.containers[container_id]
            
            # Calculate storage duration
            storage_duration = self.env.now - container.arrival_time
            
            # Record operation
            operation_record = {
                'operation_type': 'retrieval',
                'container_id': container_id,
                'block_id': block.block_id,
                'position': container.stack_position,
                'start_time': start_time,
                'end_time': self.env.now,
                'operation_time': self.env.now - start_time,
                'storage_duration': storage_duration,
                'rtg_used': True
            }
            self.operations_history.append(operation_record)
            
            logger.info(f"Retrieved container {container_id} from block {block.block_id} after {storage_duration:.1f} hours")
    
    def _find_optimal_block(self, container: Container) -> Optional[YardBlock]:
        """Find the optimal yard block for container storage"""
        suitable_blocks = []
        
        for block in self.yard_blocks.values():
            # Check container type compatibility
            if container.container_type not in block.container_types:
                continue
            
            # Check reefer requirements
            if container.is_refrigerated and not block.is_reefer_block:
                continue
            if not container.is_refrigerated and block.is_reefer_block:
                continue
            
            # Check availability
            if len(block.available_positions) > 0:
                suitable_blocks.append(block)
        
        if not suitable_blocks:
            return None
        
        # Select block with lowest utilization
        return min(suitable_blocks, key=lambda b: b.utilization)
    
    def _select_optimal_position(self, positions: List[Tuple[int, int, int]], container: Container) -> Tuple[int, int, int]:
        """Select optimal position within a block"""
        # Prefer ground level positions (tier 0)
        ground_positions = [pos for pos in positions if pos[2] == 0]
        if ground_positions:
            return random.choice(ground_positions)
        
        # Otherwise, select lowest available tier
        return min(positions, key=lambda pos: pos[2])
    
    def _calculate_storage_time(self, position: Tuple[int, int, int], block: YardBlock) -> float:
        """Calculate time required for storage operation"""
        row, bay, tier = position
        
        # Base time for RTG movement
        base_time = 3.0  # minutes
        
        # Add time based on position
        row_time = row * 0.5  # Time to move to row
        bay_time = bay * 0.2  # Time to move along bay
        tier_time = tier * 1.0  # Time to stack up
        
        total_time = base_time + row_time + bay_time + tier_time
        return total_time / 60.0  # Convert to hours
    
    def _calculate_retrieval_time(self, position: Tuple[int, int, int], block: YardBlock) -> float:
        """Calculate time required for retrieval operation"""
        # Similar to storage time but slightly faster
        return self._calculate_storage_time(position, block) * 0.8
    
    def _is_container_accessible(self, container: Container, block: YardBlock) -> bool:
        """Check if container can be accessed without moving others"""
        row, bay, tier = container.stack_position
        
        # Check if any containers are stacked above
        for t in range(tier + 1, block.tiers):
            if (row, bay, t) in block.current_containers:
                return False
        
        return True
    
    def _handle_blocking_containers(self, target_container: Container, block: YardBlock):
        """Handle containers blocking access to target container"""
        row, bay, tier = target_container.stack_position
        
        # Find all blocking containers
        blocking_containers = []
        for t in range(tier + 1, block.tiers):
            pos = (row, bay, t)
            if pos in block.current_containers:
                blocking_containers.append(block.current_containers[pos])
        
        # Move blocking containers to temporary positions
        for blocking_container in reversed(blocking_containers):  # Top to bottom
            # Find temporary position
            temp_positions = block.available_positions
            if temp_positions:
                temp_pos = temp_positions[0]
                
                # Move container
                old_pos = blocking_container.stack_position
                del block.current_containers[old_pos]
                
                blocking_container.stack_position = temp_pos
                block.current_containers[temp_pos] = blocking_container
                
                # Simulate move time
                move_time = self._calculate_storage_time(temp_pos, block)
                yield self.env.timeout(move_time)
                
                logger.info(f"Moved blocking container {blocking_container.container_id} from {old_pos} to {temp_pos}")
    
    def get_yard_statistics(self) -> Dict[str, Any]:
        """Get current yard statistics"""
        total_capacity = sum(block.capacity for block in self.yard_blocks.values())
        total_containers = sum(len(block.current_containers) for block in self.yard_blocks.values())
        
        block_stats = {}
        for block_id, block in self.yard_blocks.items():
            block_stats[block_id] = {
                'utilization': block.utilization,
                'containers': len(block.current_containers),
                'capacity': block.capacity,
                'rtg_count': block.rtg_count
            }
        
        return {
            'overall_utilization': (total_containers / total_capacity) * 100,
            'total_containers': total_containers,
            'total_capacity': total_capacity,
            'total_operations': len(self.operations_history),
            'block_statistics': block_stats,
            'performance_metrics': self.performance_metrics
        }
    
    def update_performance_metrics(self):
        """Update performance metrics based on operations history"""
        if not self.operations_history:
            return
        
        # Calculate average storage time
        storage_times = [op['storage_duration'] for op in self.operations_history 
                        if op['operation_type'] == 'retrieval' and 'storage_duration' in op]
        
        if storage_times:
            self.performance_metrics['average_storage_time'] = np.mean(storage_times)
        
        # Calculate yard utilization
        total_capacity = sum(block.capacity for block in self.yard_blocks.values())
        total_containers = sum(len(block.current_containers) for block in self.yard_blocks.values())
        self.performance_metrics['yard_utilization'] = (total_containers / total_capacity) * 100
        
        # Calculate retrieval efficiency (operations per hour)
        if self.env.now > 0:
            total_operations = len([op for op in self.operations_history if op['operation_type'] == 'retrieval'])
            self.performance_metrics['retrieval_efficiency'] = total_operations / self.env.now
        
        # Update total containers handled
        self.performance_metrics['total_containers_handled'] = len(self.operations_history)