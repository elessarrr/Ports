# Core simulation modules
# Contains the main simulation engine components

from .port_simulation import PortSimulation
from .simulation_controller import SimulationController
from .berth_manager import BerthManager, Berth
from .ship_manager import ShipManager, Ship
from .container_handler import ContainerHandler

# Import from other modules for compatibility
try:
    from ..logistics.yard_manager import Container
except ImportError:
    Container = None

# Create aliases for compatibility
Vessel = Ship  # Vessel is an alias for Ship
BerthOptimizer = None  # Will be imported from AI module if needed
ResourceManager = None  # Will be implemented later

__all__ = [
    'PortSimulation',
    'SimulationController', 
    'BerthManager',
    'ShipManager',
    'ContainerHandler',
    'Berth',
    'Ship',
    'Vessel',
    'Container',
    'BerthOptimizer',
    'ResourceManager'
]