# Dashboard modules
# Contains web interface and visualization components

# Create aliases for compatibility with main module imports
DashboardApp = None  # To be implemented as a class wrapper

def create_dashboard():
    """Create and run the dashboard"""
    from .streamlit_app import main
    return main()

def run_dashboard():
    """Run the dashboard"""
    from .streamlit_app import main
    return main()

def get_marine_traffic_integration():
    """Get MarineTrafficIntegration class"""
    try:
        from .marine_traffic_integration import MarineTrafficIntegration
        return MarineTrafficIntegration
    except ImportError:
        return None

# For backward compatibility
MarineTrafficIntegration = get_marine_traffic_integration()

__all__ = [
    'DashboardApp',
    'create_dashboard',
    'run_dashboard',
    'MarineTrafficIntegration'
]