"""Configuration settings for Hong Kong Port Digital Twin

This module contains all configuration parameters for the port simulation.
Modify these values to adjust simulation behavior without changing core logic.

Based on Hong Kong Port's actual specifications and operational characteristics.
"""

# Port Configuration
PORT_CONFIG = {
    'num_berths': 24,  # Hong Kong has multiple container terminals with 24 berths
    'operating_hours': 24,   # Hours per day
    'port_name': 'Hong Kong Port',
    'total_annual_capacity': 18000000,  # TEU per year
}

# Detailed Berth Configurations
# Based on Hong Kong's major container terminals (Kwai Tsing, etc.)
BERTH_CONFIGS = [
    # Terminal 1 - Large Container Berths
    {'berth_id': 1, 'berth_name': 'CT1-North', 'max_capacity_teu': 18000, 'crane_count': 6, 'berth_type': 'container'},
    {'berth_id': 2, 'berth_name': 'CT1-South', 'max_capacity_teu': 18000, 'crane_count': 6, 'berth_type': 'container'},
    {'berth_id': 3, 'berth_name': 'CT2-North', 'max_capacity_teu': 20000, 'crane_count': 8, 'berth_type': 'container'},
    {'berth_id': 4, 'berth_name': 'CT2-South', 'max_capacity_teu': 20000, 'crane_count': 8, 'berth_type': 'container'},
    
    # Terminal 2 - Ultra Large Container Vessels (ULCV)
    {'berth_id': 5, 'berth_name': 'CT3-ULCV-1', 'max_capacity_teu': 24000, 'crane_count': 10, 'berth_type': 'container'},
    {'berth_id': 6, 'berth_name': 'CT3-ULCV-2', 'max_capacity_teu': 24000, 'crane_count': 10, 'berth_type': 'container'},
    {'berth_id': 7, 'berth_name': 'CT4-ULCV-1', 'max_capacity_teu': 22000, 'crane_count': 9, 'berth_type': 'container'},
    {'berth_id': 8, 'berth_name': 'CT4-ULCV-2', 'max_capacity_teu': 22000, 'crane_count': 9, 'berth_type': 'container'},
    
    # Terminal 3 - Medium Container Berths
    {'berth_id': 9, 'berth_name': 'CT5-Med-1', 'max_capacity_teu': 15000, 'crane_count': 5, 'berth_type': 'container'},
    {'berth_id': 10, 'berth_name': 'CT5-Med-2', 'max_capacity_teu': 15000, 'crane_count': 5, 'berth_type': 'container'},
    {'berth_id': 11, 'berth_name': 'CT6-Med-1', 'max_capacity_teu': 16000, 'crane_count': 6, 'berth_type': 'container'},
    {'berth_id': 12, 'berth_name': 'CT6-Med-2', 'max_capacity_teu': 16000, 'crane_count': 6, 'berth_type': 'container'},
    
    # Terminal 4 - Mixed Use Berths
    {'berth_id': 13, 'berth_name': 'CT7-Mixed-1', 'max_capacity_teu': 12000, 'crane_count': 4, 'berth_type': 'mixed'},
    {'berth_id': 14, 'berth_name': 'CT7-Mixed-2', 'max_capacity_teu': 12000, 'crane_count': 4, 'berth_type': 'mixed'},
    {'berth_id': 15, 'berth_name': 'CT8-Mixed-1', 'max_capacity_teu': 14000, 'crane_count': 5, 'berth_type': 'mixed'},
    {'berth_id': 16, 'berth_name': 'CT8-Mixed-2', 'max_capacity_teu': 14000, 'crane_count': 5, 'berth_type': 'mixed'},
    
    # Bulk Cargo Terminals
    {'berth_id': 17, 'berth_name': 'Bulk-North-1', 'max_capacity_teu': 8000, 'crane_count': 3, 'berth_type': 'bulk'},
    {'berth_id': 18, 'berth_name': 'Bulk-North-2', 'max_capacity_teu': 8000, 'crane_count': 3, 'berth_type': 'bulk'},
    {'berth_id': 19, 'berth_name': 'Bulk-South-1', 'max_capacity_teu': 10000, 'crane_count': 4, 'berth_type': 'bulk'},
    {'berth_id': 20, 'berth_name': 'Bulk-South-2', 'max_capacity_teu': 10000, 'crane_count': 4, 'berth_type': 'bulk'},
    
    # Specialized Berths
    {'berth_id': 21, 'berth_name': 'Special-1', 'max_capacity_teu': 6000, 'crane_count': 2, 'berth_type': 'mixed'},
    {'berth_id': 22, 'berth_name': 'Special-2', 'max_capacity_teu': 6000, 'crane_count': 2, 'berth_type': 'mixed'},
    {'berth_id': 23, 'berth_name': 'Maintenance-1', 'max_capacity_teu': 5000, 'crane_count': 2, 'berth_type': 'mixed'},
    {'berth_id': 24, 'berth_name': 'Emergency-1', 'max_capacity_teu': 8000, 'crane_count': 3, 'berth_type': 'mixed'},
]

# Ship Types and Characteristics
# Based on actual vessel types calling at Hong Kong Port
SHIP_TYPES = {
    'container': {
        'min_size': 1000,
        'max_size': 24000,  # Ultra Large Container Vessels (ULCV)
        'processing_rate': 120,  # TEU per hour (improved with modern cranes)
        'typical_sizes': [1500, 3000, 6000, 8000, 12000, 18000, 24000],
        'arrival_probability': 0.75,  # 75% of ships are container ships
    },
    'bulk': {
        'min_size': 5000,
        'max_size': 80000,  # Large bulk carriers
        'processing_rate': 250,  # Tons per hour
        'typical_sizes': [10000, 25000, 40000, 60000, 80000],
        'arrival_probability': 0.20,  # 20% of ships are bulk carriers
    },
    'mixed': {
        'min_size': 2000,
        'max_size': 15000,  # Multi-purpose vessels
        'processing_rate': 80,  # TEU per hour (slower due to mixed cargo)
        'typical_sizes': [3000, 6000, 9000, 12000, 15000],
        'arrival_probability': 0.05,  # 5% of ships are mixed cargo
    }
}

# Simulation Parameters
# Based on Hong Kong Port's actual operational patterns
SIMULATION_CONFIG = {
    'time_unit': 'hours',
    'default_duration': 168,  # 1 week in hours
    'ship_arrival_rate': 1.5,   # Ships per hour (realistic for HK Port)
    'peak_hours': [8, 9, 10, 14, 15, 16],  # Peak arrival times
    'peak_multiplier': 1.8,  # Arrival rate multiplier during peak hours
    'weekend_multiplier': 0.7,  # Reduced activity on weekends
    'random_seed': 42,  # For reproducible simulations
    'warm_up_period': 24,  # Hours to warm up simulation
    'use_historical_parameters': True,  # Enable historical data-driven parameters
}

def get_enhanced_simulation_config():
    """Get simulation configuration enhanced with historical data patterns.
    
    This function combines the base SIMULATION_CONFIG with parameters extracted
    from 14+ years of historical Hong Kong port data to create more realistic
    simulation scenarios.
    
    Returns:
        Dict: Enhanced simulation configuration with historical patterns
    """
    enhanced_config = SIMULATION_CONFIG.copy()
    
    # Only enhance if historical parameters are enabled
    if not enhanced_config.get('use_historical_parameters', False):
        return enhanced_config
    
    try:
        # Import here to avoid circular imports
        from hk_port_digital_twin.src.utils.data_loader import extract_historical_simulation_parameters
        
        # Extract historical parameters
        historical_params = extract_historical_simulation_parameters()
        
        if historical_params:
            # Update ship arrival rate with historical data
            if 'ship_arrival_rate' in historical_params:
                enhanced_config['ship_arrival_rate'] = historical_params['ship_arrival_rate']
            
            # Update seasonal patterns
            seasonal_patterns = historical_params.get('seasonal_patterns', {})
            if seasonal_patterns:
                enhanced_config['peak_multiplier'] = seasonal_patterns.get('peak_multiplier', enhanced_config['peak_multiplier'])
                enhanced_config['seasonal_low_multiplier'] = seasonal_patterns.get('low_multiplier', 0.7)
                enhanced_config['peak_months'] = seasonal_patterns.get('peak_months', [11, 12])
                enhanced_config['low_months'] = seasonal_patterns.get('low_months', [6, 7])
            
            # Add ship type distribution from historical data
            ship_type_dist = historical_params.get('ship_type_distribution', {})
            if ship_type_dist:
                enhanced_config['historical_ship_type_distribution'] = ship_type_dist
            
            # Add operational efficiency parameters
            operational_efficiency = historical_params.get('operational_efficiency', {})
            if operational_efficiency:
                enhanced_config['processing_rate_multiplier'] = operational_efficiency.get('processing_rate_multiplier', 1.0)
                enhanced_config['crane_efficiency_factor'] = operational_efficiency.get('crane_efficiency_factor', 1.0)
                enhanced_config['target_berth_utilization'] = operational_efficiency.get('berth_utilization_target', 0.85)
            
            # Add metadata about historical analysis
            analysis_metadata = historical_params.get('analysis_metadata', {})
            enhanced_config['historical_data_metadata'] = {
                'data_period': historical_params.get('data_period', 'Unknown'),
                'years_of_data': analysis_metadata.get('years_of_data', 0),
                'total_data_points': analysis_metadata.get('total_data_points', 0),
                'trend_direction': analysis_metadata.get('trend_direction', 'stable'),
                'seasonality_strength': analysis_metadata.get('seasonality_strength', 0.1)
            }
            
            # Add volume characteristics
            volume_chars = historical_params.get('volume_characteristics', {})
            if volume_chars:
                enhanced_config['historical_volume_characteristics'] = volume_chars
            
            enhanced_config['enhanced_with_historical_data'] = True
            
        else:
            enhanced_config['enhanced_with_historical_data'] = False
            
    except Exception as e:
        # Log error but don't fail - fall back to base configuration
        print(f"Warning: Could not enhance simulation config with historical data: {e}")
        enhanced_config['enhanced_with_historical_data'] = False
    
    return enhanced_config

# Operational Parameters
OPERATIONAL_CONFIG = {
    'average_docking_time': 0.5,  # Hours to dock a ship
    'average_undocking_time': 0.3,  # Hours to undock a ship
    'berth_changeover_time': 0.2,  # Hours between ships at same berth
    'crane_setup_time': 0.1,  # Hours to setup cranes for new ship
    'weather_delay_probability': 0.05,  # 5% chance of weather delays
    'equipment_failure_probability': 0.02,  # 2% chance of equipment issues
}

# Performance Metrics Targets
# Based on Hong Kong Port's KPIs
PERFORMANCE_TARGETS = {
    'average_waiting_time': 2.0,  # Target: max 2 hours waiting
    'berth_utilization_rate': 0.85,  # Target: 85% berth utilization
    'ship_turnaround_time': 24.0,  # Target: 24 hours average turnaround
    'container_moves_per_hour': 120,  # Target: 120 TEU/hour
    'queue_length_threshold': 5,  # Alert if more than 5 ships waiting
}

# Dashboard Preferences Configuration
# Controls the behavior and layout of the dashboard interface
DASHBOARD_PREFERENCES = {
    # Tab Structure Settings
    'use_consolidated_scenarios': True,  # Use new consolidated scenarios tab
    'scenarios_sections_expanded': True,  # Default expanded state for sections
    'navigation_style': 'anchor_links',  # Navigation style: 'anchor_links' or 'sidebar'
    
    # Section Display Settings
    'show_section_navigation': True,  # Show navigation sidebar in consolidated tab
    'enable_expand_collapse_all': True,  # Show expand/collapse all buttons
    'section_auto_scroll': True,  # Auto-scroll to section when navigating
    
    # User Experience Settings
    'remember_section_states': True,  # Remember expanded/collapsed states
    'show_section_descriptions': True,  # Show descriptive text for each section
    'enable_quick_export': True,  # Enable quick export buttons in sections
    
    # Performance Settings
    'lazy_load_sections': False,  # Load sections on-demand (future feature)
    'cache_section_data': True,  # Cache section data for better performance
}

# Default Section States for Consolidated Scenarios Tab
# Controls which sections are expanded by default
DEFAULT_SECTION_STATES = {
    'overview': True,     # Scenario Selection & Overview
    'operations': True,   # Operational Impact
    'analytics': True,    # Performance Analytics
    'cargo': True,        # Cargo Analysis
    'advanced': True,     # Advanced Analysis
}

def get_dashboard_preferences():
    """Get current dashboard preferences with user overrides.
    
    This function returns the dashboard preferences, allowing for user-specific
    overrides stored in session state or user configuration files.
    
    Returns:
        Dict: Current dashboard preferences
    """
    preferences = DASHBOARD_PREFERENCES.copy()
    
    # In a future implementation, this could load user-specific preferences
    # from a database, configuration file, or session state
    
    return preferences

def get_default_section_states():
    """Get default section states for the consolidated scenarios tab.
    
    Returns:
        Dict: Default expanded/collapsed states for each section
    """
    return DEFAULT_SECTION_STATES.copy()