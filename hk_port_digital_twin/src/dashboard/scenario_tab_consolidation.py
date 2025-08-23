"""Consolidated Scenarios Tab Module

This module contains all logic for the new consolidated Scenarios tab that brings together
all scenario-dependent dashboard features under a single, organized interface.

The module provides:
- Unified scenario selection and overview
- Operational impact analysis (ships & berths)
- Performance analytics
- Cargo analysis
- Advanced scenario modeling

All content is organized in expandable sections with anchor navigation for improved UX.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
import sys
import os

# Add the config directory to the path to import settings
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))
try:
    from settings import get_dashboard_preferences, get_default_section_states
except ImportError:
    # Fallback if settings module is not available
    def get_dashboard_preferences():
        # Import streamlit here to avoid circular imports
        import streamlit as st
        return {
            'show_section_descriptions': True,
            'enable_expand_collapse_all': True,
            'show_section_navigation': st.session_state.get('show_section_navigation', True),
            'remember_section_states': st.session_state.get('remember_section_states', True),
            'scenarios_sections_expanded': st.session_state.get('scenarios_sections_expanded', False),
            'section_auto_scroll': True,
            'enable_quick_export': True
        }
    
    def get_default_section_states():
        return {
            'overview': True,
            'operations': False,
            'analytics': False,
            'cargo': False,
            'advanced': False
        }

# Import existing visualization and data functions
# These will be imported as needed from existing modules


class ConsolidatedScenariosTab:
    """Main class for managing the consolidated scenarios tab functionality."""
    
    def __init__(self):
        """Initialize the consolidated scenarios tab."""
        self.sections = {
            'overview': {
                # 'title': 'Scenario Selection & Overview',
                # 'icon': '',
                # 'description': 'Select and configure simulation scenarios with key performance indicators'
            },
            'operations': {
                # 'title': 'Operational Impact',
                # 'icon': '🚢', 
                # 'description': 'Monitor ships, berths, and operational metrics affected by scenarios'
            },
            'analytics': {
                # 'title': 'Performance Analytics',
                # 'icon': '📈',
                # 'description': 'Analyze performance trends and KPIs across different scenarios'
            },
            'cargo': {
                # 'title': 'Cargo Analysis',
                # 'icon': '📦',
                # 'description': 'Track cargo statistics and throughput metrics by scenario'
            },
            'advanced': {
                # 'title': 'Advanced Analysis',
                # 'icon': '🔬',
                # 'description': 'Deep-dive scenario comparisons and advanced simulation features'
            }
        }
        
        # Load preferences
        self.preferences = get_dashboard_preferences()
        self.default_states = get_default_section_states()
        
    def render_consolidated_tab(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render the complete consolidated scenarios tab.
        
        Args:
            scenario_data: Optional scenario data to display
        """
        self._initialize_session_state()
        
        # Header with controls
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.header("🎯 Scenarios Dashboard")
            if self.preferences.get('show_section_descriptions', True):
                st.markdown("*Comprehensive scenario analysis and planning for Hong Kong Port operations*")
        
        with col2:
            if self.preferences.get('enable_expand_collapse_all', True):
                if st.button("📖 Expand All", key="expand_all"):
                    self._expand_all_sections()
        
        with col3:
            if self.preferences.get('enable_expand_collapse_all', True):
                if st.button("📕 Collapse All", key="collapse_all"):
                    self._collapse_all_sections()
        
        # Render section navigation if enabled
        if self.preferences.get('show_section_navigation', True):
            self._render_section_navigation()
        
        # Render validation section
        self._render_validation_section()
        
        # Render all sections
        for section_key, section_info in self.sections.items():
            self._render_section(section_key, section_info, scenario_data)
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables for the consolidated tab."""
        # Check if we need to reinitialize section states due to preference changes
        should_reinitialize = False
        
        # Check if preferences have changed
        if 'last_remember_section_states' not in st.session_state:
            st.session_state.last_remember_section_states = self.preferences.get('remember_section_states', True)
            should_reinitialize = True
        elif st.session_state.last_remember_section_states != self.preferences.get('remember_section_states', True):
            st.session_state.last_remember_section_states = self.preferences.get('remember_section_states', True)
            should_reinitialize = True
            
        if 'last_scenarios_sections_expanded' not in st.session_state:
            st.session_state.last_scenarios_sections_expanded = self.preferences.get('scenarios_sections_expanded', False)
            should_reinitialize = True
        elif st.session_state.last_scenarios_sections_expanded != self.preferences.get('scenarios_sections_expanded', False):
            st.session_state.last_scenarios_sections_expanded = self.preferences.get('scenarios_sections_expanded', False)
            should_reinitialize = True
        
        # Initialize or reinitialize section states based on preferences
        if 'consolidated_sections_state' not in st.session_state or should_reinitialize:
            if self.preferences.get('remember_section_states', True):
                # Use default states from settings
                st.session_state.consolidated_sections_state = self.default_states.copy()
            else:
                # Use preference setting for all sections
                expanded_default = self.preferences.get('scenarios_sections_expanded', False)
                st.session_state.consolidated_sections_state = {
                    section: expanded_default for section in self.sections.keys()
                }
            
        if 'active_section' not in st.session_state:
            st.session_state.active_section = 'overview'
            
        # Initialize anchor navigation state
        if 'section_anchors' not in st.session_state:
            st.session_state.section_anchors = {}
            
        # Section data cache
        if 'section_data_cache' not in st.session_state:
            st.session_state.section_data_cache = {}
    
    def render(self) -> None:
        """Render the consolidated scenarios tab (legacy method for backward compatibility)."""
        self.render_consolidated_tab()
    
    def _render_section_navigation(self) -> None:
        """Render the section navigation sidebar with anchor links."""
        with st.sidebar:
            st.subheader("📋 Section Navigation")
            
            # Quick status overview
            expanded_count = sum(1 for state in st.session_state.consolidated_sections_state.values() if state)
            total_count = len(self.sections)
            st.caption(f"📊 {expanded_count}/{total_count} sections expanded")
            
            st.markdown("---")
            
            # Navigation buttons with status indicators
            for section_key, section_info in self.sections.items():
                is_expanded = st.session_state.consolidated_sections_state.get(section_key, False)
                status_icon = "📖" if is_expanded else "📕"
                
                button_label = f"{status_icon} {section_info.get('icon', '')} {section_info.get('title', section_key.title())}"
                
                if st.button(button_label, key=f"nav_{section_key}", use_container_width=True):
                    st.session_state.active_section = section_key
                    # Toggle section state when navigating
                    if self.preferences.get('section_auto_scroll', True):
                        st.session_state.consolidated_sections_state[section_key] = True
                        st.rerun()
    
    def _expand_all_sections(self) -> None:
        """Expand all sections."""
        for section_key in self.sections.keys():
            st.session_state.consolidated_sections_state[section_key] = True
        st.rerun()
    
    def _collapse_all_sections(self) -> None:
        """Collapse all sections."""
        for section_key in self.sections.keys():
            st.session_state.consolidated_sections_state[section_key] = False
        st.rerun()
    
    def _render_section(self, section_key: str, section_info: Dict[str, str], scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render an individual section with enhanced features.
        
        Args:
            section_key: The key identifier for the section
            section_info: Dictionary containing section metadata
            scenario_data: Optional scenario data to display
        """
        # Get current state
        is_expanded = st.session_state.consolidated_sections_state.get(section_key, False)
        
        # Get current scenario for visual indicators
        current_scenario = self._get_current_scenario()
        scenario_badge = self._get_scenario_badge(current_scenario)
        
        # Create section header with scenario indicator
        section_title = f"{section_info.get('icon', '')} {section_info.get('title', section_key.title())} {scenario_badge}"
        
        # Add anchor point for navigation
        st.markdown(f'<div id="section-{section_key}"></div>', unsafe_allow_html=True)
        
        # Create expandable section
        with st.expander(section_title, expanded=is_expanded):
            # Show current scenario context at the top of each section
            if section_key != 'overview':  # Skip for overview since it already shows scenario selection
                scenario_color = self._get_scenario_color(current_scenario)
                scenario_border = self._get_scenario_border_color(current_scenario)
                st.markdown(
                    f"""<div style="padding: 6px 10px; border-radius: 3px; background-color: {scenario_color}; 
                    border-left: 3px solid {scenario_border}; margin-bottom: 10px; font-size: 0.85em;">
                    <strong>Scenario Context:</strong> {scenario_badge} {current_scenario}
                    </div>""",
                    unsafe_allow_html=True
                )
            
            # Show description if enabled
            if self.preferences.get('show_section_descriptions', True):
                if section_info.get('description'):
                    st.markdown(f"*{section_info['description']}*")
                st.markdown("---")
            
            # Render section content
            if section_key == 'overview':
                self.render_scenario_overview_section(scenario_data)
            elif section_key == 'operations':
                self.render_operational_impact_section(scenario_data)
            elif section_key == 'analytics':
                self.render_performance_analytics_section(scenario_data)
            elif section_key == 'cargo':
                self.render_cargo_analysis_section(scenario_data)
            elif section_key == 'advanced':
                self.render_advanced_analysis_section(scenario_data)
            
            # Add quick export button if enabled
            if self.preferences.get('enable_quick_export', True):
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("Export Analysis", key=f"export_{section_key}"):
                        self._export_section_data(section_key)
                with col2:
                    if st.button(f"🔗 Copy Link", key=f"link_{section_key}"):
                        self._copy_section_link(section_key)
    
    def _export_section_data(self, section_key: str) -> None:
        """Export data for a specific section.
        
        Args:
            section_key: The section to export data for
        """
        st.info(f"Export functionality for {section_key} section will be implemented.")
        # TODO: Implement actual export functionality
    
    def _copy_section_link(self, section_key: str) -> None:
        """Copy anchor link for a specific section.
        
        Args:
            section_key: The section to create link for
        """
        st.success(f"Link copied for {section_key} section! (Feature to be implemented)")
        # TODO: Implement actual link copying with JavaScript
        
    def render_scenario_overview_section(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render scenario overview with KPIs and metrics.
        
        This section consolidates content from the original Overview tab,
        including KPI summaries, real-time simulation metrics, and enhanced metrics.
        
        Args:
            scenario_data: Current scenario configuration and data
        """
        #st.markdown("### 🎯 Scenario Selection & Overview")
        #st.markdown("Select and configure simulation scenarios for analysis")
        
        # Scenario Analysis & Comparison (migrated from existing tab)
        st.subheader("📊 Scenario Analysis & Comparison")
        st.markdown("Compare different operational scenarios to optimize port performance")
        
        # Scenario selection interface
        scenario_col1, scenario_col2 = st.columns([1, 2])
        
        with scenario_col1:
            st.subheader("🎯 Scenario Selection")
            
            # Initialize scenario tracking
            self._initialize_scenario_tracking()
            
            # Get available scenarios
            try:
                from hk_port_digital_twin.src.scenarios import list_available_scenarios
                available_scenarios = list_available_scenarios()
            except ImportError:
                available_scenarios = ['normal', 'peak_season', 'maintenance', 'typhoon_season']
            
            # Display current scenario with visual indicator
            current_scenario = self._get_current_scenario()
            scenario_color = self._get_scenario_color(current_scenario)
            scenario_badge = self._get_scenario_badge(current_scenario)
            
            # Show current scenario status with color coding
            st.markdown(
                f"""<div style="padding: 10px; border-radius: 5px; background-color: {scenario_color}; 
                border-left: 4px solid {self._get_scenario_border_color(current_scenario)}; margin-bottom: 15px;">
                <strong>Current Scenario:</strong> {scenario_badge} {current_scenario}
                </div>""",
                unsafe_allow_html=True
            )
            
            # Primary scenario selection
            primary_scenario = st.selectbox(
                "Primary Scenario",
                available_scenarios,
                help="Select the main scenario for analysis",
                key="primary_scenario_select"
            )
            
            # Detect scenario change
            scenario_changed = self._detect_scenario_change(primary_scenario)
            
            # Show scenario change indicator if changed
            if scenario_changed:
                new_color = self._get_scenario_color(self._get_current_scenario())
                new_badge = self._get_scenario_badge(self._get_current_scenario())
                st.markdown(
                    f"""<div style="padding: 10px; border-radius: 5px; background-color: {new_color}; 
                    border-left: 4px solid #28a745; margin: 10px 0;">
                    📊 <strong>Scenario Updated:</strong> {new_badge} {self._get_current_scenario()}
                    </div>""",
                    unsafe_allow_html=True
                )
                st.info("🔄 All values will be regenerated for the new scenario")
            
            # Comparison scenario selection
            comparison_scenarios = st.multiselect(
                "Comparison Scenarios",
                [s for s in available_scenarios if s != primary_scenario],
                help="Select scenarios to compare against the primary scenario"
            )
            
            # Display selected comparison scenarios with visual indicators
            if comparison_scenarios:
                st.markdown("**Selected Comparison Scenarios:**")
                for comp_scenario in comparison_scenarios:
                    comp_scenario_name = self._map_scenario_key_to_name(comp_scenario)
                    comp_color = self._get_scenario_color(comp_scenario_name)
                    comp_badge = self._get_scenario_badge(comp_scenario_name)
                    comp_border = self._get_scenario_border_color(comp_scenario_name)
                    
                    st.markdown(
                        f"""<div style="padding: 8px; border-radius: 4px; background-color: {comp_color}; 
                        border-left: 3px solid {comp_border}; margin: 5px 0; font-size: 0.9em;">
                        {comp_badge} {comp_scenario_name}
                        </div>""",
                        unsafe_allow_html=True
                    )
            
            # Analysis parameters
            st.subheader("⚙️ Analysis Parameters")
            simulation_duration = st.slider(
                "Simulation Duration (hours)",
                min_value=24,
                max_value=168,
                value=72,
                step=24
            )
            
            use_historical_data = st.checkbox(
                "Use Historical Data",
                value=True,
                help="Include historical patterns in the analysis"
            )
            
            if st.button("🔄 Run Scenario Comparison"):
                with st.spinner("Running scenario comparison..."):
                    try:
                        # Import scenario comparison functionality
                        from hk_port_digital_twin.src.scenarios.scenario_comparison import create_scenario_comparison
                        
                        # Run comparison
                        comparison_results = create_scenario_comparison(
                            primary_scenario=primary_scenario,
                            comparison_scenarios=comparison_scenarios,
                            simulation_hours=simulation_duration,
                            use_historical_data=use_historical_data
                        )
                        
                        if comparison_results:
                            st.session_state.scenario_comparison_results = comparison_results
                            st.success("Scenario comparison completed!")
                        else:
                            st.error("Failed to run scenario comparison")
                            
                    except Exception as e:
                        st.error(f"Error running scenario comparison: {str(e)}")
                        import logging
                        logging.error(f"Scenario comparison error: {e}")
        
        with scenario_col2:
            st.subheader("📊 Comparison Results")
            
            if hasattr(st.session_state, 'scenario_comparison_results') and st.session_state.scenario_comparison_results:
                results = st.session_state.scenario_comparison_results
                
                # Display comparison metrics
                if 'comparison_data' in results:
                    comparison_df = pd.DataFrame(results['comparison_data'])
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualization of comparison results
                    import plotly.express as px
                    
                    # Ship arrival rate comparison
                    if 'ship_arrival_rate' in comparison_df.columns:
                        fig_arrivals = px.bar(
                            comparison_df,
                            x='Scenario',
                            y='ship_arrival_rate',
                            title='Ship Arrival Rate by Scenario',
                            color='ship_arrival_rate',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_arrivals, use_container_width=True)
                    
                    # Processing efficiency comparison
                    if 'processing_efficiency' in comparison_df.columns:
                        fig_efficiency = px.bar(
                            comparison_df,
                            x='Scenario',
                            y='processing_efficiency',
                            title='Processing Efficiency by Scenario',
                            color='processing_efficiency',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_efficiency, use_container_width=True)
                    
                    # Container volume multiplier
                    if 'container_volume_multiplier' in comparison_df.columns:
                        fig_volume = px.line(
                            comparison_df,
                            x='Scenario',
                            y='container_volume_multiplier',
                            title='Container Volume Multiplier by Scenario',
                            markers=True
                        )
                        st.plotly_chart(fig_volume, use_container_width=True)
                
                # Export comparison results
                if 'comparison_data' in results:
                    export_csv = comparison_df.to_csv(index=False)
                    st.download_button(
                        label="Export Analysis",
                        data=export_csv,
                        file_name=f"scenario_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Run a scenario comparison to see results here")
                
                # Show available scenarios information
                st.subheader("📋 Available Scenarios")
                
                scenario_info = {
                    'normal': "Standard port operations with typical traffic patterns",
                    'peak_season': "High-volume operations during peak shipping season",
                    'maintenance': "Reduced capacity due to scheduled maintenance",
                    'typhoon_season': "Operations during typhoon season with weather disruptions"
                }
                
                for scenario, description in scenario_info.items():
                    if scenario in available_scenarios:
                        st.write(f"**{scenario.replace('_', ' ').title()}**: {description}")
            
    def render_operational_impact_section(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render operational impact analysis (ships & berths).
        
        Args:
            scenario_data: Current scenario configuration and data
        """
        st.markdown("### 🚢 Operational Impact")
        st.markdown("Real-time analysis of ships and berths including queue management, berth utilization, and vessel tracking.")
        
        # Create tabs for different operational views
        ops_tab1, ops_tab2, ops_tab3 = st.tabs(["🚢 Ship Queue", "🏗️ Berth Utilization", "📊 Live Operations"])
        
        with ops_tab1:
            self._render_ship_queue_analysis(scenario_data)
            
        with ops_tab2:
            self._render_berth_utilization_analysis(scenario_data)
            
        with ops_tab3:
            self._render_live_operations_analysis(scenario_data)
    
    def _render_ship_queue_analysis(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render ship queue analysis and management."""
        st.subheader("🚢 Ship Queue Analysis")
        
        # Get simulation data if available
        simulation_data = getattr(st.session_state, 'simulation_data', None)
        
        if simulation_data and hasattr(simulation_data, 'ship_queue'):
            # Real simulation data
            queue_data = simulation_data.ship_queue
            
            # Queue metrics
            queue_col1, queue_col2, queue_col3, queue_col4 = st.columns(4)
            with queue_col1:
                st.metric("Ships in Queue", len(queue_data))
            with queue_col2:
                avg_wait = sum(ship.get('waiting_time', 0) for ship in queue_data) / max(len(queue_data), 1)
                st.metric("Avg Wait Time", f"{avg_wait:.1f} hrs")
            with queue_col3:
                priority_ships = sum(1 for ship in queue_data if ship.get('priority', 'normal') == 'high')
                st.metric("Priority Ships", priority_ships)
            with queue_col4:
                total_cargo = sum(ship.get('cargo_volume', 0) for ship in queue_data)
                st.metric("Total Cargo", f"{total_cargo:,.0f} TEU")
            
            # Queue visualization
            if queue_data:
                queue_df = pd.DataFrame(queue_data)
                
                # Queue timeline chart
                import plotly.express as px
                fig_queue = px.bar(
                    queue_df,
                    x='ship_id',
                    y='waiting_time',
                    color='ship_type',
                    title='Ship Queue - Waiting Times',
                    labels={'waiting_time': 'Waiting Time (hours)', 'ship_id': 'Ship ID'}
                )
                st.plotly_chart(fig_queue, use_container_width=True)
                
                # Detailed queue table
                st.subheader("📋 Queue Details")
                display_columns = ['ship_id', 'ship_type', 'arrival_time', 'waiting_time', 'cargo_volume', 'priority']
                available_columns = [col for col in display_columns if col in queue_df.columns]
                st.dataframe(queue_df[available_columns], use_container_width=True)
            else:
                st.info("No ships currently in queue")
        else:
            # Sample data for demonstration
            st.info("📊 Using sample data - Start simulation for real-time queue data")
            
            # Generate scenario-aware sample queue data
            import numpy as np
            
            # Get scenario-specific values
            scenario_values = self._get_all_scenario_values()
            queue_size = int(scenario_values['queue_length'])
            
            sample_queue = []
            for i in range(queue_size):
                # Generate scenario-aware waiting times (higher for peak, lower for low season)
                base_wait = scenario_values['handling_time'] / 60  # Convert minutes to hours
                waiting_time = np.random.exponential(base_wait)
                
                # Generate scenario-aware cargo volumes
                cargo_base = scenario_values['monthly_volume'] / 30 / queue_size  # Daily average per ship
                cargo_volume = int(np.random.normal(cargo_base, cargo_base * 0.3))
                cargo_volume = max(500, min(cargo_volume, 5000))  # Reasonable bounds
                
                sample_queue.append({
                    'ship_id': f'SHIP-{i:03d}', 
                    'ship_type': np.random.choice(['Container', 'Bulk', 'Tanker']),
                    'arrival_time': f'{np.random.randint(0, 24):02d}:00', 
                    'waiting_time': waiting_time,
                    'cargo_volume': cargo_volume, 
                    'priority': np.random.choice(['normal', 'high'], p=[0.8, 0.2])
                })
            
            # Sample metrics
            queue_col1, queue_col2, queue_col3, queue_col4 = st.columns(4)
            with queue_col1:
                st.metric("Ships in Queue", len(sample_queue))
            with queue_col2:
                avg_wait = sum(ship['waiting_time'] for ship in sample_queue) / len(sample_queue)
                st.metric("Avg Wait Time", f"{avg_wait:.1f} hrs")
            with queue_col3:
                priority_ships = sum(1 for ship in sample_queue if ship['priority'] == 'high')
                st.metric("Priority Ships", priority_ships)
            with queue_col4:
                total_cargo = sum(ship['cargo_volume'] for ship in sample_queue)
                st.metric("Total Cargo", f"{total_cargo:,.0f} TEU")
            
            # Sample visualization
            queue_df = pd.DataFrame(sample_queue)
            import plotly.express as px
            fig_queue = px.bar(
                queue_df,
                x='ship_id',
                y='waiting_time',
                color='ship_type',
                title='Ship Queue - Waiting Times (Sample Data)',
                labels={'waiting_time': 'Waiting Time (hours)', 'ship_id': 'Ship ID'}
            )
            st.plotly_chart(fig_queue, use_container_width=True)
            
            # Sample queue table
            st.subheader("📋 Queue Details")
            st.dataframe(queue_df, use_container_width=True)
    
    def _render_berth_utilization_analysis(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render berth utilization analysis."""
        st.subheader("🏗️ Berth Utilization Analysis")
        
        # Get simulation data if available
        simulation_data = getattr(st.session_state, 'simulation_data', None)
        
        if simulation_data and hasattr(simulation_data, 'berth_data'):
            # Real simulation data
            berth_data = simulation_data.berth_data
            
            # Berth metrics
            berth_col1, berth_col2, berth_col3, berth_col4 = st.columns(4)
            with berth_col1:
                occupied_berths = sum(1 for berth in berth_data if berth.get('status') == 'occupied')
                st.metric("Occupied Berths", f"{occupied_berths}/{len(berth_data)}")
            with berth_col2:
                avg_utilization = sum(berth.get('utilization', 0) for berth in berth_data) / len(berth_data)
                st.metric("Avg Utilization", f"{avg_utilization:.1f}%")
            with berth_col3:
                maintenance_berths = sum(1 for berth in berth_data if berth.get('status') == 'maintenance')
                st.metric("Under Maintenance", maintenance_berths)
            with berth_col4:
                total_throughput = sum(berth.get('throughput', 0) for berth in berth_data)
                st.metric("Total Throughput", f"{total_throughput:,.0f} TEU")
            
            # Berth utilization chart
            berth_df = pd.DataFrame(berth_data)
            import plotly.express as px
            
            fig_berth = px.bar(
                berth_df,
                x='berth_id',
                y='utilization',
                color='status',
                title='Berth Utilization by Status',
                labels={'utilization': 'Utilization (%)', 'berth_id': 'Berth ID'}
            )
            st.plotly_chart(fig_berth, use_container_width=True)
            
            # Detailed berth table
            st.subheader("📋 Berth Status Details")
            display_columns = ['berth_id', 'status', 'current_ship', 'utilization', 'throughput', 'last_updated']
            available_columns = [col for col in display_columns if col in berth_df.columns]
            st.dataframe(berth_df[available_columns], use_container_width=True)
        else:
            # Sample data for demonstration
            st.info("📊 Using sample data - Start simulation for real-time berth data")
            
            # Generate scenario-aware sample berth data
            import numpy as np
            
            # Get scenario-specific values
            scenario_values = self._get_all_scenario_values()
            berth_utilization = scenario_values['utilization']
            throughput_base = scenario_values['throughput']
            
            berth_statuses = ['occupied', 'available', 'maintenance']
            # Adjust status probabilities based on scenario utilization
            if berth_utilization > 80:  # Peak season - more occupied berths
                status_probs = [0.75, 0.2, 0.05]
            elif berth_utilization < 50:  # Low season - fewer occupied berths
                status_probs = [0.4, 0.55, 0.05]
            else:  # Normal operations
                status_probs = [0.6, 0.35, 0.05]
            
            sample_berths = []
            for i in range(1, 21):  # 20 berths
                status = np.random.choice(berth_statuses, p=status_probs)
                
                # Scenario-aware utilization (varies around the base utilization)
                utilization = np.random.normal(berth_utilization, berth_utilization * 0.15)
                utilization = max(0, min(utilization, 100))  # Clamp to 0-100%
                
                # Scenario-aware throughput
                berth_throughput = int(np.random.normal(throughput_base / 20, throughput_base / 40))
                berth_throughput = max(0, berth_throughput)
                
                sample_berths.append({
                    'berth_id': f'B{i:02d}',
                    'status': status,
                    'current_ship': f'SHIP-{np.random.randint(100, 999)}' if status == 'occupied' else None,
                    'utilization': utilization,
                    'throughput': berth_throughput,
                    'last_updated': datetime.now().strftime('%H:%M:%S')
                })
            
            # Sample metrics
            berth_col1, berth_col2, berth_col3, berth_col4 = st.columns(4)
            with berth_col1:
                occupied_berths = sum(1 for berth in sample_berths if berth['status'] == 'occupied')
                st.metric("Occupied Berths", f"{occupied_berths}/{len(sample_berths)}")
            with berth_col2:
                avg_utilization = sum(berth['utilization'] for berth in sample_berths) / len(sample_berths)
                st.metric("Avg Utilization", f"{avg_utilization:.1f}%")
            with berth_col3:
                maintenance_berths = sum(1 for berth in sample_berths if berth['status'] == 'maintenance')
                st.metric("Under Maintenance", maintenance_berths)
            with berth_col4:
                total_throughput = sum(berth['throughput'] for berth in sample_berths)
                st.metric("Total Throughput", f"{total_throughput:,.0f} TEU")
            
            # Sample visualization
            berth_df = pd.DataFrame(sample_berths)
            import plotly.express as px
            
            fig_berth = px.bar(
                berth_df,
                x='berth_id',
                y='utilization',
                color='status',
                title='Berth Utilization by Status (Sample Data)',
                labels={'utilization': 'Utilization (%)', 'berth_id': 'Berth ID'}
            )
            st.plotly_chart(fig_berth, use_container_width=True)
            
            # Sample berth table
            st.subheader("📋 Berth Status Details")
            st.dataframe(berth_df, use_container_width=True)
    
    def _render_live_operations_analysis(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render live operations analysis."""
        st.subheader("📊 Live Operations Dashboard")
        
        # Real-time operational metrics
        ops_col1, ops_col2 = st.columns(2)
        
        with ops_col1:
            st.subheader("⚡ Real-time Metrics")
            
            # Current operational status
            current_time = datetime.now().strftime('%H:%M:%S')
            st.metric("Current Time", current_time)
            
            # Scenario-aware operational efficiency metrics
            import numpy as np
            scenario_values = self._get_all_scenario_values()
            kpis = scenario_values['kpis']
            
            efficiency_metrics = {
                'Port Efficiency': np.random.uniform(kpis['Port Efficiency'][0], kpis['Port Efficiency'][1]),
                'Crane Productivity': np.random.uniform(kpis['Crane Productivity'][0], kpis['Crane Productivity'][1]),
                'Truck Turnaround': np.random.uniform(kpis['Truck Turnaround'][0], kpis['Truck Turnaround'][1]),
                'Vessel Turnaround': np.random.uniform(kpis['Vessel Turnaround'][0], kpis['Vessel Turnaround'][1])
            }
            
            for metric, value in efficiency_metrics.items():
                st.metric(metric, f"{value:.1f}%")
        
        with ops_col2:
            st.subheader("📈 Performance Trends")
            
            # Generate scenario-aware trend data
            import numpy as np
            scenario_values = self._get_all_scenario_values()
            base_throughput = scenario_values['throughput']
            
            hours = list(range(24))
            # Create realistic hourly variation around base throughput
            throughput_trend = []
            for hour in hours:
                # Add daily pattern (lower at night, higher during day)
                daily_factor = 0.7 + 0.6 * np.sin((hour - 6) * np.pi / 12)
                daily_factor = max(0.4, min(daily_factor, 1.3))
                
                hourly_throughput = base_throughput * daily_factor * np.random.uniform(0.8, 1.2)
                throughput_trend.append(hourly_throughput)
            
            trend_df = pd.DataFrame({
                'Hour': hours,
                'Throughput': throughput_trend
            })
            
            import plotly.express as px
            fig_trend = px.line(
                trend_df,
                x='Hour',
                y='Throughput',
                title='24-Hour Throughput Trend',
                labels={'Hour': 'Hour of Day', 'Throughput': 'Throughput (TEU/hr)'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Operational alerts and notifications
        st.subheader("🚨 Operational Alerts")
        
        # Sample alerts
        alerts = [
            {'level': 'warning', 'message': 'Berth B05 approaching capacity limit', 'time': '14:23'},
            {'level': 'info', 'message': 'New vessel EVER-GIVEN scheduled for arrival', 'time': '14:15'},
            {'level': 'error', 'message': 'Crane C12 requires maintenance attention', 'time': '14:10'}
        ]
        
        for alert in alerts:
            if alert['level'] == 'error':
                st.error(f"🔴 {alert['time']} - {alert['message']}")
            elif alert['level'] == 'warning':
                st.warning(f"🟡 {alert['time']} - {alert['message']}")
            else:
                st.info(f"🔵 {alert['time']} - {alert['message']}")
            
    def render_performance_analytics_section(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render performance analytics.
        
        This section consolidates content from the original Analytics tab,
        providing analytical insights into scenario performance.
        
        Args:
            scenario_data: Current scenario configuration and data
        """
        st.markdown("### 📈 Performance Analytics")
        st.markdown("Deep dive into scenario performance metrics, including throughput timelines, waiting time distributions, and data export options.")
        
        # Create tabs for different analytics views
        analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
            "📈 Throughput Analysis", "⏱️ Waiting Time Analysis", "🎯 Performance Metrics"
        ])
        
        with analytics_tab1:
            self._render_throughput_analysis(scenario_data)
            
        with analytics_tab2:
            self._render_waiting_time_analysis(scenario_data)
            
        with analytics_tab3:
            self._render_performance_metrics(scenario_data)
    

    
    def _render_throughput_analysis(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render throughput analysis charts."""
        st.subheader("📈 Throughput Timeline Analysis")
        
        scenario_name = scenario_data.get('name') if scenario_data else 'Normal Operations'
        params = self._get_scenario_performance_params(scenario_name)
        
        # Generate scenario-aware sample throughput data
        import numpy as np
        hours = list(range(24))
        scenario_values = self._get_all_scenario_values()
        
        # Use scenario-aware throughput as base
        base_throughput = scenario_values['throughput']
        
        # Generate hourly throughput with daily patterns and scenario-aware variations
        throughput_data = []
        for hour in hours:
            # Add daily pattern (peak during business hours)
            daily_factor = 0.7 + 0.6 * np.sin(2 * np.pi * (hour - 6) / 24)
            daily_factor = max(0.5, min(daily_factor, 1.3))
            
            # Apply scenario-aware base with daily pattern and random variation
            hourly_throughput = base_throughput * daily_factor * np.random.normal(1, 0.1)
            hourly_throughput = max(0, hourly_throughput)
            throughput_data.append(hourly_throughput)
        
        # Set target as the scenario's base throughput
        target_throughput = base_throughput
        
        # Create throughput timeline
        throughput_df = pd.DataFrame({
            'Hour': hours,
            'Throughput (TEU/hr)': throughput_data,
            'Target': [target_throughput] * 24  # Target throughput line
        })
        
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Throughput timeline chart
        fig = go.Figure()
        
        # Add actual throughput
        fig.add_trace(go.Scatter(
            x=throughput_df['Hour'],
            y=throughput_df['Throughput (TEU/hr)'],
            mode='lines+markers',
            name='Actual Throughput',
            line=dict(color='blue', width=3)
        ))
        
        # Add target line
        fig.add_trace(go.Scatter(
            x=throughput_df['Hour'],
            y=throughput_df['Target'],
            mode='lines',
            name='Target Throughput',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title='24-Hour Throughput Performance',
            xaxis_title='Hour of Day',
            yaxis_title='Throughput (TEU/hr)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Throughput statistics
        throughput_col1, throughput_col2, throughput_col3, throughput_col4 = st.columns(4)
        with throughput_col1:
            st.metric("Avg Throughput", f"{np.mean(throughput_data):.1f} TEU/hr")
        with throughput_col2:
            st.metric("Peak Throughput", f"{np.max(throughput_data):.1f} TEU/hr")
        with throughput_col3:
            st.metric("Min Throughput", f"{np.min(throughput_data):.1f} TEU/hr")
        with throughput_col4:
            efficiency = (np.mean(throughput_data) / 100) * 100
            st.metric("Efficiency", f"{efficiency:.1f}%")
    
    def _render_waiting_time_analysis(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render waiting time distribution analysis."""
        st.subheader("⏱️ Waiting Time Distribution Analysis")
        
        scenario_name = scenario_data.get('name') if scenario_data else 'Normal Operations'
        params = self._get_scenario_performance_params(scenario_name)
        
        # Generate scenario-aware sample waiting time data
        import numpy as np
        scenario_values = self._get_all_scenario_values()
        
        # Use scenario-aware queue length as base for waiting times
        base_wait = scenario_values['queue_length']
        
        # Generate waiting times with exponential distribution but scenario-aware scale
        waiting_times = np.random.exponential(base_wait * 0.5, 1000)
        
        # Add some variation based on efficiency (higher efficiency = lower waiting times)
        efficiency_factor = scenario_values['efficiency'] / 100
        waiting_times = waiting_times * (1.5 - efficiency_factor)  # Inverse relationship
        
        # Ensure reasonable bounds
        waiting_times = np.clip(waiting_times, 0.1, 48)
        
        # Create waiting time distribution chart
        import plotly.express as px
        
        fig = px.histogram(
            x=waiting_times,
            nbins=30,
            title='Waiting Time Distribution',
            labels={'x': 'Waiting Time (hours)', 'y': 'Frequency'},
            marginal='box'  # Add box plot on top
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title='Waiting Time (hours)',
            yaxis_title='Number of Ships'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Waiting time statistics
        wait_col1, wait_col2, wait_col3, wait_col4 = st.columns(4)
        with wait_col1:
            st.metric("Avg Wait Time", f"{np.mean(waiting_times):.1f} hrs")
        with wait_col2:
            st.metric("Median Wait Time", f"{np.median(waiting_times):.1f} hrs")
        with wait_col3:
            st.metric("95th Percentile", f"{np.percentile(waiting_times, 95):.1f} hrs")
        with wait_col4:
            long_waits = np.sum(waiting_times > 4)  # Ships waiting more than 4 hours
            st.metric("Long Waits (>4hrs)", f"{long_waits} ships")
    
    def _render_performance_metrics(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render performance metrics and KPIs."""
        st.subheader("🎯 Key Performance Indicators")
        
        scenario_name = scenario_data.get('name') if scenario_data else 'Normal Operations'
        params = self._get_scenario_performance_params(scenario_name)
        
        # Performance metrics overview
        import numpy as np
        
        # Generate scenario-aware KPI data
        scenario_values = self._get_all_scenario_values()
        
        # Use scenario-aware values for KPIs
        kpis = {
            'Port Efficiency': scenario_values['efficiency'],
            'Crane Productivity': scenario_values['efficiency'] * 0.95,  # Slightly lower than port efficiency
            'Truck Turnaround': scenario_values['handling_time'] * 0.8,  # Convert to hours
            'Vessel Turnaround': scenario_values['handling_time'] * 1.2   # Slightly higher than truck
        }
        
        
        # Display KPIs in a grid
        kpi_cols = st.columns(3)
        for i, (kpi, value) in enumerate(kpis.items()):
            with kpi_cols[i % 3]:
                # Determine color based on performance
                if value >= 90:
                    delta_color = "normal"
                    delta = "Excellent"
                elif value >= 80:
                    delta_color = "normal"
                    delta = "Good"
                else:
                    delta_color = "inverse"
                    delta = "Needs Improvement"
                
                if 'Turnaround' in kpi:
                    st.metric(kpi, f"{value:.1f} hrs", delta=delta, delta_color=delta_color)
                else:
                    st.metric(kpi, f"{value:.1f}%", delta=delta, delta_color=delta_color)
        
        # Performance trend radar chart
        st.subheader("📊 Performance Radar Chart")
        
        import plotly.graph_objects as go
        
        # Normalize all KPIs to 0-100 scale for proper radar chart visualization
        normalized_kpis = {}
        target_values = {}
        
        for kpi, value in kpis.items():
            if 'Turnaround' in kpi:
                # For turnaround times (lower is better), normalize and invert
                # Assume max acceptable time is 8 hours, target is 3 hours
                max_time = 8.0
                target_time = 3.0
                normalized_value = max(0, (max_time - value) / max_time * 100)
                normalized_target = (max_time - target_time) / max_time * 100
                normalized_kpis[kpi] = normalized_value
                target_values[kpi] = normalized_target
            else:
                # For efficiency metrics (higher is better), use as-is
                normalized_kpis[kpi] = value
                target_values[kpi] = 90  # 90% target for efficiency metrics
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(normalized_kpis.values()),
            theta=list(normalized_kpis.keys()),
            fill='toself',
            name='Current Performance',
            line_color='purple',
            opacity=0.7
        ))
        
        # Add target performance line
        fig.add_trace(go.Scatterpolar(
            r=list(target_values.values()),
            theta=list(target_values.keys()),
            fill='toself',
            name='Target Performance',
            line_color='pink',
            opacity=0.4
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%'
                )
            ),
            showlegend=True,
            title="Performance vs Target Comparison (Normalized to 0-100%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation for radar chart
        st.info("📝 **Chart Note**: All metrics are normalized to 0-100% scale. For turnaround times, higher values indicate better performance (shorter times).")
            # - Performance metrics
            
    def _map_scenario_key_to_name(self, scenario_key: str) -> str:
        """
        Map scenario keys to proper scenario names for performance parameters.
        
        Args:
            scenario_key: Scenario key from session state ('peak', 'normal', 'low')
            
        Returns:
            Proper scenario name for performance parameters
        """
        scenario_mapping = {
            'peak': 'Peak Season',
            'normal': 'Normal Operations', 
            'low': 'Low Season',
            'peak_season': 'Peak Season',
            'normal_operations': 'Normal Operations',
            'low_season': 'Low Season'
        }
        return scenario_mapping.get(scenario_key, 'Normal Operations')
    
    def _get_scenario_performance_params(self, scenario_name: Optional[str]) -> Dict[str, Any]:
        """Get performance parameters based on the scenario name.
        
        Enhanced with wider, non-overlapping ranges to ensure clear differentiation
        between scenarios. All ranges are designed to maintain Peak > Normal > Low ordering.
        """
        scenario_name = scenario_name or "Normal Operations"  # Default to normal
        
        # Map scenario key to proper name if needed
        if scenario_name in ['peak', 'normal', 'low', 'peak_season', 'normal_operations', 'low_season']:
            scenario_name = self._map_scenario_key_to_name(scenario_name)
        
        if "Peak Season" in scenario_name:
            return {
                # Throughput parameters - highest range (130+ to ensure > Normal)
                "throughput_range": (130, 180),
                "throughput_target": 155,
                
                # Utilization parameters - very high (88+ to ensure > Normal)
                "utilization_range": (88, 98),
                
                # Revenue parameters - highest range (150M+ to ensure > Normal)
                "revenue_range": (150_000_000, 220_000_000),
                
                # Handling time parameters - shortest due to peak efficiency (1.0-2.0 to ensure < Normal)
                "handling_time_range": (1.0, 2.0),
                
                # Queue parameters - more ships, efficient processing
                "queue_length_range": (18, 30),
                "waiting_time_exponential_scale": 1.8,
                "long_wait_threshold": 4,
                
                # Efficiency parameters - highest performance (92+ to ensure > Normal)
                "efficiency_range": (92, 99),
                
                # Cargo volume parameters - highest range (450K+ to ensure > Normal)
                "cargo_volume_range": (450_000, 650_000),
                
                # Trade balance parameters - strong positive (100K+ to ensure > Normal)
                "trade_balance_range": (100_000, 180_000),
                
                # Monthly volume parameters - peak season patterns
                "monthly_volume_base": 50_000,
                "monthly_volume_variance": 18_000,
                
                # KPI parameters - optimized for peak performance
                "kpis": {
                    'Port Efficiency': (92, 99),
                    'Berth Utilization': (88, 98),
                    'Average Turnaround': (0.8, 1.5),  # Hours (fastest)
                    'Customer Satisfaction': (90, 98),
                    'Cost Efficiency': (85, 95),
                    'Environmental Score': (75, 88),
                    'Crane Productivity': (40, 50),  # Moves per hour (highest)
                    'Truck Turnaround': (15, 25),  # Minutes (fastest)
                    'Vessel Turnaround': (6, 10)  # Hours (fastest)
                }
            }
        elif "Low Season" in scenario_name:
            return {
                # Throughput parameters - lowest range (40-80 to ensure < Normal)
                "throughput_range": (40, 80),
                "throughput_target": 60,
                
                # Utilization parameters - low (45-70 to ensure < Normal)
                "utilization_range": (45, 70),
                
                # Revenue parameters - lowest range (60-105M to ensure < Normal)
                "revenue_range": (60_000_000, 105_000_000),
                
                # Handling time parameters - longest due to reduced efficiency (3.5-5.5 to ensure > Normal)
                "handling_time_range": (3.5, 5.5),
                
                # Queue parameters - fewer ships, longer waits
                "queue_length_range": (3, 8),
                "waiting_time_exponential_scale": 4.2,
                "long_wait_threshold": 8,
                
                # Efficiency parameters - lowest performance (60-78 to ensure < Normal)
                "efficiency_range": (60, 78),
                
                # Cargo volume parameters - lowest range (120-270K to ensure < Normal)
                "cargo_volume_range": (120_000, 270_000),
                
                # Trade balance parameters - negative (to ensure < Normal)
                "trade_balance_range": (-80_000, 15_000),
                
                # Monthly volume parameters - low season patterns
                "monthly_volume_base": 18_000,
                "monthly_volume_variance": 8_000,
                
                # KPI parameters - reduced performance
                "kpis": {
                    'Port Efficiency': (60, 78),
                    'Berth Utilization': (45, 70),
                    'Average Turnaround': (3.0, 5.0),  # Hours (slowest)
                    'Customer Satisfaction': (65, 80),
                    'Cost Efficiency': (55, 72),
                    'Environmental Score': (80, 95),  # Better due to lower activity
                    'Crane Productivity': (15, 25),  # Moves per hour (lowest)
                    'Truck Turnaround': (45, 65),  # Minutes (slowest)
                    'Vessel Turnaround': (20, 35)  # Hours (slowest)
                }
            }
        else:  # Normal Operations
            return {
                # Throughput parameters - medium range (85-125 to ensure < Peak and > Low)
                "throughput_range": (85, 125),
                "throughput_target": 105,
                
                # Utilization parameters - medium (72-85 to ensure < Peak and > Low)
                "utilization_range": (72, 85),
                
                # Revenue parameters - medium range (110-145M to ensure < Peak and > Low)
                "revenue_range": (110_000_000, 145_000_000),
                
                # Handling time parameters - moderate (2.2-3.2 to ensure > Peak and < Low)
                "handling_time_range": (2.2, 3.2),
                
                # Queue parameters - moderate levels
                "queue_length_range": (10, 16),
                "waiting_time_exponential_scale": 2.8,
                "long_wait_threshold": 5,
                
                # Efficiency parameters - medium performance (80-90 to ensure < Peak and > Low)
                "efficiency_range": (80, 90),
                
                # Cargo volume parameters - medium range (280-420K to ensure < Peak and > Low)
                "cargo_volume_range": (280_000, 420_000),
                
                # Trade balance parameters - balanced (20-90K to ensure < Peak and > Low)
                "trade_balance_range": (20_000, 90_000),
                
                # Monthly volume parameters - normal patterns
                "monthly_volume_base": 32_000,
                "monthly_volume_variance": 12_000,
                
                # KPI parameters - balanced performance
                "kpis": {
                    'Port Efficiency': (80, 90),
                    'Berth Utilization': (72, 85),
                    'Average Turnaround': (1.8, 2.8),  # Hours (between Peak and Low)
                    'Customer Satisfaction': (82, 88),
                    'Cost Efficiency': (75, 82),
                    'Environmental Score': (70, 85),
                    'Crane Productivity': (28, 38),  # Moves per hour (between Peak and Low)
                    'Truck Turnaround': (28, 38),  # Minutes (between Peak and Low)
                    'Vessel Turnaround': (12, 18)  # Hours (between Peak and Low)
                }
            }

    def _generate_scenario_values(self, scenario_name: str, value_type: str, count: int = 1, **kwargs) -> Any:
        """
        Centralized utility for generating scenario-aware values with consistent ranges.
        
        Args:
            scenario_name: Name of the scenario ('Peak Season', 'Normal Operations', 'Low Season')
            value_type: Type of value to generate ('throughput', 'utilization', 'revenue', etc.)
            count: Number of values to generate (default: 1)
            **kwargs: Additional parameters for specific value types
            
        Returns:
            Generated value(s) - single value if count=1, list if count>1
        """
        import numpy as np
        
        # Get scenario parameters
        params = self._get_scenario_performance_params(scenario_name)
        
        # Define value generation logic based on type
        if value_type == 'throughput':
            min_val, max_val = params['throughput_range']
            values = np.random.uniform(min_val, max_val, count)
            
        elif value_type == 'utilization':
            min_val, max_val = params['utilization_range']
            values = np.random.uniform(min_val, max_val, count)
            
        elif value_type == 'revenue':
            min_val, max_val = params['revenue_range']
            values = np.random.uniform(min_val, max_val, count)
            
        elif value_type == 'handling_time':
            min_val, max_val = params['handling_time_range']
            values = np.random.uniform(min_val, max_val, count)
            
        elif value_type == 'queue_length':
            min_val, max_val = params['queue_length_range']
            values = np.random.randint(min_val, max_val + 1, count)
            
        elif value_type == 'waiting_time':
            scale = params['waiting_time_exponential_scale']
            values = np.random.exponential(scale, count)
            
        elif value_type == 'efficiency':
            min_val, max_val = params['efficiency_range']
            values = np.random.uniform(min_val, max_val, count)
            
        elif value_type == 'cargo_volume':
            min_val, max_val = params['cargo_volume_range']
            values = np.random.uniform(min_val, max_val, count)
            
        elif value_type == 'trade_balance':
            min_val, max_val = params['trade_balance_range']
            values = np.random.uniform(min_val, max_val, count)
            
        elif value_type == 'monthly_volume':
            base = params['monthly_volume_base']
            variance = params['monthly_volume_variance']
            values = np.random.normal(base, variance / 3, count)  # Using variance/3 as std dev
            values = np.maximum(values, 0)  # Ensure non-negative
            
        elif value_type == 'kpi':
            kpi_name = kwargs.get('kpi_name', 'Port Efficiency')
            if kpi_name in params['kpis']:
                min_val, max_val = params['kpis'][kpi_name]
                values = np.random.uniform(min_val, max_val, count)
            else:
                # Fallback for unknown KPIs
                values = np.random.uniform(50, 90, count)
                
        elif value_type == 'berth_throughput':
            # Special case for berth throughput (integer values)
            min_val, max_val = params['throughput_range']
            values = np.random.randint(int(min_val * 0.8), int(max_val * 1.2), count)
            
        elif value_type == 'trend_data':
            # Generate trend data with scenario-appropriate baseline
            baseline = kwargs.get('baseline', params['throughput_target'])
            noise_factor = kwargs.get('noise_factor', 0.1)
            trend_length = kwargs.get('trend_length', 24)
            
            # Create base trend
            x = np.linspace(0, trend_length - 1, trend_length)
            trend = baseline + np.sin(x * 0.5) * baseline * 0.1  # Slight sinusoidal pattern
            
            # Add scenario-appropriate noise
            noise = np.random.normal(0, baseline * noise_factor, trend_length)
            values = trend + noise
            values = np.maximum(values, 0)  # Ensure non-negative
            
        elif value_type == 'investment_roi':
            # Generate investment ROI values based on provided range
            roi_range = kwargs.get('roi_range', (0.10, 0.20))  # Default 10-20% if not specified
            min_val, max_val = roi_range
            values = np.random.uniform(min_val, max_val, count)
            
        else:
            # Fallback for unknown value types
            values = np.random.uniform(0, 100, count)
            
        # Return single value or list based on count
        if count == 1:
            return float(values[0]) if len(values) > 0 else 0.0
        else:
            return values.tolist()

    def _get_all_scenario_values(self, scenario_name: str = None) -> Dict[str, Any]:
        """
        Get all scenario values as a dictionary for backward compatibility.
        
        Args:
            scenario_name: Name of the scenario (defaults to current scenario)
            
        Returns:
            Dictionary containing all scenario values
        """
        if scenario_name is None:
            scenario_name = self._get_current_scenario()
            
        # Get scenario parameters
        params = self._get_scenario_performance_params(scenario_name)
        
        return {
            'throughput': self._generate_scenario_values(scenario_name, 'throughput'),
            'utilization': self._generate_scenario_values(scenario_name, 'utilization'),
            'revenue': self._generate_scenario_values(scenario_name, 'revenue'),
            'handling_time': self._generate_scenario_values(scenario_name, 'handling_time'),
            'queue_length': self._generate_scenario_values(scenario_name, 'queue_length'),
            'waiting_time': self._generate_scenario_values(scenario_name, 'waiting_time'),
            'efficiency': self._generate_scenario_values(scenario_name, 'efficiency'),
            'cargo_volume': self._generate_scenario_values(scenario_name, 'cargo_volume'),
            'trade_balance': self._generate_scenario_values(scenario_name, 'trade_balance'),
            'monthly_volume': self._generate_scenario_values(scenario_name, 'monthly_volume'),
            'kpis': params['kpis']  # Return the full KPI ranges
        }

    def _validate_scenario_ranges(self) -> Dict[str, List[str]]:
        """
        Validate that scenario parameter ranges maintain Peak > Normal > Low ordering.
        
        Returns:
            Dictionary with validation results and any errors found
        """
        validation_results = {
            'errors': [],
            'warnings': [],
            'status': 'valid'
        }
        
        # Get parameters for all scenarios
        peak_params = self._get_scenario_performance_params('Peak Season')
        normal_params = self._get_scenario_performance_params('Normal Operations')
        low_params = self._get_scenario_performance_params('Low Season')
        
        # Define parameters that should follow Peak > Normal > Low ordering
        ascending_params = [
            'throughput_range', 'utilization_range', 'revenue_range',
            'efficiency_range', 'cargo_volume_range', 'trade_balance_range'
        ]
        
        # Define parameters that should follow Peak < Normal < Low ordering (inverse)
        descending_params = [
            'handling_time_range', 'waiting_time_exponential_scale'
        ]
        
        # Validate ascending parameters (Peak > Normal > Low)
        for param in ascending_params:
            if param in peak_params and param in normal_params and param in low_params:
                peak_min, peak_max = peak_params[param]
                normal_min, normal_max = normal_params[param]
                low_min, low_max = low_params[param]
                
                # Check minimum values ordering
                if not (peak_min > normal_min > low_min):
                    validation_results['errors'].append(
                        f"{param} minimum values don't follow Peak > Normal > Low ordering: "
                        f"Peak({peak_min}) > Normal({normal_min}) > Low({low_min})"
                    )
                
                # Check maximum values ordering
                if not (peak_max > normal_max > low_max):
                    validation_results['errors'].append(
                        f"{param} maximum values don't follow Peak > Normal > Low ordering: "
                        f"Peak({peak_max}) > Normal({normal_max}) > Low({low_max})"
                    )
                
                # Check for range overlaps
                if normal_max >= peak_min:
                    validation_results['warnings'].append(
                        f"{param} Normal and Peak ranges overlap: Normal({normal_min}-{normal_max}) vs Peak({peak_min}-{peak_max})"
                    )
                
                if low_max >= normal_min:
                    validation_results['warnings'].append(
                        f"{param} Low and Normal ranges overlap: Low({low_min}-{low_max}) vs Normal({normal_min}-{normal_max})"
                    )
        
        # Validate descending parameters (Peak < Normal < Low)
        for param in descending_params:
            if param in peak_params and param in normal_params and param in low_params:
                if isinstance(peak_params[param], tuple):
                    peak_min, peak_max = peak_params[param]
                    normal_min, normal_max = normal_params[param]
                    low_min, low_max = low_params[param]
                    
                    # Check minimum values ordering (inverse)
                    if not (peak_min < normal_min < low_min):
                        validation_results['errors'].append(
                            f"{param} minimum values don't follow Peak < Normal < Low ordering: "
                            f"Peak({peak_min}) < Normal({normal_min}) < Low({low_min})"
                        )
                else:
                    # Single value parameters
                    peak_val = peak_params[param]
                    normal_val = normal_params[param]
                    low_val = low_params[param]
                    
                    if not (peak_val < normal_val < low_val):
                        validation_results['errors'].append(
                            f"{param} values don't follow Peak < Normal < Low ordering: "
                            f"Peak({peak_val}) < Normal({normal_val}) < Low({low_val})"
                        )
        
        # Set overall status
        if validation_results['errors']:
            validation_results['status'] = 'invalid'
        elif validation_results['warnings']:
            validation_results['status'] = 'warning'
        
        return validation_results
    
    def _display_validation_results(self, validation_results: Dict[str, List[str]]) -> None:
        """
        Display validation results in the Streamlit interface.
        
        Args:
            validation_results: Results from _validate_scenario_ranges()
        """
        if validation_results['status'] == 'valid':
            st.success("✅ All scenario parameter ranges are properly ordered")
        elif validation_results['status'] == 'warning':
            st.warning("⚠️ Scenario validation completed with warnings")
            for warning in validation_results['warnings']:
                st.warning(f"Warning: {warning}")
        else:
            st.error("❌ Scenario parameter validation failed")
            for error in validation_results['errors']:
                st.error(f"Error: {error}")
            
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                     st.warning(f"Warning: {warning}")

    def _render_validation_section(self) -> None:
        """
        Render the validation section for scenario parameter consistency.
        """
        # Create a collapsible validation section
        with st.expander("🔍 Scenario Parameter Validation", expanded=False):
            st.markdown("*Check scenario parameter consistency and range ordering*")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("🔍 Run Validation", key="run_validation"):
                    # Run validation and store results in session state
                    validation_results = self._validate_scenario_ranges()
                    st.session_state.validation_results = validation_results
                    st.rerun()
            
            with col2:
                if st.button("📊 Show Parameter Ranges", key="show_ranges"):
                    st.session_state.show_parameter_ranges = not st.session_state.get('show_parameter_ranges', False)
                    st.rerun()
            
            # Display validation results if available
            if 'validation_results' in st.session_state:
                st.markdown("---")
                st.subheader("Validation Results")
                self._display_validation_results(st.session_state.validation_results)
            
            # Display parameter ranges if requested
            if st.session_state.get('show_parameter_ranges', False):
                st.markdown("---")
                st.subheader("Parameter Ranges by Scenario")
                self._display_parameter_ranges()
    
    def _display_parameter_ranges(self) -> None:
        """
        Display parameter ranges for all scenarios in a formatted table.
        """
        import pandas as pd
        
        # Get parameters for all scenarios
        scenarios = ['Peak Season', 'Normal Operations', 'Low Season']
        all_params = {}
        
        for scenario in scenarios:
            params = self._get_scenario_performance_params(scenario)
            all_params[scenario] = params
        
        # Create comparison table for key parameters
        comparison_data = []
        key_params = [
            'throughput_range', 'utilization_range', 'revenue_range',
            'efficiency_range', 'cargo_volume_range', 'handling_time_range',
            'trade_balance_range'
        ]
        
        for param in key_params:
            row = {'Parameter': param.replace('_', ' ').title()}
            for scenario in scenarios:
                if param in all_params[scenario]:
                    param_value = all_params[scenario][param]
                    if isinstance(param_value, tuple) and len(param_value) == 2:
                        row[scenario] = f"{param_value[0]:.1f} - {param_value[1]:.1f}"
                    else:
                        row[scenario] = str(param_value)
                else:
                    row[scenario] = "N/A"
            comparison_data.append(row)
        
        # Display as DataFrame
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Add color coding explanation
        st.markdown("""
        **Expected Ordering:**
        - 📈 **Ascending**: Peak > Normal > Low (throughput, utilization, revenue, efficiency, cargo volume, trade balance)
        - 📉 **Descending**: Peak < Normal < Low (handling time)
        """)

    def _initialize_scenario_tracking(self) -> None:
        """
        Initialize session state for scenario tracking and change detection.
        """
        # Initialize scenario tracking if not exists
        if 'current_scenario' not in st.session_state:
            st.session_state.current_scenario = 'Normal Operations'
            
        if 'scenario_changed' not in st.session_state:
            st.session_state.scenario_changed = False
            
        if 'scenario_change_timestamp' not in st.session_state:
            st.session_state.scenario_change_timestamp = datetime.now()
            
        # Initialize cached values for scenario-dependent data
        if 'scenario_cached_values' not in st.session_state:
            st.session_state.scenario_cached_values = {}

    def _detect_scenario_change(self, new_scenario: str) -> bool:
        """
        Detect if the scenario has changed and update session state accordingly.
        
        Args:
            new_scenario: The newly selected scenario name
            
        Returns:
            True if scenario changed, False otherwise
        """
        # Map scenario keys to proper names
        scenario_name = self._map_scenario_key_to_name(new_scenario)
        
        # Check if scenario has changed
        if st.session_state.current_scenario != scenario_name:
            # Update session state
            st.session_state.current_scenario = scenario_name
            st.session_state.scenario_changed = True
            st.session_state.scenario_change_timestamp = datetime.now()
            
            # Clear cached values for the new scenario
            st.session_state.scenario_cached_values = {}
            
            return True
        
        # Reset change flag if no change detected
        st.session_state.scenario_changed = False
        return False

    def _get_current_scenario(self) -> str:
        """
        Get the current scenario from session state.
        
        Returns:
            Current scenario name
        """
        return st.session_state.get('current_scenario', 'Normal Operations')

    def _cache_scenario_value(self, key: str, value: Any) -> None:
        """
        Cache a value for the current scenario to maintain consistency.
        
        Args:
            key: Cache key for the value
            value: Value to cache
        """
        scenario = self._get_current_scenario()
        if scenario not in st.session_state.scenario_cached_values:
            st.session_state.scenario_cached_values[scenario] = {}
        st.session_state.scenario_cached_values[scenario][key] = value

    def _get_cached_scenario_value(self, key: str, default: Any = None) -> Any:
        """
        Get a cached value for the current scenario.
        
        Args:
            key: Cache key for the value
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        scenario = self._get_current_scenario()
        return st.session_state.scenario_cached_values.get(scenario, {}).get(key, default)
    
    def _get_scenario_color(self, scenario: str) -> str:
        """
        Get the background color for a scenario based on its type.
        
        Args:
            scenario: Scenario name
            
        Returns:
            CSS color code for the scenario
        """
        scenario_lower = scenario.lower()
        
        if 'peak' in scenario_lower:
            return '#fff3cd'  # Light yellow for peak/high activity
        elif 'low' in scenario_lower:
            return '#d1ecf1'  # Light blue for low activity
        elif 'maintenance' in scenario_lower:
            return '#f8d7da'  # Light red for maintenance/disruption
        elif 'typhoon' in scenario_lower or 'storm' in scenario_lower:
            return '#e2e3e5'  # Light gray for weather disruptions
        else:
            return '#d4edda'  # Light green for normal operations
    
    def _get_scenario_border_color(self, scenario: str) -> str:
        """
        Get the border color for a scenario based on its type.
        
        Args:
            scenario: Scenario name
            
        Returns:
            CSS color code for the scenario border
        """
        scenario_lower = scenario.lower()
        
        if 'peak' in scenario_lower:
            return '#ffc107'  # Amber for peak/high activity
        elif 'low' in scenario_lower:
            return '#17a2b8'  # Blue for low activity
        elif 'maintenance' in scenario_lower:
            return '#dc3545'  # Red for maintenance/disruption
        elif 'typhoon' in scenario_lower or 'storm' in scenario_lower:
            return '#6c757d'  # Gray for weather disruptions
        else:
            return '#28a745'  # Green for normal operations
    
    def _get_scenario_badge(self, scenario: str) -> str:
        """
        Get an emoji badge for a scenario based on its type.
        
        Args:
            scenario: Scenario name
            
        Returns:
            Emoji badge representing the scenario
        """
        scenario_lower = scenario.lower()
        
        if 'peak' in scenario_lower:
            return '🔥'  # Fire for peak activity
        elif 'low' in scenario_lower:
            return '📉'  # Downward trend for low activity
        elif 'maintenance' in scenario_lower:
            return '🔧'  # Wrench for maintenance
        elif 'typhoon' in scenario_lower or 'storm' in scenario_lower:
            return '🌪️'  # Tornado for weather disruptions
        else:
            return '✅'  # Check mark for normal operations

    def _render_analytics_section(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render cargo analysis.
        
        This section consolidates content from the original Cargo Statistics tab,
        showing scenario impact on cargo handling and logistics.
        
        Args:
            scenario_data: Current scenario configuration and data
        """
        st.markdown("### 📦 Cargo Analysis")
        st.markdown("Comprehensive analysis of cargo throughput data, including shipment types, transport modes, time series analysis, and forecasting.")
        
        # Data summary section
        self._render_cargo_data_summary()
        
        # Create tabs for different cargo analysis views
        cargo_tab1, cargo_tab2, cargo_tab3, cargo_tab4, cargo_tab5, cargo_tab6 = st.tabs([
            "📊 Shipment Types", "🚛 Transport Modes", "📈 Time Series", 
            "🔮 Forecasting", "📦 Cargo Types", "🌍 Locations"
        ])
        
        with cargo_tab1:
            self._render_shipment_types_analysis()
            
        with cargo_tab2:
            self._render_transport_modes_analysis()
            
        with cargo_tab3:
            self._render_time_series_analysis()
            
        with cargo_tab4:
            self._render_forecasting_analysis()
            
        with cargo_tab5:
            self._render_cargo_types_analysis()
            
        with cargo_tab6:
            self._render_locations_analysis()
    
    def _render_cargo_data_summary(self) -> None:
        """Render cargo data summary section."""
        st.subheader("📊 Data Summary")
        
        # Generate sample cargo data summary
        import numpy as np
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            total_shipments = np.random.randint(15000, 25000)
            st.metric("Total Shipments", f"{total_shipments:,}")
            
        with summary_col2:
            total_teu = np.random.randint(800000, 1200000)
            st.metric("Total TEU", f"{total_teu:,}")
            
        with summary_col3:
            avg_shipment_size = total_teu / total_shipments
            st.metric("Avg Shipment Size", f"{avg_shipment_size:.1f} TEU")
            
        with summary_col4:
            utilization_rate = np.random.uniform(75, 95)
            st.metric("Utilization Rate", f"{utilization_rate:.1f}%")
    
    def _render_shipment_types_analysis(self) -> None:
        """Render shipment types analysis."""
        st.subheader("📊 Shipment Type Analysis")
        
        # Generate sample shipment data
        import numpy as np
        import plotly.express as px
        
        shipment_types = ['Import', 'Export', 'Transshipment', 'Domestic']
        shipment_data = {
            'Type': shipment_types,
            'Count': [np.random.randint(3000, 8000) for _ in shipment_types],
            'TEU': [np.random.randint(150000, 400000) for _ in shipment_types],
            'Avg_Size': [np.random.uniform(15, 60) for _ in shipment_types]
        }
        
        shipment_df = pd.DataFrame(shipment_data)
        
        # Display shipment statistics
        st.dataframe(shipment_df, use_container_width=True)
        
        # Shipment type distribution charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Pie chart for shipment count
            fig_count = px.pie(
                shipment_df, 
                values='Count', 
                names='Type',
                title='Shipment Count Distribution'
            )
            st.plotly_chart(fig_count, use_container_width=True)
        
        with chart_col2:
            # Bar chart for TEU volume
            fig_teu = px.bar(
                shipment_df,
                x='Type',
                y='TEU',
                title='TEU Volume by Shipment Type',
                color='TEU',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_teu, use_container_width=True)
    
    def _render_transport_modes_analysis(self) -> None:
        """Render transport modes analysis."""
        st.subheader("🚛 Transport Mode Analysis")
        
        # Generate scenario-aware transport mode data
        import numpy as np
        import plotly.express as px
        
        scenario_values = self._get_all_scenario_values()
        base_volume = scenario_values['cargo_volume']
        efficiency_factor = scenario_values['efficiency'] / 100
        
        transport_modes = ['Truck', 'Rail', 'Barge', 'Pipeline']
        
        # Different transport modes have different characteristics
        mode_factors = {
            'Truck': {'volume': 0.6, 'efficiency': 0.8, 'cost': 1.5},
            'Rail': {'volume': 0.25, 'efficiency': 1.1, 'cost': 0.8},
            'Barge': {'volume': 0.1, 'efficiency': 0.9, 'cost': 0.6},
            'Pipeline': {'volume': 0.05, 'efficiency': 1.2, 'cost': 0.4}
        }
        
        mode_data = {
            'Mode': transport_modes,
            'Volume_TEU': [int(base_volume * mode_factors[mode]['volume'] * np.random.normal(1, 0.1)) for mode in transport_modes],
            'Efficiency': [min(95, max(60, efficiency_factor * 100 * mode_factors[mode]['efficiency'] * np.random.normal(1, 0.05))) for mode in transport_modes],
            'Cost_per_TEU': [100 * mode_factors[mode]['cost'] * (2 - efficiency_factor) * np.random.normal(1, 0.1) for mode in transport_modes]
        }
        
        mode_df = pd.DataFrame(mode_data)
        
        # Display transport mode statistics
        st.dataframe(mode_df, use_container_width=True)
        
        # Transport mode analysis charts
        mode_col1, mode_col2 = st.columns(2)
        
        with mode_col1:
            # Volume by transport mode
            fig_volume = px.bar(
                mode_df,
                x='Mode',
                y='Volume_TEU',
                title='Volume by Transport Mode',
                color='Volume_TEU',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with mode_col2:
            # Efficiency vs Cost scatter plot
            fig_scatter = px.scatter(
                mode_df,
                x='Cost_per_TEU',
                y='Efficiency',
                size='Volume_TEU',
                color='Mode',
                title='Efficiency vs Cost by Transport Mode',
                hover_data=['Volume_TEU']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def _render_time_series_analysis(self) -> None:
        """Render time series analysis."""
        st.subheader("📈 Time Series Analysis")
        
        # Generate scenario-aware time series data
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        
        scenario_values = self._get_all_scenario_values()
        base_volume = scenario_values['cargo_volume']
        base_handling = scenario_values['handling_time']
        efficiency_factor = scenario_values['efficiency'] / 100
        
        # Create 12 months of data with scenario-aware patterns
        dates = [datetime.now() - timedelta(days=30*i) for i in range(12, 0, -1)]
        monthly_data = {
            'Date': dates,
            'TEU_Volume': [],
            'Shipments': [],
            'Avg_Dwell_Time': []
        }
        
        for i in range(12):
            # Seasonal patterns and scenario influence
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 12)  # Seasonal variation
            trend_factor = 0.9 + (i / 12) * 0.2  # Gradual trend
            
            # TEU Volume
            volume = base_volume * seasonal_factor * trend_factor * np.random.normal(1, 0.05)
            monthly_data['TEU_Volume'].append(max(0, volume))
            
            # Shipments (correlated with volume but different pattern)
            shipments = (volume / 20) * np.random.normal(1, 0.1)  # ~20 TEU per shipment average
            monthly_data['Shipments'].append(max(0, int(shipments)))
            
            # Dwell time (inversely related to efficiency)
            dwell_time = base_handling * (2 - efficiency_factor) * np.random.normal(1, 0.1)
            monthly_data['Avg_Dwell_Time'].append(max(1, dwell_time))
        
        time_df = pd.DataFrame(monthly_data)
        
        # Time series charts
        ts_col1, ts_col2 = st.columns(2)
        
        with ts_col1:
            # TEU volume over time
            fig_teu = px.line(
                time_df,
                x='Date',
                y='TEU_Volume',
                title='TEU Volume Over Time',
                markers=True
            )
            fig_teu.update_layout(xaxis_title='Date', yaxis_title='TEU Volume')
            st.plotly_chart(fig_teu, use_container_width=True)
        
        with ts_col2:
            # Shipments over time
            fig_shipments = px.line(
                time_df,
                x='Date',
                y='Shipments',
                title='Number of Shipments Over Time',
                markers=True,
                line_shape='spline'
            )
            fig_shipments.update_layout(xaxis_title='Date', yaxis_title='Number of Shipments')
            st.plotly_chart(fig_shipments, use_container_width=True)
        
        # Dwell time analysis
        fig_dwell = px.bar(
            time_df,
            x='Date',
            y='Avg_Dwell_Time',
            title='Average Dwell Time by Month',
            color='Avg_Dwell_Time',
            color_continuous_scale='Reds'
        )
        fig_dwell.update_layout(xaxis_title='Date', yaxis_title='Average Dwell Time (days)')
        st.plotly_chart(fig_dwell, use_container_width=True)
    
    def _render_forecasting_analysis(self) -> None:
        """Render forecasting analysis."""
        st.subheader("🔮 Cargo Volume Forecasting")
        
        # Generate scenario-aware forecasting data
        import numpy as np
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        
        scenario_values = self._get_all_scenario_values()
        base_volume = scenario_values['cargo_volume']
        efficiency_factor = scenario_values['efficiency'] / 100
        
        # Historical data (12 months) - scenario-aware
        hist_dates = [datetime.now() - timedelta(days=30*i) for i in range(12, 0, -1)]
        hist_volumes = []
        for i in range(12):
            # Generate historical trend leading to current scenario
            historical_factor = 0.8 + (i / 12) * 0.4  # Gradual increase to current level
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)  # Seasonal pattern
            volume = base_volume * historical_factor * seasonal_factor * np.random.normal(1, 0.05)
            hist_volumes.append(max(0, volume))
        
        # Forecast data (6 months ahead) - scenario-aware
        forecast_dates = [datetime.now() + timedelta(days=30*i) for i in range(1, 7)]
        forecast_volumes = []
        
        # Growth rate based on scenario efficiency
        monthly_growth = 0.01 + (efficiency_factor - 0.75) * 0.02  # Higher efficiency = higher growth
        
        for i, _ in enumerate(forecast_dates):
            trend = base_volume * (1 + monthly_growth * i)
            seasonal = np.sin(i * np.pi / 6) * base_volume * 0.05  # Seasonal variation
            noise = np.random.normal(0, base_volume * 0.02)  # Scenario-aware noise
            forecast_volumes.append(max(0, trend + seasonal + noise))
        
        # Create forecasting chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_volumes,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_volumes,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        # Confidence interval (simplified)
        upper_bound = [v * 1.1 for v in forecast_volumes]
        lower_bound = [v * 0.9 for v in forecast_volumes]
        
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
        
        fig.update_layout(
            title='Cargo Volume Forecast (6 Months)',
            xaxis_title='Date',
            yaxis_title='TEU Volume',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast metrics - scenario-aware
        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
        
        with forecast_col1:
            avg_forecast = np.mean(forecast_volumes)
            st.metric("Avg Forecast Volume", f"{avg_forecast:,.0f} TEU")
        
        with forecast_col2:
            growth_rate = ((forecast_volumes[-1] / hist_volumes[-1]) - 1) * 100
            st.metric("6-Month Growth", f"{growth_rate:.1f}%")
        
        with forecast_col3:
            # Confidence based on scenario efficiency
            base_confidence = 75
            confidence_boost = (efficiency_factor - 0.75) * 20  # Higher efficiency = higher confidence
            confidence = min(95, max(60, base_confidence + confidence_boost))
            st.metric("Forecast Confidence", f"{confidence:.0f}%")
    
    def _render_cargo_volume_revenue_analysis(self, params: Dict[str, Any]) -> None:
        """Render cargo volume and revenue analysis with scenario-specific data."""
        st.subheader("📊 Cargo Volume & Revenue Analysis")
        
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Generate scenario-aware data
        scenario_values = self._get_all_scenario_values()
        
        # Key metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_volume = scenario_values['cargo_volume']
            # Delta based on scenario efficiency
            delta_volume = (scenario_values['efficiency'] - 75) / 5  # Convert efficiency to percentage change
            st.metric(
                "Total Cargo Volume",
                f"{total_volume:,.0f} TEU",
                delta=f"{delta_volume:.1f}%"
            )
        
        with col2:
            # Revenue correlates with volume and efficiency
            total_revenue = total_volume * scenario_values['efficiency'] * 2.5  # Revenue factor
            delta_revenue = (scenario_values['efficiency'] - 75) / 4
            st.metric(
                "Total Revenue",
                f"${total_revenue:,.0f}",
                delta=f"{delta_revenue:.1f}%"
            )
        
        with col3:
            avg_handling = scenario_values['handling_time']
            # Delta inversely related to efficiency
            delta_handling = -(scenario_values['efficiency'] - 75) / 25  # Inverse relationship
            st.metric(
                "Avg Handling Time",
                f"{avg_handling:.1f} hrs",
                delta=f"{delta_handling:.1f} hrs"
            )
        
        with col4:
            # Trade balance based on throughput and efficiency
            trade_balance = scenario_values['throughput'] * scenario_values['efficiency'] * 100
            delta_balance = (scenario_values['efficiency'] - 75) / 7.5
            st.metric(
                "Trade Balance",
                f"${trade_balance:,.0f}",
                delta=f"{delta_balance:.1f}%"
            )
        
        # Volume trend chart
        st.subheader("📈 Monthly Volume Trends")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Generate scenario-aware monthly volumes with seasonal patterns
        base_volume = scenario_values['cargo_volume']
        monthly_volumes = []
        for i, month in enumerate(months):
            # Add seasonal pattern (higher in summer/fall)
            seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (i + 3) / 12)
            # Add scenario-aware variation
            monthly_volume = base_volume * seasonal_factor * np.random.normal(1, 0.1)
            monthly_volume = max(0, monthly_volume)
            monthly_volumes.append(monthly_volume)
        
        fig_volume = px.line(
            x=months, 
            y=monthly_volumes,
            title="Monthly Cargo Volume Trends",
            labels={'x': 'Month', 'y': 'Volume (TEU)'}
        )
        fig_volume.update_layout(height=400)
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Revenue breakdown
        st.subheader("💰 Revenue Breakdown")
        revenue_categories = ['Container Handling', 'Storage Fees', 'Equipment Rental', 'Logistics Services', 'Other']
        
        # Generate scenario-aware revenue distribution
        # Different scenarios have different revenue patterns
        efficiency_factor = scenario_values['efficiency'] / 100
        revenue_values = [
            total_revenue * 0.45 * efficiency_factor,  # Container Handling (main revenue)
            total_revenue * 0.25 * (1.2 - efficiency_factor * 0.2),  # Storage (inverse efficiency)
            total_revenue * 0.15 * efficiency_factor,  # Equipment Rental
            total_revenue * 0.10 * efficiency_factor,  # Logistics Services
            total_revenue * 0.05  # Other (constant)
        ]
        
        fig_revenue = px.pie(
            values=revenue_values,
            names=revenue_categories,
            title="Revenue Distribution by Service Type"
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

    def _render_cargo_types_analysis(self, params: Dict[str, Any]) -> None:
        """Render cargo types analysis."""
        st.subheader("📦 Cargo Type Analysis")
        
        # Generate scenario-aware cargo type data
        import numpy as np
        import plotly.express as px
        
        scenario_values = self._get_all_scenario_values()
        
        cargo_types = ['Containers', 'Bulk Dry', 'Bulk Liquid', 'Break Bulk', 'RoRo']
        
        # Generate scenario-aware cargo data with realistic distributions
        base_volume = scenario_values['cargo_volume']
        base_revenue = base_volume * scenario_values['efficiency'] * 2.5
        base_handling = scenario_values['handling_time']
        
        # Different cargo types have different characteristics
        cargo_factors = {
            'Containers': {'volume': 0.6, 'revenue': 0.4, 'handling': 0.8},
            'Bulk Dry': {'volume': 0.2, 'revenue': 0.15, 'handling': 1.2},
            'Bulk Liquid': {'volume': 0.1, 'revenue': 0.25, 'handling': 1.5},
            'Break Bulk': {'volume': 0.08, 'revenue': 0.15, 'handling': 2.0},
            'RoRo': {'volume': 0.02, 'revenue': 0.05, 'handling': 1.0}
        }
        
        cargo_data = {
            'Cargo_Type': cargo_types,
            'Volume_TEU': [int(base_volume * cargo_factors[ct]['volume'] * np.random.normal(1, 0.1)) for ct in cargo_types],
            'Revenue_HKD': [int(base_revenue * cargo_factors[ct]['revenue'] * np.random.normal(1, 0.15)) for ct in cargo_types],
            'Handling_Time': [base_handling * cargo_factors[ct]['handling'] * np.random.normal(1, 0.1) for ct in cargo_types]
        }
        
        cargo_df = pd.DataFrame(cargo_data)
        
        # Display cargo type statistics
        st.dataframe(cargo_df, use_container_width=True)
        
        # Cargo type analysis charts
        cargo_col1, cargo_col2 = st.columns(2)
        
        with cargo_col1:
            # Volume by cargo type
            fig_volume = px.treemap(
                cargo_df,
                path=['Cargo_Type'],
                values='Volume_TEU',
                title='Volume Distribution by Cargo Type'
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with cargo_col2:
            # Revenue vs Handling Time
            fig_revenue = px.scatter(
                cargo_df,
                x='Handling_Time',
                y='Revenue_HKD',
                size='Volume_TEU',
                color='Cargo_Type',
                title='Revenue vs Handling Time by Cargo Type',
                hover_data=['Volume_TEU']
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
    
    def _render_locations_analysis(self, params: Dict[str, Any]) -> None:
        """Render locations analysis."""
        st.subheader("🌍 Geographic Analysis")
        
        # Generate scenario-aware location data
        import numpy as np
        import plotly.express as px
        
        scenario_values = self._get_all_scenario_values()
        base_volume = scenario_values['cargo_volume']
        
        locations = ['Mainland China', 'Southeast Asia', 'Europe', 'North America', 'Other Asia']
        
        # Different regions have different trade patterns
        region_factors = {
            'Mainland China': {'import': 0.4, 'export': 0.35},
            'Southeast Asia': {'import': 0.25, 'export': 0.3},
            'Europe': {'import': 0.15, 'export': 0.2},
            'North America': {'import': 0.12, 'export': 0.1},
            'Other Asia': {'import': 0.08, 'export': 0.05}
        }
        
        location_data = {
            'Region': locations,
            'Import_TEU': [int(base_volume * region_factors[loc]['import'] * np.random.normal(1, 0.1)) for loc in locations],
            'Export_TEU': [int(base_volume * region_factors[loc]['export'] * np.random.normal(1, 0.1)) for loc in locations],
            'Trade_Balance': []
        }
        
        # Calculate trade balance based on import/export difference
        for i, loc in enumerate(locations):
            balance = location_data['Export_TEU'][i] - location_data['Import_TEU'][i]
            location_data['Trade_Balance'].append(balance * np.random.normal(100, 20))  # Convert to monetary value
        
        location_df = pd.DataFrame(location_data)
        location_df['Total_TEU'] = location_df['Import_TEU'] + location_df['Export_TEU']
        
        # Display location statistics
        st.dataframe(location_df, use_container_width=True)
        
        # Location analysis charts
        loc_col1, loc_col2 = st.columns(2)
        
        with loc_col1:
            # Import vs Export by region
            fig_trade = px.bar(
                location_df,
                x='Region',
                y=['Import_TEU', 'Export_TEU'],
                title='Import vs Export by Region',
                barmode='group'
            )
            fig_trade.update_layout(xaxis_title='Region', yaxis_title='TEU Volume')
            st.plotly_chart(fig_trade, use_container_width=True)
        
        with loc_col2:
            # Trade balance by region
            fig_balance = px.bar(
                location_df,
                x='Region',
                y='Trade_Balance',
                title='Trade Balance by Region',
                color='Trade_Balance',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            fig_balance.update_layout(xaxis_title='Region', yaxis_title='Trade Balance (TEU)')
            st.plotly_chart(fig_balance, use_container_width=True)
            
    def render_cargo_analysis_section(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render cargo analysis section.
        
        Args:
            scenario_data: Current scenario configuration and data
        """
        st.markdown("### 📦 Cargo Analysis")
        
        # Get scenario-specific parameters
        scenario_name = scenario_data.get('name', 'Normal Operations') if scenario_data else 'Normal Operations'
        params = self._get_scenario_performance_params(scenario_name)
        
        # Create tabs for different cargo analysis views
        cargo_tab1, cargo_tab2, cargo_tab3 = st.tabs([
            "📊 Volume & Revenue",
            "🚢 Cargo Types", 
            "🌍 Trade Routes"
        ])
        
        with cargo_tab1:
            self._render_cargo_volume_revenue_analysis(params)
            
        with cargo_tab2:
            self._render_cargo_types_analysis(params)
            
        with cargo_tab3:
            self._render_locations_analysis(params)

    def render_advanced_analysis_section(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render advanced analysis.
        
        This section consolidates content from the original Scenarios tab,
        providing sophisticated scenario modeling and planning tools.
        
        Args:
            scenario_data: Current scenario configuration and data
        """
        st.markdown("### 🔬 Advanced Analysis")
        st.markdown("Sophisticated scenario modeling including multi-scenario optimization, disruption impact simulation, and dynamic capacity planning.")
        
        # Create tabs for different advanced analysis features
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
            "🎯 Multi-Scenario Optimization",
            "⚠️ Disruption Impact Simulation", 
            "📈 Dynamic Capacity Planning"
        ])
        
        with analysis_tab1:
            self._render_multi_scenario_optimization()
            
        with analysis_tab2:
            self._render_disruption_impact_simulation()
            
        with analysis_tab3:
            self._render_dynamic_capacity_planning()
    
    def _render_multi_scenario_optimization(self) -> None:
        """Render multi-scenario optimization analysis."""
        st.subheader("🎯 Multi-Scenario Optimization Analysis")
        st.markdown("Optimize port operations across multiple scenarios simultaneously")
        
        # Optimization parameters
        opt_col1, opt_col2 = st.columns(2)
        
        with opt_col1:
            st.subheader("⚙️ Optimization Parameters")
            
            # Objective function selection
            objective = st.selectbox(
                "Optimization Objective",
                ["Minimize Total Waiting Time", "Maximize Throughput", "Minimize Costs", "Balanced Performance"],
                help="Select the primary optimization objective"
            )
            
            # Scenario weights
            st.subheader("📊 Scenario Weights")
            # Collapsible guidelines using details/summary HTML
            st.markdown("""
            <details>
            <summary><strong>ℹ️ Scenario Weight Guidelines</strong> (click to expand)</summary>
            <div style="margin-top: 10px; padding-left: 15px;">
            <strong>How to set scenario weights:</strong><br>
            • <strong>Higher weights</strong> (closer to 1.0) = More importance in optimization<br>
            • <strong>Lower weights</strong> (closer to 0.0) = Less importance in optimization<br>
            • <strong>Normal Operations</strong>: Base case - typically set to moderate weight (0.3-0.5)<br>
            • <strong>Peak Season</strong>: High cargo periods - increase if peak performance is critical<br>
            • <strong>Maintenance</strong>: Reduced capacity - increase if resilience during maintenance is important<br>
            • <strong>Typhoon Season</strong>: Weather constraints - increase if weather resilience is a priority
             </div>
             </details>
             """, unsafe_allow_html=True)
             
            # Add some spacing before the sliders
            st.markdown("<br>", unsafe_allow_html=True)
            
            normal_weight = st.slider("Normal Operations Weight", 0.0, 1.0, 0.4, 0.1)
            peak_weight = st.slider("Peak Season Weight", 0.0, 1.0, 0.3, 0.1)
            maintenance_weight = st.slider("Maintenance Weight", 0.0, 1.0, 0.2, 0.1)
            typhoon_weight = st.slider("Typhoon Season Weight", 0.0, 1.0, 0.1, 0.1)
            
            # Constraints
            st.subheader("🔒 Constraints")
            max_berths = st.number_input("Maximum Berths", min_value=1, max_value=50, value=20)
            max_cranes = st.number_input("Maximum Cranes", min_value=1, max_value=100, value=40)
            budget_constraint = st.number_input("Budget Constraint (M HKD)", min_value=0, value=1000)
            
            if st.button("🚀 Run Optimization"):
                with st.spinner("Running multi-scenario optimization..."):
                    # Simulate optimization process
                    import time
                    time.sleep(2)
                    
                    # Generate optimization results based on selected objective
                    optimization_results = self._generate_optimization_results(
                        objective=objective,
                        weights={
                            'normal': normal_weight,
                            'peak_season': peak_weight,
                            'maintenance': maintenance_weight,
                            'typhoon_season': typhoon_weight
                        },
                        constraints={
                            'max_berths': max_berths,
                            'max_cranes': max_cranes,
                            'budget': budget_constraint
                        }
                    )
                    
                    # Store optimization results
                    st.session_state.optimization_results = optimization_results
                    st.success(f"Optimization completed for: {objective}!")
        
        with opt_col2:
            st.subheader("📊 Optimization Results")
            
            if hasattr(st.session_state, 'optimization_results'):
                results = st.session_state.optimization_results
                
                # Display optimization objective used
                st.info(f"**Optimization Objective:** {results.get('objective', 'Unknown')}")
                
                # Display key metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Objective Value", f"{results['objective_value']:.3f}")
                with metric_col2:
                    st.metric("Optimal Berths", results['optimal_berths'])
                with metric_col3:
                    st.metric("Optimal Cranes", results['optimal_cranes'])
                
                # Scenario performance chart
                st.markdown("""
                <details>
                <summary><strong>ℹ️ Performance by Scenario Explanation</strong> (click to expand)</summary>
                <div style="margin-top: 10px; padding-left: 15px;">
                This chart shows how well the optimized port configuration performs under different operational scenarios:<br>
                • <strong>Normal</strong>: Standard operating conditions<br>
                • <strong>Peak Season</strong>: High cargo volume periods (e.g., holiday seasons)<br>
                • <strong>Maintenance</strong>: Reduced capacity due to equipment maintenance<br>
                • <strong>Typhoon Season</strong>: Weather-related operational constraints<br><br>
                Higher values (green) indicate better performance, lower values (red) indicate challenges.
                </div>
                </details>
                """, unsafe_allow_html=True)
                
                import plotly.express as px
                perf_data = pd.DataFrame([
                    {'Scenario': k.replace('_', ' ').title(), 'Performance': v}
                    for k, v in results['scenario_performance'].items()
                ])
                
                fig = px.bar(
                    perf_data,
                    x='Scenario',
                    y='Performance',
                    title='Optimization Performance Across Different Scenarios',
                    color='Performance',
                    color_continuous_scale='RdYlGn',
                    labels={'Performance': 'Performance Score (0-1)', 'Scenario': 'Operational Scenario'}
                )
                fig.update_layout(
                    yaxis=dict(range=[0, 1]),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run optimization to see results here")
    
    def _render_disruption_impact_simulation(self) -> None:
        """Render disruption impact simulation."""
        st.subheader("⚠️ Predictive Disruption Impact Simulation")
        st.markdown("Simulate and analyze the impact of various disruptions on port operations")
        
        # Disruption configuration
        disrupt_col1, disrupt_col2 = st.columns(2)
        
        with disrupt_col1:
            st.subheader("⚙️ Disruption Configuration")
            
            # Disruption type
            disruption_type = st.selectbox(
                "Disruption Type",
                ["Typhoon", "Equipment Failure", "Labor Strike", "Cyber Attack", "Supply Chain Disruption"],
                help="Select the type of disruption to simulate"
            )
            
            # Severity and duration
            severity = st.slider("Disruption Severity", 1, 10, 5, help="1 = Minor, 10 = Severe")
            duration = st.slider("Duration (hours)", 1, 168, 24)
            
            # Affected areas
            st.subheader("🎯 Affected Areas")
            affected_berths = st.multiselect(
                "Affected Berths",
                [f"Berth {i}" for i in range(1, 21)],
                default=["Berth 1", "Berth 2"]
            )
            
            affected_equipment = st.multiselect(
                "Affected Equipment",
                ["Cranes", "Trucks", "Conveyor Systems", "IT Systems"],
                default=["Cranes"]
            )
            
            # Mitigation strategies
            st.subheader("🛡️ Mitigation Strategies")
            enable_backup = st.checkbox("Enable Backup Systems", value=True)
            reroute_traffic = st.checkbox("Reroute Traffic", value=True)
            emergency_protocols = st.checkbox("Activate Emergency Protocols", value=False)
            
            if st.button("🔥 Simulate Disruption"):
                with st.spinner("Simulating disruption impact..."):
                    import time
                    import numpy as np
                    time.sleep(2)
                    
                    # Get scenario-aware baseline values
                    scenario_values = self._get_all_scenario_values()
                    base_throughput = scenario_values['throughput']
                    base_efficiency = scenario_values['efficiency']
                    base_revenue = scenario_values['revenue']
                    
                    # Calculate berth impact factor (more affected berths = higher impact)
                    total_berths = 20  # Total berths in port
                    affected_berth_count = len(affected_berths)
                    berth_impact_factor = affected_berth_count / total_berths
                    
                    # Calculate equipment impact factor
                    equipment_impact_factor = len(affected_equipment) / 4  # 4 total equipment types
                    
                    # Combined impact factor (berths have more weight)
                    combined_impact = (berth_impact_factor * 0.7) + (equipment_impact_factor * 0.3)
                    
                    # Dynamic impact calculations based on scenario and affected resources
                    impact_multiplier = severity * combined_impact
                    
                    # Throughput reduction: higher for more affected berths and higher severity
                    throughput_reduction = min(95, impact_multiplier * 15)  # Cap at 95% reduction
                    affected_throughput_pct = max(5, 100 - throughput_reduction)
                    
                    # Recovery time: longer for more affected berths, shorter with mitigation
                    base_recovery = duration * (0.8 + berth_impact_factor * 0.7)
                    mitigation_factor = 1.0
                    if enable_backup:
                        mitigation_factor *= 0.7
                    if reroute_traffic:
                        mitigation_factor *= 0.8
                    if emergency_protocols:
                        mitigation_factor *= 0.6
                    
                    recovery_time = base_recovery * mitigation_factor
                    
                    # Financial impact: based on scenario revenue and affected capacity
                    daily_revenue = base_revenue / 365  # Convert annual to daily
                    capacity_loss = throughput_reduction / 100
                    financial_impact = daily_revenue * capacity_loss * (duration / 24)
                    
                    # Generate dynamic recovery timeline with enhanced sensitivity
                    max_timeline_hours = min(int(recovery_time * 1.5) + 6, 72)  # Extend timeline based on recovery time
                    timeline_hours = list(range(0, max_timeline_hours, 1))  # Every hour for better granularity
                    timeline = []
                    
                    # Enhanced recovery rate calculation based on multiple factors
                    base_recovery_rate = 0.08 + (base_efficiency / 100) * 0.04  # Base rate varies with efficiency
                    
                    # Adjust recovery rate based on impact severity
                    severity_adjustment = 1.0 - (berth_impact_factor * 0.3)  # More affected berths = slower recovery
                    equipment_adjustment = 1.0 - (equipment_impact_factor * 0.2)  # Equipment issues slow recovery
                    
                    adjusted_recovery_rate = base_recovery_rate * severity_adjustment * equipment_adjustment
                    
                    # Calculate mitigation multiplier
                    mitigation_multiplier = 1.0
                    if enable_backup:
                        mitigation_multiplier *= 1.4  # Backup systems significantly help
                    if reroute_traffic:
                        mitigation_multiplier *= 1.25  # Traffic rerouting helps moderately
                    if emergency_protocols:
                        mitigation_multiplier *= 1.5  # Emergency protocols help most
                    
                    for hour in timeline_hours:
                        if hour == 0:
                            # Initial impact
                            throughput_pct = affected_throughput_pct
                        else:
                            # Enhanced recovery curve with dynamic parameters
                            # Different recovery phases: immediate (0-4h), intermediate (4-12h), final (12h+)
                            if hour <= 4:
                                # Slow initial recovery
                                phase_rate = adjusted_recovery_rate * 0.6
                                mitigation_effect = 1.0 if hour <= 1 else mitigation_multiplier * 0.7
                            elif hour <= 12:
                                # Accelerated intermediate recovery
                                phase_rate = adjusted_recovery_rate * 1.2
                                mitigation_effect = mitigation_multiplier
                            else:
                                # Gradual final recovery
                                phase_rate = adjusted_recovery_rate * 0.8
                                mitigation_effect = mitigation_multiplier * 1.1
                            
                            # Calculate recovery progress with phase-specific rates
                            recovery_progress = 1 - np.exp(-phase_rate * mitigation_effect * hour)
                            
                            # Add some variability based on scenario complexity
                            scenario_complexity = berth_impact_factor + equipment_impact_factor
                            if scenario_complexity > 0.5:  # High complexity scenarios have more variable recovery
                                recovery_progress *= (0.95 + 0.1 * np.random.random())  # ±5% variability
                            
                            # Calculate current throughput percentage
                            throughput_pct = affected_throughput_pct + (100 - affected_throughput_pct) * min(1.0, recovery_progress)
                        
                        timeline.append({
                            'hour': hour,
                            'throughput': min(100, throughput_pct)  # Cap at 100%
                        })
                    
                    # Store dynamic simulation results
                    st.session_state.disruption_results = {
                        'impact_score': min(10, impact_multiplier),
                        'affected_throughput': affected_throughput_pct,
                        'recovery_time': recovery_time,
                        'financial_impact': financial_impact,
                        'timeline': timeline,
                        'berth_impact_factor': berth_impact_factor,
                        'equipment_impact_factor': equipment_impact_factor,
                        'mitigation_effectiveness': (1 - mitigation_factor) * 100
                    }
                    st.success("Disruption simulation completed!")
        
        with disrupt_col2:
            st.subheader("📊 Impact Analysis")
            
            # Add collapsible explanation for Impact Analysis metrics
            st.markdown("""
            <details>
            <summary><strong>ℹ️ Impact Analysis Metrics Explanation</strong> (click to expand)</summary>
            <div style="margin-top: 10px; padding-left: 15px;">
            This section provides comprehensive analysis of disruption impacts on port operations:<br><br>
            • <strong>Impact Score</strong>: Overall severity rating (0-10 scale) based on affected berths and equipment<br>
            • <strong>Recovery Time</strong>: Estimated hours to return to 95% operational capacity<br>
            • <strong>Berth Impact Factor</strong>: Percentage of total berth capacity affected by the disruption<br>
            • <strong>Throughput Reduction</strong>: Immediate decrease in cargo handling capacity<br>
            • <strong>Financial Impact</strong>: Estimated revenue loss during disruption period<br>
            • <strong>Mitigation Effectiveness</strong>: How well selected strategies reduce overall impact<br>
            • <strong>Equipment Impact</strong>: Additional capacity loss from equipment unavailability<br><br>
            The recovery timeline shows throughput restoration over time, with milestones at 50%, 80%, and 95% recovery levels.
            </div>
            </details>
            """, unsafe_allow_html=True)
            
            if hasattr(st.session_state, 'disruption_results'):
                results = st.session_state.disruption_results
                
                # Enhanced impact metrics with additional insights
                impact_col1, impact_col2, impact_col3 = st.columns(3)
                with impact_col1:
                    st.metric("Impact Score", f"{results['impact_score']:.1f}/10")
                    st.metric("Throughput Reduction", f"{100 - results['affected_throughput']:.0f}%")
                with impact_col2:
                    st.metric("Recovery Time", f"{results['recovery_time']:.1f} hrs")
                    st.metric("Financial Impact", f"${results['financial_impact']:,.0f}")
                with impact_col3:
                    st.metric("Berth Impact Factor", f"{results['berth_impact_factor']:.1%}")
                    st.metric("Mitigation Effectiveness", f"{results['mitigation_effectiveness']:.1f}%")
                
                # Additional impact details
                st.markdown("**Impact Breakdown:**")
                impact_details_col1, impact_details_col2 = st.columns(2)
                with impact_details_col1:
                    st.write(f"• Affected Berths: {len(affected_berths)}/20 ({results['berth_impact_factor']:.1%})")
                    st.write(f"• Equipment Impact: {results['equipment_impact_factor']:.1%}")
                with impact_details_col2:
                    mitigation_status = []
                    if enable_backup:
                        mitigation_status.append("✅ Backup Systems")
                    if reroute_traffic:
                        mitigation_status.append("✅ Traffic Rerouting")
                    if emergency_protocols:
                        mitigation_status.append("✅ Emergency Protocols")
                    if not mitigation_status:
                        mitigation_status.append("❌ No Mitigation")
                    
                    st.write("**Active Mitigations:**")
                    for status in mitigation_status:
                        st.write(f"• {status}")
                
                # Enhanced recovery timeline with scenario context
                if results['timeline']:
                    st.subheader("🔄 Throughput Recovery Timeline")
                    
                    timeline_df = pd.DataFrame(results['timeline'])
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    # Create enhanced recovery chart
                    fig = go.Figure()
                    
                    # Add recovery line
                    fig.add_trace(go.Scatter(
                        x=timeline_df['hour'],
                        y=timeline_df['throughput'],
                        mode='lines+markers',
                        name='Throughput Recovery',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=6)
                    ))
                    
                    # Add 100% baseline
                    fig.add_hline(
                        y=100, 
                        line_dash="dash", 
                        line_color="green",
                        annotation_text="Normal Operations (100%)"
                    )
                    
                    # Add impact threshold
                    fig.add_hline(
                        y=results['affected_throughput'], 
                        line_dash="dot", 
                        line_color="red",
                        annotation_text=f"Initial Impact ({results['affected_throughput']:.0f}%)"
                    )
                    
                    fig.update_layout(
                        title='Throughput Recovery Timeline (Dynamic Based on Affected Berths)',
                        xaxis_title='Hours After Disruption',
                        yaxis_title='Throughput Capacity (%)',
                        yaxis=dict(range=[0, 105]),
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recovery statistics
                    recovery_col1, recovery_col2, recovery_col3 = st.columns(3)
                    with recovery_col1:
                        time_to_50 = next((t['hour'] for t in results['timeline'] if t['throughput'] >= 50), "N/A")
                        st.metric("Time to 50% Recovery", f"{time_to_50} hrs" if time_to_50 != "N/A" else "N/A")
                    with recovery_col2:
                        time_to_80 = next((t['hour'] for t in results['timeline'] if t['throughput'] >= 80), "N/A")
                        st.metric("Time to 80% Recovery", f"{time_to_80} hrs" if time_to_80 != "N/A" else "N/A")
                    with recovery_col3:
                        time_to_95 = next((t['hour'] for t in results['timeline'] if t['throughput'] >= 95), "N/A")
                        st.metric("Time to 95% Recovery", f"{time_to_95} hrs" if time_to_95 != "N/A" else "N/A")
            else:
                st.info("Run disruption simulation to see impact analysis")
    
    def _render_dynamic_capacity_planning(self) -> None:
        """Render dynamic capacity planning and investment simulation."""
        st.subheader("📈 Dynamic Capacity Planning & Investment Simulation")
        st.markdown("Plan capacity expansions and evaluate investment scenarios")
        
        # Investment planning
        invest_col1, invest_col2 = st.columns(2)
        
        with invest_col1:
            st.subheader("💰 Investment Planning")
            
            # Investment options
            st.subheader("🏗️ Infrastructure Investments")
            new_berths = st.number_input("Additional Berths", min_value=0, max_value=10, value=2)
            new_cranes = st.number_input("Additional Cranes", min_value=0, max_value=20, value=4)
            automation_level = st.slider("Automation Level", 0, 100, 30, help="Percentage of automated operations")
            
            # Investment costs (in millions HKD)
            st.subheader("💵 Investment Costs")
            berth_cost = st.number_input("Cost per Berth (M HKD)", min_value=0, value=100)
            crane_cost = st.number_input("Cost per Crane (M HKD)", min_value=0, value=20)
            automation_cost = st.number_input("Automation Cost (M HKD)", min_value=0, value=50)
            
            # Timeline
            implementation_time = st.slider("Implementation Timeline (months)", 6, 60, 24)
            analysis_period = st.slider("Analysis Period (years)", 5, 20, 10)
            
            # Economic parameters
            st.subheader("📊 Economic Parameters")
            discount_rate = st.slider("Discount Rate (%)", 1.0, 10.0, 5.0, 0.5)
            growth_rate = st.slider("Traffic Growth Rate (%/year)", 0.0, 10.0, 3.0, 0.5)
            
            if st.button("📊 Analyze Investment"):
                with st.spinner("Analyzing investment scenario..."):
                    import time
                    time.sleep(2)
                    
                    # Get current scenario for dynamic investment analysis
                    current_scenario = self._get_current_scenario()
                    scenario_params = self._get_scenario_performance_params(current_scenario)
                    
                    # Calculate investment metrics
                    total_investment = (new_berths * berth_cost + 
                                      new_cranes * crane_cost + 
                                      automation_cost * automation_level / 100)
                    
                    # Dynamic ROI calculation based on scenario
                    # Peak Season: Higher returns due to increased demand (18-25%)
                    # Normal Operations: Standard returns (12-18%)
                    # Low Season: Lower but stable returns (8-15%)
                    if "Peak Season" in current_scenario:
                        base_roi_rate = self._generate_scenario_values(current_scenario, 'investment_roi', 1, roi_range=(0.18, 0.25))
                        market_multiplier = 1.2  # Higher market demand
                        efficiency_bonus = automation_level * 0.002  # Automation bonus in peak
                    elif "Low Season" in current_scenario:
                        base_roi_rate = self._generate_scenario_values(current_scenario, 'investment_roi', 1, roi_range=(0.08, 0.15))
                        market_multiplier = 0.85  # Lower market demand
                        efficiency_bonus = automation_level * 0.003  # Higher automation value in low season
                    else:  # Normal Operations
                        base_roi_rate = self._generate_scenario_values(current_scenario, 'investment_roi', 1, roi_range=(0.12, 0.18))
                        market_multiplier = 1.0  # Standard market conditions
                        efficiency_bonus = automation_level * 0.0025  # Standard automation bonus
                    
                    # Apply scenario-specific adjustments
                    scenario_roi_rate = base_roi_rate + efficiency_bonus
                    
                    # Calculate infrastructure impact multipliers
                    berth_impact = new_berths * 0.08 * market_multiplier  # Berths have higher impact in peak
                    crane_impact = new_cranes * 0.04 * (1 + automation_level / 300)  # Cranes benefit from automation
                    
                    # Dynamic annual revenue calculation
                    base_annual_revenue = total_investment * scenario_roi_rate
                    infrastructure_bonus = berth_impact + crane_impact
                    annual_revenue_increase = base_annual_revenue * (1 + infrastructure_bonus)
                    
                    # Calculate payback period with scenario considerations
                    payback_period = total_investment / annual_revenue_increase if annual_revenue_increase > 0 else float('inf')
                    
                    # Dynamic NPV calculation with proper discounting
                    npv = sum([
                        annual_revenue_increase * (1 + growth_rate/100) ** i / (1 + discount_rate/100) ** i
                        for i in range(1, analysis_period + 1)
                    ]) - total_investment
                    
                    # Dynamic capacity increase calculation
                    base_capacity_increase = (new_berths * 10 + new_cranes * 5)
                    automation_multiplier = (1 + automation_level / 200)
                    scenario_efficiency = scenario_params.get('efficiency_range', (80, 90))[1] / 100
                    capacity_increase = base_capacity_increase * automation_multiplier * scenario_efficiency
                    
                    st.session_state.investment_results = {
                        'total_investment': total_investment,
                        'annual_revenue_increase': annual_revenue_increase,
                        'payback_period': payback_period,
                        'npv': npv,
                        'capacity_increase': capacity_increase,
                        'scenario_context': {
                            'scenario_name': current_scenario,
                            'base_roi_rate': base_roi_rate * 100,
                            'market_multiplier': market_multiplier,
                            'efficiency_bonus': efficiency_bonus * 100
                        },
                        'yearly_projections': [
                            {
                                'year': i,
                                'revenue': annual_revenue_increase * (1 + growth_rate/100) ** i,
                                'discounted_revenue': annual_revenue_increase * (1 + growth_rate/100) ** i / (1 + discount_rate/100) ** i,
                                'cumulative_profit': sum([
                                    annual_revenue_increase * (1 + growth_rate/100) ** j / (1 + discount_rate/100) ** j
                                    for j in range(1, i + 1)
                                ]) - total_investment
                            }
                            for i in range(1, analysis_period + 1)
                        ]
                    }
                    st.success(f"Investment analysis completed for {current_scenario}!")
        
        with invest_col2:
            st.subheader("📊 Investment Analysis")
            
            if hasattr(st.session_state, 'investment_results'):
                results = st.session_state.investment_results
                
                # Show scenario context
                if 'scenario_context' in results:
                    scenario_ctx = results['scenario_context']
                    st.info(f"**Analysis for {scenario_ctx['scenario_name']}**\n"
                           f"Base ROI Rate: {scenario_ctx['base_roi_rate']:.1f}% | "
                           f"Market Factor: {scenario_ctx['market_multiplier']:.1f}x | "
                           f"Automation Bonus: +{scenario_ctx['efficiency_bonus']:.1f}%")
                
                # Key metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Total Investment", f"${results['total_investment']:.0f}M")
                    st.metric("Annual Revenue Increase", f"${results['annual_revenue_increase']:.1f}M")
                with metric_col2:
                    st.metric("Payback Period", f"{results['payback_period']:.1f} years")
                    st.metric("NPV", f"${results['npv']:.1f}M")
                
                st.metric("Capacity Increase", f"{results['capacity_increase']:.0f}%")
                
                # Investment projection chart
                if results['yearly_projections']:
                    proj_df = pd.DataFrame(results['yearly_projections'])
                    import plotly.express as px
                    
                    fig = px.line(
                        proj_df,
                        x='year',
                        y='cumulative_profit',
                        title='Cumulative Profit Projection',
                        labels={'year': 'Year', 'cumulative_profit': 'Cumulative Profit (M HKD)'}
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Dynamic investment recommendation based on scenario
                scenario_name = results.get('scenario_context', {}).get('scenario_name', 'Current Scenario')
                npv = results['npv']
                payback = results['payback_period']
                
                if scenario_name == "Peak Season":
                    if npv > 200 and payback < 6:
                        st.success(f"✅ **Strong Recommendation for {scenario_name}**: Excellent returns with high demand conditions. NPV: ${npv:.1f}M")
                    elif npv > 0 and payback < 8:
                        st.info(f"📈 **Good Investment for {scenario_name}**: Solid returns during peak demand. Consider timing with seasonal patterns.")
                    else:
                        st.warning(f"⚠️ **Caution for {scenario_name}**: Even peak conditions show marginal returns. Review investment scope.")
                elif scenario_name == "Low Season":
                    if npv > 50 and payback < 8:
                        st.success(f"✅ **Resilient Investment**: Shows positive returns even in {scenario_name} conditions. NPV: ${npv:.1f}M")
                    elif npv > 0:
                        st.warning(f"⚠️ **Conservative Approach for {scenario_name}**: Positive but modest returns. Consider seasonal financing.")
                    else:
                        st.error(f"❌ **High Risk in {scenario_name}**: Investment may not be viable during low demand periods.")
                else:  # Normal Operations
                    if npv > 100 and payback < 7:
                        st.success(f"✅ **Recommended for {scenario_name}**: Balanced returns with stable demand. NPV: ${npv:.1f}M")
                    elif npv > 0 and payback < 10:
                        st.info(f"📊 **Viable for {scenario_name}**: Reasonable returns under normal conditions. Monitor market trends.")
                    else:
                        st.warning(f"⚠️ **Review Required**: Investment shows limited returns in {scenario_name}. Consider alternatives.")
            else:
                st.info("Configure and analyze an investment scenario to see results here")
    
    def _generate_optimization_results(self, objective: str, weights: Dict[str, float], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization results based on selected objective and parameters.
        
        Args:
            objective: Selected optimization objective
            weights: Scenario weights dictionary
            constraints: Optimization constraints
            
        Returns:
            Dictionary containing optimization results
        """
        import random
        
        # Set random seed for consistent results within the same session
        random.seed(hash(objective + str(sorted(weights.items())) + str(sorted(constraints.items()))) % 1000)
        
        # Base performance values that vary by objective
        if objective == "Minimize Total Waiting Time":
            base_performance = {
                'normal': random.uniform(0.88, 0.95),
                'peak_season': random.uniform(0.75, 0.85),
                'maintenance': random.uniform(0.82, 0.90),
                'typhoon_season': random.uniform(0.68, 0.78)
            }
            objective_value = random.uniform(0.82, 0.92)
            optimal_berths = random.randint(16, 22)
            optimal_cranes = random.randint(32, 42)
            
        elif objective == "Maximize Throughput":
            base_performance = {
                'normal': random.uniform(0.90, 0.97),
                'peak_season': random.uniform(0.85, 0.92),
                'maintenance': random.uniform(0.78, 0.88),
                'typhoon_season': random.uniform(0.70, 0.80)
            }
            objective_value = random.uniform(0.85, 0.95)
            optimal_berths = random.randint(18, 25)
            optimal_cranes = random.randint(35, 45)
            
        elif objective == "Minimize Costs":
            base_performance = {
                'normal': random.uniform(0.85, 0.92),
                'peak_season': random.uniform(0.72, 0.82),
                'maintenance': random.uniform(0.80, 0.87),
                'typhoon_season': random.uniform(0.65, 0.75)
            }
            objective_value = random.uniform(0.78, 0.88)
            optimal_berths = random.randint(14, 20)
            optimal_cranes = random.randint(28, 38)
            
        else:  # Balanced Performance
            base_performance = {
                'normal': random.uniform(0.87, 0.94),
                'peak_season': random.uniform(0.76, 0.86),
                'maintenance': random.uniform(0.81, 0.89),
                'typhoon_season': random.uniform(0.67, 0.77)
            }
            objective_value = random.uniform(0.80, 0.90)
            optimal_berths = random.randint(16, 22)
            optimal_cranes = random.randint(30, 40)
        
        # Apply weight adjustments to performance
        weighted_performance = {}
        for scenario, perf in base_performance.items():
            weight = weights.get(scenario, 0.25)
            # Higher weights should lead to better optimization for that scenario
            weight_adjustment = 1 + (weight - 0.25) * 0.2  # Scale weight impact
            weighted_performance[scenario] = min(0.98, perf * weight_adjustment)
        
        # Apply constraint adjustments
        constraint_factor = 1.0
        if constraints['max_berths'] < optimal_berths:
            constraint_factor *= 0.95  # Reduce performance if berth-constrained
            optimal_berths = constraints['max_berths']
        if constraints['max_cranes'] < optimal_cranes:
            constraint_factor *= 0.93  # Reduce performance if crane-constrained
            optimal_cranes = constraints['max_cranes']
        
        # Apply constraint factor to all performance values
        for scenario in weighted_performance:
            weighted_performance[scenario] *= constraint_factor
        
        objective_value *= constraint_factor
        
        return {
            'objective': objective,
            'objective_value': round(objective_value, 3),
            'optimal_berths': optimal_berths,
            'optimal_cranes': optimal_cranes,
            'scenario_performance': {k: round(v, 3) for k, v in weighted_performance.items()},
            'weights_used': weights,
            'constraints_applied': constraints
        }


# Convenience function for easy integration
def render_consolidated_scenarios_tab(scenario_data: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to render the consolidated scenarios tab.
    
    Args:
        scenario_data: Current scenario configuration and data
    """
    tab = ConsolidatedScenariosTab()
    tab.render_consolidated_tab(scenario_data)