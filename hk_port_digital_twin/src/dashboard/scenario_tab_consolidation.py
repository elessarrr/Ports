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
        return {
            'show_section_descriptions': True,
            'enable_expand_collapse_all': True,
            'show_section_navigation': True,
            'remember_section_states': True,
            'scenarios_sections_expanded': True,
            'section_auto_scroll': True,
            'enable_quick_export': True
        }
    
    def get_default_section_states():
        return {
            'overview': True,
            'operations': True,
            'analytics': True,
            'cargo': True,
            'advanced': True
        }

# Import existing visualization and data functions
# These will be imported as needed from existing modules


class ConsolidatedScenariosTab:
    """Main class for managing the consolidated scenarios tab functionality."""
    
    def __init__(self):
        """Initialize the consolidated scenarios tab."""
        self.sections = {
            'overview': {
                'title': 'Scenario Selection & Overview',
                'icon': '📊',
                'description': 'Select and configure simulation scenarios with key performance indicators'
            },
            'operations': {
                'title': 'Operational Impact',
                'icon': '🚢', 
                'description': 'Monitor ships, berths, and operational metrics affected by scenarios'
            },
            'analytics': {
                'title': 'Performance Analytics',
                'icon': '📈',
                'description': 'Analyze performance trends and KPIs across different scenarios'
            },
            'cargo': {
                'title': 'Cargo Analysis',
                'icon': '📦',
                'description': 'Track cargo statistics and throughput metrics by scenario'
            },
            'advanced': {
                'title': 'Advanced Analysis',
                'icon': '🔬',
                'description': 'Deep-dive scenario comparisons and advanced simulation features'
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
        
        # Render all sections
        for section_key, section_info in self.sections.items():
            self._render_section(section_key, section_info, scenario_data)
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables for the consolidated tab."""
        # Initialize section states based on preferences
        if 'consolidated_sections_state' not in st.session_state:
            if self.preferences.get('remember_section_states', True):
                # Use default states from settings
                st.session_state.consolidated_sections_state = self.default_states.copy()
            else:
                # Use preference setting for all sections
                expanded_default = self.preferences.get('scenarios_sections_expanded', True)
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
                is_expanded = st.session_state.consolidated_sections_state.get(section_key, True)
                status_icon = "📖" if is_expanded else "📕"
                
                button_label = f"{status_icon} {section_info['icon']} {section_info['title']}"
                
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
        is_expanded = st.session_state.consolidated_sections_state.get(section_key, True)
        
        # Create section header with anchor
        section_title = f"{section_info['icon']} {section_info['title']}"
        
        # Add anchor point for navigation
        st.markdown(f'<div id="section-{section_key}"></div>', unsafe_allow_html=True)
        
        # Create expandable section
        with st.expander(section_title, expanded=is_expanded):
            # Show description if enabled
            if self.preferences.get('show_section_descriptions', True):
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
                    if st.button(f"📊 Export {section_info['title']}", key=f"export_{section_key}"):
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
        st.markdown("### 🎯 Scenario Selection & Overview")
        st.markdown("Select and configure simulation scenarios for analysis")
        
        # Scenario Analysis & Comparison (migrated from existing tab)
        st.subheader("📊 Scenario Analysis & Comparison")
        st.markdown("Compare different operational scenarios to optimize port performance")
        
        # Scenario selection interface
        scenario_col1, scenario_col2 = st.columns([1, 2])
        
        with scenario_col1:
            st.subheader("🎯 Scenario Selection")
            
            # Get available scenarios
            try:
                from hk_port_digital_twin.src.scenarios import list_available_scenarios
                available_scenarios = list_available_scenarios()
            except ImportError:
                available_scenarios = ['normal', 'peak_season', 'maintenance', 'typhoon_season']
            
            # Primary scenario selection
            primary_scenario = st.selectbox(
                "Primary Scenario",
                available_scenarios,
                help="Select the main scenario for analysis"
            )
            
            # Comparison scenario selection
            comparison_scenarios = st.multiselect(
                "Comparison Scenarios",
                [s for s in available_scenarios if s != primary_scenario],
                help="Select scenarios to compare against the primary scenario"
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
                        label="📥 Export Comparison Results",
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
            
            # Generate sample queue data
            import numpy as np
            sample_queue = [
                {'ship_id': f'SHIP-{i:03d}', 'ship_type': np.random.choice(['Container', 'Bulk', 'Tanker']),
                 'arrival_time': f'{np.random.randint(0, 24):02d}:00', 'waiting_time': np.random.exponential(2),
                 'cargo_volume': np.random.randint(500, 3000), 'priority': np.random.choice(['normal', 'high'], p=[0.8, 0.2])}
                for i in range(np.random.randint(5, 15))
            ]
            
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
            
            # Generate sample berth data
            import numpy as np
            berth_statuses = ['occupied', 'available', 'maintenance']
            sample_berths = [
                {
                    'berth_id': f'B{i:02d}',
                    'status': np.random.choice(berth_statuses, p=[0.6, 0.3, 0.1]),
                    'current_ship': f'SHIP-{np.random.randint(100, 999)}' if np.random.random() > 0.4 else None,
                    'utilization': np.random.uniform(0, 100),
                    'throughput': np.random.randint(0, 5000),
                    'last_updated': datetime.now().strftime('%H:%M:%S')
                }
                for i in range(1, 21)  # 20 berths
            ]
            
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
            
            # Operational efficiency metrics
            import numpy as np
            efficiency_metrics = {
                'Port Efficiency': np.random.uniform(75, 95),
                'Crane Productivity': np.random.uniform(80, 100),
                'Truck Turnaround': np.random.uniform(60, 90),
                'Vessel Turnaround': np.random.uniform(70, 95)
            }
            
            for metric, value in efficiency_metrics.items():
                st.metric(metric, f"{value:.1f}%")
        
        with ops_col2:
            st.subheader("📈 Performance Trends")
            
            # Generate sample trend data
            import numpy as np
            hours = list(range(24))
            throughput_trend = [np.random.uniform(80, 120) for _ in hours]
            
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
        analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
            "📊 Data Export", "📈 Throughput Analysis", "⏱️ Waiting Time Analysis", "🎯 Performance Metrics"
        ])
        
        with analytics_tab1:
            self._render_data_export_section(scenario_data)
            
        with analytics_tab2:
            self._render_throughput_analysis(scenario_data)
            
        with analytics_tab3:
            self._render_waiting_time_analysis(scenario_data)
            
        with analytics_tab4:
            self._render_performance_metrics(scenario_data)
    
    def _render_data_export_section(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render data export functionality."""
        st.subheader("📊 Data Export & Download")
        st.markdown("Export simulation data and analytics for external analysis")
        
        # Export options
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.subheader("📈 Available Data Sets")
            
            # Berth data export
            if st.button("📥 Export Berth Data"):
                try:
                    # Get berth data from simulation or generate sample
                    simulation_data = getattr(st.session_state, 'simulation_data', None)
                    if simulation_data and hasattr(simulation_data, 'berth_data'):
                        berth_data = simulation_data.berth_data
                    else:
                        # Generate sample berth data
                        import numpy as np
                        berth_data = [
                            {
                                'berth_id': f'B{i:02d}',
                                'utilization': np.random.uniform(60, 95),
                                'throughput': np.random.randint(1000, 5000),
                                'ships_served': np.random.randint(5, 20),
                                'avg_service_time': np.random.uniform(8, 24)
                            }
                            for i in range(1, 21)
                        ]
                    
                    berth_df = pd.DataFrame(berth_data)
                    csv_data = berth_df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download Berth Data CSV",
                        data=csv_data,
                        file_name=f"berth_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.success("Berth data prepared for download!")
                except Exception as e:
                    st.error(f"Error preparing berth data: {str(e)}")
            
            # Queue data export
            if st.button("📥 Export Queue Data"):
                try:
                    # Get queue data from simulation or generate sample
                    simulation_data = getattr(st.session_state, 'simulation_data', None)
                    if simulation_data and hasattr(simulation_data, 'ship_queue'):
                        queue_data = simulation_data.ship_queue
                    else:
                        # Generate sample queue data
                        import numpy as np
                        queue_data = [
                            {
                                'ship_id': f'SHIP-{i:03d}',
                                'ship_type': np.random.choice(['Container', 'Bulk', 'Tanker']),
                                'arrival_time': f'{np.random.randint(0, 24):02d}:00',
                                'waiting_time': np.random.exponential(2),
                                'service_time': np.random.uniform(4, 16),
                                'cargo_volume': np.random.randint(500, 3000)
                            }
                            for i in range(50)
                        ]
                    
                    queue_df = pd.DataFrame(queue_data)
                    csv_data = queue_df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download Queue Data CSV",
                        data=csv_data,
                        file_name=f"queue_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.success("Queue data prepared for download!")
                except Exception as e:
                    st.error(f"Error preparing queue data: {str(e)}")
        
        with export_col2:
            st.subheader("📋 Export Options")
            
            # Timeline data export
            if st.button("📥 Export Timeline Data"):
                try:
                    # Generate sample timeline data
                    import numpy as np
                    timeline_data = [
                        {
                            'timestamp': datetime.now() - timedelta(hours=i),
                            'throughput': np.random.uniform(80, 120),
                            'queue_length': np.random.randint(5, 25),
                            'berth_utilization': np.random.uniform(60, 95),
                            'avg_waiting_time': np.random.exponential(2)
                        }
                        for i in range(168)  # 1 week of hourly data
                    ]
                    
                    timeline_df = pd.DataFrame(timeline_data)
                    csv_data = timeline_df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download Timeline CSV",
                        data=csv_data,
                        file_name=f"timeline_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.success("Timeline data prepared for download!")
                except Exception as e:
                    st.error(f"Error preparing timeline data: {str(e)}")
            
            # Complete data export (JSON)
            if st.button("📥 Export All Data (JSON)"):
                try:
                    # Compile all available data
                    all_data = {
                        'export_timestamp': datetime.now().isoformat(),
                        'scenario': scenario_data if scenario_data else 'sample_scenario',
                        'berth_data': [],
                        'queue_data': [],
                        'timeline_data': [],
                        'performance_metrics': {
                            'avg_throughput': 95.5,
                            'avg_waiting_time': 2.3,
                            'berth_utilization': 78.2,
                            'port_efficiency': 85.7
                        }
                    }
                    
                    import json
                    json_data = json.dumps(all_data, indent=2, default=str)
                    
                    st.download_button(
                        label="💾 Download Complete Data JSON",
                        data=json_data,
                        file_name=f"complete_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    st.success("Complete dataset prepared for download!")
                except Exception as e:
                    st.error(f"Error preparing complete data: {str(e)}")
    
    def _render_throughput_analysis(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Render throughput analysis charts."""
        st.subheader("📈 Throughput Timeline Analysis")
        
        # Generate sample throughput data
        import numpy as np
        hours = list(range(24))
        throughput_data = [np.random.uniform(80, 120) for _ in hours]
        
        # Create throughput timeline
        throughput_df = pd.DataFrame({
            'Hour': hours,
            'Throughput (TEU/hr)': throughput_data,
            'Target': [100] * 24  # Target throughput line
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
        
        # Generate sample waiting time data
        import numpy as np
        waiting_times = np.random.exponential(2, 1000)  # Exponential distribution
        
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
        
        # Performance metrics overview
        import numpy as np
        
        # Generate sample KPI data
        kpis = {
            'Port Efficiency': np.random.uniform(80, 95),
            'Berth Utilization': np.random.uniform(70, 90),
            'Average Turnaround': np.random.uniform(12, 20),
            'Customer Satisfaction': np.random.uniform(85, 98),
            'Cost Efficiency': np.random.uniform(75, 92),
            'Environmental Score': np.random.uniform(70, 85)
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
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(kpis.values()),
            theta=list(kpis.keys()),
            fill='toself',
            name='Current Performance',
            line_color='blue'
        ))
        
        # Add target performance line
        target_values = [90] * len(kpis)
        fig.add_trace(go.Scatterpolar(
            r=target_values,
            theta=list(kpis.keys()),
            fill='toself',
            name='Target Performance',
            line_color='red',
            opacity=0.3
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Performance vs Target Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
            # - Performance metrics
            
    def render_cargo_analysis_section(self, scenario_data: Optional[Dict[str, Any]] = None) -> None:
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
        
        # Generate sample transport mode data
        import numpy as np
        import plotly.express as px
        
        transport_modes = ['Truck', 'Rail', 'Barge', 'Pipeline']
        mode_data = {
            'Mode': transport_modes,
            'Volume_TEU': [np.random.randint(100000, 500000) for _ in transport_modes],
            'Efficiency': [np.random.uniform(70, 95) for _ in transport_modes],
            'Cost_per_TEU': [np.random.uniform(50, 200) for _ in transport_modes]
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
        
        # Generate sample time series data
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        
        # Create 12 months of data
        dates = [datetime.now() - timedelta(days=30*i) for i in range(12, 0, -1)]
        monthly_data = {
            'Date': dates,
            'TEU_Volume': [np.random.randint(80000, 120000) for _ in dates],
            'Shipments': [np.random.randint(4000, 7000) for _ in dates],
            'Avg_Dwell_Time': [np.random.uniform(2, 6) for _ in dates]
        }
        
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
        
        # Generate sample forecasting data
        import numpy as np
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        
        # Historical data (12 months)
        hist_dates = [datetime.now() - timedelta(days=30*i) for i in range(12, 0, -1)]
        hist_volumes = [np.random.randint(80000, 120000) for _ in hist_dates]
        
        # Forecast data (6 months ahead)
        forecast_dates = [datetime.now() + timedelta(days=30*i) for i in range(1, 7)]
        # Add trend and seasonality to forecast
        base_forecast = hist_volumes[-1]
        forecast_volumes = []
        for i, _ in enumerate(forecast_dates):
            trend = base_forecast * (1 + 0.02 * i)  # 2% monthly growth
            seasonal = np.sin(i * np.pi / 6) * 5000  # Seasonal variation
            noise = np.random.normal(0, 3000)  # Random variation
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
        
        # Forecast metrics
        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
        
        with forecast_col1:
            avg_forecast = np.mean(forecast_volumes)
            st.metric("Avg Forecast Volume", f"{avg_forecast:,.0f} TEU")
        
        with forecast_col2:
            growth_rate = ((forecast_volumes[-1] / hist_volumes[-1]) - 1) * 100
            st.metric("6-Month Growth", f"{growth_rate:.1f}%")
        
        with forecast_col3:
            confidence = 85  # Simulated confidence level
            st.metric("Forecast Confidence", f"{confidence}%")
    
    def _render_cargo_types_analysis(self) -> None:
        """Render cargo types analysis."""
        st.subheader("📦 Cargo Type Analysis")
        
        # Generate sample cargo type data
        import numpy as np
        import plotly.express as px
        
        cargo_types = ['Containers', 'Bulk Dry', 'Bulk Liquid', 'Break Bulk', 'RoRo']
        cargo_data = {
            'Cargo_Type': cargo_types,
            'Volume_TEU': [np.random.randint(50000, 300000) for _ in cargo_types],
            'Revenue_HKD': [np.random.randint(10000000, 80000000) for _ in cargo_types],
            'Handling_Time': [np.random.uniform(2, 12) for _ in cargo_types]
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
    
    def _render_locations_analysis(self) -> None:
        """Render locations analysis."""
        st.subheader("🌍 Geographic Analysis")
        
        # Generate sample location data
        import numpy as np
        import plotly.express as px
        
        locations = ['Mainland China', 'Southeast Asia', 'Europe', 'North America', 'Other Asia']
        location_data = {
            'Region': locations,
            'Import_TEU': [np.random.randint(50000, 200000) for _ in locations],
            'Export_TEU': [np.random.randint(40000, 180000) for _ in locations],
            'Trade_Balance': [np.random.uniform(-50000, 50000) for _ in locations]
        }
        
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
                    
                    # Store optimization results
                    st.session_state.optimization_results = {
                        'objective_value': 0.85,
                        'optimal_berths': 18,
                        'optimal_cranes': 36,
                        'scenario_performance': {
                            'normal': 0.92,
                            'peak_season': 0.78,
                            'maintenance': 0.85,
                            'typhoon_season': 0.71
                        }
                    }
                    st.success("Optimization completed!")
        
        with opt_col2:
            st.subheader("📊 Optimization Results")
            
            if hasattr(st.session_state, 'optimization_results'):
                results = st.session_state.optimization_results
                
                # Display key metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Objective Value", f"{results['objective_value']:.2f}")
                with metric_col2:
                    st.metric("Optimal Berths", results['optimal_berths'])
                with metric_col3:
                    st.metric("Optimal Cranes", results['optimal_cranes'])
                
                # Scenario performance chart
                import plotly.express as px
                perf_data = pd.DataFrame([
                    {'Scenario': k.replace('_', ' ').title(), 'Performance': v}
                    for k, v in results['scenario_performance'].items()
                ])
                
                fig = px.bar(
                    perf_data,
                    x='Scenario',
                    y='Performance',
                    title='Performance by Scenario',
                    color='Performance',
                    color_continuous_scale='RdYlGn'
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
                    time.sleep(2)
                    
                    # Store simulation results
                    st.session_state.disruption_results = {
                        'impact_score': severity * 0.1,
                        'affected_throughput': max(0, 100 - severity * 10),
                        'recovery_time': duration * (1.5 if not enable_backup else 1.0),
                        'financial_impact': severity * duration * 50000,
                        'timeline': [
                            {'hour': i, 'throughput': max(0, 100 - severity * 10 + i * 2)}
                            for i in range(0, min(duration, 48), 4)
                        ]
                    }
                    st.success("Disruption simulation completed!")
        
        with disrupt_col2:
            st.subheader("📊 Impact Analysis")
            
            if hasattr(st.session_state, 'disruption_results'):
                results = st.session_state.disruption_results
                
                # Impact metrics
                impact_col1, impact_col2 = st.columns(2)
                with impact_col1:
                    st.metric("Impact Score", f"{results['impact_score']:.1f}/10")
                    st.metric("Throughput Reduction", f"{100 - results['affected_throughput']:.0f}%")
                with impact_col2:
                    st.metric("Recovery Time", f"{results['recovery_time']:.1f} hrs")
                    st.metric("Financial Impact", f"${results['financial_impact']:,.0f}")
                
                # Recovery timeline
                if results['timeline']:
                    timeline_df = pd.DataFrame(results['timeline'])
                    import plotly.express as px
                    
                    fig = px.line(
                        timeline_df,
                        x='hour',
                        y='throughput',
                        title='Throughput Recovery Timeline',
                        labels={'hour': 'Hours After Disruption', 'throughput': 'Throughput %'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
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
                    
                    # Calculate investment metrics
                    total_investment = (new_berths * berth_cost + 
                                      new_cranes * crane_cost + 
                                      automation_cost * automation_level / 100)
                    
                    # Simulate ROI calculation
                    annual_revenue_increase = total_investment * 0.15  # 15% annual return assumption
                    payback_period = total_investment / annual_revenue_increase if annual_revenue_increase > 0 else float('inf')
                    
                    st.session_state.investment_results = {
                        'total_investment': total_investment,
                        'annual_revenue_increase': annual_revenue_increase,
                        'payback_period': payback_period,
                        'npv': annual_revenue_increase * analysis_period - total_investment,
                        'capacity_increase': (new_berths * 10 + new_cranes * 5) * (1 + automation_level / 200),
                        'yearly_projections': [
                            {
                                'year': i,
                                'revenue': annual_revenue_increase * (1 + growth_rate/100) ** i,
                                'cumulative_profit': annual_revenue_increase * i - total_investment
                            }
                            for i in range(1, analysis_period + 1)
                        ]
                    }
                    st.success("Investment analysis completed!")
        
        with invest_col2:
            st.subheader("📊 Investment Analysis")
            
            if hasattr(st.session_state, 'investment_results'):
                results = st.session_state.investment_results
                
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
                
                # Investment recommendation
                if results['npv'] > 0 and results['payback_period'] < 7:
                    st.success("✅ **Recommendation**: This investment shows positive returns and reasonable payback period.")
                elif results['npv'] > 0:
                    st.warning("⚠️ **Recommendation**: Positive NPV but long payback period. Consider phased implementation.")
                else:
                    st.error("❌ **Recommendation**: Investment may not be financially viable under current assumptions.")
            else:
                st.info("Configure and analyze an investment scenario to see results here")


# Convenience function for easy integration
def render_consolidated_scenarios_tab(scenario_data: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to render the consolidated scenarios tab.
    
    Args:
        scenario_data: Current scenario configuration and data
    """
    tab = ConsolidatedScenariosTab()
    tab.render_consolidated_tab(scenario_data)