"""Historical Data Parameter Extractor

This module extracts scenario-specific parameters from historical data analysis.
It integrates with the existing data_loader.py to derive operational parameters
for different scenarios based on seasonal patterns and historical trends.
"""

import logging
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

try:
    from hk_port_digital_twin.src.utils.data_loader import (
        get_time_series_data,
        forecast_cargo_throughput,
        get_enhanced_cargo_analysis,
        _analyze_seasonal_patterns,
        load_focused_cargo_statistics
    )
except ImportError:
    logging.warning("Could not import data_loader functions. Historical extraction will use default values.")
    get_time_series_data = None
    forecast_cargo_throughput = None
    get_enhanced_cargo_analysis = None
    _analyze_seasonal_patterns = None
    load_focused_cargo_statistics = None

from .scenario_parameters import ScenarioParameters, ALL_SCENARIOS

logger = logging.getLogger(__name__)

class HistoricalParameterExtractor:
    """Extracts scenario parameters from historical data analysis"""
    
    def __init__(self):
        """Initialize the historical parameter extractor"""
        self.seasonal_patterns = None
        self.cargo_forecasts = None
        self.enhanced_analysis = None
        self._data_loaded = False
        
    def load_historical_data(self) -> bool:
        """Load and analyze historical data
        
        Returns:
            True if data was loaded successfully, False otherwise
        """
        try:
            if not get_time_series_data or not load_focused_cargo_statistics:
                logger.warning("Data loader functions not available. Using predefined parameters.")
                return False
                
            # Load cargo statistics first
            cargo_stats = load_focused_cargo_statistics()
            if not cargo_stats:
                logger.warning("No cargo statistics data available")
                return False
                
            # Get time series data with cargo_stats parameter
            time_series_data = get_time_series_data(cargo_stats)
            if not time_series_data:
                logger.warning("No time series data available")
                return False
                
            # Analyze seasonal patterns - use shipment_types data if available
            if 'shipment_types' in time_series_data and not time_series_data['shipment_types'].empty:
                # Convert to format expected by _analyze_seasonal_patterns
                shipment_df = time_series_data['shipment_types']
                # Create a combined DataFrame with datetime index for seasonal analysis
                combined_data = pd.DataFrame()
                combined_data['total_teus'] = shipment_df.sum(axis=1)
                if 'Direct shipment cargo' in shipment_df.columns:
                    combined_data['seaborne_teus'] = shipment_df['Direct shipment cargo']
                if 'Transhipment cargo' in shipment_df.columns:
                    combined_data['river_teus'] = shipment_df['Transhipment cargo']
                
                # Convert year index to datetime
                combined_data.index = pd.to_datetime(combined_data.index, format='%Y')
                
                self.seasonal_patterns = _analyze_seasonal_patterns(combined_data)
            else:
                logger.warning("No suitable time series data for seasonal analysis")
                self.seasonal_patterns = {}
            
            # Get cargo forecasts
            self.cargo_forecasts = forecast_cargo_throughput(time_series_data, forecast_years=2)
            
            # Get enhanced cargo analysis
            self.enhanced_analysis = get_enhanced_cargo_analysis()
            
            self._data_loaded = True
            logger.info("Historical data loaded and analyzed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False
            
    def extract_scenario_parameters(self, scenario_name: str) -> Optional[ScenarioParameters]:
        """Extract scenario parameters from historical data
        
        Args:
            scenario_name: Name of the scenario to extract parameters for
            
        Returns:
            ScenarioParameters object with extracted parameters, or None if extraction fails
        """
        if not self._data_loaded:
            if not self.load_historical_data():
                logger.warning(f"Using predefined parameters for scenario: {scenario_name}")
                # Map scenario names to ALL_SCENARIOS keys
                scenario_mapping = {
                    "Peak Season": "peak",
                    "Normal Operations": "normal", 
                    "Low Season": "low"
                }
                scenario_key = scenario_mapping.get(scenario_name)
                return ALL_SCENARIOS.get(scenario_key) if scenario_key else None
                
        try:
            if scenario_name == "Peak Season":
                return self._extract_peak_season_parameters()
            elif scenario_name == "Normal Operations":
                return self._extract_normal_operations_parameters()
            elif scenario_name == "Low Season":
                return self._extract_low_season_parameters()
            else:
                logger.warning(f"Unknown scenario: {scenario_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting parameters for {scenario_name}: {e}")
            # Map scenario names to ALL_SCENARIOS keys
            scenario_mapping = {
                "Peak Season": "peak",
                "Normal Operations": "normal", 
                "Low Season": "low"
            }
            scenario_key = scenario_mapping.get(scenario_name)
            return ALL_SCENARIOS.get(scenario_key) if scenario_key else None
            
    def _extract_peak_season_parameters(self) -> ScenarioParameters:
        """Extract parameters for peak season scenario"""
        base_params = ALL_SCENARIOS["peak"]
        
        if not self.seasonal_patterns:
            return base_params
            
        try:
            # Get peak months from seasonal analysis
            monthly_patterns = self.seasonal_patterns.get('monthly_patterns', {})
            peak_months = self._identify_peak_months(monthly_patterns)
            
            # Calculate peak season multipliers
            peak_multiplier = self._calculate_peak_multiplier(monthly_patterns, peak_months)
            
            # Update parameters based on historical data
            updated_params = ScenarioParameters(
                name="Peak Season",
                description="High-volume operations during peak shipping season",
                ship_arrival_rate_multiplier=min(peak_multiplier, 2.0),  # Cap at 2x
                ship_type_distribution=self._adjust_ship_distribution_for_peak(),
                container_volume_multiplier=min(peak_multiplier * 0.9, 1.8),
                operational_efficiency_factor=max(0.85, 1.0 - (peak_multiplier - 1.0) * 0.3),
                berth_utilization_target=min(0.95, 0.8 + (peak_multiplier - 1.0) * 0.15),
                priority_factors=base_params.priority_factors,
                seasonal_months=peak_months,
                weather_impact_factor=base_params.weather_impact_factor,
                crane_efficiency_factor=max(0.9, 1.0 - (peak_multiplier - 1.0) * 0.1)
            )
            
            logger.info(f"Extracted peak season parameters with {peak_multiplier:.2f}x multiplier")
            return updated_params
            
        except Exception as e:
            logger.error(f"Error extracting peak season parameters: {e}")
            return base_params
            
    def _extract_normal_operations_parameters(self) -> ScenarioParameters:
        """Extract parameters for normal operations scenario"""
        base_params = ALL_SCENARIOS["normal"]
        
        if not self.seasonal_patterns:
            return base_params
            
        try:
            # Use median values from historical data
            monthly_patterns = self.seasonal_patterns.get('monthly_patterns', {})
            normal_months = self._identify_normal_months(monthly_patterns)
            
            # Calculate normal operation baseline
            normal_baseline = self._calculate_normal_baseline(monthly_patterns)
            
            updated_params = ScenarioParameters(
                name="Normal Operations",
                description="Standard port operations during regular periods",
                ship_arrival_rate_multiplier=normal_baseline,
                ship_type_distribution=self._get_balanced_ship_distribution(),
                container_volume_multiplier=normal_baseline,
                operational_efficiency_factor=0.92,
                berth_utilization_target=0.75,
                priority_factors=base_params.priority_factors,
                seasonal_months=normal_months,
                weather_impact_factor=base_params.weather_impact_factor,
                crane_efficiency_factor=0.95
            )
            
            logger.info(f"Extracted normal operations parameters with {normal_baseline:.2f}x baseline")
            return updated_params
            
        except Exception as e:
            logger.error(f"Error extracting normal operations parameters: {e}")
            return base_params
            
    def _extract_low_season_parameters(self) -> ScenarioParameters:
        """Extract parameters for low season scenario"""
        base_params = ALL_SCENARIOS["low"]
        
        if not self.seasonal_patterns:
            return base_params
            
        try:
            # Get low season months from seasonal analysis
            monthly_patterns = self.seasonal_patterns.get('monthly_patterns', {})
            low_months = self._identify_low_months(monthly_patterns)
            
            # Calculate low season multipliers
            low_multiplier = self._calculate_low_multiplier(monthly_patterns, low_months)
            
            updated_params = ScenarioParameters(
                name="Low Season",
                description="Reduced operations during low shipping season",
                ship_arrival_rate_multiplier=max(low_multiplier, 0.4),  # Floor at 0.4x
                ship_type_distribution=self._adjust_ship_distribution_for_low(),
                container_volume_multiplier=max(low_multiplier * 1.1, 0.5),
                operational_efficiency_factor=min(0.98, 0.92 + (1.0 - low_multiplier) * 0.1),
                berth_utilization_target=max(0.45, 0.75 - (1.0 - low_multiplier) * 0.3),
                priority_factors=base_params.priority_factors,
                seasonal_months=low_months,
                weather_impact_factor=base_params.weather_impact_factor,
                crane_efficiency_factor=min(0.98, 0.95 + (1.0 - low_multiplier) * 0.05)
            )
            
            logger.info(f"Extracted low season parameters with {low_multiplier:.2f}x multiplier")
            return updated_params
            
        except Exception as e:
            logger.error(f"Error extracting low season parameters: {e}")
            return base_params
            
    def _identify_peak_months(self, monthly_patterns: Dict) -> List[int]:
        """Identify peak months from seasonal patterns"""
        if not monthly_patterns:
            return [6, 7, 8, 9, 10, 11]  # Default peak months
            
        # Find months with highest total TEU values
        month_values = []
        for month in range(1, 13):
            total_teu = monthly_patterns.get(month, {}).get('total_teu', 0)
            month_values.append((month, total_teu))
            
        # Sort by TEU values and take top 6 months
        month_values.sort(key=lambda x: x[1], reverse=True)
        peak_months = [month for month, _ in month_values[:6]]
        
        return sorted(peak_months)
        
    def _identify_normal_months(self, monthly_patterns: Dict) -> List[int]:
        """Identify normal operation months from seasonal patterns"""
        if not monthly_patterns:
            return [3, 4, 5, 9, 10]  # Default normal months
            
        # Find months with median TEU values
        month_values = []
        for month in range(1, 13):
            total_teu = monthly_patterns.get(month, {}).get('total_teu', 0)
            month_values.append((month, total_teu))
            
        # Sort by TEU values and take middle months
        month_values.sort(key=lambda x: x[1])
        normal_months = [month for month, _ in month_values[3:8]]  # Middle 5 months
        
        return sorted(normal_months)
        
    def _identify_low_months(self, monthly_patterns: Dict) -> List[int]:
        """Identify low season months from seasonal patterns"""
        if not monthly_patterns:
            return [1, 2, 12]  # Default low months
            
        # Find months with lowest TEU values
        month_values = []
        for month in range(1, 13):
            total_teu = monthly_patterns.get(month, {}).get('total_teu', 0)
            month_values.append((month, total_teu))
            
        # Sort by TEU values and take bottom 3 months
        month_values.sort(key=lambda x: x[1])
        low_months = [month for month, _ in month_values[:3]]
        
        return sorted(low_months)
        
    def _calculate_peak_multiplier(self, monthly_patterns: Dict, peak_months: List[int]) -> float:
        """Calculate peak season multiplier from historical data"""
        if not monthly_patterns or not peak_months:
            return 1.5  # Default multiplier
            
        peak_values = []
        all_values = []
        
        for month in range(1, 13):
            total_teu = monthly_patterns.get(month, {}).get('total_teu', 0)
            all_values.append(total_teu)
            if month in peak_months:
                peak_values.append(total_teu)
                
        if not peak_values or not all_values:
            return 1.5
            
        avg_peak = sum(peak_values) / len(peak_values)
        avg_all = sum(all_values) / len(all_values)
        
        return avg_peak / avg_all if avg_all > 0 else 1.5
        
    def _calculate_normal_baseline(self, monthly_patterns: Dict) -> float:
        """Calculate normal operations baseline from historical data"""
        if not monthly_patterns:
            return 1.0
            
        # Use median of all monthly values as baseline
        all_values = []
        for month in range(1, 13):
            total_teu = monthly_patterns.get(month, {}).get('total_teu', 0)
            all_values.append(total_teu)
            
        if not all_values:
            return 1.0
            
        all_values.sort()
        median_value = all_values[len(all_values) // 2]
        avg_value = sum(all_values) / len(all_values)
        
        return median_value / avg_value if avg_value > 0 else 1.0
        
    def _calculate_low_multiplier(self, monthly_patterns: Dict, low_months: List[int]) -> float:
        """Calculate low season multiplier from historical data"""
        if not monthly_patterns or not low_months:
            return 0.7  # Default multiplier
            
        low_values = []
        all_values = []
        
        for month in range(1, 13):
            total_teu = monthly_patterns.get(month, {}).get('total_teu', 0)
            all_values.append(total_teu)
            if month in low_months:
                low_values.append(total_teu)
                
        if not low_values or not all_values:
            return 0.7
            
        avg_low = sum(low_values) / len(low_values)
        avg_all = sum(all_values) / len(all_values)
        
        return avg_low / avg_all if avg_all > 0 else 0.7
        
    def _adjust_ship_distribution_for_peak(self) -> Dict[str, float]:
        """Adjust ship type distribution for peak season"""
        return {
            'container': 0.75,  # More container ships during peak
            'bulk': 0.15,
            'mixed': 0.10
        }
        
    def _adjust_ship_distribution_for_low(self) -> Dict[str, float]:
        """Adjust ship type distribution for low season"""
        return {
            'container': 0.60,  # Fewer container ships during low season
            'bulk': 0.25,
            'mixed': 0.15
        }
        
    def _get_balanced_ship_distribution(self) -> Dict[str, float]:
        """Get balanced ship type distribution for normal operations"""
        return {
            'container': 0.65,
            'bulk': 0.20,
            'mixed': 0.15
        }
        
    def get_extraction_summary(self) -> Dict:
        """Get summary of parameter extraction process
        
        Returns:
            Dictionary containing extraction summary and statistics
        """
        return {
            'data_loaded': self._data_loaded,
            'seasonal_patterns_available': self.seasonal_patterns is not None,
            'cargo_forecasts_available': self.cargo_forecasts is not None,
            'enhanced_analysis_available': self.enhanced_analysis is not None,
            'extraction_timestamp': datetime.now().isoformat()
        }