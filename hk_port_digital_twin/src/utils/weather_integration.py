# Comments for context:
# This module integrates weather data from Hong Kong Observatory to enhance port operations simulation.
# Weather conditions significantly impact port operations - high winds can prevent crane operations,
# poor visibility affects vessel navigation, and typhoons can shut down operations entirely.
# This integration provides real-time weather data and historical patterns to improve simulation accuracy.

import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass
import os
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class WeatherCondition:
    """Represents current weather conditions affecting port operations."""
    timestamp: datetime
    temperature: float  # Celsius
    humidity: float  # Percentage
    wind_speed: float  # km/h
    wind_direction: str  # Cardinal direction
    visibility: float  # km
    weather_description: str
    impact_score: float  # 0.0 (no impact) to 1.0 (severe impact)
    operational_status: str  # 'normal', 'restricted', 'suspended'

class HKObservatoryIntegration:
    """Integration with Hong Kong Observatory weather data.
    
    This class handles fetching weather data from HK Observatory APIs and
    converting it into operational impact assessments for port simulation.
    """
    
    def __init__(self):
        # HK Observatory API endpoints (using publicly available data)
        self.base_url = "https://data.weather.gov.hk/weatherAPI/opendata"
        self.current_weather_endpoint = f"{self.base_url}/weather.php?dataType=rhrread&lang=en"
        self.forecast_endpoint = f"{self.base_url}/weather.php?dataType=fnd&lang=en"
        self.warning_endpoint = f"{self.base_url}/weather.php?dataType=warnsum&lang=en"
        
        # Cache directory for offline mode
        self.cache_dir = Path("data/weather_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Operational thresholds (based on Hong Kong port operational guidelines)
        self.operational_thresholds = {
            'wind_speed': {
                'normal': 30,      # km/h - normal operations
                'restricted': 50,  # km/h - restricted crane operations
                'suspended': 80    # km/h - operations suspended
            },
            'visibility': {
                'normal': 5.0,     # km - normal operations
                'restricted': 1.0, # km - restricted navigation
                'suspended': 0.5   # km - operations suspended
            }
        }
        
        logger.info("HK Observatory weather integration initialized")
    
    def get_current_weather(self) -> Optional[WeatherCondition]:
        """Fetch current weather conditions from HK Observatory.
        
        Returns:
            WeatherCondition object with current conditions, or None if unavailable
        """
        try:
            response = requests.get(self.current_weather_endpoint, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant weather data
            weather_data = self._parse_current_weather(data)
            
            # Calculate operational impact
            impact_score = self._calculate_impact_score(weather_data)
            operational_status = self._determine_operational_status(weather_data)
            
            weather_condition = WeatherCondition(
                timestamp=datetime.now(),
                temperature=weather_data.get('temperature', 25.0),
                humidity=weather_data.get('humidity', 70.0),
                wind_speed=weather_data.get('wind_speed', 10.0),
                wind_direction=weather_data.get('wind_direction', 'N'),
                visibility=weather_data.get('visibility', 10.0),
                weather_description=weather_data.get('description', 'Clear'),
                impact_score=impact_score,
                operational_status=operational_status
            )
            
            # Cache the data
            self._cache_weather_data(weather_condition)
            
            logger.info(f"Current weather retrieved: {operational_status} operations, impact score: {impact_score:.2f}")
            return weather_condition
            
        except Exception as e:
            logger.error(f"Error fetching current weather: {e}")
            # Try to load from cache
            return self._load_cached_weather()
    
    def get_weather_forecast(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get weather forecast for the next few days.
        
        Args:
            days: Number of days to forecast (max 9 for HK Observatory)
            
        Returns:
            List of forecast data dictionaries
        """
        try:
            response = requests.get(self.forecast_endpoint, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            forecast_data = data.get('weatherForecast', [])
            
            # Process forecast data
            processed_forecast = []
            for day_data in forecast_data[:days]:
                processed_day = {
                    'date': day_data.get('forecastDate'),
                    'max_temp': day_data.get('forecastMaxtemp', {}).get('value'),
                    'min_temp': day_data.get('forecastMintemp', {}).get('value'),
                    'humidity': day_data.get('forecastMaxrh', {}).get('value'),
                    'weather': day_data.get('forecastWeather'),
                    'wind': day_data.get('forecastWind'),
                    'impact_assessment': self._assess_forecast_impact(day_data)
                }
                processed_forecast.append(processed_day)
            
            logger.info(f"Weather forecast retrieved for {len(processed_forecast)} days")
            return processed_forecast
            
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {e}")
            return []
    
    def get_forecast(self) -> List[Dict[str, Any]]:
        """Get weather forecast data.
        
        Returns:
            List of forecast data dictionaries
        """
        try:
            # For now, return current weather as a single-item forecast
            # In a real implementation, this would fetch actual forecast data
            current = self.get_current_weather()
            if current:
                forecast_data = {
                    'timestamp': current.timestamp.isoformat(),
                    'temperature': current.temperature,
                    'humidity': current.humidity,
                    'wind_speed': current.wind_speed,
                    'visibility': current.visibility,
                    'operational_status': current.operational_status
                }
                return [forecast_data]
            return []
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {e}")
            return []
    
    def get_weather_warnings(self) -> List[Dict[str, Any]]:
        """Get current weather warnings that might affect port operations.
        
        Returns:
            List of active weather warnings
        """
        try:
            response = requests.get(self.warning_endpoint, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            warnings = []
            
            # Process different types of warnings
            for warning_type in ['WTCSGNL', 'WRAIN', 'WFROST', 'WFNTSA', 'WL', 'WTMW', 'WTS']:
                if warning_type in data and data[warning_type]:
                    warning_info = {
                        'type': warning_type,
                        'details': data[warning_type],
                        'port_impact': self._assess_warning_impact(warning_type, data[warning_type])
                    }
                    warnings.append(warning_info)
            
            if warnings:
                logger.warning(f"Active weather warnings: {len(warnings)} warnings affecting port operations")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error fetching weather warnings: {e}")
            return []
    
    def _parse_current_weather(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse HK Observatory current weather data."""
        try:
            # Extract temperature
            temperature = None
            if 'temperature' in data and data['temperature']:
                temp_data = data['temperature']['data']
                if temp_data:
                    temperature = float(temp_data[0]['value'])
            
            # Extract humidity
            humidity = None
            if 'humidity' in data and data['humidity']:
                humidity_data = data['humidity']['data']
                if humidity_data:
                    humidity = float(humidity_data[0]['value'])
            
            # Extract wind data
            wind_speed = 10.0  # Default
            wind_direction = 'N'  # Default
            if 'wind' in data and data['wind']:
                wind_data = data['wind']['data']
                if wind_data:
                    wind_speed = float(wind_data[0].get('speed', 10.0))
                    wind_direction = wind_data[0].get('direction', 'N')
            
            # Extract visibility
            visibility = 10.0  # Default good visibility
            if 'visibility' in data and data['visibility']:
                vis_data = data['visibility']['data']
                if vis_data:
                    visibility = float(vis_data[0]['value'])
            
            return {
                'temperature': temperature or 25.0,
                'humidity': humidity or 70.0,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'visibility': visibility,
                'description': 'Current conditions'
            }
            
        except Exception as e:
            logger.error(f"Error parsing weather data: {e}")
            return {
                'temperature': 25.0,
                'humidity': 70.0,
                'wind_speed': 10.0,
                'wind_direction': 'N',
                'visibility': 10.0,
                'description': 'Data unavailable'
            }
    
    def _calculate_impact_score(self, weather_data: Dict[str, Any]) -> float:
        """Calculate operational impact score based on weather conditions.
        
        Returns:
            Float between 0.0 (no impact) and 1.0 (severe impact)
        """
        impact_factors = []
        
        # Wind speed impact
        wind_speed = weather_data.get('wind_speed', 0)
        if wind_speed >= self.operational_thresholds['wind_speed']['suspended']:
            impact_factors.append(1.0)
        elif wind_speed >= self.operational_thresholds['wind_speed']['restricted']:
            impact_factors.append(0.7)
        elif wind_speed >= self.operational_thresholds['wind_speed']['normal']:
            impact_factors.append(0.3)
        else:
            impact_factors.append(0.0)
        
        # Visibility impact
        visibility = weather_data.get('visibility', 10)
        if visibility <= self.operational_thresholds['visibility']['suspended']:
            impact_factors.append(1.0)
        elif visibility <= self.operational_thresholds['visibility']['restricted']:
            impact_factors.append(0.6)
        elif visibility <= self.operational_thresholds['visibility']['normal']:
            impact_factors.append(0.2)
        else:
            impact_factors.append(0.0)
        
        # Return the maximum impact factor
        return max(impact_factors) if impact_factors else 0.0
    
    def _determine_operational_status(self, weather_data: Dict[str, Any]) -> str:
        """Determine operational status based on weather conditions."""
        impact_score = self._calculate_impact_score(weather_data)
        
        if impact_score >= 0.8:
            return 'suspended'
        elif impact_score >= 0.4:
            return 'restricted'
        else:
            return 'normal'
    
    def _assess_forecast_impact(self, forecast_data: Dict[str, Any]) -> str:
        """Assess potential operational impact from forecast data."""
        # Simple assessment based on weather description
        weather_desc = forecast_data.get('forecastWeather', '').lower()
        
        if any(term in weather_desc for term in ['typhoon', 'storm', 'severe']):
            return 'high_impact'
        elif any(term in weather_desc for term in ['rain', 'shower', 'wind']):
            return 'moderate_impact'
        else:
            return 'low_impact'
    
    def _assess_warning_impact(self, warning_type: str, warning_data: Any) -> str:
        """Assess port operational impact from weather warnings."""
        # Map warning types to operational impact
        high_impact_warnings = ['WTCSGNL']  # Typhoon signals
        moderate_impact_warnings = ['WRAIN', 'WTS']  # Heavy rain, thunderstorm
        
        if warning_type in high_impact_warnings:
            return 'high_impact'
        elif warning_type in moderate_impact_warnings:
            return 'moderate_impact'
        else:
            return 'low_impact'
    
    def _cache_weather_data(self, weather_condition: WeatherCondition):
        """Cache weather data for offline use."""
        try:
            cache_file = self.cache_dir / "current_weather.json"
            cache_data = {
                'timestamp': weather_condition.timestamp.isoformat(),
                'temperature': weather_condition.temperature,
                'humidity': weather_condition.humidity,
                'wind_speed': weather_condition.wind_speed,
                'wind_direction': weather_condition.wind_direction,
                'visibility': weather_condition.visibility,
                'weather_description': weather_condition.weather_description,
                'impact_score': weather_condition.impact_score,
                'operational_status': weather_condition.operational_status
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error caching weather data: {e}")
    
    def _load_cached_weather(self) -> Optional[WeatherCondition]:
        """Load cached weather data as fallback."""
        try:
            cache_file = self.cache_dir / "current_weather.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is not too old (max 2 hours)
                cached_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cached_time < timedelta(hours=2):
                    logger.info("Using cached weather data")
                    return WeatherCondition(
                        timestamp=cached_time,
                        temperature=cache_data['temperature'],
                        humidity=cache_data['humidity'],
                        wind_speed=cache_data['wind_speed'],
                        wind_direction=cache_data['wind_direction'],
                        visibility=cache_data['visibility'],
                        weather_description=cache_data['weather_description'],
                        impact_score=cache_data['impact_score'],
                        operational_status=cache_data['operational_status']
                    )
            
            # Return default weather if no valid cache
            logger.warning("No valid cached weather data, using default conditions")
            return self._get_default_weather()
            
        except Exception as e:
            logger.error(f"Error loading cached weather: {e}")
            return self._get_default_weather()
    
    def _get_default_weather(self) -> WeatherCondition:
        """Return default weather conditions for fallback."""
        return WeatherCondition(
            timestamp=datetime.now(),
            temperature=25.0,
            humidity=70.0,
            wind_speed=15.0,
            wind_direction='E',
            visibility=10.0,
            weather_description='Default conditions (data unavailable)',
            impact_score=0.1,
            operational_status='normal'
        )

# Utility functions for integration with simulation
def get_weather_impact_for_simulation() -> Dict[str, Any]:
    """Get current weather impact data formatted for simulation use.
    
    Returns:
        Dictionary with weather impact data for simulation
    """
    weather_integration = HKObservatoryIntegration()
    current_weather = weather_integration.get_current_weather()
    
    if current_weather:
        return {
            'impact_score': current_weather.impact_score,
            'operational_status': current_weather.operational_status,
            'wind_speed': current_weather.wind_speed,
            'visibility': current_weather.visibility,
            'description': current_weather.weather_description,
            'timestamp': current_weather.timestamp.isoformat()
        }
    else:
        # Return default impact data
        return {
            'impact_score': 0.1,
            'operational_status': 'normal',
            'wind_speed': 15.0,
            'visibility': 10.0,
            'description': 'Weather data unavailable',
            'timestamp': datetime.now().isoformat()
        }

def simulate_weather_scenario(scenario_type: str) -> Dict[str, Any]:
    """Generate weather conditions for specific scenarios.
    
    Args:
        scenario_type: 'normal', 'typhoon', 'heavy_rain', 'poor_visibility'
        
    Returns:
        Dictionary with simulated weather conditions
    """
    scenarios = {
        'normal': {
            'impact_score': 0.1,
            'operational_status': 'normal',
            'wind_speed': 15.0,
            'visibility': 10.0,
            'description': 'Clear weather, normal operations'
        },
        'typhoon': {
            'impact_score': 0.9,
            'operational_status': 'suspended',
            'wind_speed': 120.0,
            'visibility': 2.0,
            'description': 'Typhoon conditions, operations suspended'
        },
        'heavy_rain': {
            'impact_score': 0.4,
            'operational_status': 'restricted',
            'wind_speed': 35.0,
            'visibility': 3.0,
            'description': 'Heavy rain, restricted operations'
        },
        'poor_visibility': {
            'impact_score': 0.6,
            'operational_status': 'restricted',
            'wind_speed': 20.0,
            'visibility': 0.8,
            'description': 'Poor visibility, navigation restricted'
        }
    }
    
    weather_data = scenarios.get(scenario_type, scenarios['normal'])
    weather_data['timestamp'] = datetime.now().isoformat()
    
    return weather_data