# Comments for context:
# This module implements predictive models for Hong Kong Port Digital Twin.
# The goal is to predict ship arrivals, estimate processing times, and forecast
# queue lengths to enable proactive port management and optimization.
# 
# Approach: Using statistical models and machine learning techniques based on
# historical patterns, seasonal trends, and real-time data.

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ArrivalPrediction:
    """Prediction result for ship arrivals"""
    predicted_time: datetime
    confidence_interval: Tuple[datetime, datetime]
    probability: float
    ship_type: str
    factors: Dict[str, float]  # factors influencing the prediction

@dataclass
class ProcessingTimePrediction:
    """Prediction result for ship processing time"""
    estimated_hours: float
    confidence_interval: Tuple[float, float]
    factors: Dict[str, float]
    uncertainty: float

@dataclass
class QueueForecast:
    """Forecast result for port queue length"""
    timestamp: datetime
    predicted_queue_length: int
    predicted_waiting_time: float
    confidence: float
    trend: str  # 'increasing', 'decreasing', 'stable'

class ShipArrivalPredictor:
    """Predicts ship arrivals based on historical patterns and external factors"""
    
    def __init__(self):
        self.historical_data = pd.DataFrame()
        self.seasonal_patterns = {}
        self.ship_type_patterns = {}
        self.is_trained = False
        
    def load_historical_data(self, vessel_data: pd.DataFrame) -> None:
        """Load historical vessel arrival data for training"""
        try:
            # Ensure required columns exist
            required_columns = ['arrival_time', 'ship_type', 'size']
            if not all(col in vessel_data.columns for col in required_columns):
                logger.warning("Missing required columns for arrival prediction training")
                return
                
            self.historical_data = vessel_data.copy()
            
            # Convert arrival_time to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(self.historical_data['arrival_time']):
                self.historical_data['arrival_time'] = pd.to_datetime(self.historical_data['arrival_time'])
            
            # Extract time features
            self.historical_data['hour'] = self.historical_data['arrival_time'].dt.hour
            self.historical_data['day_of_week'] = self.historical_data['arrival_time'].dt.dayofweek
            self.historical_data['month'] = self.historical_data['arrival_time'].dt.month
            self.historical_data['day_of_year'] = self.historical_data['arrival_time'].dt.dayofyear
            
            logger.info(f"Loaded {len(self.historical_data)} historical arrival records")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def analyze_seasonal_patterns(self) -> None:
        """Analyze seasonal and temporal patterns in ship arrivals"""
        if self.historical_data.empty:
            logger.warning("No historical data available for pattern analysis")
            return
            
        try:
            # Hourly patterns
            hourly_arrivals = self.historical_data.groupby('hour').size()
            self.seasonal_patterns['hourly'] = hourly_arrivals.to_dict()
            
            # Daily patterns (day of week)
            daily_arrivals = self.historical_data.groupby('day_of_week').size()
            self.seasonal_patterns['daily'] = daily_arrivals.to_dict()
            
            # Monthly patterns
            monthly_arrivals = self.historical_data.groupby('month').size()
            self.seasonal_patterns['monthly'] = monthly_arrivals.to_dict()
            
            # Ship type patterns
            for ship_type in self.historical_data['ship_type'].unique():
                type_data = self.historical_data[self.historical_data['ship_type'] == ship_type]
                if len(type_data) > 0:
                    self.ship_type_patterns[ship_type] = {
                        'avg_arrivals_per_day': len(type_data) / max(1, (self.historical_data['arrival_time'].max() - self.historical_data['arrival_time'].min()).days),
                        'preferred_hours': type_data.groupby('hour').size().idxmax() if len(type_data) > 0 else 12,
                        'avg_size': type_data['size'].mean() if 'size' in type_data.columns else 1000
                    }
            
            logger.info("Seasonal pattern analysis completed")
            
        except Exception as e:
            logger.error(f"Error in seasonal pattern analysis: {e}")
    
    def train_arrival_model(self) -> None:
        """Train the arrival prediction model"""
        if self.historical_data.empty:
            logger.warning("No historical data available for training")
            return
            
        try:
            self.analyze_seasonal_patterns()
            self.is_trained = True
            logger.info("Arrival prediction model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training arrival model: {e}")
            self.is_trained = False
    
    def predict_next_arrival(self, ship_type: str = None, current_time: datetime = None) -> ArrivalPrediction:
        """Predict the next ship arrival"""
        if current_time is None:
            current_time = datetime.now()
            
        if not self.is_trained:
            logger.warning("Model not trained, using default prediction")
            # Default prediction: next arrival in 2-6 hours
            predicted_time = current_time + timedelta(hours=np.random.uniform(2, 6))
            return ArrivalPrediction(
                predicted_time=predicted_time,
                confidence_interval=(predicted_time - timedelta(hours=1), predicted_time + timedelta(hours=1)),
                probability=0.5,
                ship_type=ship_type or 'container',
                factors={'default': 1.0}
            )
        
        try:
            # Base arrival rate calculation
            if ship_type and ship_type in self.ship_type_patterns:
                base_rate = self.ship_type_patterns[ship_type]['avg_arrivals_per_day']
            else:
                # Overall average arrival rate
                total_days = max(1, (self.historical_data['arrival_time'].max() - self.historical_data['arrival_time'].min()).days)
                base_rate = len(self.historical_data) / total_days
            
            # Adjust for time of day
            current_hour = current_time.hour
            hourly_factor = self.seasonal_patterns.get('hourly', {}).get(current_hour, 1) / max(1, np.mean(list(self.seasonal_patterns.get('hourly', {1: 1}).values())))
            
            # Adjust for day of week
            current_dow = current_time.weekday()
            daily_factor = self.seasonal_patterns.get('daily', {}).get(current_dow, 1) / max(1, np.mean(list(self.seasonal_patterns.get('daily', {1: 1}).values())))
            
            # Adjust for month
            current_month = current_time.month
            monthly_factor = self.seasonal_patterns.get('monthly', {}).get(current_month, 1) / max(1, np.mean(list(self.seasonal_patterns.get('monthly', {1: 1}).values())))
            
            # Combined adjustment factor
            adjustment_factor = (hourly_factor + daily_factor + monthly_factor) / 3
            adjusted_rate = base_rate * adjustment_factor
            
            # Calculate time until next arrival (exponential distribution)
            if adjusted_rate > 0:
                hours_until_arrival = np.random.exponential(24 / adjusted_rate)
            else:
                hours_until_arrival = 4.0  # default 4 hours
            
            predicted_time = current_time + timedelta(hours=hours_until_arrival)
            
            # Calculate confidence interval (±25% of predicted time)
            uncertainty = hours_until_arrival * 0.25
            confidence_interval = (
                predicted_time - timedelta(hours=uncertainty),
                predicted_time + timedelta(hours=uncertainty)
            )
            
            # Calculate probability based on model confidence
            probability = min(0.9, 0.5 + (len(self.historical_data) / 1000) * 0.4)
            
            factors = {
                'base_rate': base_rate,
                'hourly_factor': hourly_factor,
                'daily_factor': daily_factor,
                'monthly_factor': monthly_factor,
                'adjustment_factor': adjustment_factor
            }
            
            return ArrivalPrediction(
                predicted_time=predicted_time,
                confidence_interval=confidence_interval,
                probability=probability,
                ship_type=ship_type or 'container',
                factors=factors
            )
            
        except Exception as e:
            logger.error(f"Error in arrival prediction: {e}")
            # Fallback prediction
            predicted_time = current_time + timedelta(hours=4)
            return ArrivalPrediction(
                predicted_time=predicted_time,
                confidence_interval=(predicted_time - timedelta(hours=2), predicted_time + timedelta(hours=2)),
                probability=0.3,
                ship_type=ship_type or 'container',
                factors={'error_fallback': 1.0}
            )

class ProcessingTimeEstimator:
    """Estimates ship processing times based on ship characteristics and historical data"""
    
    def __init__(self):
        self.historical_processing_times = {}
        self.size_factors = {}
        self.type_factors = {
            'container': 1.0,
            'bulk': 1.5,
            'tanker': 1.3,
            'general': 1.2,
            'passenger': 0.8,
            'ro-ro': 1.1
        }
        self.is_trained = False
    
    def load_processing_data(self, processing_data: pd.DataFrame) -> None:
        """Load historical processing time data"""
        try:
            if 'ship_type' in processing_data.columns and 'processing_time' in processing_data.columns:
                # Group by ship type and calculate statistics
                for ship_type in processing_data['ship_type'].unique():
                    type_data = processing_data[processing_data['ship_type'] == ship_type]['processing_time']
                    self.historical_processing_times[ship_type] = {
                        'mean': type_data.mean(),
                        'std': type_data.std(),
                        'median': type_data.median(),
                        'count': len(type_data)
                    }
                
                # Analyze size factors if size data is available
                if 'size' in processing_data.columns:
                    # Create size bins and analyze processing time correlation
                    processing_data['size_bin'] = pd.cut(processing_data['size'], bins=5, labels=['XS', 'S', 'M', 'L', 'XL'])
                    size_analysis = processing_data.groupby('size_bin')['processing_time'].mean()
                    
                    # Normalize to medium size (M) as baseline
                    baseline = size_analysis.get('M', size_analysis.mean())
                    for size_bin, avg_time in size_analysis.items():
                        self.size_factors[size_bin] = avg_time / baseline
                
                self.is_trained = True
                logger.info(f"Processing time model trained with {len(processing_data)} records")
                
        except Exception as e:
            logger.error(f"Error loading processing data: {e}")
    
    def estimate_processing_time(self, ship_type: str, ship_size: float = None, 
                               containers: int = 0, cargo_volume: float = 0) -> ProcessingTimePrediction:
        """Estimate processing time for a ship"""
        try:
            # Base processing time
            if self.is_trained and ship_type in self.historical_processing_times:
                base_time = self.historical_processing_times[ship_type]['mean']
                std_dev = self.historical_processing_times[ship_type]['std']
            else:
                # Default base times by ship type (hours)
                default_times = {
                    'container': 6.0,
                    'bulk': 12.0,
                    'tanker': 8.0,
                    'general': 4.0,
                    'passenger': 2.0,
                    'ro-ro': 3.0
                }
                base_time = default_times.get(ship_type, 6.0)
                std_dev = base_time * 0.3  # 30% standard deviation
            
            # Apply ship type factor
            type_factor = self.type_factors.get(ship_type, 1.0)
            
            # Apply size factor
            size_factor = 1.0
            if ship_size:
                if ship_size < 1000:
                    size_factor = 0.7
                elif ship_size < 3000:
                    size_factor = 1.0
                elif ship_size < 6000:
                    size_factor = 1.3
                else:
                    size_factor = 1.6
            
            # Container handling factor
            container_factor = 1.0
            if containers > 0:
                # Assume 30 containers per hour base rate
                container_time = containers / 30
                container_factor = 1.0 + (container_time / base_time)
            
            # Cargo volume factor
            cargo_factor = 1.0
            if cargo_volume > 0:
                # Assume 100 tons per hour base rate
                cargo_time = cargo_volume / 100
                cargo_factor = 1.0 + (cargo_time / base_time)
            
            # Calculate final estimate
            estimated_time = base_time * type_factor * size_factor * max(container_factor, cargo_factor)
            
            # Calculate uncertainty
            uncertainty = std_dev * np.sqrt(type_factor * size_factor)
            
            # Confidence interval
            confidence_interval = (
                max(0.5, estimated_time - uncertainty),
                estimated_time + uncertainty
            )
            
            factors = {
                'base_time': base_time,
                'type_factor': type_factor,
                'size_factor': size_factor,
                'container_factor': container_factor,
                'cargo_factor': cargo_factor
            }
            
            return ProcessingTimePrediction(
                estimated_hours=estimated_time,
                confidence_interval=confidence_interval,
                factors=factors,
                uncertainty=uncertainty / estimated_time if estimated_time > 0 else 0.3
            )
            
        except Exception as e:
            logger.error(f"Error estimating processing time: {e}")
            # Fallback estimate
            return ProcessingTimePrediction(
                estimated_hours=6.0,
                confidence_interval=(4.0, 8.0),
                factors={'fallback': 1.0},
                uncertainty=0.3
            )

class QueueLengthForecaster:
    """Forecasts port queue length and waiting times"""
    
    def __init__(self):
        self.historical_queue_data = pd.DataFrame()
        self.arrival_predictor = ShipArrivalPredictor()
        self.processing_estimator = ProcessingTimeEstimator()
        self.current_queue = []
        
    def load_queue_history(self, queue_data: pd.DataFrame) -> None:
        """Load historical queue length data"""
        try:
            self.historical_queue_data = queue_data.copy()
            if 'timestamp' in queue_data.columns:
                self.historical_queue_data['timestamp'] = pd.to_datetime(self.historical_queue_data['timestamp'])
                self.historical_queue_data['hour'] = self.historical_queue_data['timestamp'].dt.hour
                self.historical_queue_data['day_of_week'] = self.historical_queue_data['timestamp'].dt.dayofweek
                
            logger.info(f"Loaded {len(self.historical_queue_data)} queue history records")
            
        except Exception as e:
            logger.error(f"Error loading queue history: {e}")
    
    def update_current_queue(self, current_ships: List[Dict]) -> None:
        """Update the current queue state"""
        self.current_queue = current_ships.copy()
        logger.debug(f"Updated current queue with {len(self.current_queue)} ships")
    
    def forecast_queue_length(self, hours_ahead: int = 24, current_time: datetime = None) -> List[QueueForecast]:
        """Forecast queue length for the next specified hours"""
        if current_time is None:
            current_time = datetime.now()
            
        forecasts = []
        
        try:
            # Current queue length
            current_queue_length = len(self.current_queue)
            current_waiting_time = self._estimate_current_waiting_time()
            
            # Generate hourly forecasts
            for hour in range(hours_ahead):
                forecast_time = current_time + timedelta(hours=hour)
                
                # Predict arrivals for this hour
                expected_arrivals = self._predict_hourly_arrivals(forecast_time)
                
                # Predict departures (ships finishing processing)
                expected_departures = self._predict_hourly_departures(forecast_time)
                
                # Update queue length
                predicted_queue_length = max(0, current_queue_length + expected_arrivals - expected_departures)
                
                # Estimate waiting time
                predicted_waiting_time = self._estimate_waiting_time(predicted_queue_length, forecast_time)
                
                # Determine trend
                if hour > 0:
                    prev_length = forecasts[-1].predicted_queue_length
                    if predicted_queue_length > prev_length * 1.1:
                        trend = 'increasing'
                    elif predicted_queue_length < prev_length * 0.9:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'
                else:
                    trend = 'stable'
                
                # Calculate confidence based on historical data availability
                confidence = self._calculate_forecast_confidence(forecast_time)
                
                forecast = QueueForecast(
                    timestamp=forecast_time,
                    predicted_queue_length=int(predicted_queue_length),
                    predicted_waiting_time=predicted_waiting_time,
                    confidence=confidence,
                    trend=trend
                )
                
                forecasts.append(forecast)
                current_queue_length = predicted_queue_length
            
            logger.info(f"Generated {len(forecasts)} queue forecasts")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error in queue forecasting: {e}")
            # Return simple fallback forecast
            return [QueueForecast(
                timestamp=current_time + timedelta(hours=i),
                predicted_queue_length=max(0, current_queue_length + np.random.randint(-2, 3)),
                predicted_waiting_time=2.0,
                confidence=0.3,
                trend='stable'
            ) for i in range(hours_ahead)]
    
    def _predict_hourly_arrivals(self, forecast_time: datetime) -> float:
        """Predict number of arrivals in a given hour"""
        # Simple model based on historical patterns
        hour = forecast_time.hour
        day_of_week = forecast_time.weekday()
        
        # Base arrival rate (ships per hour)
        base_rate = 0.5  # default
        
        if not self.historical_queue_data.empty and 'arrivals' in self.historical_queue_data.columns:
            # Use historical data if available
            hour_data = self.historical_queue_data[self.historical_queue_data['hour'] == hour]
            if not hour_data.empty:
                base_rate = hour_data['arrivals'].mean()
        
        # Adjust for time patterns
        time_factors = {
            # Hour factors (peak hours have more arrivals)
            'hour': {6: 0.5, 7: 0.8, 8: 1.2, 9: 1.5, 10: 1.3, 11: 1.1, 12: 1.0,
                    13: 1.1, 14: 1.3, 15: 1.2, 16: 1.0, 17: 0.8, 18: 0.6, 19: 0.4,
                    20: 0.3, 21: 0.2, 22: 0.2, 23: 0.1, 0: 0.1, 1: 0.1, 2: 0.1,
                    3: 0.1, 4: 0.2, 5: 0.3},
            # Day of week factors (weekdays vs weekends)
            'day': {0: 1.2, 1: 1.3, 2: 1.2, 3: 1.1, 4: 1.0, 5: 0.7, 6: 0.6}
        }
        
        hour_factor = time_factors['hour'].get(hour, 1.0)
        day_factor = time_factors['day'].get(day_of_week, 1.0)
        
        return base_rate * hour_factor * day_factor
    
    def _predict_hourly_departures(self, forecast_time: datetime) -> float:
        """Predict number of departures in a given hour"""
        # Assume steady processing rate during business hours
        hour = forecast_time.hour
        
        if 6 <= hour <= 22:  # Business hours
            return 0.6  # ships per hour
        else:  # Night hours
            return 0.2  # reduced processing
    
    def _estimate_current_waiting_time(self) -> float:
        """Estimate current average waiting time"""
        if not self.current_queue:
            return 0.0
            
        # Simple estimate: queue length * average processing time
        avg_processing_time = 4.0  # hours
        return len(self.current_queue) * avg_processing_time / 3  # assuming 3 berths
    
    def _estimate_waiting_time(self, queue_length: int, forecast_time: datetime) -> float:
        """Estimate waiting time for a given queue length"""
        if queue_length <= 0:
            return 0.0
            
        # Factors affecting waiting time
        avg_processing_time = 4.0  # hours
        num_berths = 3  # assume 3 active berths
        
        # Adjust for time of day (efficiency varies)
        hour = forecast_time.hour
        if 6 <= hour <= 18:  # Day shift
            efficiency = 1.0
        elif 18 <= hour <= 22:  # Evening shift
            efficiency = 0.8
        else:  # Night shift
            efficiency = 0.6
        
        effective_processing_rate = num_berths * efficiency / avg_processing_time
        waiting_time = queue_length / effective_processing_rate
        
        return max(0.0, waiting_time)
    
    def _calculate_forecast_confidence(self, forecast_time: datetime) -> float:
        """Calculate confidence level for forecast"""
        # Confidence decreases with time horizon
        hours_ahead = (forecast_time - datetime.now()).total_seconds() / 3600
        
        # Base confidence decreases exponentially with time
        base_confidence = np.exp(-hours_ahead / 24)  # 24-hour half-life
        
        # Adjust based on historical data availability
        if not self.historical_queue_data.empty:
            data_confidence = min(1.0, len(self.historical_queue_data) / 100)
        else:
            data_confidence = 0.3
        
        return max(0.1, base_confidence * data_confidence)

def create_sample_predictions() -> Dict:
    """Create sample predictions for testing and demonstration"""
    
    # Initialize predictors
    arrival_predictor = ShipArrivalPredictor()
    processing_estimator = ProcessingTimeEstimator()
    queue_forecaster = QueueLengthForecaster()
    
    # Create sample historical data
    sample_arrivals = pd.DataFrame({
        'arrival_time': pd.date_range(start='2024-01-01', periods=100, freq='6H'),
        'ship_type': np.random.choice(['container', 'bulk', 'tanker'], 100),
        'size': np.random.uniform(1000, 8000, 100)
    })
    
    # Train models
    arrival_predictor.load_historical_data(sample_arrivals)
    arrival_predictor.train_arrival_model()
    
    # Generate predictions
    current_time = datetime.now()
    
    # Arrival predictions
    next_container = arrival_predictor.predict_next_arrival('container', current_time)
    next_bulk = arrival_predictor.predict_next_arrival('bulk', current_time)
    
    # Processing time predictions
    container_processing = processing_estimator.estimate_processing_time('container', 3000, containers=150)
    bulk_processing = processing_estimator.estimate_processing_time('bulk', 5000, cargo_volume=2000)
    
    # Queue forecasts
    queue_forecaster.update_current_queue([
        {'ship_id': 'SHIP001', 'type': 'container'},
        {'ship_id': 'SHIP002', 'type': 'bulk'},
        {'ship_id': 'SHIP003', 'type': 'container'}
    ])
    queue_forecasts = queue_forecaster.forecast_queue_length(12)  # 12 hours ahead
    
    return {
        'arrival_predictions': {
            'next_container': next_container,
            'next_bulk': next_bulk
        },
        'processing_predictions': {
            'container_ship': container_processing,
            'bulk_ship': bulk_processing
        },
        'queue_forecasts': queue_forecasts[:6]  # First 6 hours
    }

if __name__ == "__main__":
    # Demo the predictive models
    logging.basicConfig(level=logging.INFO)
    
    print("Hong Kong Port Digital Twin - Predictive Models Demo")
    print("=" * 55)
    
    predictions = create_sample_predictions()
    
    print("\nArrival Predictions:")
    for ship_type, pred in predictions['arrival_predictions'].items():
        print(f"  Next {ship_type} ship: {pred.predicted_time.strftime('%Y-%m-%d %H:%M')} (confidence: {pred.probability:.1%})")
    
    print("\nProcessing Time Predictions:")
    for ship_type, pred in predictions['processing_predictions'].items():
        print(f"  {ship_type}: {pred.estimated_hours:.1f} hours (±{pred.uncertainty:.1%})")
    
    print("\nQueue Length Forecast (next 6 hours):")
    for forecast in predictions['queue_forecasts']:
        print(f"  {forecast.timestamp.strftime('%H:%M')}: {forecast.predicted_queue_length} ships, {forecast.predicted_waiting_time:.1f}h wait ({forecast.trend})")