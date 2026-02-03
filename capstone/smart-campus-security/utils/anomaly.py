"""
Anomaly detection and prediction module
Uses IsolationForest for anomaly detection and Prophet for time series forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from loguru import logger

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    logger.warning("Prophet not available, prediction features disabled")
    PROPHET_AVAILABLE = False


class AnomalyDetector:
    """Anomaly detection using IsolationForest"""
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.model = None
        self.label_encoders = {}
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for anomaly detection
        
        Args:
            df: DataFrame with columns: timestamp, person_id, cam_id, role_type
        
        Returns:
            Feature matrix
        """
        features = df.copy()
        
        # Extract time features
        features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
        features['minute'] = pd.to_datetime(features['timestamp']).dt.minute
        features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
        features['entry_minutes'] = features['hour'] * 60 + features['minute']
        
        # Encode categorical features
        for col in ['role_type']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features[col])
            else:
                features[f'{col}_encoded'] = self.label_encoders[col].transform(features[col])
        
        # Select feature columns
        feature_cols = ['entry_minutes', 'day_of_week', 'cam_id', 'role_type_encoded']
        
        # Add person_id if available
        if 'person_id' in features.columns:
            features['person_id_filled'] = features['person_id'].fillna(-1)
            feature_cols.append('person_id_filled')
        
        return features[feature_cols].values
    
    def train(self, df: pd.DataFrame):
        """
        Train anomaly detector on historical data
        
        Args:
            df: Historical logs DataFrame
        """
        if len(df) < 10:
            logger.warning("Insufficient data for training anomaly detector")
            return
        
        X = self.prepare_features(df)
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.model.fit(X)
        self.is_trained = True
        
        logger.info(f"Anomaly detector trained on {len(df)} samples")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies
        
        Args:
            df: DataFrame to check for anomalies
        
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_trained:
            logger.warning("Anomaly detector not trained")
            return np.ones(len(df))
        
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        num_anomalies = np.sum(predictions == -1)
        logger.info(f"Detected {num_anomalies} anomalies out of {len(df)} samples")
        
        return predictions
    
    def save(self, path: str):
        """Save model to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoders': self.label_encoders,
                'contamination': self.contamination
            }, f)
        logger.info(f"Anomaly detector saved to {path}")
    
    def load(self, path: str):
        """Load model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoders = data['label_encoders']
            self.contamination = data['contamination']
            self.is_trained = True
        logger.info(f"Anomaly detector loaded from {path}")


class LateArrivalPredictor:
    """Predict late arrivals using Prophet time series forecasting"""
    
    def __init__(self):
        self.models = {}  # person_id -> Prophet model
        self.is_available = PROPHET_AVAILABLE
    
    def train_person_model(self, person_id: int, df: pd.DataFrame):
        """
        Train prediction model for a specific person
        
        Args:
            person_id: Person ID
            df: Historical attendance data for this person
        """
        if not self.is_available:
            logger.warning("Prophet not available")
            return
        
        if len(df) < 7:  # Need at least a week of data
            logger.warning(f"Insufficient data for person {person_id}")
            return
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['timestamp']),
            'y': pd.to_datetime(df['timestamp']).dt.hour * 60 + 
                 pd.to_datetime(df['timestamp']).dt.minute
        })
        
        # Train model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        model.fit(prophet_df)
        
        self.models[person_id] = model
        logger.info(f"Trained prediction model for person {person_id}")
    
    def predict_arrival_time(self, person_id: int, date: datetime) -> Optional[float]:
        """
        Predict arrival time for person on given date
        
        Args:
            person_id: Person ID
            date: Date to predict
        
        Returns:
            Predicted arrival time in minutes from midnight, or None
        """
        if not self.is_available or person_id not in self.models:
            return None
        
        model = self.models[person_id]
        
        # Create future dataframe
        future = pd.DataFrame({'ds': [date]})
        
        # Predict
        forecast = model.predict(future)
        predicted_minutes = forecast['yhat'].iloc[0]
        
        return predicted_minutes
    
    def predict_late_arrivals(self, person_ids: List[int], date: datetime,
                             threshold_minutes: int = 30) -> List[Dict]:
        """
        Predict which persons will arrive late
        
        Args:
            person_ids: List of person IDs to check
            date: Date to predict
            threshold_minutes: Minutes after typical time to consider late
        
        Returns:
            List of predictions with person_id and predicted_minutes
        """
        predictions = []
        
        for person_id in person_ids:
            predicted_time = self.predict_arrival_time(person_id, date)
            
            if predicted_time is not None:
                # Compare with historical average (simplified)
                # In production, would compare with actual average
                typical_time = 9 * 60  # 9:00 AM
                
                if predicted_time > typical_time + threshold_minutes:
                    predictions.append({
                        'person_id': person_id,
                        'predicted_minutes': predicted_time,
                        'typical_minutes': typical_time,
                        'delay_minutes': predicted_time - typical_time
                    })
        
        return predictions


class AnalyticsEngine:
    """Generate analytics and reports"""
    
    @staticmethod
    def generate_daily_report(df: pd.DataFrame, date: datetime) -> Dict:
        """
        Generate daily analytics report
        
        Args:
            df: Logs DataFrame
            date: Date to analyze
        
        Returns:
            Analytics dictionary
        """
        # Filter for date
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_df = df[df['date'] == date.date()]
        
        if len(daily_df) == 0:
            return {'error': 'No data for this date'}
        
        report = {
            'date': date.strftime('%Y-%m-%d'),
            'total_entries': len(daily_df),
            'unique_persons': daily_df['person_id'].nunique(),
            'role_breakdown': daily_df['role_type'].value_counts().to_dict(),
            'camera_usage': daily_df['cam_id'].value_counts().to_dict(),
            'avg_confidence': float(daily_df['confidence'].mean()),
            'mask_compliance': float(daily_df['has_mask'].mean() * 100),
            'helmet_compliance': float(daily_df['has_helmet'].mean() * 100),
            'peak_hour': int(pd.to_datetime(daily_df['timestamp']).dt.hour.mode()[0])
        }
        
        return report
    
    @staticmethod
    def generate_weekly_report(df: pd.DataFrame, start_date: datetime) -> Dict:
        """
        Generate weekly analytics report
        
        Args:
            df: Logs DataFrame
            start_date: Start of week
        
        Returns:
            Analytics dictionary
        """
        end_date = start_date + timedelta(days=7)
        
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        weekly_df = df[
            (df['date'] >= start_date.date()) & 
            (df['date'] < end_date.date())
        ]
        
        if len(weekly_df) == 0:
            return {'error': 'No data for this week'}
        
        # Daily trends
        daily_counts = weekly_df.groupby('date').size().to_dict()
        daily_counts = {str(k): v for k, v in daily_counts.items()}
        
        report = {
            'week_start': start_date.strftime('%Y-%m-%d'),
            'week_end': end_date.strftime('%Y-%m-%d'),
            'total_entries': len(weekly_df),
            'unique_persons': weekly_df['person_id'].nunique(),
            'daily_counts': daily_counts,
            'role_breakdown': weekly_df['role_type'].value_counts().to_dict(),
            'avg_daily_entries': float(len(weekly_df) / 7),
            'busiest_day': max(daily_counts, key=daily_counts.get) if daily_counts else None
        }
        
        return report


if __name__ == "__main__":
    # Test anomaly detector
    logger.info("Testing anomaly detection...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'person_id': np.random.randint(1, 10, n_samples),
        'cam_id': np.random.randint(0, 2, n_samples),
        'role_type': np.random.choice(['student', 'staff', 'labour'], n_samples)
    })
    
    # Train detector
    detector = AnomalyDetector()
    detector.train(sample_data)
    
    # Predict
    predictions = detector.predict(sample_data.head(10))
    print(f"Predictions: {predictions}")
    
    logger.info("Anomaly detection test completed")
