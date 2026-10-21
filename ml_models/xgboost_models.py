"""
XGBoost Models for IVF Journey Tracker.
Implements high-accuracy tabular predictions for IVF success and related tasks.
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# Default feature names
DEFAULT_FEATURES = ['age', 'bmi', 'amh', 'fsh', 'previous_ivf', 
                    'stress', 'sleep_hours', 'exercise_min']


class IVFSuccessPredictor:
    """
    XGBoost-based IVF success prediction model.
    Predicts probability of pregnancy success based on patient data.
    """
    
    MODEL_NAME = "ivf_success_xgboost"
    MODEL_TYPE = "xgboost_classifier"
    
    def __init__(self):
        self.model = None
        self.feature_names = DEFAULT_FEATURES
        self.metadata = {}
        self._load_model()
    
    def _load_model(self):
        """Load model from disk if exists."""
        model_path = MODEL_DIR / f"{self.MODEL_NAME}.pkl"
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                metadata_path = MODEL_DIR / f"{self.MODEL_NAME}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    self.feature_names = self.metadata.get('feature_names', DEFAULT_FEATURES)
                logger.info(f"Loaded {self.MODEL_NAME} model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.model = None
    
    def _prepare_features(self, patient_data) -> np.ndarray:
        """Prepare feature vector from patient data."""
        features = []
        
        for feature in self.feature_names:
            if hasattr(patient_data, feature):
                value = getattr(patient_data, feature, None)
            elif isinstance(patient_data, dict):
                value = patient_data.get(feature)
            else:
                value = None
            
            if value is None:
                # Use default values
                defaults = {
                    'age': 35, 'bmi': 24, 'amh': 2.0, 'fsh': 7.0,
                    'previous_ivf': 0, 'stress': 3, 'sleep_hours': 7, 'exercise_min': 30
                }
                value = defaults.get(feature, 0)
            
            features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, patient_data) -> Dict:
        """
        Predict IVF success probability.
        
        Args:
            patient_data: PatientData object or dict with features
            
        Returns:
            Dictionary with prediction, probability, and confidence
        """
        if self.model is None:
            # Return fallback prediction if model not loaded
            return self._fallback_prediction(patient_data)
        
        try:
            features = self._prepare_features(patient_data)
            
            # Get prediction
            prediction = self.model.predict(features)[0]
            
            # Get probability
            proba = self.model.predict_proba(features)[0]
            success_prob = float(proba[1])  # Probability of success (class 1)
            
            # Calculate confidence based on prediction margin
            confidence = self._calculate_confidence(proba)
            
            # Get feature importance
            importance = self._get_feature_importance()
            
            return {
                'prediction': int(prediction),
                'success_probability': round(success_prob * 100, 1),
                'confidence': confidence,
                'prediction_label': 'Success' if prediction == 1 else 'No Success',
                'feature_importance': importance,
                'model_type': self.MODEL_TYPE,
                'model_name': self.MODEL_NAME
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(patient_data)
    
    def _calculate_confidence(self, proba: np.ndarray) -> float:
        """Calculate confidence score from prediction probabilities."""
        # Higher margin between classes = higher confidence
        margin = abs(proba[0] - proba[1])
        confidence = min(95, 60 + margin * 35)  # Scale to 60-95
        return round(confidence, 1)
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance = self.model.feature_importances_
        return {
            name: round(float(imp), 4) 
            for name, imp in zip(self.feature_names, importance)
        }
    
    def _fallback_prediction(self, patient_data) -> Dict:
        """Fallback prediction when model not available."""
        # Use simple heuristic as fallback
        age = getattr(patient_data, 'age', 35) or 35
        amh = getattr(patient_data, 'amh_level', None)
        
        # Simple calculation
        base_prob = 35.0
        if age <= 30:
            base_prob += 15
        elif age <= 35:
            base_prob += 5
        elif age > 40:
            base_prob -= 20
        
        if amh:
            if amh >= 2.0:
                base_prob += 10
            elif amh < 1.0:
                base_prob -= 10
        
        success_prob = max(10, min(80, base_prob))
        
        return {
            'prediction': 1 if success_prob >= 50 else 0,
            'success_probability': round(success_prob, 1),
            'confidence': 50.0,
            'prediction_label': 'Success' if success_prob >= 50 else 'No Success',
            'feature_importance': {},
            'model_type': 'fallback',
            'model_name': self.MODEL_NAME,
            'note': 'Model not trained yet'
        }


class EmbryoQualityPredictor:
    """
    XGBoost-based embryo quality prediction.
    Predicts embryo quality scores based on patient and cycle data.
    """
    
    MODEL_NAME = "embryo_quality_xgboost"
    
    def __init__(self):
        self.model = None
        self.feature_names = ['age', 'amh', 'bmi', 'fsh', 'egg_quality_score', 
                           'sperm_morphology', 'fertilization_method']
        self._load_model()
    
    def _load_model(self):
        """Load model from disk if exists."""
        model_path = MODEL_DIR / f"{self.MODEL_NAME}.pkl"
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded {self.MODEL_NAME} model")
            except:
                self.model = None
    
    def _prepare_features(self, data) -> np.ndarray:
        """Prepare feature vector."""
        features = []
        
        for feature in self.feature_names:
            if hasattr(data, feature):
                value = getattr(data, feature, None)
            elif isinstance(data, dict):
                value = data.get(feature)
            else:
                value = None
            
            if value is None:
                value = 0
            features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, data) -> Dict:
        """
        Predict embryo quality.
        
        Args:
            data: Patient/Cycle data or dict
            
        Returns:
            Dictionary with quality score and grade
        """
        if self.model is None:
            return self._fallback_prediction(data)
        
        try:
            features = self._prepare_features(data)
            quality_score = self.model.predict(features)[0]
            
            # Determine grade
            if quality_score >= 80:
                grade = "A (Excellent)"
            elif quality_score >= 65:
                grade = "B (Good)"
            elif quality_score >= 45:
                grade = "C (Fair)"
            else:
                grade = "D (Poor)"
            
            return {
                'quality_score': round(float(quality_score), 1),
                'grade': grade,
                'development_probability': round(min(90, quality_score * 0.9), 1),
                'implantation_potential': round(min(85, quality_score * 0.8), 1)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(data)
    
    def _fallback_prediction(self, data) -> Dict:
        """Fallback prediction."""
        return {
            'quality_score': 65.0,
            'grade': 'B (Good)',
            'development_probability': 58.5,
            'implantation_potential': 52.0,
            'note': 'Model not trained yet'
        }


class XGBoostModelTrainer:
    """
    Trainer class for XGBoost models.
    """
    
    @staticmethod
    def train_ivf_success_model(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               feature_names: List[str]) -> Tuple[XGBClassifier, Dict]:
        """
        Train IVF success prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            
        Returns:
            Tuple of (trained model, metrics dict)
        """
        # Create model
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'feature_importance': {
                name: float(imp) 
                for name, imp in zip(feature_names, model.feature_importances_)
            }
        }
        
        # Save model
        model_path = MODEL_DIR / "ivf_success_xgboost.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': 'ivf_success_xgboost',
            'model_type': 'xgboost_classifier',
            'feature_names': feature_names,
            'metrics': metrics,
            'version': '1.0.0'
        }
        
        metadata_path = MODEL_DIR / "ivf_success_xgboost_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model trained and saved: {model_path}")
        logger.info(f"Metrics: {metrics}")
        
        return model, metrics
    
    @staticmethod
    def train_embryo_quality_model(X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  feature_names: List[str]) -> Tuple[XGBRegressor, Dict]:
        """
        Train embryo quality prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels (quality scores)
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            
        Returns:
            Tuple of (trained model, metrics dict)
        """
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred)),
            'feature_importance': {
                name: float(imp) 
                for name, imp in zip(feature_names, model.feature_importances_)
            }
        }
        
        # Save model
        model_path = MODEL_DIR / "embryo_quality_xgboost.pkl"
        joblib.dump(model, model_path)
        
        logger.info(f"Embryo quality model trained: {model_path}")
        
        return model, metrics


# Convenience functions
def get_ivf_success_predictor() -> IVFSuccessPredictor:
    """Get IVF success predictor instance."""
    return IVFSuccessPredictor()


def get_embryo_quality_predictor() -> EmbryoQualityPredictor:
    """Get embryo quality predictor instance."""
    return EmbryoQualityPredictor()


def predict_ivf_success(patient_data) -> Dict:
    """Convenience function to predict IVF success."""
    predictor = IVFSuccessPredictor()
    return predictor.predict(patient_data)


def predict_embryo_quality(data) -> Dict:
    """Convenience function to predict embryo quality."""
    predictor = EmbryoQualityPredictor()
    return predictor.predict(data)
