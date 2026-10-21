"""
Model Factory - Central hub for model selection, loading, and caching.
Automatically selects and loads the appropriate model based on the feature being used.
"""

import os
import joblib
import logging
from typing import Dict, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Base directory for saved models
MODEL_DIR = Path(__file__).parent / "saved_models"


class ModelFactory:
    """
    Factory class for managing ML models.
    Provides automatic model selection based on prediction type.
    """
    
    # Model type mappings
    MODEL_REGISTRY = {
        # IVF Success Prediction
        'ivf_success': {
            'model_type': 'xgboost',
            'filename': 'ivf_success_xgboost.pkl',
            'description': 'XGBoost model for IVF success prediction'
        },
        # Risk Assessment
        'risk_assessment': {
            'model_type': 'bnn',
            'filename': 'risk_assessment_bnn.pkl',
            'description': 'Bayesian Neural Network for risk assessment'
        },
        'ohss_risk': {
            'model_type': 'bnn',
            'filename': 'ohss_risk_bnn.pkl',
            'description': 'BNN for OHSS risk prediction'
        },
        'miscarriage_risk': {
            'model_type': 'bnn',
            'filename': 'miscarriage_risk_bnn.pkl',
            'description': 'BNN for miscarriage risk prediction'
        },
        # Treatment Recommendation
        'treatment_recommendation': {
            'model_type': 'lightgbm',
            'filename': 'treatment_recommendation_lgbm.pkl',
            'description': 'LightGBM for personalized treatment recommendations'
        },
        'protocol_selection': {
            'model_type': 'lightgbm',
            'filename': 'protocol_selection_lgbm.pkl',
            'description': 'LightGBM for protocol selection'
        },
        'dosage_prediction': {
            'model_type': 'lightgbm',
            'filename': 'dosage_prediction_lgbm.pkl',
            'description': 'LightGBM for medication dosage prediction'
        },
        # Hormone Prediction
        'hormone_prediction': {
            'model_type': 'bnn',
            'filename': 'hormone_prediction_bnn.pkl',
            'description': 'BNN for hormone level prediction'
        },
        'cycle_outcome': {
            'model_type': 'bnn',
            'filename': 'cycle_outcome_bnn.pkl',
            'description': 'BNN for cycle outcome prediction'
        },
        # Patient Classification
        'patient_classification': {
            'model_type': 'lightgbm',
            'filename': 'patient_classification_lgbm.pkl',
            'description': 'LightGBM for patient responder classification'
        },
        # Report Analysis
        'report_analysis': {
            'model_type': 'lightgbm',
            'filename': 'report_analysis_lgbm.pkl',
            'description': 'LightGBM for report analysis and auto-fill'
        },
        'abnormality_detection': {
            'model_type': 'lightgbm',
            'filename': 'abnormality_detection_lgbm.pkl',
            'description': 'LightGBM for abnormality detection in reports'
        },
        # Embryo Quality
        'embryo_quality': {
            'model_type': 'xgboost',
            'filename': 'embryo_quality_xgboost.pkl',
            'description': 'XGBoost for embryo quality prediction'
        }
    }
    
    # Cache for loaded models
    _model_cache: Dict[str, Any] = {}
    _metadata_cache: Dict[str, Dict] = {}
    
    @classmethod
    def get_model_path(cls, filename: str) -> Path:
        """Get the full path to a model file."""
        return MODEL_DIR / filename
    
    @classmethod
    def model_exists(cls, prediction_type: str) -> bool:
        """Check if a model exists for the given prediction type."""
        if prediction_type not in cls.MODEL_REGISTRY:
            return False
        
        model_info = cls.MODEL_REGISTRY[prediction_type]
        model_path = cls.get_model_path(model_info['filename'])
        return model_path.exists()
    
    @classmethod
    def load_model(cls, prediction_type: str) -> Optional[Any]:
        """
        Load a model for the given prediction type.
        Uses caching to avoid reloading models.
        
        Args:
            prediction_type: Type of prediction (e.g., 'ivf_success', 'risk_assessment')
            
        Returns:
            Loaded model or None if not found
        """
        # Check cache first
        if prediction_type in cls._model_cache:
            logger.info(f"Using cached model for {prediction_type}")
            return cls._model_cache[prediction_type]
        
        # Check if model type is registered
        if prediction_type not in cls.MODEL_REGISTRY:
            logger.warning(f"No model registered for prediction type: {prediction_type}")
            return None
        
        model_info = cls.MODEL_REGISTRY[prediction_type]
        model_path = cls.get_model_path(model_info['filename'])
        
        # Check if model file exists
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None
        
        try:
            # Load the model
            model = joblib.load(model_path)
            cls._model_cache[prediction_type] = model
            logger.info(f"Loaded model for {prediction_type}: {model_info['description']}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {prediction_type}: {str(e)}")
            return None
    
    @classmethod
    def load_metadata(cls, prediction_type: str) -> Optional[Dict]:
        """
        Load metadata for a model.
        
        Args:
            prediction_type: Type of prediction
            
        Returns:
            Metadata dictionary or None
        """
        if prediction_type in cls._metadata_cache:
            return cls._metadata_cache[prediction_type]
        
        if prediction_type not in cls.MODEL_REGISTRY:
            return None
        
        model_info = cls.MODEL_REGISTRY[prediction_type]
        metadata_path = cls.get_model_path(
            model_info['filename'].replace('.pkl', '_metadata.json')
        )
        
        if not metadata_path.exists():
            return None
        
        import json
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            cls._metadata_cache[prediction_type] = metadata
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return None
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict]:
        """
        Get list of all available models and their status.
        
        Returns:
            Dictionary of model types and their info
        """
        available = {}
        for pred_type, model_info in cls.MODEL_REGISTRY.items():
            model_path = cls.get_model_path(model_info['filename'])
            available[pred_type] = {
                'model_type': model_info['model_type'],
                'description': model_info['description'],
                'exists': model_path.exists()
            }
        return available
    
    @classmethod
    def clear_cache(cls):
        """Clear the model cache."""
        cls._model_cache.clear()
        cls._metadata_cache.clear()
        logger.info("Model cache cleared")
    
    @classmethod
    def predict(cls, prediction_type: str, features: Dict) -> Optional[Dict]:
        """
        Make a prediction using the appropriate model.
        
        Args:
            prediction_type: Type of prediction to make
            features: Input features for prediction
            
        Returns:
            Prediction results or None
        """
        model = cls.load_model(prediction_type)
        
        if model is None:
            return None
        
        try:
            # Get model type
            model_info = cls.MODEL_REGISTRY.get(prediction_type, {})
            model_type = model_info.get('model_type', 'unknown')
            
            # Prepare features based on model type
            import numpy as np
            
            if model_type == 'bnn':
                # BNN returns prediction with uncertainty
                result = cls._predict_with_uncertainty(model, features)
            else:
                # Standard prediction for XGBoost/LightGBM
                result = cls._predict_standard(model, features)
            
            # Add metadata
            metadata = cls.load_metadata(prediction_type)
            if metadata:
                result['model_version'] = metadata.get('version', 'unknown')
                result['feature_importance'] = metadata.get('feature_importance', {})
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    @classmethod
    def _predict_standard(cls, model, features: Dict) -> Dict:
        """Standard prediction for XGBoost/LightGBM models."""
        import numpy as np
        
        # Extract features in correct order
        feature_vector = cls._prepare_features(model, features)
        
        # Make prediction
        prediction = model.predict(feature_vector.reshape(1, -1))[0]
        
        # Get probability if available
        proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
            except:
                pass
        
        result = {
            'prediction': float(prediction),
            'confidence': float(np.max(proba)) if proba is not None else 0.8
        }
        
        if proba is not None:
            result['probabilities'] = proba.tolist()
        
        return result
    
    @classmethod
    def _predict_with_uncertainty(cls, model, features: Dict) -> Dict:
        """Prediction with uncertainty estimation for BNN models."""
        import numpy as np
        
        # Extract features
        feature_vector = cls._prepare_features(model, features)
        
        # BNN prediction with Monte Carlo dropout
        n_samples = 100
        predictions = []
        
        for _ in range(n_samples):
            pred = model.predict(feature_vector.reshape(1, -1))
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate mean and standard deviation
        mean_pred = float(np.mean(predictions))
        std_pred = float(np.std(predictions))
        
        return {
            'prediction': mean_pred,
            'confidence': 1.0 - std_pred,  # Lower uncertainty = higher confidence
            'uncertainty': std_pred,
            'uncertainty_interval': [
                float(np.percentile(predictions, 2.5)),
                float(np.percentile(predictions, 97.5))
            ]
        }
    
    @classmethod
    def _prepare_features(cls, model, features: Dict) -> np.ndarray:
        """Prepare feature vector in correct order."""
        import numpy as np
        
        # Get feature names from model
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        elif hasattr(model, 'feature_names'):
            feature_names = model.feature_names
        else:
            # Default feature order
            feature_names = ['age', 'bmi', 'amh', 'fsh', 'previous_ivf', 
                           'stress', 'sleep_hours', 'exercise_min']
        
        # Extract features in correct order
        feature_vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            if value is None:
                value = 0.0
            feature_vector.append(value)
        
        return np.array(feature_vector)


def get_model_for_prediction(prediction_type: str) -> Optional[Any]:
    """
    Convenience function to get a model for a prediction type.
    
    Args:
        prediction_type: Type of prediction
        
    Returns:
        Loaded model or None
    """
    return ModelFactory.load_model(prediction_type)


def predict_with_confidence(prediction_type: str, features: Dict) -> Optional[Dict]:
    """
    Convenience function to make a prediction with confidence score.
    
    Args:
        prediction_type: Type of prediction
        features: Input features
        
    Returns:
        Prediction results with confidence
    """
    return ModelFactory.predict(prediction_type, features)
