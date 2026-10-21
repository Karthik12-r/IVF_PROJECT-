"""
LightGBM Models for IVF Journey Tracker.
Implements classification and decision modeling for patient categorization, 
treatment recommendations, and report analysis.
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from lightgbm import LGBMClassifier, LGBMRegressor

logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


class PatientClassificationPredictor:
    """
    LightGBM-based patient classification.
    Classifies patients into: Low Responder, Normal Responder, High Responder
    """
    
    MODEL_NAME = "patient_classification_lgbm"
    CLASSES = ['Low Responder', 'Normal Responder', 'High Responder']
    
    def __init__(self):
        self.model = None
        self.feature_names = ['age', 'amh', 'fsh', 'bmi', 'previous_ivf_cycles', 
                            'num_eggs_retrieved', 'fertilization_rate']
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
        """Prepare feature vector from data."""
        features = []
        
        feature_mapping = {
            'age': ['age', 'age'],
            'amh': ['amh_level', 'amh'],
            'fsh': ['fsh_level', 'fsh'],
            'bmi': ['bmi', 'bmi'],
            'previous_ivf_cycles': ['previous_ivf_cycles', 'previous_ivf'],
            'num_eggs_retrieved': ['num_eggs_retrieved', 'eggs_retrieved'],
            'fertilization_rate': ['fertilization_rate', 'fertilization_rate']
        }
        
        for feature_key in self.feature_names:
            value = None
            
            if hasattr(data, feature_key):
                value = getattr(data, feature_key, None)
            elif isinstance(data, dict):
                # Try multiple possible keys
                for possible_keys in feature_mapping.values():
                    if feature_key in possible_keys:
                        value = data.get(feature_key)
                        if value is not None:
                            break
            
            if value is None:
                # Default values
                defaults = {
                    'age': 35, 'amh': 2.0, 'fsh': 7.0, 'bmi': 24,
                    'previous_ivf_cycles': 0, 'num_eggs_retrieved': 10,
                    'fertilization_rate': 0.7
                }
                value = defaults.get(feature_key, 0)
            
            features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, data) -> Dict:
        """
        Classify patient into responder category.
        
        Args:
            data: Patient data (object or dict)
            
        Returns:
            Dictionary with classification and probabilities
        """
        if self.model is None:
            return self._fallback_prediction(data)
        
        try:
            features = self._prepare_features(data)
            
            # Get prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Get class name
            class_name = self.CLASSES[int(prediction)]
            confidence = float(probabilities[int(prediction)])
            
            return {
                'classification': class_name,
                'classification_code': int(prediction),
                'confidence': round(confidence * 100, 1),
                'probabilities': {
                    cls: round(prob * 100, 1) 
                    for cls, prob in zip(self.CLASSES, probabilities)
                },
                'model_type': 'lightgbm_classifier',
                'model_name': self.MODEL_NAME
            }
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._fallback_prediction(data)
    
    def _fallback_prediction(self, data) -> Dict:
        """Fallback classification based on AMH levels."""
        amh = None
        
        if hasattr(data, 'amh_level'):
            amh = data.amh_level
        elif isinstance(data, dict):
            amh = data.get('amh_level') or data.get('amh')
        
        if amh is None:
            amh = 2.0
        
        # Simple classification based on AMH
        if amh < 1.0:
            classification = 'Low Responder'
            confidence = 70.0
        elif amh > 4.0:
            classification = 'High Responder'
            confidence = 70.0
        else:
            classification = 'Normal Responder'
            confidence = 60.0
        
        return {
            'classification': classification,
            'classification_code': self.CLASSES.index(classification),
            'confidence': confidence,
            'probabilities': {
                'Low Responder': 33.3,
                'Normal Responder': 33.3,
                'High Responder': 33.3
            },
            'model_type': 'fallback',
            'model_name': self.MODEL_NAME,
            'note': 'Model not trained - using rule-based fallback'
        }


class TreatmentRecommendationPredictor:
    """
    LightGBM-based treatment recommendation.
    Suggests medication dosage, stimulation protocol, cycle adjustments.
    """
    
    MODEL_NAME = "treatment_recommendation_lgbm"
    
    def __init__(self):
        self.model = None
        self.feature_names = ['age', 'amh', 'bmi', 'fsh', 'previous_ivf_cycles', 
                            'diagnosis_code', 'responder_type']
        self._load_model()
    
    def _load_model(self):
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
        Generate treatment recommendations.
        
        Args:
            data: Patient data
            
        Returns:
            Dictionary with treatment recommendations
        """
        if self.model is None:
            return self._fallback_prediction(data)
        
        try:
            features = self._prepare_features(data)
            recommendations = self.model.predict(features)[0]
            
            return {
                'protocol': recommendations.get('protocol', 'Standard Antagonist'),
                'fsh_dosage': recommendations.get('fsh_dosage', '150-225 IU'),
                'adjustments': recommendations.get('adjustments', []),
                'confidence': 75.0,
                'model_type': 'lightgbm',
                'model_name': self.MODEL_NAME
            }
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return self._fallback_prediction(data)
    
    def _fallback_prediction(self, data) -> Dict:
        """Fallback recommendations."""
        age = getattr(data, 'age', 35) or 35
        amh = getattr(data, 'amh_level', None) or 2.0
        
        # Determine protocol based on age and AMH
        if age <= 35 and amh >= 2.0:
            protocol = "Standard Antagonist Protocol"
            dosage = "150-225 IU FSH"
        elif age <= 35 and amh < 1.0:
            protocol = "High-Dose Stimulation"
            dosage = "300-450 IU FSH"
        elif age > 35 and amh >= 1.5:
            protocol = "Mild Stimulation Protocol"
            dosage = "100-150 IU FSH"
        else:
            protocol = "Mini-IVF Protocol"
            dosage = "75-100 IU FSH"
        
        return {
            'protocol': protocol,
            'fsh_dosage': dosage,
            'adjustments': ['Consider adding LH supplementation', 'Monitor closely for OHSS'],
            'confidence': 60.0,
            'model_type': 'fallback',
            'model_name': self.MODEL_NAME,
            'note': 'Model not trained - using rule-based fallback'
        }


class ReportAnalysisPredictor:
    """
    LightGBM-based report analysis and abnormality detection.
    Analyzes extracted OCR values and detects abnormalities.
    """
    
    MODEL_NAME = "report_analysis_lgbm"
    
    def __init__(self):
        self.model = None
        self.feature_names = ['e2_level', 'p4_level', 'fsh_level', 'lh_level', 
                            'amh_level', 'prolactin', 'tsh']
        self._load_model()
    
    def _load_model(self):
        model_path = MODEL_DIR / f"{self.MODEL_NAME}.pkl"
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded {self.MODEL_NAME} model")
            except:
                self.model = None
    
    def _prepare_features(self, data) -> np.ndarray:
        """Prepare feature vector from extracted values."""
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
        Analyze report values for abnormalities.
        
        Args:
            data: Extracted lab values (dict or object)
            
        Returns:
            Dictionary with analysis and detected abnormalities
        """
        if self.model is None:
            return self._fallback_analysis(data)
        
        try:
            features = self._prepare_features(data)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Get abnormalities
            abnormalities = []
            if isinstance(data, dict):
                # Check each value against normal ranges
                normal_ranges = {
                    'e2_level': (20, 300),
                    'p4_level': (0.1, 1.5),
                    'fsh_level': (4, 13),
                    'lh_level': (2, 12),
                    'amh_level': (1, 4),
                    'prolactin': (4, 25),
                    'tsh': (0.4, 4.0)
                }
                
                for param, (low, high) in normal_ranges.items():
                    value = data.get(param)
                    if value is not None:
                        if value < low:
                            abnormalities.append(f"{param} is below normal range ({low}-{high})")
                        elif value > high:
                            abnormalities.append(f"{param} is above normal range ({low}-{high})")
            
            return {
                'has_abnormalities': bool(abnormalities),
                'abnormalities': abnormalities,
                'risk_score': float(prediction),
                'confidence': round(float(max(probabilities)) * 100, 1),
                'auto_fill_suggestions': self._generate_auto_fill(data),
                'model_type': 'lightgbm_classifier',
                'model_name': self.MODEL_NAME
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return self._fallback_analysis(data)
    
    def _generate_auto_fill(self, data) -> Dict:
        """Generate auto-fill suggestions for missing values."""
        suggestions = {}
        
        if isinstance(data, dict):
            for key in self.feature_names:
                if data.get(key) is None:
                    suggestions[key] = "Value not detected - manual entry required"
        
        return suggestions
    
    def _fallback_analysis(self, data) -> Dict:
        """Fallback analysis using rule-based detection."""
        abnormalities = []
        
        if isinstance(data, dict):
            normal_ranges = {
                'e2_level': (20, 300),
                'p4_level': (0.1, 1.5),
                'fsh_level': (4, 13),
                'amh_level': (1, 4)
            }
            
            for param, (low, high) in normal_ranges.items():
                value = data.get(param)
                if value is not None:
                    if value < low:
                        abnormalities.append(f"{param} is below normal range")
                    elif value > high:
                        abnormalities.append(f"{param} is above normal range")
        
        return {
            'has_abnormalities': bool(abnormalities),
            'abnormalities': abnormalities,
            'risk_score': 0.5,
            'confidence': 50.0,
            'auto_fill_suggestions': {},
            'model_type': 'fallback',
            'model_name': self.MODEL_NAME,
            'note': 'Model not trained - using rule-based analysis'
        }


class LightGBMModelTrainer:
    """
    Trainer class for LightGBM models.
    """
    
    @staticmethod
    def train_patient_classification(X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray,
                                     feature_names: List[str]) -> Tuple[LGBMClassifier, Dict]:
        """Train patient classification model."""
        
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        from sklearn.metrics import accuracy_score, classification_report
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': {
                name: float(imp) 
                for name, imp in zip(feature_names, model.feature_importances_)
            }
        }
        
        # Save model
        model_path = MODEL_DIR / "patient_classification_lgbm.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': 'patient_classification_lgbm',
            'model_type': 'lightgbm_classifier',
            'feature_names': feature_names,
            'classes': ['Low Responder', 'Normal Responder', 'High Responder'],
            'metrics': metrics,
            'version': '1.0.0'
        }
        
        metadata_path = MODEL_DIR / "patient_classification_lgbm_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Patient classification model trained: {model_path}")
        
        return model, metrics
    
    @staticmethod
    def train_treatment_recommendation(X_train: np.ndarray, y_train: np.ndarray,
                                       X_test: np.ndarray, y_test: np.ndarray,
                                       feature_names: List[str]) -> Tuple[LGBMRegressor, Dict]:
        """Train treatment recommendation model."""
        
        model = LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
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
        
        model_path = MODEL_DIR / "treatment_recommendation_lgbm.pkl"
        joblib.dump(model, model_path)
        
        logger.info(f"Treatment recommendation model trained: {model_path}")
        
        return model, metrics
    
    @staticmethod
    def train_report_analysis(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             feature_names: List[str]) -> Tuple[LGBMClassifier, Dict]:
        """Train report analysis model."""
        
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        from sklearn.metrics import accuracy_score, roc_auc_score
        
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
        
        model_path = MODEL_DIR / "report_analysis_lgbm.pkl"
        joblib.dump(model, model_path)
        
        logger.info(f"Report analysis model trained: {model_path}")
        
        return model, metrics


# Convenience functions
def get_patient_classifier() -> PatientClassificationPredictor:
    """Get patient classification predictor instance."""
    return PatientClassificationPredictor()


def get_treatment_recommender() -> TreatmentRecommendationPredictor:
    """Get treatment recommendation predictor instance."""
    return TreatmentRecommendationPredictor()


def get_report_analyzer() -> ReportAnalysisPredictor:
    """Get report analysis predictor instance."""
    return ReportAnalysisPredictor()


def classify_patient(data) -> Dict:
    """Convenience function to classify patient."""
    classifier = PatientClassificationPredictor()
    return classifier.predict(data)


def recommend_treatment(data) -> Dict:
    """Convenience function to get treatment recommendations."""
    recommender = TreatmentRecommendationPredictor()
    return recommender.predict(data)


def analyze_report(data) -> Dict:
    """Convenience function to analyze medical reports."""
    analyzer = ReportAnalysisPredictor()
    return analyzer.predict(data)
