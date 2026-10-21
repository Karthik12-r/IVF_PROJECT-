"""
SHAP Explanations for IVF ML Models.
Provides explainable AI outputs for doctors - feature importance, SHAP values.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available, using basic feature importance")


class ModelExplainer:
    """
    Provides explanations for ML model predictions using SHAP values.
    """
    
    def __init__(self, model=None, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or []
        self.explainer = None
        
        if SHAP_AVAILABLE and model is not None:
            self._init_shap_explainer()
    
    def _init_shap_explainer(self):
        """Initialize SHAP explainer based on model type."""
        try:
            if hasattr(self.model, 'predict_proba'):
                # Use TreeExplainer for tree-based models
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for other models
                self.explainer = shap.KernelExplainer(self.model.predict)
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, features: np.ndarray) -> Dict:
        """
        Generate explanation for a single prediction.
        
        Args:
            features: Feature vector for a single prediction
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if self.explainer is None or not SHAP_AVAILABLE:
            return self._basic_explanation(features)
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features.reshape(1, -1))
            
            # Get feature importance
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_values = shap_values[1]  # Use positive class
            
            # Create explanation
            feature_importance = []
            for i, (name, value) in enumerate(zip(self.feature_names, features)):
                if i < len(shap_values):
                    feature_importance.append({
                        'feature': name,
                        'value': float(value),
                        'shap_value': float(shap_values[i]),
                        'impact': 'positive' if shap_values[i] > 0 else 'negative'
                    })
            
            # Sort by absolute SHAP value
            feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            return {
                'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                'feature_importance': feature_importance,
                'top_contributors': feature_importance[:5],
                'explanation_method': 'shap'
            }
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return self._basic_explanation(features)
    
    def _basic_explanation(self, features: np.ndarray) -> Dict:
        """Generate basic explanation using feature importance."""
        # Use model feature importance if available
        importance = []
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for name, imp in zip(self.feature_names, importances):
                importance.append({
                    'feature': name,
                    'importance': float(imp),
                    'value': float(features[len(importance)]) if len(importance) < len(features) else 0
                })
        else:
            # Create dummy importance based on feature values
            for i, name in enumerate(self.feature_names):
                value = float(features[i]) if i < len(features) else 0
                importance.append({
                    'feature': name,
                    'importance': abs(value),
                    'value': value
                })
        
        # Sort by importance
        importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'feature_importance': importance,
            'top_contributors': importance[:5],
            'explanation_method': 'feature_importance'
        }
    
    def generate_global_explanation(self, X: np.ndarray, n_samples: int = 100) -> Dict:
        """
        Generate global explanation for the entire dataset.
        
        Args:
            X: Feature matrix
            n_samples: Number of samples to use for explanation
            
        Returns:
            Dictionary with global feature importance
        """
        if self.explainer is None or not SHAP_AVAILABLE:
            return self._basic_global_explanation(X)
        
        try:
            # Sample data if too large
            if len(X) > n_samples:
                indices = np.random.choice(len(X), n_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            feature_importance = []
            for name, value in zip(self.feature_names, mean_shap):
                feature_importance.append({
                    'feature': name,
                    'mean_shap': float(value),
                    'importance_rank': None
                })
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x['mean_shap'], reverse=True)
            
            # Add rank
            for i, feat in enumerate(feature_importance):
                feat['importance_rank'] = i + 1
            
            return {
                'global_feature_importance': feature_importance,
                'explanation_method': 'shap',
                'n_samples_analyzed': len(X_sample)
            }
            
        except Exception as e:
            logger.error(f"Error generating global explanation: {e}")
            return self._basic_global_explanation(X)
    
    def _basic_global_explanation(self, X: np.ndarray) -> Dict:
        """Generate basic global explanation."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = [
                {'feature': name, 'importance': float(imp), 'importance_rank': i+1}
                for i, (name, imp) in enumerate(zip(self.feature_names, importances))
            ]
        else:
            # Use feature statistics
            feature_importance = [
                {'feature': name, 'importance': float(np.std(X[:, i])), 'importance_rank': i+1}
                for i, name in enumerate(self.feature_names)
            ]
        
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'global_feature_importance': feature_importance,
            'explanation_method': 'feature_statistics',
            'n_samples_analyzed': len(X)
        }


class PredictionExplainer:
    """
    High-level interface for explaining IVF predictions.
    """
    
    @staticmethod
    def explain_ivf_success(model, patient_data, feature_names: List[str]) -> Dict:
        """
        Explain IVF success prediction.
        
        Args:
            model: Trained model
            patient_data: Patient data (object or dict)
            feature_names: List of feature names
            
        Returns:
            Explanation dictionary
        """
        # Prepare features
        if hasattr(patient_data, '__dict__'):
            features = [getattr(patient_data, name, 0) for name in feature_names]
        else:
            features = [patient_data.get(name, 0) for name in feature_names]
        
        features = np.array(features, dtype=float)
        
        # Create explainer
        explainer = ModelExplainer(model, feature_names)
        
        return explainer.explain_prediction(features)
    
    @staticmethod
    def explain_risk_assessment(model, patient_data, feature_names: List[str]) -> Dict:
        """
        Explain risk assessment prediction.
        
        Args:
            model: Trained model
            patient_data: Patient data
            feature_names: List of feature names
            
        Returns:
            Explanation dictionary
        """
        features = [getattr(patient_data, name, 0) if hasattr(patient_data, name) 
                    else patient_data.get(name, 0) for name in feature_names]
        features = np.array(features, dtype=float)
        
        explainer = ModelExplainer(model, feature_names)
        
        return explainer.explain_prediction(features)
    
    @staticmethod
    def generate_report(model, X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Generate comprehensive explanation report.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Comprehensive explanation report
        """
        explainer = ModelExplainer(model, feature_names)
        
        return explainer.generate_global_explanation(X)


def explain_prediction(model, features: np.ndarray, feature_names: List[str]) -> Dict:
    """
    Convenience function to explain a prediction.
    
    Args:
        model: Trained model
        features: Feature vector
        feature_names: List of feature names
        
    Returns:
        Explanation dictionary
    """
    explainer = ModelExplainer(model, feature_names)
    return explainer.explain_prediction(features)


def get_feature_importance(model, feature_names: List[str]) -> List[Dict]:
    """
    Get feature importance from a model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        List of feature importance dictionaries
    """
    if hasattr(model, 'feature_importances_'):
        importance = [
            {'feature': name, 'importance': float(imp)}
            for name, imp in zip(feature_names, model.feature_importances_)
        ]
        importance.sort(key=lambda x: x['importance'], reverse=True)
        return importance
    else:
        return [{'feature': name, 'importance': 0.0} for name in feature_names]


def generate_clinical_explanation(prediction: Dict, explanation: Dict) -> str:
    """
    Generate clinical explanation for doctors.
    
    Args:
        prediction: Prediction results
        explanation: Feature importance/shap values
        
    Returns:
        Human-readable clinical explanation
    """
    top_features = explanation.get('top_contributors', [])
    
    if not top_features:
        return "Unable to generate detailed explanation."
    
    # Build explanation
    lines = ["Key factors influencing this prediction:"]
    
    for i, feat in enumerate(top_features[:5], 1):
        feature = feat.get('feature', 'Unknown')
        impact = feat.get('impact', feat.get('shap_value', 0))
        
        if isinstance(impact, str):
            impact_text = f"({impact})"
        else:
            impact_text = f"(SHAP: {impact:.3f})" if isinstance(impact, (int, float)) else ""
        
        lines.append(f"{i}. {feature.replace('_', ' ').title()} {impact_text}")
    
    return "\n".join(lines)
