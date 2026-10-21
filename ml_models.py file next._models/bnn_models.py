"""
Bayesian Neural Network Models for IVF Journey Tracker.
Implements probabilistic predictions with uncertainty estimation for:
- Risk Assessment (OHSS, Miscarriage, Complications)
- Hormone Level and Cycle Outcome Prediction
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# Try to import torch, if not available use sklearn approximation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using sklearn approximation for BNN")


class BayesianNeuralNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Bayesian Neural Network for probabilistic predictions.
    Uses Monte Carlo dropout for uncertainty estimation.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 64):
        if not TORCH_AVAILABLE:
            self.input_dim = input_dim
            self.output_dim = output_dim
            return
            
        super().__init__()
        
        # Bayesian linear layers with dropout for uncertainty
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout for Bayesian approximation
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return None
            
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class BNNRiskAssessor:
    """
    Bayesian Neural Network for risk assessment.
    Predicts OHSS risk, miscarriage risk, and complications with uncertainty.
    """
    
    MODEL_NAME = "risk_assessment_bnn"
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['age', 'amh', 'bmi', 'fsh', 'e2_level', 'p4_level',
                           'num_follicles', 'previous_ohss', 'pcos_diagnosis']
        self._load_model()
    
    def _load_model(self):
        """Load model from disk."""
        model_path = MODEL_DIR / f"{self.MODEL_NAME}.pkl"
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.is_trained = True
                logger.info(f"Loaded {self.MODEL_NAME} model")
            except:
                self.model = None
    
    def _prepare_features(self, data) -> np.ndarray:
        """Prepare feature vector."""
        features = []
        
        feature_mapping = {
            'age': ['age', 'age'],
            'amh': ['amh_level', 'amh'],
            'bmi': ['bmi', 'bmi'],
            'fsh': ['fsh_level', 'fsh'],
            'e2_level': ['e2_level', 'estradiol'],
            'p4_level': ['p4_level', 'progesterone'],
            'num_follicles': ['num_follicles', 'follicles'],
            'previous_ohss': ['previous_ohss', 'ohss_history'],
            'pcos_diagnosis': ['pcos_diagnosis', 'has_pcos']
        }
        
        for feature_key in self.feature_names:
            value = None
            
            if hasattr(data, feature_key):
                value = getattr(data, feature_key, None)
            elif isinstance(data, dict):
                for possible in feature_mapping.get(feature_key, [feature_key]):
                    value = data.get(possible)
                    if value is not None:
                        break
            
            if value is None:
                value = 0
            
            features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, data: Dict, n_samples: int = 100) -> Dict:
        """
        Predict risk with uncertainty estimation.
        
        Args:
            data: Patient data
            n_samples: Number of Monte Carlo samples for uncertainty
            
        Returns:
            Dictionary with risk probabilities and confidence intervals
        """
        if self.model is None or not self.is_trained:
            return self._fallback_prediction(data)
        
        try:
            features = self._prepare_features(data)
            features_scaled = self.scaler.transform(features)
            
            # Convert to torch tensor
            if TORCH_AVAILABLE:
                features_tensor = torch.FloatTensor(features_scaled)
                
                # Monte Carlo sampling for uncertainty
                predictions = []
                self.model.train()  # Enable dropout
                
                for _ in range(n_samples):
                    with torch.no_grad():
                        pred = self.model(features_tensor).numpy()[0, 0]
                        predictions.append(pred)
                
                predictions = np.array(predictions)
                
                # Calculate statistics
                mean_pred = float(np.mean(predictions))
                std_pred = float(np.std(predictions))
                
                return {
                    'risk_probability': round(mean_pred * 100, 1),
                    'confidence': round((1 - std_pred) * 100, 1),
                    'uncertainty': round(std_pred * 100, 1),
                    'uncertainty_interval': [
                        round(np.percentile(predictions, 2.5) * 100, 1),
                        round(np.percentile(predictions, 97.5) * 100, 1)
                    ],
                    'risk_level': self._get_risk_level(mean_pred),
                    'model_type': 'bnn',
                    'model_name': self.MODEL_NAME
                }
            else:
                # Fallback to sklearn approximation
                prob = self.model.predict_proba(features_scaled)[0, 1]
                return {
                    'risk_probability': round(prob * 100, 1),
                    'confidence': 70.0,
                    'uncertainty': 15.0,
                    'uncertainty_interval': [round(prob * 80, 1), round(prob * 120, 1)],
                    'risk_level': self._get_risk_level(prob),
                    'model_type': 'bnn_sklearn',
                    'model_name': self.MODEL_NAME,
                    'note': 'Using sklearn approximation'
                }
                
        except Exception as e:
            logger.error(f"Risk prediction error: {e}")
            return self._fallback_prediction(data)
    
    def _get_risk_level(self, probability: float) -> str:
        """Get risk level category."""
        if probability < 0.2:
            return "Low"
        elif probability < 0.5:
            return "Moderate"
        elif probability < 0.75:
            return "High"
        else:
            return "Very High"
    
    def _fallback_prediction(self, data: Dict) -> Dict:
        """Fallback risk prediction using rules."""
        age = data.get('age', 35) if isinstance(data, dict) else getattr(data, 'age', 35)
        amh = data.get('amh_level', 2.0) if isinstance(data, dict) else getattr(data, 'amh_level', 2.0)
        bmi = data.get('bmi', 24) if isinstance(data, dict) else getattr(data, 'bmi', 24)
        
        # Simple OHSS risk calculation
        risk_score = 0.1
        
        if amh > 4.0:
            risk_score += 0.3
        if age < 30:
            risk_score += 0.2
        if bmi > 30:
            risk_score += 0.1
        
        risk_prob = min(0.9, risk_score)
        
        return {
            'risk_probability': round(risk_prob * 100, 1),
            'confidence': 50.0,
            'uncertainty': 20.0,
            'uncertainty_interval': [round(risk_prob * 70 * 100, 1), round(risk_prob * 130 * 100, 1)],
            'risk_level': self._get_risk_level(risk_prob),
            'model_type': 'fallback',
            'model_name': self.MODEL_NAME,
            'note': 'Model not trained - using rule-based fallback'
        }


class BNNHormonePredictor:
    """
    Bayesian Neural Network for hormone level and cycle outcome prediction.
    Provides probabilistic forecasting with confidence intervals.
    """
    
    MODEL_NAME = "hormone_prediction_bnn"
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['day_of_cycle', 'fsh_level', 'e2_level', 'p4_level',
                           'lh_level', 'follicle_size', 'endometrial_thickness']
        self._load_model()
    
    def _load_model(self):
        model_path = MODEL_DIR / f"{self.MODEL_NAME}.pkl"
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.is_trained = True
                logger.info(f"Loaded {self.MODEL_NAME} model")
            except:
                self.model = None
    
    def _prepare_features(self, data) -> np.ndarray:
        """Prepare feature vector for hormone prediction."""
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
    
    def predict(self, data: Dict, n_samples: int = 100) -> Dict:
        """
        Predict hormone levels with uncertainty.
        
        Args:
            data: Current hormone and cycle data
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with predicted hormone trends and confidence
        """
        if self.model is None or not self.is_trained:
            return self._fallback_prediction(data)
        
        try:
            features = self._prepare_features(data)
            features_scaled = self.scaler.transform(features)
            
            if TORCH_AVAILABLE:
                features_tensor = torch.FloatTensor(features_scaled)
                
                predictions = []
                self.model.train()
                
                for _ in range(n_samples):
                    with torch.no_grad():
                        pred = self.model(features_tensor).numpy()[0, 0]
                        predictions.append(pred)
                
                predictions = np.array(predictions)
                
                mean_pred = float(np.mean(predictions))
                std_pred = float(np.std(predictions))
                
                return {
                    'predicted_value': round(mean_pred, 2),
                    'unit': 'ng/mL',
                    'confidence': round((1 - std_pred) * 100, 1),
                    'uncertainty': round(std_pred, 2),
                    'prediction_interval': [
                        round(np.percentile(predictions, 2.5), 2),
                        round(np.percentile(predictions, 97.5), 2)
                    ],
                    'trend': self._get_trend(mean_pred),
                    'model_type': 'bnn',
                    'model_name': self.MODEL_NAME
                }
            else:
                pred = self.model.predict(features_scaled)[0]
                return {
                    'predicted_value': round(float(pred), 2),
                    'unit': 'ng/mL',
                    'confidence': 70.0,
                    'uncertainty': 0.5,
                    'prediction_interval': [round(float(pred) * 0.8, 2), round(float(pred) * 1.2, 2)],
                    'trend': 'Stable',
                    'model_type': 'bnn_sklearn',
                    'model_name': self.MODEL_NAME,
                    'note': 'Using sklearn approximation'
                }
                
        except Exception as e:
            logger.error(f"Hormone prediction error: {e}")
            return self._fallback_prediction(data)
    
    def _get_trend(self, value: float) -> str:
        """Determine trend direction."""
        if value < 0.3:
            return "Declining"
        elif value > 0.7:
            return "Rising"
        else:
            return "Stable"
    
    def _fallback_prediction(self, data: Dict) -> Dict:
        """Fallback hormone prediction."""
        return {
            'predicted_value': 2.0,
            'unit': 'ng/mL',
            'confidence': 50.0,
            'uncertainty': 1.0,
            'prediction_interval': [1.0, 3.0],
            'trend': 'Stable',
            'model_type': 'fallback',
            'model_name': self.MODEL_NAME,
            'note': 'Model not trained - using rule-based fallback'
        }


class BNNCycleOutcomePredictor:
    """
    Bayesian Neural Network for cycle outcome prediction.
    Predicts probability of success with confidence intervals.
    """
    
    MODEL_NAME = "cycle_outcome_bnn"
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['age', 'amh', 'num_eggs', 'fertilization_rate',
                           'embryos_day5', 'embryo_grade', 'transfer_day']
        self._load_model()
    
    def _load_model(self):
        model_path = MODEL_DIR / f"{self.MODEL_NAME}.pkl"
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.is_trained = True
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
    
    def predict(self, data: Dict, n_samples: int = 100) -> Dict:
        """
        Predict cycle outcome with uncertainty.
        
        Args:
            data: Cycle data
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with outcome probabilities and confidence
        """
        if self.model is None or not self.is_trained:
            return self._fallback_prediction(data)
        
        try:
            features = self._prepare_features(data)
            features_scaled = self.scaler.transform(features)
            
            if TORCH_AVAILABLE:
                features_tensor = torch.FloatTensor(features_scaled)
                
                predictions = []
                self.model.train()
                
                for _ in range(n_samples):
                    with torch.no_grad():
                        pred = self.model(features_tensor).numpy()[0, 0]
                        predictions.append(pred)
                
                predictions = np.array(predictions)
                
                success_prob = float(np.mean(predictions))
                std_pred = float(np.std(predictions))
                
                return {
                    'success_probability': round(success_prob * 100, 1),
                    'confidence': round((1 - std_pred) * 100, 1),
                    'uncertainty': round(std_pred * 100, 1),
                    'prediction_interval': [
                        round(np.percentile(predictions, 2.5) * 100, 1),
                        round(np.percentile(predictions, 97.5) * 100, 1)
                    ],
                    'model_type': 'bnn',
                    'model_name': self.MODEL_NAME
                }
            else:
                prob = self.model.predict_proba(features_scaled)[0, 1]
                return {
                    'success_probability': round(prob * 100, 1),
                    'confidence': 70.0,
                    'uncertainty': 15.0,
                    'prediction_interval': [round(prob * 80 * 100, 1), round(prob * 120 * 100, 1)],
                    'model_type': 'bnn_sklearn',
                    'model_name': self.MODEL_NAME,
                    'note': 'Using sklearn approximation'
                }
                
        except Exception as e:
            logger.error(f"Cycle outcome prediction error: {e}")
            return self._fallback_prediction(data)
    
    def _fallback_prediction(self, data: Dict) -> Dict:
        """Fallback cycle outcome prediction."""
        return {
            'success_probability': 35.0,
            'confidence': 50.0,
            'uncertainty': 20.0,
            'prediction_interval': [15.0, 55.0],
            'model_type': 'fallback',
            'model_name': self.MODEL_NAME,
            'note': 'Model not trained - using rule-based fallback'
        }


class BNNModelTrainer:
    """
    Trainer class for Bayesian Neural Network models.
    """
    
    @staticmethod
    def train_bnn_risk_model(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            feature_names: List[str]) -> Tuple[Any, Dict]:
        """Train BNN risk assessment model."""
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if TORCH_AVAILABLE:
            # Create PyTorch BNN
            model = BayesianNeuralNetwork(
                input_dim=X_train.shape[1],
                output_dim=1,
                hidden_dim=64
            )
            
            # Convert data to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
            
            # Training
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            for epoch in range(100):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            
            with torch.no_grad():
                predictions = model(X_test_tensor).numpy()
            
            from sklearn.metrics import roc_auc_score
            
            metrics = {
                'roc_auc': float(roc_auc_score(y_test, predictions)),
                'feature_count': len(feature_names)
            }
        else:
            # Use sklearn LogisticRegression as approximation
            from sklearn.linear_model import LogisticRegression
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            from sklearn.metrics import roc_auc_score
            
            predictions = model.predict_proba(X_test_scaled)[:, 1]
            
            metrics = {
                'roc_auc': float(roc_auc_score(y_test, predictions)),
                'feature_count': len(feature_names)
            }
            
            # Save scaler with model
            joblib.dump((model, scaler), MODEL_DIR / "risk_assessment_bnn.pkl")
            logger.info("Using sklearn approximation for BNN")
            
            return model, metrics
        
        # Save model and scaler
        joblib.dump((model, scaler), MODEL_DIR / "risk_assessment_bnn.pkl")
        
        # Save metadata
        metadata = {
            'model_name': 'risk_assessment_bnn',
            'model_type': 'bnn',
            'feature_names': feature_names,
            'metrics': metrics,
            'version': '1.0.0'
        }
        
        metadata_path = MODEL_DIR / "risk_assessment_bnn_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"BNN risk model trained: {MODEL_DIR / 'risk_assessment_bnn.pkl'}")
        
        return model, metrics
    
    @staticmethod
    def train_bnn_hormone_model(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               feature_names: List[str]) -> Tuple[Any, Dict]:
        """Train BNN hormone prediction model."""
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if TORCH_AVAILABLE:
            model = BayesianNeuralNetwork(
                input_dim=X_train.shape[1],
                output_dim=1,
                hidden_dim=64
            )
            
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            for epoch in range(100):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            
            from sklearn.metrics import mean_squared_error, r2_score
            
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            with torch.no_grad():
                predictions = model(X_test_tensor).numpy()
            
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
                'r2': float(r2_score(y_test, predictions))
            }
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            
            model = GradientBoostingRegressor(random_state=42, n_estimators=100)
            model.fit(X_train_scaled, y_train)
            
            predictions = model.predict(X_test_scaled)
            
            from sklearn.metrics import mean_squared_error, r2_score
            
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
                'r2': float(r2_score(y_test, predictions))
            }
        
        joblib.dump((model, scaler), MODEL_DIR / "hormone_prediction_bnn.pkl")
        
        logger.info(f"BNN hormone model trained")
        
        return model, metrics


# Convenience functions
def get_risk_assessor() -> BNNRiskAssessor:
    """Get risk assessor instance."""
    return BNNRiskAssessor()


def get_hormone_predictor() -> BNNHormonePredictor:
    """Get hormone predictor instance."""
    return BNNHormonePredictor()


def get_cycle_outcome_predictor() -> BNNCycleOutcomePredictor:
    """Get cycle outcome predictor instance."""
    return BNNCycleOutcomePredictor()


def assess_risk(data: Dict) -> Dict:
    """Convenience function for risk assessment."""
    assessor = BNNRiskAssessor()
    return assessor.predict(data)


def predict_hormone(data: Dict) -> Dict:
    """Convenience function for hormone prediction."""
    predictor = BNNHormonePredictor()
    return predictor.predict(data)


def predict_cycle_outcome(data: Dict) -> Dict:
    """Convenience function for cycle outcome prediction."""
    predictor = BNNCycleOutcomePredictor()
    return predictor.predict(data)
