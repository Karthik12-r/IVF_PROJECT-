"""
Model Training Pipeline for IVF Journey Tracker.
Trains and evaluates all ML models using the IVF dataset.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, mean_squared_error, r2_score

# Import model trainers
from xgboost_models import XGBoostModelTrainer
from lightgbm_models import LightGBMModelTrainer
from bnn_models import BNNModelTrainer

logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# Dataset path
DATA_PATH = Path(__file__).parent.parent / "ivf_5000_dataset.xlsx"


class ModelTrainingPipeline:
    """
    Centralized training pipeline for all IVF ML models.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.models_trained = {}
        self.metrics = {}
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the IVF dataset."""
        try:
            df = pd.read_excel(DATA_PATH)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Create sample data if dataset not found
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for training."""
        np.random.seed(self.random_state)
        n_samples = 5000
        
        data = {
            'age': np.random.randint(22, 45, n_samples),
            'bmi': np.random.uniform(18, 35, n_samples),
            'amh': np.random.uniform(0.5, 8.0, n_samples),
            'fsh': np.random.uniform(4, 15, n_samples),
            'previous_ivf': np.random.randint(0, 5, n_samples),
            'stress': np.random.randint(1, 10, n_samples),
            'sleep_hours': np.random.uniform(5, 10, n_samples),
            'exercise_min': np.random.randint(0, 120, n_samples),
            'IVF_Success': np.random.randint(0, 2, n_samples)
        }
        
        # Add some correlation between features and outcome
        df = pd.DataFrame(data)
        
        # Success probability based on features
        success_prob = (
            0.3 +
            (df['age'] < 35).astype(float) * 0.2 +
            (df['amh'] > 2).astype(float) * 0.15 +
            (df['fsh'] < 10).astype(float) * 0.1 +
            (df['bmi'] < 25).astype(float) * 0.1 +
            np.random.uniform(-0.1, 0.1, n_samples)
        )
        
        df['IVF_Success'] = (success_prob > 0.5).astype(int)
        
        logger.info("Created sample dataset")
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'IVF_Success') -> Tuple:
        """Prepare features and target for training."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        feature_names = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train.values, X_test.values, y_train.values, y_test.values, feature_names
    
    def train_xgboost_models(self, X_train, X_test, y_train, y_test, feature_names) -> Dict:
        """Train XGBoost models."""
        logger.info("Training XGBoost models...")
        
        # IVF Success Model
        model, metrics = XGBoostModelTrainer.train_ivf_success_model(
            X_train, y_train, X_test, y_test, feature_names
        )
        
        self.models_trained['ivf_success_xgboost'] = model
        self.metrics['ivf_success_xgboost'] = metrics
        
        logger.info(f"IVF Success XGBoost - Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return metrics
    
    def train_lightgbm_models(self, X_train, X_test, y_train, y_test, feature_names) -> Dict:
        """Train LightGBM models."""
        logger.info("Training LightGBM models...")
        
        # Create synthetic classification labels for patient classification
        # 0: Low Responder, 1: Normal Responder, 2: High Responder
        amh_threshold_low = 1.0
        amh_threshold_high = 4.0
        
        # Use AMH-based classification
        X_with_amh = X_train.copy()
        amh_idx = feature_names.index('amh') if 'amh' in feature_names else 2
        
        y_classification = np.zeros(len(y_train))
        for i in range(len(y_train)):
            amh_val = X_with_amh[i, amh_idx]
            if amh_val < amh_threshold_low:
                y_classification[i] = 0  # Low
            elif amh_val > amh_threshold_high:
                y_classification[i] = 2  # High
            else:
                y_classification[i] = 1  # Normal
        
        # Train patient classification
        try:
            model, metrics = LightGBMModelTrainer.train_patient_classification(
                X_train, y_classification, X_test, y_test, feature_names
            )
            self.models_trained['patient_classification_lgbm'] = model
            self.metrics['patient_classification_lgbm'] = metrics
            logger.info(f"Patient Classification - Accuracy: {metrics.get('accuracy', 0):.4f}")
        except Exception as e:
            logger.error(f"Error training patient classification: {e}")
        
        return self.metrics.get('patient_classification_lgbm', {})
    
    def train_bnn_models(self, X_train, X_test, y_train, y_test, feature_names) -> Dict:
        """Train BNN models."""
        logger.info("Training BNN models...")
        
        # For risk assessment, create synthetic risk labels
        # Based on age and AMH
        age_idx = feature_names.index('age') if 'age' in feature_names else 0
        amh_idx = feature_names.index('amh') if 'amh' in feature_names else 2
        
        y_risk = np.zeros(len(y_train))
        for i in range(len(y_train)):
            age = X_train[i, age_idx]
            amh = X_train[i, amh_idx]
            
            risk = 0.1
            if amh > 4.0:
                risk += 0.3
            if age < 30:
                risk += 0.2
            if amh < 1.0:
                risk += 0.2
            
            y_risk[i] = 1 if risk > 0.3 else 0
        
        try:
            model, metrics = BNNModelTrainer.train_bnn_risk_model(
                X_train, y_risk, X_test, y_test, feature_names
            )
            self.models_trained['risk_assessment_bnn'] = model
            self.metrics['risk_assessment_bnn'] = metrics
            logger.info("BNN Risk model trained")
        except Exception as e:
            logger.error(f"Error training BNN risk model: {e}")
        
        return self.metrics.get('risk_assessment_bnn', {})
    
    def train_all_models(self) -> Dict:
        """Train all models in the pipeline."""
        logger.info("=" * 50)
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 50)
        
        # Load data
        df = self.load_dataset()
        
        # Prepare features
        X_train, X_test, y_train, y_test, feature_names = self.prepare_features(df)
        
        # Train models
        xgb_metrics = self.train_xgboost_models(X_train, X_test, y_train, y_test, feature_names)
        lgbm_metrics = self.train_lightgbm_models(X_train, X_test, y_train, y_test, feature_names)
        bnn_metrics = self.train_bnn_models(X_train, X_test, y_train, y_test, feature_names)
        
        logger.info("=" * 50)
        logger.info("Model Training Complete")
        logger.info("=" * 50)
        
        # Summary
        summary = {
            'xgboost_models': xgb_metrics,
            'lightgbm_models': lgbm_metrics,
            'bnn_models': bnn_metrics,
            'total_models_trained': len(self.models_trained)
        }
        
        return summary
    
    def evaluate_model(self, model_name: str, X_test, y_test) -> Dict:
        """Evaluate a trained model."""
        if model_name not in self.models_trained:
            return {'error': 'Model not found'}
        
        model = self.models_trained[model_name]
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics


def train_all_models():
    """Convenience function to train all models."""
    pipeline = ModelTrainingPipeline()
    return pipeline.train_all_models()


def main():
    """Main training function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run training
    results = train_all_models()
    
    print("\n" + "=" * 50)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        if model_name != 'total_models_trained':
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
    
    print(f"\nTotal models trained: {results.get('total_models_trained', 0)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
