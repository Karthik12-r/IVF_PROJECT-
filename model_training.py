"""
Model Training Pipeline for IVF Journey Tracker.
Trains and evaluates all ML models using the IVF dataset.

Dataset split:
    70% Training
    15% Validation
    15% Testing
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    mean_squared_error,
    r2_score
)

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

    Dataset:
        70% Training
        15% Validation
        15% Testing
    """

    def __init__(
        self,
        test_size: float = 0.15,
        validation_size: float = 0.15,
        random_state: int = 42
    ):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

        self.models_trained = {}
        self.metrics = {}

    # ============================================================
    # LOAD DATASET
    # ============================================================

    def load_dataset(self) -> pd.DataFrame:
        """Load the IVF dataset."""

        try:
            df = pd.read_excel(DATA_PATH)

            logger.info(
                f"Loaded dataset with shape: {df.shape}"
            )

            return df

        except Exception as e:

            logger.error(
                f"Error loading dataset: {e}"
            )

            # Create sample data if dataset is not found
            return self._create_sample_data()

    # ============================================================
    # CREATE SAMPLE DATA
    # ============================================================

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for training."""

        np.random.seed(self.random_state)

        n_samples = 5000

        data = {
            'age': np.random.randint(
                22, 45, n_samples
            ),

            'bmi': np.random.uniform(
                18, 35, n_samples
            ),

            'amh': np.random.uniform(
                0.5, 8.0, n_samples
            ),

            'fsh': np.random.uniform(
                4, 15, n_samples
            ),

            'previous_ivf': np.random.randint(
                0, 5, n_samples
            ),

            'stress': np.random.randint(
                1, 10, n_samples
            ),

            'sleep_hours': np.random.uniform(
                5, 10, n_samples
            ),

            'exercise_min': np.random.randint(
                0, 120, n_samples
            ),

            'IVF_Success': np.random.randint(
                0, 2, n_samples
            )
        }

        df = pd.DataFrame(data)

        # Success probability based on features
        success_prob = (
            0.3
            + (df['age'] < 35).astype(float) * 0.2
            + (df['amh'] > 2).astype(float) * 0.15
            + (df['fsh'] < 10).astype(float) * 0.1
            + (df['bmi'] < 25).astype(float) * 0.1
            + np.random.uniform(
                -0.1,
                0.1,
                n_samples
            )
        )

        df['IVF_Success'] = (
            success_prob > 0.5
        ).astype(int)

        logger.info(
            "Created sample dataset"
        )

        return df

    # ============================================================
    # PREPARE FEATURES
    # ============================================================

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'IVF_Success'
    ) -> Tuple:
        """
        Prepare features and split dataset into:

        70% Training
        15% Validation
        15% Testing
        """

        # Separate features and target
        X = df.drop(
            columns=[target_col]
        )

        y = df[target_col]

        feature_names = list(
            X.columns
        )

        # --------------------------------------------------------
        # STEP 1
        # Separate 15% TEST data
        # --------------------------------------------------------

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # --------------------------------------------------------
        # STEP 2
        # Separate 15% VALIDATION data
        #
        # Remaining data = 85%
        #
        # 15 / 85 = 0.17647
        # --------------------------------------------------------

        validation_ratio = (
            self.validation_size /
            (1 - self.test_size)
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=validation_ratio,
            random_state=self.random_state,
            stratify=y_temp
        )

        # --------------------------------------------------------
        # LOG DATASET SIZES
        # --------------------------------------------------------

        logger.info("=" * 60)
        logger.info("DATASET SPLIT")
        logger.info("=" * 60)

        logger.info(
            f"Total samples      : {len(X)}"
        )

        logger.info(
            f"Training samples   : {len(X_train)} "
            f"({len(X_train) / len(X) * 100:.2f}%)"
        )

        logger.info(
            f"Validation samples : {len(X_val)} "
            f"({len(X_val) / len(X) * 100:.2f}%)"
        )

        logger.info(
            f"Testing samples    : {len(X_test)} "
            f"({len(X_test) / len(X) * 100:.2f}%)"
        )

        logger.info("=" * 60)

        return (
            X_train.values,
            X_val.values,
            X_test.values,
            y_train.values,
            y_val.values,
            y_test.values,
            feature_names
        )

    # ============================================================
    # XGBOOST
    # ============================================================

    def train_xgboost_models(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names
    ) -> Dict:
        """Train XGBoost models."""

        logger.info(
            "Training XGBoost models..."
        )

        # IVF Success Model
        model, metrics = (
            XGBoostModelTrainer.train_ivf_success_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                feature_names
            )
        )

        self.models_trained[
            'ivf_success_xgboost'
        ] = model

        self.metrics[
            'ivf_success_xgboost'
        ] = metrics

        logger.info(
            "IVF Success XGBoost - "
            f"Test Accuracy: "
            f"{metrics.get('test_accuracy', metrics.get('accuracy', 0)):.4f}"
        )

        if 'validation_accuracy' in metrics:

            logger.info(
                "IVF Success XGBoost - "
                f"Validation Accuracy: "
                f"{metrics['validation_accuracy']:.4f}"
            )

        return metrics

    # ============================================================
    # LIGHTGBM
    # ============================================================

    def train_lightgbm_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names
    ) -> Dict:
        """Train LightGBM models."""

        logger.info(
            "Training LightGBM models..."
        )

        # Create synthetic classification labels
        # 0: Low Responder
        # 1: Normal Responder
        # 2: High Responder

        amh_threshold_low = 1.0
        amh_threshold_high = 4.0

        X_with_amh = X_train.copy()

        amh_idx = (
            feature_names.index('amh')
            if 'amh' in feature_names
            else 2
        )

        y_classification = np.zeros(
            len(y_train)
        )

        for i in range(len(y_train)):

            amh_val = X_with_amh[
                i,
                amh_idx
            ]

            if amh_val < amh_threshold_low:

                y_classification[i] = 0

            elif amh_val > amh_threshold_high:

                y_classification[i] = 2

            else:

                y_classification[i] = 1

        try:

            model, metrics = (
                LightGBMModelTrainer
                .train_patient_classification(
                    X_train,
                    y_classification,
                    X_test,
                    y_test,
                    feature_names
                )
            )

            self.models_trained[
                'patient_classification_lgbm'
            ] = model

            self.metrics[
                'patient_classification_lgbm'
            ] = metrics

            logger.info(
                "Patient Classification - "
                f"Accuracy: "
                f"{metrics.get('accuracy', 0):.4f}"
            )

        except Exception as e:

            logger.error(
                f"Error training patient classification: {e}"
            )

        return self.metrics.get(
            'patient_classification_lgbm',
            {}
        )

    # ============================================================
    # BNN
    # ============================================================

    def train_bnn_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names
    ) -> Dict:
        """Train BNN models."""

        logger.info(
            "Training BNN models..."
        )

        # Create synthetic risk labels
        # Based on age and AMH

        age_idx = (
            feature_names.index('age')
            if 'age' in feature_names
            else 0
        )

        amh_idx = (
            feature_names.index('amh')
            if 'amh' in feature_names
            else 2
        )

        y_risk = np.zeros(
            len(y_train)
        )

        for i in range(len(y_train)):

            age = X_train[
                i,
                age_idx
            ]

            amh = X_train[
                i,
                amh_idx
            ]

            risk = 0.1

            if amh > 4.0:
                risk += 0.3

            if age < 30:
                risk += 0.2

            if amh < 1.0:
                risk += 0.2

            y_risk[i] = (
                1 if risk > 0.3
                else 0
            )

        try:

            model, metrics = (
                BNNModelTrainer.train_bnn_risk_model(
                    X_train,
                    y_risk,
                    X_test,
                    y_test,
                    feature_names
                )
            )

            self.models_trained[
                'risk_assessment_bnn'
            ] = model

            self.metrics[
                'risk_assessment_bnn'
            ] = metrics

            logger.info(
                "BNN Risk model trained"
            )

        except Exception as e:

            logger.error(
                f"Error training BNN risk model: {e}"
            )

        return self.metrics.get(
            'risk_assessment_bnn',
            {}
        )

    # ============================================================
    # VALIDATION EVALUATION
    # ============================================================

    def evaluate_validation_set(
        self,
        X_val,
        y_val
    ):
        """
        Evaluate trained models using
        the validation dataset.
        """

        logger.info("=" * 60)
        logger.info("VALIDATION SET EVALUATION")
        logger.info("=" * 60)

        for model_name, model in (
            self.models_trained.items()
        ):

            try:

                y_val_pred = model.predict(
                    X_val
                )

                validation_accuracy = (
                    accuracy_score(
                        y_val,
                        y_val_pred
                    )
                )

                self.metrics.setdefault(
                    model_name,
                    {}
                )

                self.metrics[
                    model_name
                ][
                    'validation_accuracy'
                ] = float(
                    validation_accuracy
                )

                logger.info(
                    f"{model_name} - "
                    f"Validation Accuracy: "
                    f"{validation_accuracy:.4f}"
                )

            except Exception as e:

                logger.warning(
                    f"Could not evaluate "
                    f"{model_name} on validation set: "
                    f"{e}"
                )

    # ============================================================
    # TRAIN ALL MODELS
    # ============================================================

    def train_all_models(self) -> Dict:
        """Train all models in the pipeline."""

        logger.info("=" * 60)
        logger.info(
            "Starting Model Training Pipeline"
        )
        logger.info("=" * 60)

        # --------------------------------------------------------
        # Load data
        # --------------------------------------------------------

        df = self.load_dataset()

        # --------------------------------------------------------
        # Prepare features
        # --------------------------------------------------------

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            feature_names
        ) = self.prepare_features(df)

        # --------------------------------------------------------
        # Train XGBoost
        # --------------------------------------------------------

        xgb_metrics = (
            self.train_xgboost_models(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                feature_names
            )
        )

        # --------------------------------------------------------
        # Train LightGBM
        # --------------------------------------------------------

        lgbm_metrics = (
            self.train_lightgbm_models(
                X_train,
                X_test,
                y_train,
                y_test,
                feature_names
            )
        )

        # --------------------------------------------------------
        # Train BNN
        # --------------------------------------------------------

        bnn_metrics = (
            self.train_bnn_models(
                X_train,
                X_test,
                y_train,
                y_test,
                feature_names
            )
        )

        # --------------------------------------------------------
        # Validation evaluation
        # --------------------------------------------------------

        self.evaluate_validation_set(
            X_val,
            y_val
        )

        logger.info("=" * 60)
        logger.info(
            "Model Training Complete"
        )
        logger.info("=" * 60)

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        summary = {
            'xgboost_models': self.metrics.get(
                'ivf_success_xgboost',
                xgb_metrics
            ),

            'lightgbm_models': self.metrics.get(
                'patient_classification_lgbm',
                lgbm_metrics
            ),

            'bnn_models': self.metrics.get(
                'risk_assessment_bnn',
                bnn_metrics
            ),

            'total_models_trained':
                len(self.models_trained),

            'dataset_split': {
                'training': 0.70,
                'validation': 0.15,
                'testing': 0.15
            }
        }

        return summary

    # ============================================================
    # EVALUATE MODEL
    # ============================================================

    def evaluate_model(
        self,
        model_name: str,
        X_test,
        y_test
    ) -> Dict:
        """Evaluate a trained model."""

        if model_name not in self.models_trained:

            return {
                'error': 'Model not found'
            }

        model = self.models_trained[
            model_name
        ]

        y_pred = model.predict(
            X_test
        )

        metrics = {
            'accuracy': accuracy_score(
                y_test,
                y_pred
            ),

            'classification_report':
                classification_report(
                    y_test,
                    y_pred,
                    output_dict=True
                )
        }

        return metrics


# ================================================================
# CONVENIENCE FUNCTION
# ================================================================

def train_all_models():
    """Convenience function to train all models."""

    pipeline = ModelTrainingPipeline()

    return pipeline.train_all_models()


# ================================================================
# MAIN
# ================================================================

def main():
    """Main training function."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            '%(asctime)s - '
            '%(name)s - '
            '%(levelname)s - '
            '%(message)s'
        )
    )

    # Run training
    results = train_all_models()

    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)

    for model_name, metrics in results.items():

        if model_name not in [
            'total_models_trained',
            'dataset_split'
        ]:

            print(
                f"\n{model_name}:"
            )

            if isinstance(metrics, dict):

                for metric_name, value in metrics.items():

                    if isinstance(
                        value,
                        (float, int)
                    ):

                        print(
                            f"  {metric_name}: "
                            f"{value:.4f}"
                        )

    print(
        f"\nTotal models trained: "
        f"{results.get('total_models_trained', 0)}"
    )

    print(
        "\nDataset split:"
    )

    print(
        "  Training   : 70%"
    )

    print(
        "  Validation : 15%"
    )

    print(
        "  Testing    : 15%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()