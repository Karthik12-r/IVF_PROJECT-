"""
XGBoost Models for IVF Journey Tracker.
Implements high-accuracy tabular predictions for IVF success
and related tasks.

Dataset split for IVF Success model:
    70% Training
    15% Validation
    15% Testing
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# DEFAULT FEATURE NAMES
# ============================================================

DEFAULT_FEATURES = [
    'age',
    'bmi',
    'amh',
    'fsh',
    'previous_ivf',
    'stress',
    'sleep_hours',
    'exercise_min'
]


# ============================================================
# IVF SUCCESS PREDICTOR
# ============================================================

class IVFSuccessPredictor:
    """
    XGBoost-based IVF success prediction model.

    Predicts probability of pregnancy success
    based on patient data.
    """

    MODEL_NAME = "ivf_success_xgboost"
    MODEL_TYPE = "xgboost_classifier"

    def __init__(self):

        self.model = None
        self.feature_names = DEFAULT_FEATURES
        self.metadata = {}

        self._load_model()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    def _load_model(self):
        """Load model from disk if it exists."""

        model_path = MODEL_DIR / f"{self.MODEL_NAME}.pkl"

        if model_path.exists():

            try:

                self.model = joblib.load(
                    model_path
                )

                metadata_path = (
                    MODEL_DIR /
                    f"{self.MODEL_NAME}_metadata.json"
                )

                if metadata_path.exists():

                    with open(
                        metadata_path,
                        'r'
                    ) as f:

                        self.metadata = json.load(f)

                    self.feature_names = (
                        self.metadata.get(
                            'feature_names',
                            DEFAULT_FEATURES
                        )
                    )

                logger.info(
                    f"Loaded {self.MODEL_NAME} model"
                )

            except Exception as e:

                logger.error(
                    f"Error loading model: {e}"
                )

                self.model = None

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    def _prepare_features(
        self,
        patient_data
    ) -> np.ndarray:
        """Prepare feature vector from patient data."""

        features = []

        for feature in self.feature_names:

            if hasattr(
                patient_data,
                feature
            ):

                value = getattr(
                    patient_data,
                    feature,
                    None
                )

            elif isinstance(
                patient_data,
                dict
            ):

                value = patient_data.get(
                    feature
                )

            else:

                value = None

            # Default values
            if value is None:

                defaults = {
                    'age': 35,
                    'bmi': 24,
                    'amh': 2.0,
                    'fsh': 7.0,
                    'previous_ivf': 0,
                    'stress': 3,
                    'sleep_hours': 7,
                    'exercise_min': 30
                }

                value = defaults.get(
                    feature,
                    0
                )

            features.append(
                float(value)
            )

        return np.array(
            features
        ).reshape(1, -1)

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(
        self,
        patient_data
    ) -> Dict:
        """
        Predict IVF success probability.

        Args:
            patient_data:
                PatientData object or dictionary.

        Returns:
            Dictionary containing prediction,
            probability and confidence.
        """

        if self.model is None:

            return self._fallback_prediction(
                patient_data
            )

        try:

            features = (
                self._prepare_features(
                    patient_data
                )
            )

            # Prediction
            prediction = (
                self.model.predict(
                    features
                )[0]
            )

            # Probability
            proba = (
                self.model.predict_proba(
                    features
                )[0]
            )

            success_prob = float(
                proba[1]
            )

            # Confidence
            confidence = (
                self._calculate_confidence(
                    proba
                )
            )

            # Feature importance
            importance = (
                self._get_feature_importance()
            )

            return {
                'prediction': int(
                    prediction
                ),

                'success_probability':
                    round(
                        success_prob * 100,
                        1
                    ),

                'confidence':
                    confidence,

                'prediction_label':
                    (
                        'Success'
                        if prediction == 1
                        else 'No Success'
                    ),

                'feature_importance':
                    importance,

                'model_type':
                    self.MODEL_TYPE,

                'model_name':
                    self.MODEL_NAME
            }

        except Exception as e:

            logger.error(
                f"Prediction error: {e}"
            )

            return self._fallback_prediction(
                patient_data
            )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    def _calculate_confidence(
        self,
        proba: np.ndarray
    ) -> float:
        """Calculate confidence score."""

        margin = abs(
            proba[0] - proba[1]
        )

        confidence = min(
            95,
            60 + margin * 35
        )

        return round(
            confidence,
            1
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    def _get_feature_importance(
        self
    ) -> Dict[str, float]:
        """Get feature importance scores."""

        if (
            self.model is None
            or not hasattr(
                self.model,
                'feature_importances_'
            )
        ):

            return {}

        importance = (
            self.model.feature_importances_
        )

        return {
            name: round(
                float(imp),
                4
            )
            for name, imp in zip(
                self.feature_names,
                importance
            )
        }

    # --------------------------------------------------------
    # FALLBACK PREDICTION
    # --------------------------------------------------------

    def _fallback_prediction(
        self,
        patient_data
    ) -> Dict:
        """Fallback prediction when model is unavailable."""

        age = getattr(
            patient_data,
            'age',
            35
        ) or 35

        amh = getattr(
            patient_data,
            'amh_level',
            None
        )

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

        success_prob = max(
            10,
            min(
                80,
                base_prob
            )
        )

        return {
            'prediction':
                1 if success_prob >= 50 else 0,

            'success_probability':
                round(
                    success_prob,
                    1
                ),

            'confidence':
                50.0,

            'prediction_label':
                (
                    'Success'
                    if success_prob >= 50
                    else 'No Success'
                ),

            'feature_importance':
                {},

            'model_type':
                'fallback',

            'model_name':
                self.MODEL_NAME,

            'note':
                'Model not trained yet'
        }


# ============================================================
# EMBRYO QUALITY PREDICTOR
# ============================================================

class EmbryoQualityPredictor:
    """
    XGBoost-based embryo quality prediction.

    Predicts embryo quality scores based on
    patient and cycle data.
    """

    MODEL_NAME = "embryo_quality_xgboost"

    def __init__(self):

        self.model = None

        self.feature_names = [
            'age',
            'amh',
            'bmi',
            'fsh',
            'egg_quality_score',
            'sperm_morphology',
            'fertilization_method'
        ]

        self._load_model()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    def _load_model(self):
        """Load model from disk if it exists."""

        model_path = (
            MODEL_DIR /
            f"{self.MODEL_NAME}.pkl"
        )

        if model_path.exists():

            try:

                self.model = joblib.load(
                    model_path
                )

                logger.info(
                    f"Loaded {self.MODEL_NAME} model"
                )

            except Exception as e:

                logger.error(
                    f"Error loading embryo model: {e}"
                )

                self.model = None

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    def _prepare_features(
        self,
        data
    ) -> np.ndarray:
        """Prepare feature vector."""

        features = []

        for feature in self.feature_names:

            if hasattr(
                data,
                feature
            ):

                value = getattr(
                    data,
                    feature,
                    None
                )

            elif isinstance(
                data,
                dict
            ):

                value = data.get(
                    feature
                )

            else:

                value = None

            if value is None:

                value = 0

            features.append(
                float(value)
            )

        return np.array(
            features
        ).reshape(1, -1)

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(
        self,
        data
    ) -> Dict:
        """
        Predict embryo quality.
        """

        if self.model is None:

            return self._fallback_prediction(
                data
            )

        try:

            features = (
                self._prepare_features(
                    data
                )
            )

            quality_score = (
                self.model.predict(
                    features
                )[0]
            )

            if quality_score >= 80:

                grade = "A (Excellent)"

            elif quality_score >= 65:

                grade = "B (Good)"

            elif quality_score >= 45:

                grade = "C (Fair)"

            else:

                grade = "D (Poor)"

            return {
                'quality_score':
                    round(
                        float(quality_score),
                        1
                    ),

                'grade':
                    grade,

                'development_probability':
                    round(
                        min(
                            90,
                            quality_score * 0.9
                        ),
                        1
                    ),

                'implantation_potential':
                    round(
                        min(
                            85,
                            quality_score * 0.8
                        ),
                        1
                    )
            }

        except Exception as e:

            logger.error(
                f"Prediction error: {e}"
            )

            return self._fallback_prediction(
                data
            )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    def _fallback_prediction(
        self,
        data
    ) -> Dict:
        """Fallback prediction."""

        return {
            'quality_score':
                65.0,

            'grade':
                'B (Good)',

            'development_probability':
                58.5,

            'implantation_potential':
                52.0,

            'note':
                'Model not trained yet'
        }


# ============================================================
# XGBOOST MODEL TRAINER
# ============================================================

class XGBoostModelTrainer:
    """
    Trainer class for XGBoost models.
    """

    # ========================================================
    # IVF SUCCESS MODEL
    # ========================================================

    @staticmethod
    def train_ivf_success_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[XGBClassifier, Dict]:
        """
        Train IVF success prediction model.

        Dataset usage:

            X_train / y_train
                -> Model training

            X_val / y_val
                -> Validation / model monitoring

            X_test / y_test
                -> Final evaluation

        Args:
            X_train: Training features
            y_train: Training labels

            X_val: Validation features
            y_val: Validation labels

            X_test: Test features
            y_test: Test labels

            feature_names:
                List of feature names

        Returns:
            Tuple of trained model and metrics.
        """

        logger.info(
            "Starting XGBoost IVF success training"
        )

        # ----------------------------------------------------
        # CREATE MODEL
        # ----------------------------------------------------

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        # ----------------------------------------------------
        # TRAIN MODEL
        #
        # Validation set is supplied through eval_set.
        # ----------------------------------------------------

        logger.info(
            "Training XGBoost model..."
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val)
            ],
            verbose=False
        )

        logger.info(
            "XGBoost model training completed"
        )

        # ----------------------------------------------------
        # IMPORT METRICS
        # ----------------------------------------------------

        from sklearn.metrics import (
            accuracy_score,
            roc_auc_score,
            classification_report
        )

        # ====================================================
        # VALIDATION EVALUATION
        # ====================================================

        y_val_pred = model.predict(
            X_val
        )

        y_val_proba = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        validation_accuracy = (
            accuracy_score(
                y_val,
                y_val_pred
            )
        )

        validation_roc_auc = (
            roc_auc_score(
                y_val,
                y_val_proba
            )
        )

        validation_report = (
            classification_report(
                y_val,
                y_val_pred,
                output_dict=True
            )
        )

        # ====================================================
        # FINAL TEST EVALUATION
        # ====================================================

        y_test_pred = model.predict(
            X_test
        )

        y_test_proba = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        test_accuracy = (
            accuracy_score(
                y_test,
                y_test_pred
            )
        )

        test_roc_auc = (
            roc_auc_score(
                y_test,
                y_test_proba
            )
        )

        test_report = (
            classification_report(
                y_test,
                y_test_pred,
                output_dict=True
            )
        )

        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        feature_importance = {
            name: float(importance)
            for name, importance in zip(
                feature_names,
                model.feature_importances_
            )
        }

        # ====================================================
        # METRICS
        # ====================================================

        metrics = {

            # Validation metrics
            'validation_accuracy':
                float(
                    validation_accuracy
                ),

            'validation_roc_auc':
                float(
                    validation_roc_auc
                ),

            'validation_classification_report':
                validation_report,

            # Test metrics
            'test_accuracy':
                float(
                    test_accuracy
                ),

            'test_roc_auc':
                float(
                    test_roc_auc
                ),

            'test_classification_report':
                test_report,

            # Feature importance
            'feature_importance':
                feature_importance
        }

        # ----------------------------------------------------
        # LOG RESULTS
        # ----------------------------------------------------

        logger.info(
            "=" * 60
        )

        logger.info(
            "XGBOOST IVF SUCCESS RESULTS"
        )

        logger.info(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

        logger.info(
            f"Validation ROC-AUC: "
            f"{validation_roc_auc:.4f}"
        )

        logger.info(
            f"Test Accuracy: "
            f"{test_accuracy:.4f}"
        )

        logger.info(
            f"Test ROC-AUC: "
            f"{test_roc_auc:.4f}"
        )

        logger.info(
            "=" * 60
        )

        # ====================================================
        # SAVE MODEL
        # ====================================================

        model_path = (
            MODEL_DIR /
            "ivf_success_xgboost.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        # ====================================================
        # SAVE METADATA
        # ====================================================

        metadata = {

            'model_name':
                'ivf_success_xgboost',

            'model_type':
                'xgboost_classifier',

            'feature_names':
                feature_names,

            'metrics':
                metrics,

            'dataset_split': {
                'training': 0.70,
                'validation': 0.15,
                'testing': 0.15
            },

            'version':
                '2.0.0'
        }

        metadata_path = (
            MODEL_DIR /
            "ivf_success_xgboost_metadata.json"
        )

        with open(
            metadata_path,
            'w'
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2
            )

        logger.info(
            f"Model trained and saved: "
            f"{model_path}"
        )

        logger.info(
            f"Metadata saved: "
            f"{metadata_path}"
        )

        return model, metrics

    # ========================================================
    # EMBRYO QUALITY MODEL
    # ========================================================

    @staticmethod
    def train_embryo_quality_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[XGBRegressor, Dict]:
        """
        Train embryo quality prediction model.

        This function retains the original
        train/test interface.
        """

        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        # Train
        model.fit(
            X_train,
            y_train
        )

        from sklearn.metrics import (
            mean_squared_error,
            r2_score
        )

        # Test prediction
        y_pred = model.predict(
            X_test
        )

        # Metrics
        metrics = {

            'rmse':
                float(
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            y_pred
                        )
                    )
                ),

            'r2':
                float(
                    r2_score(
                        y_test,
                        y_pred
                    )
                ),

            'feature_importance': {
                name: float(imp)
                for name, imp in zip(
                    feature_names,
                    model.feature_importances_
                )
            }
        }

        # Save model
        model_path = (
            MODEL_DIR /
            "embryo_quality_xgboost.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        logger.info(
            f"Embryo quality model trained: "
            f"{model_path}"
        )

        return model, metrics


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_ivf_success_predictor() -> IVFSuccessPredictor:
    """Get IVF success predictor instance."""

    return IVFSuccessPredictor()


def get_embryo_quality_predictor() -> EmbryoQualityPredictor:
    """Get embryo quality predictor instance."""

    return EmbryoQualityPredictor()


def predict_ivf_success(
    patient_data
) -> Dict:
    """Convenience function to predict IVF success."""

    predictor = IVFSuccessPredictor()

    return predictor.predict(
        patient_data
    )


def predict_embryo_quality(
    data
) -> Dict:
    """Convenience function to predict embryo quality."""

    predictor = EmbryoQualityPredictor()

    return predictor.predict(
        data
    )