"""
Data Preprocessing and Feature Engineering utilities for IVF ML models.
Handles data normalization, missing values, and feature creation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data cleaning, normalization, and missing value imputation.
    """
    
    # Define feature ranges for normalization
    FEATURE_RANGES = {
        'age': (18, 55),
        'bmi': (15, 40),
        'amh_level': (0, 10),
        'fsh_level': (0, 30),
        'e2_level': (0, 500),  # Estradiol
        'p4_level': (0, 30),   # Progesterone
        'num_eggs_retrieved': (0, 50),
        'fertilization_rate': (0, 1),
        'num_embryos_day5': (0, 30),
        'partner_age': (18, 65),
    }
    
    # Default values for missing data
    DEFAULT_VALUES = {
        'age': 35,
        'bmi': 24,
        'amh_level': 2.0,
        'fsh_level': 7.0,
        'e2_level': 50,
        'p4_level': 0.8,
        'previous_ivf_cycles': 0,
        'previous_pregnancies': 0,
        'partner_age': None,
    }
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler_params = {}
        self.categorical_encoders = {}
    
    def normalize_numeric(self, value: float, feature_name: str) -> float:
        """
        Normalize numeric values to [0, 1] range.
        
        Args:
            value: The value to normalize
            feature_name: Name of the feature for range lookup
            
        Returns:
            Normalized value between 0 and 1
        """
        if value is None:
            return 0.5
        
        if feature_name not in self.FEATURE_RANGES:
            return value
        
        min_val, max_val = self.FEATURE_RANGES[feature_name]
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)
    
    def preprocess_patient_data(self, patient_data) -> Dict:
        """
        Preprocess patient data from PatientData model.
        
        Args:
            patient_data: PatientData model instance
            
        Returns:
            Dictionary of preprocessed features
        """
        if not patient_data:
            return {}
        
        processed = {}
        
        # Numeric features - normalize them
        numeric_features = [
            'age', 'bmi', 'amh_level', 'fsh_level',
            'partner_age', 'previous_ivf_cycles', 'previous_pregnancies'
        ]
        
        for feature in numeric_features:
            value = getattr(patient_data, feature, None)
            if value is None:
                value = self.DEFAULT_VALUES.get(feature, 0)
            processed[feature] = self.normalize_numeric(value, feature)
        
        # Categorical features
        if patient_data.diagnosis:
            processed['has_pcos'] = 'PCOS' in patient_data.diagnosis.upper()
            processed['has_endometriosis'] = 'ENDOMETRIOSIS' in patient_data.diagnosis.upper()
        else:
            processed['has_pcos'] = False
            processed['has_endometriosis'] = False
        
        if patient_data.medical_history:
            processed['has_thyroid_issues'] = 'THYROID' in patient_data.medical_history.upper()
            processed['has_diabetes'] = 'DIABETES' in patient_data.medical_history.upper()
        else:
            processed['has_thyroid_issues'] = False
            processed['has_diabetes'] = False
        
        processed['has_allergies'] = bool(patient_data.allergies)
        
        # Lifestyle factors (simple encoding)
        if patient_data.lifestyle_factors:
            lifestyle = patient_data.lifestyle_factors.lower()
            processed['smoker'] = 'smok' in lifestyle
            processed['alcohol_use'] = 'alcohol' in lifestyle or 'drink' in lifestyle
        else:
            processed['smoker'] = False
            processed['alcohol_use'] = False
        
        return processed
    
    def preprocess_cycle_data(self, ivf_cycle) -> Dict:
        """
        Preprocess IVF cycle data.
        
        Args:
            ivf_cycle: IVFCycle model instance
            
        Returns:
            Dictionary of preprocessed cycle features
        """
        if not ivf_cycle:
            return {}
        
        processed = {}
        
        # Protocol encoded
        protocol_map = {
            'antagonist': 1.0,
            'agonist': 0.7,
            'micro': 0.8,
            'mini': 0.9,
            'natural': 0.5
        }
        
        protocol = ivf_cycle.protocol.lower() if ivf_cycle.protocol else 'antagonist'
        processed['protocol_encoded'] = protocol_map.get(protocol, 0.5)
        
        # Success outcome encoding
        outcome_map = {
            'BFP': 1.0,  # Big Fat Positive
            'Live Birth': 1.0,
            'BFN': 0.0,  # Big Fat Negative
            'Miscarriage': 0.3,
            None: 0.5
        }
        processed['outcome_encoded'] = outcome_map.get(ivf_cycle.outcome, 0.5)
        
        # Numeric features from cycle
        processed['num_eggs_retrieved'] = self.normalize_numeric(
            ivf_cycle.num_eggs_retrieved or 10, 'num_eggs_retrieved'
        )
        processed['fertilization_rate'] = ivf_cycle.fertilization_rate or 0.5
        processed['num_embryos_day5'] = self.normalize_numeric(
            ivf_cycle.num_embryos_day5 or 3, 'num_embryos_day5'
        )
        processed['num_embryos_transferred'] = self.normalize_numeric(
            ivf_cycle.num_embryos_transferred or 2, 'num_embryos_day5'
        )
        
        return processed
    
    def preprocess_wellness_data(self, wellness_logs: List) -> Dict:
        """
        Preprocess wellness log data into aggregate features.
        
        Args:
            wellness_logs: List of WellnessLog instances
            
        Returns:
            Dictionary of aggregated wellness features
        """
        if not wellness_logs:
            return {
                'avg_mood': 3.0,
                'avg_stress': 3.0,
                'avg_sleep_quality': 3.0,
                'avg_energy': 3.0,
                'avg_sleep_hours': 7.0,
                'avg_exercise_minutes': 30,
                'max_stress_last_week': 3.0
            }
        
        moods = [log.mood_rating for log in wellness_logs if log.mood_rating]
        stress = [log.stress_level for log in wellness_logs if log.stress_level]
        sleep_quality = [log.sleep_quality for log in wellness_logs if log.sleep_quality]
        energy = [log.energy_level for log in wellness_logs if log.energy_level]
        sleep_hours = [log.sleep_hours for log in wellness_logs if log.sleep_hours]
        exercise = [log.exercise_minutes for log in wellness_logs if log.exercise_minutes]
        
        processed = {
            'avg_mood': np.mean(moods) if moods else 3.0,
            'avg_stress': np.mean(stress) if stress else 3.0,
            'avg_sleep_quality': np.mean(sleep_quality) if sleep_quality else 3.0,
            'avg_energy': np.mean(energy) if energy else 3.0,
            'avg_sleep_hours': np.mean(sleep_hours) if sleep_hours else 7.0,
            'avg_exercise_minutes': np.mean(exercise) if exercise else 30,
            'max_stress_last_week': max(stress) if stress else 3.0,
        }
        
        # Normalize
        processed['avg_mood'] = self.normalize_numeric(processed['avg_mood'] / 5.0, 'age')
        processed['avg_stress'] = self.normalize_numeric(processed['avg_stress'] / 5.0, 'age')
        processed['avg_sleep_quality'] = self.normalize_numeric(processed['avg_sleep_quality'] / 5.0, 'age')
        processed['avg_energy'] = self.normalize_numeric(processed['avg_energy'] / 5.0, 'age')
        processed['avg_sleep_hours'] = self.normalize_numeric(processed['avg_sleep_hours'], 'age')
        processed['max_stress_last_week'] = self.normalize_numeric(processed['max_stress_last_week'] / 5.0, 'age')
        
        return processed
    
    def handle_missing_values(self, data: Dict, strategy: str = 'mean') -> Dict:
        """
        Handle missing values in processed data.
        
        Args:
            data: Dictionary of features
            strategy: 'mean', 'median', or 'drop'
            
        Returns:
            Data with missing values handled
        """
        filled_data = data.copy()
        
        for key, value in filled_data.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                if strategy == 'mean':
                    # Use default value or 0.5
                    filled_data[key] = self.DEFAULT_VALUES.get(key, 0.5)
        
        return filled_data


class FeatureEngineering:
    """
    Advanced feature engineering for improved model performance.
    Creates derived features from raw data.
    """
    
    @staticmethod
    def create_ivf_success_features(patient_data: Dict, cycle_data: Dict, wellness_data: Dict = None) -> Dict:
        """
        Create engineered features for IVF success prediction.
        
        Args:
            patient_data: Preprocessed patient data
            cycle_data: Preprocessed cycle data
            wellness_data: Preprocessed wellness data
            
        Returns:
            Dictionary with engineered features
        """
        features = {**patient_data, **cycle_data}
        
        if wellness_data:
            features.update(wellness_data)
        
        # Interaction features
        if 'age' in patient_data and 'partner_age' in patient_data:
            features['combined_age_score'] = (patient_data['age'] + patient_data['partner_age']) / 2
        
        if 'amh_level' in patient_data and 'fsh_level' in patient_data:
            # AMH/FSH ratio is clinically meaningful
            amh = patient_data['amh_level'] if patient_data['amh_level'] > 0.1 else 0.5
            fsh = patient_data['fsh_level'] if patient_data['fsh_level'] > 0.1 else 1.0
            features['amh_fsh_ratio'] = np.clip(amh / fsh, 0, 10)
        
        if 'num_eggs_retrieved' in cycle_data and 'fertilization_rate' in cycle_data:
            features['effective_embryos'] = (cycle_data['num_eggs_retrieved'] * 
                                           cycle_data['fertilization_rate'])
        
        # Age-related risk factor
        age = patient_data.get('age', 35)
        features['age_risk_factor'] = max(0, (age - 35) / 20)  # Increases with age
        features['age_optimal'] = 1.0 if age <= 30 else (1.0 - (age - 30) / 30)  # Peaks at 30
        
        return features
    
    @staticmethod
    def create_risk_assessment_features(patient_data: Dict, cycle_data: Dict) -> Dict:
        """
        Create features specific to risk assessment.
        
        Args:
            patient_data: Preprocessed patient data
            cycle_data: Preprocessed cycle data
            
        Returns:
            Dictionary with risk-specific features
        """
        features = {**patient_data, **cycle_data}
        
        # OHSS risk factors
        age = patient_data.get('age', 35)
        features['ohss_risk_score'] = 0.0
        
        if patient_data.get('has_pcos', False):
            features['ohss_risk_score'] += 0.3
        
        if patient_data.get('amh_level', 0) > 0.7:  # High ovarian reserve
            features['ohss_risk_score'] += 0.3
        
        if age < 30:
            features['ohss_risk_score'] += 0.2
        
        # Miscarriage risk factors
        features['miscarriage_risk_score'] = 0.0
        if age > 35:
            features['miscarriage_risk_score'] += min(0.5, (age - 35) / 10)
        
        if patient_data.get('has_diabetes', False):
            features['miscarriage_risk_score'] += 0.2
        
        # Complications risk
        features['complications_risk_score'] = 0.0
        if patient_data.get('smoker', False):
            features['complications_risk_score'] += 0.2
        if patient_data.get('bmi', 0.5) > 0.7:  # High BMI
            features['complications_risk_score'] += 0.15
        
        return features
    
    @staticmethod
    def create_hormone_prediction_features(patient_data: Dict, medical_activities: List = None) -> Dict:
        """
        Create features for hormone level and cycle outcome prediction.
        
        Args:
            patient_data: Preprocessed patient data
            medical_activities: List of medical activities
            
        Returns:
            Dictionary with hormone-specific features
        """
        features = patient_data.copy()
        
        # Hormonal profiles based on diagnosis
        if patient_data.get('has_pcos', False):
            features['elevated_testosterone'] = True
            features['insulin_resistance_likely'] = True
        else:
            features['elevated_testosterone'] = False
            features['insulin_resistance_likely'] = False
        
        if patient_data.get('has_thyroid_issues', False):
            features['thyroid_imbalance'] = True
        else:
            features['thyroid_imbalance'] = False
        
        # BMI interaction with hormones
        bmi_score = patient_data.get('bmi', 0.5)
        features['bmi_hormone_modifier'] = 1.0 - (abs(bmi_score - 0.5) * 0.5)
        
        return features
    
    @staticmethod
    def create_patient_classification_features(patient_data: Dict, cycle_history: List = None) -> Dict:
        """
        Create features for patient responder classification.
        
        Args:
            patient_data: Preprocessed patient data
            cycle_history: List of previous IVF cycles
            
        Returns:
            Dictionary with classification features
        """
        features = patient_data.copy()
        
        # Responder classification indicators
        # Based on AMH and FSH levels
        amh = patient_data.get('amh_level', 0.5)
        fsh = patient_data.get('fsh_level', 0.5)
        
        # AMH > 2 = high reserve, AMH < 1 = low reserve
        # FSH > 10 = poor ovarian reserve
        features['reserve_score'] = amh / (fsh + 0.1)  # Avoid division by zero
        
        # Age factor for classification
        age = patient_data.get('age', 35)
        features['age_classification_score'] = 1.0 if age <= 35 else (1.0 - (age - 35) / 20)
        
        # Prior cycle outcomes
        if cycle_history:
            num_cycles = len(cycle_history)
            successful_cycles = sum(1 for c in cycle_history if c.get('outcome') in ['BFP', 'Live Birth'])
            features['success_rate_historical'] = successful_cycles / max(num_cycles, 1)
        else:
            features['success_rate_historical'] = 0.5
        
        return features
    
    @staticmethod
    def create_treatment_recommendation_features(patient_data: Dict, cycle_data: Dict = None) -> Dict:
        """
        Create features for personalized treatment recommendations.
        
        Args:
            patient_data: Preprocessed patient data
            cycle_data: Preprocessed cycle data
            
        Returns:
            Dictionary with treatment-specific features
        """
        features = {**patient_data}
        
        if cycle_data:
            features.update(cycle_data)
        
        # Medication dosage indicators
        age = patient_data.get('age', 35)
        amh = patient_data.get('amh_level', 0.5)
        
        # FSH starting dose considerations
        features['low_responder_indicator'] = 1.0 if amh < 0.3 else 0.0
        features['high_responder_indicator'] = 1.0 if amh > 0.8 else 0.0
        
        # Protocol selection factors
        features['protocol_flexibility_score'] = 1.0 - abs(amh - 0.5)
        
        if patient_data.get('has_pcos', False):
            features['needs_modified_protocol'] = True
            features['metformin_candidate'] = True
        else:
            features['needs_modified_protocol'] = False
            features['metformin_candidate'] = False
        
        return features


class TimeSeriesFeatureEngineering:
    """
    Engineered features for time-series predictions (hormone trends, wellness trajectory).
    """
    
    @staticmethod
    def extract_trend_features(wellness_logs: List, days: int = 30) -> Dict:
        """
        Extract trend features from recent wellness logs.
        
        Args:
            wellness_logs: List of WellnessLog instances
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend features
        """
        if not wellness_logs:
            return {
                'mood_trend': 0.0,
                'stress_trend': 0.0,
                'sleep_quality_trend': 0.0,
                'energy_trend': 0.0
            }
        
        features = {}
        
        # Calculate trends (slope of linear fit)
        if len(wellness_logs) > 1:
            moods = np.array([log.mood_rating or 3 for log in wellness_logs])
            stress = np.array([log.stress_level or 3 for log in wellness_logs])
            sleep_quality = np.array([log.sleep_quality or 3 for log in wellness_logs])
            energy = np.array([log.energy_level or 3 for log in wellness_logs])
            
            x = np.arange(len(moods))
            
            features['mood_trend'] = np.polyfit(x, moods, 1)[0]
            features['stress_trend'] = np.polyfit(x, stress, 1)[0]
            features['sleep_quality_trend'] = np.polyfit(x, sleep_quality, 1)[0]
            features['energy_trend'] = np.polyfit(x, energy, 1)[0]
        else:
            features['mood_trend'] = 0.0
            features['stress_trend'] = 0.0
            features['sleep_quality_trend'] = 0.0
            features['energy_trend'] = 0.0
        
        return features
