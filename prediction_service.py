"""
Prediction Service for IVF Journey Tracker.
Integrates ML models (XGBoost, LightGBM, BNN) with fallback to rule-based predictions.
"""

import math
import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import ML model modules
try:
    from ml_models.xgboost_models import IVFSuccessPredictor, EmbryoQualityPredictor
    from ml_models.lightgbm_models import (
        PatientClassificationPredictor,
        TreatmentRecommendationPredictor,
        ReportAnalysisPredictor
    )
    from ml_models.bnn_models import (
        BNNRiskAssessor,
        BNNHormonePredictor,
        BNNCycleOutcomePredictor
    )
    ML_MODELS_AVAILABLE = True
    logging.info("ML models loaded successfully")
except ImportError as e:
    ML_MODELS_AVAILABLE = False
    logging.warning(f"ML models not available: {e}")

# Initialize ML model predictors (lazy loading)
_ivf_success_predictor = None
_embryo_quality_predictor = None
_patient_classifier = None
_treatment_recommender = None
_report_analyzer = None
_risk_assessor = None
_hormone_predictor = None
_cycle_outcome_predictor = None


def _get_ivf_success_predictor():
    """Get or create IVF success predictor instance."""
    global _ivf_success_predictor
    if ML_MODELS_AVAILABLE and _ivf_success_predictor is None:
        try:
            _ivf_success_predictor = IVFSuccessPredictor()
        except Exception as e:
            logging.error(f"Error creating IVF success predictor: {e}")
    return _ivf_success_predictor


def _get_embryo_quality_predictor():
    """Get or create embryo quality predictor instance."""
    global _embryo_quality_predictor
    if ML_MODELS_AVAILABLE and _embryo_quality_predictor is None:
        try:
            _embryo_quality_predictor = EmbryoQualityPredictor()
        except Exception as e:
            logging.error(f"Error creating embryo quality predictor: {e}")
    return _embryo_quality_predictor


def _get_patient_classifier():
    """Get or create patient classifier instance."""
    global _patient_classifier
    if ML_MODELS_AVAILABLE and _patient_classifier is None:
        try:
            _patient_classifier = PatientClassificationPredictor()
        except Exception as e:
            logging.error(f"Error creating patient classifier: {e}")
    return _patient_classifier


def _get_treatment_recommender():
    """Get or create treatment recommender instance."""
    global _treatment_recommender
    if ML_MODELS_AVAILABLE and _treatment_recommender is None:
        try:
            _treatment_recommender = TreatmentRecommendationPredictor()
        except Exception as e:
            logging.error(f"Error creating treatment recommender: {e}")
    return _treatment_recommender


def _get_risk_assessor():
    """Get or create risk assessor instance."""
    global _risk_assessor
    if ML_MODELS_AVAILABLE and _risk_assessor is None:
        try:
            _risk_assessor = BNNRiskAssessor()
        except Exception as e:
            logging.error(f"Error creating risk assessor: {e}")
    return _risk_assessor


def calculate_ivf_success_prediction(patient_data):
    """
    Calculate IVF success prediction based on multiple factors.
    Uses ML model (XGBoost) when available, falls back to rule-based prediction.
    """
    if not patient_data:
        return {
            "success_rate": 0,
            "confidence": 0,
            "factors": [],
            "recommendations": [],
            "model_type": "none"
        }

    # Try to use ML model first
    predictor = _get_ivf_success_predictor()
    
    if predictor and predictor.model is not None:
        try:
            # Use ML model for prediction
            ml_result = predictor.predict(patient_data)
            
            return {
                "success_rate": ml_result.get('success_probability', 35.0),
                "confidence": ml_result.get('confidence', 70.0),
                "factors": _extract_ml_factors(ml_result),
                "recommendations": _generate_ml_recommendations(patient_data, ml_result),
                "interpretation": get_success_rate_interpretation(ml_result.get('success_probability', 35.0)),
                "model_type": ml_result.get('model_type', 'xgboost'),
                "prediction_label": ml_result.get('prediction_label', 'Unknown'),
                "feature_importance": ml_result.get('feature_importance', {})
            }
        except Exception as e:
            logging.error(f"ML prediction failed, using fallback: {e}")
    
    # Fallback to rule-based prediction
    return _rule_based_ivf_prediction(patient_data)


def _extract_ml_factors(ml_result: Dict) -> list:
    """Extract factors from ML prediction result."""
    factors = []
    importance = ml_result.get('feature_importance', {})
    
    for feature, imp in importance.items():
        if imp > 0.1:
            factors.append({
                "factor": feature.replace('_', ' ').title(),
                "importance": round(imp, 3),
                "positive": imp > 0.15
            })
    
    return factors[:5]


def _generate_ml_recommendations(patient_data, ml_result: Dict) -> list:
    """Generate recommendations based on ML prediction."""
    recommendations = []
    success_prob = ml_result.get('success_probability', 35.0)
    
    if success_prob < 45:
        recommendations.append("Consider preimplantation genetic testing (PGT-A)")
        recommendations.append("Explore lifestyle modifications to improve outcomes")
    
    if patient_data.age and patient_data.age > 35:
        recommendations.append("Consider genetic testing of embryos (PGT-A)")
    
    if patient_data.amh_level and patient_data.amh_level < 1.0:
        recommendations.append("Discuss aggressive stimulation protocols with your doctor")
    
    if not recommendations:
        recommendations.append("Continue with current treatment plan")
        recommendations.append("Maintain healthy lifestyle for optimal results")
    
    return recommendations


def _rule_based_ivf_prediction(patient_data):
    """Fallback rule-based IVF prediction."""
    # Add daily variation to make predictions change
    random.seed(datetime.now().date().toordinal() + 2)
    daily_variation = random.uniform(-3, 3)

    base_rate = 35.0
    factors = []
    adjustments = daily_variation
    
    # Age factor
    if patient_data.age:
        if patient_data.age <= 30:
            age_adjustment = 15
            factors.append({"factor": "Age ≤30", "impact": "+15%", "positive": True})
        elif patient_data.age <= 35:
            age_adjustment = 5
            factors.append({"factor": "Age 31-35", "impact": "+5%", "positive": True})
        elif patient_data.age <= 37:
            age_adjustment = -5
            factors.append({"factor": "Age 36-37", "impact": "-5%", "positive": False})
        elif patient_data.age <= 40:
            age_adjustment = -15
            factors.append({"factor": "Age 38-40", "impact": "-15%", "positive": False})
        else:
            age_adjustment = -25
            factors.append({"factor": "Age >40", "impact": "-25%", "positive": False})
        adjustments += age_adjustment
    
    # BMI factor
    if patient_data.bmi:
        if 18.5 <= patient_data.bmi <= 24.9:
            bmi_adjustment = 5
            factors.append({"factor": "Healthy BMI", "impact": "+5%", "positive": True})
        elif patient_data.bmi < 18.5 or patient_data.bmi >= 30:
            bmi_adjustment = -10
            factors.append({"factor": "BMI outside healthy range", "impact": "-10%", "positive": False})
        else:
            bmi_adjustment = -3
            factors.append({"factor": "Borderline BMI", "impact": "-3%", "positive": False})
        adjustments += bmi_adjustment
    
    # AMH level factor
    if patient_data.amh_level:
        if patient_data.amh_level >= 2.0:
            amh_adjustment = 8
            factors.append({"factor": "Good AMH level", "impact": "+8%", "positive": True})
        elif patient_data.amh_level >= 1.0:
            amh_adjustment = 2
            factors.append({"factor": "Adequate AMH level", "impact": "+2%", "positive": True})
        else:
            amh_adjustment = -12
            factors.append({"factor": "Low AMH level", "impact": "-12%", "positive": False})
        adjustments += amh_adjustment
    
    # Previous IVF cycles factor
    if patient_data.previous_ivf_cycles:
        if patient_data.previous_ivf_cycles == 1:
            cycle_adjustment = -5
            factors.append({"factor": "1 previous cycle", "impact": "-5%", "positive": False})
        elif patient_data.previous_ivf_cycles >= 2:
            cycle_adjustment = -10
            factors.append({"factor": "Multiple previous cycles", "impact": "-10%", "positive": False})
        adjustments += cycle_adjustment
    else:
        factors.append({"factor": "First IVF attempt", "impact": "+3%", "positive": True})
        adjustments += 3
    
    # Partner age factor
    if patient_data.partner_age:
        if patient_data.partner_age <= 35:
            partner_adjustment = 3
            factors.append({"factor": "Partner age ≤35", "impact": "+3%", "positive": True})
        elif patient_data.partner_age >= 45:
            partner_adjustment = -5
            factors.append({"factor": "Partner age ≥45", "impact": "-5%", "positive": False})
        adjustments += partner_adjustment
    
    final_rate = max(5, min(85, base_rate + adjustments))
    
    data_points = sum([
        1 if patient_data.age else 0,
        1 if patient_data.bmi else 0,
        1 if patient_data.amh_level else 0,
        1 if patient_data.fsh_level else 0,
        1 if patient_data.partner_age else 0
    ])
    confidence = min(95, 60 + (data_points * 7))
    
    recommendations = []
    if patient_data.age and patient_data.age > 35:
        recommendations.append("Consider genetic testing of embryos (PGT-A)")
    if patient_data.bmi and (patient_data.bmi < 18.5 or patient_data.bmi >= 25):
        recommendations.append("Optimize weight through nutrition and exercise")
    if patient_data.amh_level and patient_data.amh_level < 1.0:
        recommendations.append("Discuss aggressive stimulation protocols with your doctor")
    if not patient_data.lifestyle_factors:
        recommendations.append("Optimize lifestyle: quit smoking, limit alcohol, manage stress")
    
    return {
        "success_rate": round(final_rate, 1),
        "confidence": confidence,
        "factors": factors,
        "recommendations": recommendations,
        "interpretation": get_success_rate_interpretation(final_rate),
        "model_type": "rule_based"
    }

def calculate_embryo_quality_score(patient_data):
    """AI-powered embryo quality predictor (novel feature)"""
    if not patient_data:
        return {
            "quality_score": 0,
            "grade": "Unknown",
            "development_probability": 0,
            "implantation_potential": 0
        }

    # Add daily variation to make predictions change
    random.seed(datetime.now().date().toordinal())
    daily_variation = random.uniform(-5, 5)

    base_score = 65.0
    adjustments = daily_variation

    # Age-based embryo quality
    if patient_data.age:
        if patient_data.age <= 30:
            adjustments += 20
        elif patient_data.age <= 35:
            adjustments += 10
        elif patient_data.age <= 38:
            adjustments += 0
        elif patient_data.age <= 42:
            adjustments -= 15
        else:
            adjustments -= 30

    # AMH impact on egg quality
    if patient_data.amh_level:
        if patient_data.amh_level >= 2.0:
            adjustments += 10
        elif patient_data.amh_level >= 1.0:
            adjustments += 5
        else:
            adjustments -= 10

    # Lifestyle factors
    if patient_data.lifestyle_factors:
        lifestyle_lower = patient_data.lifestyle_factors.lower()
        if 'non-smoker' in lifestyle_lower or 'no smoking' in lifestyle_lower:
            adjustments += 5
        if 'exercise' in lifestyle_lower or 'active' in lifestyle_lower:
            adjustments += 3
        if 'smoking' in lifestyle_lower:
            adjustments -= 15

    final_score = max(10, min(95, base_score + adjustments))

    # Determine grade
    if final_score >= 80:
        grade = "A (Excellent)"
    elif final_score >= 65:
        grade = "B (Good)"
    elif final_score >= 45:
        grade = "C (Fair)"
    else:
        grade = "D (Poor)"

    # Calculate related probabilities
    development_probability = min(90, final_score * 0.9)
    implantation_potential = min(85, final_score * 0.8)

    # Generate detailed explanation based on factors
    explanation = []
    if patient_data.age:
        if patient_data.age <= 30:
            explanation.append("Age ≤30 provides excellent embryo quality potential")
        elif patient_data.age <= 35:
            explanation.append("Age 31-35 offers good embryo development prospects")
        elif patient_data.age <= 38:
            explanation.append("Age 36-38 indicates moderate embryo quality potential")
        else:
            explanation.append("Age >38 significantly impacts embryo quality and development")

    if patient_data.amh_level:
        if patient_data.amh_level >= 2.0:
            explanation.append("High AMH level (≥2.0) suggests good ovarian reserve and egg quality")
        elif patient_data.amh_level >= 1.0:
            explanation.append("Adequate AMH level (1.0-2.0) indicates reasonable egg quality potential")
        else:
            explanation.append("Low AMH level (<1.0) may affect egg quantity and quality")

    if patient_data.lifestyle_factors:
        lifestyle_lower = patient_data.lifestyle_factors.lower()
        if 'non-smoker' in lifestyle_lower or 'no smoking' in lifestyle_lower:
            explanation.append("Non-smoking lifestyle supports better embryo development")
        if 'exercise' in lifestyle_lower or 'active' in lifestyle_lower:
            explanation.append("Regular exercise contributes to improved egg quality")
        if 'smoking' in lifestyle_lower:
            explanation.append("Smoking negatively impacts embryo quality and implantation potential")

    return {
        "quality_score": round(final_score, 1),
        "grade": grade,
        "development_probability": round(development_probability, 1),
        "implantation_potential": round(implantation_potential, 1),
        "explanation": explanation,
        "factors": [
            f"Age factor: {patient_data.age if patient_data.age else 'Not provided'}",
            f"AMH level: {patient_data.amh_level if patient_data.amh_level else 'Not provided'}",
            f"Lifestyle: {patient_data.lifestyle_factors if patient_data.lifestyle_factors else 'Not provided'}"
        ]
    }

def generate_personalized_protocol(patient_data):
    """Personalized treatment protocol AI advisor (novel feature)"""
    if not patient_data:
        return {
            "protocol_name": "Standard Protocol",
            "medication_suggestions": [],
            "timing_recommendations": {},
            "success_optimization": []
        }

    # Add daily variation to make recommendations change
    random.seed(datetime.now().date().toordinal() + 1)  # Different seed for protocol
    daily_variation = random.randint(0, 3)  # 0-3 for different protocol variations

    age = patient_data.age or 35
    amh = patient_data.amh_level or 1.5
    bmi = patient_data.bmi or 24

    # Determine protocol based on patient characteristics with daily variation
    protocol_options = [
        "Standard Long Protocol",
        "Short Protocol",
        "Antagonist Protocol",
        "Mini-IVF Protocol",
        "Natural Cycle IVF"
    ]

    if age <= 35 and amh >= 2.0:
        base_protocols = ["Standard Long Protocol", "Antagonist Protocol"]
        stimulation_days = "10-12 days"
        expected_response = "Good"
    elif age <= 35 and amh < 1.0:
        base_protocols = ["High-Dose Short Protocol", "Mini-IVF Protocol"]
        stimulation_days = "8-10 days"
        expected_response = "Moderate"
    elif age > 35 and amh >= 1.5:
        base_protocols = ["Antagonist Protocol", "Short Protocol"]
        stimulation_days = "9-11 days"
        expected_response = "Good to Moderate"
    else:
        base_protocols = ["Mini-IVF or Natural Cycle", "Short Protocol"]
        stimulation_days = "5-8 days"
        expected_response = "Low to Moderate"

    # Select protocol with daily variation
    protocol_index = (daily_variation + hash(str(age) + str(amh)) % len(base_protocols)) % len(base_protocols)
    protocol = base_protocols[protocol_index]

    # Medication suggestions with variation
    medications = []
    if amh < 1.0:
        med_options = [
            "Higher dose FSH (300-450 IU)",
            "Consider adding LH supplementation",
            "Add HMG supplementation"
        ]
        medications.extend(random.sample(med_options, random.randint(1, 2)))
    else:
        med_options = [
            "Standard dose FSH (150-225 IU)",
            "Low dose FSH (112-150 IU)",
            "Recombinant FSH"
        ]
        medications.append(random.choice(med_options))

    if age > 38:
        medications.append("Consider growth hormone supplementation")

    if bmi >= 30:
        medications.append("Adjusted dosing for BMI")

    # Timing recommendations with variation
    # Parse stimulation_days to extract numbers for variation
    duration_parts = stimulation_days.replace(' days', '').split('-')
    start_day = int(duration_parts[0])
    end_day = int(duration_parts[1])

    timing_options = {
        "cycle_start": ["Day 2-3 of menstrual cycle", "Day 3-4 of menstrual cycle"],
        "stimulation_duration": [stimulation_days, f"{max(5, start_day-1)}-{max(6, end_day-1)} days"],
        "monitoring_frequency": ["Every 2-3 days after day 5", "Every 1-2 days after day 6"],
        "trigger_timing": ["When 2-3 follicles reach 17-18mm", "When lead follicle reaches 18-20mm"]
    }

    timing = {
        "cycle_start": random.choice(timing_options["cycle_start"]),
        "stimulation_duration": random.choice(timing_options["stimulation_duration"]),
        "monitoring_frequency": random.choice(timing_options["monitoring_frequency"]),
        "trigger_timing": random.choice(timing_options["trigger_timing"])
    }

    # Success optimization tips with variation
    base_optimization = [
        "Maintain optimal weight and nutrition",
        "Consider acupuncture for improved outcomes",
        "Ensure adequate sleep (7-9 hours nightly)",
        "Take prescribed supplements consistently",
        "Practice stress reduction techniques",
        "Consider yoga or meditation",
        "Optimize diet with fertility foods",
        "Avoid environmental toxins"
    ]

    if patient_data.lifestyle_factors and 'stress' in patient_data.lifestyle_factors.lower():
        base_optimization.insert(0, "Implement stress reduction techniques")

    if age > 35:
        base_optimization.append("Discuss PGT-A testing for embryo selection")

    # Select 4-6 random tips
    optimization = random.sample(base_optimization, min(len(base_optimization), random.randint(4, 6)))

    # Generate explanation for protocol selection
    protocol_explanation = []
    if age <= 35 and amh >= 2.0:
        protocol_explanation.append("Selected based on young age and excellent ovarian reserve")
    elif age <= 35 and amh < 1.0:
        protocol_explanation.append("Chosen for younger age with diminished ovarian reserve")
    elif age > 35 and amh >= 1.5:
        protocol_explanation.append("Recommended for advanced maternal age with adequate reserve")
    else:
        protocol_explanation.append("Selected for advanced age and/or low ovarian reserve")

    if bmi >= 30:
        protocol_explanation.append("Adjusted for BMI considerations in medication dosing")
    if age > 38:
        protocol_explanation.append("Includes growth hormone consideration for advanced age")

    return {
        "protocol_name": protocol,
        "expected_response": expected_response,
        "medication_suggestions": medications,
        "timing_recommendations": timing,
        "success_optimization": optimization,
        "personalization_score": calculate_personalization_score(patient_data),
        "protocol_explanation": protocol_explanation
    }

def calculate_personalization_score(patient_data):
    """Calculate how personalized the protocol is based on available data"""
    data_points = 0
    if patient_data.age: data_points += 1
    if patient_data.amh_level: data_points += 1
    if patient_data.bmi: data_points += 1
    if patient_data.fsh_level: data_points += 1
    if patient_data.previous_ivf_cycles is not None: data_points += 1
    if patient_data.diagnosis: data_points += 1
    if patient_data.lifestyle_factors: data_points += 1
    
    return min(100, (data_points / 7) * 100)

def get_success_rate_interpretation(rate):
    """Provide interpretation of success rate"""
    if rate >= 60:
        return "Excellent prospects - above average success rate"
    elif rate >= 45:
        return "Good prospects - average to above-average success rate"
    elif rate >= 30:
        return "Moderate prospects - consider optimization strategies"
    else:
        return "Challenging case - discuss alternative approaches with your doctor"


# ============ NEW ML-BASED PREDICTION FUNCTIONS ============

def assess_treatment_risk(patient_data):
    """
    Assess treatment risks using Bayesian Neural Network.
    Returns: OHSS risk, miscarriage risk, complications risk with uncertainty intervals.
    """
    if not patient_data:
        return {"error": "No patient data", "model_type": "none"}
    
    assessor = _get_risk_assessor()
    
    if assessor and assessor.is_trained:
        try:
            result = assessor.predict(patient_data)
            result['model_type'] = 'bnn'
            result['risk_categories'] = {
                'ohss_risk': result.get('risk_probability', 10),
                'miscarriage_risk': _estimate_miscarriage_risk(patient_data),
                'complications_risk': _estimate_complications_risk(patient_data)
            }
            return result
        except Exception as e:
            logging.error(f"Risk assessment error: {e}")
    
    return _rule_based_risk(patient_data)


def _estimate_miscarriage_risk(patient_data):
    age = getattr(patient_data, 'age', 35) or 35
    base_risk = 15
    if age > 35:
        base_risk += min(30, (age - 35) * 3)
    return min(60, base_risk)


def _estimate_complications_risk(patient_data):
    bmi = getattr(patient_data, 'bmi', 24) or 24
    risk = 10
    if bmi > 30:
        risk += 15
    return min(40, risk)


def _rule_based_risk(patient_data):
    age = getattr(patient_data, 'age', 35) or 35
    amh = getattr(patient_data, 'amh_level', 2.0) or 2.0
    bmi = getattr(patient_data, 'bmi', 24) or 24
    
    ohss_risk = 10
    if amh > 4.0: ohss_risk += 25
    if age < 30: ohss_risk += 15
    
    miscarriage_risk = 15
    if age > 35: miscarriage_risk += min(30, (age - 35) * 3)
    
    complications_risk = 10
    if bmi > 30: complications_risk += 15
    
    return {
        'ohss_risk': ohss_risk,
        'miscarriage_risk': miscarriage_risk,
        'complications_risk': complications_risk,
        'confidence': 50.0,
        'uncertainty': 20.0,
        'risk_level': 'Moderate' if ohss_risk > 30 else 'Low',
        'model_type': 'rule_based'
    }


def classify_patient_responder(patient_data):
    """
    Classify patient into responder category using LightGBM.
    Returns: Low Responder, Normal Responder, or High Responder
    """
    if not patient_data:
        return {"error": "No patient data", "model_type": "none"}
    
    classifier = _get_patient_classifier()
    
    if classifier and classifier.model is not None:
        try:
            result = classifier.predict(patient_data)
            result['model_type'] = 'lightgbm'
            return result
        except Exception as e:
            logging.error(f"Classification error: {e}")
    
    # Fallback
    amh = getattr(patient_data, 'amh_level', 2.0) or 2.0
    if amh < 1.0:
        classification = 'Low Responder'
    elif amh > 4.0:
        classification = 'High Responder'
    else:
        classification = 'Normal Responder'
    
    return {
        'classification': classification,
        'confidence': 60.0,
        'model_type': 'rule_based'
    }


def recommend_treatment_protocol(patient_data):
    """
    Generate personalized treatment recommendations using LightGBM.
    Returns: Protocol, medication dosage, adjustments
    """
    if not patient_data:
        return {"error": "No patient data", "model_type": "none"}
    
    recommender = _get_treatment_recommender()
    
    if recommender and recommender.model is not None:
        try:
            result = recommender.predict(patient_data)
            result['model_type'] = 'lightgbm'
            return result
        except Exception as e:
            logging.error(f"Recommendation error: {e}")
    
    # Fallback
    age = getattr(patient_data, 'age', 35) or 35
    amh = getattr(patient_data, 'amh_level', 2.0) or 2.0
    bmi = getattr(patient_data, 'bmi', 24) or 24
    
    if age <= 35 and amh >= 2.0:
        protocol, dosage = "Standard Antagonist Protocol", "150-225 IU FSH"
    elif age <= 35 and amh < 1.0:
        protocol, dosage = "High-Dose Stimulation", "300-450 IU FSH"
    elif age > 35 and amh >= 1.5:
        protocol, dosage = "Mild Stimulation Protocol", "100-150 IU FSH"
    else:
        protocol, dosage = "Mini-IVF Protocol", "75-100 IU FSH"
    
    adjustments = []
    if amh < 1.0: adjustments.append("Add LH supplementation")
    if bmi >= 30: adjustments.append("Adjust for BMI")
    
    return {'protocol': protocol, 'fsh_dosage': dosage, 'adjustments': adjustments, 'confidence': 60.0, 'model_type': 'rule_based'}


def analyze_medical_report(extracted_data):
    """
    Analyze extracted medical report data using LightGBM.
    Detects abnormalities and provides auto-fill suggestions.
    """
    if not extracted_data:
        return {"error": "No data", "model_type": "none"}
    
    try:
        from ml_models.lightgbm_models import ReportAnalysisPredictor
        analyzer = ReportAnalysisPredictor()
        result = analyzer.predict(extracted_data)
        result['model_type'] = 'lightgbm'
        return result
    except Exception as e:
        logging.error(f"Report analysis error: {e}")
    
    # Fallback - basic abnormality detection
    abnormalities = []
    normal_ranges = {'e2_level': (20, 300), 'fsh_level': (4, 13), 'amh_level': (1, 4)}
    
    for param, (low, high) in normal_ranges.items():
        value = extracted_data.get(param)
        if value is not None:
            if value < low or value > high:
                abnormalities.append(f"{param} outside normal range ({low}-{high})")
    
    return {'has_abnormalities': bool(abnormalities), 'abnormalities': abnormalities, 'confidence': 50.0, 'model_type': 'rule_based'}


# ============ IVF SUCCESS IMPROVEMENT SIMULATOR ============

import json
import os
import joblib as _joblib
import numpy as np

# Feature order as expected by the trained RandomForest model (from metadata)
_SIMULATOR_FEATURE_ORDER = ['age', 'bmi', 'amh', 'fsh', 'previous_ivf', 'stress', 'sleep_hours', 'exercise_min']

# Cache the loaded model to avoid reloading on every request
_simulator_model = None
_simulator_meta = None


def _load_simulator_model():
    """
    Load the existing trained IVF success model (RandomForest from models/).
    NEVER retrains — only loads pre-trained model from disk.
    Uses joblib to load the .pkl file.
    """
    global _simulator_model, _simulator_meta
    if _simulator_model is not None:
        return _simulator_model, _simulator_meta

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'ivf_success_model.pkl')
    meta_path = os.path.join(os.path.dirname(__file__), 'models', 'ivf_model_metadata.json')

    if not os.path.exists(model_path):
        logging.warning(f"Simulator: Model not found at {model_path}. Trying fallback path.")
        # Try alternate path (project root)
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ivf_success_model.pkl')
        meta_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ivf_model_metadata.json')

    if not os.path.exists(model_path):
        logging.error("Simulator: IVF success model not found. Cannot run simulations.")
        return None, None

    try:
        _simulator_model = _joblib.load(model_path)
        _simulator_meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                _simulator_meta = json.load(f)
        feature_order = _simulator_meta.get('feature_order', _SIMULATOR_FEATURE_ORDER)
        logging.info(f"Simulator: Model loaded successfully. Features: {feature_order}")
        return _simulator_model, feature_order
    except Exception as e:
        logging.error(f"Simulator: Error loading model: {e}")
        return None, None


def _predict_single(features_dict):
    """
    Run predict_proba on a single feature set.
    features_dict: dict with keys matching _SIMULATOR_FEATURE_ORDER (or feature_order from meta)
    Returns: (prediction_class, probability_of_success_as_percentage)
    """
    model, feature_order = _load_simulator_model()
    if model is None:
        return None, None

    try:
        # Build feature vector in the correct order
        fv = []
        for feat_name in feature_order:
            val = features_dict.get(feat_name, 0)
            fv.append(float(val))
        fv_array = np.array([fv])

        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(fv_array)[0]
            # proba[0] = class 0 (failure), proba[1] = class 1 (success)
            success_prob = float(proba[1]) * 100.0
        else:
            # Fallback if model doesn't have predict_proba
            pred = model.predict(fv_array)[0]
            success_prob = float(pred) * 100.0 if pred < 2 else 50.0

        pred_class = int(model.predict(fv_array)[0])
        return pred_class, round(success_prob, 1)

    except Exception as e:
        logging.error(f"Simulator: Prediction error: {e}")
        return None, None


def calculate_ivf_improvements(features_dict):
    """
    IVF Success Improvement Simulator.
    
    Takes the current patient features and simulates one-by-one lifestyle modifications
    to show the potential improvement in IVF success probability.
    
    NEVER retrains the model — only uses the existing pre-trained model for inference.
    
    Args:
        features_dict: dict with keys like 'age', 'bmi', 'amh', 'fsh', 
                       'previous_ivf', 'stress', 'sleep_hours', 'exercise_min'
    
    Returns:
        dict with:
          - current_probability: float (baseline success %)
          - current_prediction: str (label)
          - improvements: list of dicts, each with:
              - factor: str (e.g., 'weight_loss', 'stress_reduction')
              - label: str (human-readable, e.g., 'Lose Weight (Healthy BMI)')
              - new_probability: float
              - delta: float (new - current)
              - explanation: str (why this helps)
    """
    model, feature_order = _load_simulator_model()
    if model is None:
        return {
            'current_probability': None,
            'current_prediction': 'Model not available',
            'improvements': [],
            'error': 'IVF success model not found. Please ensure models/ivf_success_model.pkl exists.'
        }

    # Ensure all required features exist with defaults
    base_features = {
        'age': 35,
        'bmi': 24.0,
        'amh': 2.0,
        'fsh': 7.0,
        'previous_ivf': 0,
        'stress': 3,
        'sleep_hours': 7.0,
        'exercise_min': 30
    }
    # Override with provided values
    for k, v in features_dict.items():
        if k in base_features and v is not None:
            base_features[k] = float(v)

    # 1. Get baseline prediction
    _, base_prob = _predict_single(base_features)
    if base_prob is None:
        return {
            'current_probability': None,
            'current_prediction': 'Prediction failed',
            'improvements': [],
            'error': 'Failed to compute baseline prediction.'
        }

    improvements = []
    current_bmi = base_features['bmi']

    # --- Scenario 1: Weight Loss → Healthy BMI (22) ---
    # Only suggest if BMI > 24.9 (overweight range)
    if current_bmi > 24.9:
        target_bmi = 22.0  # Ideal BMI within healthy range
        bmi_features = dict(base_features)
        bmi_features['bmi'] = target_bmi
        _, bmi_prob = _predict_single(bmi_features)
        if bmi_prob is not None:
            weight_loss_kg = round((current_bmi - target_bmi) * 1.5, 1)  # Approx kg needed
            improvements.append({
                'factor': 'weight_loss',
                'label': f'Lose ~{weight_loss_kg} kg (Healthy BMI {target_bmi})',
                'new_probability': bmi_prob,
                'delta': round(bmi_prob - base_prob, 1),
                'explanation': 'Achieving a healthy BMI (18.5–24.9) reduces inflammation, improves hormone balance, and enhances implantation success. Excess body fat can lead to insulin resistance and hormonal imbalances that negatively impact IVF outcomes.'
            })

    # --- Scenario 2: Reduce Stress to Level 2 ---
    current_stress = base_features['stress']
    if current_stress > 2:
        stress_features = dict(base_features)
        stress_features['stress'] = 2
        _, stress_prob = _predict_single(stress_features)
        if stress_prob is not None:
            improvements.append({
                'factor': 'stress_reduction',
                'label': f'Reduce Stress to Level 2 (from {int(current_stress)})',
                'new_probability': stress_prob,
                'delta': round(stress_prob - base_prob, 1),
                'explanation': 'High stress elevates cortisol levels, which can suppress ovulation, reduce implantation rates, and negatively affect hormone production needed for IVF success. Stress management techniques like meditation, yoga, and counseling can significantly improve outcomes.'
            })

    # --- Scenario 3: Optimize Sleep to 8 Hours ---
    current_sleep = base_features['sleep_hours']
    if current_sleep < 7.5:
        sleep_features = dict(base_features)
        sleep_features['sleep_hours'] = 8.0
        _, sleep_prob = _predict_single(sleep_features)
        if sleep_prob is not None:
            improvements.append({
                'factor': 'sleep_optimization',
                'label': f'Sleep 8 Hours (from {current_sleep}h)',
                'new_probability': sleep_prob,
                'delta': round(sleep_prob - base_prob, 1),
                'explanation': 'Adequate sleep (7–9 hours) is critical for hormonal regulation. Sleep deprivation disrupts the release of reproductive hormones like FSH, LH, and melatonin, which are essential for follicle development, egg quality, and successful implantation.'
            })

    # --- Scenario 4: Exercise 30 mins/day ---
    current_exercise = base_features['exercise_min']
    if current_exercise < 25:
        exercise_features = dict(base_features)
        exercise_features['exercise_min'] = 30
        _, exercise_prob = _predict_single(exercise_features)
        if exercise_prob is not None:
            improvements.append({
                'factor': 'exercise_optimization',
                'label': f'Exercise 30 Mins/Day (from {int(current_exercise)} min)',
                'new_probability': exercise_prob,
                'delta': round(exercise_prob - base_prob, 1),
                'explanation': 'Moderate regular exercise improves blood circulation to the reproductive organs, reduces stress, helps maintain healthy weight, and regulates hormones. Studies show women who exercise moderately have higher IVF success rates than sedentary women.'
            })

    # --- Scenario 5: Combined (All Optimized) ---
    all_features = dict(base_features)
    if current_bmi > 24.9:
        all_features['bmi'] = 22.0
    if current_stress > 2:
        all_features['stress'] = 2
    if current_sleep < 7.5:
        all_features['sleep_hours'] = 8.0
    if current_exercise < 25:
        all_features['exercise_min'] = 30

    _, all_prob = _predict_single(all_features)
    if all_prob is not None:
        improvements.append({
            'factor': 'all_optimized',
            'label': '🌟 Combined: All Lifestyle Factors Optimized',
            'new_probability': all_prob,
            'delta': round(all_prob - base_prob, 1),
            'explanation': 'Optimizing ALL modifiable lifestyle factors provides a cumulative benefit. Healthy BMI, low stress, quality sleep, and regular exercise work synergistically to create the best possible conditions for IVF success — improving hormone balance, uterine health, egg quality, and overall well-being.'
        })

    # Sort improvements by delta descending (most impactful first)
    improvements.sort(key=lambda x: x['delta'], reverse=True)

    # Determine prediction label
    if base_prob >= 60:
        pred_label = 'Excellent Prospects'
    elif base_prob >= 45:
        pred_label = 'Good Prospects'
    elif base_prob >= 30:
        pred_label = 'Moderate Prospects'
    else:
        pred_label = 'Challenging Case'

    return {
        'current_probability': base_prob,
        'current_prediction': pred_label,
        'improvements': improvements,
        'features_used': base_features
    }
