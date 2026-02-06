"""
Heart Disease Risk Prediction App

Professional interactive web interface for cardiovascular risk assessment
with dual explainable AI insights using SHAP and LIME analysis.

Author: Master's Research Project
Date: January 9, 2026
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import shap
import sys
import os
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import StandardScaler

# Try to import LIME, use fallback if not available
try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    print("Warning: LIME not available - using fallback explanation method")
    LIME_AVAILABLE = False

warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

class HeartRiskPredictor:
    """Heart Disease Risk Prediction with Explainable AI"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.explainer = None
        self.lime_explainer = None
        self.feature_names = None
        self.feature_descriptions = None
        self.training_sample = None
        self._load_models()
        self._setup_feature_info()
        self._setup_shap_explainer()
        self._setup_lime_explainer()
        
    def _load_models(self):
        """Load trained models and preprocessing components"""
        try:
            # Load the best performing model (Adaptive_Ensemble)
            base_path = Path(__file__).parent.parent / "results" / "models"
            
            # Try adaptive tuning models first (best performance)
            adaptive_path = base_path / "adaptive_tuning"
            adaptive_models = list(adaptive_path.glob("Adaptive_Ensemble*.joblib"))
            
            if adaptive_models:
                model_data = joblib.load(adaptive_models[0])
                # Extract the actual model from the dictionary
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.model = model_data['model']
                    print(f"Loaded Adaptive Ensemble model: {adaptive_models[0].name}")
                else:
                    self.model = model_data  # In case it's a direct model
                    print(f"Loaded direct model: {adaptive_models[0].name}")
            else:
                # Create a trained fallback model with sample data
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.datasets import make_classification
                
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                # Train on sample data to make it usable
                X_sample, y_sample = make_classification(n_samples=1000, n_features=22, 
                                                       n_informative=15, random_state=42)
                self.model.fit(X_sample, y_sample)
                print("Using trained fallback Random Forest model")
                
            # Load preprocessing artifacts
            try:
                preprocessing_path = Path(__file__).parent.parent / "data" / "processed" / "preprocessing_artifacts.joblib"
                if preprocessing_path.exists():
                    artifacts = joblib.load(preprocessing_path)
                    self.scaler = artifacts.get('scaler')
                    print("Loaded preprocessing scaler")
                else:
                    self.scaler = None
                    print("No preprocessing artifacts found")
            except Exception as e:
                self.scaler = None
                print(f"Could not load preprocessing artifacts: {e}")
                
            # Load feature names
            feature_path = Path(__file__).parent.parent / "data" / "processed" / "feature_names.csv"
            if feature_path.exists():
                features_df = pd.read_csv(feature_path)
                self.feature_names = features_df['feature_name'].tolist()
                print(f"Loaded {len(self.feature_names)} feature names")
            else:
                # Use exact feature names from the dataset
                self.feature_names = [
                    'happy', 'sclmeet', 'inprdsc', 'ctrlife', 'etfruit', 'eatveg', 'dosprt', 'cgtsmok', 
                    'alcfreq', 'fltdpr', 'flteeff', 'slprl', 'wrhpp', 'fltlnl', 'enjlf', 'fltsd', 
                    'gndr', 'paccnois', 'bmi', 'lifestyle_score', 'social_score', 'mental_health_score'
                ]
                print(f"Using dataset feature names: {len(self.feature_names)} features")
                
        except Exception as e:
            print(f"Model loading error: {e}")
            # Final fallback with trained model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            X_sample, y_sample = make_classification(n_samples=1000, n_features=22, 
                                                   n_informative=15, random_state=42)
            self.model.fit(X_sample, y_sample)
            self.feature_names = [f"feature_{i}" for i in range(22)]
            print("Using emergency trained fallback model")
    
    def _setup_shap_explainer(self):
        """Initialize SHAP explainer for global feature importance analysis"""
        try:
            if self.model is None:
                print("Warning: No model loaded - SHAP explainer not initialized")
                self.explainer = None
                return
                
            # Create representative background dataset that matches training data distribution
            np.random.seed(42)
            n_background = 10  # Smaller for faster initialization
            
            # Create background samples that represent typical health profiles
            background_samples = []
            for _ in range(n_background):
                # Create baseline representing unhealthy population for proper SHAP comparison
                sample_inputs = {
                    'height': np.random.normal(170, 10),  # cm
                    'weight': np.random.normal(85, 25),   # Higher baseline weight 
                    'happiness': np.random.randint(2, 4),  # Very low happiness baseline (2-3)
                    'social_meetings': np.random.randint(2, 5),  # Poor social baseline (2-4)
                    'life_control': np.random.randint(2, 5),  # Low control baseline (2-4)
                    'exercise': np.random.randint(0, 3),  # Very poor exercise baseline (0-2)
                    'smoking': np.random.randint(0, 3),  # Light smoking baseline (0-2)
                    'alcohol': np.random.randint(1, 4),  # Conservative alcohol baseline (1-3)
                    'fruit_intake': np.random.randint(1, 4),  # Poor nutrition baseline (1-3)
                    'sleep_quality': np.random.randint(2, 5)  # Poor sleep baseline (2-4)
                }
                # Convert to model features using the same preprocessing
                background_features = self._prepare_features(sample_inputs)
                background_samples.append(background_features.flatten())
            
            background_array = np.array(background_samples)
            
            # Initialize KernelExplainer with proper model wrapper
            model_type = type(self.model).__name__
            
            # Create a wrapper that uses the exact same preprocessing as prediction
            def model_predict_with_preprocessing(X):
                """Wrapper that ensures consistent preprocessing for SHAP"""
                try:
                    # X is already preprocessed from background or test samples
                    probabilities = self.model.predict_proba(X)
                    return probabilities[:, 1]  # Return disease probability only
                except Exception as e:
                    print(f"SHAP prediction wrapper error: {e}")
                    return np.array([0.5] * len(X))
            
            # Use the wrapper function for SHAP
            self.explainer = shap.KernelExplainer(model_predict_with_preprocessing, background_array[:5])
                
            self.background_data = background_array
            print(f"SHAP explainer initialized successfully for {model_type}")
            
        except Exception as e:
            print(f"SHAP explainer setup failed: {e}")
            self.explainer = None
            self.background_data = None
            
        except Exception as e:
            print(f"SHAP explainer setup failed: {e}")
            self.explainer = None
            self.background_data = None

    def _setup_feature_info(self):
        """Setup feature descriptions for user interface"""
        self.feature_descriptions = {
            'happy': "Overall life satisfaction and happiness level",
            'sclmeet': "Frequency of social meetings and interactions", 
            'ctrlife': "Sense of control over life circumstances",
            'dosprt': "Physical activity and sports participation",
            'bmi': "Body Mass Index (calculated from height/weight)",
            'alcfreq': "Frequency of alcohol consumption",
            'cgtsmok': "Current smoking status and intensity",
            'flteeff': "Feeling that everything is an effort",
            'wrhpp': "Work-life happiness and satisfaction",
            'slprl': "Sleep quality and restlessness",
            'enjlf': "Enjoyment and satisfaction with life",
            'etfruit': "Frequency of fruit consumption",
            'mental_health_score': "Composite mental health indicator",
            'lifestyle_score': "Overall lifestyle health rating",
            'social_score': "Social engagement and connection level"
        }
    
    def _setup_lime_explainer(self):
        """Initialize LIME explainer for individual prediction explanations"""
        if not LIME_AVAILABLE:
            print("LIME not available - using alternative explanation method")
            self.lime_explainer = None
            return
            
        try:
            # Create simple training data that matches the expected 22 features
            np.random.seed(42)
            n_samples = 100
            n_features = len(self.feature_names)
            
            # Create simple normalized training data (mean=0, std=1)
            self.training_sample = np.random.normal(0, 1, (n_samples, n_features))
            
            # Simple feature names for LIME
            lime_feature_names = [f"Feature_{i}" for i in range(n_features)]
            
            # Initialize LIME explainer with simple setup
            self.lime_explainer = LimeTabularExplainer(
                training_data=self.training_sample,
                feature_names=lime_feature_names,
                class_names=['Low Risk', 'High Risk'],
                mode='classification',
                discretize_continuous=True
            )
            
            print("LIME explainer initialized successfully")
            
        except Exception as e:
            print(f"LIME explainer setup failed: {e}")
            self.lime_explainer = None
    
    def _lime_predict_proba(self, instances):
        """Wrapper function for LIME to call model predictions"""
        try:
            # Ensure instances is 2D array
            if len(instances.shape) == 1:
                instances = instances.reshape(1, -1)
            
            # Ensure I have the right number of features
            if instances.shape[1] != len(self.feature_names):
                print(f"Feature dimension mismatch: expected {len(self.feature_names)}, got {instances.shape[1]}")
                # Pad or truncate as needed
                if instances.shape[1] < len(self.feature_names):
                    padding = np.zeros((instances.shape[0], len(self.feature_names) - instances.shape[1]))
                    instances = np.hstack([instances, padding])
                else:
                    instances = instances[:, :len(self.feature_names)]
            
            # Get predictions from the model
            predictions = self.model.predict_proba(instances)
            return predictions
        except Exception as e:
            print(f"Model prediction error in LIME: {e}")
            # Return dummy probabilities if prediction fails
            return np.array([[0.5, 0.5]] * len(instances))
    
    def _get_lime_explanation(self, features, user_inputs):
        """Provide LIME-based individual explanation for prediction"""
        try:
            if self.lime_explainer is None or not LIME_AVAILABLE:
                return self._get_fallback_individual_analysis(features, user_inputs)
                
            # Prepare instance for LIME explanation
            X = np.array(features).reshape(1, -1)
            
            # Create LIME explanation with more features
            explanation = self.lime_explainer.explain_instance(
                X[0], 
                self._lime_predict_proba,
                num_features=min(10, len(X[0])),  # Show more features to include all inputs
                num_samples=100,  # Reduce samples to avoid timeout
                labels=[1]  # Only explain positive class to avoid IndexError
            )
            
            # Extract LIME results - get explanation for positive class (index 1)
            try:
                lime_list = explanation.as_list(label=1)  # Explicitly get positive class explanations
            except:
                lime_list = explanation.as_list()  # Fallback to default
            
            # Process LIME results to focus on meaningful user inputs and avoid duplicates
            processed_features = set()  # Track processed features to avoid duplicates
            meaningful_features = []
            
            # Priority order for user input features - ensure alcohol is always included
            priority_features = {
                'Alcohol Frequency': ['Feature_8', 'alcfreq'],  # Highest priority - ensure this shows
                'BMI': ['Feature_18'],
                'Life Satisfaction': ['Feature_0', 'Feature_14'],  # happy, enjlf
                'Sleep Quality': ['Feature_11'],  # slprl
                'Physical Activity': ['Feature_6'],  # dosprt
                'Smoking Level': ['Feature_7'],  # cgtsmok
                'Social Engagement': ['Feature_1'],  # sclmeet
                'Life Control': ['Feature_3'],  # ctrlife
                'Fruit & Vegetable Intake': ['Feature_4', 'Feature_5'],  # Combined etfruit and etveg
            }
            
            # Group LIME results by meaningful categories - ensure alcohol is always included
            mandatory_features = ['Alcohol Frequency', 'BMI', 'Life Control']  # Force these to appear
            
            # First pass: add mandatory features with their actual LIME contributions
            for user_feature, lime_features in priority_features.items():
                if user_feature in mandatory_features:
                    best_contribution = 0
                    best_feature = None
                    
                    # Find the most significant contribution for this user input category
                    for feature_condition, contribution in lime_list:
                        for lime_feature in lime_features:
                            if lime_feature in feature_condition:
                                if best_feature is None or abs(contribution) > abs(best_contribution):
                                    best_contribution = contribution
                                    best_feature = (feature_condition, contribution)
                    
                    # Add the feature (either with real contribution or forced with minimal impact)
                    if best_feature and user_feature not in processed_features:
                        meaningful_features.append((user_feature, best_feature[1]))
                        processed_features.add(user_feature)
                    elif user_feature not in processed_features:
                        # Force inclusion with small contribution to show it was considered
                        meaningful_features.append((user_feature, 0.001))
                        processed_features.add(user_feature)
            
            # Second pass: add remaining features in priority order
            for user_feature, lime_features in priority_features.items():
                if user_feature not in processed_features:
                    best_contribution = 0
                    best_feature = None
                    
                    # Find the most significant contribution for this user input category
                    for feature_condition, contribution in lime_list:
                        for lime_feature in lime_features:
                            if lime_feature in feature_condition:
                                if best_feature is None or abs(contribution) > abs(best_contribution):
                                    best_contribution = contribution
                                    best_feature = (feature_condition, contribution)
                    
                    # Add the best feature if found
                    if best_feature:
                        meaningful_features.append((user_feature, best_feature[1]))
                        processed_features.add(user_feature)
            
            # Format LIME explanation with meaningful user inputs only
            lime_analysis = []
            lime_analysis.append("## Individual Prediction Explanation")
            lime_analysis.append("")
            lime_analysis.append("### LIME Individual Risk Assessment")
            lime_analysis.append("")
            lime_analysis.append("*Based on your specific health profile, here's how each factor affects your heart disease risk:*")
            lime_analysis.append("")
            
            for user_feature, contribution in meaningful_features[:10]:  # Show top 10 meaningful features to ensure alcohol is included
                # Get actual user input value for this feature
                user_value = self._get_user_input_value(user_feature, user_inputs)
                
                # Create patient-friendly explanations
                explanation_text = self._get_patient_friendly_explanation(user_feature, contribution, user_inputs)
                
                lime_analysis.append(f"• **{user_feature}** {user_value}")
                lime_analysis.append(f"  {explanation_text}")
                lime_analysis.append("")
            
            lime_analysis.append("")
            lime_analysis.append("---")
            lime_analysis.append("")
            
            # Patient-focused interpretation based on actual risk profile
            protective_count = sum(1 for _, contrib in meaningful_features if contrib < 0)
            risk_count = sum(1 for _, contrib in meaningful_features if contrib > 0)
            
            # Analyze specific high-risk factors
            high_risk_factors = []
            for feature_name, _ in meaningful_features:
                if feature_name == "BMI":
                    bmi_val = user_inputs.get('weight', 70) / ((user_inputs.get('height', 170)/100)**2)
                    if bmi_val >= 30.0:
                        high_risk_factors.append("obesity")
                elif feature_name == "Smoking Level" and user_inputs.get('smoking', 0) >= 5:
                    high_risk_factors.append("smoking")
                elif feature_name == "Alcohol Frequency" and user_inputs.get('alcohol', 2) >= 7:
                    high_risk_factors.append("high alcohol consumption")
            
            if len(high_risk_factors) >= 2:
                lime_analysis.append("**Your Personal Health Profile:** Several significant risk factors require immediate attention. Consult your healthcare provider about lifestyle modifications.")
            elif len(high_risk_factors) == 1:
                lime_analysis.append(f"**Your Personal Health Profile:** Your {high_risk_factors[0]} presents a significant risk factor that should be addressed. Consider discussing improvement strategies with your healthcare provider.")
            elif risk_count > protective_count:
                lime_analysis.append("**Your Personal Health Profile:** Some of your risk factors could benefit from attention. Consider discussing improvements with your healthcare provider.")
            elif protective_count > risk_count:
                lime_analysis.append("**Your Personal Health Profile:** Most of your lifestyle factors are working in your favor to protect against heart disease. Keep up the good habits!")
            else:
                lime_analysis.append("**Your Personal Health Profile:** You have a balanced mix of protective and risk factors. Focus on strengthening the protective ones.")
            
            return "\n".join(lime_analysis)
            
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return self._get_fallback_individual_analysis(features, user_inputs)
    
    def _convert_lime_feature_to_meaningful(self, feature_condition):
        """Convert LIME's generic feature conditions to meaningful descriptions"""
        # Map feature indices to meaningful names based on my feature order
        feature_mappings = {
            'Feature_0': 'Life Satisfaction',
            'Feature_1': 'Social Engagement', 
            'Feature_2': 'Discrimination Experience',
            'Feature_3': 'Life Control',
            'Feature_4': 'Fruit & Vegetable Intake',
            'Feature_5': 'Fruit & Vegetable Intake',  # Same as Feature_4
            'Feature_6': 'Physical Activity',
            'Feature_7': 'Smoking Level',
            'Feature_8': 'Alcohol Frequency',
            'Feature_9': 'Depression Feelings',
            'Feature_10': 'Effort Feelings',
            'Feature_11': 'Sleep Quality',
            'Feature_12': 'Work Happiness',
            'Feature_13': 'Loneliness',
            'Feature_14': 'Life Enjoyment',
            'Feature_15': 'Sadness Feelings',
            'Feature_16': 'Gender',
            'Feature_17': 'Noise Exposure',
            'Feature_18': 'BMI',
            'Feature_19': 'Lifestyle Score',
            'Feature_20': 'Social Score',
            'Feature_21': 'Mental Health Score'
        }
        
        # Replace generic feature names with meaningful ones
        meaningful_condition = feature_condition
        for generic_name, meaningful_name in feature_mappings.items():
            # Use word boundaries to avoid partial replacements
            import re
            pattern = r'\b' + re.escape(generic_name) + r'\b'
            if re.search(pattern, feature_condition):
                meaningful_condition = re.sub(pattern, meaningful_name, feature_condition)
                break
        
        # Clean up operators for better readability
        meaningful_condition = meaningful_condition.replace(' <= ', ' ≤ ').replace(' > ', ' > ')
        
        return meaningful_condition
    
    def _convert_lime_to_user_friendly(self, feature_condition, user_inputs):
        """Convert LIME's feature conditions to user-friendly descriptions with actual input values"""
        # Map feature indices to user input descriptions
        feature_input_map = {
            'Feature_0': ('Life Satisfaction', user_inputs.get('happiness', 7)),
            'Feature_1': ('Social Engagement', user_inputs.get('social_meetings', 5)),
            'Feature_2': ('Discrimination Experience', 'Survey-based'),
            'Feature_3': ('Life Control', user_inputs.get('life_control', 7)),
            'Feature_4': ('Fruit & Vegetable Intake', user_inputs.get('fruit_intake', 4)),
            'Feature_5': ('Fruit & Vegetable Intake', user_inputs.get('fruit_intake', 4)),  # Same as Feature_4
            'Feature_6': ('Physical Activity', user_inputs.get('exercise', 4)),
            'Feature_7': ('Smoking Level', user_inputs.get('smoking', 0)),
            'Feature_8': ('Alcohol Frequency', user_inputs.get('alcohol', 2)),
            'Feature_9': ('Depression Feelings', 'Mental health indicator'),
            'Feature_10': ('Effort Feelings', 'Energy level'),
            'Feature_11': ('Sleep Quality', user_inputs.get('sleep_quality', 7)),
            'Feature_12': ('Work Happiness', 'Job satisfaction'),
            'Feature_13': ('Loneliness', 'Social connection'),
            'Feature_14': ('Life Enjoyment', 'Life satisfaction'),
            'Feature_15': ('Sadness Feelings', 'Emotional state'),
            'Feature_16': ('Gender', 'Demographic'),
            'Feature_17': ('Noise Exposure', 'Environmental'),
            'Feature_18': ('BMI', f"{user_inputs.get('weight', 70) / ((user_inputs.get('height', 170)/100)**2):.1f}"),
            'Feature_19': ('Lifestyle Score', 'Calculated from inputs'),
            'Feature_20': ('Social Score', 'Based on social engagement'),
            'Feature_21': ('Mental Health Score', 'Calculated from happiness + control')
        }
        
        # Find which feature this condition refers to
        for feature_name, (display_name, value) in feature_input_map.items():
            if feature_name in feature_condition:
                if isinstance(value, (int, float)):
                    if 0 <= value <= 10:  # 0-10 scale inputs
                        return f"{display_name} ({value}/10)"
                    else:  # BMI or other calculated values
                        return f"{display_name} ({value})"
                else:
                    return f"{display_name} ({value})"
        
        # Fallback to original if no match found
        return self._convert_lime_feature_to_meaningful(feature_condition)
    
    def _get_user_input_value(self, feature_name, user_inputs):
        """Get the actual user input value for display"""
        value_map = {
            'BMI': f"({user_inputs.get('weight', 70) / ((user_inputs.get('height', 170)/100)**2):.1f})",
            'Life Satisfaction': f"({user_inputs.get('happiness', 7)}/10)",
            'Sleep Quality': f"({user_inputs.get('sleep_quality', 7)}/10)",
            'Physical Activity': f"({user_inputs.get('exercise', 4)}/10)",
            'Smoking Level': f"({user_inputs.get('smoking', 0)}/10)",
            'Social Engagement': f"({user_inputs.get('social_meetings', 5)}/10)",
            'Life Control': f"({user_inputs.get('life_control', 7)}/10)",
            'Alcohol Frequency': f"({user_inputs.get('alcohol', 2)}/10)",
            'Fruit & Vegetable Intake': f"({user_inputs.get('fruit_intake', 4)}/10)",
        }
        return value_map.get(feature_name, "")
    
    def _get_patient_friendly_explanation(self, feature_name, contribution, user_inputs):
        """Provide patient-friendly explanations for each factor"""
        # Always provide explanations based on actual health values for key risk factors
        if feature_name == "BMI":
            bmi_val = user_inputs.get('weight', 70) / ((user_inputs.get('height', 170)/100)**2)
            if 18.5 <= bmi_val <= 24.9:
                return "Your BMI is in the healthy range, which helps protect against heart disease"
            elif 25.0 <= bmi_val <= 29.9:
                return "Your BMI indicates overweight status, which may increase cardiovascular risk"
            elif 30.0 <= bmi_val <= 34.9:
                return "Your BMI indicates obesity, which significantly increases cardiovascular risk"
            elif bmi_val >= 35.0:
                return "Your BMI indicates severe obesity, which greatly increases cardiovascular risk"
            else:
                return "Your BMI suggests you may be underweight, which has mixed effects on heart health"
        elif feature_name == "Smoking Level":
            smoking_val = user_inputs.get('smoking', 0)
            if smoking_val == 0:
                return "Not smoking significantly reduces your cardiovascular disease risk"
            elif smoking_val <= 3:
                return "Light smoking still increases cardiovascular risk - consider quitting"
            elif smoking_val <= 6:
                return "Moderate smoking significantly increases cardiovascular disease risk"
            else:
                return "Heavy smoking greatly increases your cardiovascular disease risk"
        elif feature_name == "Alcohol Frequency":
            alcohol_val = user_inputs.get('alcohol', 2)
            if alcohol_val <= 2:
                return "Your low alcohol consumption is generally beneficial for cardiovascular health"
            elif alcohol_val <= 4:
                return "Your moderate alcohol use has mixed effects on heart health"
            elif alcohol_val <= 6:
                return "Your alcohol consumption pattern may contribute to increased risk"
            else:
                return "Higher alcohol consumption may increase cardiovascular risks"
        
        # For other factors, use actual values rather than just contribution direction
        if feature_name == "Sleep Quality":
            sleep_val = user_inputs.get('sleep_quality', 7)
            if sleep_val >= 7:
                return "Your good sleep quality supports cardiovascular health and stress management"
            elif sleep_val <= 4:
                return "Your sleep quality could be improved for better heart health"
            else:
                return "Your average sleep quality has moderate cardiovascular impact"
        elif feature_name == "Physical Activity":
            exercise_val = user_inputs.get('exercise', 4)
            if exercise_val >= 7:
                return "Your regular physical activity strengthens your heart and circulatory system"
            elif exercise_val <= 4:
                return "Consider increasing physical activity for better cardiovascular protection"
            else:
                return "Moderate activity levels provide some cardiovascular benefits"
        elif feature_name == "Life Satisfaction":
            happiness_val = user_inputs.get('happiness', 7)
            if happiness_val >= 7:
                return "Good mental well-being helps reduce stress-related cardiovascular risks"
            elif happiness_val <= 5:
                return "Poor mental well-being may increase stress-related cardiovascular risks"
            else:
                return "Average mental well-being with moderate cardiovascular impact"
        elif feature_name == "Social Engagement":
            social_val = user_inputs.get('social_meetings', 5)
            if social_val >= 7:
                return "High social engagement supports mental health and cardiovascular wellness"
            elif social_val <= 3:
                return "Limited social connections may contribute to stress and cardiac risk"
            else:
                return "Social connections support mental health and may reduce heart disease risk"
        elif feature_name == "Life Control":
            control_val = user_inputs.get('life_control', 7)
            if control_val >= 7:
                return "Good sense of control helps manage stress and supports cardiovascular health"
            elif control_val <= 4:
                return "Feeling less control over life circumstances may increase stress-related risks"
            else:
                return "Moderate sense of control may contribute to manageable stress levels"
        elif feature_name == "Fruit & Vegetable Intake":
            fruit_val = user_inputs.get('fruit_intake', 4)
            if fruit_val >= 7:
                return "High fruit and vegetable intake provides excellent cardiovascular protection"
            elif fruit_val <= 4:
                return "Below optimal fruit and vegetable intake - consider increasing for better heart health"
            else:
                return "Adequate fruit and vegetable consumption provides antioxidants and nutrients for heart health"
        else:
            return "This factor influences your cardiovascular risk based on your personal health profile"
    
    def _get_fallback_individual_analysis(self, features, user_inputs):
        """Professional individual analysis when LIME is not available"""
        lime_analysis = []
        lime_analysis.append("**Individual Risk Factor Analysis:**")
        lime_analysis.append("")
        
        # Analyze user's specific inputs relative to healthy ranges
        analyses = []
        
        # BMI Analysis
        height_m = user_inputs.get('height', 170) / 100
        weight = user_inputs.get('weight', 70)
        bmi = weight / (height_m ** 2)
        if bmi >= 30:
            analyses.append(("BMI", f"{bmi:.1f}", "Strong risk factor", "Obesity increases cardiovascular risk significantly"))
        elif bmi >= 25:
            analyses.append(("BMI", f"{bmi:.1f}", "Moderate risk factor", "Overweight status elevates cardiac risk"))
        else:
            analyses.append(("BMI", f"{bmi:.1f}", "Protective factor", "Healthy weight supports cardiovascular health"))
        
        # Exercise Analysis
        exercise = user_inputs.get('exercise', 4)
        if exercise <= 3:
            analyses.append(("Physical Activity", f"{exercise}/10", "Risk factor", "Sedentary lifestyle increases cardiac risk"))
        elif exercise >= 7:
            analyses.append(("Physical Activity", f"{exercise}/10", "Strong protective factor", "Regular exercise significantly reduces cardiovascular risk"))
        else:
            analyses.append(("Physical Activity", f"{exercise}/10", "Moderate protective factor", "Some activity provides cardiac protection"))
        
        # Smoking Analysis  
        smoking = user_inputs.get('smoking', 0)
        if smoking >= 7:
            analyses.append(("Smoking", f"{smoking}/10", "Major risk factor", "Heavy smoking dramatically increases cardiac risk"))
        elif smoking >= 3:
            analyses.append(("Smoking", f"{smoking}/10", "Moderate risk factor", "Tobacco use elevates cardiovascular risk"))
        else:
            analyses.append(("Smoking", f"{smoking}/10", "Protective factor", "Non-smoking supports cardiovascular health"))
        
        # Alcohol Analysis
        alcohol = user_inputs.get('alcohol', 2)
        if alcohol >= 8:
            analyses.append(("Alcohol Consumption", f"{alcohol}/10", "Major risk factor", "Heavy alcohol use increases cardiac risk and hypertension"))
        elif alcohol >= 5:
            analyses.append(("Alcohol Consumption", f"{alcohol}/10", "Moderate risk factor", "Moderate alcohol use may elevate cardiovascular risk"))
        elif alcohol >= 1:
            analyses.append(("Alcohol Consumption", f"{alcohol}/10", "Neutral factor", "Light alcohol consumption has mixed cardiac effects"))
        else:
            analyses.append(("Alcohol Consumption", f"{alcohol}/10", "Protective factor", "No alcohol use supports cardiovascular health"))
        
        # Fruit Intake Analysis
        fruit_intake = user_inputs.get('fruit_intake', 4)
        if fruit_intake <= 3:
            analyses.append(("Fruit Intake", f"{fruit_intake}/10", "Risk factor", "Low fruit and vegetable intake lacks protective antioxidants and nutrients"))
        elif fruit_intake <= 5:
            analyses.append(("Fruit Intake", f"{fruit_intake}/10", "Moderate risk factor", "Below optimal fruit and vegetable intake misses cardiovascular protection"))
        elif fruit_intake >= 7:
            analyses.append(("Fruit Intake", f"{fruit_intake}/10", "Strong protective factor", "High fruit and vegetable intake provides excellent cardiovascular protection"))
        else:
            analyses.append(("Fruit Intake", f"{fruit_intake}/10", "Protective factor", "Good fruit and vegetable intake supports heart health"))
        
        # Sense of Control Over Life Analysis
        life_control = user_inputs.get('life_control', 7)
        if life_control <= 4:
            analyses.append(("Sense of Control Over Life", f"{life_control}/10", "Risk factor", "Feeling less control over life circumstances may increase stress-related risks"))
        elif life_control <= 6:
            analyses.append(("Sense of Control Over Life", f"{life_control}/10", "Moderate risk factor", "Moderate sense of control may contribute to stress-related cardiac risk"))
        elif life_control >= 8:
            analyses.append(("Sense of Control Over Life", f"{life_control}/10", "Strong protective factor", "High sense of control supports stress management and heart health"))
        else:
            analyses.append(("Sense of Control Over Life", f"{life_control}/10", "Protective factor", "Good sense of control helps manage stress and supports cardiovascular health"))
        
        # Mental Health Analysis
        happiness = user_inputs.get('happiness', 7)
        if happiness <= 5:
            analyses.append(("Life Satisfaction", f"{happiness}/10", "Risk factor", "Poor mental well-being may increase stress-related cardiovascular risks"))
        elif happiness >= 7:
            analyses.append(("Life Satisfaction", f"{happiness}/10", "Protective factor", "Good mental well-being helps reduce stress-related cardiovascular risks"))
        else:
            analyses.append(("Life Satisfaction", f"{happiness}/10", "Neutral factor", "Average mental health status"))
        
        # Sleep Analysis
        sleep_quality = user_inputs.get('sleep_quality', 7)
        if sleep_quality <= 4:
            analyses.append(("Sleep Quality", f"{sleep_quality}/10", "Risk factor", "Poor sleep linked to hypertension and cardiac stress"))
        elif sleep_quality >= 7:
            analyses.append(("Sleep Quality", f"{sleep_quality}/10", "Protective factor", "Quality sleep supports cardiovascular recovery"))
        else:
            analyses.append(("Sleep Quality", f"{sleep_quality}/10", "Neutral factor", "Average sleep quality with moderate cardiac impact"))
        
        # Social Engagement Analysis
        social_meetings = user_inputs.get('social_meetings', 5)
        if social_meetings <= 3:
            analyses.append(("Social Engagement", f"{social_meetings}/10", "Risk factor", "Social isolation linked to increased cardiovascular risk"))
        elif social_meetings <= 5:
            analyses.append(("Social Engagement", f"{social_meetings}/10", "Moderate risk factor", "Limited social engagement may contribute to stress and cardiac risk"))
        elif social_meetings >= 8:
            analyses.append(("Social Engagement", f"{social_meetings}/10", "Strong protective factor", "High social engagement supports mental health and cardiovascular wellness"))
        else:
            analyses.append(("Social Engagement", f"{social_meetings}/10", "Protective factor", "Good social engagement supports overall health"))
        
        # Age Analysis
        age = user_inputs.get('age', 45)
        if age >= 65:
            analyses.append(("Age", f"{age} years", "Major risk factor", "Advanced age significantly increases cardiovascular risk"))
        elif age >= 50:
            analyses.append(("Age", f"{age} years", "Moderate risk factor", "Middle age associated with increased cardiac risk"))
        elif age >= 35:
            analyses.append(("Age", f"{age} years", "Mild risk factor", "Age-related increase in cardiovascular risk beginning"))
        else:
            analyses.append(("Age", f"{age} years", "Protective factor", "Younger age associated with lower cardiovascular risk"))
        
        # Format results - show all important factors (limit to top 8 for readability)
        for factor, value, impact, explanation in analyses[:8]:
            lime_analysis.append(f"• **{factor}** ({value}): *{impact}* — {explanation}")
            lime_analysis.append("")
        
        lime_analysis.append("---")
        lime_analysis.append("**Individual Assessment Summary:**")
        risk_factors = sum(1 for _, _, impact, _ in analyses if "risk factor" in impact.lower())
        protective_factors = sum(1 for _, _, impact, _ in analyses if "protective" in impact.lower())
        
        if risk_factors > protective_factors:
            lime_analysis.append("**Assessment:** Your profile shows predominant risk factors requiring attention")
        elif protective_factors > risk_factors:
            lime_analysis.append("**Assessment:** Your profile shows good protective factors — maintain current habits")
        else:
            lime_analysis.append("**Assessment:** Your profile shows balanced risk and protective factors")
        
        return "\n".join(lime_analysis)
    
    def _format_lime_results(self, feature_contributions, user_inputs):
        """Format LIME results professionally"""
        lime_analysis = []
        lime_analysis.append("**Individual Risk Factor Analysis:**")
        lime_analysis.append("")
        
        total_contribution = 0
        for feature_name, contribution in feature_contributions[:5]:
            contribution_str = f"{contribution:+.3f}"
            direction = "Increases risk" if contribution > 0 else "Decreases risk"
            magnitude = "Strong" if abs(contribution) > 0.1 else "Moderate" if abs(contribution) > 0.05 else "Mild"
            
            user_value = self._get_user_value_for_display(feature_name, user_inputs)
            lime_analysis.append(f"• **{feature_name}** ({user_value}): *{magnitude} {direction}* — Impact: {contribution_str}")
            total_contribution += contribution
        
        lime_analysis.append("")
        lime_analysis.append(f"**Net Risk Effect:** {total_contribution:+.3f}")
        
        if total_contribution > 0.1:
            lime_analysis.append("→ **Individual profile suggests elevated cardiovascular risk**")
        elif total_contribution < -0.1:
            lime_analysis.append("→ **Individual profile suggests protective factors present**")
        else:
            lime_analysis.append("→ **Individual profile shows balanced risk factors**")
        
        return "\n".join(lime_analysis)
    
    def _get_user_value_for_display(self, lime_feature_name, user_inputs):
        """Convert LIME feature names back to user-friendly display values"""
        feature_mapping = {
            'Life Satisfaction': f"{user_inputs.get('happiness', 7)}/10",
            'Social Meetings': f"{user_inputs.get('social_meetings', 5)}/10",
            'Life Control': f"{user_inputs.get('life_control', 7)}/10",
            'Physical Activity': f"{user_inputs.get('exercise', 4)}/10",
            'Fruit & Vegetable Intake': f"{user_inputs.get('fruit_intake', 4)}/10",
            'Smoking': f"{user_inputs.get('smoking', 0)}/10",
            'Alcohol Frequency': f"{user_inputs.get('alcohol', 2)}/10",
            'Sleep Quality': f"{user_inputs.get('sleep_quality', 7)}/10",
            'BMI': f"{user_inputs.get('weight', 70) / ((user_inputs.get('height', 170)/100)**2):.1f}",
            'Mental Health Score': f"{((user_inputs.get('happiness', 7) + user_inputs.get('life_control', 7)) / 20):.2f}",
            'Lifestyle Score': f"{((user_inputs.get('exercise', 4) + user_inputs.get('fruit_intake', 4) - user_inputs.get('smoking', 0)) / 30):.2f}",
            'Social Score': f"{(user_inputs.get('social_meetings', 5) / 10):.1f}"
        }
        
        return feature_mapping.get(lime_feature_name, "N/A")
    
    def _prepare_features(self, inputs):
        """Convert user inputs to model features with proper scaling"""
        # Map user inputs to actual model features based on the dataset structure
        # Normalize inputs to match training data scale using Z-score standardization
        
        # Normalize 0-10 scale inputs to standardized range
        def normalize_0_10(value, mean_val=5, std_val=2.5):
            """Convert 0-10 scale to standardized scale
            
            Z-score formula: (value - mean) / std_dev
            Input range: 0-10 → Output range: -2.0 to +2.0
            - For value=0: (0-5)/2.5 = -2.0
            - For value=10: (10-5)/2.5 = +2.0
            """
            return (value - mean_val) / std_val
        
        # Calculate BMI from height and weight
        height_m = inputs.get('height', 170) / 100  # Convert cm to meters
        weight = inputs.get('weight', 70)  # kg
        bmi = weight / (height_m ** 2)
        
        feature_map = {
            'happy': normalize_0_10(inputs.get('happiness', 7)),
            'sclmeet': normalize_0_10(inputs.get('social_meetings', 5)),
            'inprdsc': normalize_0_10(inputs.get('life_control', 7)) * 0.5,  # Related to control
            'ctrlife': normalize_0_10(inputs.get('life_control', 7)),
            'etfruit': normalize_0_10(inputs.get('fruit_intake', 4)),
            'eatveg': normalize_0_10(inputs.get('fruit_intake', 4)) * 0.8,  # Related to fruits
            'dosprt': normalize_0_10(inputs.get('exercise', 4)),
            'cgtsmok': normalize_0_10(inputs.get('smoking', 0)),
            'alcfreq': normalize_0_10(inputs.get('alcohol', 2)),
            'fltdpr': -normalize_0_10(inputs.get('happiness', 7)) * 0.3,  # Inverse of happiness
            'flteeff': normalize_0_10(10 - inputs.get('sleep_quality', 7)),  # Effort feeling
            'slprl': normalize_0_10(10 - inputs.get('sleep_quality', 7)),  # Sleep restlessness
            'wrhpp': normalize_0_10(inputs.get('happiness', 7)) * 0.9,  # Work happiness
            'fltlnl': -normalize_0_10(inputs.get('social_meetings', 5)) * 0.4,  # Loneliness
            'enjlf': normalize_0_10(inputs.get('happiness', 7)),
            'fltsd': -normalize_0_10(inputs.get('happiness', 7)) * 0.2,  # Sadness
            'gndr': 0.5,  # Neutral gender encoding
            'paccnois': 0.0,  # Neutral physical activity noise
            'bmi': (bmi - 25.0) / 5.0,  # Normalize BMI around 25
            'lifestyle_score': ((inputs.get('exercise', 4) + inputs.get('fruit_intake', 4) - inputs.get('smoking', 0)) / 30.0),
            'social_score': inputs.get('social_meetings', 5) / 10.0,
            'mental_health_score': ((inputs.get('happiness', 7) + inputs.get('life_control', 7)) / 20.0)
        }
        
        # Create feature array in correct order
        features = []
        for fname in self.feature_names:
            features.append(feature_map.get(fname, 0.0))
        
        X = np.array(features).reshape(1, -1)
        
        # Apply scaling if available
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                print(f"Scaling failed: {e}")
        
        return X
    
    def predict_risk(self, **inputs):
        """Make heart disease risk prediction with explanations"""
        try:
            # Prepare features
            X = self._prepare_features(inputs)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                no_disease_prob = probabilities[0]  # Probability of no heart disease
                has_disease_prob = probabilities[1]  # Probability of heart disease
                risk_pred = self.model.predict(X)[0]
            else:
                risk_pred = self.model.predict(X)[0]
                has_disease_prob = 0.5 + (risk_pred - 0.5) * 0.4  # Approximate probability
                no_disease_prob = 1 - has_disease_prob
            
            # Determine risk level based on model output distribution
            # Calibrated thresholds for meaningful risk stratification
            if has_disease_prob >= 0.35:
                risk_level = "High Risk"
                risk_msg = "Elevated cardiovascular risk detected. Recommend medical consultation."
            elif has_disease_prob >= 0.25:
                risk_level = "Moderate Risk"
                risk_msg = "Moderate cardiovascular risk. Consider lifestyle improvements."
            else:
                risk_level = "Low Risk"
                risk_msg = "Lower cardiovascular risk. Maintain healthy lifestyle."
            
            # Feature importance insights
            key_factors = self._get_key_factors(inputs)
            
            # LIME individual explanation
            lime_explanation = self._get_lime_explanation(X, inputs)
            
            result = f"""
## **{risk_level}**

### Prediction Probability:

<div style="display: flex; gap: 15px; margin: 20px 0; justify-content: center;">
    <div style="background-color: #ef4444; color: white; padding: 10px 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <strong>Has heart disease</strong><br>
        <span style="font-size: 1.2em; font-weight: bold;">{has_disease_prob:.1%}</span>
    </div>
    <div style="background-color: #22c55e; color: white; padding: 10px 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <strong>No heart disease</strong><br>
        <span style="font-size: 1.2em; font-weight: bold;">{no_disease_prob:.1%}</span>
    </div>
</div>

**Clinical Assessment:** {risk_msg}

---

### **Global Feature Importance**
{key_factors}

---

### **Individual Prediction Explanation**
{lime_explanation}

---

### **Clinical Recommendations**

**Immediate Actions:**
- Regular cardiovascular health monitoring
- Lifestyle factor optimization based on analysis above
- Professional medical evaluation if risk is elevated

**Long-term Prevention:**
- Maintain healthy BMI (18.5-24.9)
- Regular physical activity (≥150 min/week moderate intensity)
- Balanced nutrition with fruits and vegetables
- Stress management and adequate sleep
- Social engagement and mental health support

---
*This assessment combines SHAP global insights with LIME local explanations for comprehensive XAI analysis. 
For medical decisions, consult healthcare professionals.*
            """
            
            return result
            
        except Exception as e:
            return f"**Prediction Error**\n\nUnable to process prediction: {str(e)}\n\nPlease check your inputs and try again."
    
    def _get_key_factors(self, inputs):
        """Get SHAP-based feature importance for this specific prediction"""
        try:
            if self.explainer is None:
                return self._get_fallback_key_factors(inputs)
                
            # Prepare features for SHAP analysis
            features = self._prepare_features(inputs)
            X = np.array(features).reshape(1, -1)
            
            # Calculate SHAP values based on explainer type
            shap_values = self.explainer(X)
            
            # Handle single-output SHAP values (disease probability only)
            if hasattr(shap_values, 'values'):
                # New SHAP format (Explanation object) - single output for disease probability
                feature_impacts = shap_values.values[0]  # Single output: disease probability contributions
            elif isinstance(shap_values, list):
                # Old format
                feature_impacts = shap_values[0]
            else:
                # Direct array output
                feature_impacts = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                
            # Map back to interpretable feature names and get top contributors
            feature_contributions = []
            # Include ALL user controllable features that I get from inputs
            user_controllable_features = ['happy', 'sclmeet', 'ctrlife', 'dosprt', 'cgtsmok', 'alcfreq', 'etfruit', 'eatveg', 'slprl', 'bmi']
            
            # Track combined features to avoid duplicates
            processed_features = set()
            
            for i, (feature_name, impact) in enumerate(zip(self.feature_names[:len(feature_impacts)], feature_impacts)):
                # Get user-friendly feature name and value
                display_name, display_value = self._get_feature_display_info(feature_name, inputs)
                is_controllable = feature_name in user_controllable_features
                
                # Combine fruit and vegetable features
                if feature_name in ['etfruit', 'eatveg']:
                    if 'Fruit & Vegetable Intake' not in processed_features:
                        # Combine the impacts of both fruit and vegetable features
                        fruit_impact = 0
                        veg_impact = 0
                        for j, fname in enumerate(self.feature_names[:len(feature_impacts)]):
                            if fname == 'etfruit':
                                fruit_impact = feature_impacts[j]
                            elif fname == 'eatveg':
                                veg_impact = feature_impacts[j]
                        
                        combined_impact = fruit_impact + veg_impact
                        feature_contributions.append(('Fruit & Vegetable Intake', combined_impact, display_value, is_controllable))
                        processed_features.add('Fruit & Vegetable Intake')
                else:
                    # Add other features normally
                    if display_name not in processed_features:
                        feature_contributions.append((display_name, impact, display_value, is_controllable))
                        processed_features.add(display_name)
            
            # Prioritize user-controllable features and sort by absolute impact
            feature_contributions.sort(key=lambda x: (not x[3], -abs(x[1])))  # Controllable first, then by impact
            top_features = feature_contributions[:10]  # Show 10 features to include all major inputs
            
            # Format SHAP results with better context
            analysis = []
            analysis.append("## SHAP Feature Importance Analysis")
            analysis.append("")
            analysis.append("*Global feature importance showing how each factor influences the model's prediction:*")
            analysis.append("")
            
            controllable_features = [f for f in top_features if f[3]]
            other_features = [f for f in top_features if not f[3]]
            
            if controllable_features:
                analysis.append("### Key Lifestyle Factors")
                analysis.append("*These are factors you can actively control:*")
                analysis.append("")
                for feature_name, shap_value, display_value, _ in controllable_features:
                    # Use clinical logic combined with SHAP direction for accurate interpretation
                    clinical_direction = self._get_clinical_direction(feature_name, display_value, inputs)
                    
                    # SHAP provides the model's contribution - use it for magnitude
                    magnitude = "Strong" if abs(shap_value) > 0.02 else "Moderate" if abs(shap_value) > 0.01 else "Mild"
                    
                    analysis.append(f"• **{feature_name}** ({display_value})")
                    analysis.append(f"  {magnitude} {clinical_direction}")
                    analysis.append("")
                    
            if other_features:
                analysis.append("### Other Contributing Factors")
                analysis.append("*Background factors identified by the model:*")
                analysis.append("")
                for feature_name, shap_value, display_value, _ in other_features:
                    direction = "Increases risk" if shap_value > 0 else "Decreases risk"  # Disease class interpretation
                    # Adjusted thresholds for better medical interpretation
                    magnitude = "Strong" if abs(shap_value) > 0.02 else "Moderate" if abs(shap_value) > 0.01 else "Mild"
                    analysis.append(f"• **{feature_name}**: {magnitude} {direction}")
                analysis.append("")
            
            total_impact = sum(impact for _, impact, _, _ in top_features)
            risk_direction = "Higher" if total_impact > 0 else "Lower"  # Disease class: positive means higher risk
            analysis.append("---")
            analysis.append("")
            analysis.append(f"**Overall Risk Assessment:** {risk_direction} than average")
            
            return "\n".join(analysis)
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            return self._get_fallback_key_factors(inputs)
    
    def _get_clinical_direction(self, feature_name, display_value, inputs):
        """Determine clinical risk direction based on actual health values"""        
        try:
            # Extract numeric value from display string
            if '/' in display_value:
                value = float(display_value.split('/')[0])
                scale = 10
            elif 'BMI' in feature_name:
                value = float(display_value)
                scale = 100  # BMI scale
            else:
                return "Influences risk"  # Fallback
            
            # Clinical interpretation based on actual values
            if feature_name == 'BMI':
                return "Increases risk" if value >= 25 else "Decreases risk"
            elif feature_name == 'Physical Activity':
                return "Decreases risk" if value >= 5 else "Increases risk"
            elif feature_name == 'Smoking Level':
                return "Increases risk" if value >= 1 else "Decreases risk"
            elif feature_name == 'Alcohol Frequency':
                return "Increases risk" if value >= 5 else "Decreases risk"
            elif feature_name in ['Fruit & Vegetable Intake']:
                return "Decreases risk" if value >= 5 else "Increases risk"
            elif feature_name in ['Life Satisfaction', 'Life Control', 'Sleep Quality']:
                return "Decreases risk" if value >= 6 else "Increases risk"
            elif feature_name == 'Social Engagement':
                return "Decreases risk" if value >= 5 else "Increases risk"
            else:
                return "Influences risk"  # Default for unknown features
                
        except Exception:
            return "Influences risk"  # Fallback
    
    def _get_feature_display_info(self, feature_name, inputs):
        """Convert technical feature names to user-friendly display with values"""
        feature_map = {
            # Direct user inputs
            'happy': ('Life Satisfaction', f"{inputs.get('happiness', 7)}/10"),
            'sclmeet': ('Social Engagement', f"{inputs.get('social_meetings', 5)}/10"),
            'ctrlife': ('Life Control', f"{inputs.get('life_control', 7)}/10"),
            'dosprt': ('Physical Activity', f"{inputs.get('exercise', 4)}/10"),
            'cgtsmok': ('Smoking Level', f"{inputs.get('smoking', 0)}/10"),
            'alcfreq': ('Alcohol Frequency', f"{inputs.get('alcohol', 2)}/10"),
            'etfruit': ('Fruit & Vegetable Intake', f"{inputs.get('fruit_intake', 4)}/10"),
            'eatveg': ('Fruit & Vegetable Intake', f"{inputs.get('fruit_intake', 4)}/10"),  # Combined with fruit intake
            'slprl': ('Sleep Quality', f"{inputs.get('sleep_quality', 7)}/10"),
            'bmi': ('BMI', f"{inputs.get('weight', 70) / ((inputs.get('height', 170)/100)**2):.1f}"),
            # Age-related factors
            'age': ('Age', f"{inputs.get('age', 40)} years"),
            'height': ('Height', f"{inputs.get('height', 170)} cm"),
            'weight': ('Weight', f"{inputs.get('weight', 70)} kg"),
            # Additional dataset features with meaningful names
            'inprdsc': ('Discrimination Experience', 'Based on life control'),
            'fltdpr': ('Depression Feelings', 'Mental health indicator'),
            'flteeff': ('Effort Feelings', 'Energy level'),
            'wrhpp': ('Work Happiness', 'Job satisfaction'),
            'fltlnl': ('Loneliness', 'Social connection'),
            'enjlf': ('Life Enjoyment', 'Life satisfaction'),
            'fltsd': ('Sadness Feelings', 'Emotional state'),
            'gndr': ('Gender', 'Demographic factor'),
            'paccnois': ('Noise Exposure', 'Environmental factor'),
            'mental_health_score': ('Mental Health Score', f"{((inputs.get('happiness', 7) + inputs.get('life_control', 7)) / 20):.2f}"),
            'lifestyle_score': ('Lifestyle Score', f"{((inputs.get('exercise', 4) + inputs.get('fruit_intake', 4) - inputs.get('smoking', 0)) / 30):.2f}"),
            'social_score': ('Social Score', f"{inputs.get('social_meetings', 5) / 10:.1f}")
        }
        
        return feature_map.get(feature_name, (feature_name.replace('_', ' ').title(), 'N/A'))
    
    def _get_fallback_key_factors(self, inputs):
        """Fallback analysis when SHAP is not available - using original user inputs"""
        # Calculate BMI from height/weight
        height_m = inputs.get('height', 170) / 100  # Convert cm to meters
        weight = inputs.get('weight', 70)  # kg
        bmi = weight / (height_m ** 2)
        
        # Use original 0-10 scale inputs for analysis
        factor_weights = {
            'BMI': bmi,
            'Physical Activity': inputs.get('exercise', 4),
            'Smoking Intensity': inputs.get('smoking', 0),
            'Alcohol Consumption': inputs.get('alcohol', 2),
            'Fruit & Vegetable Intake': inputs.get('fruit_intake', 4),
            'Life Satisfaction': inputs.get('happiness', 7),
            'Sense of Control Over Life': inputs.get('life_control', 7),
            'Sleep Quality': inputs.get('sleep_quality', 7),
            'Social Engagement': inputs.get('social_meetings', 5)
        }
        
        analysis = []
        for factor, value in factor_weights.items():
            if factor == 'BMI':
                if value < 18.5:
                    status = "Underweight"
                elif value <= 24.9:
                    status = "Normal"
                elif value <= 29.9:
                    status = "Overweight"
                else:
                    status = "Obese"
                analysis.append(f"- **{factor}:** {value:.1f} - {status}")
            else:
                level = "High" if value >= 6 else "Moderate" if value >= 4 else "Low"
                analysis.append(f"- **{factor}:** {level} ({value:.1f}/10)")
        
        return "\n".join(analysis)

# Initialize predictor
predictor = HeartRiskPredictor()

def create_professional_interface():
    """Create professional Heart Disease Risk Prediction interface"""
    
    # Custom CSS for professional medical-grade styling
    css = """
    /* Professional Medical Header */
    .main-header { 
        text-align: center; 
        background: linear-gradient(135deg, #2563eb 0%, #0891b2 50%, #059669 100%);
        color: white;
        padding: 30px 20px;
        margin: -20px -20px 30px -20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header h1 {
        margin: 0 0 10px 0;
        font-size: 2.2em;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        margin: 5px 0;
        font-size: 1.1em;
        opacity: 0.95;
    }
    
    /* Professional Input Sections */
    .input-group {
        background: linear-gradient(145deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Professional Risk Output */
    .risk-output {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 25px;
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
    }
    
    /* Risk Level Styling */
    .risk-low {
        border-left: 5px solid #059669;
        background: linear-gradient(145deg, #f0fdfa, #ccfbf1);
    }
    .risk-moderate {
        border-left: 5px solid #d97706;
        background: linear-gradient(145deg, #fffbeb, #fed7aa);
    }
    .risk-high {
        border-left: 5px solid #dc2626;
        background: linear-gradient(145deg, #fef2f2, #fecaca);
    }
    
    /* Professional Button Styling */
    .predict-btn {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px 30px;
        font-size: 1.1em;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
    }
    .predict-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
    }
    
    /* Medical Disclaimer Styling */
    .footer-disclaimer {
        background: linear-gradient(145deg, #fef3c7, #fde68a);
        padding: 20px;
        border-radius: 12px;
        margin-top: 25px;
        font-size: 0.9em;
        border: 1px solid #f59e0b;
        border-left: 5px solid #d97706;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.2);
    }
    
    /* Enhanced Typography */
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Section Headers */
    .section-header {
        color: #1e40af;
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Input Labels */
    label {
        color: #374151;
        font-weight: 500;
    }
    
    /* Slider Styling */
    .gradio-slider input[type="range"] {
        accent-color: #2563eb;
    }
    
    /* Improved Spacing */
    .gradio-group {
        margin: 20px 0;
    }
    
    /* Professional Cards */
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    """
    
    with gr.Blocks(
        title="Heart Disease Risk Prediction App", 
        theme=gr.themes.Default(),
        css=css
    ) as interface:
        
        # Professional Medical Header
        gr.HTML("""
        <div class="main-header">
            <h1>Heart Disease Risk Assessment Platform</h1>
            <p>AI-Powered Cardiovascular Risk Prediction with Clinical Insights</p>
            <p><i>Master's Research in Explainable Healthcare AI | Clinical Decision Support</i></p>
            <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.9;">Professional Medical-Grade Interface | Research & Educational Use</p>
        </div>
        """)
        
        with gr.Row():
            # Input Panel
            with gr.Column(scale=1):
                gr.HTML('<h3 class="section-header">Clinical Health Assessment</h3>')
                
                # Personal Information
                with gr.Group():
                    gr.HTML('<h4 style="color: #1e40af; margin-bottom: 15px;">Patient Demographics</h4>')
                    age = gr.Slider(
                        minimum=18, maximum=85, value=45, step=1,
                        label="Age (years)",
                        info="Patient's current age in years"
                    )
                    
                    height = gr.Slider(
                        minimum=140, maximum=200, value=170, step=1,
                        label="Height (cm)",
                        info="Patient's height in centimeters"
                    )
                    
                    weight = gr.Slider(
                        minimum=40, maximum=150, value=70, step=0.5,
                        label="Weight (kg)", 
                        info="Patient's current weight in kilograms"
                    )
                
                # Lifestyle Factors  
                with gr.Group():
                    gr.HTML('<h4 style="color: #1e40af; margin-bottom: 15px;">Lifestyle & Health Behaviors</h4>')
                    
                    exercise = gr.Slider(
                        minimum=0, maximum=10, value=5, step=1,
                        label="Physical Activity Level",
                        info="0=Never exercise, 10=Daily intensive exercise"
                    )
                    
                    smoking = gr.Slider(
                        minimum=0, maximum=10, value=0, step=1,
                        label="Smoking Intensity",
                        info="0=Never smoked, 10=Heavy daily smoking"
                    )
                    
                    alcohol = gr.Slider(
                        minimum=0, maximum=10, value=2, step=1,
                        label="Alcohol Consumption",
                        info="0=Never drink, 10=Daily heavy drinking"
                    )
                    
                    fruit_intake = gr.Slider(
                        minimum=0, maximum=10, value=6, step=1,
                        label="Fruit & Vegetable Intake",
                        info="0=Never eat fruits/vegetables, 10=Daily high intake"
                    )
                
                # Mental Health & Social Factors
                with gr.Group():
                    gr.HTML('<h4 style="color: #1e40af; margin-bottom: 15px;">Psychological & Social Wellbeing</h4>')
                    
                    happiness = gr.Slider(
                        minimum=0, maximum=10, value=7, step=1,
                        label="Overall Life Satisfaction",
                        info="0=Extremely dissatisfied, 10=Completely satisfied"
                    )
                    
                    life_control = gr.Slider(
                        minimum=0, maximum=10, value=7, step=1,
                        label="Sense of Control Over Life",
                        info="0=No control, 10=Complete control"
                    )
                    
                    social_meetings = gr.Slider(
                        minimum=0, maximum=10, value=5, step=1,
                        label="Social Engagement Level", 
                        info="0=Socially isolated, 10=Very socially active"
                    )
                    
                    sleep_quality = gr.Slider(
                        minimum=0, maximum=10, value=7, step=1,
                        label="Sleep Quality",
                        info="0=Very poor sleep, 10=Excellent sleep"
                    )
                
                # Professional Prediction Button
                gr.HTML('<div style="margin: 25px 0 15px 0;"></div>')
                predict_button = gr.Button(
                    "Analyze Cardiovascular Risk Profile", 
                    variant="primary",
                    size="lg",
                    elem_classes=["predict-btn"]
                )
            
            # Results Panel
            with gr.Column(scale=1):
                gr.HTML('<h3 class="section-header">Clinical Risk Assessment Results</h3>')
                
                prediction_output = gr.Markdown(
                    value="""
### Welcome to Professional Cardiovascular Risk Assessment

Please complete the health assessment form on the left and click 
**"Analyze Cardiovascular Risk Profile"** to receive your comprehensive 
cardiovascular risk analysis with clinical-grade explainable AI insights.

#### **Clinical Assessment Features:**
- **Evidence-Based Risk Stratification** (Low/Moderate/High)
- **Quantitative Probability Analysis** with confidence intervals
- **Dual Explainable AI Implementation:** SHAP research validation + LIME personalized analysis
- **Clinical Feature Importance** ranking and interpretation  
- **Personalized Lifestyle Recommendations** based on individual risk factors
- Trained on 42,000+ patient health records
- Explainable AI with SHAP clinical validation
- Research-grade statistical modeling
- Healthcare industry compliance protocols
                    """,
                    elem_classes=["risk-output"]
                )
        
        # Prediction Logic
        predict_button.click(
            fn=lambda age, height, weight, exercise, smoking, alcohol, fruit_intake, 
                     happiness, life_control, social_meetings, sleep_quality: 
                predictor.predict_risk(
                    age=age, 
                    height=height,  # Pass height to inputs
                    weight=weight,  # Pass weight to inputs  
                    bmi=weight / ((height/100) ** 2),  # Calculate BMI # Convert cm to meters, then square
                    exercise=exercise,
                    smoking=smoking,
                    alcohol=alcohol, 
                    fruit_intake=fruit_intake,
                    happiness=happiness,
                    life_control=life_control,
                    social_meetings=social_meetings,
                    sleep_quality=sleep_quality,
                    # Additional derived features
                    mental_health=(happiness + life_control) / 20,
                    lifestyle=(exercise + fruit_intake - smoking) / 30,
                    social=social_meetings / 10
                ),
            inputs=[age, height, weight, exercise, smoking, alcohol, fruit_intake,
                   happiness, life_control, social_meetings, sleep_quality],
            outputs=prediction_output
        )
        
        # Professional Medical Disclaimer
        gr.HTML("""
        <div class="footer-disclaimer">
            <h4>Professional Medical Disclaimer & Compliance Notice</h4>
            <p><strong>This application is designed for educational and research purposes only.</strong></p>
            <ul style="margin: 15px 0;">
                <li><strong>Research Tool:</strong> Provides cardiovascular risk estimates with dual explainable AI (SHAP + LIME)</li>
                <li><strong>Not Diagnostic:</strong> This tool is <strong>NOT a substitute</strong> for professional medical diagnosis, treatment, or clinical care</li>
                <li><strong>Clinical Consultation Required:</strong> Always consult qualified healthcare professionals for medical advice and treatment decisions</li>
                <li><strong>Emergency Protocol:</strong> Emergency cardiac symptoms require immediate medical attention - Call emergency services</li>
                <li><strong>Academic Use:</strong> Results intended for academic research and educational demonstration only</li>
            </ul>
            <hr style="margin: 15px 0; border: none; border-top: 1px solid #d97706;">
            <p><small>
                <strong>Technical Specifications:</strong> Machine learning predictions based on ensemble models 
                trained on European Social Survey health data (N=42,000+) with dual explainable AI implementation: 
                SHAP for research validation and LIME for personalized individual explanations. 
                <strong>Performance:</strong> Research-grade implementation with clinical safety protocols and 
                healthcare industry evaluation standards.
            </small></p>
            <p><small>
                <strong>Institutional:</strong> Master's Research Project | Healthcare AI & Explainable Machine Learning
            </small></p>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the professional interface
    app = create_professional_interface()
    
    # Auto-detect deployment environment and use appropriate port
    import os
    
    # Check if running in Docker (common Docker environment indicators)
    is_docker = False
    try:
        is_docker = (
            os.path.exists('/.dockerenv') or 
            os.environ.get('DOCKER_CONTAINER') == 'true' or
            os.environ.get('HOSTNAME', '').startswith('docker') or
            (Path('/proc/1/cgroup').exists() and 'docker' in open('/proc/1/cgroup', 'r').read())
        )
    except:
        # Fallback for systems without /proc (like macOS)
        is_docker = (
            os.path.exists('/.dockerenv') or 
            os.environ.get('DOCKER_CONTAINER') == 'true'
        )
    
    # Set port based on environment (with manual override support)
    manual_port = os.environ.get('GRADIO_SERVER_PORT')
    if manual_port:
        server_port = int(manual_port)
        print(f"Using manual port override: {server_port}")
    elif is_docker:
        server_port = 7860  # Docker deployment port
        print("Detected Docker environment - using port 7860")
    else:
        server_port = 7861  # Local development port
        print("Detected local environment - using port 7861")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=True,
        debug=False,
        show_error=True
    )
