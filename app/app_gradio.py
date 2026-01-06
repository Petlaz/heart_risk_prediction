"""
Interactive Gradio Web Interface for Heart Risk Prediction

This module provides a user-friendly web interface for making heart risk predictions
and viewing explanations using trained models and explainability techniques.
"""

import gradio as gr
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def predict_heart_risk(age, gender, smoking, exercise, health_rating):
    """
    Make heart risk prediction based on input features
    
    Args:
        age: Age of the person
        gender: Gender (Male/Female)  
        smoking: Smoking status
        exercise: Exercise frequency
        health_rating: Self-rated health
        
    Returns:
        prediction: Risk prediction and explanation
    """
    # Placeholder prediction logic
    # In production, this would load the trained model and make real predictions
    
    risk_score = np.random.random()  # Placeholder
    
    if risk_score > 0.7:
        risk_level = "High Risk"
        color = "🔴"
    elif risk_score > 0.4:
        risk_level = "Moderate Risk" 
        color = "🟡"
    else:
        risk_level = "Low Risk"
        color = "🟢"
    
    result = f"""
    {color} **{risk_level}**
    
    Risk Score: {risk_score:.2f}
    
    **Key Factors:**
    - Health Rating: {health_rating}
    - Exercise Level: {exercise}
    - Smoking Status: {smoking}
    - Age: {age}
    - Gender: {gender}
    
    *Note: This is a demonstration interface. In production, this would use the trained model.*
    """
    
    return result

def create_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="Heart Risk Prediction", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🫀 Heart Risk Prediction with Explainable AI
            
            This interactive tool provides heart disease risk predictions based on lifestyle and health factors.
            The predictions are generated using machine learning models trained on European Social Survey data.
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Features")
                
                age = gr.Slider(
                    minimum=18, 
                    maximum=100, 
                    value=50, 
                    label="Age",
                    info="Age in years"
                )
                
                gender = gr.Radio(
                    choices=["Male", "Female"], 
                    value="Male", 
                    label="Gender"
                )
                
                smoking = gr.Dropdown(
                    choices=["Never", "Former", "Occasional", "Daily"], 
                    value="Never", 
                    label="Smoking Status"
                )
                
                exercise = gr.Dropdown(
                    choices=["Never", "Less than once a month", "Once a month", 
                            "Several times a month", "Once a week", 
                            "Several times a week", "Every day"], 
                    value="Once a week", 
                    label="Exercise Frequency"
                )
                
                health_rating = gr.Dropdown(
                    choices=["Very good", "Good", "Fair", "Bad", "Very bad"], 
                    value="Good", 
                    label="Self-rated Health"
                )
                
                predict_btn = gr.Button("Predict Heart Risk", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Prediction Results")
                prediction_output = gr.Markdown(value="Enter your information and click 'Predict Heart Risk' to get started.")
        
        predict_btn.click(
            fn=predict_heart_risk,
            inputs=[age, gender, smoking, exercise, health_rating],
            outputs=prediction_output
        )
        
        gr.Markdown(
            """
            ---
            **Disclaimer:** This tool is for educational and research purposes only. 
            It should not be used as a substitute for professional medical advice, 
            diagnosis, or treatment.
            """
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )