# Heart Disease Risk Prediction - Technology Stack Documentation

**Project:** Heart Disease Risk Prediction with Explainable AI  
**Author:** Peter Ugonna Obi  
**Date:** January 30, 2026

---

## Overview

This document provides a comprehensive overview of the backend and frontend frameworks, libraries, and technologies used in the Heart Disease Risk Prediction project.

---

## Backend Technologies

### **Core Framework**
- **Python 3.9**
  - Primary programming language
  - Used for all machine learning, data processing, and web application logic

- **Gradio 4.x**
  - Full-stack web framework for ML applications
  - Handles HTTP requests, API endpoints, and server logic
  - Provides automatic REST API generation
  - Manages real-time prediction processing

### **Machine Learning Stack**
- **scikit-learn**
  - Primary ML library for model training and prediction
  - Used for: RandomForestClassifier, StandardScaler, train_test_split
  - Model evaluation metrics and cross-validation

- **pandas**
  - Data manipulation and analysis
  - CSV file processing and DataFrame operations
  - Feature engineering and data preprocessing

- **numpy**
  - Numerical computing and array operations
  - Mathematical calculations for normalization and feature scaling

- **joblib**
  - Model serialization and persistence
  - Loading pre-trained models and preprocessing artifacts
  - Efficient handling of large numpy arrays

### **Explainable AI**
- **SHAP (SHapley Additive exPlanations)**
  - Model interpretability and global feature importance analysis
  - Research-grade explainability for comprehensive insights
  
- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Individual patient-level explanations
  - Personalized risk factor analysis
  - Local interpretability for specific predictions
  - Professional fallback system for robust deployment
  - TreeExplainer for ensemble models
  - Research-grade explainability implementation

- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Individual prediction explanations for each user
  - Personalized risk factor analysis
  - Real-time local interpretability
  - Professional fallback system when LIME unavailable

### **Data Visualization**
- **matplotlib**
  - Statistical plots and model performance visualizations
  - Research analysis charts and graphs

### **Data Processing Libraries**
- **pathlib** - Modern file system path handling
- **json** - Configuration and data serialization
- **warnings** - Error handling and logging
- **os/sys** - System integration and environment detection

---

## Frontend Technologies

### **Web Interface Framework**
- **Gradio Interface Components**
  - Slider widgets for user inputs
  - Button components for interactions
  - Markdown rendering for results display
  - Form handling and validation

### **Custom Styling**
- **CSS3**
  - Professional medical-grade interface styling
  - Responsive design for multiple screen sizes
  - Custom color schemes and typography
  - Medical compliance visual standards

- **HTML5**
  - Semantic markup for accessibility
  - Professional medical interface structure

### **Interactive Elements**
- **JavaScript** (Built-in Gradio)
  - Real-time form interactions
  - Dynamic result updates
  - Client-side validation

---

## Development & Deployment Stack

### **Containerization**
- **Docker**
  - Application containerization
  - Multi-stage build process
  - Production-ready deployment configuration
  - Environment isolation and consistency

- **Docker Compose**
  - Service orchestration
  - Development environment setup
  - Port mapping and volume management

### **Base Infrastructure**
- **Python 3.9-slim Docker Image**
  - Lightweight container base
  - Optimized for production deployment
  - Security-hardened Python environment

### **Development Tools**
- **Git**
  - Version control and collaboration
  - Branch management and code history

- **VS Code**
  - Primary development environment
  - Python extension and debugging tools

- **Jupyter Notebooks**
  - Data analysis and exploratory development
  - Model training and evaluation
  - Research documentation

---

## Architecture Pattern

### **Design Pattern**
- **Model-View-Controller (MVC)**
  - Model: Machine learning models and data processing
  - View: Gradio web interface components
  - Controller: Prediction logic and user interaction handling

### **Application Architecture**
- **Microservice-Ready Design**
  - Containerized for cloud deployment
  - API-first architecture with Gradio
  - Scalable and maintainable codebase

- **Separation of Concerns**
  - Data processing (src/ directory)
  - Model training (notebooks/ directory)
  - Web application (app/ directory)
  - Configuration management (config files)

---

## Deployment Configuration

### **Environment Detection**
- **Automatic Port Configuration**
  - Docker: Port 7860
  - Local development: Port 7861
  - Manual override support via environment variables

### **Production Features**
- **Health Check Endpoints**
- **Error Handling and Logging**
- **Medical Disclaimer and Compliance**
- **Professional Interface Standards**

---

## Summary for Supervisor

**Quick Response:**
*"The project uses **Python 3.9 with Gradio** as the main full-stack framework, handling both backend ML processing and frontend web interface. The machine learning stack includes **scikit-learn, pandas, numpy, and SHAP** for explainable AI. The application is **containerized with Docker** for production deployment and features **custom CSS styling** for a professional medical-grade interface."*

**Technical Stack:**
- **Backend:** Python 3.9, Gradio, scikit-learn, pandas, numpy, SHAP, LIME
- **Frontend:** Gradio components, custom CSS3, responsive design
- **Deployment:** Docker containerization, automatic environment detection
- **Development:** Git version control, Jupyter notebooks, VS Code

**Architecture:** MVC pattern with microservice-ready design, API-first approach, dual explainable AI (SHAP + LIME), and healthcare compliance standards.

---

*This technology stack demonstrates modern, industry-standard tools appropriate for academic research and professional healthcare AI development.*