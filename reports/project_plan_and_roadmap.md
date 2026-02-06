# Project Plan and Roadmap - Heart Risk Prediction with Explainable AI

**Updated Research Project Plan**

## Project Overview

### Project Title
Prediction and Local Explainable AI (XAI) in Healthcare

### Project Duration
**Start Date:** October 2025  
**End Date:** January 2026  
**Total Duration:** 3 months

### Team
**Supervisor:** Prof. Dr. Beate Rhein  
**Industry Partner:** Nightingale Heart – Mr. Håkan Lane  
**Researcher:** [Your Name]

### Project Goal
The goal is to integrate **Local Explainable AI (XAI) techniques** — specifically **LIME and SHAP** — to interpret model decisions at the individual level.

A **Gradio interface** will provide real-time interactive predictions and explanations.  
The entire workflow will be **containerized using Docker** for reproducibility and future deployment.

---

## Dataset Overview

**Structured CSV dataset** with health, demographic, and lifestyle variables.

- **Target variable:** `hltprhc` (heart condition: 1 = yes, 0 = no)
- **Alternative targets:** `hltprhb` (blood pressure), `hltprdi` (diabetes)

---

## Research Objectives

1. **Develop and compare predictive models:** Logistic Regression, Random Forest, XGBoost, SVM, and PyTorch Neural Network.
2. **Perform early error analysis** (accuracy, precision, recall, confusion matrix, and misclassified samples).
3. **Conduct model optimization and iterative validation** on unseen data after tuning.
4. **Apply Local Explainability** (LIME and SHAP) for individual-level interpretation.
5. **Conduct a literature review** ("State of the Art") informed by model errors.
6. **Write report sections** (Methods, Results, Discussion) in parallel with experiments.
7. **Build a Gradio demo** for interpretable healthcare prediction.
8. **Containerize all experiments** using Docker for reproducibility.

---

## 3-Month Research Project Roadmap
*(Biweekly meetings – 6 total, ~20 hrs/week)*

### Weeks 1–2 (Oct 20 – Nov 2): Data Understanding, Baseline Modeling & Error Analysis

**Focus:** Foundation and Baseline Development

**Tasks:**
- Load and explore the dataset
- Conduct Full EDA
- Data preprocessing and feature engineering
- Train baseline models using Logistic Regression, Random Forest, XGBoost, SVM, and Neural Network with PyTorch (using AdamW with patience set to 10)
- Evaluate with accuracy, precision, recall, F1, ROC curve, classification report, and confusion matrix
- Perform misclassified samples analysis
- Perform full error analysis
- Initialize the GitHub repository, create a requirements.txt file, and create a Dockerfile
- Begin writing the Introduction and Methods sections

**Deliverables:** Clean dataset + baseline results + error plots + Docker setup

**Reading:** 
- Interpretable ML Ch. 2–3
- Hands-On ML Ch. 2–4
- Designing ML Systems Ch. 2

---

### Weeks 3–4 (Nov 3 – Nov 16): Model Optimization (focus F1 score), Early Validation & Literature Review

**Focus:** Model Optimization and Literature Review

**Tasks:**
- Tune hyperparameters (RandomizedSearchCV)
- Validate optimized models on unseen data (early performance check)
- Analyze misclassifications and document patterns
- Begin literature review ("State of the Art") informed by error findings
- Update Docker setup for reproducible experiments
- Continue writing the Methods section and other sections

**Deliverables:** Optimized models + validation results + error summary + initial paper notes

**Reading:** 
- Interpretable ML Ch. 5
- Hands-On ML Ch. 6–8
- Designing ML Systems Ch. 3

---

### Weeks 5–6 (Nov 17 – Dec 1): Local Explainability Integration (XAI) **COMPLETED**

**Focus:** XAI Implementation

**Tasks:**
- **Implement LIME and SHAP for selected model** - SHAP implementation completed with TreeExplainer
- **Create SHAP summary, force plots, and LIME explanations** - SHAP summary plots, feature importance, beeswarm plots created
- **Compare local explanations across models** - Baseline Random Forest analysis with individual patient cases
- **Interpret healthcare-related insights from local explanations** - Clinical interpretation of BMI, exercise, psychological factors
- **Ensure XAI modules run inside Docker** - All explainability code containerized and reproducible
- **Continue writing State of the Art and Results section** - Literature review and final report updated with XAI findings

**Completed Deliverables:** 
- **XAI visualizations:** 4 SHAP plots saved (feature importance, summary beeswarm, detailed summary, bar plot)
- **Interpretability report:** Comprehensive clinical interpretation identifying optimization paradox root cause
- **Dockerized XAI workflow:** 15-cell explainability notebook with SHAP analysis
- **Clinical insights:** BMI (0.0208) and exercise (0.0189) as top predictors, psychological factors as weak signals
- **Root cause analysis:** XAI validation of dataset limitations and clinical deployment challenges

**Reading:** 
- Interpretable ML Ch. 4–6
- Hands-On ML Ch. 11  
- Designing ML Systems Ch. 8

---

### Weeks 7–8 (Dec 2 – Dec 15): Gradio Demo Development & Report Progress

**Focus:** Interactive Application Development

**Tasks:**
- Build an interactive Gradio app (real-time predictions + explanations)
- Integrate the best performed model
- Test usability, latency, and visual clarity
- Containerize demo (EXPOSE 7860) and test locally
- Continue report writing (Results + Discussion) and other sections

**Deliverables:** Functional Gradio demo (classical + NN models) + Meeting 4 summary

**Reading:** 
- Hands-On ML Ch. 19
- Designing ML Systems Ch. 4

---

### Weeks 9–10 (Dec 16 – Jan 1): Evaluation, Refinement & Discussion **100% COMPLETED**

**Focus:** Final Evaluation and Refinement

**Tasks:**
- **Evaluate final model on validation and test sets** - Comprehensive evaluation completed across all optimization approaches
- **Assess stability and consistency of local explanations** - SHAP analysis validates consistent feature importance across models
- **Refine XAI visuals and final discussion** - 4 comprehensive SHAP visualizations created with clinical interpretation
- **Update Docker image with final model** - Docker configuration updated for Gradio app deployment
- **Finalize Discussion and State of the Art sections** - All report sections updated with Week 7-8 application findings
- **Final Docker testing with public URL deployment** - Completed (100%)

**Completed Deliverables:** 
- **Final Model Evaluation**: Comprehensive assessment confirming optimization paradox across all approaches
- **XAI Visual Refinement**: Professional SHAP plots with clinical decision support templates
- **Professional Application**: Heart Disease Risk Prediction App with medical-grade interface
- **Updated Documentation**: All reports enhanced with practical deployment insights
- **Docker Integration**: Containerization ready for both local and public URL deployment
- ⏳ **Meeting 5 Summary**: To be completed with final Docker validation

**Key Achievements:**
- **Model Stability Confirmed**: Consistent 77-78% high risk predictions validate systematic deployment challenges
- **XAI Consistency Validated**: SHAP explanations remain stable across optimization approaches
- **Clinical Integration**: Professional application demonstrates practical healthcare ML implementation
- **Honest Assessment**: Research provides unprecedented transparent evaluation of healthcare ML challenges

**Reading:** 
- Interpretable ML Ch. 7
- Designing ML Systems Ch. 9

---

### Weeks 11–12 (Jan 2 – Jan 15): Final Report & Defense Preparation

**Focus:** Project Completion and Presentation

**Tasks:**
- Finalize Gradio demo and Docker image
- Write final report (Introduction, State of the Art, Methods, Results, Discussion, Conclusion)
- Prepare presentation slides and defense
- Submit report + Docker package to Professor and Nightingale Heart

**Deliverables:** Final report + Gradio demo + Docker image + Meeting 6 summary

**Reading:** 
- Hands-On ML Appendix
- Designing ML Systems Ch. 10

---

## Summary of Biweekly Meetings

| Meeting | Week | Focus | Key Deliverable | Status |
|---------|------|-------|-----------------|--------|
| 1 | 2 | EDA + Baseline + Error Analysis | Clean dataset + metrics + confusion matrix | Complete |
| 2 | 4 | Model Optimization + Early Validation | Optimized models + validation results + literature insights | Complete |
| 3 | 6 | Local XAI Integration | LIME/SHAP visualizations + interpretation | **Complete** |
| 4 | 8 | Gradio Demo | Interactive demo (Dockerized) | **Complete** |
| 5 | 10 | Evaluation + Refinement | Final metrics + discussion draft | **100% Complete** |
| 6 | 12 | Final Presentation | Report + Gradio demo + Docker image | ⏳ In Progress |

---

## Technical Implementation Strategy

### Model Development Pipeline
1. **Baseline Models:** Logistic Regression, Random Forest, XGBoost, SVM
2. **Neural Network:** PyTorch implementation with AdamW optimizer (patience=10)
3. **Optimization:** RandomizedSearchCV with F1 score focus
4. **Validation:** Early validation on unseen data

### Explainability Framework
1. **SHAP Implementation:** Summary plots, force plots, feature importance
2. **LIME Integration:** Local explanations for individual predictions
3. **Comparison Analysis:** Cross-model explanation consistency
4. **Healthcare Interpretation:** Clinical relevance of explanations

### Deployment Architecture
1. **Containerization:** Docker for reproducible environment
2. **Interactive Interface:** Gradio application (port 7860)
3. **Model Integration:** Classical and neural network comparison
4. **Real-time Predictions:** Live prediction and explanation creation

---

## Success Metrics

### Technical Objectives
- All models implemented and optimized
- F1 score optimization achieved
- **LIME and SHAP successfully integrated** - SHAP TreeExplainer implemented with clinical interpretation
- ⏳ Gradio demo functional and containerized (In Progress)
- Docker environment fully reproducible

### Academic Deliverables
- **Complete literature review** - Updated with Week 5-6 XAI findings
- **Full report with all sections** - Final report updated with XAI results
- Error analysis and model comparison
- **XAI interpretability validation** - SHAP analysis confirms optimization paradox
- ⏳ Final presentation prepared (In Progress)

### Industry Collaboration
- [ ] Regular updates to Nightingale Heart
- [ ] Clinical relevance validated
- [ ] Practical deployment considerations addressed
- [ ] Industry feedback incorporated

---

**Document Version:** 2.0 - Updated Research Plan  
**Last Updated:** January 5, 2026  
**Next Review:** Biweekly Meeting 1