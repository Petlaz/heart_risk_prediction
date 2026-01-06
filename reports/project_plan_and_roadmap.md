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

## 🧩 3-Month Research Project Roadmap
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
- Continue writing the Methods section

**Deliverables:** Optimized models + validation results + error summary + initial paper notes

**Reading:** 
- Interpretable ML Ch. 5
- Hands-On ML Ch. 6–8
- Designing ML Systems Ch. 3

---

### Weeks 5–6 (Nov 17 – Dec 1): Local Explainability Integration (XAI)

**Focus:** XAI Implementation

**Tasks:**
- Implement LIME and SHAP for selected model
- Create SHAP summary, force plots, and LIME explanations
- Compare local explanations across models
- Interpret healthcare-related insights from local explanations
- Ensure XAI modules run inside Docker
- Continue writing State of the Art and Results sections

**Deliverables:** XAI visualizations + interpretability report + Dockerized XAI workflow

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
- Continue report writing (Results + Discussion)

**Deliverables:** Functional Gradio demo (classical + NN models) + Meeting 4 summary

**Reading:** 
- Hands-On ML Ch. 19
- Designing ML Systems Ch. 4

---

### Weeks 9–10 (Dec 16 – Jan 1): Evaluation, Refinement & Discussion

**Focus:** Final Evaluation and Refinement

**Tasks:**
- Evaluate final model on validation and test sets
- Assess stability and consistency of local explanations
- Refine XAI visuals and final discussion
- Update Docker image with final model
- Finalize Discussion and State of the Art sections

**Deliverables:** Evaluation results + refined XAI visuals + updated demo + Meeting 5 summary

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

## 📅 Summary of Biweekly Meetings

| Meeting | Week | Focus | Key Deliverable |
|---------|------|-------|-----------------|
| 1 | 2 | EDA + Baseline + Error Analysis | Clean dataset + metrics + confusion matrix |
| 2 | 4 | Model Optimization + Early Validation | Optimized models + validation results + literature insights |
| 3 | 6 | Local XAI Integration | LIME/SHAP visualizations + interpretation |
| 4 | 8 | Gradio Demo | Interactive demo (Dockerized) |
| 5 | 10 | Evaluation + Refinement | Final metrics + discussion draft |
| 6 | 12 | Final Presentation | Report + Gradio demo + Docker image |

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
4. **Real-time Predictions:** Live prediction and explanation generation

---

## Success Metrics

### Technical Objectives
- [ ] All models implemented and optimized
- [ ] F1 score optimization achieved
- [ ] LIME and SHAP successfully integrated
- [ ] Gradio demo functional and containerized
- [ ] Docker environment fully reproducible

### Academic Deliverables
- [ ] Complete literature review
- [ ] Full report with all sections
- [ ] Error analysis and model comparison
- [ ] XAI interpretability validation
- [ ] Final presentation prepared

### Industry Collaboration
- [ ] Regular updates to Nightingale Heart
- [ ] Clinical relevance validated
- [ ] Practical deployment considerations addressed
- [ ] Industry feedback incorporated

---

**Document Version:** 2.0 - Updated Research Plan  
**Last Updated:** January 5, 2026  
**Next Review:** Biweekly Meeting 1