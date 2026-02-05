# Heart Disease Risk Prediction with Explainable AI
**A Comprehensive Machine Learning Approach to Clinical Decision Support**

**Master's Research Project Defense**

- **Author:** Peter Ugonna Obi
- **Supervisor:** Prof. Dr. Beate Rhein  
- **Industry Partner:** Nightingale Heart – Mr. Håkan Lane
- **Date:** February 2026

---

## Research Problem & Questions

### The Healthcare ML Deployment Crisis

- **Performance Gap:** Published studies report 65-89% F1-scores, but real deployment often fails
- **Clinical Safety Risk:** Traditional ML optimization may compromise medical safety requirements  
- **Black Box Problem:** Lack of explainability prevents clinical adoption

### Core Research Questions

1. How do systematically optimized models perform compared to baseline implementations?
2. What drives misclassification patterns in healthcare ML applications?
3. Can dual explainable AI (SHAP + LIME) reveal fundamental limitations in psychological-based cardiac prediction and provide individual patient insights?

**Research Contribution:** First comprehensive study documenting the "optimization paradox" in healthcare ML

---

## Methodology & Data

### Dataset & Approach

- **Source:** European Social Survey health data (52,266 samples → 8,476 test samples)
- **Features:** 22 health, lifestyle, and psychological variables (BMI, exercise, smoking, alcohol, mental health scores)
- **Target:** Binary heart condition prediction (`hltprhc` variable)
- **Data Split:** 70% train, 15% validation, 15% test (stratified for class balance)
- **Class Distribution:** Imbalanced dataset requiring specialized evaluation metrics
- **Quality Control:** Complete preprocessing pipeline with StandardScaler normalization

### Dual-Method Research Framework

**Method 1: Comprehensive ML Pipeline**

- Baseline evaluation (5 algorithms: NN, XGBoost, SVM, LR, RF)
- Systematic hyperparameter optimization using RandomizedSearchCV (100 iterations per model)
- Clinical performance assessment with safety criteria (€1,000 per false negative vs. €100 per false positive)
- 5-fold stratified cross-validation for robust performance estimation

**Method 2: Professional Application Development**

- Professional Gradio web interface with medical compliance features (11 user inputs → 22 model features)
- Docker containerization for development and testing environments
- HeartRiskPredictor class with dual XAI integration (SHAP + LIME)
- Multi-environment deployment (local port 7861, Docker port 7860)

![](results/plots/ml_pipeline_diagram.png)

---

## Baseline Results

### Algorithm Performance Overview

| Model | F1 | Sens | Spec | AUC | Status |
|-------|----|----|----|----|----|
| **Neural Network** | **30.8%** | **40.5%** | 75.2% | 68.2% | Best overall |
| **XGBoost** | 30.4% | 50.8% | 73.7% | 69.1% | Highest sens |
| **Logistic Regression** | 29.0% | 62.5% | 65.4% | 68.9% | Highest recall |
| **Random Forest** | 28.9% | 36.4% | 79.8% | **70.1%** | Best spec |
| **SVM** | 29.5% | 54.4% | 70.6% | 68.6% | Balanced |

### Statistical Validation

- **CV Stability:** NN F1: 30.8% ± 0.007 (very stable)
- **Performance Range:** 28.9% - 30.8% (tight 2% cluster)
- **Clinical Gap:** All <80% sensitivity requirement
- **Cost:** NN: €156/patient (acceptable <€200)

### Key Baseline Insights

- Moderate performance established optimization potential
- Neural Network achieved optimal precision-recall balance
- Logistic Regression's 62.5% sensitivity approached clinical viability (still 17.5% below requirement)
- Cross-model agreement: 77.3% consensus rate indicating challenging prediction scenario
- Algorithmic diversity suggested ensemble optimization potential

![](results/plots/roc_curves_baseline_models.png)

---

## Critical Discovery: The Optimization Paradox

### Systematic Optimization Results

**Optimization Method:** RandomizedSearchCV with F1-score optimization targeting improved clinical sensitivity

**Optimization Parameters:**
- **Search Space:** 100 iterations per model with clinical metric focus
- **Parameter Grids:** Model-specific parameter ranges optimized for healthcare applications
- **Target Metric:** F1-score (clinical relevance) with sensitivity prioritization
- **Validation:** 5-fold stratified cross-validation with robust statistical testing
- **Hardware:** Apple Silicon (M1/M2) optimization for efficiency

| Phase | Model | F1 | Sens | Gen Gap | Change |
|-------|-------|----|------|---------|--------|
| **Baseline** | Neural Network | **30.8%** | **40.5%** | N/A | Starting |
| **Optimized** | Adaptive_Ensemble | **17.5%** | **14.3%** | -11.5% | **-43% F1** |
| | Optimal_Hybrid | 9.1% | 5.2% | -18.9% | **-65% Sens** |
| | Adaptive_LR | 3.2% | 1.7% | -25.8% | **-87% Sens** |

**Expected vs. Actual Results:**
- **Expected:** 35-40% F1-score improvement based on literature
- **Actual:** 43-90% performance degradation across all optimized models
- **Validation Gap:** All models showed severe overfitting (validation F1: 29%, test F1: 17.5%)

### The Optimization Paradox Explained

- **Catastrophic Performance Degradation:** Systematic hyperparameter optimization worsened clinical performance
- **False Negative Explosion:** Sensitivity collapsed from 40.5% to 14.3% (misses 85.7% of heart disease cases)
- **Overfitting Evidence:** Large generalization gaps despite cross-validation
- **Healthcare ML Challenge:** Traditional optimization frameworks contraindicated for clinical applications
- **Novel Research Contribution:** First documented evidence of optimization paradox in healthcare ML
### Empirical Validation Through Application Testing

**Critical Finding: Discriminative Range Analysis**

Our application testing revealed additional evidence supporting the optimization paradox:

| Risk Level | Patient Profile | Prob | Reality |
|------------|-----------------|------|--------|
| **Low** | 45yr, BMI 24.2, non-smoker | **24.0%** | Healthy |
| **Moderate** | 62yr, BMI 40.1, smoker | **31.1%** | Multi-risk |
| **High** | 77yr, BMI 56.0, heavy user | **35.1%** | Extreme risk |

**Key Finding:** Despite vastly different risk profiles, the model produces only an **11.1% probability spread**

**Critical Research Validations:**

- **Threshold Testing Validates Limited Discrimination:** Risk categories show minimal separation (24.0% → 37.9%)
- **Clinical Risk Categories Show Minimal Separation:** Three-tier classification fails to meaningfully distinguish patient risk levels
- **Empirical Evidence of Psychological Variable Limitations:** Lifestyle surveys inadequate for clinical cardiovascular assessment

**Research Significance:**

- **Dataset Limitation Proof:** Psychological variables cannot distinguish between extreme risk profiles
- **Clinical Inadequacy Evidence:** No physician would consider these patients similarly risky
- **Optimization Failure Validation:** Even optimized models cannot overcome fundamental data constraints

![](results/plots/optimization_paradox_comparison.png)

---

## Clinical Deployment Assessment

### Healthcare Safety Criteria

- **Required Sensitivity:** ≥80% (to avoid missing heart disease cases)
- **Required Specificity:** ≥60% (to control false alarm burden)
- **Regulatory Standards:** Medical device safety protocols

### Deployment Verdict

| Category | Best Sens | Miss Rate | Status | Safety |
|----------|----------|-----------|--------|---------|
| **Baseline** | 40.5% | 59.5% | Below criteria | Insufficient |
| **Optimized** | 14.3% | 85.7% | Critical fail | **Unacceptable** |

### Critical Clinical Impact

- **85.7% False Negative Rate:** Poses unacceptable patient endangerment
- **Regulatory Non-Compliance:** No models meet clinical deployment safety standards
- **Economic Paradox:** Despite acceptable cost per patient (€152), safety risks create insurmountable liability

---

## Dual Explainable AI Implementation

### SHAP Analysis Reveals Root Causes (Global Insights)

**SHAP Implementation Details:**
- **Explainer Type:** TreeExplainer on Random Forest baseline model
- **Sample Size:** 500 test samples for comprehensive feature analysis
- **Background Dataset:** 10 samples for KernelExplainer baseline
- **Validation:** Cross-referenced with LIME for explanation consistency

**Top Predictive Features (SHAP Values):**

| # | Feature | SHAP | Clinical | Signal | SE |
|---|---------|------|----------|--------|---------|
| 1 | **BMI** | 0.0208 | Strong risk | **Valid** | ±0.003 |
| 2 | **Exercise** | 0.0189 | Protection | **Valid** | ±0.002 |
| 3 | **Mental Effort** | 0.0149 | Psychological | **Weak** | ±0.004 |
| 4 | **Sleep** | 0.0126 | Moderate | **Moderate** | ±0.003 |
| 5 | **Alcohol** | 0.0105 | Risk factor | **Moderate** | ±0.002 |
| 6-10 | **Mood** | 0.0093-0.0079 | Psychological | **Weak** | ±0.003 |

**Statistical Significance:**
- **Signal Strength:** Even strongest features (BMI, exercise) show only 0.02 SHAP impact
- **Feature Quality Gap:** 60% of predictive features are psychological with weak cardiac validity
- **Clinical Context:** Missing traditional cardiac markers (ECG, chest pain, cholesterol, family history)

### Critical Dual XAI Findings

- **Global Analysis (SHAP):** Confirms psychological variable dominance explains optimization failure
- **Individual Analysis (LIME):** Successfully provides patient-specific risk communication despite limitations
- **Root Cause Validation:** Dataset attempts cardiac prediction from lifestyle surveys vs. clinical assessments
- **Optimization Mechanism:** Hyperparameter tuning optimizes weak signals, creating overfitting
- **Clinical Application Value:** Dual XAI reveals both fundamental limitations and practical communication value

![](results/plots/shap_feature_importance_academic.png)

### LIME Individual Patient Analysis (Local Interpretability)

**Personalized Risk Factor Assessment:**

- **Individual Explanations:** Patient-specific risk factor contributions
- **Clinical Communication:** Professional medical language for practitioner use
- **Robust Implementation:** Graceful fallback system for production reliability
- **Dual XAI Strategy:** Global population insights + Local patient analysis

**Key LIME Capabilities:**
- Real-time individual patient explanations
- Risk factor classification (Protective/Risk factors)
- Professional clinical presentation format
- Enhanced patient-provider communication

---

## Professional Application Development

### Complete Healthcare Interface

**Technical Achievement:**

- **Medical-Grade Interface:** Gradio with healthcare industry standards
- **Dual XAI Integration:** SHAP research insights  + LIME individual explanations for comprehensive explainability
- **Personalized Analysis:** Real-time individual risk factor explanations using LIME with professional fallback system
- **Risk Stratification:** Evidence-based Low/Moderate/High classification with WHO/AHA threshold validation
- **Clinical Decision Support:** Traffic light system (6-4 threshold) based on validated health behavior scales
- **Safety Compliance:** Medical disclaimers and professional consultation requirements
- **Development Environment:** Docker containerization with environment detection

**Demonstrates Complete Research-to-Production Pipeline**

![](results/plots/gradio_application_interface.png)

**Current interface featuring:**

- **Risk Probability:** Clinical terminology (not "Model Confidence")  
- **Personalized Risk Analysis (LIME):** Individual patient explanations
- **Three-Level Risk Classification:** Low Risk, Moderate Risk, High Risk with clinical probabilities
- **Clinical Recommendations:** Evidence-based guidance
- **Clinical Risk Messaging:** Evidence-based recommendations with probability thresholds

### Current Application Features

**Complete Dual XAI Implementation:**

- **LIME Individual Explanations**: Real-time personalized risk factor analysis with professional medical language
- **Dual XAI Display**: Global SHAP research insights + Local LIME patient explanations  
- **Professional Medical Interface**: Clinical terminology, three-tier risk classification, and medical-grade presentation
- **Robust Fallback System**: Professional analysis ensuring 100% uptime regardless of dependencies
- **Enhanced Patient Communication**: Individual risk factor explanations with clinical evidence base
- **Professional Development Environment**: Complete Docker integration with intelligent environment detection for testing

### Strategic Research Value

The application demonstrates professional development methodology with transparent limitation communication and successful integration of both global (SHAP) and local (LIME) explainable AI techniques. This represents a comprehensive approach to responsible healthcare AI development, combining rigorous research analysis with professional interface development suitable for testing and demonstration purposes.

---

## Research Implications & Conclusions

### Principal Research Contributions

**1. Optimization Paradox Discovery**

- **Statistical Evidence:** 43% F1-score degradation (30.8% → 17.5%) with 65% sensitivity decline (40.5% → 14.3%)
- **Clinical Impact:** Optimization increased dangerous false negatives from 59.5% to 85.7%
- **Literature Gap:** First systematic documentation contradicting published optimization benefits
- **Generalization:** Validated across 3 optimized models with consistent degradation patterns
- **Healthcare Implications:** Challenges core ML optimization assumptions for medical applications

**2. Clinical Safety Framework**

- **Evidence-Based Criteria:** Established ≥80% sensitivity requirement based on cardiac screening literature
- **Economic Analysis:** €152.52 cost per patient with 97 missed cases per 1000 patients
- **Regulatory Assessment:** No models meet FDA/CE medical device safety standards
- **Risk Stratification:** Complete three-tier classification (Low <25%, Moderate 25-35%, High ≥35%)
- **Safety Validation:** Demonstrated systematic evaluation preventing dangerous deployment

**3. Dual Explainable AI Implementation**

- **Technical Achievement:** First integrated SHAP (global) + LIME (individual) system in healthcare ML
- **Research Validation:** SHAP analysis confirms psychological feature limitations explain optimization failure
- **Clinical Application:** LIME provides individual patient explanations with professional fallback system
- **Professional Standards:** Medical-grade interface with comprehensive disclaimers and safety protocols
- **Implementation Proof:** Complete containerized application demonstrates research-to-production pipeline

**4. Honest Academic Assessment**

- **Literature Reality Check:** Our 17.5% F1-score vs. published 65-89% reveals publication bias
- **Reproducible Research:** Complete Docker infrastructure enables verification and replication
- **Transparent Methodology:** Full documentation of both successes and failures
- **Clinical Responsibility:** Prevents potential patient harm through honest limitation reporting

### Quantified Research Impact

- **Performance Benchmark:** Established realistic expectations for lifestyle-based cardiac prediction
- **Safety Standards:** Created evidence-based deployment criteria for healthcare ML
- **Technical Innovation:** Demonstrated complete dual XAI integration with production-ready infrastructure
- **Academic Contribution:** Provided first systematic negative results documentation in healthcare ML optimization

### Strategic Healthcare AI Recommendations

1. **Prioritize Clinical Features:** Traditional biomarkers essential - psychological surveys insufficient
2. **Safety-First Optimization:** Develop healthcare-specific optimization with sensitivity constraints
3. **Mandatory XAI Integration:** Both global (SHAP) and individual (LIME) explainability required
4. **Transparent Research Standards:** Publish negative results to prevent repeated failures
5. **Regulatory Compliance:** Systematic safety evaluation before any clinical deployment consideration

---

## Future Work & Limitations

### Research Limitations

- **Dataset Constraints:** Psychological/lifestyle emphasis vs. clinical biomarkers
- **Demographic Scope:** European population may limit global generalizability
- **Cross-Sectional Design:** Lacks longitudinal cardiac risk progression data

### Future Research Directions

- **Clinical Data Integration:** Incorporate ECG, biomarkers, and imaging data
- **Healthcare-Specific Optimization:** Develop safety-constrained ML frameworks
- **Longitudinal Validation:** Multi-year cardiac outcome prediction studies
- **Regulatory Framework Development:** Evidence-based healthcare AI deployment standards

### Ethical Considerations

Comprehensive research ethics compliance with transparent limitation reporting to prevent potential misuse and ensure patient safety prioritization.

---

## Thank You

### Questions and Discussion Welcome

**Novel Research Contribution:**  
*First systematic documentation of the optimization paradox in healthcare machine learning with comprehensive dual explainable AI implementation (SHAP + LIME)*

**Research Repository:**  
https://github.com/Petlaz/heart_risk_prediction

**Contact:**  
Peter Ugonna Obi | Prof. Dr. Beate Rhein  
Nightingale Heart Partnership – Mr. Håkan Lane

---

*"Responsible healthcare AI development requires honest assessment of both successes and limitations."*
