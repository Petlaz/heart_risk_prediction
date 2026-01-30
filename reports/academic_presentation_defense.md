# Heart Disease Risk Prediction with Explainable AI
**A Comprehensive Machine Learning Approach to Clinical Decision Support**

**Master's Research Project Defense**

- **Author:** Peter Ugonna Obi
- **Supervisor:** Prof. Dr. Beate Rhein  
- **Industry Partner:** Nightingale Heart – Mr. Håkan Lane
- **Date:** January 2026

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
- **Features:** 22 health, lifestyle, and psychological variables
- **Target:** Binary heart condition prediction

### Dual-Method Research Framework

**Method 1: Comprehensive ML Pipeline**

- Baseline evaluation (5 algorithms: NN, XGBoost, SVM, LR, RF)
- Systematic hyperparameter optimization using RandomizedSearchCV
- Clinical performance assessment with safety criteria

**Method 2: Production-Ready Application**

- Professional Gradio web interface with medical compliance
- Docker containerization for deployment readiness

![](results/plots/ml_pipeline_diagram.png)

---

## Baseline Results

### Algorithm Performance Overview

| Model | F1-Score | Sensitivity | Specificity | Clinical Assessment |
|-------|----------|-------------|-------------|---------------------|
| **Neural Network** | **30.8%** | **40.5%** | 75.2% | Best overall performance |
| **XGBoost** | 30.4% | 50.8% | 73.7% | Highest sensitivity |
| **Logistic Regression** | 29.0% | 62.5% | 65.4% | Highest recall |
| **Random Forest** | 28.9% | 36.4% | 79.8% | Best specificity |

### Key Baseline Insights

- Moderate performance established optimization potential
- Neural Network achieved optimal precision-recall balance
- Logistic Regression's 62.5% sensitivity approached clinical viability
- Algorithmic diversity suggested ensemble optimization potential

![](results/plots/roc_curves_baseline_models.png)

---

## Critical Discovery: The Optimization Paradox

### Systematic Optimization Results

**Optimization Method:** RandomizedSearchCV with F1-score optimization targeting improved clinical sensitivity

| Phase | Best Model | F1-Score | Sensitivity | Performance Change |
|-------|------------|----------|-------------|-------------------|
| **Baseline** | Neural Network | **30.8%** | **40.5%** | Starting performance |
| **Optimized** | Adaptive_Ensemble | **17.5%** | **14.3%** | **43% F1 decline** |
| | | | | **65% sensitivity decline** |

### The Optimization Paradox Explained

- **Catastrophic Performance Degradation:** Systematic hyperparameter optimization worsened clinical performance
- **False Negative Explosion:** Sensitivity collapsed from 40.5% to 14.3%
- **Healthcare ML Challenge:** Traditional optimization frameworks contraindicated for clinical applications
- **Novel Research Contribution:** First documented evidence of optimization paradox in healthcare ML

![](results/plots/optimization_paradox_comparison.png)

---

## Clinical Deployment Assessment

### Healthcare Safety Criteria

- **Required Sensitivity:** ≥80% (to avoid missing heart disease cases)
- **Required Specificity:** ≥60% (to control false alarm burden)
- **Regulatory Standards:** Medical device safety protocols

### Deployment Verdict

| Model Category | Best Sensitivity | Missed Cases (%) | Clinical Status | Safety Assessment |
|---------------|------------------|------------------|------------------|-------------------|
| **Baseline Models** | 40.5% | 59.5% | Below criteria | Insufficient |
| **Optimized Models** | 14.3% | 85.7% | Critical failure | **Unacceptable** |

### Critical Clinical Impact

- **85.7% False Negative Rate:** Poses unacceptable patient endangerment
- **Regulatory Non-Compliance:** No models meet clinical deployment safety standards
- **Economic Paradox:** Despite acceptable cost per patient (€152), safety risks create insurmountable liability

---

## Dual Explainable AI Implementation

### SHAP Analysis Reveals Root Causes (Global Insights)

**Top Predictive Features (SHAP Values):**

| Rank | Feature | SHAP Value | Clinical Validity | Predictive Signal |
|------|---------|------------|------------------|------------------|
| 1 | **BMI** | 0.0208 | Strong cardiac risk factor | **Valid** |
| 2 | **Physical Activity** | 0.0189 | Established cardiac protection | **Valid** |
| 3 | **Mental Effort** | 0.0149 | Psychological indicator | **Weak** |
| 4 | **Sleep Quality** | 0.0126 | Moderate cardiac relevance | **Moderate** |
| 5 | **Happiness/Mood** | 0.0093-0.0079 | Psychological factors | **Weak** |

### Critical Dual XAI Findings

- **Global Analysis (SHAP):** 60% of top features are psychological variables with weak cardiac predictive validity
- **Individual Analysis (LIME):** Enables personalized risk communication despite dataset limitations
- **Missing clinical markers:** No ECG, biomarkers, or imaging data
- **Optimization paradox mechanism:** Hyperparameter tuning cannot overcome fundamental dataset limitations
- **Clinical Application Value:** Dual XAI approach provides both research insights and practical patient communication

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
- **Deployment Ready:** Docker containerization with environment detection

**Demonstrates Complete Research-to-Production Pipeline**

![](results/plots/gradio_application_interface.png)

**Current interface featuring:**

- **Risk Probability:** Clinical terminology (not "Model Confidence")  
- **Personalized Risk Analysis (LIME):** Individual patient explanations
- **Key Risk Factors Summary:** Professional emoji indicators (🔴🟡✅)
- **Clinical Recommendations:** Evidence-based guidance
- **Professional Formatting:** Medical-grade presentation with em-dash formatting

### Current Application Features

**Complete Dual XAI Implementation:**

- **LIME Individual Explanations**: Real-time personalized risk factor analysis with professional medical language
- **Dual XAI Display**: Global SHAP research insights + Local LIME patient explanations  
- **Professional Medical Interface**: Clinical terminology, emoji indicators, and proper formatting
- **Robust Fallback System**: Professional analysis ensuring 100% uptime regardless of dependencies
- **Enhanced Patient Communication**: Individual risk factor explanations with clinical evidence base
- **Production-Ready Deployment**: Complete Docker integration with intelligent environment detection

### Strategic Research Value

The updated interface demonstrates **complete research-to-production methodology** with transparent limitation communication and successful integration of both global (SHAP) and local (LIME) explainable AI techniques. This represents a comprehensive approach to responsible healthcare AI development, combining rigorous research analysis with practical clinical deployment capabilities.

---

## Research Implications & Conclusions

### Principal Research Contributions

**1. Optimization Paradox Discovery**

- First documentation of systematic optimization degrading healthcare ML performance
- 65% sensitivity decline challenges core ML optimization assumptions
- Critical implications for healthcare AI safety protocols

**2. Clinical Safety Framework**

- Established evidence-based deployment criteria (≥80% sensitivity)
- Demonstrated systematic evaluation of regulatory compliance
- Highlighted patient safety vs. algorithmic performance tensions

**3. Dual Explainable AI Implementation**

- SHAP analysis revealed dataset limitation mechanisms at global level
- LIME integration provided individual patient-level explanations
- Demonstrated complete XAI pipeline from research to clinical application
- Professional fallback system ensures robust deployment

### Strategic Healthcare AI Recommendations

1. **Prioritize Clinical Features:** Traditional biomarkers essential for viable cardiac prediction
2. **Safety-First Optimization:** Develop healthcare-specific optimization frameworks
3. **Mandatory XAI Integration:** Explainability required for clinical validation
4. **Transparent Research Standards:** Honest assessment of deployment failures

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
