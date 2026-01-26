# Heart Risk Prediction with Explainable AI
**A Comprehensive Machine Learning Approach to Clinical Decision Support**

**Master's Research Project Defense**

- **Author:** Peter Ugonna Obi
- **Supervisor:** Prof. Dr. Beate Rhein  
- **Industry Partner:** Nightingale Heart – Mr. Håkan Lane
- **Date:** January 2026

::: notes
Open with the clinical urgency: heart disease remains the leading global cause of death (17.9M annually, WHO 2023). This investigation addresses a critical gap between published ML performance claims and practical deployment outcomes in healthcare settings.
:::

---

## Research Problem & Questions

### The Healthcare ML Deployment Crisis

- **Performance Gap:** Published studies report 65-89% F1-scores, but real deployment often fails
- **Clinical Safety Risk:** Traditional ML optimization may compromise medical safety requirements  
- **Black Box Problem:** Lack of explainability prevents clinical adoption

### Core Research Questions

1. How do systematically optimized models perform compared to baseline implementations?
2. What drives misclassification patterns in healthcare ML applications?
3. Can explainable AI reveal fundamental limitations in psychological-based cardiac prediction?

**Research Contribution:** First comprehensive study documenting the "optimization paradox" in healthcare ML

::: notes
Frame this as addressing a critical methodological crisis in healthcare AI. The optimization paradox represents our novel contribution to the field.
:::

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

::: notes
Emphasize the methodological innovation of testing complete research-to-production pipeline, unlike studies that stop at cross-validation.
:::

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

::: notes
Position these results as reasonable starting points that created expectation for improvement through systematic optimization.
:::

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

::: notes
This is your key research contribution. Position this as a fundamental discovery that challenges core ML assumptions in healthcare contexts.
:::

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

::: notes
Emphasize the clinical safety implications. The optimization paradox isn't just an academic finding—it has serious patient safety consequences.
:::

---

## Explainable AI Insights

### SHAP Analysis Reveals Root Causes

**Top Predictive Features (SHAP Values):**

| Rank | Feature | SHAP Value | Clinical Validity | Predictive Signal |
|------|---------|------------|------------------|------------------|
| 1 | **BMI** | 0.0208 | Strong cardiac risk factor | **Valid** |
| 2 | **Physical Activity** | 0.0189 | Established cardiac protection | **Valid** |
| 3 | **Mental Effort** | 0.0149 | Psychological indicator | **Weak** |
| 4 | **Sleep Quality** | 0.0126 | Moderate cardiac relevance | **Moderate** |
| 5 | **Happiness/Mood** | 0.0093-0.0079 | Psychological factors | **Weak** |

### Critical XAI Findings

- **60% of top features** are psychological variables with weak cardiac predictive validity
- **Missing clinical markers:** No ECG, biomarkers, or imaging data
- **Optimization paradox mechanism:** Hyperparameter tuning cannot overcome fundamental dataset limitations

![](results/plots/shap_feature_importance_academic.png)

::: notes
SHAP analysis provides the mechanistic explanation for the optimization paradox. We're asking models to predict heart disease from happiness surveys rather than clinical assessments.
:::

---

## Professional Application Development

### Complete Healthcare Interface

**Technical Achievement:**

- **Medical-Grade Interface:** Gradio with healthcare industry standards
- **SHAP-Informed Design:** Application uses research findings (Weeks 5-6) to inform feature selection rather than real-time XAI computation
- **Risk Stratification:** Evidence-based Low/Moderate/High classification with WHO/AHA threshold validation
- **Clinical Decision Support:** Traffic light system (6-4 threshold) based on validated health behavior scales
- **Safety Compliance:** Medical disclaimers and professional consultation requirements
- **Deployment Ready:** Docker containerization with environment detection

**Demonstrates Complete Research-to-Production Pipeline**

![](results/plots/gradio_application_interface.png)

### Strategic Research Value

Despite model limitations, demonstrates methodological capability for responsible healthcare AI development with transparent limitation communication and separation of research analysis from practical deployment.

::: notes
Position the application as demonstrating complete methodology while maintaining transparency about limitations—a model for responsible healthcare AI research.
:::

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

**3. Explainable AI Integration**

- SHAP analysis revealed dataset limitation mechanisms
- Psychological features insufficient for reliable cardiac prediction
- Evidence-based feature engineering recommendations

### Strategic Healthcare AI Recommendations

1. **Prioritize Clinical Features:** Traditional biomarkers essential for viable cardiac prediction
2. **Safety-First Optimization:** Develop healthcare-specific optimization frameworks
3. **Mandatory XAI Integration:** Explainability required for clinical validation
4. **Transparent Research Standards:** Honest assessment of deployment failures

::: notes
Conclude by positioning your research as establishing new methodological standards for responsible healthcare AI development and contributing to evidence-based healthcare AI policy.
:::

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

::: notes
Acknowledge limitations while positioning future work as building on your methodological contributions. Emphasize the ethical responsibility of honest research reporting.
:::

---

## Thank You

### Questions and Discussion Welcome

**Novel Research Contribution:**  
*First systematic documentation of the optimization paradox in healthcare machine learning*

**Research Repository:**  
https://github.com/Petlaz/heart_risk_prediction

**Contact:**  
Peter Ugonna Obi | Prof. Dr. Beate Rhein  
Nightingale Heart Partnership – Mr. Håkan Lane

---

*"Responsible healthcare AI development requires honest assessment of both successes and limitations."*

::: notes
Be prepared to discuss: (1) The methodological implications of the optimization paradox, (2) How your findings challenge current healthcare ML research practices, (3) The clinical safety framework you've established, (4) Future research directions building on your contributions.
:::