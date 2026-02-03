# Heart Diesease Risk Prediction with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange.svg)](https://gradio.app/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-green.svg)](https://scikit-learn.org/)
[![XAI](https://img.shields.io/badge/XAI-SHAP-lightblue.svg)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Master's%20Project-purple.svg)](#)

## 🫀 Project Overview

An advanced machine learning system for predicting heart disease risk using comprehensive health and lifestyle data from the European Social Survey. The project integrates state-of-the-art ML algorithms with explainable AI (XAI) techniques, delivering insights through a professional web application with Docker containerization.

**Status**: Production Complete | **Performance**: Research-grade analysis | **Focus**: Clinical applicability

## ⭐ Key Features

🧠 **Machine Learning Pipeline**
- 5 baseline algorithms with comprehensive evaluation  
- Advanced hyperparameter optimization using RandomizedSearchCV
- Clinical safety assessment and application readiness evaluation

🔍 **Explainable AI (XAI)**  
- SHAP implementation for global feature importance analysis
- LIME integration for individual patient-level explanations
- Dual XAI approach: global insights + personalized risk factors
- Clinical interpretation of model decisions
- Root cause investigation of optimization challenges

**Production Web Application**
- Professional medical interface with evidence-based risk assessment
- Three-tier classification system (Low/Moderate/High risk) 
- Comprehensive explainable AI with SHAP and LIME analysis
- Clinical interpretation with patient-friendly explanations
- Medical-grade styling with comprehensive safety guidance

**Docker Containerization**
- Complete containerization with automatic environment detection
- Configuration with optimized port management
- Professional logging and system validation protocols

**Academic Documentation**
- Comprehensive final report and literature review
- 58 peer-reviewed references (2019-2026)
- Master's thesis-quality documentation and analysis

## Research Achievements

### Model Performance Analysis
| Model | F1-Score | Sensitivity | Specificity | Clinical Status |
|-------|----------|-------------|-------------|------------------|
| **Adaptive_Ensemble** | **0.175** | 14.3% | 98.4% | Best performance |
| Optimal_Hybrid | 0.091 | 5.4% | 99.1% | Extremely conservative |
| Adaptive_LR | 0.032 | 1.8% | 99.4% | Clinically unusable |

### Key Findings
- **Performance Gap**: Significant disparity vs. published benchmarks (0.65-0.92 F1) 
- **Clinical Safety**: All models fail clinical safety criteria (≥80% sensitivity required)
- **Feature Analysis**: BMI (0.0208) and Physical Activity (0.0189) top predictors
- **Medical Risk**: 822 missed cases per 8,476 patients represents unacceptable clinical risk

### Clinical Implications
- **Dataset Limitations**: Missing traditional cardiac risk factors (ECG, chest pain, family history)
- **Feature Quality**: Lifestyle/psychological variables insufficient for clinical-grade prediction
- **Optimization Paradox**: Standard ML optimization may worsen healthcare performance
- **Academic Impact**: Demonstrates need for honest assessment in healthcare ML research

## � Installation

### Prerequisites
- Python 3.8+ 
- Git
- Docker (optional, for containerization)

### Clone Repository
```bash
# Clone the repository
git clone https://github.com/Petlaz/heart_risk_prediction.git
cd heart_risk_prediction

# Create virtual environment (recommended)
python -m venv heart_risk_env
source heart_risk_env/bin/activate  # On Windows: heart_risk_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Web Application Development

[![App Status](https://img.shields.io/badge/App-Production%20Ready-brightgreen?style=for-the-badge&logo=gradio)](app/app_gradio.py)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue?style=for-the-badge&logo=docker)](docker/docker-compose.yml)
[![XAI](https://img.shields.io/badge/XAI-SHAP%20%2B%20LIME-orange?style=for-the-badge)](results/explainability/)

### Docker Setup (Recommended)
```bash
# Build and run the containerized application
docker build -f docker/Dockerfile -t heart-risk-app .
docker run -d --name heart-risk-container -p 7860:7860 heart-risk-app

# Access the application
open http://localhost:7860
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app/app_gradio.py

# Access at http://localhost:7861
```

### Application Features
- **Automatic Environment Detection**: Intelligent port configuration based on execution context
- **Professional Medical Interface**: Clinical-grade styling with comprehensive safety protocols
- **Three-Tier Risk Assessment**: Low/Moderate/High classification with clinical thresholds
- **Dual Explainable AI**: SHAP global analysis and LIME individual explanations
- **Evidence-Based Interpretation**: Clinical risk factor analysis with patient-friendly language
- **Containerization Ready**: Docker setup with optimized performance

## Research Methodology

### Implementation Framework
1. **Hyperparameter Optimization**: RandomizedSearchCV with F1-score optimization and Apple Silicon efficiency
2. **Model Validation**: Test set validation revealing performance hierarchy and overfitting patterns
3. **Error Analysis**: Comprehensive post-optimization investigation with clinical safety assessment
4. **Literature Review**: 58 peer-reviewed references (2019-2026) with healthcare ML methodology analysis
5. **Explainable AI**: SHAP implementation for clinical feature importance and root cause analysis
6. **Application Framework**: Professional Docker containerization with medical-grade interface

### Performance Results
| Model | Test F1-Score | Sensitivity | Specificity | Clinical Assessment |
|-------|---------------|-------------|-------------|---------------------|
| **Adaptive_Ensemble** | **0.175** | 14.3% | 98.4% | Best performance but clinically unsafe |
| Optimal_Hybrid | 0.091 | 5.4% | 99.1% | Extremely conservative, misses 94.6% cases |
| Adaptive_LR | 0.032 | 1.8% | 99.4% | Nearly unusable for screening |

### Key Findings
- **Clinical Safety Failure**: All models fail healthcare safety criteria (required ≥80% sensitivity)
- **Performance Gap**: Significant disparity vs. published benchmarks (65-92% F1) indicates methodological challenges
- **Psychological Insufficiency**: Mental health variables inadequate for clinical-grade heart disease prediction
- **Economic Impact**: 822 missed cases per 8,476 patients represents unacceptable medical risk

## Project Structure

```
heart_risk_prediction/
├── app/                         # Web Application
│   └── app_gradio.py            # Professional medical interface
├── data/                        # Dataset & Processing  
│   ├── raw/heart_data.csv        # European Social Survey data
│   └── processed/               # Clean splits & preprocessing artifacts
├── docker/                      # Containerization
│   ├── Dockerfile               # Optimized container image
│   ├── docker-compose.yml       # Multi-service orchestration
│   └── entrypoint_app.sh        # Professional startup script
├── notebooks/                   # Research Pipeline (Complete)
│   ├── 01_eda.ipynb             # Data exploration & analysis
│   ├── 02_data_processing.ipynb # Preprocessing & feature engineering 
│   ├── 03_modeling.ipynb        # Model training & optimization
│   ├── 04_error_analysis.ipynb  # Clinical safety assessment
│   └── 05_explainability.ipynb  # SHAP analysis & XAI implementation
├── reports/                     # Academic Documentation
│   ├── final_report.md          # Master's thesis report
│   └── literature_review.md     # 58-reference state-of-the-art review
├── results/                     # Model Artifacts & Analysis
│   ├── explainability/          # SHAP visualizations & clinical insights
│   ├── models/                  # Trained models & preprocessing artifacts
│   └── metrics/                 # Performance evaluation results
└── src/                        # Core Implementation
    ├── tuning/                  # Hyperparameter optimization
    └── analysis/                # Clinical decision support
```

## Research Pipeline

### Jupyter Notebooks
Systematic 5-notebook research methodology:

1. **01_eda.ipynb** - Data exploration and statistical analysis
2. **02_data_processing.ipynb** - Preprocessing and feature engineering  
3. **03_modeling.ipynb** - Baseline models and hyperparameter optimization
4. **04_error_analysis.ipynb** - Clinical safety assessment and diagnostics
5. **05_explainability.ipynb** - SHAP implementation and XAI analysis

### Dataset Information
- **Source**: European Social Survey Health Dataset
- **Sample Size**: 8,476 patients with 22 health/lifestyle features  
- **Target**: Binary heart disease indicator (18.4% positive rate)
- **Splits**: 70% train / 15% validation / 15% test (stratified)

### Configuration
Modify `src/config.yaml` for:
- Model hyperparameters and training settings
- Data processing and feature engineering 
- Clinical evaluation thresholds
- XAI explanation parameters

## Research Methodology & Critical Findings

### Optimization Framework
1. **Hyperparameter Optimization**: RandomizedSearchCV with F1-score focus
2. **Model Validation**: Comprehensive test set evaluation 
3. **Clinical Assessment**: Healthcare safety criteria analysis
4. **Error Investigation**: Systematic diagnostic and safety evaluation
5. **XAI Implementation**: SHAP analysis for feature importance and root cause investigation

### Critical Research Findings

**Clinical Safety Crisis**
- All models fail healthcare safety criteria (required ≥80% sensitivity)
- 822 missed heart disease cases per 8,476 patients = unacceptable medical risk
- Best model sensitivity: only 14.3% (misses 85.7% of actual cases)

**Performance vs. Literature Gap** 
- Our results: 17.5% F1-score vs. published benchmarks: 65-92% F1
- Significant methodological challenges identified
- Demonstrates need for honest assessment in healthcare ML

**Root Cause Analysis (via SHAP)**
- Missing traditional cardiac risk factors (ECG, chest pain, family history)
- Dataset emphasizes psychological/lifestyle factors with weak predictive signal  
- Optimization paradox: weak predictors cannot be optimized for clinical performance

## Academic Documentation

### Research Reports
- **[Final Report](reports/final_report.md)**: Master's thesis with comprehensive methodology and findings
- **[Literature Review](reports/literature_review.md)**: 58 peer-reviewed references (2019-2026)
- **[Technical Documentation](reports/)**: Progress reports and implementation specifications

### Research Impact
- **Academic Contribution**: Methodology for honest assessment of healthcare ML challenges
- **Clinical Insight**: Demonstrates gap between ML research and clinical application
- **Technical Innovation**: Complete end-to-end pipeline with professional containerization
- **Safety Focus**: Patient-centered evaluation prioritizing clinical safety over accuracy

---

## Contributing

### Academic Research
1. Review methodology in [final report](reports/final_report.md)
2. Examine implementation in research notebooks
3. Propose improvements via GitHub issues
4. Follow professional documentation standards

### Technical Development  
- Fork repository for experimental features
- Create descriptive feature branches
- Maintain clinical evaluation compatibility
- Submit PRs with comprehensive testing

---

## License & Acknowledgments

**License**: MIT License - see [LICENSE](LICENSE) file

**Acknowledgments**:
- European Social Survey for health dataset
- SHAP team for explainability framework  
- Gradio team for web interface capabilities
- PyTorch and Scikit-learn communities

---

**Project Status**: Production Complete | **Last Updated**: January 2026 | **Author**: Peter Ugoona Obi
