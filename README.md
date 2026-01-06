# Heart Risk Prediction with Explainable AI

## Project Overview

This project develops an interpretable machine learning system for predicting heart disease risk using the European Social Survey health data. The system combines high-performance prediction models with explainable AI (XAI) techniques to provide clinically relevant insights.

## Features

- **Data Processing**: Comprehensive EDA and preprocessing pipeline
- **Model Training**: Multiple ML algorithms with hyperparameter tuning  
- **Explainability**: SHAP and LIME implementations for model interpretability
- **Clinical Integration**: Decision support templates for healthcare professionals
- **Interactive Interface**: Gradio-based web application for predictions and explanations
- **Containerization**: Docker support for reproducible deployment

## Quick Start

### Local Development

1. **Clone and setup environment**:
   ```bash
   git clone <repository-url>
   cd heart_risk_prediction
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the Gradio application**:
   ```bash
   python app/app_gradio.py
   ```

### Docker Deployment

```bash
docker-compose up --build
```

The application will be available at `http://localhost:7860`

## Project Structure

```
heart_risk_prediction/
├── .github/                     # GitHub workflows and configurations
├── .gitignore                   # Git ignore patterns
├── LICENSE                      # MIT License
├── README.md                    # This file - project documentation
├── requirements.txt             # Host environment dependencies
├── app/                         # Gradio demo application
│   ├── __init__.py
│   └── app_gradio.py           # Interactive web interface for predictions & explanations
├── data/
│   ├── raw/                     # Original survey datasets (read-only)
│   │   └── heart_data.csv      # European Social Survey health data
│   ├── processed/               # Clean splits + artifacts
│   │   └── feature_names.csv   # Feature mapping and descriptions
│   ├── data_dictionary_backup.md
│   └── data_dictionary.md       # Auto-generated feature documentation
├── docker/
│   ├── Dockerfile               # Container runtime for notebooks + Gradio
│   ├── docker-compose.yml       # Multi-service orchestration
│   ├── entrypoint_app.sh        # Application startup script
│   ├── README.md               # Docker setup documentation
│   └── requirements.txt         # Docker-specific dependencies
├── notebooks/
│   ├── 01_eda.ipynb            # Data exploration and EDA
│   ├── 02_data_preprocessing.ipynb # Data preprocessing pipeline
│   ├── 03_modeling.ipynb       # Model training and evaluation
│   ├── 04_error_analysis.ipynb # Error analysis and diagnostics
│   └── 05_explainability.ipynb # XAI implementation and testing
├── reports/
│   ├── biweekly_meeting_1.md   # Sprint 1-2 progress summary
│   ├── biweekly_meeting_2.md   # Sprint 3-4 progress summary  
│   ├── biweekly_meeting_3.md   # Sprint 5-6 progress summary
│   ├── biweekly_meeting_4.md   # Sprint 7-8 progress summary
│   ├── biweekly_meeting_5.md   # Sprint 9-10 progress summary
│   ├── biweekly_meeting_6.md   # Sprint 11-12 progress summary
│   ├── final_report.md         # Academic thesis/final report
│   ├── literature_review.md    # Research background and related work
│   └── project_plan_and_roadmap.md # Project timeline and milestones
├── results/
│   ├── confusion_matrices/     # Model confusion matrix visualizations
│   │   └── .gitkeep
│   ├── explainability/        # XAI artifacts and visualizations
│   │   └── clinical/          # Clinical decision support templates
│   ├── explanations/          # Local explanation artifacts
│   │   ├── README.md
│   │   └── clinical_decision_support_template.md
│   ├── metrics/               # Performance metrics and diagnostics
│   │   └── classification_reports/
│   ├── models/                # Trained model artifacts
│   └── plots/                 # Visualization outputs
└── src/
    ├── __init__.py
    ├── config.yaml            # Project configuration settings
    ├── data_preprocessing.py  # EDA + preprocessing pipeline
    ├── train_models.py        # Model training orchestration
    └── utils.py               # Shared utilities and helpers
```

## Usage

### Jupyter Notebooks

The analysis is organized across 5 comprehensive notebooks:

1. **01_eda.ipynb**: Initial data exploration and visualization
2. **02_data_preprocessing.ipynb**: Data cleaning, feature engineering, and splitting
3. **03_modeling.ipynb**: Model training, evaluation, and selection
4. **04_error_analysis.ipynb**: Error analysis and model diagnostics
5. **05_explainability.ipynb**: XAI implementation and testing

### Python Scripts

- **data_preprocessing.py**: Automated data processing pipeline
- **train_models.py**: Model training with hyperparameter optimization
- **evaluate_models.py**: Model evaluation and performance analysis
- **explainability.py**: Generate SHAP/LIME explanations

### Configuration

Modify `src/config.yaml` to adjust:
- Model hyperparameters
- Data processing settings
- Explanation generation parameters
- Output directories

## Data

The project uses the European Social Survey health dataset with features including:
- Demographics (age, gender, country)
- Lifestyle factors (exercise, diet, smoking, alcohol)
- Mental health indicators (happiness, depression, stress)
- Medical history (diabetes, blood pressure)
- Social factors (community engagement, noise exposure)

## Models

Implemented algorithms:
- Logistic Regression
- Random Forest
- Support Vector Machine
- Neural Network
- Gradient Boosting (XGBoost)

## Explainability

XAI techniques implemented:
- **SHAP**: Feature importance and interaction analysis
- **LIME**: Local instance explanations
- **Clinical Templates**: Healthcare professional decision support

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- European Social Survey for providing the health dataset
- SHAP and LIME libraries for explainability frameworks
- Gradio team for the interactive interface framework
