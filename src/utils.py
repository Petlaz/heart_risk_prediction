"""
Shared utilities and helper functions for the Heart Risk Prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging

def load_config(config_path="src/config.yaml"):
    """Load project configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def save_plot(fig, filename, plots_dir="results/plots/"):
    """Save matplotlib figure to plots directory"""
    ensure_dir(plots_dir)
    filepath = Path(plots_dir) / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath

def load_feature_names():
    """Load feature name mappings"""
    try:
        feature_names_path = "data/processed/feature_names.csv"
        return pd.read_csv(feature_names_path)
    except FileNotFoundError:
        logging.warning("Feature names file not found")
        return None

def create_summary_stats(df):
    """Generate summary statistics for a dataframe"""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else None
    }
    return summary

def plot_confusion_matrix(conf_matrix, model_name, save_path=None):
    """Plot and optionally save confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def calculate_feature_importance(model, feature_names):
    """Extract feature importance from model if available"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance_df
    elif hasattr(model, 'coef_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        return importance_df
    else:
        return None

def create_results_directory_structure():
    """Create the complete results directory structure"""
    directories = [
        "results/",
        "results/confusion_matrices/",
        "results/explainability/",
        "results/explainability/clinical/",
        "results/explanations/",
        "results/metrics/",
        "results/metrics/classification_reports/",
        "results/models/",
        "results/plots/"
    ]
    
    for directory in directories:
        ensure_dir(directory)

def format_percentage(value):
    """Format decimal as percentage string"""
    return f"{value * 100:.1f}%"

def get_model_performance_summary(metrics_list):
    """Create a summary of model performance"""
    summary_df = pd.DataFrame(metrics_list)
    
    # Sort by ROC AUC score
    summary_df = summary_df.sort_values('roc_auc', ascending=False)
    
    # Format percentage columns
    percentage_cols = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']
    for col in percentage_cols:
        if col in summary_df.columns:
            summary_df[f'{col}_formatted'] = summary_df[col].apply(format_percentage)
    
    return summary_df

class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def check_missing_values(df, threshold=0.5):
        """Check for columns with high missing value percentages"""
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > threshold]
        return high_missing
    
    @staticmethod
    def check_data_types(df, expected_types=None):
        """Validate data types match expectations"""
        if expected_types:
            type_mismatches = {}
            for col, expected_type in expected_types.items():
                if col in df.columns:
                    if df[col].dtype != expected_type:
                        type_mismatches[col] = {
                            'expected': expected_type,
                            'actual': df[col].dtype
                        }
            return type_mismatches
        return {}
    
    @staticmethod
    def check_value_ranges(df, expected_ranges=None):
        """Check if numeric values fall within expected ranges"""
        if expected_ranges:
            range_violations = {}
            for col, (min_val, max_val) in expected_ranges.items():
                if col in df.columns:
                    out_of_range = df[(df[col] < min_val) | (df[col] > max_val)][col]
                    if not out_of_range.empty:
                        range_violations[col] = {
                            'expected_range': (min_val, max_val),
                            'violations': out_of_range.tolist()[:10]  # First 10 violations
                        }
            return range_violations
        return {}