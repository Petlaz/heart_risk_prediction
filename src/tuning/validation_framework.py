"""
Validation Framework for Heart Risk Prediction Models
Tests optimized models on unseen data and provides comprehensive performance analysis.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ValidationFramework:
    """Comprehensive validation framework for optimized models."""
    
    def __init__(self, project_root: str = None):
        """Initialize validation framework with project paths."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        self.data_path = self.project_root / 'data' / 'processed'
        self.models_path = self.project_root / 'results' / 'models'
        self.results_path = self.project_root / 'results'
        
        # Ensure results directories exist
        (self.results_path / 'metrics').mkdir(exist_ok=True)
        (self.results_path / 'plots').mkdir(exist_ok=True)
        
        # Model configurations
        self.optimized_models = {
            'Adaptive_LR': {
                'path': 'adaptive_tuning/Adaptive_LR_complexity_increased_20260108_233028.joblib',
                'type': 'sklearn',
                'expected_f1': 0.290
            },
            'Enhanced_Ensemble': {
                'path': 'enhanced_techniques/Enhanced_Ensemble_weighted_ensemble_20260108_231559.joblib', 
                'type': 'sklearn',
                'expected_f1': 0.284
            },
            'Enhanced_NN': {
                'path': 'enhanced_techniques/Enhanced_NN_regularized_20260108_231559.joblib',
                'type': 'sklearn', 
                'expected_f1': 0.282
            },
            'Optimal_Hybrid': {
                'path': 'adaptive_tuning/Optimal_Hybrid_optimal_hybrid_20260108_233028.joblib',
                'type': 'sklearn',
                'expected_f1': 0.275
            },
            'Adaptive_Ensemble': {
                'path': 'adaptive_tuning/Adaptive_Ensemble_complexity_optimized_20260108_233028.joblib',
                'type': 'sklearn',
                'expected_f1': 0.255
            },
            'Enhanced_LR': {
                'path': 'enhanced_techniques/Enhanced_LR_cost_sensitive_20260108_231559.joblib',
                'type': 'sklearn',
                'expected_f1': 0.140
            }
        }
        
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load test dataset for validation."""
        print("LOADING: Loading test dataset...")
        
        test_df = pd.read_csv(self.data_path / 'test.csv')
        
        # The target column is 'hltprhc' (health problems/condition)
        if 'hltprhc' in test_df.columns:
            X_test = test_df.drop('hltprhc', axis=1)
            y_test = test_df['hltprhc']
        elif 'target' in test_df.columns:
            X_test = test_df.drop('target', axis=1)
            y_test = test_df['target']
        else:
            # Assume last column is target
            X_test = test_df.iloc[:, :-1] 
            y_test = test_df.iloc[:, -1]
            
        print(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"Class distribution: {dict(y_test.value_counts().sort_index())}")
        
        return X_test, y_test
        
    def load_scaler(self):
        """Load the fitted scaler for data preprocessing."""
        # Check for preprocessing artifacts first
        preprocessing_path = self.data_path / 'preprocessing_artifacts.joblib'
        if preprocessing_path.exists():
            artifacts = joblib.load(preprocessing_path)
            if isinstance(artifacts, dict) and 'scaler' in artifacts:
                return artifacts['scaler']
        
        # Check for standalone scaler
        scaler_path = self.models_path / 'standard_scaler.joblib'
        if scaler_path.exists():
            return joblib.load(scaler_path)
        else:
            print("No scaler found - using raw features")
            return None
            
    def load_optimized_models(self) -> Dict[str, Any]:
        """Load all optimized models for validation."""
        print("LOADING: Loading optimized models...")
        
        loaded_models = {}
        scaler = self.load_scaler()
        
        for model_name, config in self.optimized_models.items():
            model_path = self.models_path / config['path']
            
            if model_path.exists():
                try:
                    model_data = joblib.load(model_path)
                    
                    # Extract actual model from saved structure
                    if isinstance(model_data, dict) and 'model' in model_data:
                        actual_model = model_data['model']
                    else:
                        actual_model = model_data
                        
                    loaded_models[model_name] = {
                        'model': actual_model,
                        'scaler': scaler,
                        'config': config,
                        'metadata': model_data if isinstance(model_data, dict) else None
                    }
                    print(f"{model_name}: Loaded successfully")
                except Exception as e:
                    print(f"{model_name}: Failed to load - {e}")
            else:
                print(f"{model_name}: Model file not found at {model_path}")
            
        print(f"Successfully loaded {len(loaded_models)} models")
        return loaded_models
        
    def calculate_clinical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate healthcare-specific performance metrics."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value/Precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Cost analysis (False Positive: €100, False Negative: €1000)
        cost_per_patient = (fp * 100 + fn * 1000) / len(y_true)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'cost_per_patient_eur': cost_per_patient
        }
        
    def validate_single_model(self, model_name: str, model_info: Dict, 
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Validate a single optimized model on test data."""
        print(f"\nValidating {model_name}...")
        
        model = model_info['model']
        scaler = model_info['scaler'] 
        config = model_info['config']
        
        # Preprocess data
        if scaler is not None:
            X_test_processed = scaler.transform(X_test)
        else:
            X_test_processed = X_test.values
            
        # Predictions
        try:
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        except Exception as e:
            print(f"Prediction failed for {model_name}: {e}")
            return None
            
        # Standard metrics
        metrics = {
            'model_name': model_name,
            'f1_score': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'expected_f1': config['expected_f1']
        }
        
        # Clinical metrics
        clinical_metrics = self.calculate_clinical_metrics(y_test, y_pred)
        metrics.update(clinical_metrics)
        
        # Performance comparison
        f1_difference = metrics['f1_score'] - config['expected_f1']
        metrics['f1_difference'] = f1_difference
        metrics['performance_status'] = self._assess_performance(f1_difference)
        
        # Detailed classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"{model_name} Results:")
        print(f"   F1 Score: {metrics['f1_score']:.3f} (Expected: {config['expected_f1']:.3f})")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"   Specificity: {metrics['specificity']:.3f}")
        print(f"   Status: {metrics['performance_status']}")
        
        return metrics
        
    def _assess_performance(self, f1_difference: float) -> str:
        """Assess model performance relative to expectations."""
        if f1_difference >= 0.02:
            return "EXCEEDS_EXPECTATIONS"
        elif f1_difference >= -0.02:
            return "MEETS_EXPECTATIONS" 
        elif f1_difference >= -0.05:
            return "BELOW_EXPECTATIONS"
        else:
            return "SIGNIFICANT_DEGRADATION"
            
    def statistical_significance_test(self, results: List[Dict]) -> Dict[str, Any]:
        """Test statistical significance of performance differences."""
        print("\nConducting statistical significance analysis...")
        
        # Compare top performers
        results_sorted = sorted(results, key=lambda x: x['f1_score'], reverse=True)
        
        if len(results_sorted) < 2:
            return {"message": "Need at least 2 models for comparison"}
            
        best_model = results_sorted[0]
        second_model = results_sorted[1]
        
        # Calculate confidence intervals (approximation using bootstrap logic)
        best_f1 = best_model['f1_score']
        second_f1 = second_model['f1_score'] 
        
        # Approximate standard error based on sample size
        n_samples = best_model['true_positives'] + best_model['false_negatives'] + \
                   best_model['true_negatives'] + best_model['false_positives']
        se_approx = np.sqrt(best_f1 * (1 - best_f1) / n_samples)
        
        # 95% confidence intervals
        ci_lower = best_f1 - 1.96 * se_approx
        ci_upper = best_f1 + 1.96 * se_approx
        
        # Check if confidence intervals overlap
        se_second = np.sqrt(second_f1 * (1 - second_f1) / n_samples)
        ci_second_lower = second_f1 - 1.96 * se_second
        ci_second_upper = second_f1 + 1.96 * se_second
        
        significant_difference = ci_lower > ci_second_upper or ci_upper < ci_second_lower
        
        significance_results = {
            'best_model': best_model['model_name'],
            'best_f1': best_f1,
            'best_ci_lower': ci_lower,
            'best_ci_upper': ci_upper,
            'second_model': second_model['model_name'], 
            'second_f1': second_f1,
            'second_ci_lower': ci_second_lower,
            'second_ci_upper': ci_second_upper,
            'statistically_significant': significant_difference,
            'f1_difference': best_f1 - second_f1
        }
        
        print(f"Best Model: {significance_results['best_model']} (F1: {best_f1:.3f})")
        print(f"Second Model: {significance_results['second_model']} (F1: {second_f1:.3f})")
        print(f"Difference: {significance_results['f1_difference']:.3f}")
        print(f"Statistically Significant: {significant_difference}")
        
        return significance_results
        
    def create_performance_visualizations(self, results: List[Dict], output_dir: Path):
        """Create comprehensive performance visualization plots."""
        print("\nCreating performance visualizations...")
        
        # Performance comparison plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: F1 Score Comparison
        plt.subplot(2, 3, 1)
        model_names = [r['model_name'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        expected_f1s = [r['expected_f1'] for r in results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, f1_scores, width, label='Actual F1', alpha=0.8)
        plt.bar(x + width/2, expected_f1s, width, label='Expected F1', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('F1 Score')
        plt.title('F1 Score: Actual vs Expected')
        plt.xticks(x, [name.replace('_', '\n') for name in model_names], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Clinical Metrics Comparison
        plt.subplot(2, 3, 2)
        sensitivity = [r['sensitivity'] for r in results]
        specificity = [r['specificity'] for r in results]
        
        x = np.arange(len(model_names))
        plt.bar(x - width/2, sensitivity, width, label='Sensitivity', alpha=0.8)
        plt.bar(x + width/2, specificity, width, label='Specificity', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Clinical Metrics: Sensitivity vs Specificity')
        plt.xticks(x, [name.replace('_', '\n') for name in model_names], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cost Analysis
        plt.subplot(2, 3, 3)
        costs = [r['cost_per_patient_eur'] for r in results]
        
        plt.bar(model_names, costs, alpha=0.8, color='orange')
        plt.xlabel('Models')
        plt.ylabel('Cost per Patient (EUR)')
        plt.title('Healthcare Cost Impact')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Precision-Recall Trade-off
        plt.subplot(2, 3, 4)
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        
        plt.scatter(recalls, precisions, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            plt.annotate(name.replace('_', '\n'), (recalls[i], precisions[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Trade-off')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Performance Status Overview
        plt.subplot(2, 3, 5)
        status_counts = {}
        for r in results:
            status = r['performance_status'].split(' ')[1]  # Remove emoji
            status_counts[status] = status_counts.get(status, 0) + 1
            
        plt.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
        plt.title('Performance Status Distribution')
        
        # Plot 6: F1 Score Ranking
        plt.subplot(2, 3, 6)
        sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
        sorted_names = [r['model_name'] for r in sorted_results]
        sorted_f1s = [r['f1_score'] for r in sorted_results]
        
        colors = ['gold', 'silver', '#CD7F32', 'lightblue'][:len(sorted_names)]
        plt.barh(sorted_names, sorted_f1s, color=colors, alpha=0.8)
        plt.xlabel('F1 Score')
        plt.title('Model Performance Ranking')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance visualizations saved to {output_dir / 'validation_performance_overview.png'}")
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation framework on all optimized models."""
        print("STARTING COMPREHENSIVE MODEL VALIDATION")
        print("=" * 60)
        
        # Load data and models
        X_test, y_test = self.load_test_data()
        models = self.load_optimized_models()
        
        if not models:
            print("No models loaded successfully. Aborting validation.")
            return None
            
        # Validate each model
        validation_results = []
        for model_name, model_info in models.items():
            result = self.validate_single_model(model_name, model_info, X_test, y_test)
            if result:
                validation_results.append(result)
                
        # Statistical analysis
        significance_results = self.statistical_significance_test(validation_results)
        
        # Create visualizations  
        plots_dir = self.results_path / 'plots'
        self.create_performance_visualizations(validation_results, plots_dir)
        
        # Compile comprehensive results
        comprehensive_results = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'test_dataset_info': {
                'n_samples': len(X_test),
                'n_features': len(X_test.columns),
                'class_distribution': dict(y_test.value_counts().sort_index())
            },
            'individual_results': validation_results,
            'statistical_significance': significance_results,
            'summary': self._create_validation_summary(validation_results, significance_results)
        }
        
        # Save results
        results_file = self.results_path / 'metrics' / 'validation_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            json_safe_results = self._make_json_serializable(comprehensive_results)
            json.dump(json_safe_results, f, indent=2)
            
        print(f"\nSAVED: Validation results saved to {results_file}")
        
        # Print summary
        self._print_validation_summary(comprehensive_results)
        
        return comprehensive_results
        
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle scalar numpy types
            return obj.item()
        else:
            return obj
            
    def _create_validation_summary(self, results: List[Dict], significance: Dict) -> Dict[str, Any]:
        """Create executive summary of validation results."""
        if not results:
            return {
                'best_model': 'None',
                'best_f1_score': 0.0,
                'models_exceeding_expectations': 0,
                'models_meeting_expectations': 0,
                'models_below_expectations': 0,
                'avg_f1_score': 0.0,
                'avg_sensitivity': 0.0,
                'avg_specificity': 0.0,
                'lowest_cost_model': 'None',
                'statistically_significant_difference': False
            }
            
        sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
        
        return {
            'best_model': sorted_results[0]['model_name'],
            'best_f1_score': sorted_results[0]['f1_score'],
            'models_exceeding_expectations': len([r for r in results if 'EXCEEDS' in r['performance_status']]),
            'models_meeting_expectations': len([r for r in results if 'MEETS' in r['performance_status']]),
            'models_below_expectations': len([r for r in results if 'BELOW' in r['performance_status']]),
            'avg_f1_score': np.mean([r['f1_score'] for r in results]),
            'avg_sensitivity': np.mean([r['sensitivity'] for r in results]),
            'avg_specificity': np.mean([r['specificity'] for r in results]),
            'lowest_cost_model': min(results, key=lambda x: x['cost_per_patient_eur'])['model_name'],
            'statistically_significant_difference': significance.get('statistically_significant', False)
        }
        
    def _print_validation_summary(self, results: Dict[str, Any]):
        """Print executive summary of validation results."""
        summary = results['summary']
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY REPORT")
        print("="*60)
        
        print(f"\nBEST PERFORMING MODEL: {summary['best_model']}")
        print(f"   F1 Score: {summary['best_f1_score']:.3f}")
        
        print(f"\nPERFORMANCE DISTRIBUTION:")
        print(f"   Exceeding Expectations: {summary['models_exceeding_expectations']} models")
        print(f"   Meeting Expectations: {summary['models_meeting_expectations']} models") 
        print(f"   Below Expectations: {summary['models_below_expectations']} models")
        
        print(f"\nAVERAGE PERFORMANCE:")
        print(f"   F1 Score: {summary['avg_f1_score']:.3f}")
        print(f"   Sensitivity: {summary['avg_sensitivity']:.3f}")
        print(f"   Specificity: {summary['avg_specificity']:.3f}")
        
        print(f"\nCOST EFFECTIVE: MOST COST-EFFECTIVE: {summary['lowest_cost_model']}")
        
        print(f"\nSTATISTICAL SIGNIFICANCE: {summary['statistically_significant_difference']}")
        
        print("\nVALIDATION COMPLETE - Models ready for production consideration!")


def main():
    """Main execution function for validation framework."""
    # Initialize validation framework
    validator = ValidationFramework()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    if results:
        print(f"\nValidation completed successfully!")
        print(f"Results saved to: {validator.results_path / 'metrics' / 'validation_results.json'}")
        print(f"Plots saved to: {validator.results_path / 'plots' / 'validation_performance_overview.png'}")
    else:
        print("Validation failed. Check model files and data availability.")


if __name__ == "__main__":
    main()

# Run: python src/tuning/validation_framework.py
