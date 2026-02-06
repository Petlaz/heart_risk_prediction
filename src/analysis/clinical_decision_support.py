"""
Clinical Decision Support Analysis
Comprehensive clinical utility assessment for heart risk prediction models.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


class ClinicalDecisionSupport:
    """Clinical decision support analysis for heart risk prediction models."""
    
    def __init__(self, project_root: str = None):
        """Initialize clinical decision support analysis."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        self.data_path = self.project_root / 'data' / 'processed'
        self.models_path = self.project_root / 'results' / 'models'
        self.results_path = self.project_root / 'results' / 'explainability' / 'clinical'
        
        # Create clinical results directory
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Clinical thresholds and costs
        self.clinical_params = {
            'sensitivity_threshold': 0.80,  # Minimum acceptable sensitivity
            'specificity_threshold': 0.60,  # Minimum acceptable specificity
            'cost_false_negative': 1000,    # Cost of missed heart risk case
            'cost_false_positive': 100,     # Cost of unnecessary referral
            'cost_screening': 50,           # Cost of screening test
            'prevalence': 0.25              # Estimated heart disease prevalence
        }
        
    def load_clinical_test_data(self):
        """Load test data and preprocessing artifacts for clinical evaluation."""
        print("Loading clinical test data...")
        
        # Load test data
        test_df = pd.read_csv(self.data_path / 'test.csv')
        if 'hltprhc' in test_df.columns:
            X_test = test_df.drop('hltprhc', axis=1)
            y_test = test_df['hltprhc']
        else:
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]
            
        # Load preprocessing artifacts
        preprocessing = joblib.load(self.data_path / 'preprocessing_artifacts.joblib')
        scaler = preprocessing['scaler']
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   Test cohort size: {len(X_test)} patients")
        print(f"   Positive cases: {y_test.sum()} ({y_test.mean():.2%})")
        
        return X_test, y_test, X_test_scaled, scaler
        
    def evaluate_clinical_model_performance(self, model_path, X_test_scaled, y_test):
        """Evaluate single model for clinical decision support."""
        try:
            # Load model
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
            else:
                model = model_data
                
            # Generate predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate clinical metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Clinical performance metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Clinical utility scores
            clinical_utility = self._calculate_clinical_utility(tp, fp, tn, fn)
            
            return {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_proba,
                'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
                'clinical_metrics': {
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'ppv': ppv,
                    'npv': npv
                },
                'clinical_utility': clinical_utility,
                'meets_clinical_threshold': (
                    sensitivity >= self.clinical_params['sensitivity_threshold'] and 
                    specificity >= self.clinical_params['specificity_threshold']
                )
            }
            
        except Exception as e:
            print(f"Failed to evaluate model {model_path}: {e}")
            return None
            
    def _calculate_clinical_utility(self, tp, fp, tn, fn):
        """Calculate clinical utility metrics."""
        total_patients = tp + fp + tn + fn
        prevalence = self.clinical_params['prevalence']
        
        # Cost calculations
        cost_fn = self.clinical_params['cost_false_negative']
        cost_fp = self.clinical_params['cost_false_positive']
        cost_screen = self.clinical_params['cost_screening']
        
        # Total costs
        total_cost = (fn * cost_fn) + (fp * cost_fp) + (total_patients * cost_screen)
        cost_per_patient = total_cost / total_patients
        
        # Net benefit calculation
        threshold_probability = cost_fp / (cost_fn + cost_fp)
        net_benefit = (tp / total_patients) - (fp / total_patients) * (threshold_probability / (1 - threshold_probability))
        
        # Lives saved calculation
        lives_saved_per_1000 = (tp / total_patients) * 1000
        
        return {
            'total_cost': total_cost,
            'cost_per_patient': cost_per_patient,
            'net_benefit': net_benefit,
            'threshold_probability': threshold_probability,
            'lives_saved_per_1000': lives_saved_per_1000,
            'missed_cases_per_1000': (fn / total_patients) * 1000,
            'unnecessary_referrals_per_1000': (fp / total_patients) * 1000
        }
        
    def analyze_threshold_optimization(self, y_test, y_proba):
        """Analyze optimal clinical decision thresholds."""
        print("Analyzing threshold optimization for clinical use...")
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        threshold_results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
            
            # Clinical metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Clinical utility
            clinical_utility = self._calculate_clinical_utility(tp, fp, tn, fn)
            
            threshold_results.append({
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
                'net_benefit': clinical_utility['net_benefit'],
                'cost_per_patient': clinical_utility['cost_per_patient'],
                'lives_saved_per_1000': clinical_utility['lives_saved_per_1000'],
                'meets_clinical_criteria': (
                    sensitivity >= self.clinical_params['sensitivity_threshold'] and 
                    specificity >= self.clinical_params['specificity_threshold']
                )
            })
            
        return threshold_results
        
    def create_clinical_risk_stratification(self, y_test, y_proba):
        """Create clinical risk stratification analysis."""
        print("Creating clinical risk stratification...")
        
        # Risk categories
        risk_categories = {
            'Low Risk': (0.0, 0.3),
            'Medium Risk': (0.3, 0.7), 
            'High Risk': (0.7, 1.0)
        }
        
        stratification_results = {}
        
        for category, (low_thresh, high_thresh) in risk_categories.items():
            mask = (y_proba >= low_thresh) & (y_proba < high_thresh)
            category_y_test = y_test[mask]
            category_proba = y_proba[mask]
            
            if len(category_y_test) > 0:
                actual_risk = category_y_test.mean()
                predicted_risk = category_proba.mean()
                patient_count = len(category_y_test)
                positive_cases = category_y_test.sum()
                
                stratification_results[category] = {
                    'patient_count': int(patient_count),
                    'positive_cases': int(positive_cases),
                    'actual_risk': actual_risk,
                    'predicted_risk': predicted_risk,
                    'risk_range': f"{low_thresh:.1f}-{high_thresh:.1f}",
                    'calibration': abs(actual_risk - predicted_risk)
                }
            else:
                stratification_results[category] = {
                    'patient_count': 0,
                    'positive_cases': 0,
                    'actual_risk': 0,
                    'predicted_risk': 0,
                    'risk_range': f"{low_thresh:.1f}-{high_thresh:.1f}",
                    'calibration': 0
                }
                
        return stratification_results
        
    def create_clinical_decision_support_report(self, model_name, evaluation_results, threshold_analysis, risk_stratification):
        """Create comprehensive clinical decision support report."""
        print(f"Creating clinical decision support report for {model_name}...")
        
        # Extract key metrics
        clinical_metrics = evaluation_results['clinical_metrics']
        clinical_utility = evaluation_results['clinical_utility']
        confusion_matrix_data = evaluation_results['confusion_matrix']
        
        # Find optimal threshold
        optimal_threshold = max(threshold_analysis, key=lambda x: x['net_benefit'])
        clinical_threshold = next((t for t in threshold_analysis if t['meets_clinical_criteria']), None)
        
        # Create report content
        report = {
            'model_name': model_name,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'clinical_performance': {
                'default_threshold': {
                    'sensitivity': clinical_metrics['sensitivity'],
                    'specificity': clinical_metrics['specificity'],
                    'ppv': clinical_metrics['ppv'],
                    'npv': clinical_metrics['npv'],
                    'meets_criteria': evaluation_results['meets_clinical_threshold']
                },
                'optimal_threshold': optimal_threshold,
                'clinical_acceptable_threshold': clinical_threshold
            },
            'clinical_utility_analysis': clinical_utility,
            'risk_stratification': risk_stratification,
            'threshold_analysis_summary': {
                'total_thresholds_tested': len(threshold_analysis),
                'clinically_acceptable_thresholds': sum(1 for t in threshold_analysis if t['meets_clinical_criteria']),
                'max_net_benefit': max(t['net_benefit'] for t in threshold_analysis),
                'min_cost_per_patient': min(t['cost_per_patient'] for t in threshold_analysis)
            },
            'clinical_recommendations': self._create_clinical_recommendations(
                evaluation_results, optimal_threshold, clinical_threshold
            ),
            'confusion_matrix': confusion_matrix_data
        }
        
        return report
        
    def _create_clinical_recommendations(self, evaluation_results, optimal_threshold, clinical_threshold):
        """Create clinical recommendations based on model evaluation."""
        recommendations = []
        
        # Sensitivity check
        sensitivity = evaluation_results['clinical_metrics']['sensitivity']
        if sensitivity < 0.6:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Safety',
                'recommendation': f'Model sensitivity ({sensitivity:.2%}) is below acceptable clinical threshold. High risk of missed diagnoses.',
                'action': 'Consider alternative models or adjust threshold to increase sensitivity.'
            })
        elif sensitivity < 0.8:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Performance',
                'recommendation': f'Model sensitivity ({sensitivity:.2%}) is moderate. Consider threshold optimization.',
                'action': 'Evaluate threshold adjustment to improve sensitivity while maintaining specificity.'
            })
            
        # Specificity check
        specificity = evaluation_results['clinical_metrics']['specificity']
        if specificity < 0.5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Cost',
                'recommendation': f'Model specificity ({specificity:.2%}) is low, leading to excessive false positives.',
                'action': 'High healthcare costs due to unnecessary referrals. Consider model improvement.'
            })
            
        # Clinical utility check
        net_benefit = evaluation_results['clinical_utility']['net_benefit']
        if net_benefit < 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Utility',
                'recommendation': 'Model shows negative net clinical benefit. May cause more harm than benefit.',
                'action': 'Do not deploy. Requires significant model improvement or alternative approach.'
            })
        elif net_benefit < 0.05:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Utility',
                'recommendation': f'Model net benefit ({net_benefit:.3f}) is marginal.',
                'action': 'Consider if modest benefit justifies implementation costs and risks.'
            })
            
        # Cost analysis
        cost_per_patient = evaluation_results['clinical_utility']['cost_per_patient']
        if cost_per_patient > 200:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Economics',
                'recommendation': f'High cost per patient (€{cost_per_patient:.0f}) may limit cost-effectiveness.',
                'action': 'Conduct formal health economic evaluation before implementation.'
            })
            
        # Threshold optimization
        if clinical_threshold:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Optimization',
                'recommendation': f'Clinical threshold of {clinical_threshold["threshold"]:.2f} available.',
                'action': f'Consider using threshold {clinical_threshold["threshold"]:.2f} for clinical deployment.'
            })
        else:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Threshold',
                'recommendation': 'No threshold meets minimum clinical criteria.',
                'action': 'Model requires improvement before clinical deployment consideration.'
            })
            
        return recommendations
        
    def create_clinical_visualizations(self, model_name, threshold_analysis, risk_stratification, evaluation_results):
        """Create clinical decision support visualizations."""
        print("Creating clinical decision support visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Sensitivity vs Specificity Trade-off
        ax1 = axes[0, 0]
        thresholds = [t['threshold'] for t in threshold_analysis]
        sensitivities = [t['sensitivity'] for t in threshold_analysis]
        specificities = [t['specificity'] for t in threshold_analysis]
        
        ax1.plot(thresholds, sensitivities, 'o-', label='Sensitivity', linewidth=2, markersize=4)
        ax1.plot(thresholds, specificities, 's-', label='Specificity', linewidth=2, markersize=4)
        ax1.axhline(y=self.clinical_params['sensitivity_threshold'], color='red', linestyle='--', alpha=0.7, label='Min Sensitivity')
        ax1.axhline(y=self.clinical_params['specificity_threshold'], color='orange', linestyle='--', alpha=0.7, label='Min Specificity')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Performance')
        ax1.set_title('Clinical Performance vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Net Benefit Analysis
        ax2 = axes[0, 1]
        net_benefits = [t['net_benefit'] for t in threshold_analysis]
        
        ax2.plot(thresholds, net_benefits, 'o-', linewidth=2, markersize=4, color='green')
        ax2.axhline(y=0, color='red', linestyle='-', alpha=0.7, label='No Benefit')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Net Benefit')
        ax2.set_title('Net Clinical Benefit by Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Color positive net benefit points
        for i, (thresh, nb) in enumerate(zip(thresholds, net_benefits)):
            if nb > 0:
                ax2.scatter(thresh, nb, color='green', s=30, zorder=5)
                
        # Plot 3: Cost per Patient
        ax3 = axes[0, 2]
        costs = [t['cost_per_patient'] for t in threshold_analysis]
        
        ax3.plot(thresholds, costs, 'o-', linewidth=2, markersize=4, color='red')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Cost per Patient (EUR)')
        ax3.set_title('Cost Analysis by Threshold')
        ax3.grid(True, alpha=0.3)
        
        # Highlight minimum cost
        min_cost_idx = costs.index(min(costs))
        ax3.scatter(thresholds[min_cost_idx], costs[min_cost_idx], color='green', s=100, zorder=5, label='Minimum Cost')
        ax3.legend()
        
        # Plot 4: Risk Stratification
        ax4 = axes[1, 0]
        categories = list(risk_stratification.keys())
        patient_counts = [risk_stratification[cat]['patient_count'] for cat in categories]
        positive_cases = [risk_stratification[cat]['positive_cases'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x - width/2, patient_counts, width, label='Total Patients', alpha=0.7)
        ax4.bar(x + width/2, positive_cases, width, label='Positive Cases', alpha=0.7)
        ax4.set_xlabel('Risk Category')
        ax4.set_ylabel('Patient Count')
        ax4.set_title('Risk Stratification Distribution')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        
        # Add risk percentages
        for i, cat in enumerate(categories):
            if patient_counts[i] > 0:
                risk_pct = (positive_cases[i] / patient_counts[i]) * 100
                ax4.text(i, patient_counts[i] + max(patient_counts) * 0.05, 
                        f'{risk_pct:.1f}%', ha='center', va='bottom')
        
        # Plot 5: Clinical Impact per 1000 Patients
        ax5 = axes[1, 1]
        lives_saved = [t['lives_saved_per_1000'] for t in threshold_analysis]
        
        ax5.plot(thresholds, lives_saved, 'o-', linewidth=2, markersize=4, color='purple')
        ax5.set_xlabel('Threshold')
        ax5.set_ylabel('Lives Saved per 1000 Patients')
        ax5.set_title('Clinical Impact Assessment')
        ax5.grid(True, alpha=0.3)
        
        # Highlight maximum lives saved
        max_lives_idx = lives_saved.index(max(lives_saved))
        ax5.scatter(thresholds[max_lives_idx], lives_saved[max_lives_idx], color='red', s=100, zorder=5, label='Maximum Impact')
        ax5.legend()
        
        # Plot 6: Confusion Matrix Visualization
        ax6 = axes[1, 2]
        cm_data = evaluation_results['confusion_matrix']
        cm_matrix = np.array([[cm_data['tn'], cm_data['fp']], 
                             [cm_data['fn'], cm_data['tp']]])
        
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax6,
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        ax6.set_title('Confusion Matrix (Default Threshold)')
        
        plt.suptitle(f'Clinical Decision Support Analysis: {model_name}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save visualization
        output_file = self.results_path / f'{model_name}_clinical_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Clinical visualizations saved to {output_file}")
        return output_file
        
    def run_comprehensive_clinical_analysis(self):
        """Run comprehensive clinical decision support analysis."""
        print("STARTING COMPREHENSIVE CLINICAL DECISION SUPPORT ANALYSIS")
        print("=" * 80)
        
        # Load clinical test data
        X_test, y_test, X_test_scaled, scaler = self.load_clinical_test_data()
        
        # Model configurations (based on post-optimization results)
        clinical_models = {
            'Adaptive_Ensemble': {
                'path': self.models_path / 'adaptive_tuning' / 'Adaptive_Ensemble_complexity_optimized_20260108_233028.joblib',
                'description': 'Best performing ensemble model',
                'expected_performance': 'Marginal but potentially viable'
            }
        }
        
        comprehensive_results = {}
        
        for model_name, model_config in clinical_models.items():
            if model_config['path'].exists():
                print(f"\nAnalyzing {model_name} for clinical deployment...")
                
                # Evaluate model
                evaluation = self.evaluate_clinical_model_performance(
                    model_config['path'], X_test_scaled, y_test
                )
                
                if evaluation:
                    # Threshold optimization
                    threshold_analysis = self.analyze_threshold_optimization(
                        y_test, evaluation['probabilities']
                    )
                    
                    # Risk stratification
                    risk_stratification = self.create_clinical_risk_stratification(
                        y_test, evaluation['probabilities']
                    )
                    
                    # Clinical report
                    clinical_report = self.create_clinical_decision_support_report(
                        model_name, evaluation, threshold_analysis, risk_stratification
                    )
                    
                    # Visualizations
                    viz_file = self.create_clinical_visualizations(
                        model_name, threshold_analysis, risk_stratification, evaluation
                    )
                    
                    comprehensive_results[model_name] = {
                        'clinical_report': clinical_report,
                        'threshold_analysis': threshold_analysis,
                        'risk_stratification': risk_stratification,
                        'visualization_file': str(viz_file)
                    }
                    
                    # Print summary
                    self._print_clinical_summary(model_name, clinical_report)
                    
        # Save comprehensive results
        results_file = self.results_path / 'comprehensive_clinical_analysis.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
            
        print(f"\nComprehensive clinical analysis saved to {results_file}")
        
        return comprehensive_results
        
    def _print_clinical_summary(self, model_name, clinical_report):
        """Print clinical summary for a model."""
        print(f"\nCLINICAL SUMMARY - {model_name}")
        print("-" * 50)
        
        default_perf = clinical_report['clinical_performance']['default_threshold']
        utility = clinical_report['clinical_utility_analysis']
        
        print(f"Clinical Performance:")
        print(f"   Sensitivity: {default_perf['sensitivity']:.2%} (catches {default_perf['sensitivity']*100:.1f}% of cases)")
        print(f"   Specificity: {default_perf['specificity']:.2%} (avoids {default_perf['specificity']*100:.1f}% false alarms)")
        print(f"   PPV: {default_perf['ppv']:.2%}")
        print(f"   NPV: {default_perf['npv']:.2%}")
        
        print(f"\nClinical Utility:")
        print(f"   Cost per patient: €{utility['cost_per_patient']:.2f}")
        print(f"   Net benefit: {utility['net_benefit']:.4f}")
        print(f"   Lives saved per 1000: {utility['lives_saved_per_1000']:.1f}")
        print(f"   Missed cases per 1000: {utility['missed_cases_per_1000']:.1f}")
        
        print(f"\nClinical Criteria:")
        criteria_met = "MEETS" if default_perf['meets_criteria'] else "FAILS"
        print(f"   {criteria_met} clinical deployment criteria")
        
        recommendations = clinical_report['clinical_recommendations']
        high_priority = [r for r in recommendations if r['priority'] in ['CRITICAL', 'HIGH']]
        
        if high_priority:
            print(f"\nCritical Recommendations:")
            for rec in high_priority[:3]:  # Top 3 priority recommendations
                print(f"   {rec['priority']}: {rec['recommendation']}")


def main():
    """Main execution function for clinical analysis."""
    # Initialize clinical decision support framework
    clinical_analyzer = ClinicalDecisionSupport()
    
    # Run comprehensive clinical analysis
    results = clinical_analyzer.run_comprehensive_clinical_analysis()
    
    if results:
        print(f"\nClinical decision support analysis completed successfully!")
        print(f"Results directory: {clinical_analyzer.results_path}")
    else:
        print("Clinical analysis failed. Check model files and data availability.")


if __name__ == "__main__":
    main()

# Usage: python src/analysis/clinical_decision_support.py