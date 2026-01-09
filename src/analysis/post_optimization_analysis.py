"""
Post-Optimization Error Analysis Framework
Comprehensive error analysis for optimized heart risk prediction models.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PostOptimizationAnalysis:
    """Advanced error analysis for post-optimization model evaluation."""
    
    def __init__(self, project_root: str = None):
        """Initialize error analysis framework."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        self.data_path = self.project_root / 'data' / 'processed'
        self.models_path = self.project_root / 'results' / 'models'
        self.results_path = self.project_root / 'results'
        
        # Ensure results directories exist
        (self.results_path / 'explanations').mkdir(exist_ok=True)
        (self.results_path / 'plots').mkdir(exist_ok=True)
        
        # Model configurations (updated based on test set validation)
        self.models_config = {
            'Adaptive_Ensemble': {
                'path': 'adaptive_tuning/Adaptive_Ensemble_complexity_optimized_20260108_233028.joblib',
                'test_f1': 0.175,
                'status': 'winner',
                'description': 'Best performing ensemble on test set'
            },
            'Optimal_Hybrid': {
                'path': 'adaptive_tuning/Optimal_Hybrid_optimal_hybrid_20260108_233028.joblib',
                'test_f1': 0.091,
                'status': 'poor_generalization',
                'description': 'Large generalization gap'
            },
            'Adaptive_LR': {
                'path': 'adaptive_tuning/Adaptive_LR_complexity_increased_20260108_233028.joblib',
                'test_f1': 0.032,
                'status': 'failed',
                'description': 'Severe overfitting to validation set'
            }
        }
        
    def load_data_and_models(self):
        """Load test data and available models for analysis."""
        print("📂 Loading data and models for error analysis...")
        
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
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Load models and generate predictions
        models_data = {}
        for model_name, config in self.models_config.items():
            model_path = self.models_path / config['path']
            
            if model_path.exists():
                try:
                    model_data = joblib.load(model_path)
                    if isinstance(model_data, dict) and 'model' in model_data:
                        model = model_data['model']
                    else:
                        model = model_data
                        
                    # Generate predictions
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    models_data[model_name] = {
                        'model': model,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba,
                        'config': config,
                        'metadata': model_data if isinstance(model_data, dict) else None
                    }
                    print(f"✅ Loaded {model_name}")
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
                    
        return X_test, y_test, X_test_scaled_df, models_data
        
    def analyze_misclassification_patterns(self, X_test, y_test, models_data):
        """Analyze misclassification patterns across models."""
        print("\n🔍 Analyzing misclassification patterns...")
        
        analysis_results = {}
        feature_names = X_test.columns.tolist()
        
        for model_name, model_info in models_data.items():
            print(f"\n📊 Analyzing {model_name}...")
            
            y_pred = model_info['predictions']
            y_proba = model_info['probabilities']
            
            # Identify misclassified samples
            misclassified_mask = (y_test != y_pred)
            false_positives = (y_test == 0) & (y_pred == 1)
            false_negatives = (y_test == 1) & (y_pred == 0)
            
            # Feature analysis for misclassified samples
            misclassified_features = X_test[misclassified_mask]
            correct_features = X_test[~misclassified_mask]
            
            # Calculate feature differences
            feature_differences = {}
            for feature in feature_names:
                if len(misclassified_features) > 0 and len(correct_features) > 0:
                    miscl_mean = misclassified_features[feature].mean()
                    correct_mean = correct_features[feature].mean()
                    difference = abs(miscl_mean - correct_mean)
                    feature_differences[feature] = {
                        'misclassified_mean': miscl_mean,
                        'correct_mean': correct_mean,
                        'abs_difference': difference
                    }
            
            # Sort features by impact on misclassification
            sorted_features = sorted(feature_differences.items(), 
                                   key=lambda x: x[1]['abs_difference'], 
                                   reverse=True)
            
            # Analyze by prediction confidence
            low_confidence_mask = (y_proba > 0.3) & (y_proba < 0.7)
            high_confidence_errors = misclassified_mask & (~low_confidence_mask)
            
            analysis_results[model_name] = {
                'total_samples': len(y_test),
                'total_misclassified': misclassified_mask.sum(),
                'false_positives': false_positives.sum(),
                'false_negatives': false_negatives.sum(),
                'misclassification_rate': misclassified_mask.mean(),
                'high_confidence_errors': high_confidence_errors.sum(),
                'low_confidence_predictions': low_confidence_mask.sum(),
                'feature_impact': dict(sorted_features[:10]),  # Top 10 features
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'test_f1': model_info['config']['test_f1'],
                'status': model_info['config']['status']
            }
            
            print(f"   Total misclassified: {misclassified_mask.sum()}/{len(y_test)} ({misclassified_mask.mean():.3f})")
            print(f"   False positives: {false_positives.sum()}")
            print(f"   False negatives: {false_negatives.sum()}")
            print(f"   High confidence errors: {high_confidence_errors.sum()}")
            
        return analysis_results
        
    def feature_based_error_correlation(self, X_test, y_test, models_data):
        """Analyze feature correlation with prediction errors."""
        print("\n📈 Analyzing feature-based error correlations...")
        
        feature_error_analysis = {}
        
        for model_name, model_info in models_data.items():
            y_pred = model_info['predictions']
            
            # Create error indicators
            error_indicators = (y_test != y_pred).astype(int)
            
            # Calculate correlation between features and errors
            correlations = {}
            for feature in X_test.columns:
                correlation = np.corrcoef(X_test[feature], error_indicators)[0, 1]
                correlations[feature] = correlation
                
            # Sort by absolute correlation
            sorted_correlations = sorted(correlations.items(), 
                                       key=lambda x: abs(x[1]), 
                                       reverse=True)
            
            feature_error_analysis[model_name] = {
                'feature_error_correlations': dict(sorted_correlations),
                'top_error_features': sorted_correlations[:5],
                'model_status': model_info['config']['status']
            }
            
            print(f"   {model_name} - Top error-correlated features:")
            for feature, corr in sorted_correlations[:3]:
                print(f"     {feature}: {corr:.3f}")
                
        return feature_error_analysis
        
    def cross_model_error_comparison(self, y_test, models_data):
        """Compare error patterns across different models."""
        print("\n🔄 Analyzing cross-model error patterns...")
        
        model_names = list(models_data.keys())
        cross_model_analysis = {
            'model_agreement': {},
            'unique_errors': {},
            'consensus_patterns': {}
        }
        
        # Model agreement analysis
        predictions_df = pd.DataFrame({
            name: info['predictions'] 
            for name, info in models_data.items()
        })
        
        # Calculate agreement between models
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                agreement = (predictions_df[model1] == predictions_df[model2]).mean()
                cross_model_analysis['model_agreement'][f"{model1}_vs_{model2}"] = agreement
                
        # Find samples where models disagree
        disagreement_patterns = {}
        for idx in predictions_df.index:
            row_predictions = predictions_df.loc[idx]
            unique_predictions = len(row_predictions.unique())
            
            if unique_predictions > 1:  # Models disagree
                pattern = tuple(row_predictions.values)
                if pattern not in disagreement_patterns:
                    disagreement_patterns[pattern] = []
                disagreement_patterns[pattern].append(idx)
                
        # Analyze unique errors per model
        for model_name, model_info in models_data.items():
            y_pred = model_info['predictions']
            model_errors = set(y_test.index[y_test != y_pred])
            
            # Find errors unique to this model
            other_models_errors = set()
            for other_name, other_info in models_data.items():
                if other_name != model_name:
                    other_errors = set(y_test.index[y_test != other_info['predictions']])
                    other_models_errors.update(other_errors)
                    
            unique_errors = model_errors - other_models_errors
            shared_errors = model_errors & other_models_errors
            
            cross_model_analysis['unique_errors'][model_name] = {
                'unique_error_count': len(unique_errors),
                'shared_error_count': len(shared_errors),
                'total_errors': len(model_errors),
                'unique_error_rate': len(unique_errors) / len(model_errors) if model_errors else 0
            }
            
        cross_model_analysis['disagreement_patterns'] = {
            str(pattern): len(indices) 
            for pattern, indices in disagreement_patterns.items()
        }
        
        return cross_model_analysis
        
    def clinical_risk_assessment(self, y_test, models_data):
        """Assess clinical risk implications of model errors."""
        print("\n🏥 Conducting clinical risk assessment...")
        
        clinical_analysis = {}
        
        for model_name, model_info in models_data.items():
            y_pred = model_info['predictions']
            y_proba = model_info['probabilities']
            
            # Confusion matrix elements
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Clinical metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Risk analysis
            false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
            false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
            
            # Cost analysis (€100 per false positive, €1000 per false negative)
            cost_per_patient = (fp * 100 + fn * 1000) / len(y_test)
            
            # Threshold analysis for potential improvement
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            threshold_analysis = {}
            
            for threshold in thresholds:
                thresh_pred = (y_proba >= threshold).astype(int)
                thresh_tn, thresh_fp, thresh_fn, thresh_tp = confusion_matrix(y_test, thresh_pred).ravel()
                thresh_sensitivity = thresh_tp / (thresh_tp + thresh_fn) if (thresh_tp + thresh_fn) > 0 else 0
                thresh_specificity = thresh_tn / (thresh_tn + thresh_fp) if (thresh_tn + thresh_fp) > 0 else 0
                
                threshold_analysis[threshold] = {
                    'sensitivity': thresh_sensitivity,
                    'specificity': thresh_specificity,
                    'false_negatives': int(thresh_fn),
                    'false_positives': int(thresh_fp)
                }
                
            clinical_analysis[model_name] = {
                'current_performance': {
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'ppv': ppv,
                    'npv': npv,
                    'false_negative_rate': false_negative_rate,
                    'false_positive_rate': false_positive_rate
                },
                'clinical_impact': {
                    'missed_cases': int(fn),
                    'unnecessary_referrals': int(fp),
                    'cost_per_patient_eur': cost_per_patient,
                    'total_cost_eur': cost_per_patient * len(y_test)
                },
                'threshold_analysis': threshold_analysis,
                'model_status': model_info['config']['status'],
                'test_f1': model_info['config']['test_f1']
            }
            
            print(f"   {model_name}:")
            print(f"     Sensitivity: {sensitivity:.3f} (catches {sensitivity*100:.1f}% of cases)")
            print(f"     Specificity: {specificity:.3f} (avoids {specificity*100:.1f}% false alarms)")
            print(f"     Missed cases: {fn} ({false_negative_rate*100:.1f}%)")
            print(f"     Cost per patient: €{cost_per_patient:.2f}")
            
        return clinical_analysis
        
    def create_error_visualizations(self, analysis_results, feature_analysis, cross_model_analysis, clinical_analysis):
        """Create comprehensive error analysis visualizations."""
        print("\n📊 Creating error analysis visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Plot 1: Misclassification rates by model
        ax1 = axes[0, 0]
        model_names = list(analysis_results.keys())
        miscl_rates = [analysis_results[name]['misclassification_rate'] for name in model_names]
        test_f1s = [analysis_results[name]['test_f1'] for name in model_names]
        
        colors = ['green' if rate < 0.2 else 'orange' if rate < 0.3 else 'red' for rate in miscl_rates]
        bars = ax1.bar(model_names, miscl_rates, color=colors, alpha=0.7)
        ax1.set_title('Misclassification Rates by Model')
        ax1.set_ylabel('Misclassification Rate')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add F1 scores as text on bars
        for bar, f1 in zip(bars, test_f1s):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'F1: {f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: False Positives vs False Negatives
        ax2 = axes[0, 1]
        fp_counts = [analysis_results[name]['false_positives'] for name in model_names]
        fn_counts = [analysis_results[name]['false_negatives'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        ax2.bar(x - width/2, fp_counts, width, label='False Positives', alpha=0.7)
        ax2.bar(x + width/2, fn_counts, width, label='False Negatives', alpha=0.7)
        ax2.set_title('False Positives vs False Negatives')
        ax2.set_ylabel('Count')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        
        # Plot 3: Feature error correlation heatmap (for best model)
        ax3 = axes[0, 2]
        best_model = min(model_names, key=lambda x: analysis_results[x]['misclassification_rate'])
        best_model_features = feature_analysis[best_model]['top_error_features']
        
        features = [item[0] for item in best_model_features]
        correlations = [item[1] for item in best_model_features]
        
        y_pos = np.arange(len(features))
        bars = ax3.barh(y_pos, correlations)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(features)
        ax3.set_xlabel('Error Correlation')
        ax3.set_title(f'Top Error Features ({best_model})')
        
        # Color bars by correlation magnitude
        for bar, corr in zip(bars, correlations):
            bar.set_color('red' if abs(corr) > 0.1 else 'orange' if abs(corr) > 0.05 else 'lightblue')
        
        # Plot 4: Clinical Impact - Sensitivity vs Specificity
        ax4 = axes[1, 0]
        sensitivities = [clinical_analysis[name]['current_performance']['sensitivity'] for name in model_names]
        specificities = [clinical_analysis[name]['current_performance']['specificity'] for name in model_names]
        
        scatter = ax4.scatter(sensitivities, specificities, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax4.annotate(name, (sensitivities[i], specificities[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Sensitivity (Recall)')
        ax4.set_ylabel('Specificity')
        ax4.set_title('Sensitivity vs Specificity Trade-off')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Cost Analysis
        ax5 = axes[1, 1]
        costs = [clinical_analysis[name]['clinical_impact']['cost_per_patient_eur'] for name in model_names]
        
        bars = ax5.bar(model_names, costs, alpha=0.7)
        ax5.set_title('Cost per Patient (EUR)')
        ax5.set_ylabel('Cost (EUR)')
        ax5.tick_params(axis='x', rotation=45)
        
        # Color by cost level
        for bar, cost in zip(bars, costs):
            bar.set_color('green' if cost < 100 else 'orange' if cost < 200 else 'red')
        
        # Plot 6: Model Agreement Analysis
        ax6 = axes[1, 2]
        agreement_data = cross_model_analysis['model_agreement']
        if agreement_data:
            comparisons = list(agreement_data.keys())
            agreements = list(agreement_data.values())
            
            y_pos = np.arange(len(comparisons))
            bars = ax6.barh(y_pos, agreements)
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels([comp.replace('_vs_', ' vs ') for comp in comparisons])
            ax6.set_xlabel('Agreement Rate')
            ax6.set_title('Cross-Model Agreement')
            
            # Color by agreement level
            for bar, agreement in zip(bars, agreements):
                bar.set_color('green' if agreement > 0.8 else 'orange' if agreement > 0.6 else 'red')
        
        # Plot 7: Unique vs Shared Errors
        ax7 = axes[2, 0]
        unique_errors = [cross_model_analysis['unique_errors'][name]['unique_error_count'] for name in model_names]
        shared_errors = [cross_model_analysis['unique_errors'][name]['shared_error_count'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        ax7.bar(x - width/2, unique_errors, width, label='Unique Errors', alpha=0.7)
        ax7.bar(x + width/2, shared_errors, width, label='Shared Errors', alpha=0.7)
        ax7.set_title('Unique vs Shared Errors')
        ax7.set_ylabel('Error Count')
        ax7.set_xticks(x)
        ax7.set_xticklabels(model_names, rotation=45)
        ax7.legend()
        
        # Plot 8: Threshold Analysis for Best Model
        ax8 = axes[2, 1]
        best_clinical = clinical_analysis[best_model]['threshold_analysis']
        thresholds = list(best_clinical.keys())
        sens_values = [best_clinical[t]['sensitivity'] for t in thresholds]
        spec_values = [best_clinical[t]['specificity'] for t in thresholds]
        
        ax8.plot(thresholds, sens_values, 'o-', label='Sensitivity', linewidth=2)
        ax8.plot(thresholds, spec_values, 's-', label='Specificity', linewidth=2)
        ax8.set_xlabel('Threshold')
        ax8.set_ylabel('Performance')
        ax8.set_title(f'Threshold Analysis ({best_model})')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Performance Summary
        ax9 = axes[2, 2]
        performance_data = {
            'Test F1': [analysis_results[name]['test_f1'] for name in model_names],
            'Sensitivity': [clinical_analysis[name]['current_performance']['sensitivity'] for name in model_names],
            'Specificity': [clinical_analysis[name]['current_performance']['specificity'] for name in model_names]
        }
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, (metric, values) in enumerate(performance_data.items()):
            ax9.bar(x + i*width, values, width, label=metric, alpha=0.7)
        
        ax9.set_title('Performance Summary')
        ax9.set_ylabel('Score')
        ax9.set_xticks(x + width)
        ax9.set_xticklabels(model_names, rotation=45)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the visualization
        output_file = self.results_path / 'plots' / 'error_analysis_comprehensive.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Error analysis visualizations saved to {output_file}")
        
    def run_comprehensive_error_analysis(self):
        """Run complete post-optimization error analysis."""
        print("🚀 STARTING COMPREHENSIVE POST-OPTIMIZATION ERROR ANALYSIS")
        print("=" * 70)
        
        # Load data and models
        X_test, y_test, X_test_scaled, models_data = self.load_data_and_models()
        
        if not models_data:
            print("❌ No models loaded successfully. Aborting error analysis.")
            return None
            
        # Run all analysis components
        print("\n📊 Running analysis components...")
        
        # 1. Misclassification pattern analysis
        misclass_analysis = self.analyze_misclassification_patterns(X_test, y_test, models_data)
        
        # 2. Feature-based error correlation
        feature_error_analysis = self.feature_based_error_correlation(X_test, y_test, models_data)
        
        # 3. Cross-model error comparison
        cross_model_analysis = self.cross_model_error_comparison(y_test, models_data)
        
        # 4. Clinical risk assessment
        clinical_analysis = self.clinical_risk_assessment(y_test, models_data)
        
        # 5. Create visualizations
        self.create_error_visualizations(
            misclass_analysis, 
            feature_error_analysis, 
            cross_model_analysis, 
            clinical_analysis
        )
        
        # Compile comprehensive results
        comprehensive_results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'test_dataset_info': {
                'n_samples': len(X_test),
                'n_features': len(X_test.columns),
                'class_distribution': dict(y_test.value_counts().sort_index())
            },
            'misclassification_analysis': misclass_analysis,
            'feature_error_analysis': feature_error_analysis,
            'cross_model_analysis': cross_model_analysis,
            'clinical_risk_analysis': clinical_analysis,
            'summary': self._create_error_analysis_summary(misclass_analysis, clinical_analysis)
        }
        
        # Save results
        results_file = self.results_path / 'explanations' / 'post_optimization_error_analysis.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
            
        print(f"\n💾 Error analysis results saved to {results_file}")
        
        # Print summary
        self._print_error_analysis_summary(comprehensive_results)
        
        return comprehensive_results
        
    def _create_error_analysis_summary(self, misclass_analysis, clinical_analysis):
        """Create executive summary of error analysis results."""
        model_performances = []
        for model_name, analysis in misclass_analysis.items():
            model_performances.append({
                'model': model_name,
                'misclassification_rate': analysis['misclassification_rate'],
                'test_f1': analysis['test_f1'],
                'status': analysis['status'],
                'false_negatives': analysis['false_negatives'],
                'false_positives': analysis['false_positives']
            })
            
        # Sort by performance (lowest misclassification rate)
        model_performances.sort(key=lambda x: x['misclassification_rate'])
        
        return {
            'best_performing_model': model_performances[0]['model'],
            'lowest_misclassification_rate': model_performances[0]['misclassification_rate'],
            'model_ranking': [model['model'] for model in model_performances],
            'total_models_analyzed': len(model_performances),
            'clinical_concern_models': [
                model['model'] for model in model_performances 
                if clinical_analysis[model['model']]['current_performance']['sensitivity'] < 0.2
            ]
        }
        
    def _print_error_analysis_summary(self, results):
        """Print executive summary of error analysis results."""
        summary = results['summary']
        
        print("\n" + "="*70)
        print("🎯 ERROR ANALYSIS SUMMARY REPORT")
        print("="*70)
        
        print(f"\n🏆 BEST PERFORMING MODEL: {summary['best_performing_model']}")
        print(f"   Misclassification Rate: {summary['lowest_misclassification_rate']:.3f}")
        
        print(f"\n📊 MODEL RANKING (by misclassification rate):")
        for i, model in enumerate(summary['model_ranking'], 1):
            rate = results['misclassification_analysis'][model]['misclassification_rate']
            f1 = results['misclassification_analysis'][model]['test_f1']
            status = results['misclassification_analysis'][model]['status']
            print(f"   {i}. {model}: {rate:.3f} misclass rate, {f1:.3f} F1 ({status})")
        
        print(f"\n🏥 CLINICAL ASSESSMENT:")
        for model in summary['model_ranking']:
            clinical = results['clinical_risk_analysis'][model]
            sensitivity = clinical['current_performance']['sensitivity']
            missed_cases = clinical['clinical_impact']['missed_cases']
            print(f"   {model}: {sensitivity:.1%} sensitivity, {missed_cases} missed cases")
            
        if summary['clinical_concern_models']:
            print(f"\n⚠️ CLINICAL CONCERN MODELS (sensitivity < 20%):")
            for model in summary['clinical_concern_models']:
                print(f"   - {model}")
        
        print("\n✅ ERROR ANALYSIS COMPLETE - Insights ready for clinical review!")


def main():
    """Main execution function for error analysis."""
    # Initialize error analysis framework
    analyzer = PostOptimizationAnalysis()
    
    # Run comprehensive error analysis
    results = analyzer.run_comprehensive_error_analysis()
    
    if results:
        print(f"\n🎉 Error analysis completed successfully!")
        print(f"📁 Results saved to: {analyzer.results_path / 'explanations' / 'post_optimization_error_analysis.json'}")
        print(f"📊 Plots saved to: {analyzer.results_path / 'plots' / 'error_analysis_comprehensive.png'}")
    else:
        print("❌ Error analysis failed. Check model files and data availability.")


if __name__ == "__main__":
    main()

# Usage: python src/analysis/post_optimization_analysis.py