"""
Misclassification Deep Dive Analysis
Detailed investigation of specific misclassification patterns and root causes.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class MisclassificationAnalysis:
    """Deep dive analysis of misclassification patterns and causes."""
    
    def __init__(self, project_root: str = None):
        """Initialize misclassification analysis framework."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        self.data_path = self.project_root / 'data' / 'processed'
        self.models_path = self.project_root / 'results' / 'models'
        self.results_path = self.project_root / 'results' / 'explanations'
        
        # Feature descriptions for clinical interpretation
        self.feature_descriptions = {
            'happy': 'Happiness/mood score',
            'sclmeet': 'Social meeting frequency',
            'inprdsc': 'Income/productivity score',
            'ctrlife': 'Control over life',
            'etfruit': 'Eating fruit frequency',
            'eatveg': 'Eating vegetables frequency',
            'dosprt': 'Physical activity/sport',
            'cgtsmok': 'Cigarette smoking status',
            'alcfreq': 'Alcohol frequency',
            'fltdpr': 'Feeling depressed',
            'flteeff': 'Feeling ineffective',
            'slprl': 'Sleep related issues',
            'wrhpp': 'Work/life happiness',
            'fltlnl': 'Feeling lonely',
            'enjlf': 'Enjoying life',
            'fltsd': 'Feeling sad',
            'gndr': 'Gender',
            'paccnois': 'Physical activity/noise',
            'bmi': 'Body mass index',
            'lifestyle_score': 'Overall lifestyle score',
            'social_score': 'Social interaction score',
            'mental_health_score': 'Mental health composite score'
        }
        
    def load_misclassification_data(self, model_name):
        """Load data and create misclassification indices."""
        print(f"Loading misclassification data for {model_name}...")
        
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
        X_test_scaled = scaler.transform(X_test)
        
        # Load model and generate predictions
        model_paths = {
            'Adaptive_Ensemble': 'adaptive_tuning/Adaptive_Ensemble_complexity_optimized_20260108_233028.joblib',
            'Optimal_Hybrid': 'adaptive_tuning/Optimal_Hybrid_optimal_hybrid_20260108_233028.joblib',
            'Adaptive_LR': 'adaptive_tuning/Adaptive_LR_complexity_increased_20260108_233028.joblib'
        }
        
        model_path = self.models_path / model_paths[model_name]
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
            
        # Generate predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Identify misclassification types
        misclass_indices = {
            'false_positives': X_test.index[(y_test == 0) & (y_pred == 1)].tolist(),
            'false_negatives': X_test.index[(y_test == 1) & (y_pred == 0)].tolist(),
            'true_positives': X_test.index[(y_test == 1) & (y_pred == 1)].tolist(),
            'true_negatives': X_test.index[(y_test == 0) & (y_pred == 0)].tolist()
        }
        
        return X_test, y_test, y_pred, y_proba, misclass_indices, scaler
        
    def analyze_false_positive_patterns(self, X_test, y_test, y_pred, y_proba, fp_indices):
        """Deep analysis of false positive misclassifications."""
        print("Analyzing false positive patterns...")
        
        if not fp_indices:
            return {'message': 'No false positives found', 'analysis': {}}
            
        # Extract false positive samples
        fp_samples = X_test.loc[fp_indices]
        fp_probabilities = y_proba[X_test.index.get_indexer(fp_indices)]
        
        # Compare with true negatives
        tn_indices = X_test.index[(y_test == 0) & (y_pred == 0)].tolist()
        tn_samples = X_test.loc[tn_indices] if tn_indices else pd.DataFrame()
        
        # Feature analysis
        fp_analysis = {}
        
        for feature in X_test.columns:
            if len(fp_samples) > 0:
                fp_mean = fp_samples[feature].mean()
                fp_std = fp_samples[feature].std()
                
                if len(tn_samples) > 0:
                    tn_mean = tn_samples[feature].mean()
                    tn_std = tn_samples[feature].std()
                    
                    # Statistical difference
                    effect_size = abs(fp_mean - tn_mean) / max(fp_std, tn_std, 0.01)
                else:
                    tn_mean, tn_std, effect_size = 0, 0, 0
                    
                fp_analysis[feature] = {
                    'fp_mean': fp_mean,
                    'fp_std': fp_std,
                    'tn_mean': tn_mean,
                    'tn_std': tn_std,
                    'effect_size': effect_size,
                    'description': self.feature_descriptions.get(feature, feature)
                }
                
        # Sort features by effect size
        sorted_features = sorted(fp_analysis.items(), key=lambda x: x[1]['effect_size'], reverse=True)
        
        # Confidence analysis
        high_confidence_fp = [idx for idx in fp_indices if y_proba[X_test.index.get_loc(idx)] > 0.7]
        medium_confidence_fp = [idx for idx in fp_indices if 0.3 <= y_proba[X_test.index.get_loc(idx)] <= 0.7]
        low_confidence_fp = [idx for idx in fp_indices if y_proba[X_test.index.get_loc(idx)] < 0.3]
        
        # Outlier detection
        outlier_analysis = self._detect_outliers(fp_samples, X_test)
        
        return {
            'total_false_positives': len(fp_indices),
            'confidence_distribution': {
                'high_confidence': len(high_confidence_fp),
                'medium_confidence': len(medium_confidence_fp),
                'low_confidence': len(low_confidence_fp)
            },
            'top_discriminating_features': dict(sorted_features[:5]),
            'outlier_analysis': outlier_analysis,
            'sample_characteristics': {
                'mean_probability': np.mean(fp_probabilities),
                'std_probability': np.std(fp_probabilities),
                'min_probability': np.min(fp_probabilities),
                'max_probability': np.max(fp_probabilities)
            }
        }
        
    def analyze_false_negative_patterns(self, X_test, y_test, y_pred, y_proba, fn_indices):
        """Deep analysis of false negative misclassifications."""
        print("Analyzing false negative patterns...")
        
        if not fn_indices:
            return {'message': 'No false negatives found', 'analysis': {}}
            
        # Extract false negative samples
        fn_samples = X_test.loc[fn_indices]
        fn_probabilities = y_proba[X_test.index.get_indexer(fn_indices)]
        
        # Compare with true positives
        tp_indices = X_test.index[(y_test == 1) & (y_pred == 1)].tolist()
        tp_samples = X_test.loc[tp_indices] if tp_indices else pd.DataFrame()
        
        # Feature analysis
        fn_analysis = {}
        
        for feature in X_test.columns:
            if len(fn_samples) > 0:
                fn_mean = fn_samples[feature].mean()
                fn_std = fn_samples[feature].std()
                
                if len(tp_samples) > 0:
                    tp_mean = tp_samples[feature].mean()
                    tp_std = tp_samples[feature].std()
                    
                    # Statistical difference
                    effect_size = abs(fn_mean - tp_mean) / max(fn_std, tp_std, 0.01)
                else:
                    tp_mean, tp_std, effect_size = 0, 0, 0
                    
                fn_analysis[feature] = {
                    'fn_mean': fn_mean,
                    'fn_std': fn_std,
                    'tp_mean': tp_mean,
                    'tp_std': tp_std,
                    'effect_size': effect_size,
                    'description': self.feature_descriptions.get(feature, feature)
                }
                
        # Sort features by effect size
        sorted_features = sorted(fn_analysis.items(), key=lambda x: x[1]['effect_size'], reverse=True)
        
        # Risk factor analysis for missed cases
        risk_factor_analysis = self._analyze_missed_risk_factors(fn_samples)
        
        # Severity analysis (how far below threshold)
        threshold_distance = 0.5 - fn_probabilities  # Distance below default threshold
        
        return {
            'total_false_negatives': len(fn_indices),
            'severity_analysis': {
                'mean_threshold_distance': np.mean(threshold_distance),
                'max_threshold_distance': np.max(threshold_distance),
                'cases_very_low_risk': np.sum(fn_probabilities < 0.2),
                'cases_borderline': np.sum((fn_probabilities >= 0.2) & (fn_probabilities < 0.4))
            },
            'top_discriminating_features': dict(sorted_features[:5]),
            'risk_factor_analysis': risk_factor_analysis,
            'sample_characteristics': {
                'mean_probability': np.mean(fn_probabilities),
                'std_probability': np.std(fn_probabilities),
                'min_probability': np.min(fn_probabilities),
                'max_probability': np.max(fn_probabilities)
            }
        }
        
    def _detect_outliers(self, misclass_samples, full_dataset):
        """Detect outlier patterns in misclassified samples."""
        if len(misclass_samples) == 0:
            return {'outlier_features': {}, 'outlier_count': 0}
            
        outlier_features = {}
        total_outliers = 0
        
        for feature in misclass_samples.columns:
            # Calculate Z-scores relative to full dataset
            feature_mean = full_dataset[feature].mean()
            feature_std = full_dataset[feature].std()
            
            if feature_std > 0:
                z_scores = np.abs((misclass_samples[feature] - feature_mean) / feature_std)
                outliers = z_scores > 2.5  # 2.5 standard deviations
                
                if outliers.any():
                    outlier_features[feature] = {
                        'outlier_count': int(outliers.sum()),
                        'outlier_percentage': float(outliers.mean() * 100),
                        'max_z_score': float(z_scores.max()),
                        'mean_z_score': float(z_scores.mean())
                    }
                    total_outliers += outliers.sum()
                    
        return {
            'outlier_features': outlier_features,
            'total_outlier_instances': int(total_outliers),
            'samples_with_outliers': len([f for f in outlier_features if outlier_features[f]['outlier_count'] > 0])
        }
        
    def _analyze_missed_risk_factors(self, fn_samples):
        """Analyze risk factors in missed (false negative) cases."""
        if len(fn_samples) == 0:
            return {}
            
        risk_analysis = {}
        
        # Clinical risk factor combinations
        clinical_patterns = {
            'high_stress_low_happiness': (fn_samples['fltdpr'] > 1) & (fn_samples['happy'] < -1),
            'social_isolation': (fn_samples['sclmeet'] < -1) & (fn_samples['fltlnl'] > 1),
            'lifestyle_risk_factors': (fn_samples['cgtsmok'] > 0) & (fn_samples['alcfreq'] > 1),
            'poor_mental_health': fn_samples['mental_health_score'] < -1,
            'low_physical_activity': fn_samples['dosprt'] < -1
        }
        
        for pattern_name, pattern_mask in clinical_patterns.items():
            if pattern_mask.any():
                risk_analysis[pattern_name] = {
                    'case_count': int(pattern_mask.sum()),
                    'percentage': float(pattern_mask.mean() * 100),
                    'description': self._get_pattern_description(pattern_name)
                }
                
        return risk_analysis
        
    def _get_pattern_description(self, pattern_name):
        """Get clinical description for risk patterns."""
        descriptions = {
            'high_stress_low_happiness': 'High stress with low happiness scores',
            'social_isolation': 'Social isolation patterns with loneliness',
            'lifestyle_risk_factors': 'Smoking and alcohol risk factors',
            'poor_mental_health': 'Poor overall mental health scores',
            'low_physical_activity': 'Low physical activity levels'
        }
        return descriptions.get(pattern_name, pattern_name)
        
    def perform_clustering_analysis(self, X_test, misclass_indices):
        """Perform clustering analysis on misclassified samples."""
        print("ANALYSIS: Performing clustering analysis on misclassifications...")
        
        # Combine all misclassified samples
        all_misclass = misclass_indices['false_positives'] + misclass_indices['false_negatives']
        
        if len(all_misclass) < 3:
            return {'message': 'Insufficient misclassified samples for clustering'}
            
        misclass_samples = X_test.loc[all_misclass]
        
        # Standardize features for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        misclass_scaled = scaler.fit_transform(misclass_samples)
        
        # Determine optimal number of clusters
        max_clusters = min(5, len(all_misclass) - 1)
        if max_clusters < 2:
            return {'message': 'Too few samples for meaningful clustering'}
            
        # Perform clustering
        optimal_k = min(3, max_clusters)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(misclass_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        
        for cluster_id in range(optimal_k):
            cluster_mask = cluster_labels == cluster_id
            cluster_samples = misclass_samples[cluster_mask]
            cluster_indices = np.array(all_misclass)[cluster_mask].tolist()
            
            # Determine cluster composition
            fp_in_cluster = len([idx for idx in cluster_indices if idx in misclass_indices['false_positives']])
            fn_in_cluster = len([idx for idx in cluster_indices if idx in misclass_indices['false_negatives']])
            
            # Feature characteristics
            cluster_features = {}
            for feature in misclass_samples.columns:
                cluster_features[feature] = {
                    'mean': float(cluster_samples[feature].mean()),
                    'std': float(cluster_samples[feature].std()),
                    'description': self.feature_descriptions.get(feature, feature)
                }
                
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'sample_count': int(cluster_mask.sum()),
                'false_positives': fp_in_cluster,
                'false_negatives': fn_in_cluster,
                'dominant_error_type': 'false_positive' if fp_in_cluster > fn_in_cluster else 'false_negative',
                'feature_characteristics': cluster_features,
                'cluster_description': self._describe_cluster_pattern(cluster_features)
            }
            
        return {
            'total_clusters': optimal_k,
            'total_misclassified': len(all_misclass),
            'cluster_analysis': cluster_analysis,
            'clustering_quality': self._evaluate_clustering_quality(misclass_scaled, cluster_labels)
        }
        
    def _describe_cluster_pattern(self, cluster_features):
        """Create clinical description of cluster patterns."""
        # Identify key characteristics
        descriptions = []
        
        # Age patterns
        happy_mean = cluster_features.get('happy', {}).get('mean', 0)
        if happy_mean > 1:
            descriptions.append("high happiness")
        elif happy_mean < -1:
            descriptions.append("low happiness")
            
        # Gender patterns
        gndr_mean = cluster_features.get('gndr', {}).get('mean', 0)
        if gndr_mean > 0.5:
            descriptions.append("predominantly male")
        elif gndr_mean < -0.5:
            descriptions.append("predominantly female")
            
        # Clinical indicators
        mental_health_mean = cluster_features.get('mental_health_score', {}).get('mean', 0)
        if mental_health_mean < -1:
            descriptions.append("poor mental health")
            
        lifestyle_mean = cluster_features.get('lifestyle_score', {}).get('mean', 0)
        if lifestyle_mean < -1:
            descriptions.append("poor lifestyle factors")
            
        if not descriptions:
            descriptions.append("mixed clinical characteristics")
            
        return ", ".join(descriptions[:3])  # Limit to top 3 characteristics
        
    def _evaluate_clustering_quality(self, data, labels):
        """Evaluate clustering quality metrics."""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        if len(np.unique(labels)) < 2:
            return {'silhouette_score': 0, 'calinski_harabasz_score': 0}
            
        silhouette = silhouette_score(data, labels)
        calinski = calinski_harabasz_score(data, labels)
        
        return {
            'silhouette_score': float(silhouette),
            'calinski_harabasz_score': float(calinski)
        }
        
    def create_misclassification_visualizations(self, model_name, X_test, misclass_indices, fp_analysis, fn_analysis, clustering_results):
        """Create comprehensive misclassification visualizations."""
        print("Creating misclassification visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        
        # Plot 1: Misclassification distribution
        ax1 = axes[0, 0]
        categories = ['False Positives', 'False Negatives', 'True Positives', 'True Negatives']
        counts = [len(misclass_indices[key]) for key in ['false_positives', 'false_negatives', 'true_positives', 'true_negatives']]
        colors = ['red', 'orange', 'green', 'lightgreen']
        
        ax1.bar(categories, counts, color=colors, alpha=0.7)
        ax1.set_title('Classification Distribution')
        ax1.set_ylabel('Sample Count')
        ax1.tick_params(axis='x', rotation=45)
        
        for i, count in enumerate(counts):
            ax1.text(i, count + max(counts)*0.02, str(count), ha='center', va='bottom')
        
        # Plot 2: False Positive Feature Analysis
        ax2 = axes[0, 1]
        if fp_analysis.get('top_discriminating_features'):
            fp_features = list(fp_analysis['top_discriminating_features'].keys())[:5]
            fp_effects = [fp_analysis['top_discriminating_features'][f]['effect_size'] for f in fp_features]
            
            ax2.barh(fp_features, fp_effects, color='red', alpha=0.7)
            ax2.set_title('False Positive: Top Discriminating Features')
            ax2.set_xlabel('Effect Size')
        else:
            ax2.text(0.5, 0.5, 'No False Positives', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('False Positive Analysis')
        
        # Plot 3: False Negative Feature Analysis
        ax3 = axes[0, 2]
        if fn_analysis.get('top_discriminating_features'):
            fn_features = list(fn_analysis['top_discriminating_features'].keys())[:5]
            fn_effects = [fn_analysis['top_discriminating_features'][f]['effect_size'] for f in fn_features]
            
            ax3.barh(fn_features, fn_effects, color='orange', alpha=0.7)
            ax3.set_title('False Negative: Top Discriminating Features')
            ax3.set_xlabel('Effect Size')
        else:
            ax3.text(0.5, 0.5, 'No False Negatives', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('False Negative Analysis')
        
        # Plot 4: Confidence Distribution for FP
        ax4 = axes[1, 0]
        if 'confidence_distribution' in fp_analysis:
            conf_data = fp_analysis['confidence_distribution']
            conf_labels = ['High (>0.7)', 'Medium (0.3-0.7)', 'Low (<0.3)']
            conf_values = [conf_data['high_confidence'], conf_data['medium_confidence'], conf_data['low_confidence']]
            
            ax4.pie(conf_values, labels=conf_labels, autopct='%1.1f%%', startangle=90)
            ax4.set_title('False Positive Confidence Distribution')
        else:
            ax4.text(0.5, 0.5, 'No False Positives', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('FP Confidence Distribution')
        
        # Plot 5: False Negative Severity Analysis
        ax5 = axes[1, 1]
        if 'severity_analysis' in fn_analysis:
            severity = fn_analysis['severity_analysis']
            severity_labels = ['Very Low Risk\n(<0.2)', 'Borderline\n(0.2-0.4)', 'Other\n(≥0.4)']
            other_count = fn_analysis['total_false_negatives'] - severity['cases_very_low_risk'] - severity['cases_borderline']
            severity_values = [severity['cases_very_low_risk'], severity['cases_borderline'], other_count]
            
            ax5.pie(severity_values, labels=severity_labels, autopct='%1.1f%%', startangle=90)
            ax5.set_title('False Negative Risk Level Distribution')
        else:
            ax5.text(0.5, 0.5, 'No False Negatives', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('FN Risk Distribution')
        
        # Plot 6: Clustering Results
        ax6 = axes[1, 2]
        if clustering_results and 'cluster_analysis' in clustering_results:
            cluster_data = clustering_results['cluster_analysis']
            cluster_names = list(cluster_data.keys())
            cluster_sizes = [cluster_data[name]['sample_count'] for name in cluster_names]
            
            ax6.bar(cluster_names, cluster_sizes, alpha=0.7)
            ax6.set_title('Misclassification Clusters')
            ax6.set_ylabel('Sample Count')
            ax6.tick_params(axis='x', rotation=45)
        else:
            ax6.text(0.5, 0.5, 'Insufficient Data\nfor Clustering', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Clustering Analysis')
        
        # Plot 7: Risk Factor Analysis (FN only)
        ax7 = axes[2, 0]
        if 'risk_factor_analysis' in fn_analysis and fn_analysis['risk_factor_analysis']:
            risk_factors = list(fn_analysis['risk_factor_analysis'].keys())[:5]
            risk_percentages = [fn_analysis['risk_factor_analysis'][rf]['percentage'] for rf in risk_factors]
            
            ax7.barh(risk_factors, risk_percentages, color='purple', alpha=0.7)
            ax7.set_title('Missed Risk Factor Patterns')
            ax7.set_xlabel('Percentage of False Negatives')
        else:
            ax7.text(0.5, 0.5, 'No Risk Pattern\nData Available', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Risk Factor Analysis')
        
        # Plot 8: Outlier Analysis
        ax8 = axes[2, 1]
        if fp_analysis.get('outlier_analysis', {}).get('outlier_features'):
            outlier_data = fp_analysis['outlier_analysis']['outlier_features']
            if outlier_data:
                outlier_features = list(outlier_data.keys())[:5]
                outlier_percentages = [outlier_data[f]['outlier_percentage'] for f in outlier_features]
                
                ax8.bar(outlier_features, outlier_percentages, color='red', alpha=0.7)
                ax8.set_title('Outlier Features (False Positives)')
                ax8.set_ylabel('Outlier Percentage')
                ax8.tick_params(axis='x', rotation=45)
            else:
                ax8.text(0.5, 0.5, 'No Outliers\nDetected', ha='center', va='center', transform=ax8.transAxes)
                ax8.set_title('Outlier Analysis')
        else:
            ax8.text(0.5, 0.5, 'No Outlier\nData Available', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Outlier Analysis')
        
        # Plot 9: Model Performance Summary
        ax9 = axes[2, 2]
        total_samples = sum(counts)
        accuracy = (counts[2] + counts[3]) / total_samples if total_samples > 0 else 0
        precision = counts[2] / (counts[2] + counts[0]) if (counts[2] + counts[0]) > 0 else 0
        recall = counts[2] / (counts[2] + counts[1]) if (counts[2] + counts[1]) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1_score]
        
        bars = ax9.bar(metrics, values, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
        ax9.set_title('Model Performance Metrics')
        ax9.set_ylabel('Score')
        ax9.set_ylim(0, 1)
        
        for bar, value in zip(bars, values):
            ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(f'Misclassification Deep Dive Analysis: {model_name}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save visualization
        output_file = self.results_path / f'{model_name}_misclassification_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Misclassification visualizations saved to {output_file}")
        return output_file
        
    def run_comprehensive_misclassification_analysis(self, model_name='Adaptive_Ensemble'):
        """Run comprehensive misclassification analysis for a specific model."""
        print(f"STARTING COMPREHENSIVE MISCLASSIFICATION ANALYSIS")
        print(f"Target Model: {model_name}")
        print("=" * 70)
        
        try:
            # Load data and misclassifications
            X_test, y_test, y_pred, y_proba, misclass_indices, scaler = self.load_misclassification_data(model_name)
            
            # Analyze false positives
            fp_analysis = self.analyze_false_positive_patterns(
                X_test, y_test, y_pred, y_proba, misclass_indices['false_positives']
            )
            
            # Analyze false negatives
            fn_analysis = self.analyze_false_negative_patterns(
                X_test, y_test, y_pred, y_proba, misclass_indices['false_negatives']
            )
            
            # Clustering analysis
            clustering_results = self.perform_clustering_analysis(X_test, misclass_indices)
            
            # Create visualizations
            viz_file = self.create_misclassification_visualizations(
                model_name, X_test, misclass_indices, fp_analysis, fn_analysis, clustering_results
            )
            
            # Compile comprehensive results
            comprehensive_results = {
                'model_name': model_name,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'dataset_info': {
                    'total_samples': len(X_test),
                    'total_features': len(X_test.columns),
                    'class_distribution': dict(y_test.value_counts().sort_index())
                },
                'misclassification_summary': {
                    'false_positives': len(misclass_indices['false_positives']),
                    'false_negatives': len(misclass_indices['false_negatives']),
                    'true_positives': len(misclass_indices['true_positives']),
                    'true_negatives': len(misclass_indices['true_negatives']),
                    'total_misclassified': len(misclass_indices['false_positives']) + len(misclass_indices['false_negatives'])
                },
                'false_positive_analysis': fp_analysis,
                'false_negative_analysis': fn_analysis,
                'clustering_analysis': clustering_results,
                'visualization_file': str(viz_file)
            }
            
            # Save results
            results_file = self.results_path / f'{model_name}_misclassification_deep_dive.json'
            with open(results_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
                
            print(f"\nSAVED: Misclassification analysis saved to {results_file}")
            
            # Print summary
            self._print_misclassification_summary(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            print(f"ERROR: Misclassification analysis failed: {e}")
            return None
            
    def _print_misclassification_summary(self, results):
        """Print executive summary of misclassification analysis."""
        summary = results['misclassification_summary']
        
        print("\n" + "="*70)
        print("MISCLASSIFICATION ANALYSIS SUMMARY")
        print("="*70)
        
        total_samples = summary['false_positives'] + summary['false_negatives'] + summary['true_positives'] + summary['true_negatives']
        accuracy = (summary['true_positives'] + summary['true_negatives']) / total_samples
        
        print(f"\nClassification Results:")
        print(f"   True Positives: {summary['true_positives']}")
        print(f"   True Negatives: {summary['true_negatives']}")
        print(f"   False Positives: {summary['false_positives']} (unnecessary referrals)")
        print(f"   False Negatives: {summary['false_negatives']} (missed cases)")
        print(f"   Accuracy: {accuracy:.3f}")
        
        # False Positive insights
        if results['false_positive_analysis'] and 'top_discriminating_features' in results['false_positive_analysis']:
            print(f"\nFalse Positive Insights:")
            fp_features = results['false_positive_analysis']['top_discriminating_features']
            for feature, data in list(fp_features.items())[:3]:
                print(f"   {data['description']}: Effect size {data['effect_size']:.3f}")
                
        # False Negative insights
        if results['false_negative_analysis'] and 'top_discriminating_features' in results['false_negative_analysis']:
            print(f"\nFalse Negative Insights:")
            fn_features = results['false_negative_analysis']['top_discriminating_features']
            for feature, data in list(fn_features.items())[:3]:
                print(f"   {data['description']}: Effect size {data['effect_size']:.3f}")
                
        # Clustering insights
        if results['clustering_analysis'] and 'cluster_analysis' in results['clustering_analysis']:
            clusters = results['clustering_analysis']['cluster_analysis']
            print(f"\nCLUSTERING INSIGHTS: Clustering Insights:")
            for cluster_name, cluster_data in clusters.items():
                print(f"   {cluster_name}: {cluster_data['sample_count']} samples - {cluster_data['cluster_description']}")
        
        print("\nMISCLASSIFICATION ANALYSIS COMPLETE")


def main():
    """Main execution function for misclassification analysis."""
    # Initialize misclassification analysis framework
    analyzer = MisclassificationAnalysis()
    
    # Run comprehensive misclassification analysis
    results = analyzer.run_comprehensive_misclassification_analysis('Adaptive_Ensemble')
    
    if results:
        print(f"\nMisclassification analysis completed successfully!")
        print(f"Results saved to: {analyzer.results_path}")
    else:
        print("ERROR: Misclassification analysis failed. Check model files and data availability.")


if __name__ == "__main__":
    main()

# Usage: python src/analysis/misclassification_deep_dive.py