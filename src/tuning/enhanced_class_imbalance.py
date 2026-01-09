"""
Enhanced class imbalance techniques for heart risk prediction.

Implements comprehensive improvements based on diagnostic results:
1. Advanced Feature Engineering: Domain-specific heart risk features
2. Heavy Regularization: Address severe overfitting
3. Cost-sensitive Learning: Aggressive class weight adjustment  
4. Ensemble Methods: Combine complementary models
5. Neural Network Redesign: Proper architecture and initialization
6. Feature Selection: Remove noisy features
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import os
import psutil
from datetime import datetime
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    """Advanced feature engineering for heart risk prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_indices = None
        
    def create_domain_features(self, X, feature_names):
        """Create domain-specific heart risk features with consistent output."""
        print("🔧 Creating domain-specific heart risk features...")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(X, columns=feature_names)
        original_cols = list(df.columns)
        
        # 1. Lifestyle Risk Score
        lifestyle_cols = ['etfruit', 'eatveg', 'dosprt', 'cgtsmok', 'alcfreq']
        available_lifestyle = [col for col in lifestyle_cols if col in df.columns]
        
        if available_lifestyle:
            df['lifestyle_risk_enhanced'] = df[available_lifestyle].mean(axis=1)
        
        # 2. Mental Health Composite
        mental_cols = ['fltdpr', 'flteeff', 'fltlnl', 'fltsd', 'wrhpp']
        available_mental = [col for col in mental_cols if col in df.columns]
        
        if available_mental:
            df['mental_health_composite'] = df[available_mental].mean(axis=1)
            df['mental_distress_flag'] = (df['mental_health_composite'] < -0.5).astype(int)
        
        # 3. Social Connection Index
        social_cols = ['sclmeet', 'happy', 'enjlf']
        available_social = [col for col in social_cols if col in df.columns]
        
        if available_social:
            df['social_connection_index'] = df[available_social].mean(axis=1)
        
        # 4. BMI Risk Categories (if BMI exists)
        if 'bmi' in df.columns:
            df['bmi_risk_high'] = (df['bmi'] > 1).astype(int)
            df['bmi_risk_low'] = (df['bmi'] < -1).astype(int)
        
        # 5. Control and Productivity
        if 'ctrlife' in df.columns and 'inprdsc' in df.columns:
            df['control_productivity'] = df['ctrlife'] * df['inprdsc']
        
        # 6. Physical Activity Risk
        if 'dosprt' in df.columns:
            df['low_activity'] = (df['dosprt'] < -0.5).astype(int)
        
        # 7. Smoking and Drinking Interaction
        if 'cgtsmok' in df.columns and 'alcfreq' in df.columns:
            df['smoking_drinking_risk'] = df['cgtsmok'] * df['alcfreq']
        
        # Calculate number of new features
        new_features_count = len(df.columns) - len(original_cols)
        print(f"✅ Created {new_features_count} new features")
        
        # Return consistent feature names
        feature_names_out = list(df.columns)
        return df.values, feature_names_out
    
    def fit_transform(self, X_train, y_train, feature_names):
        """Full feature engineering pipeline with consistent transforms."""
        print(f"🔧 Starting feature engineering on {X_train.shape}")
        
        # 1. Create domain features
        X_enhanced, enhanced_names = self.create_domain_features(X_train, feature_names)
        print(f"After domain features: {X_enhanced.shape}")
        
        # 2. Feature selection (select most informative features)
        self.feature_selector = SelectKBest(f_classif, k=min(25, X_enhanced.shape[1]))
        X_selected = self.feature_selector.fit_transform(X_enhanced, y_train)
        print(f"After feature selection: {X_selected.shape}")
        
        # 3. Scale features
        X_final = self.scaler.fit_transform(X_selected)
        print(f"After scaling: {X_final.shape}")
        
        return X_final
    
    def transform(self, X, feature_names):
        """Transform new data using fitted pipeline."""
        # Apply same domain feature creation
        X_enhanced, _ = self.create_domain_features(X, feature_names)
        
        # Apply same feature selection
        X_selected = self.feature_selector.transform(X_enhanced)
        
        # Apply scaling
        X_final = self.scaler.transform(X_selected)
        
        return X_final

class RegularizedNeuralNetwork(nn.Module):
    """Heavily regularized neural network for class imbalance."""
    
    def __init__(self, input_size, hidden_layers=[128, 64, 32], dropout_rates=[0.5, 0.4, 0.3]):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization for stable training
            layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            
            # Heavy dropout for regularization
            layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Final layer with reduced capacity
        layers.append(nn.Linear(prev_size, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization."""
        if isinstance(module, nn.Linear):
            # Xavier initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class EnhancedNeuralNetworkWrapper:
    """Enhanced Neural Network with proper regularization and class weighting."""
    
    def __init__(self, hidden_layers=[128, 64, 32], learning_rate=0.001, epochs=200, 
                 batch_size=256, dropout_rates=[0.5, 0.4, 0.3], weight_decay=0.01, 
                 class_weights=None, random_state=42):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rates = dropout_rates
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Train the enhanced neural network."""
        torch.manual_seed(self.random_state)
        
        # Scale features (additional scaling for NN)
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        # Create model
        self.model = RegularizedNeuralNetwork(
            X.shape[1], 
            self.hidden_layers, 
            self.dropout_rates
        )
        
        # Calculate class weights if not provided
        if self.class_weights is None:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            class_weights = torch.FloatTensor(class_weights)
        else:
            class_weights = torch.FloatTensor(self.class_weights)
        
        # Training setup with class-weighted loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay  # L2 regularization
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=20, factor=0.5
        )
        
        # Training loop with early stopping
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 30
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.numpy()
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.numpy()

class CostSensitiveClassifier:
    """Wrapper for cost-sensitive learning with aggressive class weighting."""
    
    def __init__(self, base_model, cost_ratio=10.0):
        self.base_model = base_model
        self.cost_ratio = cost_ratio  # How much more to weight minority class
        
    def fit(self, X, y):
        """Fit with cost-sensitive weighting."""
        # Calculate aggressive class weights
        unique_classes, class_counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        
        # Create aggressive class weights
        class_weights = {}
        for cls, count in zip(unique_classes, class_counts):
            if cls == minority_class:
                class_weights[cls] = self.cost_ratio
            else:
                class_weights[cls] = 1.0
        
        print(f"💰 Cost-sensitive weights: {class_weights}")
        
        # Apply weights to model
        if hasattr(self.base_model, 'class_weight'):
            self.base_model.class_weight = class_weights
        elif isinstance(self.base_model, xgb.XGBClassifier):
            # For XGBoost, use scale_pos_weight
            self.base_model.scale_pos_weight = self.cost_ratio
        
        # Fit the model
        self.base_model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

class EnhancedEnsemble:
    """Enhanced ensemble combining complementary models."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = []
        self.weights = []
        
    def create_complementary_models(self):
        """Create models with complementary strengths."""
        
        # 1. Regularized Logistic Regression (underfitting tendency - good for precision)
        lr_regularized = LogisticRegression(
            C=0.01,  # Strong regularization
            class_weight='balanced',
            max_iter=2000,
            random_state=self.random_state
        )
        
        # 2. Heavily Regularized Random Forest (overfitting tendency - regularized)
        rf_regularized = RandomForestClassifier(
            n_estimators=50,  # Reduced
            max_depth=3,      # Heavily limited
            min_samples_split=20,  # Increased
            min_samples_leaf=10,   # Increased
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=self.random_state
        )
        
        # 3. Heavily Regularized XGBoost (overfitting tendency - regularized)
        xgb_regularized = xgb.XGBClassifier(
            n_estimators=50,   # Reduced
            max_depth=3,       # Limited
            learning_rate=0.05, # Reduced
            reg_alpha=10,      # Strong L1
            reg_lambda=10,     # Strong L2
            subsample=0.6,     # Regularization
            colsample_bytree=0.6,  # Regularization
            scale_pos_weight=10,   # Class balance
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        return [
            ('lr_regularized', lr_regularized),
            ('rf_regularized', rf_regularized), 
            ('xgb_regularized', xgb_regularized)
        ]
    
    def fit(self, X, y, X_val, y_val):
        """Fit ensemble with validation-based weighting."""
        models = self.create_complementary_models()
        
        print("🎭 Training Enhanced Ensemble...")
        
        model_performances = []
        for name, model in models:
            print(f"  Training {name}...")
            
            # Fit model
            model.fit(X, y)
            
            # Evaluate on validation set
            val_pred = model.predict(X_val)
            val_f1 = f1_score(y_val, val_pred)
            
            print(f"    {name} validation F1: {val_f1:.4f}")
            
            self.models.append((name, model))
            model_performances.append(val_f1)
        
        # Calculate weights based on performance
        performances = np.array(model_performances)
        # Use softmax to convert to weights (gives more weight to better performers)
        self.weights = np.exp(performances * 5) / np.sum(np.exp(performances * 5))
        
        print(f"📊 Ensemble weights: {dict(zip([name for name, _ in self.models], self.weights))}")
        
        return self
    
    def predict(self, X):
        """Weighted ensemble prediction."""
        predictions = np.array([model.predict(X) for _, model in self.models])
        
        # Weighted majority vote
        weighted_votes = np.zeros(X.shape[0])
        for i, weight in enumerate(self.weights):
            weighted_votes += weight * predictions[i]
        
        return (weighted_votes > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Weighted ensemble probability prediction."""
        all_probas = np.array([model.predict_proba(X) for _, model in self.models])
        
        # Weighted average of probabilities
        weighted_probas = np.zeros_like(all_probas[0])
        for i, weight in enumerate(self.weights):
            weighted_probas += weight * all_probas[i]
        
        return weighted_probas

def load_processed_data():
    """Load processed heart risk data with full splits."""
    print("📂 Loading processed heart risk data...")
    
    try:
        # Load all data splits
        train_data = pd.read_csv('/Users/peter/Desktop/heart_risk_prediction/data/processed/train.csv')
        val_data = pd.read_csv('/Users/peter/Desktop/heart_risk_prediction/data/processed/validation.csv')
        test_data = pd.read_csv('/Users/peter/Desktop/heart_risk_prediction/data/processed/test.csv')
        
        # Separate features and targets - target column is 'hltprhc'
        X_train = train_data.drop('hltprhc', axis=1).values
        y_train = train_data['hltprhc'].values
        X_val = val_data.drop('hltprhc', axis=1).values
        y_val = val_data['hltprhc'].values
        X_test = test_data.drop('hltprhc', axis=1).values
        y_test = test_data['hltprhc'].values
        
        # Get feature names
        feature_names = list(train_data.columns[:-1])  # All except target
        
        print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_names
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def analyze_model_performance(train_score, val_score, model_name, technique):
    """Analyze model performance to detect overfitting/underfitting."""
    gap = train_score - val_score
    
    # Define thresholds
    OVERFITTING_THRESHOLD = 0.05  
    UNDERFITTING_THRESHOLD = 0.30  
    GOOD_PERFORMANCE_THRESHOLD = 0.35  
    
    # Determine status
    if gap > OVERFITTING_THRESHOLD:
        if val_score < UNDERFITTING_THRESHOLD:
            status = "SEVERE_OVERFITTING"
            status_emoji = "🔴"
            description = f"Severe overfitting (gap: {gap:.3f}, low val performance)"
        else:
            status = "OVERFITTING"
            status_emoji = "🟠"
            description = f"Overfitting detected (gap: {gap:.3f})"
    elif train_score < UNDERFITTING_THRESHOLD and val_score < UNDERFITTING_THRESHOLD:
        status = "UNDERFITTING"
        status_emoji = "🟡"
        description = f"Underfitting (both scores low: train={train_score:.3f}, val={val_score:.3f})"
    elif train_score >= GOOD_PERFORMANCE_THRESHOLD and val_score >= GOOD_PERFORMANCE_THRESHOLD and abs(gap) <= OVERFITTING_THRESHOLD:
        status = "GOOD"
        status_emoji = "🟢"
        description = f"Good performance (balanced scores)"
    else:
        status = "ACCEPTABLE"
        status_emoji = "🟡"
        description = f"Acceptable performance (gap: {gap:.3f})"
    
    analysis = {
        'model_name': model_name,
        'technique': technique,
        'train_score': train_score,
        'val_score': val_score,
        'gap': gap,
        'status': status,
        'status_emoji': status_emoji,
        'description': description
    }
    
    print(f"  {status_emoji} {model_name} ({technique}): {description}")
    
    return analysis

class EnhancedModelSaver:
    """Enhanced model saver with better organization."""
    
    def __init__(self, base_path='/Users/peter/Desktop/heart_risk_prediction/results/models'):
        self.base_path = base_path
        self.enhanced_path = os.path.join(base_path, 'enhanced_techniques')
        os.makedirs(self.enhanced_path, exist_ok=True)
        
    def save_model(self, model, model_name, technique, timestamp):
        """Save a trained model with metadata."""
        filename = f"{model_name}_{technique}_{timestamp}.joblib"
        filepath = os.path.join(self.enhanced_path, filename)
        
        model_data = {
            'model': model,
            'model_name': model_name,
            'technique': technique,
            'timestamp': timestamp,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Saved {model_name} ({technique}) to {filepath}")
        return filepath
    
    def save_results(self, results, timestamp):
        """Save experiment results."""
        results_path = os.path.join(self.enhanced_path, f'enhanced_results_{timestamp}.json')
        
        json_results = {
            'timestamp': timestamp,
            'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_type': 'Enhanced Class Imbalance Techniques',
            'results': results
        }
        
        import json
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"📊 Saved enhanced results to {results_path}")
        return results_path

def test_enhanced_techniques():
    """Test enhanced class imbalance techniques with comprehensive improvements."""
    print("🚀 TESTING ENHANCED CLASS IMBALANCE TECHNIQUES")
    print("=" * 70)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_processed_data()
    
    # Initialize feature engineer
    feature_engineer = EnhancedFeatureEngineer()
    
    # Apply enhanced feature engineering
    print(f"\n🔧 ENHANCED FEATURE ENGINEERING")
    print("=" * 50)
    X_train_enhanced = feature_engineer.fit_transform(X_train, y_train, feature_names)
    X_val_enhanced = feature_engineer.transform(X_val, feature_names)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Enhanced features: {X_train_enhanced.shape[1]}")
    print(f"Feature reduction: {((X_train.shape[1] - X_train_enhanced.shape[1]) / X_train.shape[1] * 100):.1f}%")
    
    # Initialize model saver
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_saver = EnhancedModelSaver()
    
    print(f"💾 Models will be saved to: {model_saver.enhanced_path}")
    
    results = {}
    
    # Test models with enhanced techniques
    print(f"\n🧪 TESTING ENHANCED MODELS")
    print("=" * 50)
    
    # 1. Enhanced Logistic Regression with Cost-Sensitive Learning
    print("\n💰 COST-SENSITIVE LOGISTIC REGRESSION:")
    lr_enhanced = CostSensitiveClassifier(
        LogisticRegression(C=0.1, max_iter=2000, random_state=42),
        cost_ratio=15.0
    )
    lr_enhanced.fit(X_train_enhanced, y_train)
    lr_train_pred = lr_enhanced.predict(X_train_enhanced)
    lr_val_pred = lr_enhanced.predict(X_val_enhanced)
    lr_train_f1 = f1_score(y_train, lr_train_pred)
    lr_val_f1 = f1_score(y_val, lr_val_pred)
    lr_analysis = analyze_model_performance(lr_train_f1, lr_val_f1, "Enhanced_LR", "cost_sensitive")
    
    # 2. Enhanced Neural Network
    print("\n🧠 ENHANCED NEURAL NETWORK:")
    nn_enhanced = EnhancedNeuralNetworkWrapper(
        hidden_layers=[128, 64, 32],
        learning_rate=0.001,
        epochs=200,
        dropout_rates=[0.5, 0.4, 0.3],
        weight_decay=0.01,
        random_state=42
    )
    nn_enhanced.fit(X_train_enhanced, y_train)
    nn_train_pred = nn_enhanced.predict(X_train_enhanced)
    nn_val_pred = nn_enhanced.predict(X_val_enhanced)
    nn_train_f1 = f1_score(y_train, nn_train_pred)
    nn_val_f1 = f1_score(y_val, nn_val_pred)
    nn_analysis = analyze_model_performance(nn_train_f1, nn_val_f1, "Enhanced_NN", "regularized")
    
    # 3. Enhanced Ensemble
    print("\n🎭 ENHANCED ENSEMBLE:")
    ensemble_enhanced = EnhancedEnsemble(random_state=42)
    ensemble_enhanced.fit(X_train_enhanced, y_train, X_val_enhanced, y_val)
    ensemble_train_pred = ensemble_enhanced.predict(X_train_enhanced)
    ensemble_val_pred = ensemble_enhanced.predict(X_val_enhanced)
    ensemble_train_f1 = f1_score(y_train, ensemble_train_pred)
    ensemble_val_f1 = f1_score(y_val, ensemble_val_pred)
    ensemble_analysis = analyze_model_performance(ensemble_train_f1, ensemble_val_f1, "Enhanced_Ensemble", "weighted_ensemble")
    
    # Store results
    results = {
        'Enhanced_LR': {
            'val_f1': lr_val_f1,
            'train_f1': lr_train_f1,
            'analysis': lr_analysis
        },
        'Enhanced_NN': {
            'val_f1': nn_val_f1,
            'train_f1': nn_train_f1,
            'analysis': nn_analysis
        },
        'Enhanced_Ensemble': {
            'val_f1': ensemble_val_f1,
            'train_f1': ensemble_train_f1,
            'analysis': ensemble_analysis
        }
    }
    
    # Save models
    print(f"\n💾 Saving enhanced models...")
    model_saver.save_model(lr_enhanced, 'Enhanced_LR', 'cost_sensitive', timestamp)
    model_saver.save_model(nn_enhanced, 'Enhanced_NN', 'regularized', timestamp)
    model_saver.save_model(ensemble_enhanced, 'Enhanced_Ensemble', 'weighted_ensemble', timestamp)
    
    # Summary
    print(f"\n📊 ENHANCED RESULTS SUMMARY")
    print("=" * 50)
    
    best_model = max(results.keys(), key=lambda k: results[k]['val_f1'])
    best_score = results[best_model]['val_f1']
    
    print(f"🏆 Best Model: {best_model}")
    print(f"🏆 Best Validation F1: {best_score:.4f}")
    
    for model_name, data in results.items():
        analysis = data['analysis']
        print(f"  {analysis['status_emoji']} {model_name}: Val F1 = {data['val_f1']:.4f}, Status = {analysis['status']}")
    
    # Save results
    model_saver.save_results(results, timestamp)
    
    return results

if __name__ == "__main__":
    results = test_enhanced_techniques()