#!/usr/bin/env python3
"""
PyTorch Neural Network Baseline Model for Heart Disease Risk Prediction
Optimized for Apple Silicon (MPS) with proper error handling and monitoring.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import time
from datetime import datetime
import json


class HeartDiseaseNN(nn.Module):
    """
    Neural Network architecture for heart disease risk prediction
    """
    def __init__(self, input_dim):
        super(HeartDiseaseNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


class PyTorchWrapper:
    """
    Sklearn-style wrapper for PyTorch model
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).cpu().numpy().astype(int).ravel()
        return predictions
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
            outputs = self.model(X_tensor)
            proba = torch.sigmoid(outputs).cpu().numpy().ravel()
            return np.column_stack([1 - proba, proba])
    
    def fit(self, X, y):
        return self


def setup_device():
    """
    Setup optimal device for Apple Silicon
    """
    print("🔥 SETTING UP PYTORCH DEVICE")
    print("=" * 50)
    
    # Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ Using Apple Silicon GPU acceleration (MPS)")
        # Additional optimizations for Apple Silicon
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using NVIDIA GPU acceleration (CUDA)")
    else:
        device = torch.device('cpu')
        print(f"⚠️ Using CPU (consider enabling GPU acceleration)")

    print(f"Device: {device}")
    return device


def load_data():
    """
    Load processed datasets
    """
    print("📥 LOADING PROCESSED DATASETS")
    print("=" * 50)
    
    # Load train/validation datasets
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")
    
    # Load feature metadata
    feature_names_df = pd.read_csv("data/processed/feature_names.csv")
    feature_names = feature_names_df['feature_name'].tolist()
    
    print(f"✅ Training set: {train_df.shape}")
    print(f"✅ Validation set: {val_df.shape}")
    print(f"✅ Features: {len(feature_names)} predictors")
    
    # Separate features and targets
    X_train = train_df[feature_names]
    y_train = train_df['hltprhc']
    X_val = val_df[feature_names]
    y_val = val_df['hltprhc']
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Features: {X_train.shape[1]}")
    
    # Check class distribution
    class_counts = np.bincount(y_train)
    print(f"Class distribution: {class_counts[0]} negative, {class_counts[1]} positive")
    
    return X_train, X_val, y_train, y_val, feature_names, class_counts


def create_data_loaders(X_train, X_val, y_train, y_val, device, batch_size=128):
    """
    Create PyTorch data loaders with smaller batch size for stability
    """
    print("🔄 CREATING DATA LOADERS")
    print("=" * 50)
    
    try:
        # Convert to PyTorch tensors - Use float32 for MPS compatibility
        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1)).to(device)
        X_val_tensor = torch.FloatTensor(X_val.values).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values.reshape(-1, 1)).to(device)
        
        print(f"✅ Tensors created successfully")
        print(f"Training tensor shape: {X_train_tensor.shape}")
        print(f"Validation tensor shape: {X_val_tensor.shape}")
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ Data loaders created with batch size: {batch_size}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        raise


def train_neural_network(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                        patience=5, max_epochs=50):
    """
    Train neural network with early stopping and progress monitoring
    Reduced epochs and patience for faster execution
    """
    print("🚀 STARTING NEURAL NETWORK TRAINING")
    print("=" * 50)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training configuration:")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Patience: {patience}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    
    start_time = time.time()
    
    try:
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0
            train_batches_processed = 0
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                try:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_batches_processed += 1
                    
                    # Progress update every 50 batches
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        avg_loss = train_loss / train_batches_processed
                        print(f"    Batch {batch_idx:3d}/{len(train_loader):3d}, Avg Loss: {avg_loss:.6f}")
                        
                except Exception as e:
                    print(f"❌ Error in training batch {batch_idx}: {e}")
                    raise
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_batches_processed = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    try:
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        val_batches_processed += 1
                    except Exception as e:
                        print(f"❌ Error in validation batch: {e}")
                        raise
            
            # Calculate average losses
            train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
            val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"    ✅ New best validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
            
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            
            print(f"Epoch {epoch+1:3d}/{max_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Time = {epoch_time:.1f}s, Total = {total_time:.1f}s")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience = {patience})")
                break
                
            # Safety check - if training is taking too long, stop
            if total_time > 1800:  # 30 minutes max
                print(f"⚠️ Training timeout after {total_time:.0f}s - stopping early")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"✅ Loaded best model state (val_loss: {best_val_loss:.6f})")
        
        total_training_time = time.time() - start_time
        print(f"✅ Training completed in {total_training_time:.1f}s")
        
        return train_losses, val_losses
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise


def evaluate_model(model, X_train, X_val, y_train, y_val, device):
    """
    Evaluate the trained model
    """
    print("📊 EVALUATING NEURAL NETWORK MODEL")
    print("=" * 50)
    
    # Create wrapper
    nn_wrapper = PyTorchWrapper(model, device)
    
    # Training predictions
    print("Getting training predictions...")
    y_pred_train_nn = nn_wrapper.predict(X_train)
    y_pred_proba_train_nn = nn_wrapper.predict_proba(X_train)[:, 1]
    
    # Validation predictions  
    print("Getting validation predictions...")
    y_pred_val_nn = nn_wrapper.predict(X_val)
    y_pred_proba_val_nn = nn_wrapper.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    nn_metrics = {
        'model_name': 'Neural Network',
        'cv_f1_mean': np.nan,  # Skip CV for NN
        'cv_f1_std': np.nan,
        'train_accuracy': accuracy_score(y_train, y_pred_train_nn),
        'train_precision': precision_score(y_train, y_pred_train_nn),
        'train_recall': recall_score(y_train, y_pred_train_nn),
        'train_f1': f1_score(y_train, y_pred_train_nn),
        'val_accuracy': accuracy_score(y_val, y_pred_val_nn),
        'val_precision': precision_score(y_val, y_pred_val_nn),
        'val_recall': recall_score(y_val, y_pred_val_nn),
        'val_f1': f1_score(y_val, y_pred_val_nn),
        'val_auc': roc_auc_score(y_val, y_pred_proba_val_nn)
    }
    
    # Print results
    print(f"📊 COMPREHENSIVE RESULTS:")
    print(f"Training   - Acc: {nn_metrics['train_accuracy']:.4f}, Prec: {nn_metrics['train_precision']:.4f}, Rec: {nn_metrics['train_recall']:.4f}, F1: {nn_metrics['train_f1']:.4f}")
    print(f"Validation - Acc: {nn_metrics['val_accuracy']:.4f}, Prec: {nn_metrics['val_precision']:.4f}, Rec: {nn_metrics['val_recall']:.4f}, F1: {nn_metrics['val_f1']:.4f}, AUC: {nn_metrics['val_auc']:.4f}")
    
    return nn_metrics, y_pred_val_nn, y_pred_proba_val_nn


def save_results(model, nn_metrics, y_pred_val_nn, y_pred_proba_val_nn, train_losses, val_losses, device):
    """
    Save model and results for loading in notebook
    """
    print("💾 SAVING RESULTS")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("results/models", exist_ok=True)
    
    # Save model state
    model_path = "results/models/neural_network_baseline.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'HeartDiseaseNN',
        'device': str(device),
        'training_complete': True
    }, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save metrics and predictions
    results_path = "results/models/neural_network_results.joblib"
    joblib.dump({
        'metrics': nn_metrics,
        'val_predictions': y_pred_val_nn,
        'val_probabilities': y_pred_proba_val_nn,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'timestamp': datetime.now().isoformat()
    }, results_path)
    print(f"✅ Results saved to: {results_path}")
    
    # Save summary
    summary_path = "results/models/neural_network_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("NEURAL NETWORK BASELINE MODEL RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write(f"Device used: {device}\n")
        f.write(f"Training time: {len(train_losses)} epochs\n\n")
        f.write("COMPREHENSIVE PERFORMANCE METRICS:\n")
        f.write("-" * 35 + "\n")
        f.write("TRAINING METRICS:\n")
        f.write(f"  Training Accuracy:  {nn_metrics['train_accuracy']:.4f}\n")
        f.write(f"  Training Precision: {nn_metrics['train_precision']:.4f}\n")
        f.write(f"  Training Recall:    {nn_metrics['train_recall']:.4f}\n")
        f.write(f"  Training F1-Score:  {nn_metrics['train_f1']:.4f}\n\n")
        f.write("VALIDATION METRICS:\n")
        f.write(f"  Validation Accuracy:  {nn_metrics['val_accuracy']:.4f}\n")
        f.write(f"  Validation Precision: {nn_metrics['val_precision']:.4f}\n")
        f.write(f"  Validation Recall:    {nn_metrics['val_recall']:.4f}\n")
        f.write(f"  Validation F1-Score:  {nn_metrics['val_f1']:.4f}\n")
        f.write(f"  Validation AUC:       {nn_metrics['val_auc']:.4f}\n\n")
        f.write("TRAINING SUMMARY:\n")
        f.write(f"  Final training loss:   {train_losses[-1]:.6f}\n")
        f.write(f"  Best validation loss:  {min(val_losses):.6f}\n")
    print(f"✅ Summary saved to: {summary_path}")


def main():
    """
    Main training pipeline
    """
    try:
        print("🔥 PYTORCH NEURAL NETWORK BASELINE TRAINING")
        print("=" * 60)
        print(f"Start time: {datetime.now()}")
        print()
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Setup device
        device = setup_device()
        
        # Apple Silicon specific seed
        if device.type == 'mps':
            torch.mps.manual_seed(42)
        
        print()
        
        # Load data
        X_train, X_val, y_train, y_val, feature_names, class_counts = load_data()
        print()
        
        # Create data loaders with smaller batch size for stability
        train_loader, val_loader = create_data_loaders(X_train, X_val, y_train, y_val, device, batch_size=128)
        print()
        
        # Initialize model
        print("🏗️ INITIALIZING MODEL")
        print("=" * 50)
        input_dim = X_train.shape[1]
        model = HeartDiseaseNN(input_dim).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Setup loss function and optimizer
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        print(f"Positive weight for class imbalance: {pos_weight.item():.2f}")
        print()
        
        # Train model with reduced epochs for faster execution
        train_losses, val_losses = train_neural_network(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            patience=5, max_epochs=30  # Reduced for faster execution
        )
        print()
        
        # Evaluate model
        nn_metrics, y_pred_val_nn, y_pred_proba_val_nn = evaluate_model(
            model, X_train, X_val, y_train, y_val, device
        )
        print()
        
        # Save results
        save_results(model, nn_metrics, y_pred_val_nn, y_pred_proba_val_nn, train_losses, val_losses, device)
        print()
        
        print("🎉 NEURAL NETWORK TRAINING COMPLETED SUCCESSFULLY!")
        print(f"End time: {datetime.now()}")
        
        return True
        
    except Exception as e:
        print(f"❌ TRAINING FAILED: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)