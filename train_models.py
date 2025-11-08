"""
Train models for risk and return prediction
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.data_generation import generate_synthetic_dataset
from src.preprocessing import preprocess_data
from src.models import train_multiple_models
from src.evaluation import (plot_confusion_matrix, plot_roc_curve, 
                           plot_feature_importance, plot_model_comparison,
                           print_evaluation_summary)
import matplotlib.pyplot as plt

def train_and_evaluate(target='returned', save_models=True):
    """
    Train and evaluate models for the specified target
    
    Args:
        target: 'returned' or 'risk_level'
        save_models: Whether to save trained models
    """
    print("="*80)
    print(f"Training Models for {target.upper()} Prediction")
    print("="*80)
    
    # Load or generate dataset
    data_path = 'data/raw/ecommerce_orders.csv'
    if os.path.exists(data_path):
        print(f"\nLoading dataset from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("\nDataset not found. Generating new dataset...")
        df = generate_synthetic_dataset(n_samples=10000, random_seed=42)
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv(data_path, index=False)
    
    print(f"Dataset loaded: {len(df)} samples")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders = \
        preprocess_data(df, target=target, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(feature_names)}")
    
    # Train models
    print("\n" + "="*80)
    print("Training Models")
    print("="*80)
    results = train_multiple_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Create visualizations
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    os.makedirs('models', exist_ok=True)
    
    # Model comparison
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    plot_model_comparison(results, metric='accuracy')
    plt.savefig(f'models/{target}_model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: models/{target}_model_comparison.png")
    
    # Detailed plots for each model
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 3, figsize=(18, 6*n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, result) in enumerate(results.items()):
        # Confusion matrix
        plot_confusion_matrix(y_test, result['y_pred'], model_name, ax=axes[idx, 0])
        
        # ROC curve
        plot_roc_curve(y_test, result['y_pred_proba'], model_name, ax=axes[idx, 1])
        
        # Feature importance
        if result['feature_importance']:
            plot_feature_importance(result['feature_importance'], model_name, 
                                  top_n=10, ax=axes[idx, 2])
    
    plt.tight_layout()
    plt.savefig(f'models/{target}_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: models/{target}_detailed_analysis.png")
    plt.close()
    
    # Save best model
    if save_models:
        # Find best model by F1 score
        best_model_name = max(results.keys(), 
                             key=lambda x: results[x]['metrics']['f1_score'])
        best_model = results[best_model_name]['predictor']
        
        model_path = f'models/{target}_best_model_{best_model_name}.joblib'
        best_model.save(model_path)
        print(f"\nBest model ({best_model_name}) saved to: {model_path}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    return results

if __name__ == "__main__":
    # Train models for return prediction
    print("\n" + "="*80)
    print("RETURN PREDICTION MODELS")
    print("="*80)
    return_results = train_and_evaluate(target='returned', save_models=True)
    
    # Train models for risk prediction
    print("\n\n" + "="*80)
    print("RISK PREDICTION MODELS")
    print("="*80)
    risk_results = train_and_evaluate(target='risk_level', save_models=True)

