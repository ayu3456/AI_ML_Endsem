"""
Generate comprehensive results file with all metrics and statistics
"""
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.kaggle_data_loader import load_kaggle_dataset
from src.preprocessing import preprocess_data
from src.models import train_multiple_models, RiskReturnPredictor
import numpy as np

def generate_results_file():
    """Generate comprehensive results file"""
    
    print("Generating comprehensive results file...")
    
    # Load Kaggle dataset
    kaggle_path = 'data/raw/ecommerce_returns_kaggle.csv'
    if not os.path.exists(kaggle_path):
        print(f"ERROR: Kaggle dataset not found at {kaggle_path}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/...")
        return None
    
    print(f"Loading Kaggle dataset from {kaggle_path}...")
    df = load_kaggle_dataset(kaggle_path)
    
    # Get return prediction results
    print("Training return prediction models...")
    X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders = \
        preprocess_data(df, target='returned', test_size=0.2, random_state=42)
    return_results = train_multiple_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Get risk prediction results
    print("Training risk prediction models...")
    X_train_risk, X_test_risk, y_train_risk, y_test_risk, feature_names_risk, scaler_risk, label_encoders_risk = \
        preprocess_data(df, target='risk_level', test_size=0.2, random_state=42)
    risk_results = train_multiple_models(X_train_risk, X_test_risk, y_train_risk, y_test_risk, feature_names_risk)
    
    # Generate results text
    results_text = []
    results_text.append("="*80)
    results_text.append("RISK AND RETURN PREDICTION - COMPREHENSIVE RESULTS")
    results_text.append("="*80)
    results_text.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append("\n")
    
    # Dataset Overview
    results_text.append("="*80)
    results_text.append("1. DATASET OVERVIEW")
    results_text.append("="*80)
    results_text.append(f"\nTotal Samples: {len(df):,}")
    results_text.append(f"Total Features: {len(df.columns)}")
    results_text.append(f"Return Rate: {df['returned'].mean():.2%}")
    results_text.append(f"High Risk Rate: {df['risk_level'].mean():.2%}")
    results_text.append(f"\nTraining Set: {len(X_train):,} samples")
    results_text.append(f"Test Set: {len(X_test):,} samples")
    results_text.append(f"Features Used: {len(feature_names)}")
    
    # Return Prediction Results
    results_text.append("\n" + "="*80)
    results_text.append("2. RETURN PREDICTION RESULTS")
    results_text.append("="*80)
    
    for model_name, result in return_results.items():
        metrics = result['metrics']
        results_text.append(f"\n{model_name.upper()} MODEL:")
        results_text.append(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        results_text.append(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        results_text.append(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        results_text.append(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        results_text.append(f"  ROC-AUC:   {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
        results_text.append(f"\n  Confusion Matrix:")
        cm = metrics['confusion_matrix']
        results_text.append(f"    True Negatives:  {cm[0][0]}")
        results_text.append(f"    False Positives: {cm[0][1]}")
        results_text.append(f"    False Negatives: {cm[1][0]}")
        results_text.append(f"    True Positives:  {cm[1][1]}")
        
        if result['feature_importance']:
            results_text.append(f"\n  Top 10 Feature Importance:")
            for i, (feature, importance) in enumerate(result['feature_importance'][:10], 1):
                results_text.append(f"    {i:2d}. {feature:30s}: {importance:.4f} ({importance*100:.2f}%)")
    
    # Find best model for returns
    best_return_model = max(return_results.keys(), 
                           key=lambda x: return_results[x]['metrics']['f1_score'])
    results_text.append(f"\n\nBEST MODEL FOR RETURN PREDICTION: {best_return_model.upper()}")
    best_return_metrics = return_results[best_return_model]['metrics']
    results_text.append(f"  F1-Score: {best_return_metrics['f1_score']:.4f}")
    results_text.append(f"  Accuracy: {best_return_metrics['accuracy']:.4f}")
    
    # Risk Prediction Results
    results_text.append("\n" + "="*80)
    results_text.append("3. RISK PREDICTION RESULTS")
    results_text.append("="*80)
    
    for model_name, result in risk_results.items():
        metrics = result['metrics']
        results_text.append(f"\n{model_name.upper()} MODEL:")
        results_text.append(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        results_text.append(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        results_text.append(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        results_text.append(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        results_text.append(f"  ROC-AUC:   {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
        results_text.append(f"\n  Confusion Matrix:")
        cm = metrics['confusion_matrix']
        results_text.append(f"    True Negatives:  {cm[0][0]}")
        results_text.append(f"    False Positives: {cm[0][1]}")
        results_text.append(f"    False Negatives: {cm[1][0]}")
        results_text.append(f"    True Positives:  {cm[1][1]}")
        
        if result['feature_importance']:
            results_text.append(f"\n  Top 10 Feature Importance:")
            for i, (feature, importance) in enumerate(result['feature_importance'][:10], 1):
                results_text.append(f"    {i:2d}. {feature:30s}: {importance:.4f} ({importance*100:.2f}%)")
    
    # Find best model for risk
    best_risk_model = max(risk_results.keys(), 
                         key=lambda x: risk_results[x]['metrics']['f1_score'])
    results_text.append(f"\n\nBEST MODEL FOR RISK PREDICTION: {best_risk_model.upper()}")
    best_risk_metrics = risk_results[best_risk_model]['metrics']
    results_text.append(f"  F1-Score: {best_risk_metrics['f1_score']:.4f}")
    results_text.append(f"  Accuracy: {best_risk_metrics['accuracy']:.4f}")
    
    # Dataset Statistics
    results_text.append("\n" + "="*80)
    results_text.append("4. DATASET STATISTICS")
    results_text.append("="*80)
    results_text.append("\nNumerical Features Summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['returned', 'risk_level']:
            results_text.append(f"\n  {col}:")
            results_text.append(f"    Mean:   {df[col].mean():.2f}")
            results_text.append(f"    Std:    {df[col].std():.2f}")
            results_text.append(f"    Min:    {df[col].min():.2f}")
            results_text.append(f"    Max:    {df[col].max():.2f}")
    
    results_text.append("\n\nCategorical Features Distribution:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        results_text.append(f"\n  {col}:")
        value_counts = df[col].value_counts()
        for value, count in value_counts.head(10).items():
            results_text.append(f"    {value}: {count} ({count/len(df)*100:.2f}%)")
    
    # Example Predictions
    results_text.append("\n" + "="*80)
    results_text.append("5. EXAMPLE PREDICTIONS")
    results_text.append("="*80)
    
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    best_return_predictor = return_results[best_return_model]['predictor']
    best_risk_predictor = risk_results[best_risk_model]['predictor']
    
    results_text.append(f"\nSample predictions from test set (showing 10 random samples):")
    results_text.append(f"\n{'Sample':<10} {'Return Prob':<15} {'Risk Prob':<15} {'Actual Return':<15} {'Actual Risk':<15} {'Return Correct':<15} {'Risk Correct':<15}")
    results_text.append("-" * 100)
    
    correct_return = 0
    correct_risk = 0
    
    for idx in sample_indices:
        return_prob = best_return_predictor.predict_proba(X_test.iloc[[idx]])[0, 1]
        risk_prob = best_risk_predictor.predict_proba(X_test_risk.iloc[[idx]])[0, 1]
        actual_return = y_test.iloc[idx]
        actual_risk = y_test_risk.iloc[idx]
        pred_return = 1 if return_prob > 0.5 else 0
        pred_risk = 1 if risk_prob > 0.5 else 0
        
        return_correct = "✓" if pred_return == actual_return else "✗"
        risk_correct = "✓" if pred_risk == actual_risk else "✗"
        
        if pred_return == actual_return:
            correct_return += 1
        if pred_risk == actual_risk:
            correct_risk += 1
        
        results_text.append(f"{idx:<10} {return_prob*100:>6.2f}%{'':<6} {risk_prob*100:>6.2f}%{'':<6} {'Yes' if actual_return==1 else 'No':<15} {'High' if actual_risk==1 else 'Low':<15} {return_correct:<15} {risk_correct:<15}")
    
    results_text.append(f"\nAccuracy on these samples:")
    results_text.append(f"  Return Prediction: {correct_return}/10 ({correct_return*10}%)")
    results_text.append(f"  Risk Prediction: {correct_risk}/10 ({correct_risk*10}%)")
    
    # Summary
    results_text.append("\n" + "="*80)
    results_text.append("6. SUMMARY")
    results_text.append("="*80)
    results_text.append(f"\nBest Return Prediction Model: {best_return_model.upper()}")
    results_text.append(f"  - F1-Score: {best_return_metrics['f1_score']:.4f}")
    results_text.append(f"  - Accuracy: {best_return_metrics['accuracy']:.4f}")
    results_text.append(f"  - ROC-AUC: {best_return_metrics['roc_auc']:.4f}")
    
    results_text.append(f"\nBest Risk Prediction Model: {best_risk_model.upper()}")
    results_text.append(f"  - F1-Score: {best_risk_metrics['f1_score']:.4f}")
    results_text.append(f"  - Accuracy: {best_risk_metrics['accuracy']:.4f}")
    results_text.append(f"  - ROC-AUC: {best_risk_metrics['roc_auc']:.4f}")
    
    results_text.append("\n" + "="*80)
    results_text.append("END OF RESULTS")
    results_text.append("="*80)
    
    # Write to file
    output_file = 'RESULTS_DETAILED.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(results_text))
    
    print(f"\nDetailed results saved to: {output_file}")
    print(f"Total lines: {len(results_text)}")
    
    return output_file

if __name__ == "__main__":
    generate_results_file()

