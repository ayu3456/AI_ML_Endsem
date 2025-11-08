"""
Main script to run the complete risk and return prediction pipeline
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.data_generation import generate_synthetic_dataset
from src.preprocessing import preprocess_data
from src.models import RiskReturnPredictor
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Run the complete pipeline"""
    print("="*80)
    print("RISK AND RETURN PREDICTION FOR E-COMMERCE PRODUCTS")
    print("="*80)
    
    # Step 1: Generate or load dataset
    print("\n[Step 1] Dataset Preparation")
    print("-" * 80)
    data_path = 'data/raw/ecommerce_orders.csv'
    
    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("Generating synthetic dataset...")
        os.makedirs('data/raw', exist_ok=True)
        df = generate_synthetic_dataset(n_samples=10000, random_seed=42)
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to {data_path}")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Return rate: {df['returned'].mean():.2%}")
    print(f"High risk rate: {df['risk_level'].mean():.2%}")
    
    # Step 2: Data exploration
    print("\n[Step 2] Data Exploration")
    print("-" * 80)
    
    # Create some basic visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Return rate by category
    return_by_category = df.groupby('product_category')['returned'].mean().sort_values(ascending=False)
    axes[0, 0].barh(return_by_category.index, return_by_category.values)
    axes[0, 0].set_xlabel('Return Rate')
    axes[0, 0].set_title('Return Rate by Product Category')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Risk score distribution
    axes[0, 1].hist(df['risk_score'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Risk Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Risk Score Distribution')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Return rate by price range
    df['price_range'] = pd.cut(df['product_price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    return_by_price = df.groupby('price_range')['returned'].mean()
    axes[1, 0].bar(return_by_price.index, return_by_price.values)
    axes[1, 0].set_ylabel('Return Rate')
    axes[1, 0].set_title('Return Rate by Price Range')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Correlation heatmap (selected features)
    numeric_cols = ['product_price', 'product_rating', 'num_reviews', 
                   'customer_age', 'customer_purchase_history', 
                   'quantity', 'total_order_value', 'risk_score', 'returned']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
    axes[1, 1].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/data_exploration.png', dpi=300, bbox_inches='tight')
    print("Saved: models/data_exploration.png")
    plt.close()
    
    # Remove temporary visualization column
    df = df.drop('price_range', axis=1, errors='ignore')
    
    # Step 3: Train return prediction model
    print("\n[Step 3] Training Return Prediction Model")
    print("-" * 80)
    X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders = \
        preprocess_data(df, target='returned', test_size=0.2, random_state=42)
    
    return_predictor = RiskReturnPredictor(model_type='random_forest', random_state=42)
    return_predictor.train(X_train, y_train)
    return_metrics, y_pred_return, y_pred_proba_return = return_predictor.evaluate(X_test, y_test)
    
    print("\nReturn Prediction Results:")
    print(f"  Accuracy:  {return_metrics['accuracy']:.4f}")
    print(f"  Precision: {return_metrics['precision']:.4f}")
    print(f"  Recall:    {return_metrics['recall']:.4f}")
    print(f"  F1-Score:  {return_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {return_metrics['roc_auc']:.4f}")
    
    # Step 4: Train risk prediction model
    print("\n[Step 4] Training Risk Prediction Model")
    print("-" * 80)
    X_train_risk, X_test_risk, y_train_risk, y_test_risk, feature_names_risk, scaler_risk, label_encoders_risk = \
        preprocess_data(df, target='risk_level', test_size=0.2, random_state=42)
    
    risk_predictor = RiskReturnPredictor(model_type='xgboost', random_state=42)
    risk_predictor.train(X_train_risk, y_train_risk)
    risk_metrics, y_pred_risk, y_pred_proba_risk = risk_predictor.evaluate(X_test_risk, y_test_risk)
    
    print("\nRisk Prediction Results:")
    print(f"  Accuracy:  {risk_metrics['accuracy']:.4f}")
    print(f"  Precision: {risk_metrics['precision']:.4f}")
    print(f"  Recall:    {risk_metrics['recall']:.4f}")
    print(f"  F1-Score:  {risk_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {risk_metrics['roc_auc']:.4f}")
    
    # Step 5: Feature importance
    print("\n[Step 5] Feature Importance Analysis")
    print("-" * 80)
    
    return_importance = return_predictor.get_feature_importance(feature_names)
    risk_importance = risk_predictor.get_feature_importance(feature_names_risk)
    
    print("\nTop 10 Features for Return Prediction:")
    for feature, importance in return_importance[:10]:
        print(f"  {feature:30s}: {importance:.4f}")
    
    print("\nTop 10 Features for Risk Prediction:")
    for feature, importance in risk_importance[:10]:
        print(f"  {feature:30s}: {importance:.4f}")
    
    # Step 6: Save models
    print("\n[Step 6] Saving Models")
    print("-" * 80)
    os.makedirs('models', exist_ok=True)
    return_predictor.save('models/return_predictor.joblib')
    risk_predictor.save('models/risk_predictor.joblib')
    print("Models saved to models/ directory")
    
    # Step 7: Create prediction example
    print("\n[Step 7] Example Predictions")
    print("-" * 80)
    
    # Select a few test samples
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    print("\nSample Predictions:")
    print("-" * 80)
    for idx in sample_indices:
        return_prob = return_predictor.predict_proba(X_test.iloc[[idx]])[0, 1]
        risk_prob = risk_predictor.predict_proba(X_test_risk.iloc[[idx]])[0, 1]
        actual_return = y_test.iloc[idx]
        actual_risk = y_test_risk.iloc[idx]
        
        print(f"\nSample {idx+1}:")
        print(f"  Predicted Return Probability: {return_prob:.2%}")
        print(f"  Predicted Risk Probability:  {risk_prob:.2%}")
        print(f"  Actual Returned: {'Yes' if actual_return == 1 else 'No'}")
        print(f"  Actual Risk Level: {'High' if actual_risk == 1 else 'Low'}")
    
    print("\n" + "="*80)
    print("Pipeline Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check models/ directory for saved models and visualizations")
    print("  2. Run train_models.py for detailed model comparison")
    print("  3. Use the saved models for predictions on new data")

if __name__ == "__main__":
    main()

