"""
Model evaluation and visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

def plot_confusion_matrix(y_true, y_pred, model_name, ax=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Returned', 'Returned'],
                yticklabels=['Not Returned', 'Returned'])
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    return ax

def plot_roc_curve(y_true, y_pred_proba, model_name, ax=None):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_feature_importance(feature_importance, model_name, top_n=15, ax=None):
    """Plot feature importance"""
    if feature_importance is None:
        return None
    
    # Get top N features
    top_features = feature_importance[:top_n]
    features, importances = zip(*top_features)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax

def plot_model_comparison(results, metric='f1_score'):
    """Compare models using a bar chart"""
    model_names = list(results.keys())
    metric_values = [results[m]['metrics'][metric] for m in model_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, metric_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, ax

def create_evaluation_report(results, target_name='Return'):
    """Create a comprehensive evaluation report"""
    fig = plt.figure(figsize=(16, 12))
    
    # Model comparison
    ax1 = plt.subplot(2, 3, 1)
    plot_model_comparison(results, metric='accuracy')
    plt.title('Model Accuracy Comparison')
    
    ax2 = plt.subplot(2, 3, 2)
    plot_model_comparison(results, metric='f1_score')
    plt.title('Model F1-Score Comparison')
    
    ax3 = plt.subplot(2, 3, 3)
    plot_model_comparison(results, metric='roc_auc')
    plt.title('Model ROC-AUC Comparison')
    
    # Confusion matrices - Note: This function needs y_true and y_pred, not cm directly
    # This is a placeholder - actual confusion matrices should be plotted separately
    
    plt.tight_layout()
    return fig

def print_evaluation_summary(results):
    """Print a summary of all model evaluations"""
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\n  Top 5 Important Features:")
        if result['feature_importance']:
            for feature, importance in result['feature_importance'][:5]:
                print(f"    - {feature}: {importance:.4f}")
    
    print("\n" + "="*80)

