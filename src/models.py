"""
Machine learning models for risk and return prediction
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
from xgboost import XGBClassifier
import joblib

class RiskReturnPredictor:
    """Class to train and evaluate models for risk and return prediction"""
    
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize the predictor
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'logistic')
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.feature_importance_ = None
    
    def _create_model(self):
        """Create the specified model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                solver='liblinear'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For logistic regression, use absolute coefficients
            self.feature_importance_ = np.abs(self.model.coef_[0])
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics, y_pred, y_pred_proba
    
    def get_feature_importance(self, feature_names):
        """Get feature importance as a dictionary"""
        if self.feature_importance_ is None:
            return None
        
        importance_dict = dict(zip(feature_names, self.feature_importance_))
        return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    def save(self, filepath):
        """Save the model"""
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load(filepath):
        """Load a saved model"""
        return joblib.load(filepath)

def train_multiple_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train multiple models and compare their performance
    
    Returns:
        Dictionary with model results
    """
    models = ['random_forest', 'xgboost', 'logistic']
    results = {}
    
    for model_type in models:
        print(f"\nTraining {model_type}...")
        predictor = RiskReturnPredictor(model_type=model_type)
        predictor.train(X_train, y_train)
        
        metrics, y_pred, y_pred_proba = predictor.evaluate(X_test, y_test)
        feature_importance = predictor.get_feature_importance(feature_names)
        
        results[model_type] = {
            'predictor': predictor,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_importance': feature_importance
        }
        
        print(f"{model_type} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return results

