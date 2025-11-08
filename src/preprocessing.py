"""
Data preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df, target='returned', test_size=0.2, random_state=42):
    """
    Preprocess the dataset for machine learning
    
    Args:
        df: Input DataFrame
        target: Target variable name ('returned' or 'risk_level')
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Drop unnecessary columns and columns that cause data leakage
    # return_reason and Days_to_Return are only known AFTER return happens
    df = df.drop(['order_id', 'days_to_return', 'Days_to_Return', 
                  'return_reason', 'product_id', 'user_id'], axis=1, errors='ignore')
    
    # Separate features and target
    if target == 'returned':
        y = df['returned']
        X = df.drop(['returned', 'risk_level', 'risk_score'], axis=1)
    elif target == 'risk_level':
        y = df['risk_level']
        X = df.drop(['returned', 'risk_level', 'risk_score'], axis=1)
    else:
        raise ValueError("Target must be 'returned' or 'risk_level'")
    
    # Feature engineering
    X = create_features(X)
    
    # Encode categorical variables
    # Include both object and category dtypes
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            X_train_scaled.columns.tolist(), scaler, label_encoders)

def create_features(df):
    """
    Create additional features from existing ones
    """
    df = df.copy()
    
    # Price per unit (if quantity is available)
    if 'quantity' in df.columns and 'product_price' in df.columns:
        df['price_per_unit'] = df['product_price'] / df['quantity']
    
    # High value order flag
    if 'total_order_value' in df.columns:
        df['high_value_order'] = (df['total_order_value'] > df['total_order_value'].quantile(0.75)).astype(int)
    
    # Low rating flag
    if 'product_rating' in df.columns:
        df['low_rating'] = (df['product_rating'] < 3.5).astype(int)
    
    # New customer flag
    if 'customer_purchase_history' in df.columns:
        df['new_customer'] = (df['customer_purchase_history'] == 0).astype(int)
    
    # Review density (reviews per rating point)
    if 'num_reviews' in df.columns and 'product_rating' in df.columns:
        df['review_density'] = df['num_reviews'] / (df['product_rating'] + 1)
    
    # Peak hours (9 AM - 5 PM)
    if 'order_hour' in df.columns:
        df['peak_hours'] = ((df['order_hour'] >= 9) & (df['order_hour'] <= 17)).astype(int)
    
    # Weekend flag
    if 'order_day_of_week' in df.columns:
        df['weekend'] = (df['order_day_of_week'] >= 5).astype(int)
    
    # Holiday season (Nov-Dec)
    if 'order_month' in df.columns:
        df['holiday_season'] = df['order_month'].isin([11, 12]).astype(int)
    
    return df

