"""
Data loader for Kaggle e-commerce returns dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime

def load_kaggle_dataset(filepath='data/raw/ecommerce_returns_kaggle.csv'):
    """
    Load and preprocess the Kaggle e-commerce returns dataset
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        Preprocessed DataFrame compatible with our models
    """
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Convert dates
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df['Return_Date'] = pd.to_datetime(df['Return_Date'], errors='coerce')
    
    # Extract temporal features from Order_Date
    df['order_month'] = df['Order_Date'].dt.month
    df['order_day_of_week'] = df['Order_Date'].dt.dayofweek
    df['order_hour'] = np.random.randint(0, 24, len(df))  # Not available in dataset, use random
    
    # Create return status binary (1 = Returned, 0 = Not Returned)
    df['returned'] = (df['Return_Status'] == 'Returned').astype(int)
    
    # Calculate days to return (fill NaN with 0 for non-returned items)
    df['Days_to_Return'] = df['Days_to_Return'].fillna(0)
    
    # Create risk score based on available features (before renaming columns)
    df['risk_score'] = calculate_risk_score(df)
    
    # Create risk level (High Risk = 1, Low Risk = 0)
    # High risk if: returned, high price, certain categories, certain payment methods
    df['risk_level'] = (df['risk_score'] > 5).astype(int)
    
    # Calculate total order value
    df['total_order_value'] = df['Product_Price'] * df['Order_Quantity']
    
    # Rename columns to match our existing code structure
    column_mapping = {
        'Product_Category': 'product_category',
        'Product_Price': 'product_price',
        'Order_Quantity': 'quantity',
        'User_Age': 'customer_age',
        'User_Location': 'customer_region',
        'Payment_Method': 'payment_method',
        'Shipping_Method': 'shipping_method',
        'Discount_Applied': 'discount_applied',
        'Order_ID': 'order_id',
        'Product_ID': 'product_id',
        'User_ID': 'user_id',
        'Return_Reason': 'return_reason',
        'User_Gender': 'user_gender'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Add missing features that our models expect (with default/calculated values)
    # Product rating - not available, estimate based on return rate and category
    # Need to calculate before renaming, so use original column names
    temp_df = df.copy()
    df['product_rating'] = estimate_product_rating(temp_df)
    
    # Number of reviews - not available, use random or estimate
    df['num_reviews'] = np.random.randint(10, 500, len(df))
    
    # Customer purchase history - not available, estimate based on user_id frequency
    user_order_counts = df['user_id'].value_counts()
    df['customer_purchase_history'] = df['user_id'].map(user_order_counts) - 1  # -1 because current order doesn't count
    
    # Select and reorder columns to match expected structure
    final_columns = [
        'order_id', 'product_category', 'product_price', 'product_rating', 
        'num_reviews', 'customer_age', 'customer_purchase_history', 
        'customer_region', 'quantity', 'total_order_value', 
        'shipping_method', 'payment_method', 'order_month', 
        'order_day_of_week', 'order_hour', 'risk_score', 'risk_level', 
        'returned', 'Days_to_Return', 'discount_applied', 'return_reason', 'user_gender'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in final_columns if col in df.columns]
    df = df[available_columns]
    
    return df

def calculate_risk_score(df):
    """
    Calculate risk score based on available features
    """
    risk_score = np.zeros(len(df))
    
    # Higher price = higher risk
    price_median = df['Product_Price'].median()
    risk_score += (df['Product_Price'] > price_median).astype(int) * 1.5
    
    # Certain categories have higher return rates
    high_risk_categories = ['Electronics', 'Clothing']
    risk_score += df['Product_Category'].isin(high_risk_categories).astype(int) * 1
    
    # Higher quantity = higher risk
    risk_score += (df['Order_Quantity'] > 2).astype(int) * 0.5
    
    # Certain payment methods = higher risk
    high_risk_payment = ['Cash on Delivery', 'Gift Card']
    if 'Payment_Method' in df.columns:
        risk_score += df['Payment_Method'].isin(high_risk_payment).astype(int) * 1
    
    # If already returned, high risk
    if 'Return_Status' in df.columns:
        risk_score += (df['Return_Status'] == 'Returned').astype(int) * 2
    
    # Higher discount might indicate clearance items = higher risk
    if 'Discount_Applied' in df.columns:
        risk_score += (df['Discount_Applied'] > 30).astype(int) * 0.5
    
    # Normalize to 0-10 scale
    risk_score = np.clip(risk_score, 0, 10)
    
    return risk_score

def estimate_product_rating(df):
    """
    Estimate product rating based on return rate and category
    Higher return rate = lower rating
    """
    # Base rating
    rating = np.full(len(df), 3.5)
    
    # Adjust based on return status (calculate before renaming)
    if 'Return_Status' in df.columns:
        returned_mask = df['Return_Status'] == 'Returned'
        rating[returned_mask] -= 0.8  # Returned items likely have lower ratings
    
    # Adjust based on category (some categories have naturally higher/lower ratings)
    category_adjustments = {
        'Electronics': -0.2,
        'Clothing': -0.1,
        'Books': 0.1,
        'Home': 0.0,
        'Toys': 0.0
    }
    
    category_col = 'Product_Category' if 'Product_Category' in df.columns else 'product_category'
    for category, adjustment in category_adjustments.items():
        mask = df[category_col] == category
        rating[mask] += adjustment
    
    # Add some randomness
    rating += np.random.normal(0, 0.3, len(df))
    
    # Clip to valid range
    rating = np.clip(rating, 1.0, 5.0)
    
    return rating

