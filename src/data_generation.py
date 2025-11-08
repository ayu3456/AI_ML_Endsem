"""
Synthetic dataset generation for e-commerce product orders
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_dataset(n_samples=10000, random_seed=42):
    """
    Generate synthetic dataset for e-commerce product orders
    
    Args:
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic order data
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Product categories
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 
                  'Sports', 'Toys', 'Beauty', 'Automotive', 'Food', 'Furniture']
    
    # Shipping methods
    shipping_methods = ['Standard', 'Express', 'Overnight', 'Free Shipping']
    
    # Customer locations (regions)
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Payment methods
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash on Delivery']
    
    data = []
    
    for i in range(n_samples):
        # Product features
        category = np.random.choice(categories)
        base_price = np.random.uniform(10, 500)
        
        # Price varies by category
        if category == 'Electronics':
            price = np.random.uniform(50, 1000)
        elif category == 'Clothing':
            price = np.random.uniform(10, 200)
        elif category == 'Furniture':
            price = np.random.uniform(100, 2000)
        else:
            price = base_price
        
        # Product rating (affects return probability)
        rating = np.random.uniform(2.5, 5.0)
        
        # Number of reviews
        num_reviews = int(np.random.exponential(100))
        
        # Customer features
        customer_age = np.random.randint(18, 80)
        customer_purchase_history = np.random.poisson(5)  # Previous purchases
        region = np.random.choice(regions)
        
        # Order features
        quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.05, 0.05])
        shipping_method = np.random.choice(shipping_methods, 
                                          p=[0.4, 0.3, 0.1, 0.2])
        payment_method = np.random.choice(payment_methods,
                                         p=[0.4, 0.3, 0.2, 0.1])
        
        # Time features
        order_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        month = order_date.month
        day_of_week = order_date.weekday()
        hour = np.random.randint(0, 24)
        
        # Calculate total order value
        total_value = price * quantity
        
        # Risk factors (affect return probability)
        # Lower rating = higher risk
        # Electronics and Furniture have higher return rates
        # Higher price = higher risk
        # Cash on delivery = higher risk
        
        risk_score = 0
        if rating < 3.5:
            risk_score += 2
        if category in ['Electronics', 'Furniture', 'Clothing']:
            risk_score += 1
        if price > 200:
            risk_score += 1
        if payment_method == 'Cash on Delivery':
            risk_score += 1
        if shipping_method == 'Overnight':
            risk_score += 0.5
        if customer_purchase_history < 2:
            risk_score += 1
        
        # Normalize risk score to 0-10
        risk_score = min(risk_score, 10)
        
        # Return probability based on risk factors
        base_return_prob = 0.15  # Base return rate
        
        # Adjust based on factors
        if rating < 3.0:
            base_return_prob += 0.25
        if category in ['Electronics', 'Clothing']:
            base_return_prob += 0.10
        if price > 300:
            base_return_prob += 0.15
        if quantity > 2:
            base_return_prob += 0.05
        if customer_purchase_history == 0:
            base_return_prob += 0.10
        if payment_method == 'Cash on Delivery':
            base_return_prob += 0.15
        
        # Add some randomness
        return_prob = min(base_return_prob + np.random.uniform(-0.1, 0.1), 0.95)
        returned = 1 if np.random.random() < return_prob else 0
        
        # Risk level (High Risk if risk_score > 5 or returned)
        risk_level = 1 if (risk_score > 5 or returned == 1) else 0
        
        # Days to return (if returned)
        days_to_return = None
        if returned == 1:
            days_to_return = np.random.randint(1, 30)
        
        data.append({
            'order_id': f'ORD_{i+1:06d}',
            'product_category': category,
            'product_price': round(price, 2),
            'product_rating': round(rating, 2),
            'num_reviews': num_reviews,
            'customer_age': customer_age,
            'customer_purchase_history': customer_purchase_history,
            'customer_region': region,
            'quantity': quantity,
            'total_order_value': round(total_value, 2),
            'shipping_method': shipping_method,
            'payment_method': payment_method,
            'order_month': month,
            'order_day_of_week': day_of_week,
            'order_hour': hour,
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'returned': returned,
            'days_to_return': days_to_return
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_samples=10000)
    print(f"Generated {len(df)} samples")
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())
    print("\nReturn rate:", df['returned'].mean())
    print("Risk level distribution:")
    print(df['risk_level'].value_counts())

