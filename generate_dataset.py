"""
Generate synthetic dataset for e-commerce product orders
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_generation import generate_synthetic_dataset
import pandas as pd

def main():
    """Generate and save the dataset"""
    print("="*60)
    print("Generating Synthetic E-commerce Dataset")
    print("="*60)
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate dataset
    print("\nGenerating 10,000 samples...")
    df = generate_synthetic_dataset(n_samples=10000, random_seed=42)
    
    # Save to CSV
    output_path = 'data/raw/ecommerce_orders.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nReturn rate: {df['returned'].mean():.2%}")
    print(f"High risk rate: {df['risk_level'].mean():.2%}")
    
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset statistics:")
    print(df.describe())
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()

