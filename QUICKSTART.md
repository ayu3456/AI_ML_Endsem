# Quick Start Guide

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Or if you prefer using a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Project

### Option 1: Run the Complete Pipeline
```bash
python3 main.py
```
This will:
- Generate or load the dataset
- Perform data exploration
- Train both return and risk prediction models
- Create visualizations
- Save trained models

### Option 2: Generate Dataset Only
```bash
python3 generate_dataset.py
```
This generates a synthetic dataset with 10,000 samples and saves it to `data/raw/ecommerce_orders.csv`.

### Option 3: Train Models Only
```bash
python3 train_models.py
```
This trains multiple models (Random Forest, XGBoost, Logistic Regression) for both return and risk prediction, and creates detailed evaluation visualizations.

### Option 4: Use Jupyter Notebook
```bash
jupyter notebook notebooks/main_analysis.ipynb
```
This provides an interactive environment to explore the data and models.

## Project Outputs

After running the scripts, you'll find:

- **Dataset**: `data/raw/ecommerce_orders.csv`
- **Trained Models**: `models/return_predictor.joblib`, `models/risk_predictor.joblib`
- **Visualizations**: 
  - `models/data_exploration.png`
  - `models/returned_model_comparison.png`
  - `models/returned_detailed_analysis.png`
  - `models/risk_level_model_comparison.png`
  - `models/risk_level_detailed_analysis.png`

## Dataset Features

The synthetic dataset includes:
- **Product Features**: category, price, rating, number of reviews
- **Customer Features**: age, purchase history, region
- **Order Features**: quantity, shipping method, payment method, total value
- **Temporal Features**: month, day of week, hour
- **Target Variables**: return status, risk level, risk score

## Model Performance

The models predict:
1. **Return Probability**: Likelihood of a product being returned
2. **Risk Level**: High/Low risk classification

Models are evaluated using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Feature importance analysis
- Confusion matrices

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python3 main.py`
3. Explore the results in the `models/` directory
4. Use the Jupyter notebook for interactive analysis

