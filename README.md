# Risk and Return Prediction for E-commerce Products

This project predicts the risk and return probability of products ordered from an Amazon-like e-commerce platform using machine learning.

## Project Overview

The project includes:
- **Risk Prediction**: Predicts the likelihood of a product order having issues (returns, complaints, refunds)
- **Return Prediction**: Predicts the probability of a product being returned

## Features

- Synthetic dataset generation for e-commerce orders
- Feature engineering and data preprocessing
- Multiple ML models (Random Forest, XGBoost, Logistic Regression)
- Model evaluation and visualization
- Risk scoring system

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Dataset
```bash
python generate_dataset.py
```

### Train Models
```bash
python train_models.py
```

### Run Complete Pipeline
```bash
jupyter notebook main_analysis.ipynb
```

Or run the main script:
```bash
python main.py
```

## Project Structure

```
RiskReturnProject/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── main_analysis.ipynb
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── generate_dataset.py
├── train_models.py
├── main.py
├── requirements.txt
└── README.md
```

## Dataset Features

The dataset includes:
- Product features (price, category, rating, reviews)
- Customer features (age, purchase history, location)
- Order features (quantity, shipping method, time of day)
- Seasonal features (month, day of week)
- Target variables (return_status, risk_score)

## Models

- **Risk Prediction**: Binary classification (High Risk / Low Risk)
- **Return Prediction**: Binary classification (Returned / Not Returned)

## Results

Models are evaluated using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Feature importance analysis
- Confusion matrices

