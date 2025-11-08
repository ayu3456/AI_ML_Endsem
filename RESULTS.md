# Risk and Return Prediction - Project Results

**Project Date:** Generated on execution  
**Dataset:** Kaggle E-commerce Returns Synthetic Data (10,000 orders)  
**Models Trained:** Random Forest, XGBoost, Logistic Regression

---

## 1. Dataset Overview

### Dataset Statistics
- **Total Samples:** 10,000 orders
- **Total Features:** 24 (after feature engineering)
- **Return Rate:** 50.52% (5,052 returned, 4,948 not returned)
- **High Risk Rate:** 5.53%

### Dataset Features
- **Product Features:** category, price, quantity, discount applied
- **Customer Features:** age, gender, location, purchase history (estimated)
- **Order Features:** payment method, shipping method, order date
- **Temporal Features:** month, day of week, hour
- **Estimated Features:** product rating, number of reviews (estimated from patterns)
- **Target Variables:** return status, risk level, risk score

---

## 2. Return Prediction Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **89.81%** | 89.81% | 87.39% | 88.58% | **99.73%** |
| **XGBoost** | 89.70% | 89.70% | **87.39%** | **88.54%** | 99.72% |
| **Logistic Regression** | 88.35% | 88.35% | 85.22% | 86.76% | 98.95% |

**Best Model for Return Prediction:** Random Forest (Best ROC-AUC: 99.73%)

### Feature Importance - Return Prediction

#### Random Forest Model:
1. **product_rating** - 59.98% (Most Important!)
2. **low_rating** - 21.36%
3. **review_density** - 5.10%
4. **num_reviews** - 2.28%
5. **discount_applied** - 1.28%
6. **total_order_value** - 1.19%
7. **product_price** - 1.18%
8. **product_category** - 1.14%
9. **price_per_unit** - 1.13%
10. **customer_region** - 1.07%

#### XGBoost Model:
1. **product_rating** - 56.99%
2. **low_rating** - 23.92%
3. **review_density** - 5.10%
4. **num_reviews** - 2.34%
5. **discount_applied** - 1.30%

#### Logistic Regression Model:
1. **product_rating** - High importance
2. **low_rating** - High importance
3. **review_density** - Moderate importance
4. **num_reviews** - Moderate importance
5. **discount_applied** - Moderate importance

---

## 3. Risk Prediction Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | 96.35% | 93.18% | 36.94% | 52.90% | 97.94% |
| **XGBoost** | **98.60%** | 90.29% | **83.78%** | **86.92%** | **99.24%** |
| **Logistic Regression** | 94.60% | 54.29% | 17.12% | 26.03% | 91.17% |

**Best Model for Risk Prediction:** XGBoost (Best F1-Score: 86.92%, Best ROC-AUC: 99.24%)

### Feature Importance - Risk Prediction

#### Random Forest Model:
1. **product_rating** - 17.31%
2. **discount_applied** - 15.03%
3. **payment_method** - 9.23%
4. **total_order_value** - 8.43%
5. **product_price** - 8.29%
6. **product_category** - 7.85%
7. **quantity** - 6.12%
8. **price_per_unit** - 4.23%
9. **num_reviews** - 3.45%
10. **order_month** - 2.98%

#### XGBoost Model:
1. **product_price** - 26.92%
2. **product_category** - 13.28%
3. **payment_method** - 13.05%
4. **quantity** - 12.83%
5. **product_rating** - 12.13%
6. **discount_applied** - 7.30%
7. **price_per_unit** - 1.83%
8. **num_reviews** - 1.53%
9. **total_order_value** - 1.45%
10. **order_month** - 1.44%

---

## 4. Key Insights

### Most Important Features Across All Models

1. **Product Rating** - Dominates return prediction (60% importance) - Most critical factor
2. **Product Price** - Most important for risk prediction (27% importance)
3. **Product Category** - Significant factor in both predictions (13% for risk)
4. **Payment Method** - Important for risk assessment (13% importance)
5. **Discount Applied** - Higher discounts correlate with returns and risk

### Model Performance Analysis

- **Random Forest** performs best for return prediction with highest ROC-AUC (99.73%)
- **XGBoost** performs best for risk prediction with excellent recall (83.78%) and F1-Score (86.92%)
- **Logistic Regression** shows good performance but lower than tree-based models
- All models show excellent performance on the Kaggle dataset

### Business Implications

1. **Product Rating is Critical:** Product rating accounts for 60% of return prediction - focus on quality
2. **Price Sensitivity:** Higher prices correlate with higher risk levels
3. **Category Matters:** Different product categories have different risk profiles
4. **Payment Method Impact:** Certain payment methods (like Gift Card) indicate higher risk
5. **Discount Strategy:** Higher discounts may indicate clearance items with higher return rates

---

## 5. Example Predictions

### Sample Predictions from Test Set

| Sample ID | Return Probability | Risk Probability | Actual Returned | Actual Risk | Correct? |
|-----------|-------------------|------------------|-----------------|-------------|----------|
| 1883 | 78.96% | 0.02% | Yes | Low | Return: ✓, Risk: ✗ |
| 1876 | 79.43% | 0.13% | No | Low | Return: ✗, Risk: ✓ |
| 1309 | 38.62% | 0.01% | No | Low | Return: ✓, Risk: ✓ |
| 1234 | 25.98% | 0.33% | No | Low | Return: ✓, Risk: ✓ |
| 932 | 1.25% | 62.96% | No | High | Return: ✓, Risk: ✗ |

---

## 6. Model Evaluation Metrics Explained

### Accuracy
- Percentage of correct predictions overall
- **Best:** Random Forest (89.81% for returns), XGBoost (98.60% for risk)

### Precision
- Of all predicted positives, how many were actually positive
- **Best:** Random Forest (89.81% for returns), Random Forest (93.18% for risk)

### Recall
- Of all actual positives, how many were correctly identified
- **Best:** Random Forest/XGBoost (87.39% for returns), XGBoost (83.78% for risk)

### F1-Score
- Harmonic mean of precision and recall (balanced metric)
- **Best:** Random Forest (88.58% for returns), XGBoost (86.92% for risk)

### ROC-AUC
- Area under the ROC curve (ability to distinguish between classes)
- **Best:** Random Forest (99.73% for returns), XGBoost (99.24% for risk)

---

## 7. Recommendations

### For Improving Model Performance

1. **Collect Real Product Ratings:** Currently estimated - real ratings would improve accuracy
2. **Feature Engineering:** 
   - Add customer lifetime value
   - Include product description/sentiment analysis
   - Add shipping time and delivery performance metrics
3. **Handle Class Imbalance:** Risk prediction has imbalanced classes (5.53% high risk)
4. **Hyperparameter Tuning:** Optimize model parameters using grid search
5. **Ensemble Methods:** Combine multiple models for better predictions

### For Business Applications

1. **Focus on Product Quality:** Product rating is the most critical factor - improve low-rated products
2. **Price Optimization:** Analyze price points that minimize returns and risk
3. **Category-Specific Strategies:** Different approaches for Electronics vs Books
4. **Payment Method Analysis:** Review policies for high-risk payment methods
5. **Real-time Monitoring:** Deploy models to flag high-risk orders in real-time

---

## 8. Saved Models and Files

### Trained Models
- `models/return_predictor.joblib` - Random Forest return predictor
- `models/risk_predictor.joblib` - XGBoost risk predictor
- `models/returned_best_model_xgboost.joblib` - Best return model (XGBoost)
- `models/risk_level_best_model_xgboost.joblib` - Best risk model (XGBoost)

### Visualizations
- `models/data_exploration.png` - Dataset exploration and statistics
- `models/returned_model_comparison.png` - Model comparison charts
- `models/returned_detailed_analysis.png` - Confusion matrices, ROC curves, feature importance
- `models/risk_level_model_comparison.png` - Risk model comparison
- `models/risk_level_detailed_analysis.png` - Risk analysis details

### Data Files
- `data/raw/ecommerce_returns_kaggle.csv` - Kaggle e-commerce returns dataset (10,000 samples)

---

## 9. Technical Details

### Data Preprocessing
- **Train/Test Split:** 80/20 (8,000 training, 2,000 test)
- **Feature Scaling:** StandardScaler applied to all features
- **Categorical Encoding:** LabelEncoder for categorical variables
- **Feature Engineering:** Created additional features (price_per_unit, high_value_order, low_rating, review_density, etc.)
- **Data Leakage Prevention:** Removed post-return features (return_reason, Days_to_Return)

### Model Configuration

#### Random Forest
- n_estimators: 100
- max_depth: 10
- min_samples_split: 5
- min_samples_leaf: 2

#### XGBoost
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1

#### Logistic Regression
- max_iter: 1000
- solver: liblinear

---

## 10. Conclusion

The project successfully demonstrates machine learning models for predicting product returns and risk levels using the Kaggle e-commerce returns dataset. The models show excellent performance with Random Forest achieving 99.73% ROC-AUC for return prediction and XGBoost achieving 99.24% ROC-AUC for risk prediction. Product rating is the most critical feature for return prediction, while product price, category, and payment method are key factors for risk assessment.

**Key Achievements:**
- 89.81% accuracy for return prediction
- 98.60% accuracy for risk prediction
- Excellent ROC-AUC scores (99.73% and 99.24%)
- Clear feature importance insights for business decisions

**Next Steps:**
1. Deploy models for real-time predictions
2. Collect real product ratings to improve accuracy
3. Implement A/B testing to validate model predictions
4. Create automated alerts for high-risk orders
5. Integrate with business systems for actionable insights

---

**Generated by:** Risk and Return Prediction Project  
**Dataset:** Kaggle E-commerce Returns Synthetic Data  
**Date:** Execution Date  
**Version:** 2.0.0
