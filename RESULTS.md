# Risk and Return Prediction - Project Results

**Project Date:** Generated on execution  
**Dataset:** 10,000 synthetic e-commerce orders  
**Models Trained:** Random Forest, XGBoost, Logistic Regression

---

## 1. Dataset Overview

### Dataset Statistics
- **Total Samples:** 10,000 orders
- **Total Features:** 19 original features + 3 engineered features = 22 features
- **Return Rate:** 31.12%
- **High Risk Rate:** 31.18%

### Dataset Features
- **Product Features:** category, price, rating, number of reviews
- **Customer Features:** age, purchase history, region
- **Order Features:** quantity, shipping method, payment method, total value
- **Temporal Features:** month, day of week, hour
- **Target Variables:** return status, risk level, risk score

---

## 2. Return Prediction Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **70.60%** | 59.66% | 16.88% | 26.32% | 66.75% |
| **XGBoost** | 69.70% | 53.01% | **22.67%** | **31.76%** | 64.59% |
| **Logistic Regression** | 68.35% | 40.98% | 4.02% | 7.32% | 60.23% |

**Best Model for Return Prediction:** XGBoost (Best F1-Score: 31.76%)

### Feature Importance - Return Prediction

#### Random Forest Model:
1. **product_rating** - 17.67%
2. **product_price** - 10.78%
3. **total_order_value** - 8.41%
4. **price_per_unit** - 8.21%
5. **review_density** - 6.88%
6. **num_reviews** - 6.20%
7. **customer_age** - 6.06%
8. **order_hour** - 5.01%
9. **product_category** - 4.48%
10. **payment_method** - 4.35%

#### XGBoost Model:
1. **product_rating** - 10.66%
2. **product_price** - 9.47%
3. **payment_method** - 9.13%
4. **product_category** - 5.96%
5. **review_density** - 5.54%

#### Logistic Regression Model:
1. **product_price** - 36.74%
2. **price_per_unit** - 34.13%
3. **product_rating** - 31.01%
4. **num_reviews** - 29.34%
5. **total_order_value** - 28.49%

---

## 3. Risk Prediction Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **68.95%** | 50.78% | 15.71% | 23.99% | 63.42% |
| **XGBoost** | 68.65% | 49.49% | **23.56%** | **31.92%** | 62.78% |
| **Logistic Regression** | 68.20% | 43.75% | 6.73% | 11.67% | 58.73% |

**Best Model for Risk Prediction:** XGBoost (Best F1-Score: 31.92%)

### Feature Importance - Risk Prediction

#### Random Forest Model:
1. **product_rating** - 11.86%
2. **payment_method** - 10.49%
3. **product_price** - 9.61%
4. **product_category** - 5.65%
5. **customer_age** - 5.26%
6. **total_order_value** - 5.22%
7. **review_density** - 5.18%
8. **order_month** - 5.08%
9. **price_per_unit** - 5.03%
10. **num_reviews** - 5.00%

---

## 4. Key Insights

### Most Important Features Across All Models

1. **Product Rating** - Consistently the most important feature for both return and risk prediction
2. **Product Price** - Strong predictor of both return probability and risk level
3. **Payment Method** - Significant factor in risk assessment
4. **Product Category** - Important for understanding return patterns
5. **Total Order Value** - Higher value orders show different return patterns

### Model Performance Analysis

- **XGBoost** performs best overall with the highest F1-Scores for both tasks
- **Random Forest** has the highest accuracy but lower recall (misses many positive cases)
- **Logistic Regression** shows poor performance, especially in recall, indicating the problem is non-linear

### Business Implications

1. **Product Rating is Critical:** Low-rated products have significantly higher return rates
2. **Price Sensitivity:** Higher-priced items and higher total order values correlate with returns
3. **Payment Method Matters:** Cash on delivery and certain payment methods increase risk
4. **Customer Segmentation:** New customers (purchase_history = 0) show different return patterns

---

## 5. Example Predictions

### Sample Predictions from Test Set

| Sample ID | Return Probability | Risk Probability | Actual Returned | Actual Risk | Correct? |
|-----------|-------------------|------------------|-----------------|-------------|----------|
| 553 | 32.27% | 15.47% | Yes | High | Return: ✓, Risk: ✗ |
| 1921 | 34.44% | 48.80% | Yes | High | Return: ✓, Risk: ✓ |
| 1148 | 27.63% | 38.88% | Yes | High | Return: ✓, Risk: ✓ |
| 1159 | 56.45% | 29.80% | Yes | High | Return: ✓, Risk: ✓ |
| 1479 | 23.36% | 53.55% | No | Low | Return: ✓, Risk: ✗ |

---

## 6. Model Evaluation Metrics Explained

### Accuracy
- Percentage of correct predictions overall
- **Best:** Random Forest (70.60% for returns, 68.95% for risk)

### Precision
- Of all predicted positives, how many were actually positive
- **Best:** Random Forest (59.66% for returns, 50.78% for risk)

### Recall
- Of all actual positives, how many were correctly identified
- **Best:** XGBoost (22.67% for returns, 23.56% for risk)

### F1-Score
- Harmonic mean of precision and recall (balanced metric)
- **Best:** XGBoost (31.76% for returns, 31.92% for risk)

### ROC-AUC
- Area under the ROC curve (ability to distinguish between classes)
- **Best:** Random Forest (66.75% for returns, 63.42% for risk)

---

## 7. Recommendations

### For Improving Model Performance

1. **Collect More Data:** Increase dataset size for better generalization
2. **Feature Engineering:** 
   - Add customer lifetime value
   - Include product description/sentiment analysis
   - Add shipping time and delivery performance metrics
3. **Handle Class Imbalance:** Use SMOTE or other techniques to balance classes
4. **Hyperparameter Tuning:** Optimize model parameters using grid search
5. **Ensemble Methods:** Combine multiple models for better predictions

### For Business Applications

1. **Focus on Product Rating:** Implement strategies to improve low-rated products
2. **Price Optimization:** Analyze price points that minimize returns
3. **Payment Method Analysis:** Review policies for high-risk payment methods
4. **Customer Segmentation:** Different strategies for new vs. returning customers
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
- `data/raw/ecommerce_orders.csv` - Complete dataset (10,000 samples)

---

## 9. Technical Details

### Data Preprocessing
- **Train/Test Split:** 80/20 (8,000 training, 2,000 test)
- **Feature Scaling:** StandardScaler applied to all features
- **Categorical Encoding:** LabelEncoder for categorical variables
- **Feature Engineering:** Created 3 additional features (price_per_unit, high_value_order, low_rating, etc.)

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

The project successfully demonstrates machine learning models for predicting product returns and risk levels in an e-commerce setting. While the models show moderate performance, they provide valuable insights into the factors that influence returns and risk. The XGBoost model performs best overall, with product rating being the most critical feature for both predictions.

**Next Steps:**
1. Deploy models for real-time predictions
2. Collect more data to improve model accuracy
3. Implement A/B testing to validate model predictions
4. Create automated alerts for high-risk orders
5. Integrate with business systems for actionable insights

---

**Generated by:** Risk and Return Prediction Project  
**Date:** Execution Date  
**Version:** 1.0.0

