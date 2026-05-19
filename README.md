# COE546-ML-Project:
Machine Learning project focused on predicting whether a user session in a food delivery application results in an order placement.

# User Behavior Prediction in Food Delivery Applications

The project was developed for a Kaggle competition using ensemble tree-based models and extensive behavioral feature engineering. The primary evaluation metric is **ROC-AUC**, with additional consideration for **F1-score** due to severe class imbalance.


## Project Overview

Food delivery platforms generate large amounts of behavioral and transactional data during user sessions. Predicting whether a session will convert into an order can help improve:

* Personalized recommendations
* Promotion targeting
* User retention strategies
* Conversion optimization

This project approaches the problem as a **binary classification task** using engineered behavioral, temporal, and promotional features.

## Dataset

* **Training samples:** 297,236
* **Test samples:** 99,639
* **Target:** `order_placed`
* **Positive class ratio:** 2.9%
* **Features:** 17 anonymized variables (`f2`–`f17`)

### Data Characteristics

The dataset contains:

* Numerical features
* Categorical variables
* Timestamp-based session data
* Structured missing values
* Severe class imbalance


## Exploratory Data Analysis (EDA)

Key findings from EDA included:

### Cart Behavior

* Users with non-zero cart value or cart items were significantly more likely to place an order.
* Cart-related variables showed the strongest class separation.

### Customer Type

* Returning users converted at substantially higher rates than new users.

### Promotional Features

* Certain offer categories had very high conversion rates.
* Missing promotional fields occurred systematically and carried predictive information.

### Temporal Patterns

* Conversion rates varied strongly by:

  * Hour of day
  * Day of week
* Late-night sessions showed elevated conversion behavior.


## Feature Engineering

A large portion of the project focused on transforming raw session data into behavioral signals.

### Engineered Feature Groups

#### Temporal Features

* Session duration
* Time-to-event
* Remaining session time
* Cyclical hour/day encodings

#### Cart & Transaction Features

* Log-transformed cart value
* Cart item intensity
* Average item value
* Cart velocity

#### Offer & Promotion Features

* Offer eligibility
* Discount ratios
* Threshold proximity
* Promotion generosity

#### Behavioral Features

* Fast-action indicators
* Event position within session
* Returning-user interactions

#### Missingness Signals

* Structured missing-value indicators


## Modeling Approach

Two gradient boosting models were trained using 5-fold stratified cross-validation:

### Models Used

* CatBoost
* LightGBM

### Validation Strategy

* Stratified 5-Fold Cross Validation
* Out-of-fold (OOF) predictions
* ROC-AUC evaluation

### Ensemble

Final predictions were generated using a weighted ensemble of CatBoost and LightGBM outputs.


## Results

| Model            | OOF ROC-AUC  |
| ---------------- | ------------ |
| CatBoost         | 0.971487     |
| LightGBM         | 0.971949     |
| Ensemble (40/60) | **0.972290** |

The ensemble achieved the best overall validation performance.


## Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* LightGBM
* Matplotlib
* Kaggle Notebooks


## Future Improvements

Potential future work includes:

* Rank-based ensemble blending
* Additional interaction features
* Hyperparameter optimization
* Pseudo-labeling
* Advanced imbalance-aware training strategies

## Report
Full project report: [COE546- Project Report - Snih, Antoun, Ammar.pdf]

## Author

Developed by Lara Snih - Ghadi Ammar - Jana Antoun as part of a Machine Learning Course Project
