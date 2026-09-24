# Smart Logistics Analytics: Machine Learning & Optimization

An end-to-end business analytics project that combines exploratory data analysis,
machine learning, and linear optimization to analyze logistics delays and
demonstrate ML-informed resource allocation.

## Project Overview

This project examines a logistics dataset containing 1,000 operational
observations across 10 fleet assets.

The project follows a complete analytics workflow:

**Data Cleaning → Exploratory Analysis → Machine Learning → Model Validation → Optimization**

The analysis focuses on two business questions:

1. What operational factors are associated with logistics delays, and can those
   delays be predicted using machine learning?
2. How can limited inventory replenishment be allocated across shortage
   observations using predicted operational risk as a prioritization signal?

## Dataset

The dataset contains 1,000 observations and 16 original variables, including:

- Fleet asset ID
- Inventory levels
- Shipment status
- Traffic conditions
- Waiting time
- Temperature and humidity
- Asset utilization
- Forecast demand
- Logistics delay indicators

Data-quality analysis identified 263 missing values in the logistics delay
reason field. These observations were investigated and retained using an
`Unknown` category rather than being removed.

No duplicate records or numerical outliers were identified.

## Exploratory Data Analysis

Key findings included:

- 56.6% of observations experienced a logistics delay.
- Heavy traffic showed the strongest relationship with delays.
- Every Heavy-traffic observation in the dataset was associated with a delay.
- Delay rates varied across trucks and time periods.
- Waiting time, asset utilization, inventory shortage, and several continuous
  variables showed little relationship with the delay target.
- Forecast demand exceeded available inventory in 261 observations (26.1%).
- Geographic coordinates were broadly distributed across the globe and did not
  reveal meaningful logistics clusters.

The EDA also identified potential target leakage. `Shipment_Status` and
`Logistics_Delay_Reason` were excluded from predictive modeling because they
could contain information unavailable at prediction time.

## Machine Learning

Four classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Models were compared using stratified 5-fold cross-validation.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.728 | 0.861 | 0.618 | 0.719 | 0.769 |
| Decision Tree | 0.672 | 0.709 | 0.722 | 0.715 | 0.665 |
| Random Forest | 0.726 | 0.861 | 0.616 | 0.717 | 0.768 |
| Gradient Boosting | **0.734** | 0.833 | 0.662 | **0.737** | **0.779** |

Gradient Boosting produced the strongest overall cross-validation performance.

Hyperparameter tuning increased its cross-validation ROC-AUC from **0.779 to
0.782**.

The tuned model used:

- Learning rate: 0.1
- Maximum tree depth: 2
- Number of estimators: 50

On the held-out test set, the tuned Gradient Boosting model achieved:

- Accuracy: 75.0%
- Precision: 90.9%
- Recall: 61.9%
- F1-score: 73.7%
- ROC-AUC: 0.746

Logistic Regression achieved a held-out ROC-AUC of 0.800, demonstrating the
importance of considering both cross-validation and held-out performance when
comparing models.

## Model Interpretation

Feature-importance analysis showed that `Traffic_Status_Heavy` overwhelmingly
dominated the Gradient Boosting model.

This finding was consistent with the exploratory analysis, where all
Heavy-traffic observations were associated with logistics delays.

## Optimization

The project extends predictive analytics into a prescriptive scenario.

There were 261 observations where forecast demand exceeded available inventory,
representing a total shortage of **16,592 units**.

A linear optimization model was developed to allocate limited replenishment
inventory across shortage observations. Predicted delay probabilities from the
machine-learning model were used as a prioritization signal.

Under a hypothetical 500-unit replenishment constraint:

- Total shortage before allocation: 16,592 units
- Inventory allocated: 500 units
- Remaining shortage: 16,092 units

Scenario analysis evaluated replenishment capacities of 250, 500, 1,000, and
2,000 units.

| Replenishment Capacity | Remaining Shortage | Shortage Reduction |
|---:|---:|---:|
| 250 | 16,342 | 1.5% |
| 500 | 16,092 | 3.0% |
| 1,000 | 15,592 | 6.0% |
| 2,000 | 14,592 | 12.1% |

## Project Structure

```text
Smart-Logistics-Analytics/
├── data/
│   ├── smart_logistics_dataset.csv
│   └── smart_logistics_cleaned.csv
├── models/
│   └── final_gradient_boosting_model.pkl
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_machine_learning.ipynb
│   └── 04_optimization.ipynb
├── README.md
├── requirements.txt
└── .gitignore