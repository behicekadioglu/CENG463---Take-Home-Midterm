# Question 1: Regression Analysis – Regularization, Feature Engineering, and Model Selection

## Objective
The objective of this task is to rigorously compare multiple regression models using advanced regularization techniques, extensive feature engineering, and statistical significance testing.

## Dataset
* **Dataset Used:** California Housing dataset.

## Implementation Tasks

### 1. Exploratory Data Analysis (EDA)
* Perform deep EDA including visualization of feature distributions and correlation heatmaps.
* Identify pairwise interactions and detect outliers using IQR or Z-score methods.

### 2. Feature Engineering & Selection
* **Transformations:** Apply log transforms for skewed targets [cite: 38].
* **Feature Creation:** Generate polynomial features (degree 2 and 3) and interaction terms.
* **Selection:** Use Recursive Feature Elimination (RFE) or feature importance from tree-based models to select top features.

### 3. Model Selection & Optimization
Compare at least four regression models:
* **Linear Regression**.
* **Ridge and Lasso:** Using `RidgeCV` and `LassoCV` for cross-validated alpha selection.
* **Elastic Net:** With tuned L1 ratio.
* **Gradient Boosting:** XGBoost or LightGBM with hyperparameter optimization via random search or Bayesian optimization.
* **Robust Regression:** Implement `HuberRegressor` as a fix for violations of homoscedasticity or normality.

### 4. Experimental Setup & Evaluation
* **Validation Strategy:** 5-fold cross-validation repeated 3 times, reporting mean and standard deviation for all metrics.
* **Primary Metrics:** RMSE, MAE, $R^{2}$, Adjusted $R^{2}$, MAPE, and Explained Variance Score.
* **Statistical Testing:** Perform paired t-tests or Wilcoxon signed-rank tests to determine the significance of differences between models.

## Analysis & Discussion
* **Residual Analysis:** Provide residual plots (fitted vs. residuals) and Q-Q plots to discuss violations of model assumptions.
* **Model Trade-offs:** Conclude with a detailed discussion on the bias-variance trade-off observed across the implemented models.
