"""
CENG 463 – Machine Learning Take-Home Midterm
Question 1

Requirements:
  pip install pandas numpy matplotlib seaborn statsmodels scikit-learn xgboost scipy
"""

# =============================================================================
# SECTION 1: Imports, Reproducibility, and Configuration
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNet, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

from xgboost import XGBRegressor
from scipy import stats
from scipy.stats import wilcoxon


# Constant random seed for reproducibility
RANDOM_SEED = 22
np.random.seed(RANDOM_SEED)



# =============================================================================
# SECTION 2: Data Loading
# =============================================================================

# California Housing Dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

print(f"Shape            : {df.shape}")
print(f"Columns          : {list(df.columns)}")
print(df.head())

print("=" * 70)



# =============================================================================
# SECTION 3: Exploratory Data Analysis (EDA) Visualizations
# =============================================================================

# Feature distributions
df.hist(bins=40, figsize=(15, 10), color='green', edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=16)
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300)
plt.close()


# Pairwise relationships
columns = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal'
]

sns.pairplot(df[columns].sample(2000, random_state=RANDOM_SEED), diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15})
plt.savefig('pairwise_relationships.png', dpi=300)
plt.close()


# Correlation heatmap
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

 

# =============================================================================
# SECTION 4: Outlier Detection
# =============================================================================

# Identifying outliers using the Z-score method
z_scores = np.abs(stats.zscore(df, nan_policy='omit'))

outliers_z = (z_scores > 3).sum(axis=0)
outliers_z = pd.Series(outliers_z, index=df.columns)

print('Outlier counts per feature (Z-score > 3):')
print(outliers_z.sort_values(ascending=False))
    
print("=" * 70) 



# =============================================================================
# SECTION 5: Data Preprocessing & Feature Engineering
# =============================================================================

# We will apply a log transformation to the target variable to reduce skewness
y_final = np.log1p(df['MedHouseVal'])

# Final feature matrix and target vector
X = df.drop('MedHouseVal', axis=1)
y = y_final


# Full polynomial features up to degree 3 (includes powers + interactions)
poly = PolynomialFeatures(degree=3, include_bias=False)

X_poly_and_interactions = poly.fit_transform(X)

# DataFrame that contains all features
poly_feature_names = poly.get_feature_names_out(X.columns)
X_poly_and_int_df = pd.DataFrame(X_poly_and_interactions, columns=poly_feature_names, index=X.index)


# RFE with Linear Regression
rfe = RFE(estimator=LinearRegression(), n_features_to_select=10)
rfe.fit(X_poly_and_int_df, y)

selected_features = X_poly_and_int_df.columns[rfe.support_].tolist()
print('RFE selected features:', selected_features)

print("=" * 70)


# DataFrame with only RFE-selected features
X_selected_df = X_poly_and_int_df[selected_features].copy()

print(f'Selected-feature DataFrame shape: {X_selected_df.shape}')
print(X_selected_df.head())

print("=" * 70)



# =============================================================================
# SECTION 6: Model Pipelines & Hyperparameter Setup
# =============================================================================

# Required repeated CV and metric dictionary
rkf = RepeatedKFold(n_splits=5,
                    n_repeats=3,
                    random_state=RANDOM_SEED)

metric_names = ['RMSE', 'MAE', 'R2', 'Adj_R2', 'MAPE', 'EVS']


alphas = np.logspace(-3, 3, 100)

# Required Pipelines
lr_model = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])

ridge_model = Pipeline([('scaler', StandardScaler()), ('reg', RidgeCV(alphas=alphas, cv=5))])

lasso_model = Pipeline([('scaler', StandardScaler()), ('reg', LassoCV(alphas=alphas, cv=5, max_iter=10000))])

en_base = Pipeline([('scaler', StandardScaler()), ('reg', ElasticNet(max_iter=10000))])
en_param_dist = {
    'reg__alpha': alphas,
    'reg__l1_ratio': np.linspace(0.1, 0.9, 9)
    }
en_opt = RandomizedSearchCV(
    en_base, en_param_dist, n_iter=20, cv=5,
    random_state=RANDOM_SEED, scoring='neg_mean_squared_error', n_jobs=-1
    )

xgb_base = Pipeline([('scaler', StandardScaler()), ('reg', XGBRegressor(random_state=RANDOM_SEED))])
xgb_param_dist = {
    'reg__n_estimators': [100, 300, 500],
    'reg__learning_rate': [0.01, 0.05, 0.1],
    'reg__max_depth': [3, 5, 7],
    'reg__subsample': [0.7, 0.8, 0.9],
    'reg__colsample_bytree': [0.7, 0.8, 0.9]
    }
xgb_opt = RandomizedSearchCV(
    xgb_base, xgb_param_dist, n_iter=20, cv=5,
    random_state=RANDOM_SEED, scoring='neg_mean_squared_error', n_jobs=-1
    )

final_models = {
    'Linear': lr_model,
    'Ridge': ridge_model,
    'Lasso': lasso_model,
    'ElasticNet': en_opt,
    'XGBoost': xgb_opt
    }



# =============================================================================
# SECTION 7: Cross-Validation & Model Training
# =============================================================================

raw_results = {name: {m: [] for m in metric_names} for name in final_models}

print(f'Starting repeated CV for {len(final_models)} models with interaction terms...')
# Training all the models
for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_selected_df, y), start=1):
    X_train, X_test = X_selected_df.iloc[train_idx], X_selected_df.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    n_samples = len(y_test)
    n_features = X_test.shape[1]

    for model_name, model in final_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        evs = explained_variance_score(y_test, preds)

        denom = max(n_samples - n_features - 1, 1)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / denom

        # Avoid divide-by-zero when y_test has zeros.
        y_test_safe = np.where(np.abs(y_test) < 1e-12, 1e-12, y_test)
        mape = np.mean(np.abs((y_test - preds) / y_test_safe)) * 100

        raw_results[model_name]['RMSE'].append(rmse)
        raw_results[model_name]['MAE'].append(mae)
        raw_results[model_name]['R2'].append(r2)
        raw_results[model_name]['Adj_R2'].append(adj_r2)
        raw_results[model_name]['MAPE'].append(mape)
        raw_results[model_name]['EVS'].append(evs)

    print(f'Completed split {fold_idx}/{rkf.get_n_splits()}')



# =============================================================================
# SECTION 8: Alternative Robust Model (Huber Regressor)
# =============================================================================

# As an alternative solution, training a HuberRegressor
huber_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', HuberRegressor(max_iter=2000))
])

huber_params = {
    'reg__epsilon': [1.1, 1.35, 1.5, 1.75, 1.9],
    'reg__alpha': np.logspace(-4, 1, 20)
}

huber_search = RandomizedSearchCV(
    huber_pipe, huber_params, n_iter=15, cv=5,
    random_state=RANDOM_SEED, n_jobs=-1, scoring='neg_mean_squared_error'
)


cv_huber = cross_validate(
    huber_search, X_selected_df, y, cv=rkf,
    scoring={
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2',
        'mape': 'neg_mean_absolute_percentage_error',
        'evs': 'explained_variance'
    },
    n_jobs=-1, return_estimator=True
)


# Store raw fold results for significance testing
n, p = X.shape[0], X.shape[1]

raw_results['Huber'] = {
    "RMSE": -cv_huber['test_rmse'],
    "MAE": -cv_huber['test_mae'],
    "R2": cv_huber['test_r2'],
    "Adj_R2": 1 - (1 - cv_huber['test_r2']) * (n - 1) / (n - p - 1),
    "MAPE": -cv_huber['test_mape'] * 100,
    "EVS": cv_huber['test_evs']
}

# Store the final model and metrics for later
final_models['Huber'] = cv_huber['estimator'][-1]



# =============================================================================
# SECTION 9: Aggregating Summary Metrics & Export
# =============================================================================

summary_rows = []

for model_name in final_models:
    row = {'Model': model_name}
    for metric in metric_names:
        mean_val = float(np.mean(raw_results[model_name][metric]))
        std_val = float(np.std(raw_results[model_name][metric]))
        row[metric] = f'{mean_val:.4f} +/- {std_val:.4f}'
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print('Final summary metrics:')
print(summary_df)

summary_df.to_csv('model_summary_metrics.csv', index=False)



# =============================================================================
# SECTION 10: Model Diagnostics (Residuals & Q-Q Plots)
# =============================================================================

# Get the Fitted vs Residuals Plots and Normal Q-Q Plots for all models
fig, axes = plt.subplots(len(final_models), 2, figsize=(15, 4 * len(final_models)))

for i, (name, model) in enumerate(final_models.items()):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # --- Column 0: Fitted vs Residuals ---
    ax_res = axes[i, 0]
    ax_res.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', color='skyblue')
    ax_res.axhline(0, color='red', linestyle='--', linewidth=2)
    ax_res.set_title(f"{name}: Fitted vs Residuals", fontsize=14)
    ax_res.set_xlabel("Predicted Values")
    ax_res.set_ylabel("Residuals")

    # --- Column 1: Normal Q-Q Plot ---
    ax_qq = axes[i, 1]
    sm.qqplot(residuals, line='45', ax=ax_qq)
    ax_qq.get_lines()[1].set_color("red")
    ax_qq.set_title(f"{name}: Normal Q-Q Plot", fontsize=14)

plt.tight_layout()
plt.savefig('residuals_qq_plots.png', dpi=300)
plt.close()



# =============================================================================
# SECTION 11: Statistical Significance Testing
# =============================================================================

# Apply Wilcoxon Signed-Rank Test with Linear model as baseline
baseline = "Linear"
comparisons = ["Ridge", "Lasso", "ElasticNet", "XGBoost", "Huber"]
sig_results = []


for model_name in comparisons:
    # Logic: Use MAE for Huber, RMSE for everything else
    metric = "MAE" if model_name == "Huber" else "RMSE"

    try:
        stat, p_value = wilcoxon(raw_results[baseline][metric],
                                 raw_results[model_name][metric])

        is_significant = "Yes" if p_value < 0.05 else "No"

    except ValueError:
        # Fails if the two sets of 15 values are identical
        # Identical results mean no significant difference
        p_value = 1.0
        is_significant = "No"

    sig_results.append({
        "Comparison": f"{baseline} vs {model_name}",
        "Metric Used": metric,
        "p-value": round(p_value, 5),
        "Significant (α=0.05)": is_significant
    })


# Apply Wilcoxon Signed-Rank Test with XGBoost and HuberRegressor to see if HuberRegressor is more improved
try:
    stat, p_value = wilcoxon(raw_results["XGBoost"]["MAE"],
                                 raw_results["Huber"]["MAE"])

    is_significant = "Yes" if p_value < 0.05 else "No"

except ValueError:
    # Fails if the two sets of 15 values are identical
    # Identical results mean no significant difference
    p_value = 1.0
    is_significant = "No"

sig_results.append({
    "Comparison": f"XGBoost vs Huber",
    "Metric Used": "MAE",
    "p-value": round(p_value, 5),
    "Significant (α=0.05)": is_significant})

# Create the final significance DataFrame to see the results
df_sig_final = pd.DataFrame(sig_results)
print(df_sig_final)

df_sig_final.to_csv('model_significance_results.csv', index=False)