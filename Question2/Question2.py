"""
CENG 463 – Machine Learning Take-Home Midterm
Question 2

Dataset: Credit Card Fraud Detection
  → Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  → Place 'creditcard.csv' in the same directory as this script.

Requirements:
  pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports & Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score,
    brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models
from scikeras.wrappers import KerasClassifier


SEED = 22
np.random.seed(SEED)



# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load Dataset
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("creditcard.csv")

print(f"Shape            : {df.shape}")
print(f"Columns          : {list(df.columns)}")
print(f"\nClass distribution:\n{df['Class'].value_counts()}")

# number of fraud samples (fraud = 1)
n_minority = df["Class"].sum()

# number of legit samples (all - fraud)
n_majority = len(df) - n_minority

# imbalance ratio (IR) = majority / minority
IR = n_majority / n_minority

print(f"\nImbalance Ratio (IR): {IR:.1f}:1  ({n_minority} fraud / {n_majority} legit)")



# ─────────────────────────────────────────────────────────────────────────────
# 2.  Exploratory Data Analysis
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Class distribution bar chart
df["Class"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "crimson"],
                                edgecolor="black")
axes[0].set_title("Class Distribution")
axes[0].set_xticklabels(["Legit (0)", "Fraud (1)"], rotation=0)
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom")


# Transaction amount by class
df.groupby("Class")["Amount"].apply(list)
axes[1].boxplot([df[df["Class"] == 0]["Amount"], df[df["Class"] == 1]["Amount"]],
                labels=["Legit", "Fraud"])
axes[1].set_title("Transaction Amount by Class")
axes[1].set_ylabel("Amount (USD)")
axes[1].set_yscale("log")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150)
plt.close()
print("Saved: eda_overview.png")


# Correlation heatmap of top features
corr = df.drop(columns=["Time"]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
plt.close()
print("Saved: eda_correlation.png")



# ─────────────────────────────────────────────────────────────────────────────
# 3.  Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

# drop 'Time', it is not useful for modeling and 
# can cause data leakage (since it is relative to the first transaction, not absolute time)
df = df.drop(columns=["Time"])

# log-transform 'Amount', range of 'Amount' is very wide, 
# log-transform can compress it and make it more suitable for modeling
df["Amount"] = np.log1p(df["Amount"])


X = df.drop(columns=["Class"]).values
y = df["Class"].values

print(f"Feature matrix shape : {X.shape}")
print(f"Positive rate        : {y.mean()*100:.4f}%")


# calculate class weights, used by several models
classes = np.unique(y)
cw = compute_class_weight("balanced", classes=classes, y=y)
class_weight_dict = {0: cw[0], 1: cw[1]}

print(f"Computed class weights: {class_weight_dict}")



# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metrics Helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, label=""):
    """Return a dict of all required evaluation metrics."""
    return {
        "Model"            : label,
        "Precision"        : precision_score(y_true, y_pred, zero_division=0),
        "Recall"           : recall_score(y_true, y_pred, zero_division=0),
        "F1 (macro)"       : f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (micro)"       : f1_score(y_true, y_pred, average="micro", zero_division=0),
        "ROC-AUC"          : roc_auc_score(y_true, y_prob),
        "PR-AUC"           : average_precision_score(y_true, y_prob),
        "MCC"              : matthews_corrcoef(y_true, y_pred),
        "Balanced Acc."    : balanced_accuracy_score(y_true, y_pred),
        "Brier Score"      : brier_score_loss(y_true, y_prob),
    }


def evaluate_pipeline(pipe, X, y, label, n_splits=5):
    """
    Run stratified k-fold CV (default 5-folds), 
    collect Out Of Fold (OOF) predictions, 
    compute metrics.
    Returns (metrics_dict, oof_probs, oof_preds, oof_true).
    """

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # initialize OOF arrays
    oof_prob = np.zeros(len(y))
    oof_pred = np.zeros(len(y), dtype=int)

    for fold, (tr, val) in enumerate(cv.split(X, y)):
        pipe.fit(X[tr], y[tr])
        
        oof_prob[val] = pipe.predict_proba(X[val])[:, 1]
        oof_pred[val] = pipe.predict(X[val])

    metrics = compute_metrics(y, oof_pred, oof_prob, label)

    print(f"  [{label}]  PR-AUC={metrics['PR-AUC']:.4f}  "
          f"F1-macro={metrics['F1 (macro)']:.4f}  MCC={metrics['MCC']:.4f}")
    
    return metrics, oof_prob, oof_pred



# ─────────────────────────────────────────────────────────────────────────────
# 5.  Baseline: Four Classifiers (no resampling, cost-sensitive weights)
# ─────────────────────────────────────────────────────────────────────────────

# To use MLP with class weights, we need to define a Keras model factory function
# Define the Keras Model Factory
def create_keras_mlp(meta):
    # meta["n_features_in_"] is automatically provided by SciKeras
    n_features = meta["n_features_in_"]
    
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Setup Weights and Parameters for Keras MLP
keras_class_weights = {0: 1.0, 1: float(IR)}


scale_pos = IR   # for XGBoost

# Pipelines for baseline models
baseline_pipes = {
    "LogReg": ImbPipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=SEED))
    ]),

    "RF": ImbPipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)),
    ]),

    "XGBoost (SPW)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("clf",    xgb.XGBClassifier(scale_pos_weight=scale_pos, n_estimators=300,
                                     use_label_encoder=False, eval_metric="logloss",
                                     random_state=SEED, n_jobs=-1)),
    ]),

    "MLP (Keras CW)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("clf",    KerasClassifier(
                        model=create_keras_mlp,
                        epochs=20,
                        batch_size=32,
                        class_weight=keras_class_weights,
                        validation_split=0.1,
                        random_state=SEED,
                        verbose=1 # Set to 1 to see training progress
                    ))
    ]),
}

# Training Loop for Baselines
baseline_results = []
baseline_oof = {}

# Sample weights for models that use the .fit(sample_weight=...) syntax
# Note: Keras uses class_weight inside the wrapper, so we don't need to pass
# sample_weight_array to it manually in the evaluate_pipeline call.
sample_weight_array = np.array([class_weight_dict[int(label)] for label in y])

for name, pipe in baseline_pipes.items():

    print(f"Training {name}...")
    metrics, oof_prob, oof_pred = evaluate_pipeline(pipe, X, y, name)
    
    baseline_results.append(metrics)
    baseline_oof[name] = (oof_prob, oof_pred)


df_baseline = pd.DataFrame(baseline_results).set_index("Model")
print("\n" + "=" * 70)
print("FINAL BASELINE RESULTS")
print("=" * 70)
print(df_baseline.round(4).to_string())



# ─────────────────────────────────────────────────────────────────────────────
# 6.  Resampling Strategies  (SMOTE / ADASYN / RandomUnderSampler)
# ─────────────────────────────────────────────────────────────────────────────

# We use Random Forest as the base learner for resampling comparison
def make_rf_pipe(sampler):
    return ImbPipeline([
        ("scaler",  StandardScaler()),
        ("sampler", sampler),
        ("clf",     RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                           random_state=SEED)),
    ])


# we used three resampling strategies: SMOTE, ADASYN, and RandomUnderSampler
resampling_configs = {
    "RF + SMOTE"       : make_rf_pipe(SMOTE(random_state=SEED)),
    "RF + ADASYN"      : make_rf_pipe(ADASYN(random_state=SEED)),
    "RF + RandUnder"   : make_rf_pipe(RandomUnderSampler(random_state=SEED)),
}

resampling_results = []
resampling_oof = {}

# training and evaluating each resampling strategy
for name, pipe in resampling_configs.items():
    metrics, oof_prob, oof_pred = evaluate_pipeline(pipe, X, y, name)
    resampling_results.append(metrics)
    resampling_oof[name] = (oof_prob, oof_pred)

df_resampling = pd.DataFrame(resampling_results).set_index("Model")
print("\nResampling Results:")
print(df_resampling.round(4).to_string())



# ─────────────────────────────────────────────────────────────────────────────
# 7.  Cost-Sensitive Random Forest (No Resampling)
# ─────────────────────────────────────────────────────────────────────────────

# class_weight="balanced" automatically handles the cost-sensitive penalties
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200, 
        class_weight="balanced", 
        n_jobs=-1, 
        random_state=SEED
    ))
])

metrics, oof_prob, oof_pred = evaluate_pipeline(rf_pipe, X, y, "Random Forest (Weighted)")

df_rf_result = pd.DataFrame([metrics]).set_index("Model")
print("\nRandom Forest Results:")
print(df_rf_result.round(4).to_string())



# ─────────────────────────────────────────────────────────────────────────────
# 8.  Combined Results Table
# ─────────────────────────────────────────────────────────────────────────────
df_all = pd.concat([df_baseline, df_resampling, df_rf_result])

print(df_all.round(4).to_string())

# Save to CSV
df_all.to_csv("q2_all_results.csv")
print("\nSaved: q2_all_results.csv")



# ─────────────────────────────────────────────────────────────────────────────
# 9.  Probability Calibration on Best Two Models
# ─────────────────────────────────────────────────────────────────────────────

# Identify best 2 models by PR-AUC
best_two = df_all["PR-AUC"].nlargest(2).index.tolist()
print(f"Best two models (by PR-AUC): {best_two}")

# For calibration we need estimators; map names back to pipelines
all_pipes = {**baseline_pipes, **resampling_configs}

# Re-build best pipes with Platt scaling (sigmoid) and Isotonic regression
calibration_results = []

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Reliability Diagrams (Calibration Curves)", fontsize=13)

for ax, model_name in zip(axes, best_two):
    base_pipe = all_pipes[model_name]

    # Full fit on all data for calibration curves (CV=5 internal)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_prob_raw  = np.zeros(len(y))
    oof_prob_sig  = np.zeros(len(y))
    oof_prob_iso  = np.zeros(len(y))

    for tr, val in cv_inner.split(X, y):
        # Raw
        base_pipe.fit(X[tr], y[tr])
        oof_prob_raw[val] = base_pipe.predict_proba(X[val])[:, 1]

        # Platt (sigmoid)
        cal_sig = CalibratedClassifierCV(base_pipe, method="sigmoid", cv=3)
        cal_sig.fit(X[tr], y[tr])
        oof_prob_sig[val] = cal_sig.predict_proba(X[val])[:, 1]

        # Isotonic
        cal_iso = CalibratedClassifierCV(base_pipe, method="isotonic", cv=3)
        cal_iso.fit(X[tr], y[tr])
        oof_prob_iso[val] = cal_iso.predict_proba(X[val])[:, 1]

    for label, probs in [("Raw", oof_prob_raw),
                          ("Platt (sigmoid)", oof_prob_sig),
                          ("Isotonic", oof_prob_iso)]:
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        bs = brier_score_loss(y, probs)
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{label}  (BS={bs:.4f})")
        calibration_results.append({
            "Model": model_name, "Calibration": label, "Brier Score": round(bs, 5)
        })

    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title(model_name, fontsize=10)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("calibration_curves.png", dpi=150)
plt.close()
print("Saved: calibration_curves.png")


df_cal = pd.DataFrame(calibration_results)
print("\nBrier Scores (Calibration Comparison):")
print(df_cal.to_string(index=False))
df_cal.to_csv("calibration_brier_scores.csv", index=False)



# ─────────────────────────────────────────────────────────────────────────────
# 10.  Precision-Recall Curves & Optimal Threshold
# ─────────────────────────────────────────────────────────────────────────────

# Collect OOF probs for all models
all_oof = {**baseline_oof, **resampling_oof}

fig, ax = plt.subplots(figsize=(9, 6))
threshold_rows = []

for name, (oof_prob, _) in all_oof.items():
    prec, rec, thresholds = precision_recall_curve(y, oof_prob)
    auprc = average_precision_score(y, oof_prob)

    # F1-maximisation threshold
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1    = f1_scores[best_idx]

    ax.plot(rec, prec, label=f"{name}  (AUPRC={auprc:.3f})")

    # Recompute preds at optimal threshold
    y_pred_opt = (oof_prob >= best_thresh).astype(int)
    threshold_rows.append({
        "Model"         : name,
        "Opt. Threshold": round(best_thresh, 4),
        "F1 @ Threshold": round(best_f1, 4),
        "Precision"     : round(precision_score(y, y_pred_opt, zero_division=0), 4),
        "Recall"        : round(recall_score(y, y_pred_opt, zero_division=0), 4),
    })
    print(f"  {name:<20}  best_thresh={best_thresh:.4f}  F1={best_f1:.4f}")

ax.axhline(n_minority / len(y), linestyle="--", color="gray",
           label=f"Baseline (prevalence={n_minority/len(y):.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves (OOF)")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig("pr_curves.png", dpi=150)
plt.close()
print("Saved: pr_curves.png")

df_thresh = pd.DataFrame(threshold_rows)
print("\nOptimal Threshold Analysis:")
print(df_thresh.to_string(index=False))
df_thresh.to_csv("optimal_thresholds.csv", index=False)



# ─────────────────────────────────────────────────────────────────────────────
# 11.  ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for name, (oof_prob, _) in all_oof.items():
    fpr, tpr, _ = roc_curve(y, oof_prob)
    auc = roc_auc_score(y, oof_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (OOF)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()
print("\nSaved: roc_curves.png")




# ─────────────────────────────────────────────────────────────────────────────
# 12.  Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────

n_models = len(all_oof)
ncols = 3
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for ax_i, (name, (oof_prob, oof_pred)) in enumerate(all_oof.items()):
    cm = confusion_matrix(y, oof_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(
        ax=axes[ax_i], colorbar=False, cmap="Blues"
    )
    tn, fp, fn, tp = cm.ravel()
    axes[ax_i].set_title(f"{name}\nFN={fn}  FP={fp}", fontsize=9)

# Hide unused axes
for ax_j in range(ax_i + 1, len(axes)):
    axes[ax_j].set_visible(False)

plt.suptitle("Confusion Matrices (OOF Predictions)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrices.png")

# FN vs FP cost discussion
print("\n--- FN vs FP Cost Discussion ---")
best_xgb_name = "XGBoost (SPW)"
cm = confusion_matrix(y, all_oof[best_xgb_name][1])
tn, fp, fn, tp = cm.ravel()
print(f"  Model : {best_xgb_name}")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")



# ─────────────────────────────────────────────────────────────────────────────
# 13.  Summary Figure  (Bar chart – F1 macro & PR-AUC across all models)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metric_pairs = [("F1 (macro)", "F1 Score (macro)"), ("PR-AUC", "PR-AUC")]
colors = plt.cm.tab10(np.linspace(0, 1, len(df_all)))

for ax, (col, title) in zip(axes, metric_pairs):
    vals = df_all[col].sort_values(ascending=False)
    bars = ax.bar(range(len(vals)), vals.values, color=colors[:len(vals)], edgecolor="black")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

plt.suptitle("Model Comparison – F1 (macro) and PR-AUC", fontsize=12)
plt.tight_layout()
plt.savefig("model_comparison_bar.png", dpi=150)
plt.close()
print("\nSaved: model_comparison_bar.png")