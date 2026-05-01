# Question 2: Classification – Extreme Imbalance, Cost-Sensitive Learning, and Calibration

## 🎯 Objective
The primary goal of this task is to address severe class imbalance using advanced resampling techniques, cost-sensitive learning methods, and probability calibration.

## 📊 Dataset
* **Dataset Used:** Extremely imbalanced dataset (credit card fraud dataset).
* **Imbalance Ratio (IR):** Report the specific IR for the selected dataset.

## 🛠️ Implementation Tasks

### 1. Resampling Strategies
Implement three strategies using a pipeline to avoid data leakage:
* **SMOTE** (Synthetic Minority Over-sampling Technique).
* **ADASYN** (Adaptive Synthetic).
* **Random Undersampling**.

### 2. Model Implementation & Cost-Sensitivity
Compare at least four classifiers:
* **Logistic Regression** (Baseline).
* **Random Forest**.
* **XGBoost** (using `scale_pos_weight`).
* **Neural Network (MLP)** (using class weights inversely proportional to frequencies).

### 3. Probability Calibration
Perform calibration on the best two models:
* **Methods:** Platt scaling or isotonic regression.
* **Evaluation:** Assessment via Brier scores and reliability diagrams (calibration curves).

### 4. Evaluation Metrics
* Precision, Recall, F1-score (macro and micro).
* ROC-AUC, PR-AUC (Precision-Recall AUC).
* Matthews Correlation Coefficient (MCC).
* Balanced Accuracy.

## 🔍 Analysis & Discussion
* **Threshold Optimization:** Determine optimal threshold using F1-maximization or cost-weighted decision.
* **Precision-Recall Trade-off:** Analyze the trade-off using PR curves.
* **Cost Analysis:** Discuss the confusion matrix and the real-world cost of false negatives vs. false positives.
