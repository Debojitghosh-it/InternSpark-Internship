#!/usr/bin/env python3
"""
06_mitigation.py
Bias Mitigation Strategies

Implements and compares three mitigation approaches:
1. Remove sensitive features
2. Reweighting
3. Threshold optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def compute_reweighting_weights(df_train):
    """Compute sample weights to balance approval rates across groups."""
    weights = np.ones(len(df_train))

    for gender in df_train['gender'].unique():
        for race in df_train['race'].unique():
            mask = (df_train['gender'] == gender) & (df_train['race'] == race)
            group_approval = df_train.loc[mask, 'approved'].mean()
            if group_approval > 0:
                weights[mask] = 1.0 / (group_approval + 0.1)

    weights = weights / weights.mean()
    return weights


def optimize_thresholds(y_proba, y_true, sensitive_attr):
    """Find group-specific thresholds that equalize TPR across groups."""
    groups = sensitive_attr.unique()
    thresholds = {}
    overall_tpr = None

    for group in groups:
        mask = sensitive_attr == group
        yt = y_true[mask]
        yp = y_proba[mask]

        best_thresh = 0.5
        best_diff = float('inf')

        for thresh in np.linspace(0.1, 0.9, 81):
            pred = (yp >= thresh).astype(int)
            tp = ((pred == 1) & (yt == 1)).sum()
            fn = ((pred == 0) & (yt == 1)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

            if overall_tpr is None:
                overall_tpr = tpr
                best_thresh = thresh
            else:
                diff = abs(tpr - overall_tpr)
                if diff < best_diff:
                    best_diff = diff
                    best_thresh = thresh

        thresholds[group] = best_thresh

    return thresholds


def evaluate_model(name, y_true, y_pred, y_proba, sensitive):
    """Evaluate model fairness and performance."""
    results = {'model': name}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['auc'] = roc_auc_score(y_true, y_proba)

    gender_rates = sensitive.groupby(sensitive['gender']).apply(
        lambda g: y_pred[sensitive['gender'] == g.name].mean()
    )
    race_rates = sensitive.groupby(sensitive['race']).apply(
        lambda g: y_pred[sensitive['race'] == g.name].mean()
    )

    results['gender_disparity'] = gender_rates.max() - gender_rates.min()
    results['race_disparity'] = race_rates.max() - race_rates.min()

    return results


def run_mitigation(data_path="data/loan_approval_data.csv",
                   artifacts_path="models/artifacts.pkl",
                   output_dir="outputs"):
    """Run all mitigation strategies and compare results."""

    print("=" * 60)
    print("Bias Mitigation Strategies")
    print("=" * 60)

    df = pd.read_csv(data_path)

    # Encode categorical features
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_edu = LabelEncoder()
    df['gender_enc'] = le_gender.fit_transform(df['gender'])
    df['race_enc'] = le_race.fit_transform(df['race'])
    df['education_enc'] = le_edu.fit_transform(df['education'])

    feature_cols = ['age', 'income', 'credit_score', 'employment_years',
                    'debt_to_income', 'loan_amount', 'loan_to_income']
    all_features = feature_cols + ['gender_enc', 'race_enc', 'education_enc']
    fair_features = feature_cols  # without sensitive attributes

    X_all = df[all_features]
    X_fair = df[fair_features]
    y = df['approved']

    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y, test_size=0.25, random_state=42, stratify=y
    )
    X_train_fair, X_test_fair, _, _ = train_test_split(
        X_fair, y, test_size=0.25, random_state=42, stratify=y
    )

    sensitive_test = df.loc[X_test_all.index, ['gender', 'race']]
    y_test_vals = y_test.values

    # Load baseline artifacts
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    y_pred_base = artifacts['y_pred']
    y_proba_base = artifacts['y_proba']

    # === STRATEGY 1: Remove Sensitive Features ===
    print("\n--- Strategy 1: Remove Sensitive Features ---")
    model_fair = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    model_fair.fit(X_train_fair, y_train)
    y_pred_fair = model_fair.predict(X_test_fair)
    y_proba_fair = model_fair.predict_proba(X_test_fair)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test_vals, y_pred_fair):.4f}")

    # === STRATEGY 2: Reweighting ===
    print("\n--- Strategy 2: Reweighting ---")
    df_train = df.loc[X_train_all.index]
    sample_weights = compute_reweighting_weights(df_train)

    model_reweighted = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    model_reweighted.fit(X_train_all, y_train, sample_weight=sample_weights)
    y_pred_reweighted = model_reweighted.predict(X_test_all)
    y_proba_reweighted = model_reweighted.predict_proba(X_test_all)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test_vals, y_pred_reweighted):.4f}")

    # === STRATEGY 3: Threshold Optimization ===
    print("\n--- Strategy 3: Threshold Optimization ---")
    gender_thresholds = optimize_thresholds(y_proba_base, y_test_vals, sensitive_test['gender'])
    print(f"Optimized thresholds by gender: {gender_thresholds}")

    y_pred_thresholded = y_pred_base.copy()
    for group, thresh in gender_thresholds.items():
        mask = sensitive_test['gender'] == group
        y_pred_thresholded[mask] = (y_proba_base[mask] >= thresh).astype(int)
    print(f"Accuracy: {accuracy_score(y_test_vals, y_pred_thresholded):.4f}")

    # === COMPARISON ===
    print("\n=== MITIGATION COMPARISON ===")
    comparison = pd.DataFrame([
        evaluate_model('Baseline (with sensitive)', y_test_vals, y_pred_base, y_proba_base, sensitive_test),
        evaluate_model('Remove Features', y_test_vals, y_pred_fair, y_proba_fair, sensitive_test),
        evaluate_model('Reweighting', y_test_vals, y_pred_reweighted, y_proba_reweighted, sensitive_test),
        evaluate_model('Threshold Optimized', y_test_vals, y_pred_thresholded, y_proba_base, sensitive_test)
    ])
    print(comparison.to_string(index=False))

    # === VISUALIZATION ===
    fig, ax = plt.subplots(figsize=(10, 6))

    models = comparison['model']
    x = np.arange(len(models))
    width = 0.25

    acc_vals = comparison['accuracy']
    gd_vals = comparison['gender_disparity']
    rd_vals = comparison['race_disparity']

    bars1 = ax.bar(x - width, acc_vals, width, label='Accuracy', 
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x, gd_vals, width, label='Gender Disparity',
                   color='#e74c3c', edgecolor='black')
    bars3 = ax.bar(x + width, rd_vals, width, label='Race Disparity',
                   color='#f39c12', edgecolor='black')

    ax.set_ylabel('Score')
    ax.set_title('Mitigation Strategy Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline', 'Remove\nFeatures', 'Reweighted', 'Threshold\nOptimized'], 
                       fontsize=10)
    ax.legend()
    ax.set_ylim(0, 0.7)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/05_mitigation_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n✅ Mitigation plot saved to {output_dir}/05_mitigation_comparison.png")

    return comparison


if __name__ == "__main__":
    run_mitigation()
