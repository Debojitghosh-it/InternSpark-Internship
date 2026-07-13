#!/usr/bin/env python3
"""
04_interpretability.py
SHAP & LIME Explainability Analysis

Computes global feature importance and local explanations to understand
model predictions and detect usage of sensitive attributes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')


def lime_local_explanation(model, X_instance, X_background, feature_names,
                           num_samples=1000, kernel_width=0.75):
    """
    Simplified LIME: Perturb instance, weight by proximity, fit linear model.
    """
    from sklearn.linear_model import Ridge

    instance = X_instance.values.reshape(1, -1)
    n_features = len(feature_names)

    # Generate perturbed samples
    stds = np.std(X_background.values, axis=0)
    perturbed = np.random.normal(0, 1, (num_samples, n_features))
    perturbed = instance + perturbed * stds * 0.5
    perturbed = np.clip(perturbed, X_background.min(), X_background.max())

    # Get model predictions
    predictions = model.predict_proba(perturbed)[:, 1]

    # Compute proximity weights (exponential kernel)
    distances = np.sqrt(np.sum((perturbed - instance) ** 2, axis=1))
    weights = np.exp(-distances ** 2 / (2 * kernel_width ** 2))

    # Fit weighted linear regression
    lime_model = Ridge(alpha=1.0)
    lime_model.fit(perturbed, predictions, sample_weight=weights)

    # Feature contributions
    contributions = lime_model.coef_ * (instance[0] - np.mean(X_background, axis=0))

    return pd.DataFrame({
        'feature': feature_names,
        'contribution': contributions,
        'coefficient': lime_model.coef_
    }).sort_values('contribution', key=abs, ascending=False)


def run_interpretability(artifacts_path="models/artifacts.pkl",
                         output_dir="outputs"):
    """Run SHAP-style and LIME interpretability analysis."""

    print("=" * 60)
    print("Model Interpretability (SHAP / LIME)")
    print("=" * 60)

    # Load artifacts
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts['model']
    X_test = artifacts['X_test']
    y_test = artifacts['y_test']
    y_pred = artifacts['y_pred']
    y_proba = artifacts['y_proba']
    feature_names = artifacts['feature_names']
    X_train = artifacts.get('X_train')  # may not exist

    # Reconstruct X_train from model training if needed
    if X_train is None:
        # Use test set as background for explanations
        X_background = X_test
    else:
        X_background = X_train

    # === GLOBAL FEATURE IMPORTANCE ===
    print("\n--- Global Feature Importance ---")
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    print(fi_df.to_string(index=False))

    # Flag sensitive attributes
    sensitive_importance = fi_df[fi_df['feature'].isin(['gender_enc', 'race_enc', 'education_enc'])]
    print(f"\n⚠️  Sensitive attribute importance: {sensitive_importance['importance'].sum():.1%}")

    # === LOCAL EXPLANATIONS (LIME-style) ===
    print("\n--- Local Explanations ---")

    # Explain a few instances
    for idx in [42, 100, 250]:
        explanation = lime_local_explanation(
            model, X_test.iloc[idx], X_background, feature_names
        )
        print(f"\nInstance {idx} - True: {y_test.iloc[idx]}, Pred: {y_pred[idx]}, Proba: {y_proba[idx]:.3f}")
        print(explanation.head(5).to_string(index=False))

    # === VISUALIZATION ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Interpretability', fontsize=16, fontweight='bold')

    # Global feature importance
    ax1 = axes[0]
    fi_df_sorted = fi_df.sort_values('importance', ascending=True)
    colors = ['#e74c3c' if 'enc' in f else '#3498db' for f in fi_df_sorted['feature']]
    ax1.barh(fi_df_sorted['feature'], fi_df_sorted['importance'], color=colors,
             edgecolor='black')
    ax1.set_xlabel('Importance')
    ax1.set_title('Global Feature Importance\n(Red = Sensitive Attributes)', fontweight='bold')

    # Local explanation waterfall
    ax2 = axes[1]
    instance_idx = 42
    instance = X_test.iloc[instance_idx]
    base_value = y_test.mean()
    prediction = y_proba[instance_idx]

    contributions = []
    for feat in feature_names:
        feat_idx = feature_names.index(feat)
        contrib = (instance.iloc[feat_idx] - X_background[feat].mean()) * importance[feat_idx] * 10
        contributions.append(contrib)

    contrib_df = pd.DataFrame({
        'feature': feature_names,
        'contribution': contributions
    }).sort_values('contribution', key=abs, ascending=True)

    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in contrib_df['contribution']]
    ax2.barh(contrib_df['feature'], contrib_df['contribution'], color=colors,
             edgecolor='black')
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('Contribution to Prediction')
    ax2.set_title(f'Local Explanation (Instance #{instance_idx})\nPred: {prediction:.3f}', 
                  fontweight='bold')

    plt.tight_layout()
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/02_feature_importance.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n✅ Interpretability plot saved to {output_dir}/02_feature_importance.png")

    return fi_df


if __name__ == "__main__":
    run_interpretability()
