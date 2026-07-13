#!/usr/bin/env python3
"""
05_fairness_metrics.py
Fairness Metrics Across Sensitive Groups

Computes demographic parity, equal opportunity, equalized odds,
and other fairness metrics by gender and race.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def compute_fairness_metrics(y_true, y_pred, y_proba, sensitive_attr):
    """Compute key fairness metrics by group."""
    results = {}
    groups = sensitive_attr.unique()

    for group in groups:
        mask = sensitive_attr == group
        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_proba[mask]

        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        results[group] = {
            'count': mask.sum(),
            'approval_rate': yp.mean(),
            'true_approval_rate': yt.mean(),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tpr,
            'fpr': fpr,
            'fnr': fnr,
            'auc': roc_auc_score(yt, ypr) if len(np.unique(yt)) > 1 else np.nan,
            'selection_rate': yp.mean(),
            'positive_rate': yt.mean()
        }

    return pd.DataFrame(results).T


def run_fairness_analysis(artifacts_path="models/artifacts.pkl",
                          output_dir="outputs"):
    """Run complete fairness analysis."""

    print("=" * 60)
    print("Fairness Metrics Analysis")
    print("=" * 60)

    # Load artifacts
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    y_test = artifacts['y_test'].values
    y_pred = artifacts['y_pred']
    y_proba = artifacts['y_proba']
    sensitive_test = artifacts['sensitive_test']

    # === GENDER FAIRNESS ===
    print("\n=== FAIRNESS BY GENDER ===")
    gender_fairness = compute_fairness_metrics(y_test, y_pred, y_proba, sensitive_test['gender'])
    print(gender_fairness[['approval_rate', 'precision', 'recall', 'fpr', 'fnr']].to_string())

    gender_gap = gender_fairness['approval_rate'].max() - gender_fairness['approval_rate'].min()
    print(f"\nGender approval rate gap: {gender_gap:.3f} ({gender_gap*100:.1f}pp)")

    # === RACE FAIRNESS ===
    print("\n=== FAIRNESS BY RACE ===")
    race_fairness = compute_fairness_metrics(y_test, y_pred, y_proba, sensitive_test['race'])
    print(race_fairness[['approval_rate', 'precision', 'recall', 'fpr', 'fnr']].to_string())

    race_gap = race_fairness['approval_rate'].max() - race_fairness['approval_rate'].min()
    print(f"\nRace approval rate gap: {race_gap:.3f} ({race_gap*100:.1f}pp)")

    # === FAIRNESS DEFINITIONS CHECK ===
    print("\n=== FAIRNESS CRITERIA CHECK ===")

    # Demographic Parity
    print(f"Demographic Parity (Gender): {gender_gap:.3f} gap → {'VIOLATED' if gender_gap > 0.05 else 'SATISFIED'}")
    print(f"Demographic Parity (Race):   {race_gap:.3f} gap → {'VIOLATED' if race_gap > 0.05 else 'SATISFIED'}")

    # Equal Opportunity (TPR parity)
    tpr_gender_gap = gender_fairness['recall'].max() - gender_fairness['recall'].min()
    tpr_race_gap = race_fairness['recall'].max() - race_fairness['recall'].min()
    print(f"Equal Opportunity (Gender):  {tpr_gender_gap:.3f} TPR gap → {'VIOLATED' if tpr_gender_gap > 0.05 else 'SATISFIED'}")
    print(f"Equal Opportunity (Race):    {tpr_race_gap:.3f} TPR gap → {'VIOLATED' if tpr_race_gap > 0.05 else 'SATISFIED'}")

    # Equalized Odds
    fpr_gender_gap = gender_fairness['fpr'].max() - gender_fairness['fpr'].min()
    fpr_race_gap = race_fairness['fpr'].max() - race_fairness['fpr'].min()
    print(f"Equalized Odds (Gender):     TPR={tpr_gender_gap:.3f}, FPR={fpr_gender_gap:.3f} → {'VIOLATED' if max(tpr_gender_gap, fpr_gender_gap) > 0.05 else 'SATISFIED'}")
    print(f"Equalized Odds (Race):       TPR={tpr_race_gap:.3f}, FPR={fpr_race_gap:.3f} → {'VIOLATED' if max(tpr_race_gap, fpr_race_gap) > 0.05 else 'SATISFIED'}")

    # === VISUALIZATION ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Fairness Metrics Across Groups', fontsize=16, fontweight='bold')

    # Gender fairness
    ax1 = axes[0]
    metrics = ['approval_rate', 'precision', 'recall', 'fpr', 'fnr']
    x = np.arange(len(metrics))
    width = 0.35

    female_vals = [gender_fairness.loc['Female', m] for m in metrics]
    male_vals = [gender_fairness.loc['Male', m] for m in metrics]

    bars1 = ax1.bar(x - width/2, female_vals, width, label='Female', 
                    color='#e74c3c', edgecolor='black')
    bars2 = ax1.bar(x + width/2, male_vals, width, label='Male',
                    color='#3498db', edgecolor='black')

    ax1.set_ylabel('Rate')
    ax1.set_title('Fairness Metrics by Gender', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Approval\nRate', 'Precision', 'Recall', 'FPR', 'FNR'])
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Race fairness
    ax2 = axes[1]
    races = race_fairness.index
    x = np.arange(len(races))
    approval_rates = [race_fairness.loc[r, 'approval_rate'] for r in races]
    colors = ['#2ecc71' if r in ['White', 'Asian'] else '#e74c3c' for r in races]

    bars = ax2.bar(x, approval_rates, color=colors, edgecolor='black')
    ax2.axhline(y=y_test.mean(), color='navy', linestyle='--', linewidth=2,
                label=f'Overall: {y_test.mean():.3f}')
    ax2.set_ylabel('Approval Rate')
    ax2.set_title('Approval Rate by Race', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(races, rotation=30)
    ax2.legend()

    for bar, val in zip(bars, approval_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/04_fairness_metrics.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n✅ Fairness plot saved to {output_dir}/04_fairness_metrics.png")

    return gender_fairness, race_fairness


if __name__ == "__main__":
    run_fairness_analysis()
