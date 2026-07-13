#!/usr/bin/env python3
"""
02_eda_bias_detection.py
Exploratory Data Analysis & Initial Bias Detection

Performs EDA, visualizes distributions, and runs statistical tests
to detect bias across sensitive groups.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


def run_eda_bias_detection(data_path="data/loan_approval_data.csv",
                           output_dir="outputs"):
    """Run full EDA and bias detection pipeline."""

    print("=" * 60)
    print("EDA & Bias Detection")
    print("=" * 60)

    df = pd.read_csv(data_path)

    # === STATISTICAL BIAS TESTS ===
    print("\n--- Statistical Bias Tests ---")

    # Gender
    contingency_gender = pd.crosstab(df['gender'], df['approved'])
    chi2_g, p_g, _, _ = chi2_contingency(contingency_gender)
    print(f"Gender bias: chi2={chi2_g:.4f}, p-value={p_g:.6f}")

    # Race
    contingency_race = pd.crosstab(df['race'], df['approved'])
    chi2_r, p_r, _, _ = chi2_contingency(contingency_race)
    print(f"Race bias:   chi2={chi2_r:.4f}, p-value={p_r:.6f}")

    # Education
    contingency_edu = pd.crosstab(df['education'], df['approved'])
    chi2_e, p_e, _, _ = chi2_contingency(contingency_edu)
    print(f"Education:   chi2={chi2_e:.4f}, p-value={p_e:.6f}")

    # === VISUALIZATION ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Dataset Overview & Bias Detection', fontsize=16, fontweight='bold', y=1.02)

    # 1. Approval rate by gender
    ax1 = axes[0, 0]
    gender_rates = df.groupby('gender')['approved'].mean()
    colors_g = ['#e74c3c' if v < 0.55 else '#2ecc71' for v in gender_rates.values]
    bars1 = ax1.bar(gender_rates.index, gender_rates.values, color=colors_g, 
                    edgecolor='black', linewidth=1.2)
    ax1.axhline(y=df['approved'].mean(), color='navy', linestyle='--', linewidth=2,
                label=f'Overall: {df["approved"].mean():.3f}')
    ax1.set_ylabel('Approval Rate')
    ax1.set_title('Approval Rate by Gender', fontweight='bold')
    ax1.set_ylim(0, 1)
    for bar, val in zip(bars1, gender_rates.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.legend()

    # 2. Approval rate by race
    ax2 = axes[0, 1]
    race_rates = df.groupby('race')['approved'].mean().sort_values(ascending=False)
    colors_r = ['#2ecc71' if v >= df['approved'].mean() else '#e74c3c' for v in race_rates.values]
    bars2 = ax2.bar(race_rates.index, race_rates.values, color=colors_r,
                    edgecolor='black', linewidth=1.2)
    ax2.axhline(y=df['approved'].mean(), color='navy', linestyle='--', linewidth=2,
                label=f'Overall: {df["approved"].mean():.3f}')
    ax2.set_ylabel('Approval Rate')
    ax2.set_title('Approval Rate by Race', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars2, race_rates.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.legend()

    # 3. Credit score distribution
    ax3 = axes[0, 2]
    df[df['approved']==1]['credit_score'].hist(bins=30, alpha=0.6, label='Approved', 
                                                color='#2ecc71', ax=ax3)
    df[df['approved']==0]['credit_score'].hist(bins=30, alpha=0.6, label='Rejected',
                                                color='#e74c3c', ax=ax3)
    ax3.set_xlabel('Credit Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Credit Score Distribution', fontweight='bold')
    ax3.legend()

    # 4. Income vs Loan Amount
    ax4 = axes[1, 0]
    approved_mask = df['approved'] == 1
    ax4.scatter(df[~approved_mask]['income'], df[~approved_mask]['loan_amount'],
                alpha=0.3, c='#e74c3c', label='Rejected', s=10)
    ax4.scatter(df[approved_mask]['income'], df[approved_mask]['loan_amount'],
                alpha=0.3, c='#2ecc71', label='Approved', s=10)
    ax4.set_xlabel('Annual Income ($)')
    ax4.set_ylabel('Loan Amount ($)')
    ax4.set_title('Income vs Loan Amount', fontweight='bold')
    ax4.legend()

    # 5. Correlation heatmap
    ax5 = axes[1, 1]
    numeric_cols = ['age', 'income', 'credit_score', 'employment_years',
                    'debt_to_income', 'loan_amount', 'loan_to_income', 'approved']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax5,
                square=True, linewidths=0.5)
    ax5.set_title('Feature Correlation Matrix', fontweight='bold')

    # 6. Approval by credit tier
    ax6 = axes[1, 2]
    tier_rates = df.groupby('credit_tier')['approved'].mean()
    colors_t = ['#e74c3c', '#f39c12', '#2ecc71', '#27ae60', '#1abc9c']
    bars6 = ax6.bar(tier_rates.index, tier_rates.values, color=colors_t,
                    edgecolor='black', linewidth=1.2)
    ax6.set_ylabel('Approval Rate')
    ax6.set_title('Approval Rate by Credit Tier', fontweight='bold')
    ax6.set_ylim(0, 1)
    for bar, val in zip(bars6, tier_rates.values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/01_eda_bias_detection.png', dpi=150, 
                bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n✅ EDA plot saved to {output_dir}/01_eda_bias_detection.png")

    return df


if __name__ == "__main__":
    run_eda_bias_detection()
