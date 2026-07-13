#!/usr/bin/env python3
"""
01_generate_data.py
Synthetic Loan Approval Dataset with Injected Bias

Generates a realistic loan approval dataset with explicit systemic bias
injected across gender and race to simulate real-world discrimination patterns.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)


def generate_biased_loan_dataset(n_samples=5000, output_path="data/loan_approval_data.csv"):
    """
    Generate synthetic loan approval data with injected bias.

    Bias injection:
        - Male: +0.08 probability boost
        - Female: -0.05 penalty
        - Black: -0.08 vs White: +0.05
        - Hispanic: -0.04 vs Asian: +0.03
    """
    print("=" * 60)
    print("Generating Synthetic Loan Approval Dataset")
    print("=" * 60)

    # === BASE FINANCIAL FEATURES ===
    age = np.random.normal(40, 12, n_samples).clip(18, 75).astype(int)
    income = np.random.lognormal(10.8, 0.6, n_samples)
    credit_score = np.random.normal(680, 80, n_samples).clip(300, 850).astype(int)
    employment_years = np.random.poisson(5, n_samples).clip(0, 40)
    debt_to_income = np.random.beta(2, 5, n_samples) * 100
    loan_amount = np.random.lognormal(11, 0.5, n_samples)

    # === SENSITIVE ATTRIBUTES ===
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])
    race = np.random.choice(
        ['White', 'Black', 'Asian', 'Hispanic', 'Other'],
        n_samples, p=[0.60, 0.13, 0.06, 0.18, 0.03]
    )
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'],
        n_samples, p=[0.35, 0.40, 0.20, 0.05]
    )

    # === BASE APPROVAL PROBABILITY (financial merit) ===
    base_prob = (
        0.30 * (credit_score - 300) / 550 +          # credit score weight
        0.25 * (np.log(income) - 8) / 5 +             # income weight
        0.15 * (1 - debt_to_income / 100) +            # lower DTI is better
        0.10 * (employment_years / 40) +               # employment history
        0.10 * (1 - loan_amount / income / 10)         # loan-to-income ratio
    )

    # === INJECT SYSTEMIC BIAS ===
    gender_bias = np.where(gender == 'Male', 0.08, -0.05)

    race_bias_map = {
        'White': 0.05, 'Asian': 0.03, 'Hispanic': -0.04,
        'Black': -0.08, 'Other': -0.03
    }
    race_bias = np.array([race_bias_map[r] for r in race])

    age_bias = np.where((age >= 30) & (age <= 50), 0.03, -0.02)

    # Combine probabilities
    approval_prob = base_prob + gender_bias + race_bias + age_bias
    approval_prob = np.clip(approval_prob, 0.05, 0.95)

    # Generate target
    approved = np.random.binomial(1, approval_prob)

    # === CREATE DATAFRAME ===
    df = pd.DataFrame({
        'age': age,
        'income': income.round(2),
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income.round(2),
        'loan_amount': loan_amount.round(2),
        'gender': gender,
        'race': race,
        'education': education,
        'approved': approved
    })

    # Derived features
    df['loan_to_income'] = (df['loan_amount'] / df['income']).round(4)
    df['credit_tier'] = pd.cut(
        df['credit_score'],
        bins=[0, 580, 670, 740, 800, 850],
        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    )

    # === SAVE ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # === REPORT ===
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget distribution:")
    print(df['approved'].value_counts(normalize=True))
    print(f"\n--- Approval Rates by Gender ---")
    print(df.groupby('gender')['approved'].agg(['mean', 'count']))
    print(f"\n--- Approval Rates by Race ---")
    print(df.groupby('race')['approved'].agg(['mean', 'count']))
    print(f"\n✅ Dataset saved to {output_path}")

    return df


if __name__ == "__main__":
    df = generate_biased_loan_dataset()
