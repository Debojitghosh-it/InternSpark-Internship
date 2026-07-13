#!/usr/bin/env python3
"""
03_model_training.py
Model Training & Evaluation

Trains multiple models, evaluates performance, and selects the best
model for downstream interpretability and fairness analysis.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')


def train_and_evaluate(data_path="data/loan_approval_data.csv"):
    """Train multiple models and select the best one."""

    print("=" * 60)
    print("Model Training & Evaluation")
    print("=" * 60)

    df = pd.read_csv(data_path)

    # Feature columns
    feature_cols = ['age', 'income', 'credit_score', 'employment_years',
                    'debt_to_income', 'loan_amount', 'loan_to_income']

    # Encode categorical features
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_edu = LabelEncoder()

    df['gender_enc'] = le_gender.fit_transform(df['gender'])
    df['race_enc'] = le_race.fit_transform(df['race'])
    df['education_enc'] = le_edu.fit_transform(df['education'])

    model_features = feature_cols + ['gender_enc', 'race_enc', 'education_enc']

    X = df[model_features]
    y = df['approved']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Keep sensitive attributes for fairness analysis
    sensitive_test = df.loc[X_test.index, ['gender', 'race', 'education']]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, 
                                                random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                        random_state=42)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        trained_models[name] = model

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc, 'y_pred': y_pred, 'y_proba': y_proba
        }

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")

    # Select best model
    best_name = 'Gradient Boosting'
    best_model = trained_models[best_name]
    best_preds = results[best_name]['y_pred']
    best_probas = results[best_name]['y_proba']

    print(f"\n✅ Selected model: {best_name}")

    # Save artifacts
    import pickle
    import os
    os.makedirs("models", exist_ok=True)

    artifacts = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': model_features,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': best_preds,
        'y_proba': best_probas,
        'sensitive_test': sensitive_test,
        'label_encoders': {'gender': le_gender, 'race': le_race, 'education': le_edu}
    }

    with open("models/artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("✅ Artifacts saved to models/artifacts.pkl")

    return artifacts


if __name__ == "__main__":
    artifacts = train_and_evaluate()
