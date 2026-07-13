# Model Selection Report — SMS Spam Detection

## 1. Problem & Dataset

The goal was to build a supervised binary classifier that labels SMS messages as **spam** or **ham** (not spam). We used the SMS Spam Collection dataset — 5,572 labeled messages, of which 747 (13.4%) are spam and 4,825 (86.6%) are ham. This class imbalance is the main reason we evaluate models on precision, recall, F1, and ROC-AUC rather than accuracy alone.

## 2. Approach

- **Preprocessing:** lowercasing, removal of URLs/emails/digits/punctuation, whitespace normalization.
- **Feature extraction:** TF-IDF vectorization (unigrams + bigrams, top 5,000 features, English stopwords removed), fit only on the training split to avoid leakage.
- **Split:** 80/20 stratified train/test split, plus 5-fold stratified cross-validation on the training set.
- **Models compared:**
  - **Logistic Regression** — linear baseline, fast and interpretable
  - **Random Forest** (200 trees) — non-linear ensemble
  - **Multinomial Naive Bayes** — classic, lightweight text-classification baseline

## 3. Results

### Cross-validation (training set, 5-fold mean)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.947 | 0.982 | 0.616 | 0.756 | 0.988 |
| Random Forest | 0.974 | 0.990 | 0.816 | 0.895 | 0.985 |
| Naive Bayes | 0.971 | 0.998 | 0.784 | 0.878 | 0.984 |

### Held-out test set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.969 | 1.000 | 0.765 | 0.867 | 0.987 |
| **Random Forest** | **0.977** | 1.000 | **0.826** | **0.904** | 0.978 |
| Naive Bayes | 0.967 | 1.000 | 0.752 | 0.858 | **0.987** |

All three models achieved perfect precision on the test set — no ham message was misclassified as spam. The models differ mainly in **recall**: how much spam they actually catch.

## 4. Model Selection

**Random Forest is the recommended model.** It has the highest F1 score (0.904) and the highest recall (0.826) while keeping precision at 1.0, meaning it catches more spam than the alternatives without ever wrongly blocking a legitimate message. Its ROC-AUC (0.978) is marginally lower than Logistic Regression and Naive Bayes (~0.987), but the difference is small and less relevant than F1/recall for a deployed spam filter, where the practical cost of a missed spam message matters more than a small AUC gap.

**Logistic Regression** is a reasonable second choice if interpretability or training speed is a priority — it's a linear model with far fewer parameters and near-identical ROC-AUC to Naive Bayes.

**Naive Bayes**, while the fastest to train and simplest, has the lowest recall of the three, meaning more spam slips through.

## 5. Limitations

- **Dataset size and source:** ~5.5k messages from a single public collection; performance may not generalize to messages with different slang, languages, or spam tactics (e.g. modern phishing SMS).
- **Class imbalance:** only 13% of messages are spam; results should be interpreted with this in mind, and a larger/more balanced dataset would give a more robust estimate.
- **No hyperparameter tuning:** default/lightly-tuned settings were used for all models; grid search (especially for Random Forest depth/estimators) could likely improve recall further.
- **Static evaluation:** spam tactics evolve over time; a production system would need periodic retraining on fresh data.
