# Customer Churn Prediction

Predicts which telecom customers are likely to churn, using the real IBM
Telco Customer Churn dataset (7,043 customers, 21 features).

**One-line summary (for resume/interviews):** Built and compared 3
classification models (Logistic Regression, Random Forest, XGBoost) to
predict customer churn on an imbalanced dataset, using SMOTE for
class balancing and F1/ROC-AUC (not accuracy) for honest evaluation,
plus feature importance analysis to explain the key churn drivers.

## What's inside
```
churn-prediction/
├── churn_prediction.ipynb   # Full analysis, already executed - open and read straight away
├── build_notebook.py         # Script that generates + runs the notebook (for re-running from scratch)
├── data/telco_churn.csv      # Real dataset (IBM Telco Customer Churn)
├── outputs/                  # Saved chart images (EDA, confusion matrices, feature importance)
└── requirements.txt
```

## What the notebook covers
1. **EDA** — churn rate (~27%), churn vs tenure, churn vs contract type
2. **Cleaning** — fixing `TotalCharges` (stored as text with blanks)
3. **Encoding & scaling** — one-hot encoding categoricals, scaling numerics
4. **SMOTE** — balances the training set only (never touches the test set —
   that would leak information and fake your results)
5. **3 models compared via 5-fold cross-validation** on F1 score
6. **Final test-set evaluation** — precision/recall/F1/ROC-AUC + confusion
   matrices for each model
7. **Feature importance** — which factors (contract type, tenure, monthly
   charges, tech support) actually drive churn
8. **Model selection reasoning** — why recall matters more than precision
   here (missing a churner costs more than a false alarm)

## How to run it

```bash
git clone <your-repo-url>
cd churn-prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

The notebook is already executed and saved with all outputs, so you can
also just open it and read through without re-running anything. To
regenerate everything from scratch:
```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace churn_prediction.ipynb
```

## Push to GitHub

```bash
git init
git add .
git commit -m "Customer churn prediction - EDA, SMOTE, model comparison, feature importance"
git branch -M main
git remote add origin https://github.com/<your-username>/churn-prediction.git
git push -u origin main
```

## Key results (from the executed notebook)
- Dataset churn rate: 26.5% (imbalanced — accuracy alone would be misleading)
- Cross-validation F1 (on SMOTE-balanced training data): Logistic Regression
  0.81, Random Forest 0.85, XGBoost 0.84
- Real-world test-set F1 (true 27% churn rate, no SMOTE applied to test
  data): ~0.58–0.59 — notice this is much lower than CV F1. That gap is a
  genuinely good interview talking point: it shows the model isn't
  overfitting to synthetic balance, and that real churn prediction is a
  genuinely hard, imbalanced problem, not something a notebook trick fixes.
- Top churn drivers (XGBoost feature importance): contract type
  (month-to-month), tenure, monthly charges, lack of online security/tech
  support add-ons

## Interview talking points
- "Why SMOTE only on the training set, not the test set?" → applying it to
  test data would leak synthetic samples into evaluation and inflate scores
  artificially — the test set has to reflect the real world.
- "Why F1/ROC-AUC instead of accuracy?" → with 73%/27% class split, a model
  that always predicts 'no churn' gets 73% accuracy while being useless.
- "Why recall over precision for churn?" → a missed churner (false
  negative) costs the business a lost customer; a false alarm (false
  positive) just costs a wasted retention email — asymmetric cost, so
  recall matters more.
- "What would you improve?" → hyperparameter tuning (GridSearch/Optuna),
  try SMOTE variants (ADASYN, Borderline-SMOTE), add cost-sensitive
  learning instead of oversampling, deploy as an API for real-time scoring.
