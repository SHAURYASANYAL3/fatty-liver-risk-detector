import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ----------------- Generate synthetic data (adjust to your features) -----------------
np.random.seed(42)
n_samples = 5000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'bmi': np.random.uniform(18, 45, n_samples),
    'waist_cm': np.random.uniform(60, 150, n_samples),
    'diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'high_cholesterol': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
    'alcohol_use': np.random.choice([0, 1, 2], n_samples),  # 0 none, 1 moderate, 2 heavy
    'activity_level': np.random.choice([0, 1, 2], n_samples),  # 0 low, 1 medium, 2 high
    'fatigue': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'right_side_pain': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'alt_ast_ratio': np.random.uniform(0.5, 3.0, n_samples),
    # Add more if you have them
})

# Simple rule-based target (you can make more realistic)
risk_score = (
    0.4 * (data['bmi'] > 30) +
    0.3 * (data['waist_cm'] > 100) +
    0.25 * data['diabetes'] +
    0.2 * (data['alt_ast_ratio'] > 1.5) +
    0.15 * (data['alcohol_use'] == 2) -
    0.1 * (data['activity_level'] == 2) +
    np.random.normal(0, 0.1, n_samples)
)
data['risk'] = (risk_score > 0.5).astype(int)  # binary for classification

data.to_csv("data/synthetic_liver_risk.csv", index=False)

# ----------------- Train -----------------
X = data.drop('risk', axis=1)
y = data['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, preds))
print("AUC-ROC:", roc_auc_score(y_test, proba))

# Save
joblib.dump(model, "models/best_model.pkl")
print("Model saved to models/best_model.pkl")
