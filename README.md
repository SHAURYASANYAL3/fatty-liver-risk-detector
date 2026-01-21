# Fatty Liver Risk Awareness App

AI-powered web tool to estimate fatty liver risk from simple health inputs.  
**For educational/awareness purposes ONLY – not a medical diagnosis.**

⚠️ **Critical Disclaimer**  
This is **NOT** a substitute for professional medical advice, tests (ultrasound, FibroScan, blood work), or diagnosis.  
Always consult a qualified doctor. Results are based on synthetic data patterns and have no clinical validation.

## Live Demo
🚀 Try it now: [https://fatty-liver-risk-detector-u9abevxlcrz9ojxlpsrsxd.streamlit.app/](https://fatty-liver-risk-detector-u9abevxlcrz9ojxlpsrsxd.streamlit.app/)

## Screenshots
<!-- Keep your existing images -->
![Input Form](input_form.png)  
![Risk Result](risk_result.png)  
![Feature Importance](feature_contribution.png)

## Features
- Easy input form (age, BMI, waist, diabetes, ALT/AST ratio, etc.)
- Risk score (0–100%) + category (Low / Medium / High)
- SHAP-based feature importance visualization
- Personalized general health tips
- Clean, mobile-friendly Streamlit UI

## Model & Performance (on synthetic data)
- **Best model**: Random Forest (or XGBoost – whichever you pick)
- **Metrics** (hold-out test set):
  - Accuracy: ~85–92% (placeholder – fill real numbers after training)
  - AUC-ROC: ~0.89
  - F1-score (high-risk class): ~0.82
- Trained on ~5,000 synthetic samples mimicking real risk factors
- Interpretability via SHAP values

## Tech Stack
- Python 3.10+
- Streamlit (UI)
- scikit-learn (ML)
- pandas, numpy
- matplotlib, shap (for explanations)
- joblib (model saving)

## Setup & Run Locally
```bash
git clone https://github.com/SHAURYASANYAL3/fatty-liver-risk-detector.git
cd fatty-liver-risk-detector
pip install -r requirements.txt
streamlit run app.py
