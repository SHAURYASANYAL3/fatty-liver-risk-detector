import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Fatty Liver Risk Detector",
    layout="centered"
)

st.title("🧠 AI-Based Fatty Liver Risk Detector")
st.caption("⚠️ Early risk screening & awareness tool (Not a medical diagnosis)")

# =========================
# SYNTHETIC DATA GENERATION
# =========================
np.random.seed(42)
DATA_SIZE = 500

bmi = np.clip(np.random.normal(27, 5, DATA_SIZE), 18, 45)

data = pd.DataFrame({
    "BMI": bmi,
    "Alcohol": np.random.choice([0, 1, 2], DATA_SIZE, p=[0.4, 0.4, 0.2]),
    "Diabetes": np.random.choice([0, 1], DATA_SIZE, p=[0.7, 0.3]),
    "Cholesterol": np.random.choice([0, 1], DATA_SIZE, p=[0.6, 0.4]),
    "Exercise": np.random.choice([0, 1, 2], DATA_SIZE, p=[0.4, 0.35, 0.25]),
    "Fatigue": np.random.choice([0, 1], DATA_SIZE, p=[0.6, 0.4])
})

# Risk logic (semi-realistic)
risk_score = (
    0.04 * data["BMI"] +
    0.8 * data["Alcohol"] +
    1.0 * data["Diabetes"] +
    0.9 * data["Cholesterol"] +
    0.6 * data["Exercise"] +
    0.7 * data["Fatigue"]
)

prob = 1 / (1 + np.exp(-risk_score))
data["FattyLiver"] = (prob > 0.6).astype(int)

X = data.drop("FattyLiver", axis=1)
y = data["FattyLiver"]

# =========================
# MODEL PIPELINE
# =========================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

# =========================
# USER INPUT
# =========================
st.header("👤 Patient Health Information")

col1, col2 = st.columns(2)

with col1:
    height = st.number_input("Height (cm)", 100, 220, 170)
    alcohol = st.selectbox("Alcohol Intake", ["None", "Occasional", "Frequent"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    fatigue = st.selectbox("Chronic Fatigue", ["No", "Yes"])

with col2:
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    cholesterol = st.selectbox("High Cholesterol", ["No", "Yes"])
    exercise = st.selectbox("Physical Activity", ["Regular", "Occasional", "None"])

alcohol_map = {"None": 0, "Occasional": 1, "Frequent": 2}
exercise_map = {"Regular": 0, "Occasional": 1, "None": 2}

# =========================
# PREDICTION
# =========================
if st.button("🔍 Analyze Risk"):
    bmi_user = weight / ((height / 100) ** 2)

    input_df = pd.DataFrame([{
        "BMI": bmi_user,
        "Alcohol": alcohol_map[alcohol],
        "Diabetes": int(diabetes == "Yes"),
        "Cholesterol": int(cholesterol == "Yes"),
        "Exercise": exercise_map[exercise],
        "Fatigue": int(fatigue == "Yes")
    }])

    probability = model.predict_proba(input_df)[0][1]
    risk_percent = int(probability * 100)

    st.subheader("📊 Risk Assessment Result")
    st.metric("Calculated BMI", f"{bmi_user:.2f}")
    st.progress(risk_percent)

    if risk_percent < 30:
        st.success("🟢 LOW RISK")
        risk_level = "Low"
    elif risk_percent < 60:
        st.warning("🟡 MODERATE RISK")
        risk_level = "Moderate"
    else:
        st.error("🔴 HIGH RISK")
        risk_level = "High"

    st.metric("Risk Category", risk_level)
    st.metric("Estimated Risk Probability", f"{risk_percent}%")

    # =========================
    # EXPLAINABLE AI
    # =========================
    st.subheader("🧾 Risk Factor Contribution")

    coef = model.named_steps["clf"].coef_[0]
    features = X.columns

    coef_norm = coef / np.sum(np.abs(coef))

    fig, ax = plt.subplots()
    ax.barh(features, coef_norm)
    ax.set_title("Normalized Feature Influence")
    ax.set_xlabel("Relative Impact")

    st.pyplot(fig)

    # =========================
    # HEALTH GUIDANCE
    # =========================
    st.subheader("🛡️ Preventive Guidance")

    st.markdown("""
- Reduce or avoid alcohol consumption  
- Maintain gradual weight loss (5–10%)  
- Exercise at least 30 minutes daily  
- Prefer whole foods over processed foods  
- Avoid self-medication  

⚠️ Always consult a healthcare professional for diagnosis and treatment.
""")

# =========================
# FOOTER
# =========================
st.caption(
    "This tool demonstrates AI-assisted health risk screening using synthetic data. "
    "Not intended for clinical use."
)
