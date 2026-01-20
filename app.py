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
    page_title="Fatty Liver Risk Screening (Educational)",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Fatty Liver Risk Screening Tool")
st.caption(
    "Educational AI-based risk estimation using known medical risk patterns. "
    "⚠️ Not a diagnosis."
)

st.markdown("---")

# =========================
# MEDICAL MODEL (CACHED)
# =========================
@st.cache_resource
def train_model():
    np.random.seed(42)
    N = 900

    data = pd.DataFrame({
        "Age": np.random.randint(18, 75, N),
        "Sex": np.random.choice([0, 1], N),  # 0=female, 1=male
        "BMI": np.clip(np.random.normal(27, 5, N), 18, 45),
        "Waist": np.clip(np.random.normal(95, 12, N), 70, 140),
        "Alcohol": np.random.choice([0, 1, 2], N, p=[0.5, 0.35, 0.15]),
        "Diabetes": np.random.choice([0, 1], N, p=[0.7, 0.3]),
        "Dyslipidemia": np.random.choice([0, 1], N, p=[0.6, 0.4]),
        "Exercise": np.random.choice([0, 1, 2], N, p=[0.35, 0.4, 0.25]),
        "Fatigue": np.random.choice([0, 1], N, p=[0.6, 0.4]),
        "Abdominal_Discomfort": np.random.choice([0, 1], N, p=[0.75, 0.25]),
        "Brain_Fog": np.random.choice([0, 1], N, p=[0.7, 0.3]),
    })

    # Simulated liver enzyme ratio (ALT/AST)
    data["ALT_AST"] = np.clip(
        np.random.normal(1.0 + 0.4 * data["Diabetes"], 0.25, N),
        0.5, 2.5
    )

    # Medically inspired screening risk score
    risk = (
        0.03 * data["Age"] +
        0.04 * data["BMI"] +
        0.03 * data["Waist"] +
        1.2 * data["Diabetes"] +
        1.0 * data["Dyslipidemia"] +
        0.8 * (data["ALT_AST"] > 1).astype(int) +
        0.6 * data["Alcohol"] +
        0.5 * data["Fatigue"] +
        0.4 * data["Abdominal_Discomfort"] +
        0.3 * data["Brain_Fog"] -
        0.7 * (data["Exercise"] == 0).astype(int)
    )

    prob = 1 / (1 + np.exp(-risk))
    threshold = np.percentile(prob, 65)
    data["RiskLabel"] = (prob > threshold).astype(int)

    X = data.drop("RiskLabel", axis=1)
    y = data["RiskLabel"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1200))
    ])

    model.fit(X, y)
    return model, X.columns


model, feature_names = train_model()

# =========================
# USER INPUT UI
# =========================
st.header("👤 Health Information (Screening Level)")
st.caption("All questions reflect known population-level risk factors.")

with st.expander("ℹ️ What does this tool do?"):
    st.markdown("""
This tool estimates **fatty liver risk** using:
- Lifestyle factors
- Metabolic health
- Early non-specific symptoms

It **does not diagnose disease** and **does not replace medical tests**.
""")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 18, 80, 40)
    sex = st.selectbox("Sex", ["Female", "Male"])
    height = st.number_input("Height (cm)", 140, 220, 170)
    weight = st.number_input("Weight (kg)", 35, 200, 75)
    waist = st.number_input(
        "Waist Circumference (cm)",
        60, 160, 95,
        help="Belly fat is more strongly linked to fatty liver than weight alone."
    )

with col2:
    alcohol = st.selectbox(
        "Alcohol Intake",
        ["None", "Occasional", "Frequent"],
        help="Alcohol stresses the liver and increases fat storage."
    )
    diabetes = st.selectbox(
        "Diabetes / High Blood Sugar",
        ["No", "Yes"],
        help="High blood sugar is converted into fat by the liver."
    )
    dyslipidemia = st.selectbox(
        "High Cholesterol / Triglycerides",
        ["No", "Yes"],
        help="Excess blood fats are often stored in the liver."
    )
    exercise = st.selectbox(
        "Physical Activity Level",
        ["Regular", "Occasional", "Rare"],
        help="Exercise helps the liver burn fat."
    )

st.subheader("😴 Early & Common Symptoms (Often Vague)")

col3, col4 = st.columns(2)

with col3:
    fatigue = st.selectbox(
        "Persistent Fatigue",
        ["No", "Yes"],
        help="Liver stress can affect energy regulation."
    )
    abdominal = st.selectbox(
        "Right Upper Abdominal Discomfort",
        ["No", "Yes"],
        help="Fat accumulation may cause mild liver enlargement."
    )

with col4:
    brain_fog = st.selectbox(
        "Poor Concentration / Brain Fog",
        ["No", "Yes"],
        help="Metabolic imbalance can affect mental clarity."
    )
    alt_ast = st.slider(
        "ALT/AST Ratio (If Known)",
        0.5, 2.5, 1.0,
        help="A liver enzyme balance marker. Higher values suggest liver stress."
    )

# =========================
# MAP INPUTS
# =========================
if st.button("🔍 Estimate Risk"):

    bmi = round(weight / ((height / 100) ** 2), 2)

    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": 1 if sex == "Male" else 0,
        "BMI": bmi,
        "Waist": waist,
        "Alcohol": {"None": 0, "Occasional": 1, "Frequent": 2}[alcohol],
        "Diabetes": int(diabetes == "Yes"),
        "Dyslipidemia": int(dyslipidemia == "Yes"),
        "Exercise": {"Regular": 0, "Occasional": 1, "Rare": 2}[exercise],
        "Fatigue": int(fatigue == "Yes"),
        "Abdominal_Discomfort": int(abdominal == "Yes"),
        "Brain_Fog": int(brain_fog == "Yes"),
        "ALT_AST": alt_ast
    }])

    probability = model.predict_proba(input_df)[0][1]
    risk_percent = int(probability * 100)

    st.markdown("---")
    st.subheader("📊 Screening Result")

    st.metric("Calculated BMI", bmi)
    st.progress(risk_percent)

    if risk_percent < 30:
        st.success("🟢 Lower Estimated Risk")
        level = "Low"
    elif risk_percent < 60:
        st.warning("🟡 Moderate Estimated Risk")
        level = "Moderate"
    else:
        st.error("🔴 Higher Estimated Risk")
        level = "High"

    st.metric("Risk Category", level)
    st.metric("Estimated Risk Probability", f"{risk_percent}%")

    # =========================
    # EXPLAINABILITY
    # =========================
    st.subheader("🧠 What Factors Influenced This Estimate?")

    coef = model.named_steps["clf"].coef_[0]
    coef_norm = coef / np.sum(np.abs(coef))

    fig, ax = plt.subplots()
    ax.barh(feature_names, coef_norm)
    ax.set_xlabel("Relative Influence")
    ax.set_title("Feature Contribution (Educational)")

    st.pyplot(fig)

    # =========================
    # GUIDANCE
    # =========================
    st.subheader("🛡️ General Preventive Guidance")

    st.markdown("""
- Maintain healthy body weight and waist size  
- Engage in regular physical activity  
- Limit alcohol intake  
- Control blood sugar and cholesterol  
- Seek medical advice if symptoms persist  

⚠️ **This tool estimates risk only and cannot diagnose fatty liver disease.**
""")

# =========================
# FOOTER
# =========================
st.caption(
    "This educational tool uses simulated data and known medical risk patterns. "
    "It does not replace blood tests, imaging, or professional medical care."
)
