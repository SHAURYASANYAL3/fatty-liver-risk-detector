# 🩺 Fatty Liver Risk Screening Tool (Educational)

An **AI-powered Streamlit web application** that estimates **fatty liver disease risk** using
**medically inspired screening factors**, **early symptoms**, and **lifestyle indicators**.

⚠️ **Important Disclaimer**  
This tool is for **education and awareness only**.  
It **does NOT diagnose disease**, does **NOT replace medical tests**, and must **NOT be used for clinical decisions**.

---

## 🎯 Purpose of This Project

Fatty liver disease often has **no obvious symptoms in early stages**.  
This project demonstrates how **population-level medical risk factors** can be combined with
machine learning to **estimate risk**, **raise awareness**, and **encourage preventive action**.

The goal is:
- Education
- Transparency
- Responsible AI use in healthcare

---

## 🧠 What This Tool Does

- Collects **health, lifestyle, and early symptom information**
- Uses a **medically inspired AI model** trained on **synthetic data**
- Produces a **risk estimate (Low / Moderate / High)**
- Explains **why** certain factors influenced the result
- Uses **plain-language explanations** so non-medical users can understand

---

## 🧬 Medical & Health Features Included

### 👤 Demographic Factors
| Feature | Why it matters |
|------|----------------|
| Age | Risk increases with age |
| Sex | Hormonal differences affect fat metabolism |

---

### ⚖️ Body Composition & Obesity Indicators
| Feature | Simple Explanation |
|------|----------------|
| BMI (Body Mass Index) | Higher body fat increases liver fat storage |
| Waist Circumference | Belly fat directly impacts liver health |

---

### 🍬 Metabolic Health (Core Drivers)
| Feature | Simple Explanation |
|------|----------------|
| Diabetes / High Blood Sugar | Excess sugar is converted into fat by the liver |
| High Cholesterol / Triglycerides (Dyslipidemia) | Extra blood fats often end up in the liver |

> These factors together are commonly referred to as **metabolic syndrome**  
> (a cluster of conditions that overload the liver).

---

### 🍺 Lifestyle Factors
| Feature | Simple Explanation |
|------|----------------|
| Alcohol Intake | Alcohol stresses the liver and promotes fat accumulation |
| Physical Activity | Exercise helps the liver burn fat instead of storing it |

---

### 😴 Early & Common Symptoms (Often Vague)
⚠️ These symptoms are **not specific** to fatty liver but are commonly reported.

| Symptom | What it means |
|------|---------------|
| Persistent Fatigue | Liver stress can affect energy regulation |
| Right Upper Abdominal Discomfort | Mild liver enlargement may cause discomfort |
| Brain Fog / Poor Concentration | Metabolic imbalance may affect mental clarity |

---

### 🧪 Simulated Laboratory Indicator
| Indicator | Plain Explanation |
|--------|------------------|
| ALT/AST Ratio | A liver enzyme balance marker that may suggest liver stress |

⚠️ Normal enzyme levels **do not rule out fatty liver**.

---

## 🤖 AI & Machine Learning Approach

- **Model:** Logistic Regression  
- **Pipeline:**  
  - StandardScaler (normalizes inputs)  
  - Logistic Regression classifier  
- **Training Data:**  
  - Fully **synthetic**, medically inspired data  
  - Designed to reflect realistic risk patterns  
- **Explainability:**  
  - Feature contribution visualization (normalized coefficients)  
  - Helps users understand *why* risk is higher or lower  

---

## 📊 Output Provided

- Calculated BMI
- Estimated risk probability (%)
- Risk category:
  - 🟢 Low
  - 🟡 Moderate
  - 🔴 High
- Feature contribution chart
- Preventive guidance (non-medical advice)

---

## 🖥️ User Interface Highlights

- Clean, calm, healthcare-style layout
- Tooltips explaining medical terms in plain language
- Expandable educational sections
- Progress bar for risk visualization
- Clear warnings about limitations

---

## ⚠️ Ethical & Medical Boundaries

This project **intentionally does NOT**:
- Diagnose fatty liver disease
- Provide treatment or medication advice
- Use real patient data
- Model advanced or emergency symptoms (e.g., jaundice, ascites)

This project **IS**:
- Educational
- Transparent
- Medically cautious
- Suitable for learning, demos, and portfolios

---

## ▶️ Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
