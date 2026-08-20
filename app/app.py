import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import os

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="❤️", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #B23A48; margin-bottom: 0; }
    .subtitle { color: #666; font-size: 1rem; margin-top: 0; }
    div.stButton > button { background-color: #B23A48; color: white; font-weight: 600; border-radius: 8px; }
    div.stButton > button:hover { background-color: #8E2E39; color: white; }
    </style>
""", unsafe_allow_html=True)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "heart_disease_pipeline.pkl")

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Model file not found. Please ensure `heart_disease_pipeline.pkl` exists in the `models/` folder.")
    st.stop()

st.markdown('<p class="main-title">❤️ Heart Disease Risk Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A machine learning model trained on 918 patient records to estimate heart disease risk.</p>', unsafe_allow_html=True)

with st.expander("ℹ️ About this project"):
    st.markdown("""
    This tool uses a classification model (compared across Logistic Regression, KNN,
    Naive Bayes, Decision Tree, and SVM) trained on the UCI Heart Disease dataset.

    **This is an educational data science project — not a medical diagnostic tool.**
    Always consult a qualified healthcare professional for medical advice.
    """)
    st.markdown("**Model in use:** Logistic Regression — 89% accuracy, 0.934 AUC on held-out test data")
st.divider()


st.sidebar.header("🩺 Patient Information")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
    help="ASY = Asymptomatic, ATA = Atypical Angina, NAP = Non-Anginal Pain, TA = Typical Angina"
)
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

cholesterol_unknown = st.sidebar.checkbox("I don't know my cholesterol level")
cholesterol = None if cholesterol_unknown else st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)

fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", -2.6, 6.2, 1.0, step=0.1)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

predict_clicked = st.sidebar.button("🔍 Predict Risk", type="primary", use_container_width=True)


def make_gauge(probability):
    fig, ax = plt.subplots(figsize=(4, 0.6))
    ax.barh([0], [100], color="#E8E8E8", height=0.5)
    color = "#D64545" if probability >= 0.5 else "#3E9C5F"
    ax.barh([0], [probability * 100], color=color, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Risk Probability (%)")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig

def get_top_risk_factors(pipeline, input_df, top_n=3):
    """Returns the top contributing features for this specific prediction,
    using the model's coefficients (Logistic Regression) if available."""
    try:
        classifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        if not hasattr(classifier, "coef_"):
            return None  # only works for linear models like Logistic Regression

        feature_names = preprocessor.get_feature_names_out()
        transformed_input = preprocessor.transform(input_df)

        # handle sparse matrix output from OneHotEncoder
        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()

        contributions = transformed_input[0] * classifier.coef_[0]

        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contributions
        })
        contrib_df["AbsContribution"] = contrib_df["Contribution"].abs()
        contrib_df = contrib_df.sort_values(by="AbsContribution", ascending=False).head(top_n)

        FEATURE_LABELS = {
            "Sex_M": "Male",
            "Sex_F": "Female",
            "ChestPainType_ASY": "Asymptomatic Chest Pain",
            "ChestPainType_ATA": "Atypical Angina",
            "ChestPainType_NAP": "Non-Anginal Pain",
            "ChestPainType_TA": "Typical Angina",
            "RestingECG_Normal": "Normal Resting ECG",
            "RestingECG_ST": "ST-T Wave Abnormality (ECG)",
            "RestingECG_LVH": "Left Ventricular Hypertrophy (ECG)",
            "ExerciseAngina_Y": "Exercise-Induced Angina",
            "ExerciseAngina_N": "No Exercise-Induced Angina",
            "ST_Slope_Up": "Upsloping ST Segment",
            "ST_Slope_Flat": "Flat ST Segment",
            "ST_Slope_Down": "Downsloping ST Segment",
            "Age": "Age",
            "RestingBP": "Resting Blood Pressure",
            "Cholesterol": "Cholesterol Level",
            "FastingBS": "Fasting Blood Sugar",
            "MaxHR": "Maximum Heart Rate",
            "Oldpeak": "ST Depression (Oldpeak)",
            "Cholesterol_Missing": "Cholesterol Not Provided"
        }

        def clean_feature_name(name):
            raw_name = name.replace("cat__", "").replace("num__", "").replace("pass__", "")
            return FEATURE_LABELS.get(raw_name, raw_name.replace("_", " "))

        contrib_df["Feature"] = contrib_df["Feature"].apply(clean_feature_name)


    except Exception:
        return None

def generate_pdf(input_dict, result_text, confidence, timestamp):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(178, 58, 72)
    pdf.cell(0, 12, "Heart Disease Risk Prediction Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated on: {timestamp}", ln=True, align="C")
    pdf.ln(6)

    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Patient Information", ln=True)

    pdf.set_font("Helvetica", "", 11)
    labels = {
        "Age": "Age", "Sex": "Sex", "ChestPainType": "Chest Pain Type",
        "RestingBP": "Resting Blood Pressure (mm Hg)", "Cholesterol": "Cholesterol (mg/dL)",
        "FastingBS": "Fasting Blood Sugar > 120", "RestingECG": "Resting ECG",
        "MaxHR": "Max Heart Rate", "ExerciseAngina": "Exercise-Induced Angina",
        "Oldpeak": "Oldpeak (ST Depression)", "ST_Slope": "ST Slope"
    }
    for key, label in labels.items():
        value = input_dict.get(key, "N/A")
        pdf.cell(0, 8, f"{label}: {value}", ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Prediction Result", ln=True)
    if "High" in result_text:
         pdf.set_text_color(214, 69, 69)
    else:
        pdf.set_text_color(62, 156, 95)
    pdf.set_font("Helvetica", "B", 12)
    
    pdf.cell(0, 8, f"{result_text}", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Model Confidence: {confidence:.1f}%", ln=True)

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, "This report is generated by a machine learning model for educational purposes only. "
                          "It is not a medical diagnosis. Please consult a qualified healthcare professional.")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Built by Usman Haider - Data Science Student Project", ln=True, align="C")

    return bytes(pdf.output())

if predict_clicked:

    input_data = {
        "Age": age, "Sex": sex, "ChestPainType": chest_pain,
        "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak, "ST_Slope": st_slope,
        "Cholesterol_Missing": 1 if cholesterol_unknown else 0
    }
    input_df = pd.DataFrame([input_data])

    with st.spinner("Analyzing patient data..."):
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

    result_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
    confidence = probability * 100 if prediction == 1 else (1 - probability) * 100

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Result")
        if prediction == 1:
             st.error(f"⚠️ **{result_text}**")
        else:
            st.success(f"✅ **{result_text}**")
        st.metric("Model Confidence", f"{confidence:.1f}%")

        # ---- Feature importance / key risk factors ----
        top_factors = get_top_risk_factors(pipeline, input_df)
        if top_factors is not None:
            st.markdown("**Top contributing factors for this prediction:**")
            for _, row in top_factors.iterrows():
                direction = "⬆️ increases risk" if row["Contribution"] > 0 else "⬇️ decreases risk"
                st.write(f"- `{row['Feature']}` — {direction}")
        else:
            st.markdown("**Key inputs summarized:**")
            st.write(f"- Age: {age}, Sex: {sex}, Chest Pain: {chest_pain}\n- ST Slope: {st_slope}, Exercise Angina: {exercise_angina}")

    with col2:
        st.subheader("Risk Meter")
        st.pyplot(make_gauge(probability))

    st.caption(
        "This prediction is based on patterns learned from a specific dataset and "
        "should not replace professional medical advice."
    )

    st.divider()

    # ---- Downloads ----
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    download_df = input_df.copy()
    download_df["Cholesterol"] = "Not provided" if cholesterol_unknown else cholesterol
    download_df["Prediction"] = result_text
    download_df["Confidence (%)"] = round(confidence, 2)
    download_df["Generated On"] = timestamp
    download_df["Created By"] = "Usman Haider (Data Science Student Project)"
    csv_data = download_df.to_csv(index=False).encode("utf-8")
    pdf_data = generate_pdf(input_data, result_text, confidence, timestamp)

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button("📊 Download as CSV", data=csv_data, file_name="heart_disease_result.csv", mime="text/csv", use_container_width=True)
    with dcol2:
        st.download_button("📄 Download as PDF", data=pdf_data, file_name="heart_disease_report.pdf", mime="application/pdf", use_container_width=True)

else:
    st.info("👈 Fill in the patient details in the sidebar and click **Predict Risk** to see results.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey; font-size: 0.85rem;'>"
    "Built by <b>Usman Haider</b> | Data Science Student Project"
    "</p>",
    unsafe_allow_html=True
)