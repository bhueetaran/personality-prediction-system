import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 🧠 MBTI Personality Type Predictor
st.set_page_config(page_title="MBTI Personality Predictor", page_icon="🧠", layout="centered")
st.title("🧠 MBTI Personality Type Predictor")
st.markdown("Predict your MBTI personality type based on your traits and preferences.")

st.divider()

# 📦 Load Model
model_path = Path("best_model.joblib")
encoder_path = Path("label_encoder.joblib")

if not model_path.exists():
    st.error("❌ 'best_model.joblib' not found in folder.")
    st.stop()

# Load the pipeline and label encoder
model_pipeline = joblib.load(model_path)
le = joblib.load(encoder_path) if encoder_path.exists() else None
st.success("✅ Model loaded successfully!")

# 🧾 User Input
st.header("🧍 Enter Your Details")

# Education level mapping (from training data)
education_mapping = {
    "High School": 0,
    "Bachelors": 1,
    "Masters": 1,
    "PhD": 1
}

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education_label = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
    education = education_mapping[education_label]
with col2:
    interest = st.selectbox("Interest Field", ["Technology", "Arts", "Science", "Business", "Sports", "Other"])
    introversion = st.slider("Introversion Score (0-10)", 0.0, 10.0, 5.0, step=0.1)
    sensing = st.slider("Sensing Score (0-10)", 0.0, 10.0, 5.0, step=0.1)
    thinking = st.slider("Thinking Score (0-10)", 0.0, 10.0, 5.0, step=0.1)
    judging = st.slider("Judging Score (0-10)", 0.0, 10.0, 5.0, step=0.1)

st.divider()

# ⚙️ Prediction
if st.button("🔍 Predict Personality Type"):
    # Create sample DataFrame with the exact column names and order
    sample = pd.DataFrame({
        "Age": [age],
        "Education": [education],
        "Introversion Score": [introversion],
        "Sensing Score": [sensing],
        "Thinking Score": [thinking],
        "Judging Score": [judging],
        "Gender": [gender],
        "Interest": [interest]
    })

    try:
        # Use the full pipeline for prediction
        pred_encoded = model_pipeline.predict(sample)
        pred_proba = model_pipeline.predict_proba(sample)
        
        if le is not None:
            pred_label = le.inverse_transform(pred_encoded)[0]
            
            st.subheader(f"🌟 Predicted Personality Type: **{pred_label}**")
            st.markdown("🧩 This personality type reflects your decision-making and perception style.")
        else:
            st.info(f"Predicted Label Code: {pred_encoded[0]}")
    except Exception as e:
        st.error(f"🚨 Prediction failed: {str(e)}")
        st.info("Please check that all inputs are valid and the model was trained with the same data.")

st.divider()

# 📘 MBTI Information
with st.expander("📘 Learn about MBTI Personality Types"):
    st.markdown("""
| Type | Meaning | Traits |
|:------|:---------|:--------|
| **ISTJ** | Inspector | Responsible, serious, practical |
| **ISFJ** | Protector | Loyal, considerate, organized |
| **INFJ** | Counselor | Insightful, idealistic, reserved |
| **INTJ** | Mastermind | Strategic, independent, confident |
| **ISTP** | Crafter | Logical, hands-on, adaptable |
| **ISFP** | Artist | Gentle, creative, spontaneous |
| **INFP** | Mediator | Empathetic, imaginative, idealistic |
| **INTP** | Thinker | Analytical, curious, abstract |
| **ESTP** | Dynamo | Energetic, persuasive, action-oriented |
| **ESFP** | Performer | Outgoing, friendly, fun-loving |
| **ENFP** | Champion | Enthusiastic, expressive, creative |
| **ENTP** | Visionary | Inventive, quick-witted, curious |
| **ESTJ** | Director | Decisive, organized, practical |
| **ESFJ** | Caregiver | Sociable, caring, loyal |
| **ENFJ** | Teacher | Charismatic, supportive, responsible |
| **ENTJ** | Commander | Assertive, efficient, strategic |
""")

st.caption("💡 Developed using Streamlit · Model trained with Scikit-Learn & XGBoost")
