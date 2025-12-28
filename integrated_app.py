"""
Integrated Personality Predictor - OCEAN & MBTI Models
Modern Dark Theme Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Theme CSS Styling
st.markdown("""
    <style>
    body {
        background-color: #0f1419;
        color: #ffffff;
    }
    
    .main {
        background-color: #0f1419;
        color: #ffffff;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #00d4ff;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .input-section {
        background: #1e2530;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #00d4ff;
        margin-bottom: 1.5rem;
    }
    
    .input-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 1rem;
    }
    
    .result-box {
        background: #1e2530;
        padding: 2.5rem;
        border-radius: 15px;
        border: 2px solid #00d4ff;
    }
    
    .result-box-mbti {
        border-color: #ff006e;
    }
    
    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #00d4ff;
    }
    
    .result-title-mbti {
        color: #ff006e;
    }
    
    .ocean-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #0f1419;
        border-radius: 8px;
        border-left: 4px solid #00d4ff;
    }
    
    .ocean-label {
        font-weight: 600;
        color: #ffffff;
    }
    
    .ocean-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .ocean-question {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.3rem;
        font-style: italic;
    }
    
    .mbti-result {
        text-align: center;
        padding: 2rem;
    }
    
    .mbti-type {
        font-size: 3rem;
        font-weight: 900;
        color: #ff006e;
        letter-spacing: 0.2rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4) !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_ocean_model():
    try:
        with open('Ocean/personality_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_mbti_model():
    try:
        model_path = Path("MBTI/best_model.joblib")
        encoder_path = Path("MBTI/label_encoder.joblib")
        
        if not model_path.exists():
            return None, None
        
        model_pipeline = joblib.load(model_path)
        le = joblib.load(encoder_path) if encoder_path.exists() else None
        return model_pipeline, le
    except Exception as e:
        return None, None

ocean_model_data = load_ocean_model()
mbti_pipeline, mbti_encoder = load_mbti_model()

# Header
st.markdown('<h1 class="main-header">🧠 Personality Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover Your Personality with OCEAN & MBTI</p>', unsafe_allow_html=True)

st.divider()

# Input Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<div class="input-title">� Your Information</div>', unsafe_allow_html=True)


col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col3:
    education_mapping = {
        "High School": 0,
        "Bachelors": 1,
        "Masters": 1,
        "PhD": 1
    }
    education_label = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
    education = education_mapping[education_label]

with col4:
    interest = st.selectbox("Interest", ["Technology", "Arts", "Science", "Business", "Sports", "Other"])

st.markdown('</div>', unsafe_allow_html=True)

# OCEAN Questions Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<div class="input-title">🌊 OCEAN Assessment (1-5 Scale)</div>', unsafe_allow_html=True)

ocean_questions = {
    'E': ["I am the life of the party", "I start conversations", "I am outgoing and sociable"],
    'N': ["I worry about things", "I have frequent mood swings", "I get upset easily"],
    'A': ["I am interested in people", "I have a soft heart", "I feel others' emotions"],
    'C': ["I am always prepared", "I pay attention to details", "I like order"],
    'O': ["I have a vivid imagination", "I am creative and original", "I spend time reflecting on things"]
}

responses_ocean = {}
cols_ocean = st.columns(5)

traits = [('E', 'Extraversion'), ('N', 'Neuroticism'), ('A', 'Agreeableness'), ('C', 'Conscientiousness'), ('O', 'Openness')]

for idx, (trait_key, trait_name) in enumerate(traits):
    with cols_ocean[idx]:
        st.markdown(f"**{trait_name}**")
        for i in range(1, 4):
            responses_ocean[f'{trait_key}{i}'] = st.slider(
                ocean_questions[trait_key][i-1],
                min_value=1, max_value=5, value=3, step=1,
                key=f'{trait_key}{i}'
            )
        for i in range(4, 11):
            responses_ocean[f'{trait_key}{i}'] = 3

st.markdown('</div>', unsafe_allow_html=True)

# MBTI Questions Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<div class="input-title">🧠 MBTI Assessment (0-10 Scale)</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    introversion = st.slider("Introversion Score", 0.0, 10.0, 5.0, step=0.1)
    sensing = st.slider("Sensing Score", 0.0, 10.0, 5.0, step=0.1)

with col2:
    thinking = st.slider("Thinking Score", 0.0, 10.0, 5.0, step=0.1)
    judging = st.slider("Judging Score", 0.0, 10.0, 5.0, step=0.1)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #00d4ff; margin: 1.5rem 0;'><small>↓ Click to analyze your responses ↓</small></p>", unsafe_allow_html=True)

if st.button("📊 Get My Personality Results", use_container_width=True):
    
    # OCEAN Prediction
    ocean_results = None
    if ocean_model_data:
        model = ocean_model_data['model']
        scaler = ocean_model_data['scaler']
        feature_cols = ocean_model_data['feature_cols']
        
        gender_code = 1 if gender == "Male" else 2
        
        features = {}
        for col in feature_cols:
            if col in responses_ocean:
                features[col] = responses_ocean[col]
            elif col == 'age':
                features[col] = age
            elif col == 'gender':
                features[col] = gender_code
            elif col == 'engnat':
                features[col] = 1
            elif col == 'hand':
                features[col] = 1
            else:
                features[col] = 3
        
        feature_array = np.array([features[col] for col in feature_cols]).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array)
        
        prediction = model.predict(feature_scaled)[0]
        
        E_cols = [f'E{i}' for i in range(1, 11)]
        N_cols = [f'N{i}' for i in range(1, 11)]
        A_cols = [f'A{i}' for i in range(1, 11)]
        C_cols = [f'C{i}' for i in range(1, 11)]
        O_cols = [f'O{i}' for i in range(1, 11)]
        
        trait_scores = {
            'Openness': np.mean([responses_ocean[col] for col in O_cols if col in responses_ocean]) / 5.0,
            'Conscientiousness': np.mean([responses_ocean[col] for col in C_cols if col in responses_ocean]) / 5.0,
            'Extraversion': np.mean([responses_ocean[col] for col in E_cols if col in responses_ocean]) / 5.0,
            'Agreeableness': np.mean([responses_ocean[col] for col in A_cols if col in responses_ocean]) / 5.0,
            'Neuroticism': np.mean([responses_ocean[col] for col in N_cols if col in responses_ocean]) / 5.0
        }
        
        ocean_results = {
            'prediction': prediction,
            'scores': trait_scores,
            'questions': ocean_questions
        }
    
    # MBTI Prediction
    mbti_results = None
    if mbti_pipeline and mbti_encoder:
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
            pred_encoded = mbti_pipeline.predict(sample)
            pred_label = mbti_encoder.inverse_transform(pred_encoded)[0]
            mbti_results = pred_label
        except Exception as e:
            st.error(f"❌ MBTI Prediction failed: {str(e)}")
    
    # Display Results
    if ocean_results or mbti_results:
        st.markdown("---")
        st.markdown('## 📊 Your Results', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # OCEAN Results
        if ocean_results:
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="result-title">🌊 OCEAN Results</div>', unsafe_allow_html=True)
                
                trait_info = {
                    'Openness': 'Creative, imaginative, reflective',
                    'Conscientiousness': 'Organized, detail-oriented, prepared',
                    'Extraversion': 'Outgoing, conversational, sociable',
                    'Agreeableness': 'Interested in people, empathetic, kind',
                    'Neuroticism': 'Anxious, moody, emotionally reactive'
                }
                
                for trait, score in ocean_results['scores'].items():
                    st.markdown(f"""
                    <div class="ocean-item">
                        <div>
                            <div class="ocean-label">{trait}</div>
                            <div class="ocean-question">{trait_info[trait]}</div>
                        </div>
                        <span class="ocean-value">{score:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="margin-top: 1.5rem; padding: 1rem; background: #0f1419; border-radius: 8px; border-left: 4px solid #ff006e;">
                    <strong>Personality Type: {ocean_results['prediction']}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # MBTI Results
        if mbti_results:
            with col2:
                st.markdown('<div class="result-box result-box-mbti">', unsafe_allow_html=True)
                st.markdown('<div class="result-title result-title-mbti">🧠 MBTI Results</div>', unsafe_allow_html=True)
                
                # MBTI Type Descriptions
                mbti_descriptions = {
                    'INTJ': 'The Architect - Strategic, independent, innovative thinker',
                    'INTP': 'The Logician - Logical, curious, analytical problem-solver',
                    'ENTJ': 'The Commander - Bold, decisive, natural leader',
                    'ENTP': 'The Debater - Quick-witted, creative, enjoys challenges',
                    'INFJ': 'The Advocate - Idealistic, principled, dedicated helper',
                    'INFP': 'The Mediator - Idealistic, reserved, focused on personal growth',
                    'ENFJ': 'The Protagonist - Charismatic, inspiring natural leader',
                    'ENFP': 'The Campaigner - Energetic, enthusiastic, spontaneous explorer',
                    'ISTJ': 'The Logistician - Practical, factual, reliable organizer',
                    'ISFJ': 'The Defender - Protective, caring, loyal supporter',
                    'ESTJ': 'The Executive - Direct, organized, strong leader',
                    'ESFJ': 'The Consul - Caring, social, enthusiastic team player',
                    'ISTP': 'The Virtuoso - Practical, experimental, resourceful tinkerer',
                    'ISFP': 'The Adventurer - Sensitive, artistic, flexible free spirit',
                    'ESTP': 'The Entrepreneur - Bold, energetic, pragmatic doer',
                    'ESFP': 'The Entertainer - Spontaneous, outgoing, natural performer'
                }
                
                description = mbti_descriptions.get(mbti_results, 'Unique personality type')
                
                st.markdown(f"""
                <div class="mbti-result">
                    <div class="mbti-type">{mbti_results}</div>
                    <div style="margin-top: 1rem; font-size: 1rem; color: #00d4ff; font-weight: 600;">
                        {description}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown("""
<div style='text-align: center; color: #00d4ff; font-weight: 600; padding: 1rem;'>
    <strong>OCEAN:</strong> Big Five Personality Traits | <strong>MBTI:</strong> Myers-Briggs Type Indicator
</div>
""", unsafe_allow_html=True)
st.caption("🧠 Personality Predictor · Psychological Assessment Tool")
