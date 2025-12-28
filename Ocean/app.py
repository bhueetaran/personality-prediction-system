"""
Streamlit App for OCEAN Personality Prediction
Interactive web application for personality trait prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="OCEAN Personality Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
    }
    .trait-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessors"""
    try:
        with open('personality_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run train_model.py first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def get_ocean_questions():
    """Get OCEAN personality questions (2-3 per trait)"""
    questions = {
        'E': [
            "I am the life of the party",
            "I start conversations",
            "I am outgoing and sociable"
        ],
        'N': [
            "I worry about things",
            "I have frequent mood swings",
            "I get upset easily"
        ],
        'A': [
            "I am interested in people",
            "I have a soft heart",
            "I feel others' emotions"
        ],
        'C': [
            "I am always prepared",
            "I pay attention to details",
            "I like order"
        ],
        'O': [
            "I have a vivid imagination",
            "I am creative and original",
            "I spend time reflecting on things"
        ]
    }
    return questions

def calculate_trait_scores(responses):
    """Calculate OCEAN trait scores from responses"""
    E_cols = [f'E{i}' for i in range(1, 11)]
    N_cols = [f'N{i}' for i in range(1, 11)]
    A_cols = [f'A{i}' for i in range(1, 11)]
    C_cols = [f'C{i}' for i in range(1, 11)]
    O_cols = [f'O{i}' for i in range(1, 11)]
    
    scores = {
        'Extraversion': np.mean([responses[col] for col in E_cols if col in responses]),
        'Neuroticism': np.mean([responses[col] for col in N_cols if col in responses]),
        'Agreeableness': np.mean([responses[col] for col in A_cols if col in responses]),
        'Conscientiousness': np.mean([responses[col] for col in C_cols if col in responses]),
        'Openness': np.mean([responses[col] for col in O_cols if col in responses])
    }
    return scores

def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 OCEAN Personality Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover Your Big Five Personality Traits</p>', unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    label_encoders = model_data['label_encoders']
    
    # Sidebar
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    **OCEAN Model** represents the Big Five personality traits:
    - **O**penness to Experience
    - **C**onscientiousness  
    - **E**xtraversion
    - **A**greeableness
    - **N**euroticism
    
    Answer 50 questions to discover your dominant personality trait!
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model:** {model_data['model_name']}")
    st.sidebar.markdown(f"**Accuracy:** {model_data['accuracy']:.2%}")
    
    # Main Content
    tab1, tab2, tab3 = st.tabs(["📝 Questionnaire", "📊 Results", "ℹ️ About OCEAN"])
    
    with tab1:
        st.markdown("### Please answer the following questions (1-5 scale)")
        st.markdown("*1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree*")
        
        # Demographic Information
        st.markdown("---")
        st.markdown("#### Demographics (Optional)")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=13, max_value=120, value=25, step=1)
            gender = st.selectbox("Gender", ["1", "2"], format_func=lambda x: "Male" if x == "1" else "Female")
        
        with col2:
            engnat = st.selectbox("Native English Speaker", ["1", "2"], format_func=lambda x: "Yes" if x == "1" else "No")
            hand = st.selectbox("Handedness", ["1", "2"], format_func=lambda x: "Right" if x == "1" else "Left")
        
        # Personality Questions
        st.markdown("---")
        st.markdown("#### Personality Assessment Questions (15 questions total)")
        
        responses = {}
        
        # Get questions from question bank
        ocean_questions = get_ocean_questions()
        
        # Create tabs for each trait for better organization
        trait_tabs = st.tabs(["Extraversion (E)", "Neuroticism (N)", "Agreeableness (A)", 
                             "Conscientiousness (C)", "Openness (O)"])
        
        # Extraversion Questions
        with trait_tabs[0]:
            st.markdown("**Extraversion** - Social, energetic, assertive")
            for i in range(1, 4):  # Only 3 questions
                responses[f'E{i}'] = st.slider(
                    f"E{i}: {ocean_questions['E'][i-1]}",
                    min_value=1, max_value=5, value=3, step=1,
                    key=f'E{i}'
                )
            # Fill remaining E4-E10 with neutral default for model compatibility
            for i in range(4, 11):
                responses[f'E{i}'] = 3
        
        # Neuroticism Questions
        with trait_tabs[1]:
            st.markdown("**Neuroticism** - Anxious, moody, emotional")
            for i in range(1, 4):  # Only 3 questions
                responses[f'N{i}'] = st.slider(
                    f"N{i}: {ocean_questions['N'][i-1]}",
                    min_value=1, max_value=5, value=3, step=1,
                    key=f'N{i}'
                )
            # Fill remaining N4-N10 with neutral default for model compatibility
            for i in range(4, 11):
                responses[f'N{i}'] = 3
        
        # Agreeableness Questions
        with trait_tabs[2]:
            st.markdown("**Agreeableness** - Trusting, helpful, kind")
            for i in range(1, 4):  # Only 3 questions
                responses[f'A{i}'] = st.slider(
                    f"A{i}: {ocean_questions['A'][i-1]}",
                    min_value=1, max_value=5, value=3, step=1,
                    key=f'A{i}'
                )
            # Fill remaining A4-A10 with neutral default for model compatibility
            for i in range(4, 11):
                responses[f'A{i}'] = 3
        
        # Conscientiousness Questions
        with trait_tabs[3]:
            st.markdown("**Conscientiousness** - Organized, disciplined, goal-oriented")
            for i in range(1, 4):  # Only 3 questions
                responses[f'C{i}'] = st.slider(
                    f"C{i}: {ocean_questions['C'][i-1]}",
                    min_value=1, max_value=5, value=3, step=1,
                    key=f'C{i}'
                )
            # Fill remaining C4-C10 with neutral default for model compatibility
            for i in range(4, 11):
                responses[f'C{i}'] = 3
        
        # Openness Questions
        with trait_tabs[4]:
            st.markdown("**Openness** - Creative, curious, open-minded")
            for i in range(1, 4):  # Only 3 questions
                responses[f'O{i}'] = st.slider(
                    f"O{i}: {ocean_questions['O'][i-1]}",
                    min_value=1, max_value=5, value=3, step=1,
                    key=f'O{i}'
                )
            # Fill remaining O4-O10 with neutral default for model compatibility
            for i in range(4, 11):
                responses[f'O{i}'] = 3
        
        # Prepare features
        features = {}
        for col in feature_cols:
            if col in responses:
                features[col] = responses[col]
            elif col == 'age':
                features[col] = age
            elif col == 'gender':
                features[col] = int(gender)
            elif col == 'engnat':
                features[col] = int(engnat)
            elif col == 'hand':
                features[col] = int(hand)
            elif col in label_encoders:
                features[col] = 0  # Default value for country if not provided
            else:
                features[col] = 3  # Default neutral value
        
        # Predict button
        st.markdown("---")
        if st.button("🔮 Predict My Personality", type="primary", use_container_width=True):
            # Create feature array
            feature_array = np.array([features[col] for col in feature_cols]).reshape(1, -1)
            
            # Scale features
            feature_scaled = scaler.transform(feature_array)
            
            # Predict
            prediction = model.predict(feature_scaled)[0]
            probabilities = model.predict_proba(feature_scaled)[0]
            
            # Calculate trait scores
            trait_scores = calculate_trait_scores(responses)
            
            # Store in session state
            st.session_state['prediction'] = prediction
            st.session_state['probabilities'] = probabilities
            st.session_state['trait_scores'] = trait_scores
            st.session_state['class_names'] = model.classes_
            
            # Switch to results tab
            st.rerun()
    
    with tab2:
        if 'prediction' not in st.session_state:
            st.info("👈 Please complete the questionnaire in the 'Questionnaire' tab to see your results!")
        else:
            prediction = st.session_state['prediction']
            probabilities = st.session_state['probabilities']
            trait_scores = st.session_state['trait_scores']
            class_names = st.session_state['class_names']
            
            st.markdown("### 🎯 Your Personality Prediction")
            
            # Prediction card
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"## Your Dominant Trait: **{prediction}**")
                
                # Trait descriptions
                descriptions = {
                    'Extraversion': "You are outgoing, social, and energized by interactions with others. You enjoy being around people and thrive in social settings.",
                    'Neuroticism': "You experience emotions intensely and may be more sensitive to stress. You tend to worry more and experience mood swings.",
                    'Agreeableness': "You are cooperative, trusting, and empathetic. You value harmony and tend to be helpful and considerate of others.",
                    'Conscientiousness': "You are organized, disciplined, and goal-oriented. You plan ahead and follow through on your commitments.",
                    'Openness': "You are creative, curious, and open to new experiences. You enjoy art, adventure, and intellectual pursuits."
                }
                
                st.info(descriptions.get(prediction, "Personality trait information"))
            
            with col2:
                # Confidence score
                max_prob_idx = np.argmax(probabilities)
                confidence = probabilities[max_prob_idx]
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Trait Scores Visualization
            st.markdown("---")
            st.markdown("### 📊 Your Trait Scores")
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(trait_scores.values()),
                theta=list(trait_scores.keys()),
                fill='toself',
                name='Your Scores',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[1, 5]
                    )),
                showlegend=True,
                title="OCEAN Personality Profile",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart of probabilities
            st.markdown("### 📈 Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Trait': class_names,
                'Probability': probabilities
            }).sort_values('Probability', ascending=True)
            
            fig_bar = px.bar(
                prob_df,
                x='Probability',
                y='Trait',
                orientation='h',
                color='Probability',
                color_continuous_scale='Blues',
                title='Confidence Level for Each Trait'
            )
            fig_bar.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Detailed trait scores
            st.markdown("### 📋 Detailed Trait Scores")
            scores_df = pd.DataFrame(list(trait_scores.items()), columns=['Trait', 'Score'])
            scores_df = scores_df.sort_values('Score', ascending=False)
            
            for idx, row in scores_df.iterrows():
                score = row['Score']
                trait = row['Trait']
                st.progress(score / 5.0, text=f"{trait}: {score:.2f}/5.0")
    
    with tab3:
        st.markdown("## About the OCEAN/Big Five Personality Model")
        
        st.markdown("""
        The **Big Five personality model** (also known as OCEAN) is one of the most widely accepted 
        frameworks in psychology for understanding personality. It identifies five broad dimensions 
        that describe human personality:
        
        ### 🌊 The Five Traits
        
        1. **Openness to Experience (O)**
           - High: Creative, curious, open-minded, enjoys new experiences
           - Low: Practical, conventional, prefers routine
        
        2. **Conscientiousness (C)**
           - High: Organized, disciplined, reliable, goal-oriented
           - Low: Spontaneous, flexible, less organized
        
        3. **Extraversion (E)**
           - High: Outgoing, sociable, energetic, assertive
           - Low: Reserved, quiet, prefers solitude
        
        4. **Agreeableness (A)**
           - High: Trusting, helpful, empathetic, cooperative
           - Low: Competitive, skeptical, less concerned with others
        
        5. **Neuroticism (N)**
           - High: Anxious, moody, sensitive to stress
           - Low: Calm, emotionally stable, resilient
        
        ### 🎯 How to Use This Tool
        
        1. Answer the 50 questions honestly (1-5 scale)
        2. Provide basic demographic information (optional)
        3. Click "Predict My Personality" to see your results
        4. View your trait scores and dominant personality type
        
        ### 📝 Note
        
        This tool is for educational and entertainment purposes. Professional personality assessments 
        should be conducted by qualified psychologists.
        """)

if __name__ == "__main__":
    main()