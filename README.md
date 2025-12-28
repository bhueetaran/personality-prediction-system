# 🧠 Personality Prediction System using OCEAN & MBTI

This repository presents a **machine learning–based personality prediction system**
that integrates two widely used psychological frameworks:

- 🌊 **OCEAN (Big Five Personality Traits)**
- 🧠 **MBTI (Myers–Briggs Type Indicator)**

The project was developed as part of **academic coursework and independent research**
and serves as the **implementation and experimental foundation** for a corresponding
research paper on personality prediction using machine learning techniques.

---

## 📄 Research Background

This implementation is directly associated with a research study conducted by the author,
focusing on **machine learning–based personality prediction using OCEAN and MBTI models**.

The repository represents the **practical and experimental component** of the research,
demonstrating:
- Dataset preprocessing and encoding
- Model training and evaluation
- Model persistence and inference

The accompanying research paper builds upon the methodologies and results derived
from this system.

---

## 🧠 Models & Methodology

### 🌊 OCEAN 
- Traits predicted:
  - Openness
  - Conscientiousness
  - Extraversion
  - Agreeableness
  - Neuroticism
- Questionnaire-based feature representation
- Feature scaling and supervised learning
- Trained model stored using serialized objects (`.pkl`)

### 🧠 MBTI 
- Predicts one of **16 MBTI personality types**
- Uses numerical trait indicators and demographic features
- Supervised classification approach
- Model and label encoder stored using `joblib`

---

## 📊 Datasets
 
All datasets used are publicly available on **Kaggle** and can be accessed via the links below.

### 🌊 OCEAN
- Description: Questionnaire responses mapped to the five-factor personality model
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/lucasgreenwell/ocean-five-factor-personality-test-responses

### 🧠 MBTI 
- Description: Structured dataset for predicting MBTI personality categories
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/stealthtechnologies/predict-people-personality-types

---

## 🗂 Structure
```text
├─ integrated app.py
├─ README.md
├─ requirements.txt
├─ MBTI/
│  ├─ best_model.joblib
│  ├─ label_encoder.joblib
│  └─ main.ipynb
└─ Ocean/
   ├─ personality_model.pkl
   └─ main.ipynb


