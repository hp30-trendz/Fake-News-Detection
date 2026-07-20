"""
Streamlit application for Fake News Detection.
"""

from __future__ import annotations

import streamlit as st

from src.models.predictor import FakeNewsPredictor


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Fake News Detection")
st.markdown(
    """
Detect whether a news article is **FAKE** or **REAL**
using a trained Machine Learning model.
"""
)

st.sidebar.header("Project Information")

st.sidebar.info(
    """
**Algorithms Trained**

- Logistic Regression
- Naive Bayes
- Random Forest

Best Model:
**Random Forest**

Features:
- Text Preprocessing
- TF-IDF Vectorization
- Machine Learning Classification
"""
)

st.subheader("Enter News Article")

news_text = st.text_area(
    "Paste the news content below:",
    height=300,
)

if st.button("Predict"):

    if not news_text.strip():

        st.warning("Please enter some news text.")

    else:

        with st.spinner("Analyzing article..."):

            predictor = FakeNewsPredictor(
                "random_forest"
            )

            label, confidence = predictor.predict(
                news_text
            )

        st.divider()

        st.subheader("Prediction")

        if label == "REAL":

            st.success(f"✅ REAL NEWS")

        else:

            st.error(f"🚨 FAKE NEWS")

        st.metric(
            "Confidence",
            f"{confidence:.2%}",
        )

st.divider()

st.markdown(
    """
### Dataset

- Fake News: **22,848**
- Real News: **21,210**

### Text Processing

- Lowercasing
- HTML Removal
- URL Removal
- Punctuation Removal
- Stopword Removal
- Lemmatization

### Feature Extraction

- TF-IDF (5000 Features)

### Developed Using

- Python
- Scikit-learn
- Streamlit
- Pandas
- NLTK
"""
)