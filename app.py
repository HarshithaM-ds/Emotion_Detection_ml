import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text

st.title("Emotion Detection App (ML - TFIDF + Logistic Regression)")

user_input = st.text_area("Enter a sentence:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text!")
    else:
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        st.success(f"Predicted Emotion: {pred.upper()}")
