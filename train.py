import pandas as pd
import numpy as np
import re
import string
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data.csv")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["emotion"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train_tf, y_train)

print("Training accuracy:", model.score(X_train_tf, y_train))
print("Testing accuracy:", model.score(X_test_tf, y_test))

# Save
joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved!")
