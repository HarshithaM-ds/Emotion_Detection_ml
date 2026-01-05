# Emotion Detection Using Machine Learning

This project is a simple Machine Learning and NLP-based application that predicts the emotion expressed in a given text.  
The model classifies text into emotions such as joy, sadness, anger, fear, disgust, and surprise.

I built this project as part of my Machine Learning coursework to understand how text preprocessing, feature extraction, and classification work together in a real application.

---

##  Problem Statement
Text-based emotion detection is useful in areas like chatbots, sentiment analysis, and mental health monitoring.  
The aim of this project is to build a model that can understand user input text and predict the underlying emotion accurately.

---

##  Approach Used
The project follows a classical NLP and Machine Learning pipeline:

- Cleaning and preprocessing raw text
- Converting text into numerical features using TF-IDF
- Training a Logistic Regression classifier for multi-class emotion prediction
- Evaluating performance using train–test split
- Deploying the trained model using Streamlit

---

##  Algorithms and Techniques
- TF-IDF (Term Frequency–Inverse Document Frequency) for feature extraction
- Logistic Regression for emotion classification
- Cross-Entropy Loss for optimization
- Train–Test Split (80–20) for evaluation

---

##  Dataset
- Used a real-world Kaggle Emotion Dataset
- Contains thousands of natural language sentences
- Emotion labels include:
  - Joy
  - Sadness
  - Anger
  - Fear
  - Disgust
  - Surprise

Using a large and realistic dataset helped improve the model’s generalization compared to a small synthetic dataset.

---

##  Tech Stack
- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Git & GitHub

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshithaM-ds/Emotion_Detection_ml.git
   cd Emotion_Detection_ml
