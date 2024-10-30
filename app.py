import streamlit as st
import numpy as np
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

@st.cache(allow_output_mutation=True)
def load_stop_words():
    stop_words = set(stopwords.words('english'))
    return stop_words

stop_words = load_stop_words()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Load pre-trained models and vectorizers
MODEL_PATHS = {
    'Random forests': r'model\random_forests.pkl',
    'Support Vector Classifier': r'model\svc_model.pkl',
    # Add other models as needed
}
VECTORIZER_PATH = r'model\tfidf_vectorizer.pkl'

# Load vectorizer
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = joblib.load(f)

# Define function to load selected model
@st.cache(allow_output_mutation=True)
def load_model(model_name):
    with open(MODEL_PATHS[model_name], 'rb') as f:
        model = joblib.load(f)
    return model

# Streamlit App
st.title("Cyberbullying Text Classification")

# Model Selection
model_name = st.selectbox("Choose the model for inference", list(MODEL_PATHS.keys()))
model = load_model(model_name)

# Text Input
user_text = st.text_area("Enter the text to classify")

# Predict and display result
if st.button("Classify Text"):
    if user_text.strip():
        # Preprocess the text
        user_text = preprocess_text(user_text)

        # Transform the text using the loaded vectorizer
        text_vector = vectorizer.transform([user_text])
        
        # Predict using the selected model
        prediction = model.predict(text_vector)[0]
        
        # Display the result
        st.write(f"Predicted Cyberbullying Type: {prediction}")
    else:
        st.write("Please enter some text for classification.")
