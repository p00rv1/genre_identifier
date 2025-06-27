# streamlit_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import json
import tempfile
from src.feature_ext import extract_mfcc

# Load Keras model
model = tf.keras.models.load_model("genre_classifier.keras")

# Load genre mapping
with open("data.json", "r") as f:
    data = json.load(f)
    mapping = data["mapping"]

st.title("🎵 Music Genre Classifier")
st.write("Upload a `.wav` file (30 seconds max) to predict its genre.")

uploaded_file = st.file_uploader("Choose a `.wav` file", type=["wav"])

if uploaded_file is not None:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format='audio/wav')

    # Extract MFCC
    mfcc = extract_mfcc(tmp_path)

    # Pad or trim MFCC to match model input shape
    expected_length = model.input_shape[1]  # e.g., 130
    if mfcc.shape[0] > expected_length:
        mfcc = mfcc[:expected_length]
    elif mfcc.shape[0] < expected_length:
        pad_width = expected_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

    # Add batch dimension for prediction
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # shape: (1, time, n_mfcc, 1)

    # Predict genre
    prediction = model.predict(mfcc)[0]
    predicted_index = np.argmax(prediction)
    predicted_genre = mapping[predicted_index]

    st.markdown(f"### 🎧 Predicted Genre: **{predicted_genre}**")

    st.write("#### Confidence Scores:")
    for i, prob in sorted(enumerate(prediction), key=lambda x: -x[1])[:3]:
        st.write(f"- {mapping[i]}: {prob*100:.2f}%")
