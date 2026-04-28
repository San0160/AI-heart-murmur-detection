import streamlit as st
import numpy as np

from model.model_loader import load_model
from audio.preprocessing import load_audio, extract_mfcc
from ui.visualizations import plot_waveform
from utils.logger import setup_logger

logger = setup_logger("Streamlit APP")

# Load the model 
model = load_model()

st.title("heart murmur detection with LSTM")

uploaded_files = st.file_uploader("Upload a heart sound (wav or mp3)", type = ["wav", "mp3"])

if uploaded_files is not None:
    try:
        y, sr = load_audio(uploaded_files)

        st.subheader("waveform of input sound")
        fig = plot_waveform(y, sr)
        st.pyplot(fig)

        X_input = extract_mfcc(y, sr)

        prediction = model.predict(X_input)
        predicted_class = np.argmax(prediction, axis = 1)[0]

        st.subheader("prediction Result")
        st.write(f"Predicted class **{predicted_class}**")
        st.write("Raw prediction score", prediction)

        logger.info("Prediction completed sucessfully")
    
    except Exception as e:
        logger.exception("Infrence pipeline failed")
        st.error("An error occured while processing the file")