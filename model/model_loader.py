import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import streamlit as st
from huggingface_hub import hf_hub_download
from config import HF_REPO_ID, HF_MODEL_FILENAME
from utils.logger import setup_logger

logger = setup_logger("ModelLoader")

@st.cache_resource
def load_model():
    try:
        logger.info("Downloading model from hF hub")
        model_path = hf_hub_download(
            repo_id = HF_REPO_ID,
            filename = HF_MODEL_FILENAME,
            repo_type = "model"
        )

        logger.info("loading tenssorflow model")
        model = tf.keras.models.load_model(model_path, compile = False)
        model.compile(
            optimizer = "adam",
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
        )

        logger.info("Model loaded Sucessfully")
        return model
    
    except Exception as e:
        logger.exception("Failed to load model")
        st.error("Failed to load ML model try again")
        raise e