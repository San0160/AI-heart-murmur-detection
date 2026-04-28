import librosa
import numpy as np
from config import SAMPLE_RATE, N_MFCC
from utils.logger import setup_logger

logger = setup_logger("AudioProcessing")

def load_audio(uploaded_file):
    try:
        logger.info("Loading Audio file")
        y, sr = librosa.load(uploaded_file, sr = SAMPLE_RATE)
        return y, sr
    except Exception as e:
        logger.exception("Audio Loading Failed")
        raise RuntimeError("Invalid or corupted audio files") from e

def extract_mfcc(y, sr):
    try:
        logger.info("Extravting MFCC")
        mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = N_MFCC)
        mfcc_scaled = np.mean(mfcc.T, axis = 0)
        X_input = np.expand_dims(mfcc_scaled, axis = 0)
        X_input = np.expand_dims(X_input, axis = 2)

        return X_input
    
    except Exception as e:
        logger.exception("Falied to extract MFCC")
        raise RuntimeError("Feature Extraction failed") from e