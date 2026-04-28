import os

# Tensorflow optimization flag
os.environ["TF_ENABLE_ONE_DNN_OPTS"] = "0"

# Audio config
SAMPLE_RATE = 22050
N_MFCC = 52

# Hugging face model config
HF_REPO_ID = "San0160/heart_murmur"
HF_MODEL_FILENAME = "lstm_model.keras"