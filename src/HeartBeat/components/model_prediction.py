from HeartBeat.components.model_training import HeartMurmurLSTM, DataLoader
from HeartBeat.config.configuration import PredictionConfig
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class PredictionPipeline():

    def __init__(self, config: PredictionConfig):

        self.config = config

    def load_model(self):
        """
        Load the trained Heart Murmur LSTM model.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = HeartMurmurLSTM(self.config)

        model.load_state_dict(
            torch.load(
                self.config.trained_model_path,
                map_location=device
            )
        )

        model.to(device)
        model.eval()

        return model
    
    def create_prediction_loader(self, features):
        """
        Convert one MFCC feature vector into a DataLoader.
        """

        X = np.expand_dims(features, axis=0)

        X = torch.tensor(X, dtype=torch.float32)

        dataset = TensorDataset(X)

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )
  
    def predict(self, model, prediction_loader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        classes = self.config.classes

        with torch.no_grad():

            for X in prediction_loader:

                X = X[0].to(device)

                outputs = model(X)

                pred = outputs.argmax(dim=1).item()

        return classes[pred]