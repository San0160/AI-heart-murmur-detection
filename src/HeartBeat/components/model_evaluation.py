from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from HeartBeat.components.model_training import HeartMurmurLSTM, ModelTrainer
from HeartBeat.config.configuration import ModelEvaluationConfig
import os


# model components
# 1) Load test data   2) create test loader       3) evaluate model

class model_evaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_test_data(self):
        path = self.config.transformed_data_path

        X_test = np.load(path / "X_test.npy")
        y_test = np.load(path / "y_test.npy")

        return X_test, y_test
    
    def create_test_loader(self, X_test, y_test):

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        dataset = TensorDataset(X_test, y_test)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
    
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

        print(f"✓ Model loaded successfully from {self.config.trained_model_path}")

        return model

    def evaluate_model(self, model, test_loader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)

                preds = outputs.argmax(dim=1).cpu().numpy()
                labels = y_batch.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        # Classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.config.classes,
            output_dict=True
        )

        print(classification_report(
            all_labels,
            all_preds,
            target_names=self.config.classes
        ))

        # Save metrics
        metrics_df = pd.DataFrame(report).transpose()

        os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)
        metrics_df.to_csv(self.config.metrics_path, index=True)

        print(f"Metrics saved to {self.config.metrics_path}")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.config.classes,
            yticklabels=self.config.classes
        )

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
        plt.savefig(self.config.confusion_matrix_path)
        plt.close()

        return all_preds, all_labels