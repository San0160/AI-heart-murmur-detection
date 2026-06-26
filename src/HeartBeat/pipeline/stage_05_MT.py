from HeartBeat.config.configuration import configurationManager
from HeartBeat.components.model_training import create_dataloaders, HeartMurmurLSTM, train_model
from HeartBeat.logging import logger
import os
import numpy as np
import torch

class ModelTrainerTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        
        config = configurationManager()
        trainer_config = config.get_model_trainer_config()

        # Skip training if model already exists
        if trainer_config.trained_model_path.exists():
            print("✓ Trained model already exists.")
            print("Skipping model training.")
        else:
            print("No trained model found.")
            print("Starting training...")

            # Load transformed data
            data_path = trainer_config.transformed_data_path

            X_train = np.load(data_path / "X_train.npy")
            X_val   = np.load(data_path / "X_val.npy")
            X_test  = np.load(data_path / "X_test.npy")

            y_train = np.load(data_path / "y_train.npy")
            y_val   = np.load(data_path / "y_val.npy")
            y_test  = np.load(data_path / "y_test.npy")

            print("✓ Transformed data loaded successfully.")

            # Create DataLoaders
            train_loader, val_loader, test_loader = create_dataloaders(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                batch_size=trainer_config.batch_size
            )

            # Build model
            model = HeartMurmurLSTM(trainer_config)

            # Train model
            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=trainer_config.epochs,
                lr=trainer_config.learning_rate,
                patience=trainer_config.patience
            )

            # Save trained model
            torch.save(model.state_dict(), trainer_config.trained_model_path)

            print(f"✓ Model saved to {trainer_config.trained_model_path}")