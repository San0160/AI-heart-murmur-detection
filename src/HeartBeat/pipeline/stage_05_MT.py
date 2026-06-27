from HeartBeat.config.configuration import configurationManager
from HeartBeat.components.model_training import HeartMurmurLSTM, ModelTrainer
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
            print("Starting model training...")

            # Initialize trainer
            trainer = ModelTrainer(trainer_config)

            # Load transformed data
            X_train, X_val, X_test, y_train, y_val, y_test = (
                trainer.load_transformed_data()
            )

            # Compute class weights
            class_weights = trainer.compute_class_weights(y_train)

            # Create DataLoaders
            train_loader, val_loader, test_loader = trainer.create_dataloaders(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test
            )

            # Build model
            model = HeartMurmurLSTM(trainer_config)

            # Train model
            model, history = trainer.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights
            )

            # Save best model
            torch.save(
                model.state_dict(),
                trainer_config.trained_model_path
            )

            print(f"✓ Model saved to {trainer_config.trained_model_path}")