from HeartBeat.config.configuration import configurationManager
from HeartBeat.logging import logger
import os
import numpy as np
import torch
from HeartBeat.components.model_evaluation import model_evaluation

class ModelEvaulationTrainingPipeline:
    def __init__ (self):
        pass    

    def main(self):

        config = configurationManager()
        evaluation_config = config.get_model_evaluation_config()

        # Initialize evaluation component
        evaluation = model_evaluation(evaluation_config)

        # Load model
        model = evaluation.load_model()

        # Load test data
        X_test, y_test = evaluation.load_test_data()

        # Create DataLoader
        test_loader = evaluation.create_test_loader(
            X_test,
            y_test
        )

        # Evaluate
        evaluation.evaluate_model(
            model=model,
            test_loader=test_loader
        )

        print("✓ Model evaluation completed successfully.")