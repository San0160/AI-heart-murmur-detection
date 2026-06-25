from HeartBeat.config.configuration import configurationManager
from HeartBeat.components.data_transformer import DataTransformer
from HeartBeat.logging import logger
import os

class DataTransformerTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):

        Classes = [0, 1, 2]

        config = configurationManager()
        data_transformer_config = config.get_data_transformer_config()
        data_transformer = DataTransformer(config=data_transformer_config)

        save_path = data_transformer_config.local_data_file

        if all(os.path.exists(os.path.join(save_path, f)) for f in [
            "X_train.npy", "X_val.npy", "X_test.npy",
            "y_train.npy", "y_val.npy", "y_test.npy", "test_y.npy"
        ]):
            print("Transformed data already exists, skipping transformation...")

        else:
            print("Transformed data not found, running transformation...")

            # 1. Load preprocessed data
            x_data, y_data, test_x, test_y = data_transformer.load_preprocessed()

            # 2. Split and encode
            X_train, X_val, X_test, y_train, y_val, y_test, test_y = data_transformer.split_and_encode(
                x_data=x_data,
                y_data=y_data,
                test_y=test_y,
                classes=Classes
            )

            # 3. Save transformed data
            data_transformer.save_transformed(X_train, X_val, X_test, y_train, y_val, y_test, test_y)