# 6 Pipeline
from HeartBeat.config.configuration import configurationManager
from HeartBeat.components.data_processing import preprocessing
from HeartBeat.logging import logger
import os

class DataProcessingTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        Classes = ["artifact", "murmur", "normal"]
        config = configurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = preprocessing(config=data_preprocessing_config)

        save_path = data_preprocessing_config.local_data_file

        if all(os.path.exists(os.path.join(save_path, f)) for f in ["x_data.npy", "y_data.npy", "test_x.npy", "test_y.npy"]):
            print("Preprocessed data already exists, skipping preprocessing...")

        else:
            print("Preprocessed data not found, running preprocessing...")

            # 1. Set folder paths
            data_preprocessing.load_files(base_path="artifacts/data_injection/Heartbeat_Sound")

            # 2. Create label maps
            label_to_int, int_to_label, Nb_classes = data_preprocessing.create_label_maps(classes=Classes)

            # 3. Load labeled data
            categories = [
                {'folder': data_preprocessing.artifact_data,   'prefix': 'artifact',   'label': 0},
                {'folder': data_preprocessing.murmur_data,     'prefix': 'murmur',     'label': 1},
                {'folder': data_preprocessing.normal_data,     'prefix': 'normal',     'label': 2},
                {'folder': data_preprocessing.extrahls_data,   'prefix': 'extrahls',   'label': 2},
                {'folder': data_preprocessing.extrastole_data, 'prefix': 'extrastole', 'label': 2},
            ]
            all_sounds, all_labels = data_preprocessing.load_all_categories(categories)

            # 4. Load unlabeled data
            unlabeled_sounds, unlabeled_labels = data_preprocessing.load_unlabeled_data(
                folder=data_preprocessing.unlable_data,
                prefixes=["Aunlabelledtest", "Bunlabelledtest"]
            )

            # 5. Combine
            x_data, y_data, test_x, test_y = data_preprocessing.combine_data(
                all_sounds, all_labels, unlabeled_sounds, unlabeled_labels
            )

            # 6. Save
            data_preprocessing.save_preprocessed(x_data, y_data, test_x, test_y)