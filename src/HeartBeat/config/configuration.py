from HeartBeat.constant import *
from HeartBeat.utils.common import read_yaml, create_directories
from HeartBeat.entity import DataingestionConfig, DataValidationConfig, DataPreprocessingConfig, DataTransformerConfig

class configurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,     # Access to constants
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath) # read all config and params yaml files
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataingestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataingestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir = config.root_dir,
            STATUS_FILE = config.STATUS_FILE,
            ALL_REQUIRED_FILES = config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_processing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir = config.root_dir,
            local_data_file = config.local_data_file
        )

        return data_preprocessing_config
    
    def get_data_transformer_config(self) -> DataTransformerConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformer_config = DataTransformerConfig(
            root_dir = config.root_dir,
            local_data_file = config.local_data_file,
            preprocessed_data_path = config.preprocessed_data_path
        )

        return data_transformer_config
    
    