from HeartBeat.constant import *
from HeartBeat.utils.common import read_yaml, create_directories
from HeartBeat.entity import DataingestionConfig, DataValidationConfig, DataPreprocessingConfig, DataTransformerConfig, ModelTrainerConfig, ModelEvaluationConfig

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
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            transformed_data_path = Path(config.transformed_data_path),

            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,

            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            patience=config.patience,
        )
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config.model_evaluation

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),

            transformed_data_path=Path(config.transformed_data_path),
            trained_model_path=Path(config.trained_model_path),

            metrics_path=Path(config.metrics_path),
            confusion_matrix_path=Path(config.confusion_matrix_path),

            batch_size=config.batch_size,

            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,

            classes=config.classes
        )
    
    