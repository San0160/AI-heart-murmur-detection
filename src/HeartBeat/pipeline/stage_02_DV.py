from HeartBeat.config.configuration import configurationManager
from HeartBeat.components.data_validation import DataValidation
from HeartBeat.logging import logger

class DataValidationTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config = data_validation_config)
        data_validation.validate_all_files_exists()