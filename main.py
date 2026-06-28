from HeartBeat.pipeline.stage_01_DI import DataIngestionTrainingPipeline
from HeartBeat.pipeline.stage_02_DV import DataValidationTrainingPipeline
from HeartBeat.pipeline.stage_03_DP import DataProcessingTrainingPipeline
from HeartBeat.pipeline.stage_04_DT import DataTransformerTrainingPipeline
from HeartBeat.pipeline.stage_05_MT import ModelTrainerTrainingPipeline
from HeartBeat.pipeline.stage_06_ME import ModelEvaulationTrainingPipeline
from HeartBeat.pipeline.prediction import ModelPredictionTrainingPipeline
from HeartBeat.logging import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_injestion = DataIngestionTrainingPipeline()
    data_injestion.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Processing Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_processing = DataProcessingTrainingPipeline()
    data_processing.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_transformation = DataTransformerTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    model_training = ModelTrainerTrainingPipeline()
    model_training.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    model_evaluation = ModelEvaulationTrainingPipeline()
    model_evaluation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Prediction Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    model_evaluation = ModelPredictionTrainingPipeline()
    model_evaluation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e