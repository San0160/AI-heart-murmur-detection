from HeartBeat.pipeline.stage_01_DI import DataIngestionTrainingPipeline
from HeartBeat.pipeline.stage_02_DV import DataValidationTrainingPipeline
from HeartBeat.pipeline.stage_03_DP import DataProcessingTrainingPipeline
#from Text_summariser.pipeline.stage_03_dt import DataTransformationTrainingPipeline
#from Text_summariser.pipeline.stage_04_mt import ModelTrainerTrainingPipeline
#from Text_summariser.pipeline.stage_05_de import ModelEvaluationTrainingPipeline
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
    data_injestion = DataValidationTrainingPipeline()
    data_injestion.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Processing Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_injestion = DataProcessingTrainingPipeline()
    data_injestion.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e