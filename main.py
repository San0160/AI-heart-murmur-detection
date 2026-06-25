from HeartBeat.pipeline.stage_01_DI import DataIngestionTrainingPipeline

#from Text_summariser.pipeline.stage_02_dv import DatavalidationTrainingPipeline
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