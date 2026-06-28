from HeartBeat.config.configuration import configurationManager
from HeartBeat.components.data_processing import preprocessing
from HeartBeat.components.model_prediction import PredictionPipeline

class ModelPredictionTrainingPipeline:
    def __init__ (self):
        pass    

    def main(self):
        config = configurationManager()

        prediction_config = config.get_prediction_config()
        preprocessing_config = config.get_data_preprocessing_config()

        preprocessor = preprocessing(preprocessing_config)
        predictor = PredictionPipeline(prediction_config)

        model = predictor.load_model()

        audio_path = r"C:\Users\Sandeep\Desktop\Projects\AI-heart-murmur-detection\test_files\murmur__171_1307971016233_D.wav"

        features = preprocessor.preprocess_audio(audio_path)

        loader = predictor.create_prediction_loader(features)

        prediction = predictor.predict(model, loader)

        print(prediction)