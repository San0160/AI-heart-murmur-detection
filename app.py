from fastapi import FastAPI, UploadFile, File
import shutil
import os
from HeartBeat.pipeline.prediction import ModelPredictionTrainingPipeline

app = FastAPI(
    title = "Heart murmur detection",
    version = "1.0.0"
)

@app.get("/")
def home():
    return{
        "message": "Heart Murmur api is running"
    }


@app.post("/predict")
async def predict(file : UploadFile = File(...)):
    
    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction_pipeline = ModelPredictionTrainingPipeline()

    prediction = prediction_pipeline.main(file_path)   

    return {
        "prediction": prediction
    }