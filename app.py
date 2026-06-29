from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uvicorn
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

    prediction, confidence = prediction_pipeline.main(file_path)   

    return {
        "prediction": prediction,
        "status": "SUCESSS",
        "filename": file.filename,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )