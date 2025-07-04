from fastapi import FastAPI
from app.schema import IrisFeatures, PredictionResult
from app.model import predict_species

app = FastAPI()

@app.post("/predict", response_model=PredictionResult)
def predict(iris: IrisFeatures):
    result = predict_species(iris)
    return {"prediction": result}

