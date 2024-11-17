from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
 
# Load model
MODEL_PATH = r"logistic_regression_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 
# Define the expected number of features
EXPECTED_FEATURES = 40  # Número fijo de características basado en el preprocesamiento
 
# Define input schema for API
class InputData(BaseModel):
    features: List[float]
 
# Initialize FastAPI
app = FastAPI()
 
@app.post("/predict")
def predict(data: InputData):
    # Validate input length
    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain exactly {EXPECTED_FEATURES} features. Got {len(data.features)}."
        )
 
    # Perform prediction
    prediction = model.predict([data.features])[0]
    return {"prediction": int(prediction)}
 
@app.get("/")
def read_root():
    return {"message": "Model Fail Detection API is running"}

