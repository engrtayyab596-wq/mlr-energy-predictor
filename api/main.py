from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model
import pandas as pd

app = FastAPI(title='Energy Consumption Predictor')

model = None
feature_columns = None


def get_model():
    global model, feature_columns
    if model is None:
        model, feature_columns = load_model()
    return model, feature_columns


class BuildingData(BaseModel):
    Compactness: float
    Wall_Area: float
    Height: float
    Glazing_Area: float
    Glazing_Distribution: float


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model': 'loaded',
        'description': 'Energy Consumption Predictor API'
    }


@app.post('/predict')
def predict(data: BuildingData):
    model, feature_columns = get_model()
    input_df = pd.DataFrame([data.model_dump()])
    input_df = input_df.reindex(columns=feature_columns)
    prediction = model.predict(input_df)[0]

    if prediction < 15:
        efficiency = 'High Efficiency'
    elif prediction < 30:
        efficiency = 'Medium Efficiency'
    else:
        efficiency = 'Low Efficiency'

    return {
        'predicted_heating_load': round(float(prediction), 4),
        'unit': 'kWh',
        'efficiency_rating': efficiency
    }
