import joblib
import pandas as pd


def load_model(
    model_path='models/energy_pipeline.pkl',
    columns_path='models/feature_columns.pkl'
):
    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)
    return model, feature_columns


def get_efficiency_rating(prediction):
    if prediction < 15:
        return 'High Efficiency'
    elif prediction < 30:
        return 'Medium Efficiency'
    else:
        return 'Low Efficiency'


def predict(model, feature_columns, data: dict):
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(
        columns=feature_columns, fill_value=0
    )
    prediction = model.predict(input_df)[0]
    efficiency = get_efficiency_rating(prediction)
    return {
        'predicted_heating_load': round(float(prediction), 4),
        'unit': 'kWh',
        'efficiency_rating': efficiency
    }
