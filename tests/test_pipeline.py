from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

sample_building = {
    "Compactness": 0.98,
    "Wall_Area": 294.0,
    "Height": 7.0,
    "Glazing_Area": 0.0,
    "Glazing_Distribution": 0.0
}


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    assert response.json()['model'] == 'loaded'


def test_predict_endpoint_returns_correct_fields():
    response = client.post('/predict', json=sample_building)
    assert response.status_code == 200
    assert 'predicted_heating_load' in response.json()
    assert 'unit' in response.json()
    assert 'efficiency_rating' in response.json()


def test_prediction_is_positive_number():
    response = client.post('/predict', json=sample_building)
    prediction = response.json().get('predicted_heating_load')
    assert prediction > 0


def test_unit_is_kwh():
    response = client.post('/predict', json=sample_building)
    assert response.json().get('unit') == 'kWh'


def test_efficiency_rating_is_valid():
    response = client.post('/predict', json=sample_building)
    rating = response.json().get('efficiency_rating')
    assert rating in [
        'High Efficiency',
        'Medium Efficiency',
        'Low Efficiency'
    ]
