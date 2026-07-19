import sys
import os
import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.abspath('.'))


@pytest.fixture(autouse=True)
def create_test_model():
    os.makedirs('models', exist_ok=True)

    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 5),
        columns=[
            'Compactness', 'Wall_Area', 'Height',
            'Glazing_Area', 'Glazing_Distribution'
        ]
    )
    y = np.random.randn(100) * 10 + 22

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    pipeline.fit(X, y)

    joblib.dump(pipeline, 'models/energy_pipeline.pkl')
    joblib.dump(list(X.columns), 'models/feature_columns.pkl')

    yield

    if os.path.exists('models/energy_pipeline.pkl'):
        os.remove('models/energy_pipeline.pkl')
    if os.path.exists('models/feature_columns.pkl'):
        os.remove('models/feature_columns.pkl')
