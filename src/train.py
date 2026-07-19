import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def build_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    return pipeline


def train_model(pipeline, X_train, y_train):
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring='r2'
    )
    print(f"Mean R²: {cv_scores.mean():.4f}")
    print(f"Std R²:  {cv_scores.std():.4f}")
    pipeline.fit(X_train, y_train)
    return pipeline, cv_scores


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    return mae, rmse, r2


def save_model(pipeline, path='models/energy_pipeline.pkl'):
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")
