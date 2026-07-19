import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    df = pd.read_excel(filepath)
    return df


def rename_columns(df):
    df.columns = [
        'Compactness', 'Surface_Area', 'Wall_Area',
        'Roof_Area', 'Height', 'Orientation',
        'Glazing_Area', 'Glazing_Distribution',
        'Heating_Load', 'Cooling_Load'
    ]
    return df


def clean_data(df):
    df = df.drop(['Roof_Area', 'Surface_Area',
                  'Orientation', 'Cooling_Load'], axis=1)
    return df


def split_data(df):
    X = df.drop('Heating_Load', axis=1)
    y = df['Heating_Load']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
