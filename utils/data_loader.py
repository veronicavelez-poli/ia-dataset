import numpy as np
import pandas as pd

def load_pima_dataset(path="diabetes.csv"):
    df = pd.read_csv(path)

    zero_as_nan_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_nan_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].mean(), inplace=True)

    X = df.drop("Outcome", axis=1).values.astype(float)
    y = df["Outcome"].values.astype(float).reshape(-1, 1)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0

    X_norm = (X - X_mean) / X_std
    return X_norm, y, X_mean, X_std