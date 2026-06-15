import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_advanced(X_train: pd.DataFrame, y_train: pd.Series):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

