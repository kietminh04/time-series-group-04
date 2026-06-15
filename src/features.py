import pandas as pd
import numpy as np


def handle_outliers(df: pd.DataFrame, cols: list, method: str = "iqr") -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower, upper)
    return df


def create_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    df["hour"] = df[date_col].dt.hour
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def create_fourier_features(df: pd.DataFrame, date_col: str, periods: list = [24, 168]) -> pd.DataFrame:
    df = df.copy()
    t = (df[date_col] - df[date_col].min()).dt.total_seconds() / 3600.0
    for p in periods:
        df[f"sin_{p}"] = np.sin(2 * np.pi * t / p)
        df[f"cos_{p}"] = np.cos(2 * np.pi * t / p)
    return df


def create_lag_features(df: pd.DataFrame, cols: list, lags: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, cols: list, windows: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].shift(1).rolling(window=w).mean()
            df[f"{col}_roll_std_{w}"] = df[col].shift(1).rolling(window=w).std()
    return df


def scale_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cols: list):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    mean = train_df[cols].mean()
    std = train_df[cols].std() + 1e-8
    
    train_df[cols] = (train_df[cols] - mean) / std
    val_df[cols] = (val_df[cols] - mean) / std
    test_df[cols] = (test_df[cols] - mean) / std
    
    return train_df, val_df, test_df, mean, std
