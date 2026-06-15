import pandas as pd
import numpy as np


def create_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str, windows: list) -> pd.DataFrame:
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = df[target_col].shift(1).rolling(window=window).mean()
        df[f"{target_col}_roll_std_{window}"] = df[target_col].shift(1).rolling(window=window).std()
    return df


def create_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df


def create_features(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df = create_calendar_features(df, date_col)
    df = create_lag_features(df, target_col, lags=[1, 2, 3, 7])
    df = create_rolling_features(df, target_col, windows=[7, 14, 30])
    return df

