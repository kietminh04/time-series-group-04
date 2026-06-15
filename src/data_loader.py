import os
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    return df

