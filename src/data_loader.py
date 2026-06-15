import os
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    missing_values = df.isnull().sum().to_dict()
    df_diff = df["date"].diff().dropna()
    is_uniform = df_diff.nunique() == 1
    most_common_freq = df_diff.mode()[0] if not df_diff.empty else None
    return {
        "missing_values": missing_values,
        "is_uniform": is_uniform,
        "sampling_freq": most_common_freq
    }


def split_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    
    return train_df, val_df, test_df
