import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def plot_predictions(y_true: pd.Series, y_pred: pd.Series, title: str = "Model Predictions vs Actuals") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true.index, y_true.values, label='Actual', color='blue', alpha=0.7)
    ax.plot(y_true.index, y_pred.values, label='Predicted', color='orange', linestyle='--', alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    return fig

