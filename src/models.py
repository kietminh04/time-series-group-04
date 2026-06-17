import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NaiveModel:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["OT_lag_1"].values


class MovingAverageModel:
    def __init__(self, window: int = 24):
        self.window = window

    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[f"OT_roll_mean_{self.window}"].values


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.fc(self.encoder(self.proj(x))[:, -1, :]).squeeze(-1)


class PyTorchSeqRegressor:
    def __init__(self, model_type: str, input_dim: int, hidden_dim: int = 128, d_model: int = 64, nhead: int = 4, num_layers: int = 2, lr: float = 0.001, epochs: int = 50, patience: int = 10, batch_size: int = 64, device: str = "cpu", checkpoint_dir: str = "../checkpoints"):
        self.model_type = model_type.lower()
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        
        if self.model_type == "lstm":
            self.model = LSTMModel(input_dim, hidden_size=hidden_dim, num_layers=num_layers).to(self.device)
        elif self.model_type == "gru":
            self.model = GRUModel(input_dim, hidden_size=hidden_dim, num_layers=num_layers).to(self.device)
        elif self.model_type == "transformer":
            self.model = TransformerModel(input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_type}_best.pt")
        
        # Prepare training data loader
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Prepare validation data loader if val data is provided
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            val_size = len(val_dataset)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        train_losses = []
        val_losses = []

        print(f"Huấn luyện {self.model_type.upper()} (max {self.epochs} epoch, patience={self.patience})")
        for epoch in range(1, self.epochs + 1):
            # Training phase
            self.model.train()
            total_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(batch_X)
            epoch_train_loss = total_train_loss / len(train_dataset)
            train_losses.append(epoch_train_loss)
            
            # Validation phase
            epoch_val_loss = None
            if val_loader is not None:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        pred = self.model(batch_X)
                        loss = criterion(pred, batch_y)
                        total_val_loss += loss.item() * len(batch_X)
                epoch_val_loss = total_val_loss / val_size
                val_losses.append(epoch_val_loss)
            
            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                val_str = f" | val={epoch_val_loss:.6f}" if epoch_val_loss is not None else ""
                print(f"  Epoch {epoch:3d}/{self.epochs:3d}: train={epoch_train_loss:.6f}{val_str}")
            
            # Early stopping check
            if epoch_val_loss is not None:
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    torch.save(best_state, checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"  Early stopping tại epoch {epoch} (best val={best_val_loss:.6f})")
                        break
        
        # Load best state back into model
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        elif val_loader is not None:
            # If we trained with validation but somehow best_state is None
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        return train_losses, val_losses

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        self.model.eval()
        preds_list = []
        # Predict in chunks to prevent VRAM overflow
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_X, in loader:
                batch_X = batch_X.to(self.device)
                pred = self.model(batch_X)
                preds_list.append(pred.cpu().numpy())
        return np.concatenate(preds_list)


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
