# src/train_lstm.py
"""
Minimal training loop to run a smoke-test LSTM locally (CPU-friendly).
Saves model to models/lstm_baseline.pth
Run as: python -m src.train_lstm
"""
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from src.model_lstm import SimpleLSTM
PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_npz(name):
    d = np.load(os.path.join(PROCESSED, name))
    return d["X"], d["y"]

def make_dataloaders(batch_size=64):
    X_train, y_train = load_npz("windows_train.npz")
    X_val, y_val = load_npz("windows_val.npz")
    # convert to tensors
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val)
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size)

def train(epochs=3, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # load a sample to get dims
    X_train, y_train = np.load(os.path.join(PROCESSED, "windows_train.npz"))["X"], np.load(os.path.join(PROCESSED, "windows_train.npz"))["y"]
    n_features = X_train.shape[2]
    model = SimpleLSTM(n_features=n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_dl, val_dl = make_dataloaders(batch_size=64)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_train = total_loss / (len(train_dl.dataset) + 1e-9)
        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item() * xb.size(0)
        avg_val = val_loss / (len(val_dl.dataset) + 1e-9)
        print(f"Epoch {epoch} train_loss={avg_train:.6e} val_loss={avg_val:.6e}")
    # save
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_baseline.pth"))
    print("Saved LSTM model to models/lstm_baseline.pth")

if __name__ == "__main__":
    # small safety checks: ensure processed files exist
    reqs = ["windows_train.npz", "windows_val.npz", "windows_test.npz"]
    for r in reqs:
        p = os.path.join(PROCESSED, r)
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p}. Run src.dataset_windowed.build_and_save_all() first.")
    train(epochs=3)
