# src/model_lstm.py
"""
Small PyTorch LSTM model for sequence regression.
Input: (batch, seq_len, n_features) -> output scalar per sequence.
"""
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, (hn, cn) = self.lstm(x)  # out: (B, T, H)
        # take last time-step
        last = out[:, -1, :]  # (B, H)
        y = self.head(last).squeeze(1)  # (B,)
        return y
