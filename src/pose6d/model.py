import torch
import torch.nn as nn


class SymmetryFieldMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (N, in_dim) -> (N,)
        return self.net(features).squeeze(-1)
