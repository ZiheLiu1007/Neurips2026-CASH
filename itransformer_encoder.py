import torch
from torch import nn


class ITransformerEncoder(nn.Module):
    def __init__(self, feature_dim, window, z_dim=32, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.feature_dim, self.window = feature_dim, window
        self.token_embedding = nn.Sequential(
            nn.Linear(window, z_dim), nn.GELU(), nn.Linear(z_dim, z_dim)
        )
        layer = nn.TransformerEncoderLayer(
            z_dim, nhead, 4 * z_dim, dropout, batch_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.projection = nn.Linear(z_dim, z_dim)

    def forward(self, x):

        b, t, n, m = x.shape
        if t != self.window or m != self.feature_dim:
            raise ValueError(
                "Input window or feature dimension does not match the encoder"
            )
        tokens = x.permute(0, 2, 3, 1).reshape(b * n, m, t)
        h = self.blocks(self.token_embedding(tokens))
        return self.projection(h.mean(dim=1)).reshape(b, n, -1)


class VanillaTransformerEncoder(nn.Module):

    def __init__(self, feature_dim, window, z_dim=32, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.feature_dim, self.window = feature_dim, window
        self.embedding = nn.Linear(feature_dim, z_dim)
        self.position = nn.Parameter(torch.zeros(1, window, z_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            z_dim, nhead, 4 * z_dim, dropout, batch_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.projection = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        b, t, n, m = x.shape
        if t != self.window or m != self.feature_dim:
            raise ValueError(
                "Input window or feature dimension does not match the encoder"
            )
        tokens = x.permute(0, 2, 1, 3).reshape(b * n, t, m)
        h = self.blocks(self.embedding(tokens) + self.position[:, :t])
        return self.projection(h.mean(dim=1)).reshape(b, n, -1)
