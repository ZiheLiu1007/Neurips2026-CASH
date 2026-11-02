from torch import nn


class RobustPredictionHead(nn.Module):
    def __init__(self, z_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)
