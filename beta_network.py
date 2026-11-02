from torch import nn


class BetaNetwork(nn.Module):
    def __init__(self, z_dim, factor_dim, hidden_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim or z_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim or z_dim, factor_dim),
        )

    def forward(self, z):
        return self.net(z)


def pricing_returns(beta, factors):
    return (beta * factors.unsqueeze(-2)).sum(dim=-1)
