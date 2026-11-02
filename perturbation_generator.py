from torch import nn


class CaRIIntervention(nn.Module):
    def __init__(self, z_dim, target_norm=0.1):
        super().__init__()
        if target_norm <= 0:
            raise ValueError("target_norm b must be positive")
        self.target_norm = float(target_norm)
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim), nn.GELU(), nn.Linear(z_dim, z_dim)
        )

        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.normal_(self.net[-1].bias, std=0.01)

    def perturbation(self, z):
        return self.net(z)

    def forward(self, z):
        return z + self.perturbation(z)

    def regularization(self, z):
        delta = self.perturbation(z)
        return (delta.norm(p=2, dim=-1) - self.target_norm).square().mean()
