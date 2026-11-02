import torch
import torch.nn as nn


class CaRIIntervention(nn.Module):


    def __init__(self, noise_std: float = 0.1, z_dim: int = 32):


        super().__init__()
        self.noise_std = noise_std
        if not 0 < noise_std < float("inf"):
            raise ValueError("Perturbation target norm b must be finite and positive")
        self.target_norm = float(noise_std)
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim), nn.GELU(), nn.Linear(z_dim, z_dim)
        )

    def perturbation(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def regularization(self, z: torch.Tensor, target_norm=None) -> torch.Tensor:
        b = self.target_norm if target_norm is None else target_norm
        if not 0 < float(b) < float("inf"):
            raise ValueError("Perturbation target norm b must be finite and positive")
        return (self.perturbation(z).norm(p=2, dim=-1) - b).square().mean()

    def forward(self, z: torch.Tensor) -> torch.Tensor:


        B, N, z_dim = z.shape


        noise = self.perturbation(z)


        z_bar = z + noise

        return z_bar
