import torch
from torch import nn
from torch.nn import functional as F


class ReturnVAE(nn.Module):
    def __init__(self, n_assets, factor_dim=8, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_assets, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, factor_dim)
        self.log_variance = nn.Linear(hidden_dim, factor_dim)
        self.decoder = nn.Sequential(
            nn.Linear(factor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
        )

    def posterior(self, returns):
        h = self.encoder(returns)
        return self.mean(h), self.log_variance(h).clamp(-12, 12)

    def factors(self, returns):
        return self.posterior(returns)[0]

    def loss(self, returns, kl_weight=1.0):
        mean, logvar = self.posterior(returns)
        latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        reconstruction = F.mse_loss(self.decoder(latent), returns)
        kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
        return reconstruction + kl_weight * kl
