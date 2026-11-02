import torch
from torch import nn
from torch.nn import functional as F


def shuffled_pairs(x, z):
    flat_x = x.reshape(-1, x.shape[-1])
    flat_z = z.reshape(-1, z.shape[-1])
    permutation = torch.randperm(len(flat_z), device=z.device)
    return flat_x, flat_z, flat_z[permutation]


class DependencyDiscriminator(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, z):
        return self.net(torch.cat([x, z], dim=-1)).squeeze(-1)


def dependency_loss(head, x, z):
    x, matched, shuffled = shuffled_pairs(x, z)
    positive, negative = head(x, matched), head(x, shuffled)

    return F.binary_cross_entropy_with_logits(
        positive, torch.ones_like(positive)
    ) + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
