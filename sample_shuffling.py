import torch
import torch.nn as nn


class ShuffleModule(nn.Module):


    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:


        B, N, z_dim = z.shape


        if not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite shuffled representation")


        indices = torch.randperm(B * N, device=z.device)


        z_shuffled = z.reshape(B * N, z_dim)[indices].reshape(B, N, z_dim)

        return z_shuffled
