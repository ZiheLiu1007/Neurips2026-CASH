import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaNetwork(nn.Module):


    def __init__(self, z_dim: int, factor_dim: int, hidden_dim: int = None):


        super().__init__()

        self.z_dim = z_dim
        self.factor_dim = factor_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else z_dim


        self.net = nn.Sequential(
            nn.Linear(z_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, factor_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:


        B, N, z_dim = z.shape


        z_flat = z.reshape(B * N, z_dim)


        beta_flat = self.net(z_flat)
        beta_z = beta_flat.reshape(B, N, self.factor_dim)


        return beta_z


def compute_factor_pricing_returns(beta_z: torch.Tensor, F_t: torch.Tensor) -> torch.Tensor:


    if beta_z.ndim != 3 or F_t.ndim != 2 or (beta_z.shape[0], beta_z.shape[2]) != F_t.shape:
        raise ValueError("Expected loadings (B, N, K) and factors (B, K)")
    if not torch.isfinite(beta_z).all() or not torch.isfinite(F_t).all():

        raise FloatingPointError("Nonfinite factor pricing inputs")


    F_t_expanded = F_t.unsqueeze(1)


    y_pred = torch.sum(beta_z * F_t_expanded, dim=-1)


    if not torch.isfinite(y_pred).all():
        raise FloatingPointError("Nonfinite factor pricing predictions")

    return y_pred


def mse_loss(predicted: torch.Tensor, target: torch.Tensor,
             reduction: str = "mean") -> torch.Tensor:


    if predicted.shape != target.shape:
        raise ValueError("Prediction and target shapes must match exactly")
    mse = F.mse_loss(predicted, target, reduction=reduction)

    return mse


def compute_prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:


    return mse_loss(predicted, target, reduction="mean")
