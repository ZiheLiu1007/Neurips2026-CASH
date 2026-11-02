import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDiscriminatorD1(nn.Module):


    def __init__(self, feature_dim: int, z_dim: int, hidden_dim: int = 64,
                 use_inner_product: bool = False):


        super().__init__()

        self.feature_dim = feature_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.use_inner_product = use_inner_product


        if use_inner_product:
            self.x_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
            self.z_proj = nn.Linear(z_dim, hidden_dim, bias=False)

            self.fc_out = nn.Linear(hidden_dim, 1)
        else:

            self.input_proj = nn.Linear(feature_dim + z_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
            self.activation = nn.GELU()
            self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:


        B, N, P = x.shape
        _, _, dz = z.shape


        if not self.use_inner_product:
            fused = self.activation(self.norm(self.input_proj(torch.cat([x, z], dim=-1))))
            return self.fc_out(fused)
        x_proj = self.x_proj(x)
        z_proj = self.z_proj(z)


        x_proj = F.layer_norm(x_proj, x_proj.shape[-1:])
        z_proj = F.layer_norm(z_proj, z_proj.shape[-1:])

        if self.use_inner_product:


            fused = x_proj * z_proj
            logits = self.fc_out(fused)
        else:

            fused = torch.cat([x_proj, z_proj], dim=-1)
            logits = self.fc_out(fused)

        return logits

    def predict_probability(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:


        logits = self.forward(x, z)
        return torch.sigmoid(logits)


class ReturnDiscriminatorD2(nn.Module):


    def __init__(self, z_dim: int):


        super().__init__()

        self.z_dim = z_dim


        self.linear = nn.Linear(z_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:


        B, N, dz = z.shape


        z_flat = z.reshape(B * N, dz)
        y_pred_flat = self.linear(z_flat)
        y_pred = y_pred_flat.squeeze(-1).reshape(B, N)

        return y_pred


class Discriminator(nn.Module):


    def __init__(self, feature_dim: int, z_dim: int,
                 d1_hidden_dim: int = 64,
                 d1_use_inner_product: bool = False):


        super().__init__()

        self.feature_dim = feature_dim
        self.z_dim = z_dim


        self.d1 = CausalDiscriminatorD1(
            feature_dim=feature_dim,
            z_dim=z_dim,
            hidden_dim=d1_hidden_dim,
            use_inner_product=d1_use_inner_product
        )


        self.d2 = ReturnDiscriminatorD2(z_dim=z_dim)

    def forward(self, x: torch.Tensor, z_original: torch.Tensor,
                z_intervened: torch.Tensor) -> dict:


        d1_logits_original = self.d1(x, z_original)
        d1_logits_intervened = self.d1(x, z_intervened)


        d2_pred = self.d2(z_intervened)

        return {
            'd1_logits_original': d1_logits_original,
            'd1_logits_intervened': d1_logits_intervened,
            'd2_pred': d2_pred
        }


def compute_d1_loss(discriminator: Discriminator,
                    x: torch.Tensor,
                    z: torch.Tensor,
                    z_shuffled: torch.Tensor) -> torch.Tensor:


    d1_logits_original = discriminator.d1(x, z)
    d1_logits_intervened = discriminator.d1(x, z_shuffled)


    target_original = torch.ones_like(d1_logits_original)
    target_intervened = torch.zeros_like(d1_logits_intervened)


    loss_original = F.binary_cross_entropy_with_logits(
        d1_logits_original, target_original, reduction='mean'
    )

    loss_intervened = F.binary_cross_entropy_with_logits(
        d1_logits_intervened, target_intervened, reduction='mean'
    )


    loss = loss_original + loss_intervened

    return loss


def compute_d2_loss(beta_network: nn.Module,
                    vae: nn.Module,
                    x: torch.Tensor,
                    z_intervened: torch.Tensor,
                    y_true: torch.Tensor,
                    discriminator: nn.Module = None) -> torch.Tensor:


    if discriminator is None:
        raise ValueError("compute_d2_loss requires an independent discriminator.d2")


    y_pred = discriminator.d2(z_intervened)


    if not torch.isfinite(y_pred).all() or not torch.isfinite(y_true).all():

        raise FloatingPointError("Nonfinite Dpred inputs")


    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and target shapes must match exactly")
    loss = F.mse_loss(y_pred, y_true, reduction='mean')


    if torch.isnan(loss) or torch.isinf(loss):
        raise FloatingPointError("Nonfinite Dpred objective")

    return loss


def compute_discriminator_loss(discriminator: Discriminator,
                               beta_network: nn.Module,
                               vae: nn.Module,
                               x: torch.Tensor,
                               z: torch.Tensor,
                               z_shuffled: torch.Tensor,
                               z_bar: torch.Tensor,
                               y_true: torch.Tensor,
                               lambda_d1: float = 1.0,
                               lambda_d2: float = 1.0) -> dict:


    if float(lambda_d1) != 0.0:
        loss_d1 = compute_d1_loss(discriminator, x, z, z_shuffled)
    else:
        loss_d1 = torch.zeros((), device=x.device, dtype=x.dtype)


    if float(lambda_d2) != 0.0:
        loss_d2 = compute_d2_loss(beta_network, vae, x, z_bar, y_true,
                                  discriminator=discriminator)
    else:
        loss_d2 = torch.zeros((), device=x.device, dtype=x.dtype)


    loss_total = lambda_d1 * loss_d1 + lambda_d2 * loss_d2

    return {
        'loss_d1': loss_d1,
        'loss_d2': loss_d2,
        'loss_total': loss_total
    }
