import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalLayer(nn.Module):


    def __init__(self, lambda_grl: float = 1.0):


        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        return grad_reverse(x, self.lambda_grl)

    @staticmethod
    def backward(ctx, grad_output):


        return -ctx.lambda_grl * grad_output, None


class GradientReversalFunction(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:


    return x if lambda_grl == 0 else GradientReversalFunction.apply(x, lambda_grl)


class LightweightPredictionHead(nn.Module):


    def __init__(self, z_dim: int, hidden_dim: int = None, num_layers: int = 2):


        super().__init__()

        if num_layers > 2:
            raise ValueError("预测头最多2层")

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else z_dim
        self.num_layers = num_layers

        if num_layers == 1:

            self.net = nn.Linear(z_dim, 1)
        else:

            self.net = nn.Sequential(
                nn.Linear(z_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1)
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:


        B, N, z_dim = z.shape


        z_flat = z.reshape(B * N, z_dim)


        y_pred_flat = self.net(z_flat)
        y_pred = y_pred_flat.squeeze(-1).reshape(B, N)

        return y_pred


class Generator(nn.Module):


    def __init__(self, encoder: nn.Module, cari_module: nn.Module,
                 z_dim: int):


        super().__init__()


        self.encoder = encoder
        self.cari_module = cari_module

        self.z_dim = z_dim

    def forward(self, x: torch.Tensor) -> dict:


        z = self.encoder(x)


        z_intervened = self.cari_module(z)

        return {
            "z": z,
            "z_intervened": z_intervened
        }


def compute_g1_loss(generator: Generator, discriminator: nn.Module,
                    shuffle_module: nn.Module, x: torch.Tensor,
                    lambda_grl_d1: float = 1.0) -> torch.Tensor:


    outputs = generator(x)
    z = outputs["z"]


    z_shuffled = shuffle_module(z)


    if x.dim() == 4:
        x_current = x[:, -1, :, :]
    else:
        x_current = x


    z_shuffled_rev = grad_reverse(z_shuffled, lambda_grl_d1)
    z_rev = grad_reverse(z, lambda_grl_d1)


    d1_logits_z = discriminator.d1(x_current, z_rev)
    d1_logits_z_shuffled = discriminator.d1(x_current, z_shuffled_rev)

    target_real = torch.ones_like(d1_logits_z)
    target_fake = torch.zeros_like(d1_logits_z_shuffled)


    loss_z = F.binary_cross_entropy_with_logits(d1_logits_z, target_real, reduction='mean')
    loss_z_shuffled = F.binary_cross_entropy_with_logits(d1_logits_z_shuffled, target_fake, reduction='mean')
    loss = loss_z + loss_z_shuffled

    return loss


def compute_g2_loss(generator: Generator, beta_network: nn.Module,
                    vae: nn.Module, x: torch.Tensor, y_true: torch.Tensor,
                    lambda_grl_d2: float = 1.0,
                    discriminator: nn.Module = None) -> torch.Tensor:


    if discriminator is None:
        raise ValueError("compute_g2_loss requires an independent discriminator.d2")


    outputs = generator(x)
    z_intervened = outputs["z_intervened"]


    if x.dim() == 4:
        x_current = x[:, -1, :, :]
    else:
        x_current = x


    z_intervened_rev = grad_reverse(z_intervened, lambda_grl_d2)


    y_pred = discriminator.d2(z_intervened_rev)


    if not torch.isfinite(y_pred).all() or not torch.isfinite(y_true).all():

        raise FloatingPointError("Nonfinite Dpred inputs")


    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and target shapes must match exactly")
    loss = F.mse_loss(y_pred, y_true, reduction='mean')


    if torch.isnan(loss) or torch.isinf(loss):
        raise FloatingPointError("Nonfinite Dpred objective")

    return loss


def compute_g3_loss(generator: Generator, beta_network: nn.Module,
                    vae: nn.Module, x: torch.Tensor,
                    y_true: torch.Tensor,
                    mse_weight: float = 1.0,
                    ic_weight: float = 0.0) -> torch.Tensor:


    from beta_network import compute_factor_pricing_returns


    outputs = generator(x)
    z = outputs["z"]


    if x.dim() == 4:
        x_current = x[:, -1, :, :]
    else:
        x_current = x


    beta_z = beta_network(z)


    F_t = vae.get_factor_returns(y_true, use_mean=True)


    y_pred = compute_factor_pricing_returns(beta_z, F_t)


    if not torch.isfinite(y_pred).all() or not torch.isfinite(y_true).all():

        raise FloatingPointError("Nonfinite pricing inputs")


    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and target shapes must match exactly")
    loss_mse = F.mse_loss(y_pred, y_true, reduction='mean')
    loss = mse_weight * loss_mse

    if ic_weight > 0:

        pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)
        target_centered = y_true - y_true.mean(dim=1, keepdim=True)
        pred_std = pred_centered.std(dim=1, keepdim=True, unbiased=False)
        target_std = target_centered.std(dim=1, keepdim=True, unbiased=False)

        safe_mask = (pred_std > 1e-8) & (target_std > 1e-8)
        if safe_mask.any():
            cov = (pred_centered * target_centered).mean(dim=1, keepdim=True)
            corr = cov / (pred_std * target_std + 1e-8)
            corr = torch.where(safe_mask, corr, torch.zeros_like(corr))
            ic_loss = 1.0 - corr.squeeze(-1).mean()
        else:
            ic_loss = torch.tensor(1.0, device=y_pred.device, dtype=y_pred.dtype)
        loss = loss + ic_weight * ic_loss


    if torch.isnan(loss) or torch.isinf(loss):
        raise FloatingPointError("Nonfinite pricing objective")

    return loss


def compute_g4_loss(generator: Generator, x: torch.Tensor,
                    perturb_target_norm=None) -> torch.Tensor:


    outputs = generator(x)
    z = outputs["z"]
    z_intervened = outputs["z_intervened"]


    b = (generator.cari_module.target_norm if perturb_target_norm is None
         else perturb_target_norm)
    if not 0 < float(b) < float("inf"):
        raise ValueError("Perturbation target norm b must be finite and positive")
    loss = ((z_intervened - z).norm(p=2, dim=-1) - b).square().mean()

    return loss


def compute_generator_loss(generator: Generator, discriminator: nn.Module,
                           beta_network: nn.Module, vae: nn.Module,
                           shuffle_module: nn.Module, x: torch.Tensor,
                           y_true: torch.Tensor,
                           lambda_g1: float = 1.0,
                           lambda_g2: float = 1.0,
                           lambda_g3: float = 1.0,
                           lambda_g4: float = 1.0,
                           lambda_grl_d1: float = 1.0,
                           lambda_grl_d2: float = 1.0,
                           g3_mse_weight: float = 1.0,
                           g3_ic_weight: float = 0.0,
                           perturb_target_norm=None) -> dict:


    if float(lambda_g1) != 0.0:
        loss_g1 = compute_g1_loss(generator, discriminator, shuffle_module, x, lambda_grl_d1)
    else:
        loss_g1 = torch.zeros((), device=x.device, dtype=x.dtype)

    if float(lambda_g2) != 0.0:
        loss_g2 = compute_g2_loss(generator, beta_network, vae, x, y_true, lambda_grl_d2,
                                 discriminator=discriminator)
    else:
        loss_g2 = torch.zeros((), device=x.device, dtype=x.dtype)

    if float(lambda_g3) != 0.0:
        loss_g3 = compute_g3_loss(
            generator,
            beta_network,
            vae,
            x,
            y_true,
            mse_weight=g3_mse_weight,
            ic_weight=g3_ic_weight,
        )
    else:
        loss_g3 = torch.zeros((), device=x.device, dtype=x.dtype)

    if float(lambda_g4) != 0.0:
        loss_g4 = compute_g4_loss(generator, x, perturb_target_norm)
    else:
        loss_g4 = torch.zeros((), device=x.device, dtype=x.dtype)


    loss_total = (
        lambda_g1 * loss_g1 +
        lambda_g2 * loss_g2 +
        lambda_g3 * loss_g3 +
        lambda_g4 * loss_g4
    )

    return {
        'loss_g1': loss_g1,
        'loss_g2': loss_g2,
        'loss_g3': loss_g3,
        'loss_g4': loss_g4,
        'loss_total': loss_total
    }
