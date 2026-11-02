from dataclasses import dataclass, asdict
from contextlib import contextmanager
import torch
from torch import nn
from itransformer_encoder import ITransformerEncoder, VanillaTransformerEncoder
from perturbation_generator import CaRIIntervention
from factor_network import ReturnVAE
from beta_network import BetaNetwork, pricing_returns
from classification_adversary import DependencyDiscriminator
from prediction_adversary import RobustPredictionHead


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


def adversarial_input(z, enabled=True, scale=1.0):

    return _Reverse.apply(z, scale) if enabled else z


@contextmanager
def frozen(*modules):
    params = list(dict.fromkeys(p for m in modules for p in m.parameters()))
    prior = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(params, prior):
            p.requires_grad_(flag)


@dataclass
class ModelConfig:
    n_assets: int
    feature_dim: int
    window: int = 3
    z_dim: int = 32
    factor_dim: int = 8
    transformer_nhead: int = 4
    transformer_layers_time: int = 2
    transformer_dropout: float = 0.1
    vae_hidden_dim: int = 32
    beta_hidden_dim: int = 0
    d1_hidden_dim: int = 64
    perturb_target_norm: float = 0.1
    disable_itransformer: bool = False
    disable_adversarial: bool = False
    disable_grl: bool = False

    def __post_init__(self):
        for name in (
            "n_assets",
            "feature_dim",
            "window",
            "z_dim",
            "factor_dim",
            "transformer_nhead",
            "transformer_layers_time",
            "vae_hidden_dim",
            "d1_hidden_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.z_dim % self.transformer_nhead:
            raise ValueError("z_dim must be divisible by transformer_nhead")
        if self.beta_hidden_dim < 0 or not 0 <= self.transformer_dropout < 1:
            raise ValueError("Invalid beta hidden dimension or Transformer dropout")
        if not 0 < self.perturb_target_norm < float("inf"):
            raise ValueError("perturb_target_norm must be finite and positive")


class CASH(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        encoder = (
            VanillaTransformerEncoder
            if (config.disable_itransformer or config.disable_adversarial)
            else ITransformerEncoder
        )
        self.encoder = encoder(
            config.feature_dim,
            config.window,
            config.z_dim,
            config.transformer_nhead,
            config.transformer_layers_time,
            config.transformer_dropout,
        )
        self.factor = ReturnVAE(
            config.n_assets, config.factor_dim, config.vae_hidden_dim
        )
        self.beta = BetaNetwork(config.z_dim, config.factor_dim, config.beta_hidden_dim)
        self.perturb = CaRIIntervention(config.z_dim, config.perturb_target_norm)
        self.dcls = DependencyDiscriminator(
            config.feature_dim, config.z_dim, config.d1_hidden_dim
        )
        self.dpred = RobustPredictionHead(config.z_dim, config.d1_hidden_dim)

    def covariates(self, history):
        return history[:, -1]

    def forward(self, history, realized_returns):
        z = self.encoder(history)
        return pricing_returns(self.beta(z), self.factor.factors(realized_returns))

    def metadata(self):
        return asdict(self.config)
