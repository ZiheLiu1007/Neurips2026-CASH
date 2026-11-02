import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PreNormTransformerLayer(nn.Module):


    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = None,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model


        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )


        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):


        src_norm = self.norm1(src)
        attn_out, _ = self.self_attn(
            src_norm, src_norm, src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = src + self.dropout(attn_out)


        src_norm = self.norm2(src)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout(ff_out)

        return src


def generate_causal_mask(seq_len: int, device=None):


    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask


class TimeTransformerEncoder(nn.Module):


    def __init__(self, input_dim: int, d_model: int, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = None,
                 dropout: float = 0.1, max_len: int = 1000):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers


        self.input_proj = nn.Linear(input_dim, d_model)


        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)


        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])


        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, batch_size_per_group=None):


        B, T, N, P = x.shape


        if batch_size_per_group is not None and B > batch_size_per_group:
            outputs = []
            for b_start in range(0, B, batch_size_per_group):
                b_end = min(b_start + batch_size_per_group, B)
                x_group = x[b_start:b_end]
                out_group = self._forward_single(x_group)
                outputs.append(out_group)
            return torch.cat(outputs, dim=0)
        else:
            return self._forward_single(x)

    def _forward_single(self, x):

        B, T, N, P = x.shape


        x_reshaped = x.permute(0, 2, 1, 3).contiguous()
        x_flat = x_reshaped.reshape(B * N, T, P)


        x_proj = self.input_proj(x_flat)


        x_pos = self.pos_encoding(x_proj)


        causal_mask = generate_causal_mask(T, device=x.device)


        out = x_pos
        for layer in self.layers:
            out = layer(out, src_mask=causal_mask)


        out = self.output_norm(out)


        out = out.reshape(B, N, T, self.d_model)
        out = out.permute(0, 2, 1, 3).contiguous()

        return out


class CrossSectionTransformerEncoder(nn.Module):


    def __init__(self, input_dim: int, d_model: int, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = None,
                 dropout: float = 0.1, batch_size_per_time: int = None):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.batch_size_per_time = batch_size_per_time


        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()


        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])


        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x):


        B, T, N, d = x.shape


        if self.batch_size_per_time is not None and T > self.batch_size_per_time:

            outputs = []
            for t_start in range(0, T, self.batch_size_per_time):
                t_end = min(t_start + self.batch_size_per_time, T)
                x_time_group = x[:, t_start:t_end]
                out_time_group = self._forward_time_group(x_time_group)
                outputs.append(out_time_group)
            return torch.cat(outputs, dim=1)
        else:
            return self._forward_time_group(x)

    def _forward_time_group(self, x):

        B, group_T, N, d = x.shape


        x_reshaped = x.reshape(B * group_T, N, d)


        x_proj = self.input_proj(x_reshaped)


        out = x_proj
        for layer in self.layers:
            out = layer(out)


        out = self.output_norm(out)


        out = out.reshape(B, group_T, N, self.d_model)

        return out


class PositionalEncoding(nn.Module):


    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):


        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalCrossSectionEncoder(nn.Module):


    def __init__(self, input_dim: int, z_dim: int, nhead: int = 4,
                 num_layers_time: int = 2, num_layers_cross: int = 2,
                 dropout: float = 0.1, dim_feedforward: int = None,
                 batch_size_per_group: int = None,
                 batch_size_per_time: int = None,
                 window: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.window = window


        self.token_embedding = nn.Sequential(
            nn.Linear(window, z_dim), nn.GELU(), nn.Linear(z_dim, z_dim)
        )


        self.layers = nn.ModuleList([
            PreNormTransformerLayer(z_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_layers_time)
        ])
        self.output_norm = nn.LayerNorm(z_dim)
        self.projection = nn.Linear(z_dim, z_dim)


        self.batch_size_per_group = batch_size_per_group

    def forward(self, x):


        B, T, N, P = x.shape
        if T != self.window or P != self.input_dim:
            raise ValueError("Input window or feature dimension does not match the encoder")


        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite encoder input")


        if self.batch_size_per_group is not None and B > self.batch_size_per_group:
            return torch.cat([
                self._forward_single(x[start:start + self.batch_size_per_group])
                for start in range(0, B, self.batch_size_per_group)
            ], dim=0)
        return self._forward_single(x)


    def _forward_single(self, x):
        B, T, N, P = x.shape
        out = x.permute(0, 2, 3, 1).reshape(B * N, P, T)
        out = self.token_embedding(out)
        for layer in self.layers:
            out = layer(out)
        out = self.output_norm(out)


        z = self.projection(out.mean(dim=1)).reshape(B, N, self.z_dim)

        return z


class NoITransformerEncoder(nn.Module):


    def __init__(self, input_dim: int, z_dim: int, pooling: str = "last",
                 window: int = 3, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, dim_feedforward: int = None,
                 batch_size_per_group: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        if pooling not in {"last", "mean"}:
            raise ValueError(f"unsupported pooling mode: {pooling}")
        self.pooling = pooling
        self.window = window
        self.batch_size_per_group = batch_size_per_group
        self.time_transformer = TimeTransformerEncoder(
            input_dim, z_dim, nhead, num_layers, dim_feedforward, dropout
        )
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, x):


        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite encoder input")
        if x.shape[1] != self.window or x.shape[-1] != self.input_dim:
            raise ValueError("Input window or feature dimension does not match the encoder")
        x = self.time_transformer(x, batch_size_per_group=self.batch_size_per_group)
        if self.pooling == "mean":
            x_pool = x.mean(dim=1)
        else:
            x_pool = x[:, -1, :, :]
        z = x_pool
        z = self.norm(z)
        if not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite encoder output")
        return z
