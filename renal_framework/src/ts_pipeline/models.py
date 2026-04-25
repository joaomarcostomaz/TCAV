"""
models.py

Time-series model zoo:
- TSMixer
- DLinear
- iTransformer
- PatchTST-style classifier
- NBEATSx-style classifier
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


# -----------------------------
# TSMixer
# -----------------------------
class TSMixerBlock(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout),
        )
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)

    def forward(self, x):
        z = self.norm1(x)
        z = z.transpose(1, 2)
        z = self.time_mlp(z)
        z = z.transpose(1, 2)
        x = x + z

        z = self.norm2(x)
        z = self.feature_mlp(z)
        x = x + z
        return x


class TSMixerClassifier(nn.Module):
    def __init__(self, seq_len: int, n_features: int, n_blocks: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TSMixerBlock(seq_len=seq_len, n_features=n_features, hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        hidden_out = n_features // 2 if n_features >= 2 else 1
        self.head = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_out, 1),
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        h = x[:, -1, :]
        return self.head(h).squeeze(-1)


# -----------------------------
# DLinear
# -----------------------------
class DLinearClassifier(nn.Module):
    def __init__(self, seq_len: int, n_features: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.linear_seasonal = nn.Linear(seq_len, seq_len)
        self.linear_trend = nn.Linear(seq_len, seq_len)

        self.head = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Dropout(dropout),
            nn.Linear(n_features, max(1, n_features // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, n_features // 2), 1),
        )

    def moving_avg(self, x):
        pad = (self.kernel_size - 1) // 2
        x_t = x.transpose(1, 2)
        x_pad = torch.nn.functional.pad(x_t, (pad, pad), mode="replicate")
        trend = torch.nn.functional.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        return trend.transpose(1, 2)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend

        s = self.linear_seasonal(seasonal.transpose(1, 2))
        t = self.linear_trend(trend.transpose(1, 2))

        y = (s + t).transpose(1, 2)
        h = y[:, -1, :]
        return self.head(h).squeeze(-1)


# -----------------------------
# iTransformer
# -----------------------------
class InvertedTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, z):
        a, _ = self.attn(z, z, z, need_weights=False)
        z = self.norm1(z + a)
        z = self.norm2(z + self.ffn(z))
        return z


class ITransformerClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_blocks: int = 3,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                InvertedTransformerBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        z = x.transpose(1, 2)
        z = self.input_proj(z)
        for blk in self.blocks:
            z = blk(z)
        z = z.mean(dim=1)
        return self.head(z).squeeze(-1)


# -----------------------------
# PatchTST-style
# -----------------------------
class PatchTSTClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        patch_len: int = 2,
        stride: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert patch_len <= seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = 1 + (seq_len - patch_len) // stride

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.feature_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.feature_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _patchify(self, x):
        x = x.transpose(1, 2)
        return x.unfold(dimension=2, size=self.patch_len, step=self.stride)

    def forward(self, x):
        p = self._patchify(x)
        B, F, NP, PL = p.shape
        p = p.reshape(B * F, NP, PL)

        z = self.patch_embed(p)
        cls = self.cls_token.expand(B * F, -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = z + self.pos_embed[:, : z.size(1), :]
        z = self.encoder(z)
        z_cls = z[:, 0, :]

        zf = z_cls.reshape(B, F, -1)
        q = self.feature_query.expand(B, -1, -1)
        pooled, _ = self.feature_attn(q, zf, zf, need_weights=False)
        h = pooled[:, 0, :]

        return self.cls_head(h).squeeze(-1)


# -----------------------------
# NBEATSx-style
# -----------------------------
class NBEATSxBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_dim, input_dim)
        self.theta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        h = self.mlp(x)
        back = self.backcast(h)
        theta = self.theta(h)
        return back, theta


class NBEATSxClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        n_blocks: int = 4,
        hidden_dim: int = 512,
        n_layers_per_block: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = seq_len * n_features
        self.blocks = nn.ModuleList(
            [
                NBEATSxBlock(
                    input_dim=self.input_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers_per_block,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        residual = x.reshape(x.size(0), -1)
        thetas = []
        for blk in self.blocks:
            back, theta = blk(residual)
            residual = residual - back
            thetas.append(theta)
        h = torch.stack(thetas, dim=0).mean(dim=0)
        return self.head(h).squeeze(-1)


# -----------------------------
# Factory
# -----------------------------
def build_model(model_name: str, seq_len: int, n_features: int) -> nn.Module:
    m = model_name.lower()
    if m == "tsmixer":
        return TSMixerClassifier(seq_len=seq_len, n_features=n_features, n_blocks=4, hidden_dim=256, dropout=0.1)
    if m == "dlinear":
        return DLinearClassifier(seq_len=seq_len, n_features=n_features, kernel_size=3, dropout=0.1)
    if m == "itransformer":
        return ITransformerClassifier(seq_len=seq_len, n_features=n_features, d_model=128, n_heads=4, n_blocks=3, mlp_ratio=2, dropout=0.1)
    if m == "patchtst":
        return PatchTSTClassifier(seq_len=seq_len, n_features=n_features, patch_len=2, stride=1, d_model=128, n_heads=4, n_layers=3, ff_mult=2, dropout=0.1)
    if m == "nbeatsx":
        return NBEATSxClassifier(seq_len=seq_len, n_features=n_features, n_blocks=4, hidden_dim=512, n_layers_per_block=3, dropout=0.1)
    raise ValueError(f"Unknown model_name: {model_name}")