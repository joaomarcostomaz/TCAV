"""
models_v2.py

Same model zoo, but each model exposes:
- forward_representation(x): latent representation (sequence-aware or pooled)
- forward_from_representation(h): prediction from latent representation
"""

from __future__ import annotations
import torch
import torch.nn as nn


# -----------------------------
# TSMixer
# -----------------------------
class TSMixerBlock(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len), nn.Dropout(dropout),
        )
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)

    def forward(self, x):
        z = self.norm1(x).transpose(1, 2)
        z = self.time_mlp(z).transpose(1, 2)
        x = x + z
        z = self.norm2(x)
        z = self.feature_mlp(z)
        return x + z


class TSMixerClassifierV2(nn.Module):
    def __init__(self, seq_len: int, n_features: int, n_blocks: int = 4, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TSMixerBlock(seq_len, n_features, hidden_dim, dropout) for _ in range(n_blocks)])
        hdim = max(1, n_features // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hdim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim, 1),
        )

    def forward_representation(self, x):
        # returns [B,T,F]
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_from_representation(self, h):
        # expect [B,T,F] -> use last token
        if h.ndim == 2:
            h_last = h
        else:
            h_last = h[:, -1, :]
        return self.head(h_last).squeeze(-1)

    def forward(self, x):
        h = self.forward_representation(x)
        return self.forward_from_representation(h)


# -----------------------------
# DLinear
# -----------------------------
class DLinearClassifierV2(nn.Module):
    def __init__(self, seq_len: int, n_features: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.linear_seasonal = nn.Linear(seq_len, seq_len)
        self.linear_trend = nn.Linear(seq_len, seq_len)
        hdim = max(1, n_features // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Dropout(dropout),
            nn.Linear(n_features, hdim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim, 1),
        )

    def moving_avg(self, x):
        pad = (self.kernel_size - 1) // 2
        x_t = x.transpose(1, 2)
        x_pad = torch.nn.functional.pad(x_t, (pad, pad), mode="replicate")
        trend = torch.nn.functional.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        return trend.transpose(1, 2)

    def forward_representation(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        s = self.linear_seasonal(seasonal.transpose(1, 2))
        t = self.linear_trend(trend.transpose(1, 2))
        y = (s + t).transpose(1, 2)  # [B,T,F]
        return y

    def forward_from_representation(self, h):
        if h.ndim == 2:
            h_last = h
        else:
            h_last = h[:, -1, :]
        return self.head(h_last).squeeze(-1)

    def forward(self, x):
        h = self.forward_representation(x)
        return self.forward_from_representation(h)


# -----------------------------
# iTransformer
# -----------------------------
class InvertedTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, z):
        a, _ = self.attn(z, z, z, need_weights=False)
        z = self.norm1(z + a)
        z = self.norm2(z + self.ffn(z))
        return z


class ITransformerClassifierV2(nn.Module):
    def __init__(self, seq_len: int, n_features: int, d_model: int = 128, n_heads: int = 4, n_blocks: int = 3, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(seq_len, d_model)
        self.blocks = nn.ModuleList([InvertedTransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward_representation(self, x):
        # x [B,T,F] -> [B,F,d_model]
        z = x.transpose(1, 2)
        z = self.input_proj(z)
        for blk in self.blocks:
            z = blk(z)
        # map to [B,1,d_model] so SAE can be temporal-compatible
        return z.mean(dim=1, keepdim=True)

    def forward_from_representation(self, h):
        if h.ndim == 3:
            h = h[:, 0, :]
        return self.head(h).squeeze(-1)

    def forward(self, x):
        h = self.forward_representation(x)
        return self.forward_from_representation(h)


def build_model_v2(model_name: str, seq_len: int, n_features: int) -> nn.Module:
    m = model_name.lower()
    if m == "tsmixer":
        return TSMixerClassifierV2(seq_len, n_features)
    if m == "dlinear":
        return DLinearClassifierV2(seq_len, n_features)
    if m == "itransformer":
        return ITransformerClassifierV2(seq_len, n_features)
    raise ValueError(f"Unknown model_name for v2 hooks: {model_name}")