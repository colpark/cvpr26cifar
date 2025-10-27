"""
Time embeddings for discrete (DDPM) and continuous (Flow Matching) timesteps.
"""
import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for discrete timesteps (DDPM).
    From reference code: PositionalEncoding.
    """
    def __init__(self, dim, max_len=10000):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t):
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            (B, dim) embeddings
        """
        return self.pe[t]


class GaussianFourierEmbedding(nn.Module):
    """
    Gaussian Fourier features for continuous time t ∈ [0, 1] (Flow Matching).
    Projects t with random Fourier features: [sin(2πBt), cos(2πBt)]
    """
    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.dim = dim
        # B ~ N(0, scale^2)
        self.register_buffer('B', torch.randn(1, dim // 2) * scale)

    def forward(self, t):
        """
        Args:
            t: (B,) or (B, 1) float timesteps in [0, 1]
        Returns:
            (B, dim) embeddings
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)

        # 2π * B * t
        proj = 2 * math.pi * t @ self.B  # (B, dim//2)

        # [sin, cos]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, dim)


class TimeEmbeddingMLP(nn.Module):
    """
    MLP to process time embeddings (discrete or continuous) into final embedding.
    Used in U-Net and transformers.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, t_emb):
        """
        Args:
            t_emb: (B, in_dim)
        Returns:
            (B, out_dim)
        """
        return self.mlp(t_emb)


class DiscreteTimeEmbedding(nn.Module):
    """Complete time embedding for DDPM: positional encoding + MLP."""
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.encoding = SinusoidalPositionalEncoding(dim)
        self.mlp = TimeEmbeddingMLP(dim, time_emb_dim)

    def forward(self, t):
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            (B, time_emb_dim)
        """
        t_emb = self.encoding(t)
        return self.mlp(t_emb)


class ContinuousTimeEmbedding(nn.Module):
    """Complete time embedding for Flow Matching: Gaussian Fourier + MLP."""
    def __init__(self, dim, time_emb_dim, scale=16.0):
        super().__init__()
        self.encoding = GaussianFourierEmbedding(dim, scale=scale)
        self.mlp = TimeEmbeddingMLP(dim, time_emb_dim)

    def forward(self, t):
        """
        Args:
            t: (B,) float timesteps in [0, 1]
        Returns:
            (B, time_emb_dim)
        """
        t_emb = self.encoding(t)
        return self.mlp(t_emb)
