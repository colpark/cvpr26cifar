"""
Coordinate-based Flow Matching model with Fourier features.

Key innovations:
- Processes (x, y, RGB) tuples instead of image grids
- Continuous Fourier positional encoding
- Query-based decoding at arbitrary resolutions
- Scale-invariant by design
"""
import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from ..utils.time_emb import ContinuousTimeEmbedding


class FourierFeatures(nn.Module):
    """
    Continuous Fourier positional encoding for coordinates.

    Maps (x, y) ∈ [0,1]² to high-dimensional features using:
    [sin(2π f₁ x), cos(2π f₁ x), sin(2π f₁ y), cos(2π f₁ y), ...]

    Args:
        num_freqs: Number of frequency bands
        scale: Maximum frequency (higher = captures finer details)
    """
    def __init__(self, num_freqs=256, scale=50.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.scale = scale

        # Frequency bands: linearly spaced from 1 to scale
        freq_bands = torch.linspace(1.0, scale, num_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, coords):
        """
        Args:
            coords: (B, N, 2) coordinates in [0, 1]²

        Returns:
            (B, N, num_freqs*4) Fourier features
        """
        B, N, _ = coords.shape

        # Split into x and y
        x = coords[..., 0:1]  # (B, N, 1)
        y = coords[..., 1:2]  # (B, N, 1)

        # Apply frequencies
        freqs = self.freq_bands.view(1, 1, -1)  # (1, 1, num_freqs)

        # Compute Fourier features
        x_proj = 2 * math.pi * freqs * x  # (B, N, num_freqs)
        y_proj = 2 * math.pi * freqs * y  # (B, N, num_freqs)

        features = torch.cat([
            torch.sin(x_proj),
            torch.cos(x_proj),
            torch.sin(y_proj),
            torch.cos(y_proj)
        ], dim=-1)  # (B, N, num_freqs*4)

        return features


class CrossAttention(nn.Module):
    """Cross-attention for query-to-context interaction."""
    def __init__(self, query_dim, context_dim, num_heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        """
        Args:
            x: (B, N_q, D_q) queries
            context: (B, N_c, D_c) context

        Returns:
            (B, N_q, D_q)
        """
        B, N_q, D_q = x.shape

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Reshape for multi-head
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # Attention
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class SelfAttention(nn.Module):
    """Self-attention for latent processing."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, D)

        Returns:
            (B, N, D)
        """
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class FeedForward(nn.Module):
    """MLP with GELU activation."""
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CoordinateBasedFM(nn.Module):
    """
    Coordinate-based Flow Matching with dense supervision.

    Architecture:
    1. Encode sparse inputs with Fourier features
    2. Process with transformer blocks
    3. Decode queries with cross-attention
    4. Output RGB values at query coordinates

    Supports arbitrary resolution via query coordinates.

    Args:
        channel: Number of image channels (3 for RGB)
        dim: Model dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension multiplier
        num_fourier_freqs: Number of Fourier frequency bands
        fourier_scale: Maximum Fourier frequency
        dropout: Dropout rate
    """
    def __init__(self, channel=3, dim=512, depth=12, num_heads=8, mlp_ratio=4,
                 num_fourier_freqs=256, fourier_scale=50.0, dropout=0.1):
        super().__init__()
        self.channel = channel
        self.dim = dim

        # Fourier features for continuous positional encoding
        self.fourier = FourierFeatures(num_fourier_freqs, fourier_scale)
        fourier_dim = num_fourier_freqs * 4

        # Input projection: (Fourier features, RGB) → dim
        self.input_proj = nn.Linear(fourier_dim + channel, dim)

        # Query projection: Fourier features → dim
        self.query_proj = nn.Linear(fourier_dim, dim)

        # Time embedding
        self.time_embedding = ContinuousTimeEmbedding(dim // 4, dim)

        # Transformer blocks for processing inputs
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Cross-attention decoder: queries attend to processed inputs
        self.decoder_cross_attn = CrossAttention(dim, dim, num_heads, dim // num_heads, dropout)
        self.decoder_norm = nn.LayerNorm(dim)

        # Output projection: dim → RGB
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, channel)
        )

    def forward(self, query_coords, t, input_coords, input_values):
        """
        Forward pass with coordinate-based representation.

        Args:
            query_coords: (B, N_query, 2) query positions in [0,1]²
            t: (B,) continuous timesteps
            input_coords: (B, N_input, 2) sparse input positions
            input_values: (B, N_input, 3) RGB values at input positions

        Returns:
            (B, N_query, 3) predicted RGB values at query positions
        """
        B = query_coords.shape[0]

        # Time embedding
        time_emb = self.time_embedding(t)  # (B, dim)

        # Encode inputs with Fourier features
        input_feats = self.fourier(input_coords)  # (B, N_input, fourier_dim)
        input_tokens = self.input_proj(
            torch.cat([input_feats, input_values], dim=-1)
        )  # (B, N_input, dim)

        # Add time embedding to inputs
        input_tokens = input_tokens + time_emb.unsqueeze(1)

        # Process inputs through transformer
        for block in self.blocks:
            input_tokens = block(input_tokens)

        # Generate query tokens with Fourier features
        query_feats = self.fourier(query_coords)  # (B, N_query, fourier_dim)
        query_tokens = self.query_proj(query_feats)  # (B, N_query, dim)

        # Add time embedding to queries
        query_tokens = query_tokens + time_emb.unsqueeze(1)

        # Cross-attention: queries attend to processed inputs
        output_tokens = self.decoder_cross_attn(
            self.decoder_norm(query_tokens),
            input_tokens
        )  # (B, N_query, dim)

        # Project to RGB values
        output_values = self.output_proj(output_tokens)  # (B, N_query, 3)

        return output_values
