"""
Perceiver IO for Flow Matching with zero-shot super-resolution capability.
"""
import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from ..utils.time_emb import ContinuousTimeEmbedding


class CrossAttention(nn.Module):
    """Cross-attention: latents attend to inputs."""
    def __init__(self, query_dim, context_dim, num_heads=8, head_dim=64):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

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

        # Reshape for multi-head attention
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
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

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
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class PerceiverBlock(nn.Module):
    """Perceiver block: Cross-attention + Self-attention + FFN."""
    def __init__(self, latent_dim, input_dim, num_heads=8, head_dim=64, self_attn_layers=2):
        super().__init__()
        # Cross-attention from inputs
        self.cross_attn = CrossAttention(latent_dim, input_dim, num_heads, head_dim)
        self.cross_norm = nn.LayerNorm(latent_dim)

        # Self-attention layers
        self.self_attns = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(latent_dim),
                SelfAttention(latent_dim, num_heads),
                nn.LayerNorm(latent_dim),
                FeedForward(latent_dim)
            ])
            for _ in range(self_attn_layers)
        ])

    def forward(self, latents, inputs):
        """
        Args:
            latents: (B, N_latent, D_latent)
            inputs: (B, N_input, D_input)

        Returns:
            (B, N_latent, D_latent)
        """
        # Cross-attention
        latents = latents + self.cross_attn(self.cross_norm(latents), inputs)

        # Self-attention layers
        for norm1, self_attn, norm2, ffn in self.self_attns:
            latents = latents + self_attn(norm1(latents))
            latents = latents + ffn(norm2(latents))

        return latents


class PerceiverIOFM(nn.Module):
    """
    Perceiver IO for Flow Matching with resolution-agnostic design.

    Args:
        channel: Number of image channels (3 for RGB)
        latent_dim: Dimension of latent array
        num_latents: Number of latent vectors
        depth: Number of Perceiver blocks
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        input_fourier_features: Dimension of Fourier features for input
        query_fourier_features: Dimension of Fourier features for queries
    """
    def __init__(self, channel=3, latent_dim=512, num_latents=256, depth=6,
                 num_heads=8, head_dim=64, input_fourier_features=64, query_fourier_features=64):
        super().__init__()
        self.channel = channel
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        # Input projection: pixel values + 2D Fourier features
        input_dim = channel * 3 + input_fourier_features * 4  # x, sparse, mask + Fourier (sin/cos for 2D)
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Query projection: 2D Fourier features
        query_dim = query_fourier_features * 4
        self.query_proj = nn.Linear(query_dim, latent_dim)

        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Time embedding (will be added to latents)
        self.time_embedding = ContinuousTimeEmbedding(latent_dim // 4, latent_dim)

        # Perceiver blocks
        self.blocks = nn.ModuleList([
            PerceiverBlock(latent_dim, latent_dim, num_heads, head_dim)
            for _ in range(depth)
        ])

        # Output decoder: cross-attention from queries to latents
        self.decoder_cross_attn = CrossAttention(latent_dim, latent_dim, num_heads, head_dim)
        self.decoder_norm = nn.LayerNorm(latent_dim)

        # Output projection
        self.output_proj = nn.Linear(latent_dim, channel)

        # Fourier feature parameters
        self.input_fourier_scale = input_fourier_features
        self.query_fourier_scale = query_fourier_features

    def _fourier_features(self, coords, scale):
        """
        Generate Fourier features for 2D coordinates.

        Args:
            coords: (N, 2) coordinates in [-1, 1]
            scale: Number of frequency bands

        Returns:
            (N, scale*4) Fourier features
        """
        freq_bands = torch.linspace(1.0, scale/2, scale // 2, device=coords.device)

        feats = []
        for freq in freq_bands:
            feats.append(torch.sin(2 * math.pi * freq * coords[:, 0]))
            feats.append(torch.cos(2 * math.pi * freq * coords[:, 0]))
            feats.append(torch.sin(2 * math.pi * freq * coords[:, 1]))
            feats.append(torch.cos(2 * math.pi * freq * coords[:, 1]))

        return torch.stack(feats, dim=-1)

    def forward(self, x, t, sparse_input=None, mask=None):
        """
        Forward pass.

        Args:
            x: (B, C, H, W)
            t: (B,) continuous timesteps
            sparse_input: (B, C, H, W)
            mask: (B, C, H, W)

        Returns:
            (B, C, H, W) predicted velocity
        """
        assert sparse_input is not None and mask is not None

        B, C, H, W = x.shape

        # Ensure channel alignment
        if sparse_input.size(1) != C:
            sparse_input = sparse_input.repeat(1, C, 1, 1) if sparse_input.size(1) == 1 else sparse_input[:, :C]
        if mask.size(1) != C:
            mask = mask.repeat(1, C, 1, 1) if mask.size(1) == 1 else mask[:, :C]

        # Create coordinate grid
        y = torch.linspace(-1, 1, H, device=x.device)
        x_coords = torch.linspace(-1, 1, W, device=x.device)
        yy, xx = torch.meshgrid(y, x_coords, indexing='ij')
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # (H*W, 2)

        # Flatten spatial dimensions
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        sparse_flat = rearrange(sparse_input, 'b c h w -> b (h w) c')
        mask_flat = rearrange(mask, 'b c h w -> b (h w) c')

        # Concatenate pixel values
        pixels = torch.cat([x_flat, sparse_flat, mask_flat], dim=-1)  # (B, H*W, C*3)

        # Add Fourier features
        fourier_feats = self._fourier_features(coords, self.input_fourier_scale)  # (H*W, F)
        fourier_feats = repeat(fourier_feats, 'n f -> b n f', b=B)

        # Input tokens
        inputs = torch.cat([pixels, fourier_feats], dim=-1)  # (B, H*W, C*3+F)
        inputs = self.input_proj(inputs)  # (B, H*W, D)

        # Initialize latents
        latents = repeat(self.latents, 'n d -> b n d', b=B)

        # Add time embedding to latents
        time_emb = self.time_embedding(t)  # (B, D)
        latents = latents + time_emb.unsqueeze(1)

        # Process through Perceiver blocks
        for block in self.blocks:
            latents = block(latents, inputs)

        # Generate queries for output (same resolution as input by default)
        query_fourier = self._fourier_features(coords, self.query_fourier_scale)
        queries = repeat(query_fourier, 'n f -> b n f', b=B)
        queries = self.query_proj(queries)  # (B, H*W, D)

        # Decode: queries attend to latents
        output_tokens = self.decoder_cross_attn(self.decoder_norm(queries), latents)

        # Project to velocity
        output_flat = self.output_proj(output_tokens)  # (B, H*W, C)

        # Reshape to image
        output = rearrange(output_flat, 'b (h w) c -> b c h w', h=H, w=W)

        return output
