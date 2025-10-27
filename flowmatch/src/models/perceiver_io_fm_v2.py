"""
Perceiver IO for Flow Matching V2 with improved zero-shot super-resolution.

Key improvements:
1. Separate target_size parameter for query generation
2. Spatial positional encoding for latents
3. Better Fourier feature normalization for multi-scale
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


class PerceiverIOFMV2(nn.Module):
    """
    Perceiver IO V2 for Flow Matching with improved zero-shot super-resolution.

    Key improvements:
    1. Separate encoding and decoding resolutions
    2. Spatial-aware latent initialization
    3. Resolution-normalized Fourier features

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
        # Fourier features: (scale // 2) frequency bands * 4 (sin/cos for x and y)
        input_dim = channel * 3 + (input_fourier_features // 2) * 4
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Query projection: 2D Fourier features
        query_dim = (query_fourier_features // 2) * 4
        self.query_proj = nn.Linear(query_dim, latent_dim)

        # Learnable latent array with spatial structure
        # Arrange latents in a grid for better spatial awareness
        latent_h = latent_w = int(math.sqrt(num_latents))
        assert latent_h * latent_w == num_latents, f"num_latents must be a perfect square, got {num_latents}"
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Spatial positional encoding for latents
        self.latent_pos_embed = nn.Parameter(torch.randn(num_latents, latent_dim))

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
        Generate resolution-normalized Fourier features for 2D coordinates.

        Args:
            coords: (N, 2) coordinates in [-1, 1]
            scale: Number of frequency bands

        Returns:
            (N, scale*2) Fourier features (half the previous to match dimension)
        """
        freq_bands = torch.linspace(1.0, scale/2, scale // 2, device=coords.device)

        feats = []
        for freq in freq_bands:
            feats.append(torch.sin(2 * math.pi * freq * coords[:, 0]))
            feats.append(torch.cos(2 * math.pi * freq * coords[:, 0]))
            feats.append(torch.sin(2 * math.pi * freq * coords[:, 1]))
            feats.append(torch.cos(2 * math.pi * freq * coords[:, 1]))

        return torch.stack(feats, dim=-1)

    def forward(self, x, t, sparse_input=None, mask=None, target_size=None):
        """
        Forward pass with support for variable output resolution.

        Args:
            x: (B, C, H, W)
            t: (B,) continuous timesteps
            sparse_input: (B, C, H, W) or None for unconditional
            mask: (B, C, H, W) or None for unconditional
            target_size: (H_out, W_out) or None for same as input

        Returns:
            (B, C, H_out, W_out) predicted velocity
        """
        B, C, H_in, W_in = x.shape

        # Handle unconditional generation
        if sparse_input is None:
            sparse_input = torch.zeros_like(x)
        if mask is None:
            mask = torch.zeros_like(x)

        # Ensure channel alignment
        if sparse_input.size(1) != C:
            sparse_input = sparse_input.repeat(1, C, 1, 1) if sparse_input.size(1) == 1 else sparse_input[:, :C]
        if mask.size(1) != C:
            mask = mask.repeat(1, C, 1, 1) if mask.size(1) == 1 else mask[:, :C]

        # === ENCODING at input resolution ===
        # Create coordinate grid for input
        y_in = torch.linspace(-1, 1, H_in, device=x.device)
        x_in_coords = torch.linspace(-1, 1, W_in, device=x.device)
        yy_in, xx_in = torch.meshgrid(y_in, x_in_coords, indexing='ij')
        coords_in = torch.stack([yy_in.flatten(), xx_in.flatten()], dim=-1)  # (H_in*W_in, 2)

        # Flatten spatial dimensions
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        sparse_flat = rearrange(sparse_input, 'b c h w -> b (h w) c')
        mask_flat = rearrange(mask, 'b c h w -> b (h w) c')

        # Concatenate pixel values
        pixels = torch.cat([x_flat, sparse_flat, mask_flat], dim=-1)  # (B, H_in*W_in, C*3)

        # Add Fourier features
        fourier_feats_in = self._fourier_features(coords_in, self.input_fourier_scale)  # (H_in*W_in, F)
        fourier_feats_in = repeat(fourier_feats_in, 'n f -> b n f', b=B)

        # Input tokens
        inputs = torch.cat([pixels, fourier_feats_in], dim=-1)  # (B, H_in*W_in, C*3+F)
        inputs = self.input_proj(inputs)  # (B, H_in*W_in, D)

        # Initialize latents with learned embeddings + spatial positional encoding
        latents = repeat(self.latents, 'n d -> b n d', b=B)
        latents = latents + self.latent_pos_embed.unsqueeze(0)

        # Add time embedding to latents
        time_emb = self.time_embedding(t)  # (B, D)
        latents = latents + time_emb.unsqueeze(1)

        # Process through Perceiver blocks
        for block in self.blocks:
            latents = block(latents, inputs)

        # === DECODING at target resolution ===
        # Determine output resolution
        if target_size is None:
            H_out, W_out = H_in, W_in
        else:
            H_out, W_out = target_size

        # Generate queries for output resolution
        y_out = torch.linspace(-1, 1, H_out, device=x.device)
        x_out_coords = torch.linspace(-1, 1, W_out, device=x.device)
        yy_out, xx_out = torch.meshgrid(y_out, x_out_coords, indexing='ij')
        coords_out = torch.stack([yy_out.flatten(), xx_out.flatten()], dim=-1)  # (H_out*W_out, 2)

        query_fourier = self._fourier_features(coords_out, self.query_fourier_scale)
        queries = repeat(query_fourier, 'n f -> b n f', b=B)
        queries = self.query_proj(queries)  # (B, H_out*W_out, D)

        # Decode: queries attend to latents
        output_tokens = self.decoder_cross_attn(self.decoder_norm(queries), latents)

        # Project to velocity
        output_flat = self.output_proj(output_tokens)  # (B, H_out*W_out, C)

        # Reshape to image
        output = rearrange(output_flat, 'b (h w) c -> b c h w', h=H_out, w=W_out)

        return output
