"""
DiT (Diffusion Transformer) style model for Flow Matching.
Resolution-agnostic with 2D Fourier positional encoding for zero-shot SR.
"""
import torch
import torch.nn as nn
import math
from einops import rearrange
from ..utils.time_emb import ContinuousTimeEmbedding


class FourierPositionalEncoding2D(nn.Module):
    """
    2D Fourier positional encoding that is resolution-agnostic.
    """
    def __init__(self, dim, max_freq=10.0):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

        # Random Fourier features
        num_freq = dim // 4  # sin/cos for x and y
        self.register_buffer('freq_bands', torch.linspace(1.0, max_freq, num_freq))

    def forward(self, h, w, device):
        """
        Generate 2D positional encodings for h×w grid.

        Returns:
            (h*w, dim) positional embeddings
        """
        # Create normalized coordinate grid [-1, 1]
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

        # Flatten
        coords = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=-1)  # (h*w, 2)

        # Apply Fourier features
        pos_enc = []
        for freq in self.freq_bands:
            pos_enc.append(torch.sin(2 * math.pi * freq * coords[:, 0]))  # sin(y)
            pos_enc.append(torch.cos(2 * math.pi * freq * coords[:, 0]))  # cos(y)
            pos_enc.append(torch.sin(2 * math.pi * freq * coords[:, 1]))  # sin(x)
            pos_enc.append(torch.cos(2 * math.pi * freq * coords[:, 1]))  # cos(x)

        pos_enc = torch.stack(pos_enc, dim=-1)  # (h*w, dim)
        return pos_enc


class WindowedAttention(nn.Module):
    """
    Windowed self-attention for memory efficiency.
    """
    def __init__(self, dim, num_heads=8, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        """
        Args:
            x: (B, N, D) where N = h*w
            h, w: Spatial dimensions

        Returns:
            (B, N, D)
        """
        B, N, D = x.shape

        # Reshape to spatial
        x_spatial = rearrange(x, 'b (h w) d -> b h w d', h=h, w=w)

        # Window partition (simplified - full attention if small enough)
        if h * w <= 1024:  # Full attention for small resolutions
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        else:
            # For larger resolutions, use local windows
            # Simplified: just use full attention for now
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, D)

        return self.proj(out)


class DiTBlock(nn.Module):
    """
    DiT transformer block: Attention + MLP with adaptive layer norm.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(dim, num_heads, window_size)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )

        # Adaptive modulation from time embedding
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)  # scale/shift for norm1 and norm2, gate for residuals
        )

    def forward(self, x, time_emb, h, w):
        """
        Args:
            x: (B, N, D)
            time_emb: (B, D)
            h, w: Spatial dimensions
        """
        # Get modulation parameters
        mod = self.adaLN_modulation(time_emb)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Attention with adaptive norm
        x = x + gate_attn.unsqueeze(1) * self.attn(
            self.norm1(x) * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1),
            h, w
        )

        # MLP with adaptive norm
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        )

        return x


class DiTFM(nn.Module):
    """
    Diffusion Transformer for Flow Matching with resolution-agnostic design.

    Args:
        patch_size: Patch size for tokenization (2 or 4)
        dim: Model dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        channel: Number of image channels (3 for RGB)
        dropout: Dropout rate
        window_size: Window size for attention
    """
    def __init__(self, patch_size=4, dim=512, depth=12, num_heads=8, mlp_ratio=4.0,
                 channel=3, dropout=0.1, window_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.channel = channel

        # Patch embedding: [x, sparse, mask] -> tokens
        self.patch_embed = nn.Conv2d(channel * 3, dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding (resolution-agnostic)
        self.pos_encoding = FourierPositionalEncoding2D(dim)

        # Time embedding
        self.time_embedding = ContinuousTimeEmbedding(dim // 4, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio, dropout, window_size)
            for _ in range(depth)
        ])

        # Output head: tokens -> patches -> image
        self.final_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, patch_size * patch_size * channel)

    def patchify(self, x, sparse_input, mask):
        """
        Convert images to patches.

        Args:
            x: (B, C, H, W)
            sparse_input: (B, C, H, W)
            mask: (B, C, H, W)

        Returns:
            tokens: (B, N, D) where N = (H/P)*(W/P)
            h_patches, w_patches: Number of patches in each dimension
        """
        # Ensure channel alignment
        C = x.size(1)
        if sparse_input.size(1) != C:
            sparse_input = sparse_input.repeat(1, C, 1, 1) if sparse_input.size(1) == 1 else sparse_input[:, :C]
        if mask.size(1) != C:
            mask = mask.repeat(1, C, 1, 1) if mask.size(1) == 1 else mask[:, :C]

        # Concatenate
        x_concat = torch.cat([x, sparse_input, mask], dim=1)  # (B, C*3, H, W)

        # Patch embedding
        tokens = self.patch_embed(x_concat)  # (B, D, H/P, W/P)
        B, D, h_patches, w_patches = tokens.shape

        tokens = rearrange(tokens, 'b d h w -> b (h w) d')
        return tokens, h_patches, w_patches

    def unpatchify(self, tokens, h_patches, w_patches):
        """
        Convert tokens back to image.

        Args:
            tokens: (B, N, D)
            h_patches, w_patches: Number of patches

        Returns:
            (B, C, H, W)
        """
        tokens = rearrange(tokens, 'b (h w) d -> b d h w', h=h_patches, w=w_patches)

        # Project to patches
        patches = rearrange(tokens, 'b d h w -> b (h w) d')
        patches = self.output_proj(patches)  # (B, N, P*P*C)

        # Reshape to image
        B, N, _ = patches.shape
        P = self.patch_size
        C = self.channel

        patches = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                           h=h_patches, w=w_patches, p1=P, p2=P, c=C)

        return patches

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

        # Patchify
        tokens, h_patches, w_patches = self.patchify(x, sparse_input, mask)
        B, N, D = tokens.shape

        # Add positional encoding
        pos_enc = self.pos_encoding(h_patches, w_patches, tokens.device)
        tokens = tokens + pos_enc.unsqueeze(0)

        # Time embedding
        time_emb = self.time_embedding(t)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, time_emb, h_patches, w_patches)

        # Final norm
        tokens = self.final_norm(tokens)

        # Unpatchify
        output = self.unpatchify(tokens, h_patches, w_patches)

        return output
