"""
U-Net architecture for Flow Matching with continuous time embeddings.
Nearly identical to DDPM U-Net, but uses continuous time t ∈ [0, 1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..utils.time_emb import ContinuousTimeEmbedding


class ResnetBlock(nn.Module):
    """Residual block with time embedding and GroupNorm."""
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=0.0, groups=32):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        ) if time_emb_dim is not None else None

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=dim_out)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)

        self.residual_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.conv1(self.act1(self.norm1(x)))

        if time_emb is not None and self.time_mlp is not None:
            h = h + self.time_mlp(time_emb)[:, :, None, None]

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.residual_conv(x)


class Attention(nn.Module):
    """Self-attention module."""
    def __init__(self, dim, groups=32):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.to_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), qkv)

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j c -> b i c', attn, v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return self.to_out(out) + x


class ResnetAttentionBlock(nn.Module):
    """Resnet block followed by attention."""
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=0.0, groups=32):
        super().__init__()
        self.resnet = ResnetBlock(dim, dim_out, time_emb_dim, dropout, groups)
        self.attention = Attention(dim_out if dim_out else dim, groups)

    def forward(self, x, time_emb=None):
        x = self.resnet(x, time_emb)
        return self.attention(x)


class DownSample(nn.Module):
    """Downsampling via strided convolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    """Upsampling via nearest neighbor + convolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNetFM(nn.Module):
    """
    U-Net for Flow Matching with continuous time embeddings t ∈ [0, 1].
    Predicts velocity v instead of noise.

    Args:
        dim: Base channel dimension
        image_size: Input image size (32 for CIFAR-10)
        dim_multiply: Channel multipliers per resolution level
        channel: Number of image channels (3 for RGB)
        num_res_blocks: Number of ResNet blocks per level
        attn_resolutions: Resolutions at which to apply attention
        dropout: Dropout rate
        groups: Number of groups for GroupNorm
        time_emb_scale: Scale for Gaussian Fourier features
    """
    def __init__(self, dim=128, image_size=32, dim_multiply=(1, 2, 4, 4),
                 channel=3, num_res_blocks=2, attn_resolutions=(16,), dropout=0.1,
                 groups=32, time_emb_scale=16.0):
        super().__init__()
        self.dim = dim
        self.channel = channel
        self.time_emb_dim = 4 * dim
        self.num_resolutions = len(dim_multiply)
        self.image_size = image_size

        # Continuous time embedding
        self.time_embedding = ContinuousTimeEmbedding(dim, self.time_emb_dim, scale=time_emb_scale)

        # Resolution at each level
        self.resolutions = [int(image_size / (2 ** i)) for i in range(self.num_resolutions)]
        self.hidden_dims = [dim, *[dim * m for m in dim_multiply]]

        # Initial convolution: channel*3 -> dim (concatenation contract)
        self.init_conv = nn.Conv2d(channel * 3, dim, kernel_size=3, padding=1)

        # Downward path
        self.down_path = nn.ModuleList([])
        concat_dims = [dim]

        for level in range(self.num_resolutions):
            d_in, d_out = self.hidden_dims[level], self.hidden_dims[level + 1]
            for block in range(num_res_blocks):
                d_in_ = d_in if block == 0 else d_out
                if self.resolutions[level] in attn_resolutions:
                    self.down_path.append(
                        ResnetAttentionBlock(d_in_, d_out, self.time_emb_dim, dropout, groups)
                    )
                else:
                    self.down_path.append(
                        ResnetBlock(d_in_, d_out, self.time_emb_dim, dropout, groups)
                    )
                concat_dims.append(d_out)

            if level != self.num_resolutions - 1:
                self.down_path.append(DownSample(d_out))
                concat_dims.append(d_out)

        # Middle blocks
        mid_dim = self.hidden_dims[-1]
        self.middle = nn.ModuleList([
            ResnetAttentionBlock(mid_dim, mid_dim, self.time_emb_dim, dropout, groups),
            ResnetBlock(mid_dim, mid_dim, self.time_emb_dim, dropout, groups)
        ])

        # Upward path
        self.up_path = nn.ModuleList([])
        for level in reversed(range(self.num_resolutions)):
            d_out = self.hidden_dims[level + 1]
            for block in range(num_res_blocks + 1):
                d_in = concat_dims.pop()
                d_in = d_in + (self.hidden_dims[level + 2] if block == 0 and level != self.num_resolutions - 1 else d_out)

                if self.resolutions[level] in attn_resolutions:
                    self.up_path.append(
                        ResnetAttentionBlock(d_in, d_out, self.time_emb_dim, dropout, groups)
                    )
                else:
                    self.up_path.append(
                        ResnetBlock(d_in, d_out, self.time_emb_dim, dropout, groups)
                    )

            if level != 0:
                self.up_path.append(UpSample(d_out))

        # Output convolution (predicts velocity)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(groups, self.hidden_dims[1]),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dims[1], channel, kernel_size=3, padding=1)
        )

    def forward(self, x, t, sparse_input=None, mask=None):
        """
        Forward pass.

        Args:
            x: (B, C*3, H, W) concatenated input [x_t, sparse, mask]
               OR (B, C, H, W) if sparse_input and mask provided separately
            t: (B,) continuous timesteps in [0, 1]
            sparse_input: Optional (B, C, H, W) sparse input
            mask: Optional (B, C, H, W) mask

        Returns:
            (B, C, H, W) predicted velocity
        """
        # Handle both concatenated and separate inputs
        if sparse_input is not None and mask is not None:
            # Ensure channel alignment
            C = x.size(1)
            if sparse_input.size(1) != C:
                sparse_input = sparse_input.repeat(1, C, 1, 1) if sparse_input.size(1) == 1 else sparse_input[:, :C]
            if mask.size(1) != C:
                mask = mask.repeat(1, C, 1, 1) if mask.size(1) == 1 else mask[:, :C]
            x = torch.cat([x, sparse_input, mask], dim=1)
        else:
            # Unconditional generation: use zeros for sparse_input and mask
            C = x.size(1)
            sparse_input = torch.zeros_like(x)
            mask = torch.zeros_like(x)
            x = torch.cat([x, sparse_input, mask], dim=1)

        # Time embedding
        t_emb = self.time_embedding(t)

        # Initial conv
        x = self.init_conv(x)

        # Downward path with skip connections
        skips = [x]
        for layer in self.down_path:
            if isinstance(layer, (DownSample, UpSample)):
                x = layer(x)
            else:
                x = layer(x, t_emb)
            skips.append(x)

        # Middle
        for layer in self.middle:
            x = layer(x, t_emb)

        # Upward path with skip connections
        for layer in self.up_path:
            if isinstance(layer, UpSample):
                x = layer(x)
            else:
                x = torch.cat([x, skips.pop()], dim=1)
                x = layer(x, t_emb)

        # Final output
        return self.final_conv(x)
