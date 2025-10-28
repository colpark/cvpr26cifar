"""
Mamba with Coordinate-Based Flow Matching

Combines Mamba's efficient SSM with coordinate-based representation.
Uses Gaussian Fourier features for continuous positional encoding.

Key features:
- Linear complexity O(N) vs transformer's O(N²)
- Coordinate-based representation with Fourier features
- State space model for long-range dependencies
- Query-based decoding for arbitrary resolutions
- Extremely fast compared to attention-based models

Note: Requires mamba_ssm package. Install with:
    pip install mamba-ssm
"""
import torch
import torch.nn as nn
import math

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed. MambaCoordinateFM will not work.")
    print("Install with: pip install mamba-ssm")


class GaussianFourierFeatures(nn.Module):
    """
    Gaussian Fourier feature mapping for continuous coordinates.

    Maps (x, y) ∈ [0,1]² to high-dimensional features using random Fourier features.
    """
    def __init__(self, input_dim=2, mapping_size=256, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size

        # Random Gaussian matrix B ~ N(0, scale²)
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, coords):
        """
        Args:
            coords: (B, N, 2) coordinates in [0, 1]
        Returns:
            features: (B, N, mapping_size*2) Fourier features
        """
        x_proj = 2 * math.pi * coords @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MambaBlock(nn.Module):
    """Mamba SSM block with normalization and residual."""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm not installed")

        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, dim)
        Returns:
            output: (B, N, dim)
        """
        return x + self.mamba(self.norm(x))


class CrossAttention(nn.Module):
    """Simple cross-attention for query-to-context decoding."""
    def __init__(self, query_dim, context_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        B, N_q, _ = query.shape
        N_c = context.shape[1]

        q = self.to_q(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, N_c, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, N_c, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, -1)
        return self.to_out(out)


class MambaCoordinateFM(nn.Module):
    """
    Mamba SSM for coordinate-based Flow Matching.

    Architecture:
    1. Encode sparse input coordinates with Gaussian Fourier features
    2. Process inputs with Mamba SSM blocks (linear complexity)
    3. Encode query coordinates with Fourier features
    4. Cross-attend from queries to processed inputs
    5. Decode to RGB velocity prediction

    Advantages over transformer:
    - O(N) complexity vs O(N²)
    - Better for long sequences of coordinates
    - Faster training and inference
    """
    def __init__(self, channel=3, dim=512, depth=6, d_state=16, d_conv=4,
                 expand=2, fourier_mapping_size=256, fourier_scale=10.0,
                 num_heads=8, dropout=0.1):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm not installed. Install with: pip install mamba-ssm"
            )

        self.channel = channel
        self.dim = dim

        # Gaussian Fourier features for coordinates
        self.coord_encoder = GaussianFourierFeatures(
            input_dim=2,
            mapping_size=fourier_mapping_size,
            scale=fourier_scale
        )
        coord_feat_dim = fourier_mapping_size * 2

        # Time embedding (sinusoidal)
        time_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.register_buffer('time_freqs', torch.exp(
            torch.linspace(0, math.log(10000), time_dim // 2)
        ))

        # Input encoder: coordinates + RGB values → dim
        self.input_encoder = nn.Sequential(
            nn.Linear(coord_feat_dim + channel, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Mamba blocks for processing inputs
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(dim, d_state, d_conv, expand)
            for _ in range(depth)
        ])

        # Query encoder: just coordinates (no values yet)
        self.query_encoder = nn.Sequential(
            nn.Linear(coord_feat_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Cross-attention from queries to processed inputs
        self.decoder_attn = CrossAttention(dim, dim, num_heads, dropout)
        self.decoder_norm = nn.LayerNorm(dim)

        # Output: velocity prediction
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, channel)
        )

    def time_embedding(self, t):
        """
        Args:
            t: (B,) time values in [0, 1]
        Returns:
            emb: (B, dim)
        """
        freqs = self.time_freqs.unsqueeze(0) * t.unsqueeze(-1)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return self.time_mlp(emb)

    def forward(self, query_coords, t, input_coords, input_values):
        """
        Args:
            query_coords: (B, N_query, 2) - coordinates where we want predictions
            t: (B,) - time in [0, 1]
            input_coords: (B, N_input, 2) - sparse observation coordinates
            input_values: (B, N_input, 3) - RGB values at input coordinates
        Returns:
            v_pred: (B, N_query, 3) - predicted velocity at query coordinates
        """
        B = query_coords.shape[0]

        # Time embedding
        t_emb = self.time_embedding(t)  # (B, dim)

        # Encode input coordinates with Gaussian Fourier features
        input_coord_feats = self.coord_encoder(input_coords)  # (B, N_input, coord_feat_dim)

        # Combine coordinate features with RGB values
        input_feats = torch.cat([input_coord_feats, input_values], dim=-1)
        input_encoded = self.input_encoder(input_feats)  # (B, N_input, dim)

        # Add time conditioning
        input_encoded = input_encoded + t_emb.unsqueeze(1)

        # Process with Mamba blocks (linear complexity!)
        for block in self.mamba_blocks:
            input_encoded = block(input_encoded)

        # Encode query coordinates
        query_coord_feats = self.coord_encoder(query_coords)  # (B, N_query, coord_feat_dim)
        query_encoded = self.query_encoder(query_coord_feats)  # (B, N_query, dim)

        # Add time conditioning to queries
        query_encoded = query_encoded + t_emb.unsqueeze(1)

        # Decode: cross-attend from queries to processed inputs
        decoded = self.decoder_attn(self.decoder_norm(query_encoded), input_encoded)
        decoded = decoded + query_encoded  # Residual

        # Project to RGB velocity
        v_pred = self.output_proj(decoded)  # (B, N_query, 3)

        return v_pred
