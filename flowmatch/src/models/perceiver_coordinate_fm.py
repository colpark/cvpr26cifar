"""
Perceiver IO with Coordinate-Based Flow Matching

Combines Perceiver's efficient latent bottleneck with coordinate-based representation.
Uses Gaussian Fourier features for continuous positional encoding.

Key features:
- Latent compression for efficiency (256 latents vs 1024 inputs)
- Coordinate-based representation with Fourier features
- Query-based decoding for arbitrary resolutions
- Much faster than pure transformer on coordinates
"""
import torch
import torch.nn as nn
import math


class GaussianFourierFeatures(nn.Module):
    """
    Gaussian Fourier feature mapping for continuous coordinates.

    Maps (x, y) ∈ [0,1]² to high-dimensional features using random Fourier features.
    More efficient than deterministic Fourier features for high dimensions.
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
        # coords: (B, N, 2)
        # B: (2, mapping_size)
        # coords @ B: (B, N, mapping_size)

        x_proj = 2 * math.pi * coords @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PerceiverAttention(nn.Module):
    """Multi-head cross-attention for Perceiver."""
    def __init__(self, query_dim, context_dim, num_heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        """
        Args:
            query: (B, N_q, query_dim)
            context: (B, N_c, context_dim)
        Returns:
            output: (B, N_q, query_dim)
        """
        B, N_q, _ = query.shape
        N_c = context.shape[1]

        # Project to Q, K, V
        q = self.to_q(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, N_c, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, N_c, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Combine
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, -1)
        return self.to_out(out)


class PerceiverBlock(nn.Module):
    """Perceiver processing block with cross-attention and feedforward."""
    def __init__(self, latent_dim, context_dim, num_heads=8, head_dim=64,
                 mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.cross_attn = PerceiverAttention(latent_dim, context_dim, num_heads, head_dim, dropout)
        self.norm1 = nn.LayerNorm(latent_dim)

        self.self_attn = PerceiverAttention(latent_dim, latent_dim, num_heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(latent_dim)

        mlp_hidden = latent_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, latent_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(latent_dim)

    def forward(self, latents, context):
        """
        Args:
            latents: (B, N_latent, latent_dim)
            context: (B, N_context, context_dim)
        """
        # Cross-attention to context
        latents = latents + self.cross_attn(self.norm1(latents), context)

        # Self-attention among latents
        latents = latents + self.self_attn(self.norm2(latents), self.norm2(latents))

        # MLP
        latents = latents + self.mlp(self.norm3(latents))

        return latents


class PerceiverCoordinateFM(nn.Module):
    """
    Perceiver IO for coordinate-based Flow Matching.

    Architecture:
    1. Encode sparse input coordinates with Gaussian Fourier features
    2. Cross-attend from latent array to input coordinates
    3. Process latents with self-attention
    4. Cross-attend from query coordinates to latents
    5. Decode to RGB velocity prediction
    """
    def __init__(self, channel=3, latent_dim=512, num_latents=256, depth=6,
                 num_heads=8, head_dim=64, fourier_mapping_size=256,
                 fourier_scale=10.0, dropout=0.1):
        super().__init__()
        self.channel = channel
        self.latent_dim = latent_dim
        self.num_latents = num_latents

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
            nn.Linear(time_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.register_buffer('time_freqs', torch.exp(
            torch.linspace(0, math.log(10000), time_dim // 2)
        ))

        # Input encoder: coordinates + RGB values
        self.input_encoder = nn.Sequential(
            nn.Linear(coord_feat_dim + channel, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Perceiver blocks (encode inputs to latents)
        self.encoder_blocks = nn.ModuleList([
            PerceiverBlock(latent_dim, latent_dim, num_heads, head_dim, 4, dropout)
            for _ in range(depth)
        ])

        # Query encoder: just coordinates (no values yet)
        self.query_encoder = nn.Sequential(
            nn.Linear(coord_feat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Decoder: cross-attend from queries to latents
        self.decoder_attn = PerceiverAttention(latent_dim, latent_dim, num_heads, head_dim, dropout)
        self.decoder_norm = nn.LayerNorm(latent_dim)

        # Output: velocity prediction
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, channel)
        )

    def time_embedding(self, t):
        """
        Args:
            t: (B,) time values in [0, 1]
        Returns:
            emb: (B, latent_dim)
        """
        # t: (B,)
        freqs = self.time_freqs.unsqueeze(0) * t.unsqueeze(-1)  # (B, time_dim//2)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # (B, time_dim)
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
        t_emb = self.time_embedding(t)  # (B, latent_dim)

        # Encode input coordinates with Gaussian Fourier features
        input_coord_feats = self.coord_encoder(input_coords)  # (B, N_input, coord_feat_dim)

        # Combine coordinate features with RGB values
        input_feats = torch.cat([input_coord_feats, input_values], dim=-1)
        input_encoded = self.input_encoder(input_feats)  # (B, N_input, latent_dim)

        # Add time conditioning to inputs
        input_encoded = input_encoded + t_emb.unsqueeze(1)

        # Initialize latents (broadcast to batch)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, num_latents, latent_dim)

        # Process: cross-attend from latents to inputs, then self-attend
        for block in self.encoder_blocks:
            latents = block(latents, input_encoded)

        # Encode query coordinates
        query_coord_feats = self.coord_encoder(query_coords)  # (B, N_query, coord_feat_dim)
        query_encoded = self.query_encoder(query_coord_feats)  # (B, N_query, latent_dim)

        # Add time conditioning to queries
        query_encoded = query_encoded + t_emb.unsqueeze(1)

        # Decode: cross-attend from queries to latents
        decoded = self.decoder_attn(self.decoder_norm(query_encoded), latents)
        decoded = decoded + query_encoded  # Residual

        # Project to RGB velocity
        v_pred = self.output_proj(decoded)  # (B, N_query, 3)

        return v_pred
