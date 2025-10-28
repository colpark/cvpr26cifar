"""
Mamba SSM for coordinate-based Flow Matching

Closely follows ref/mamba_diffusion.ipynb implementation:
- Vectorized SSM blocks with state space dynamics
- Concatenate inputs + queries as sequence
- Process with Mamba SSM blocks
- Cross-attention for query extraction
- Gaussian Fourier features for coordinates
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianFourierFeatures(nn.Module):
    """Gaussian Fourier feature mapping for continuous coordinates."""
    def __init__(self, coord_dim=2, num_freqs=256, scale=10.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_freqs = num_freqs
        # Random Gaussian matrix B ~ N(0, scale²)
        self.register_buffer('B', torch.randn(coord_dim, num_freqs) * scale)

    def forward(self, coords):
        """
        Args:
            coords: (B, N, coord_dim) coordinates in [0, 1]
        Returns:
            features: (B, N, num_freqs*2) Fourier features
        """
        x_proj = 2 * math.pi * coords @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (B,) time values
        Returns:
            emb: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class SSMBlockFast(nn.Module):
    """
    Vectorized State Space Model (following reference implementation)

    Uses einsum for fast parallel computation of state dynamics.
    """
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # State space parameters
        self.A_log = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

        # Initialize with smaller weights
        nn.init.xavier_uniform_(self.B.weight, gain=0.5)
        nn.init.xavier_uniform_(self.C.weight, gain=0.5)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8

    def forward(self, x):
        """
        Vectorized forward pass using einsum

        Args:
            x: (B, N, d_model)
        Returns:
            y: (B, N, d_model)
        """
        B, N, D = x.shape

        # Discretization
        A = -torch.exp(self.A_log).clamp(min=self.eps, max=10.0)
        dt = 1.0 / N
        A_bar = torch.exp(dt * A)
        B_bar = torch.where(
            torch.abs(A) > self.eps,
            (A_bar - 1.0) / (A + self.eps),
            torch.ones_like(A) * dt
        )

        # Input projection (vectorized)
        Bu = self.B(x) * B_bar  # (B, N, d_state)

        # Create exponential decay matrix using einsum
        indices = torch.arange(N, device=x.device)
        decay = A_bar.unsqueeze(0).pow(
            (indices.unsqueeze(0) - indices.unsqueeze(1)).clamp(min=0).unsqueeze(-1)
        )  # (N, N, d_state)

        # Causal mask (only i >= j)
        mask = indices.unsqueeze(0) >= indices.unsqueeze(1)
        decay = decay * mask.unsqueeze(-1).float()

        # Compute all states: h[t] = sum_{s<=t} decay[t,s] * Bu[s]
        h = torch.einsum('nmd,bnd->bmd', decay, Bu)  # (B, N, d_state)
        h = torch.clamp(h, min=-10.0, max=10.0)

        # Output: y = C*h + D*x
        y = self.C(h) + self.D * x

        # Gating and residual
        gate = self.gate(x)
        y = gate * y + (1 - gate) * x

        return self.dropout(self.norm(y))


class MambaBlock(nn.Module):
    """Complete Mamba block: SSM + MLP (following reference)"""
    def __init__(self, d_model, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()

        # Expand
        self.proj_in = nn.Linear(d_model, d_model * expand_factor)

        # SSM (fast version)
        self.ssm = SSMBlockFast(d_model * expand_factor, d_state, dropout)

        # Contract
        self.proj_out = nn.Linear(d_model * expand_factor, d_model)

        # MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # SSM branch
        residual = x
        x = self.proj_in(x)
        x = self.ssm(x)
        x = self.proj_out(x)
        x = x + residual

        # MLP branch
        x = x + self.mlp(x)

        return x


class MambaSSMFM(nn.Module):
    """
    Mamba SSM for coordinate-based Flow Matching

    Following ref/mamba_diffusion.ipynb closely:
    1. Fourier encode inputs and queries
    2. Project to d_model
    3. Add time conditioning
    4. Concatenate as sequence: [inputs, queries]
    5. Process with Mamba SSM blocks
    6. Split back and cross-attend
    7. Decode to RGB
    """
    def __init__(
        self,
        channel=3,
        num_fourier_feats=256,
        d_model=512,
        num_layers=6,
        d_state=16,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.channel = channel

        # Gaussian Fourier features
        self.fourier = GaussianFourierFeatures(
            coord_dim=2,
            num_freqs=num_fourier_feats,
            scale=10.0
        )
        feat_dim = num_fourier_feats * 2

        # Project inputs and queries
        self.input_proj = nn.Linear(feat_dim + channel, d_model)
        self.query_proj = nn.Linear(feat_dim + channel, d_model)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Mamba blocks for sequence processing
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, expand_factor=2, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Cross-attention: queries attend to processed inputs
        self.query_cross_attn = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=dropout, batch_first=True
        )

        # Output decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, channel)
        )

    def forward(self, query_coords, t, input_coords, input_values, noisy_values=None):
        """
        Args:
            query_coords: (B, N_query, 2) - coordinates where we want predictions
            t: (B,) - time in [0, 1]
            input_coords: (B, N_input, 2) - sparse observation coordinates
            input_values: (B, N_input, 3) - RGB values at input coordinates
            noisy_values: (B, N_query, 3) - noisy RGB at query coords (for flow matching training)
        Returns:
            v_pred: (B, N_query, 3) - predicted velocity at query coordinates
        """
        B = query_coords.shape[0]
        N_in = input_coords.shape[1]
        N_out = query_coords.shape[1]

        # Use provided noisy values, or create dummy ones for inference
        if noisy_values is None:
            noisy_values = torch.randn(B, N_out, self.channel, device=query_coords.device)

        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))  # (B, d_model)

        # Fourier features
        input_feats = self.fourier(input_coords)  # (B, N_in, feat_dim)
        query_feats = self.fourier(query_coords)  # (B, N_out, feat_dim)

        # Encode inputs and queries
        input_tokens = self.input_proj(
            torch.cat([input_feats, input_values], dim=-1)
        )  # (B, N_in, d_model)

        query_tokens = self.query_proj(
            torch.cat([query_feats, noisy_values], dim=-1)
        )  # (B, N_out, d_model)

        # Add time embedding
        input_tokens = input_tokens + t_emb.unsqueeze(1)
        query_tokens = query_tokens + t_emb.unsqueeze(1)

        # Concatenate as sequence: [inputs, queries]
        seq = torch.cat([input_tokens, query_tokens], dim=1)  # (B, N_in+N_out, d_model)

        # Process through Mamba blocks (SSM)
        for mamba_block in self.mamba_blocks:
            seq = mamba_block(seq)

        # Split back
        input_seq = seq[:, :N_in, :]  # (B, N_in, d_model)
        query_seq = seq[:, N_in:, :]  # (B, N_out, d_model)

        # Cross-attention: queries attend to processed inputs
        output, _ = self.query_cross_attn(query_seq, input_seq, input_seq)

        # Decode to RGB velocity
        return self.decoder(output)
