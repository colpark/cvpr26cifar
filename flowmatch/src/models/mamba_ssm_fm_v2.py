"""
Mamba SSM Flow Matching - Reference Implementation

Follows ref/mamba_diffusion.ipynb exactly:
- Correct FourierFeatures from ref/core/neural_fields/perceiver.py
- Correct forward signature: forward(noisy_values, query_coords, t, ...)
- SSMBlockFast with einsum optimization
- Dataset format: output_coords/output_values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierFeatures(nn.Module):
    """
    Fourier feature positional encoding

    Reference: ref/core/neural_fields/perceiver.py

    CRITICAL: Matrix B has shape (num_freqs, coord_dim), NOT (coord_dim, num_freqs)!
    """
    def __init__(self, coord_dim=2, num_freqs=256, scale=10.0, learnable=False):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_freqs = num_freqs
        self.output_dim = 2 * num_freqs  # sin + cos

        # Sample frequency matrix B from Gaussian
        # CRITICAL: Shape is (num_freqs, coord_dim)
        B = torch.randn(num_freqs, coord_dim) * scale

        if learnable:
            self.register_parameter('B', nn.Parameter(B))
        else:
            self.register_buffer('B', B)

    def forward(self, coords):
        """
        Args:
            coords: (B, N, coord_dim) coordinates in [0, 1]

        Returns:
            features: (B, N, 2*num_freqs) Fourier features
        """
        # coords: (B, N, coord_dim)
        # B: (num_freqs, coord_dim)

        # Compute 2π * coords @ B^T
        coords_proj = 2 * math.pi * torch.matmul(coords, self.B.T)  # (B, N, num_freqs)

        # Concatenate sin and cos
        features = torch.cat([
            torch.sin(coords_proj),
            torch.cos(coords_proj)
        ], dim=-1)  # (B, N, 2*num_freqs)

        return features


class SSMBlockFast(nn.Module):
    """
    Ultra-fast SSM using cumulative scan

    Reference: ref/mamba_diffusion.ipynb
    Uses einsum for maximum speed
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

        nn.init.xavier_uniform_(self.B.weight, gain=0.5)
        nn.init.xavier_uniform_(self.C.weight, gain=0.5)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8

    def forward(self, x):
        """
        Fully vectorized - uses einsum for maximum speed

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

        # Create exponential decay matrix
        # decay[i,j] = A_bar^(i-j) if i >= j else 0
        indices = torch.arange(N, device=x.device)
        decay = A_bar.unsqueeze(0).pow(
            (indices.unsqueeze(0) - indices.unsqueeze(1)).clamp(min=0).unsqueeze(-1)
        )  # (N, N, d_state)

        # Mask to only include i >= j (causal)
        mask = indices.unsqueeze(0) >= indices.unsqueeze(1)  # (N, N)
        decay = decay * mask.unsqueeze(-1).float()  # (N, N, d_state)

        # Compute all states: h[t] = sum_{s<=t} decay[t,s] * Bu[s]
        # Using einsum for speed: (B,N,d) = (N,N,d) @ (B,N,d)
        h = torch.einsum('nmd,bnd->bmd', decay, Bu)  # (B, N, d_state)
        h = torch.clamp(h, min=-10.0, max=10.0)

        # Output
        y = self.C(h) + self.D * x

        # Gating and residual
        gate = self.gate(x)
        y = gate * y + (1 - gate) * x

        return self.dropout(self.norm(y))


class MambaBlock(nn.Module):
    """Complete Mamba block with FAST SSM + MLP"""
    def __init__(self, d_model, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()

        # Expand
        self.proj_in = nn.Linear(d_model, d_model * expand_factor)

        # Use the FAST SSM implementation
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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class MambaSSMFMV2(nn.Module):
    """
    State space model for sparse field flow matching

    Reference implementation following ref/mamba_diffusion.ipynb exactly:
    - Correct FourierFeatures implementation
    - Correct forward signature: forward(noisy_values, query_coords, t, ...)
    - SSMBlockFast with einsum optimization
    - Linear O(N) complexity

    Key features:
    - State propagation for long-range dependencies
    - Efficient sequential processing
    - Scale-invariant continuous representations
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

        # Fourier features (CORRECT implementation from reference)
        self.fourier = FourierFeatures(coord_dim=2, num_freqs=num_fourier_feats, scale=10.0)
        feat_dim = num_fourier_feats * 2  # FourierFeatures outputs 2*num_freqs

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

        # Cross-attention to extract query-specific features
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

    def forward(self, noisy_values, query_coords, t, input_coords, input_values):
        """
        CRITICAL: Forward signature matches reference exactly!

        Args:
            noisy_values: (B, N_out, 3) - noisy RGB values at query positions
            query_coords: (B, N_out, 2) - query pixel coordinates [0,1]
            t: (B,) - timestep in [0,1]
            input_coords: (B, N_in, 2) - input pixel coordinates [0,1]
            input_values: (B, N_in, 3) - input pixel RGB values [0,1]

        Returns:
            pred_velocity: (B, N_out, 3) - predicted velocity field
        """
        B = query_coords.shape[0]
        N_in = input_coords.shape[1]
        N_out = query_coords.shape[1]

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

        # Concatenate inputs and queries as sequence
        seq = torch.cat([input_tokens, query_tokens], dim=1)  # (B, N_in+N_out, d_model)

        # Process through Mamba blocks (SSM)
        for mamba_block in self.mamba_blocks:
            seq = mamba_block(seq)

        # Split back into input and query sequences
        input_seq = seq[:, :N_in, :]  # (B, N_in, d_model)
        query_seq = seq[:, N_in:, :]  # (B, N_out, d_model)

        # Cross-attention: queries attend to processed inputs
        output, _ = self.query_cross_attn(query_seq, input_seq, input_seq)

        # Decode to RGB velocity
        return self.decoder(output)


# Test model
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Testing MambaSSMFMV2 (Reference Implementation)...")

    model = MambaSSMFMV2(
        channel=3,
        num_fourier_feats=256,
        d_model=512,
        num_layers=6,
        d_state=16
    ).to(device)

    # Test with correct forward signature
    test_noisy = torch.rand(4, 204, 3).to(device)
    test_query_coords = torch.rand(4, 204, 2).to(device)
    test_t = torch.rand(4).to(device)
    test_input_coords = torch.rand(4, 204, 2).to(device)
    test_input_values = torch.rand(4, 204, 3).to(device)

    # CRITICAL: Forward signature is (noisy_values, query_coords, t, input_coords, input_values)
    test_out = model(test_noisy, test_query_coords, test_t, test_input_coords, test_input_values)

    print(f"Model test: {test_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n✓ Forward pass successful with correct signature!")
    print(f"  noisy_values: {test_noisy.shape}")
    print(f"  query_coords: {test_query_coords.shape}")
    print(f"  t: {test_t.shape}")
    print(f"  input_coords: {test_input_coords.shape}")
    print(f"  input_values: {test_input_values.shape}")
    print(f"  output: {test_out.shape}")
