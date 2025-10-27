"""
OmniField / Cascaded PerceiverIO for Navier–Stokes (inference only)
-------------------------------------------------------------------
Self‑contained module to load an OmniField checkpoint and run inference
on sparse‑masked inputs. Mirrors your training architecture:
  - Cascaded encoder blocks with masked cross‑attention
  - Gaussian Fourier Features (GFF) positional encoding
  - Queries are fixed positional embeddings (saved in ckpt as `pos_queries`)

Expected checkpoint keys (as saved in your script):
  'model_state_dict', 'fourier_state_dict', 'pos_queries',
  'config': { 'H','W','SPARSITY','POS_EMBED_DIM','INPUT_DIM','QUERIES_DIM','LOGITS_DIM', ... }

Usage
-----
from models.omnifield import OmniFieldNavierStokesModel, path_for
ckpt = path_for("NavierStokesCheckpoints/OMNIFIELD", dataset_tag="full", sparsity_pct=5)
m = OmniFieldNavierStokesModel(ckpt, device="cuda:0")
# x_sparse: [B,1,H,W] (normalized), mask: [B,H,W] (bool)
pred = m.predict(x_sparse, mask)  # [B,1,H,W]
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# -----------------------
# Helpers
# -----------------------

def exists(x):
    return x is not None


def create_coordinate_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    return rearrange(torch.stack([yy, xx], dim=-1), "h w c -> (h w) c")


# -----------------------
# Gaussian Fourier Features (GFF)
# -----------------------
class GaussianFourierFeatures(nn.Module):
    def __init__(self, in_features: int, mapping_size: int, scale: float = 5.0):
        super().__init__()
        self.register_buffer("B", torch.randn((in_features, mapping_size), dtype=torch.float32) * scale)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords [..., 2]
        proj = coords @ self.B  # [..., M]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [..., 2M]


# -----------------------
# Core blocks (match training shapes/semantics)
# -----------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context) and "context" in kwargs:
            kwargs.update(context=self.norm_context(kwargs["context"]))
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, g = x.chunk(2, dim=-1)
        return x * F.gelu(g)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner = heads * dim_head
        context_dim = context_dim if exists(context_dim) else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = context if exists(context) else x
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            # mask expected as [B, N] (bool)
            mask = mask.to(torch.bool)
            mask = repeat(mask, "b n -> (b h) n", h=h)
            max_neg = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask[:, None, :], max_neg)

        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


def get_sinusoidal_embeddings(n: int, d: int) -> torch.Tensor:
    assert d % 2 == 0
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * -(torch.log(torch.tensor(10000.0)) / d))
    pe = torch.zeros(n, d, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CascadedBlock(nn.Module):
    def __init__(self, dim, n_latents, input_dim, cross_heads, cross_dim_head, self_heads, self_dim_head, residual_dim=None):
        super().__init__()
        self.latents = nn.Parameter(get_sinusoidal_embeddings(n_latents, dim), requires_grad=False)
        self.cross_attn = PreNorm(dim, Attention(dim, input_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=input_dim)
        self.self_attn = PreNorm(dim, Attention(dim, heads=self_heads, dim_head=self_dim_head))
        self.residual_proj = nn.Linear(residual_dim, dim) if (residual_dim is not None and residual_dim != dim) else None
        self.ff = PreNorm(dim, FeedForward(dim))

    def forward(self, x, context, mask=None, residual=None):
        b = context.size(0)
        latents = repeat(self.latents, "n d -> b n d", b=b)
        latents = self.cross_attn(latents, context=context, mask=mask) + latents
        if residual is not None:
            latents = latents + (self.residual_proj(residual) if self.residual_proj is not None else residual)
        latents = self.self_attn(latents) + latents
        latents = self.ff(latents) + latents
        return latents


class CascadedPerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        queries_dim,
        logits_dim=None,
        latent_dims=(256, 256, 256),
        num_latents=(512, 512, 512),
        cross_heads=1,
        cross_dim_head=64,
        self_heads=8,
        self_dim_head=64,
        decoder_ff=True,
        self_layers=4,
    ):
        super().__init__()
        assert len(latent_dims) == len(num_latents)
        self.encoder_blocks = nn.ModuleList()
        prev_dim = None
        for dim, n_lat in zip(latent_dims, num_latents):
            self.encoder_blocks.append(
                CascadedBlock(
                    dim=dim,
                    n_latents=n_lat,
                    input_dim=input_dim,
                    cross_heads=cross_heads,
                    cross_dim_head=cross_dim_head,
                    self_heads=self_heads,
                    self_dim_head=self_dim_head,
                    residual_dim=prev_dim,
                )
            )
            prev_dim = dim
        final_latent_dim = latent_dims[-1]

        self.self_attn_blocks = nn.Sequential(
            *[
                nn.Sequential(
                    PreNorm(final_latent_dim, Attention(final_latent_dim, heads=self_heads, dim_head=self_dim_head)),
                    PreNorm(final_latent_dim, FeedForward(final_latent_dim)),
                )
                for _ in range(self_layers)
            ]
        )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, final_latent_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=final_latent_dim,
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(self, data, mask=None, queries=None):
        b = data.size(0)
        residual = None
        for block in self.encoder_blocks:
            residual = block(x=residual, context=data, mask=mask, residual=residual)
        for sa_block in self.self_attn_blocks:
            residual = sa_block[0](residual) + residual
            residual = sa_block[1](residual) + residual
        if queries is None:
            return residual
        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=b)
        x = self.decoder_cross_attn(queries, context=residual)
        x = x + queries
        if self.decoder_ff:
            x = x + self.decoder_ff(x)
        return self.to_logits(x)


# -----------------------
# Config & wrapper
# -----------------------
@dataclass
class OmniConfig:
    H: int
    W: int
    POS_EMBED_DIM: int
    INPUT_DIM: int
    QUERIES_DIM: int
    LOGITS_DIM: int
    SPARSITY: Optional[float] = None


class OmniFieldNavierStokesModel:
    """Inference wrapper for OmniField.

    API:
        m = OmniFieldNavierStokesModel(ckpt)
        pred = m.predict(x_sparse, mask)  # [B,1,H,W]
    """

    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = torch.load(ckpt_path, map_location=self.device)
        sd: Dict[str, Any] = ckpt["model_state_dict"]
        fe_sd: Optional[Dict[str, Any]] = ckpt.get("fourier_state_dict") or ckpt.get("fourier_encoder_state_dict")
        cfg_raw: Dict[str, Any] = ckpt.get("config", {})

        # Basic dims
        H = int(cfg_raw.get("H", 64))
        W = int(cfg_raw.get("W", 64))
        pos_dim = int(cfg_raw.get("QUERIES_DIM", cfg_raw.get("POS_EMBED_DIM", 192)))
        in_dim = int(cfg_raw.get("INPUT_DIM", pos_dim + 1))
        logits_dim = int(cfg_raw.get("LOGITS_DIM", 1))
        self.cfg = OmniConfig(
            H=H,
            W=W,
            POS_EMBED_DIM=int(cfg_raw.get("POS_EMBED_DIM", pos_dim)),
            INPUT_DIM=in_dim,
            QUERIES_DIM=pos_dim,
            LOGITS_DIM=logits_dim,
            SPARSITY=cfg_raw.get("SPARSITY"),
        )

        # Infer encoder shapes from state_dict
        latent_dims, num_latents = self._infer_cascade_shapes(sd)
        self_layers = self._infer_self_layers(sd)

        model = CascadedPerceiverIO(
            input_dim=self.cfg.INPUT_DIM,
            queries_dim=self.cfg.QUERIES_DIM,
            logits_dim=self.cfg.LOGITS_DIM,
            latent_dims=tuple(latent_dims) if latent_dims else (256, 256, 256),
            num_latents=tuple(num_latents) if num_latents else (512, 512, 512),
            cross_heads=1,
            cross_dim_head=64,
            self_heads=8,
            self_dim_head=64,
            decoder_ff=True,
            self_layers=self_layers if self_layers is not None else 4,
        ).to(self.device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        self.model = model

        # Fourier encoder (Gaussian) + coords
        mapping_size = self.cfg.POS_EMBED_DIM // 2
        self.fourier = GaussianFourierFeatures(2, mapping_size, scale=5.0).to(self.device)
        if fe_sd is not None:
            self.fourier.load_state_dict(fe_sd, strict=True)
        self.fourier.eval()

        self.coords = create_coordinate_grid(self.cfg.H, self.cfg.W, self.device)
        if "pos_queries" in ckpt:
            pq = ckpt["pos_queries"].to(self.device)
            self.pos_queries = pq if pq.ndim == 2 else pq.squeeze(0)
        else:
            self.pos_queries = self._encode_positions()

    # ---- inference utils ----
    def _encode_positions(self) -> torch.Tensor:
        with torch.no_grad():
            pos = self.fourier(self.coords)  # [N, Dpos]
        pos.requires_grad_(False)
        return pos

    @staticmethod
    def _infer_cascade_shapes(sd: Dict[str, Any]):
        idxs = []
        for k, v in sd.items():
            if k.startswith("encoder_blocks.") and k.endswith(".latents"):
                try:
                    i = int(k.split(".")[1])
                except Exception:
                    continue
                n, d = v.shape
                idxs.append((i, n, d))
        if not idxs:
            return None, None
        idxs.sort()
        latent_dims = [d for _, _, d in idxs]
        num_latents = [n for _, n, _ in idxs]
        return latent_dims, num_latents

    @staticmethod
    def _infer_self_layers(sd: Dict[str, Any]) -> Optional[int]:
        idxs = set()
        prefix = "self_attn_blocks."
        for k in sd.keys():
            if k.startswith(prefix):
                try:
                    i = int(k.split(".")[1])
                    idxs.add(i)
                except Exception:
                    pass
        return (max(idxs) + 1) if idxs else None

    def _prepare_inputs(self, x_sparse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_sparse: [B,1,H,W]
        b, _, h, w = x_sparse.shape
        assert h == self.cfg.H and w == self.cfg.W
        pixels = rearrange(x_sparse, "b c h w -> b (h w) c")  # [B,N,1]
        pos = repeat(self.coords, "n d -> b n d", b=b)
        pos_emb = self.fourier(pos)  # [B,N,Dpos]
        inp = torch.cat([pixels, pos_emb], dim=-1)  # [B,N,1+Dpos]
        return inp, pixels, pos_emb

    # ---- public API ----
    @torch.no_grad()
    def predict(self, x_sparse: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.model.eval(); self.fourier.eval()
        x_sparse = x_sparse.to(self.device)
        inp, _pix, _pos = self._prepare_inputs(x_sparse)

        mask_flat = None
        if exists(mask):
            mask = mask.to(self.device)
            if mask.dim() == 3:
                mask_flat = rearrange(mask, "b h w -> b (h w)").bool()
            elif mask.dim() == 2:
                mask_flat = mask.bool()
            else:
                raise ValueError("mask must be [B,H,W] or [B,N]")

        q = self.pos_queries  # [N,Dpos]
        pred_pix = self.model(inp, mask=mask_flat, queries=q)  # [B,N,1]
        pred = rearrange(pred_pix, "b (h w) c -> b c h w", h=self.cfg.H, w=self.cfg.W)
        return pred

    @torch.no_grad()
    def predict_K(self, x_sparse: torch.Tensor, mask: Optional[torch.Tensor] = None, K: int = 10) -> torch.Tensor:
        p = self.predict(x_sparse, mask)
        return repeat(p, "b c h w -> k b c h w", k=K)


# -----------------------
# Path helper
# -----------------------

def sparsity_tag(sparsity_pct: int) -> str:
    assert sparsity_pct in {2, 5, 20}
    return f"{sparsity_pct}pct"


def path_for(root: str, dataset_tag: str, sparsity_pct: int) -> str:
    return os.path.join(root, dataset_tag, sparsity_tag(sparsity_pct), "best.pt")


# -----------------------
# Quick viz (optional)
# -----------------------
@torch.no_grad()
def quick_panel(x_sparse: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, title: str = ""):
    import matplotlib.pyplot as plt
    xs = x_sparse[0, 0].detach().cpu().numpy()
    yt = y[0, 0].detach().cpu().numpy()
    pr = pred[0, 0].detach().cpu().numpy()
    import matplotlib
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1); plt.imshow(xs, cmap="viridis"); plt.title("x_t (sparse)"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(yt, cmap="viridis"); plt.title("y_{t+1} (target)"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(pr, cmap="viridis"); plt.title("prediction"); plt.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout(); plt.show()



