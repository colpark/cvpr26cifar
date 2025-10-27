"""
SCENT / PerceiverIO for Navier–Stokes next‑step forecasting (inference only)
--------------------------------------------------------------------------
Self‑contained module to load a trained SCENT (PerceiverIO) checkpoint and run
inference + quick visualizations. No external project imports required.

Expected checkpoint layout (as saved by your training script):
NavierStokesCheckpoints/SCENT/<dataset_tag>/<sparsity_tag>/best.pt
  - keys: {
      'model_state_dict', 'fourier_encoder_state_dict', 'pos_queries',
      'config': {
          'H','W','SPARSITY','POS_EMBED_DIM','INPUT_DIM','QUERIES_DIM',
          'LOGITS_DIM','LR','BATCH_SIZE'
      },
      ...
    }

Example
-------
from models.scent import SCENTNavierStokesModel, path_for
ckpt = path_for("NavierStokesCheckpoints/SCENT", dataset_tag="full", sparsity_pct=2)
model = SCENTNavierStokesModel(ckpt, device="cuda:0")
# x_sparse: [B,1,H,W] (normalized), mask: [B,H,W] (bool)
pred = model.predict(x_sparse, mask)  # [B,1,H,W]

Notes
-----
* Predictions are in the same (normalized) space as training unless you
  unnormalize externally using your mean/std.
* For CRPS with deterministic model, replicate predictions (K copies).
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# -----------------------
# Basic helpers
# -----------------------

def exists(x):
    return x is not None


def create_coordinate_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Return grid of shape [(h*w), 2] with coords in [-1, 1]."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    return rearrange(torch.stack([yy, xx], dim=-1), "h w c -> (h w) c")


# -----------------------
# Fourier features (positional encoder)
# -----------------------
class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_bands: int = 16):
        super().__init__()
        fourier_dim = in_features * 2 * n_bands
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )
        self.register_buffer("freqs", 2 ** torch.arange(n_bands) * torch.pi)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: [B, N, d] -> returns [B, N, out_features]."""
        b, n, d = coords.shape
        proj = coords.unsqueeze(-1) * self.freqs  # [B,N,d,bands]
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        feats = rearrange(feats, "b n d bands -> b n (d bands)")
        return self.mlp(feats)


# -----------------------
# PerceiverIO (minimal, matches training script)
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
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


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
        inner_dim = dim_head * heads
        context_dim = context_dim if exists(context_dim) else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = context if exists(context) else x
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


def cache_fn(f):
    cache = None

    def cached_fn(*args, _cache=True, **kwargs):
        nonlocal cache
        if not _cache:
            return f(*args, **kwargs)
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim=None,
        num_latents=256,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        decoder_ff=True,
    ):
        super().__init__()
        self.register_buffer("latents", torch.randn(num_latents, latent_dim))

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head
                    ),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )

        get_latent_attn = lambda: PreNorm(
            latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head)
        )
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([get_latent_attn(**cache_args), get_latent_ff(**cache_args)])
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=latent_dim,
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(self, data, mask=None, queries=None):
        b, *_ = data.shape
        x = repeat(self.latents, "n d -> b n d", b=b)
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        if not exists(queries):
            return x
        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=b)
        latents = self.decoder_cross_attn(queries, context=x)
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        return self.to_logits(latents)


# -----------------------
# Inference adapter
# -----------------------
@dataclass
class SCENTConfig:
    H: int
    W: int
    POS_EMBED_DIM: int
    INPUT_DIM: int
    QUERIES_DIM: int
    LOGITS_DIM: int
    # optional, not always saved
    SPARSITY: Optional[float] = None


class SCENTNavierStokesModel:
    """Convenience wrapper around PerceiverIO + FourierFeatures for NS inference.

    API:
        m = SCENTNavierStokesModel(ckpt_path)
        pred = m.predict(x_sparse, mask)  # [B,1,H,W]
    """

    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = torch.load(ckpt_path, map_location=self.device)
        sd: Dict[str, Any] = ckpt["model_state_dict"]
        fe_sd: Optional[Dict[str, Any]] = ckpt.get("fourier_encoder_state_dict")
        cfg_raw: Dict[str, Any] = ckpt.get("config", {})

        # Infer config (robust to partial configs)
        pos_dim = int(cfg_raw.get("QUERIES_DIM", cfg_raw.get("POS_EMBED_DIM", 96)))
        in_dim = int(cfg_raw.get("INPUT_DIM", pos_dim + 1))
        logits_dim = int(cfg_raw.get("LOGITS_DIM", 1))
        H = int(cfg_raw.get("H", 64))
        W = int(cfg_raw.get("W", 64))
        self.cfg = SCENTConfig(
            H=H,
            W=W,
            POS_EMBED_DIM=int(cfg_raw.get("POS_EMBED_DIM", pos_dim)),
            INPUT_DIM=in_dim,
            QUERIES_DIM=pos_dim,
            LOGITS_DIM=logits_dim,
            SPARSITY=cfg_raw.get("SPARSITY"),
        )

        # Infer structural args from state_dict
        num_latents, latent_dim = sd["latents"].shape
        depth = self._infer_depth(sd)
        # Heads & dim_head match your training defaults
        model = PerceiverIO(
            depth=depth,
            dim=self.cfg.INPUT_DIM,
            queries_dim=self.cfg.QUERIES_DIM,
            logits_dim=self.cfg.LOGITS_DIM,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            decoder_ff=True,
        ).to(self.device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        self.model = model

        # Fourier encoder + fixed grid queries
        self.fourier = FourierFeatures(2, self.cfg.POS_EMBED_DIM).to(self.device)
        if fe_sd is not None:
            self.fourier.load_state_dict(fe_sd, strict=True)
        self.fourier.eval()

        self.coords = create_coordinate_grid(self.cfg.H, self.cfg.W, self.device)
        if "pos_queries" in ckpt:
            pq = ckpt["pos_queries"].to(self.device)
            if pq.ndim == 2 and pq.shape[0] == self.cfg.H * self.cfg.W:
                self.pos_queries = pq
            else:
                # Fallback: recompute
                self.pos_queries = self._encode_positions()
        else:
            self.pos_queries = self._encode_positions()

    # -------- utils --------
    @staticmethod
    def _infer_depth(sd: Dict[str, Any]) -> int:
        """Count distinct indices in keys like 'layers.<i>.'"""
        layers = set()
        prefix = "layers."
        for k in sd.keys():
            if k.startswith(prefix):
                try:
                    i = int(k.split(".")[1])
                    layers.add(i)
                except Exception:
                    pass
        return (max(layers) + 1) if layers else 6

    def _encode_positions(self) -> torch.Tensor:
        with torch.no_grad():
            pos = self.fourier(self.coords.unsqueeze(0)).squeeze(0)  # [N, Dpos]
        pos.requires_grad_(False)
        return pos

    def _prepare_inputs(self, x_sparse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_sparse: [B,1,H,W] -> (inp, queries)
        inp: [B, N, 1+Dpos], queries: [N, Dpos] (broadcasted in model)
        """
        assert x_sparse.dim() == 4 and x_sparse.shape[1] == 1
        b, _, h, w = x_sparse.shape
        assert h == self.cfg.H and w == self.cfg.W, "Input HW mismatch"
        pixels = rearrange(x_sparse, "b c h w -> b (h w) c")  # [B,N,1]
        pos = repeat(self.coords, "n d -> b n d", b=b)
        pos_emb = self.fourier(pos)  # [B,N,Dpos]
        inp = torch.cat([pixels, pos_emb], dim=-1)  # [B,N,1+Dpos]
        return inp, self.pos_queries

    # -------- public API --------
    @torch.no_grad()
    def predict(
        self,
        x_sparse: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ensemble_K: int = 1,
    ) -> torch.Tensor:
        """Run a forward pass and return [B,1,H,W].

        Args:
            x_sparse: normalized sparse input [B,1,H,W].
            mask: optional visibility mask [B,H,W] (bool or 0/1).
            ensemble_K: if >1, returns the mean over K identical copies
                        (deterministic model). For CRPS, use `predict_K`.
        """
        self.model.eval(); self.fourier.eval()
        x_sparse = x_sparse.to(self.device)
        q = self.pos_queries  # [N,Dpos]
        inp, _ = self._prepare_inputs(x_sparse)
        if exists(mask):
            mask = mask.to(self.device)
        pred = self.model(inp, mask=mask, queries=q)  # [B,N,1]
        pred = rearrange(pred, "b (h w) c -> b c h w", h=self.cfg.H, w=self.cfg.W)
        if ensemble_K <= 1:
            return pred
        else:
            return pred  # identical copies would average to itself

    @torch.no_grad()
    def predict_K(
        self,
        x_sparse: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        K: int = 10,
    ) -> torch.Tensor:
        """Return K identical predictions for CRPS helper: [K,B,1,H,W]."""
        p = self.predict(x_sparse, mask=mask, ensemble_K=1)
        return repeat(p, "b c h w -> k b c h w", k=K)


# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def crps_from_ensemble(preds_K: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute CRPS given ensemble preds.

    Args:
        preds_K: [K,B,C,H,W]
        target:  [B,C,H,W]
    Returns:
        scalar tensor (mean over batch/spatial/channel)
    """
    K = preds_K.shape[0]
    preds = rearrange(preds_K, "k b c h w -> k b (h w) c")
    tgt = rearrange(target, "b c h w -> b (h w) c")
    term1 = (preds - tgt.unsqueeze(0)).abs().mean(dim=0)  # [B,N,C]
    diffs = (preds.unsqueeze(0) - preds.unsqueeze(1)).abs()  # [K,K,B,N,C]
    term2 = 0.5 * diffs.mean(dim=(0, 1))  # [B,N,C]
    return (term1 - term2).mean()


# -----------------------
# Paths / discovery helpers
# -----------------------

def sparsity_tag(sparsity_pct: int) -> str:
    assert sparsity_pct in {2, 5, 20}
    return f"{sparsity_pct}pct"


def path_for(root: str, dataset_tag: str, sparsity_pct: int) -> str:
    """Build the standard checkpoint path.

    Args:
        root: e.g., "NavierStokesCheckpoints/SCENT"
        dataset_tag: "full" or "10pct"
        sparsity_pct: 2, 5, or 20
    """
    sp = sparsity_tag(sparsity_pct)
    return os.path.join(root, dataset_tag, sp, "best.pt")


# -----------------------
# Quick viz utility (optional)
# -----------------------
@torch.no_grad()
def quick_panel(x_sparse: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, title: str = ""):
    """Matplotlib triple panel for sanity checks.
    All tensors shaped [1,1,H,W] or [B,1,H,W] (uses first element).
    """
    import matplotlib.pyplot as plt

    xs = x_sparse[0, 0].detach().cpu().numpy()
    yt = y[0, 0].detach().cpu().numpy()
    pr = pred[0, 0].detach().cpu().numpy()

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(xs, cmap="viridis")
    plt.title("x_t (sparse)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(yt, cmap="viridis")
    plt.title("y_{t+1} (target)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pr, cmap="viridis")
    plt.title("prediction")
    plt.axis("off")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()
