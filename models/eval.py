"""
Evaluation utilities for Navier–Stokes models on the fixed validation set
-------------------------------------------------------------------------
- Works with any model exposing:
    predict(x_sparse: [B,1,H,W], mask: [B,H,W]|None) -> [B,1,H,W]
    predict_K(x_sparse, mask, K: int) -> [K,B,1,H,W]   (optional)
- Computes normalized metrics (MSE, RMSE, MAE, CRPS) and unnormalized versions
  (RMSE_u, MAE_u, CRPS_u) given a provided STD.

Typical use
-----------
from models.ns_data import open_memmap, build_val_pairs_for, make_ns_val_loader
from models.scent import SCENTNavierStokesModel, path_for
from models.eval import evaluate_loader, pretty_print_metrics

BASE_DIR = "/pscratch/sd/d/dpark1/NSData"; NAME = "100k"
SPARSITY = 0.02; MEAN, STD = 0.0, 2.4036
DATASET_TAG = "full"  # or "10pct"

mem = open_memmap(BASE_DIR, NAME)
val_pairs, T = build_val_pairs_for(mem, DATASET_TAG)
val_loader, meta = make_ns_val_loader(mem, val_pairs, sparsity=SPARSITY, mean=MEAN, std=STD, batch_size=32)

ckpt = path_for("NavierStokesCheckpoints/SCENT", dataset_tag=DATASET_TAG, sparsity_pct=2)
model = SCENTNavierStokesModel(ckpt, device="cuda:0")
metrics = evaluate_loader(model, val_loader, crps_k=10, std=STD)
pretty_print_metrics(metrics)
"""
from __future__ import annotations

from typing import Dict, Optional, Iterable

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


# -----------------------
# CRPS
# -----------------------
@torch.no_grad()
def crps_from_ensemble(preds_K: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CRPS for ensemble predictions.

    Args:
      preds_K: [K,B,1,H,W] (or [K,B,C,H,W])
      target : [B,1,H,W]   (or [B,C,H,W])
    Returns:
      scalar tensor
    """
    if preds_K.dim() == 5:
        K, B, C, H, W = preds_K.shape
    else:
        raise ValueError("preds_K must be [K,B,C,H,W]")

    preds = rearrange(preds_K, "k b c h w -> k b (h w) c")
    tgt = rearrange(target, "b c h w -> b (h w) c")
    term1 = (preds - tgt.unsqueeze(0)).abs().mean(dim=0)  # [B,N,C]
    diffs = (preds.unsqueeze(0) - preds.unsqueeze(1)).abs()  # [K,K,B,N,C]
    term2 = 0.5 * diffs.mean(dim=(0, 1))  # [B,N,C]
    return (term1 - term2).mean()


# -----------------------
# Core evaluation
# -----------------------
@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    *,
    crps_k: int = 10,
    std: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate a model on the provided DataLoader.

    The loader must yield (x_sparse, y, mask, sample_id).
    If the model exposes predict_K, it is used for CRPS; otherwise, we
    replicate the deterministic prediction K times.
    """
    mse_sum = 0.0
    mae_sum = 0.0
    crps_sum = 0.0
    n_batches = 0

    use_predict_K = hasattr(model, "predict_K") and callable(getattr(model, "predict_K"))

    for x_sparse, y, mask, sid in loader:
        # Forward; model handles device placement for inputs
        pred = model.predict(x_sparse, mask)  # [B,1,H,W]
        device = pred.device
        y = y.to(device)

        # normalized-space metrics per batch (mean reduction)
        mse_b = F.mse_loss(pred, y, reduction="mean").item()
        mae_b = F.l1_loss(pred, y, reduction="mean").item()

        if use_predict_K:
            preds_K = model.predict_K(x_sparse, mask, K=crps_k)
        else:
            preds_K = repeat(pred, "b c h w -> k b c h w", k=crps_k)

        # ensure target on same device as preds_K
        y_k = y  # already on device
        crps_b = crps_from_ensemble(preds_K, y_k).item()

        mse_sum += mse_b
        mae_sum += mae_b
        crps_sum += crps_b
        n_batches += 1

    # averages over batches (matches your earlier script semantics)
    mse = mse_sum / max(1, n_batches)
    rmse = mse ** 0.5
    mae = mae_sum / max(1, n_batches)
    crps = crps_sum / max(1, n_batches)

    out = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "CRPS": crps,
    }

    if std is not None:
        out.update(
            {
                "RMSE_u": rmse * float(std),
                "MAE_u": mae * float(std),
                "CRPS_u": crps * float(std),
            }
        )
    return out


# -----------------------
# Convenience: SCENT one-liner
# -----------------------
@torch.no_grad()
def evaluate_scent_checkpoint(ckpt_path: str, loader, *, crps_k: int = 10, std: Optional[float] = None):
    from models.scent import SCENTNavierStokesModel

    model = SCENTNavierStokesModel(ckpt_path)
    return evaluate_loader(model, loader, crps_k=crps_k, std=std)


# -----------------------
# Pretty print
# -----------------------

def pretty_print_metrics(metrics: Dict[str, float], prefix: str = ""):
    keys = ["MSE", "RMSE", "MAE", "CRPS", "RMSE_u", "MAE_u", "CRPS_u"]
    kv = [k for k in keys if k in metrics]
    line = prefix + " | ".join(f"{k}={metrics[k]:.4f}" for k in kv)
    print(line)


# -----------------------
# Batch grid runner (optional helper)
# -----------------------
@torch.no_grad()
def eval_grid_scent(
    root: str,
    dataset_tags: Iterable[str] = ("full", "10pct"),
    sparsity_pcts: Iterable[int] = (2, 5, 20),
    loader=None,
    *,
    crps_k: int = 10,
    std: Optional[float] = None,
):
    """Evaluate all 6 SCENT ckpts using a SINGLE provided loader.

    NOTE: Only use this if your `loader`'s sparsity matches the ckpts you're
    evaluating. Otherwise prefer `eval_grid_scent_autoload` below, which builds
    the proper val loader per sparsity.
    """
    from models.scent import path_for, SCENTNavierStokesModel

    rows = []
    for dt in dataset_tags:
        for sp in sparsity_pcts:
            ckpt = path_for(root, dataset_tag=dt, sparsity_pct=sp)
            model = SCENTNavierStokesModel(ckpt)
            metrics = evaluate_loader(model, loader, crps_k=crps_k, std=std)
            rows.append({"dataset_tag": dt, "sparsity_pct": sp, **metrics})
    return rows


@torch.no_grad()
def eval_grid_scent_autoload(
    root: str,
    *,
    mem,
    dataset_tags: Iterable[str] = ("full", "10pct"),
    sparsity_pcts: Iterable[int] = (2, 5, 20),
    mean: float,
    std: Optional[float] = None,
    batch_size: int = 32,
    mask_seed: int = 0,
    crps_k: int = 10,
):
    """Evaluate all SCENT ckpts with the CORRECT sparsity-specific val loader.

    This builds the fixed validation split once per dataset_tag, then for each
    sparsity percentage constructs a DataLoader with that sparsity so masks
    match the ckpt's training regime.
    """
    from models.scent import path_for, SCENTNavierStokesModel
    from models.ns_data import build_val_pairs_for, make_ns_val_loader

    rows = []
    for dt in dataset_tags:
        val_pairs, _T = build_val_pairs_for(mem, dt)
        for sp in sparsity_pcts:
            sp_f = sp / 100.0
            val_loader, _meta = make_ns_val_loader(
                mem=mem,
                val_pairs=val_pairs,
                sparsity=sp_f,
                mean=mean,
                std=std if std is not None else 1.0,
                batch_size=batch_size,
                mask_seed=mask_seed,
                verbose=False,
            )
            ckpt = path_for(root, dataset_tag=dt, sparsity_pct=sp)
            model = SCENTNavierStokesModel(ckpt)
            metrics = evaluate_loader(model, val_loader, crps_k=crps_k, std=std)
            rows.append({"dataset_tag": dt, "sparsity_pct": sp, **metrics})
    return rows


@torch.no_grad()
def eval_grid_omnifield_autoload(
    root: str,
    *,
    mem,
    dataset_tags: Iterable[str] = ("full", "10pct"),
    sparsity_pcts: Iterable[int] = (2, 5, 20),
    mean: float,
    std: Optional[float] = None,
    batch_size: int = 32,
    mask_seed: int = 0,
    crps_k: int = 10,
):
    """Evaluate all OmniField ckpts with the CORRECT sparsity-specific val loader.

    Builds the fixed validation split once per dataset_tag, then for each
    sparsity percentage constructs a DataLoader with that sparsity so masks
    match the ckpt's training regime.
    """
    from models.omnifield import path_for, OmniFieldNavierStokesModel
    from models.ns_data import build_val_pairs_for, make_ns_val_loader

    rows = []
    for dt in dataset_tags:
        val_pairs, _T = build_val_pairs_for(mem, dt)
        for sp in sparsity_pcts:
            sp_f = sp / 100.0
            val_loader, _meta = make_ns_val_loader(
                mem=mem,
                val_pairs=val_pairs,
                sparsity=sp_f,
                mean=mean,
                std=std if std is not None else 1.0,
                batch_size=batch_size,
                mask_seed=mask_seed,
                verbose=False,
            )
            ckpt = path_for(root, dataset_tag=dt, sparsity_pct=sp)
            model = OmniFieldNavierStokesModel(ckpt)
            metrics = evaluate_loader(model, val_loader, crps_k=crps_k, std=std)
            rows.append({"dataset_tag": dt, "sparsity_pct": sp, **metrics})
    return rows

