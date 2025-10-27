"""
Navier–Stokes sparse next-step dataloader (shared across all models)
-------------------------------------------------------------------
- Recreates the exact fixed validation split you trained with (val_pairs=4900),
  using the same RNG seeds. This guarantees the *same samples* for SCENT,
  Diffusion, DDO, etc., as long as BASE_DIR/NAME point to the same memmap.
- Produces tensors: x_sparse [B,1,H,W], y [B,1,H,W], mask [B,H,W] (bool), sample_id [B]
- Deterministic per-sample mask via sample_id and mask_seed.

Typical use
-----------
from models.ns_data import (
    open_memmap,
    build_fixed_item_splits,
    make_ns_val_loader,
    make_subset_loader,
)

mem = open_memmap(BASE_DIR, NAME)
_, val_pairs, T = build_fixed_item_splits(mem, target_train_pairs=4900)  # dataset_tag = '10pct'
val_loader, meta = make_ns_val_loader(
    mem=mem,
    val_pairs=val_pairs,
    sparsity=0.20,
    mean=0.0, std=2.4036,
    batch_size=8,
    mask_seed=0,
)

# one batch
x_sparse, y, mask, sid = next(iter(val_loader))

# If you want *exact* same few examples across all models:
subset_loader = make_subset_loader(meta["dataset"], indices=[0, 1, 2], batch_size=3)

Notes
-----
* Validation split is independent of the training split size; so it is identical
  for dataset_tag 'full' and '10pct' when you use the same memmap + seeds.
* sample_id = item_idx * T + t, where t is the pair's time index.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from mmap_ninja import RaggedMmap
except Exception as e:  # helpful error if package missing
    RaggedMmap = None


# -----------------------
# Constants (match training)
# -----------------------
VAL_SEED = 12345
TRAIN_SEED = 0


# -----------------------
# Memory map helpers
# -----------------------

def open_memmap(base_dir: str, name: str):
    """Open RaggedMmap at base_dir/name."""
    assert RaggedMmap is not None, "mmap_ninja (RaggedMmap) not available. pip install mmap-ninja"
    path = os.path.join(base_dir, name)
    assert os.path.exists(path), f"Memmap path missing: {path}"
    return RaggedMmap(path, mode="r")


# -----------------------
# Fixed splits (deterministic, disjoint)
# -----------------------

def build_fixed_item_splits(
    mem,
    target_train_pairs: int,
    val_pairs: int = 4900,
    train_seed: int = TRAIN_SEED,
    val_seed: int = VAL_SEED,
):
    """Replicate your trainer's logic to build fixed train/val pairs.

    Returns: (train_pairs, val_pairs, T, train_items, val_items)
    """
    num_items_total = len(mem)
    assert num_items_total > 0

    T = int(mem[0].shape[0])
    assert T >= 2, "Need T>=2"
    pairs_per_item = T - 1

    items_for_val = int(np.ceil(val_pairs / pairs_per_item))
    items_for_train = int(np.ceil(target_train_pairs / pairs_per_item))

    rng_val = np.random.default_rng(val_seed)
    val_items = np.sort(rng_val.choice(num_items_total, size=items_for_val, replace=False))

    remaining = np.setdiff1d(np.arange(num_items_total), val_items, assume_unique=True)
    assert len(remaining) >= items_for_train, "Not enough remaining items to satisfy train size"

    rng_tr = np.random.default_rng(train_seed)
    train_items = np.sort(rng_tr.choice(remaining, size=items_for_train, replace=False))

    def pairs_from_items(items):
        pairs = [(int(it), int(t)) for it in items for t in range(T - 1)]
        return np.asarray(pairs, dtype=np.int64)

    val_ip = pairs_from_items(val_items)[:val_pairs]
    train_ip = pairs_from_items(train_items)[:target_train_pairs]

    # sanity: disjoint
    assert set(map(tuple, val_ip)).isdisjoint(set(map(tuple, train_ip))), "Train/val pairs overlap!"
    return train_ip, val_ip, T, train_items, val_items


# -----------------------
# Dataset (fixed per-sample mask)
# -----------------------
class NavierStokesSparseNextStep(Dataset):
    def __init__(
        self,
        memmap,
        index_pairs,
        sparsity: float = 0.05,
        normalize: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        to_unit_range: bool = False,
        seed: int = 42,
        verbose: bool = True,
        fixed_mask_per_sample: bool = True,
        mask_seed: int = 0,
    ):
        self.mem = memmap
        self.index_pairs = np.asarray(index_pairs, dtype=np.int64)
        self.normalize = bool(normalize)
        self.mean = float(mean)
        self.std = float(std)
        self.to_unit_range = bool(to_unit_range)
        self.sparsity = float(sparsity)
        self.rng = np.random.default_rng(seed)  # only used if not fixed
        self.fixed_mask_per_sample = bool(fixed_mask_per_sample)
        self.mask_seed = int(mask_seed)

        it0 = int(self.index_pairs[0, 0])
        self.T = int(self.mem[it0].shape[0])
        H, W = int(self.mem[it0].shape[1]), int(self.mem[it0].shape[2])
        self.H, self.W = H, W
        if verbose:
            print(
                f"[NS Sparse] pairs={len(self.index_pairs)}, T={self.T}, HW=({H},{W}), "
                f"sparsity={self.sparsity}, normalize={self.normalize}, unit_range={self.to_unit_range}, "
                f"fixed_mask_per_sample={self.fixed_mask_per_sample}"
            )

    def __len__(self):
        return len(self.index_pairs)

    def _z(self, a):
        return (a - self.mean) / self.std

    def _unit(self, a):
        return np.tanh(a)

    def __getitem__(self, i):
        item_idx, t = map(int, self.index_pairs[i])
        seq = self.mem[item_idx]  # (T,H,W)
        x = np.asarray(seq[t], dtype=np.float32)
        y = np.asarray(seq[t + 1], dtype=np.float32)

        if self.normalize:
            x = self._z(x)
            y = self._z(y)
        if self.to_unit_range:
            x = self._unit(x)
            y = self._unit(y)

        # deterministic per-sample mask keyed by sample_id
        sid = int(item_idx * self.T + t)
        if self.fixed_mask_per_sample:
            rng = np.random.default_rng(self.mask_seed + sid)
            mask = rng.random(x.shape, dtype=np.float32) < self.sparsity
        else:
            mask = self.rng.random(x.shape, dtype=np.float32) < self.sparsity

        x_sparse = x * mask.astype(np.float32)

        x_sparse = torch.from_numpy(x_sparse[None, ...])  # (1,H,W)
        y = torch.from_numpy(y[None, ...])  # (1,H,W)
        mask = torch.from_numpy(mask.astype(np.bool_))  # (H,W)
        sample_id = torch.tensor(sid, dtype=torch.long)
        return x_sparse, y, mask, sample_id


# -----------------------
# Loader builders
# -----------------------
@dataclass
class LoaderMeta:
    dataset: Dataset
    H: int
    W: int
    sparsity: float
    mean: float
    std: float
    pairs: np.ndarray  # (N,2) item_idx, t


def make_ns_val_loader(
    mem,
    val_pairs: np.ndarray,
    sparsity: float,
    mean: float,
    std: float,
    batch_size: int = 64,
    mask_seed: int = 0,
    verbose: bool = True,
):
    """Create the fixed-validation DataLoader used during training.

    Returns: (val_loader, meta)
    """
    val_ds = NavierStokesSparseNextStep(
        mem,
        val_pairs,
        sparsity=sparsity,
        normalize=True,
        mean=mean,
        std=std,
        to_unit_range=False,
        seed=42,
        verbose=verbose,
        fixed_mask_per_sample=True,
        mask_seed=mask_seed,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    meta = LoaderMeta(
        dataset=val_ds,
        H=val_ds.H,
        W=val_ds.W,
        sparsity=float(sparsity),
        mean=float(mean),
        std=float(std),
        pairs=val_pairs,
    )
    return loader, meta


def make_subset_loader(dataset: Dataset, indices: Iterable[int], batch_size: int = 1):
    """Deterministic subset (no shuffle) to reuse *exact* rows across models."""
    subset = Subset(dataset, list(indices))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


# -----------------------
# Small helpers for mapping tags
# -----------------------

def target_pairs_for(dataset_tag: str) -> int:
    tag = dataset_tag.strip().lower()
    if tag == "full":
        return 49000
    if tag == "10pct":
        return 4900
    # allow custom numeric fallback like "12345pairs"
    if tag.endswith("pairs"):
        try:
            return int(tag[:-5])
        except Exception:
            pass
    raise ValueError(f"Unrecognized dataset_tag: {dataset_tag}")


def build_val_pairs_for(mem, dataset_tag: str, val_pairs: int = 4900):
    """Helper to get val_pairs array consistent with a tag ('full'|'10pct')."""
    train_pairs_target = target_pairs_for(dataset_tag)
    _train_ip, val_ip, T, *_ = build_fixed_item_splits(mem, train_pairs_target, val_pairs=val_pairs)
    return val_ip, T
