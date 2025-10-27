# evaluate_ns.py
import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

# ---- your project imports ----
from src.model_original import Unet
from src.diffusion import GaussianDiffusion, DDIM_Sampler
from src.dataset import NavierStokesNextStepDDPM
from src.sparsity import SparsityController
from mmap_ninja import RaggedMmap


# ------------------------- utils -------------------------

@torch.no_grad()
def _atanh_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Numerically safe inverse tanh. Assumes x in (-1,1); clamps to avoid infs.
    """
    x = x.clamp(min=-1 + eps, max=1 - eps)
    # atanh(x) = 0.5 * ln((1+x)/(1-x)) ; use log1p for stability
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

@torch.no_grad()
def denorm_to_physical(x: torch.Tensor, *, normalize: bool, to_unit_range: bool,
                       mean: float, std: float) -> torch.Tensor:
    """
    Inverts the dataset transforms from NavierStokesNextStepDDPM.__getitem__:
      if to_unit_range: x <- tanh( (arr - mean)/std )
      elif normalize:   x <- (arr - mean)/std
      else:             x <- arr
    Returns arr (physical units).
    """
    if to_unit_range:
        z = _atanh_clamped(x)
        return z * std + mean
    elif normalize:
        return x * std + mean
    else:
        return x


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_split_indices(mem, train_max_items: int, seed: int):
    """
    Recreates the same style of split you used in training:
      - choose 'train_max_items' unique items for train
      - test = all remaining items
    Returns (train_items, test_items) as numpy arrays (sorted).
    """
    rng = np.random.default_rng(seed)
    num_items_total = len(mem)
    pick = min(train_max_items, num_items_total)
    chosen_train = rng.choice(num_items_total, size=pick, replace=False)
    chosen_train = np.sort(chosen_train)

    all_items = np.arange(num_items_total, dtype=np.int64)
    test_items = np.setdiff1d(all_items, chosen_train, assume_unique=True)
    return chosen_train, test_items


def build_pairs_for_items(mem, items: np.ndarray):
    """
    Build (t -> t+1) pairs for all items given.
    Assumes fixed T across those items.
    """
    assert len(items) > 0
    T = int(mem[int(items[0])].shape[0])
    assert T >= 2, "Need at least 2 frames per item."
    pairs = [(int(it), int(t)) for it in items for t in range(T - 1)]
    pairs = np.asarray(pairs, dtype=np.int64)
    return pairs, T


@torch.no_grad()
def apply_colormap(tensor):
    """
    tensor: (B, 1, H, W) in arbitrary range. Returns (B, 3, H, W) in [0,1]
    (Uses matplotlib on CPU; fine for a few grids.)
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('viridis')
    t = tensor.detach().cpu().numpy()
    # normalize each image to [0,1] per-image
    t = (t - t.min(axis=(2, 3), keepdims=True)) / (t.max(axis=(2, 3), keepdims=True) - t.min(axis=(2, 3), keepdims=True) + 1e-8)
    rgb_list = []
    for img in t:
        img2d = img[0]
        rgba = cmap(img2d)
        rgb = torch.from_numpy(rgba[..., :3]).permute(2, 0, 1)  # (3,H,W)
        rgb_list.append(rgb)
    return torch.stack(rgb_list, dim=0)  # (B,3,H,W)


def make_sparse_batch(y, sample_ids, sparsity_ctrl: SparsityController, device):
    """
    y: (B,1,H,W) -> return sparse_input, cond_mask, target_mask as (B,1,H,W) on device
    """
    B, C, H, W = y.shape
    cond_masks, target_masks = sparsity_ctrl.get_masks(B, C, [int(s) for s in sample_ids])
    cond_mask   = torch.stack(cond_masks,  dim=0).to(device)
    target_mask = torch.stack(target_masks, dim=0).to(device)
    sparse_input = y * cond_mask
    return sparse_input, cond_mask, target_mask


def batch_metrics(y, y_hat, target_mask):
    """
    Returns dict of scalar tensors: mse_full, mse_unknown
    (y, y_hat, target_mask): (B,1,H,W) in same scale ([-1,1] if using min1to1=True).
    """
    eps = 1e-12
    mse_full = torch.mean((y_hat - y) ** 2)

    # Unknown region only (where target_mask == 1)
    denom = target_mask.sum()
    if denom <= 0:
        mse_unknown = torch.tensor(0.0, device=y.device)
    else:
        mse_unknown = torch.sum(((y_hat - y) ** 2) * target_mask) / (denom + eps)

    return {
        'mse_full': mse_full,
        'mse_unknown': mse_unknown,
    }


# ---------------------- CRPS helpers ----------------------

def empirical_crps_components(samples, y, mask=None):
    """
    Empirical CRPS:
      CRPS(F,y) = E|X - y| - 0.5 E|X - X'|
    samples: (S, B, 1, H, W)
    y:       (B, 1, H, W)
    mask:    (B, 1, H, W) or None
    Returns scalar tensors: crps_full, crps_unknown
    """
    S, B, _, H, W = samples.shape

    # E|X - y|
    abs_xy = torch.abs(samples - y.unsqueeze(0))  # (S,B,1,H,W)
    e_abs_xy = abs_xy.mean(dim=0)                 # (B,1,H,W)

    # 0.5 E|X - X'|
    N = B * H * W
    flat = samples.view(S, N)
    diff = flat.unsqueeze(0) - flat.unsqueeze(1)          # (S,S,N)
    e_abs_xx = torch.abs(diff).mean(dim=(0, 1)) * 0.5     # (N,)
    e_abs_xx = e_abs_xx.view(B, 1, H, W)

    crps_map = e_abs_xy - e_abs_xx                        # (B,1,H,W)

    crps_full = crps_map.mean()

    if mask is None:
        crps_unk = torch.tensor(float("nan"), device=y.device)
    else:
        denom = mask.sum()
        crps_unk = (crps_map * mask).sum() / (denom + 1e-12)

    return crps_full, crps_unk


def sample_ensemble(diffusion, sampler, batch_size, sparse, cond_mask, S):
    """
    Returns (S, B, 1, H, W) samples in [-1,1].
    Uses DDPM if sampler is None; otherwise uses the provided sampler (DDIM).
    """
    outs = []
    for _ in range(S):
        if sampler is None:
            y_hat = diffusion.sample(
                batch_size=batch_size,
                clip=True,
                min1to1=True,
                sparse_input=sparse,
                perceiver_input=None,
                mask=cond_mask
            )
        else:
            y_hat = sampler.sample(
                diffusion_model=diffusion,
                batch_size=batch_size,
                noise=None,
                return_all_timestep=False,
                clip=True,
                min1to1=True,
                sparse_input=sparse,
                perceiver_input=None,
                mask=cond_mask
            )
        outs.append(y_hat)
    return torch.stack(outs, dim=0)  # (S,B,1,H,W)


# ------------------------- main --------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DDPM/DDIM on unseen Navier–Stokes data")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML (mirror settings)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trainer checkpoint (e.g., model_latest.pt)")
    parser.add_argument("--mmap_dir", type=str, required=True, help="Path to RaggedMmap directory (BASE/NAME)")
    parser.add_argument("--device", type=str, default="cuda:2", choices=["cuda", "cpu", "cuda:1","cuda:2"])
    parser.add_argument("--train_max_items", type=int, default=1000, help="How many items were used for training")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for building the training subset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--small_eval_items", type=int, default=100, help="Size of smaller test split (items)")
    parser.add_argument("--save_dir", type=str, default="./eval_results")
    parser.add_argument("--save_grids", action="store_true", default=True, help="Save qualitative grids")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM instead of vanilla DDPM sampling")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta (set >0 for stochastic samples)")
    parser.add_argument("--crps_samples", type=int, default=10, help="If >0, compute CRPS with this many samples")
    parser.add_argument("--only_small", action="store_true", default=True,
                        help="Skip the full split and evaluate only the small (K items) split (default: True).")
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- Load config ----
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    unet_cfg    = cfg["unet"].copy()
    trainer_cfg = cfg["trainer"].copy()

    # ensure single-channel base (since model_input is 3=1*3)
    unet_cfg["channel"] = 1

    image_size = unet_cfg["image_size"]
    device = torch.device(args.device)

    # ---- Build model ----
    unet = Unet(**unet_cfg).to(device)
    diffusion = GaussianDiffusion(unet, image_size=image_size).to(device)

    # ---- Load checkpoint saved by Trainer.save(...) ----
    ckpt = torch.load(args.ckpt, map_location=device)
    diffusion.load_state_dict(ckpt["model"], strict=True)
    diffusion.eval()
    unet.eval()

    # ---- Data (memmap) ----
    mem = RaggedMmap(args.mmap_dir, mode="r")

    # splits
    train_items, test_items_all = build_split_indices(mem, args.train_max_items, args.seed)
    if len(test_items_all) == 0:
        raise RuntimeError("No unseen items left after excluding training items. "
                           "Reduce --train_max_items or point to a larger mmap.")

    # full + small item lists
    test_pairs_full, T = build_pairs_for_items(mem, test_items_all)
    print(f"[Eval] Full test: {len(test_items_all)} items → {len(test_pairs_full)} pairs, T={T}")

    K = min(args.small_eval_items, len(test_items_all))
    small_items = test_items_all[:K]
    test_pairs_small, _ = build_pairs_for_items(mem, small_items)
    print(f"[Eval] Small test: {len(small_items)} items → {len(test_pairs_small)} pairs")

    # ---- Datasets / loaders (mirror training normalization) ----
    ds_kwargs = dict(
        memmap=mem,
        normalize=True,
        mean=0.0,
        std=2.4036,
        to_unit_range=True,   # you trained with this in your snippet
        verbose=True
    )

    # flags for denormalization
    mean_for_denorm = float(ds_kwargs["mean"])
    std_for_denorm  = float(ds_kwargs["std"])
    norm_flag       = bool(ds_kwargs["normalize"])
    unit_flag       = bool(ds_kwargs["to_unit_range"])

    loader_full = None
    if not args.only_small:
        test_ds_full = NavierStokesNextStepDDPM(index_pairs=test_pairs_full, **ds_kwargs)
        loader_full  = DataLoader(test_ds_full, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    test_ds_small = NavierStokesNextStepDDPM(index_pairs=test_pairs_small, **ds_kwargs)
    loader_small  = DataLoader(test_ds_small, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    # ---- SparsityController ----
    sparsity = SparsityController(
        image_size=image_size,
        mode=trainer_cfg.get("sparsity_mode", "fixed_instance"),
        pattern=trainer_cfg.get("sparsity_pattern", "random"),
        sparsity=trainer_cfg.get("sparsity_level", 0.1),
        block_size=trainer_cfg.get("block_size", 5),
        num_blocks=trainer_cfg.get("num_blocks", 6),
    )

    # ---- Sampler ----
    if args.use_ddim:
        ddim_sampler = DDIM_Sampler(
            diffusion, ddim_sampling_steps=50, eta=args.ddim_eta,
            sample_every=10_000, calculate_fid=False, generate_image=False, clip=True
        )
    else:
        ddim_sampler = None

    # ---- Eval loop ----
    def run_eval(loader, tag):
        all_mse_full, all_mse_full_phys = [], []
        all_crps_full, all_crps_unknown = [], []
        all_crps_full_phys, all_crps_unknown_phys = [], []

        first_batch_saved = False
        for y, sample_id in tqdm(loader, desc=f"Eval {tag}", leave=False):
            y = y.to(device).float()
            sids = [int(s.item()) for s in sample_id]

            sparse, cond_mask, target_mask = make_sparse_batch(y, sids, sparsity, device)

            # prediction
            if ddim_sampler is None:
                y_hat = diffusion.sample(
                    batch_size=y.size(0), clip=True, min1to1=True,
                    sparse_input=sparse, perceiver_input=None, mask=cond_mask
                )
            else:
                y_hat = ddim_sampler.sample(
                    diffusion_model=diffusion, batch_size=y.size(0), noise=None,
                    return_all_timestep=False, clip=True, min1to1=True,
                    sparse_input=sparse, perceiver_input=None, mask=cond_mask
                )

            # normalized-space MSE
            m = batch_metrics(y, y_hat, target_mask)
            all_mse_full.append(m["mse_full"].item())

            # ---- Physical-space MSE ----
            y_phys = denorm_to_physical(y, normalize=norm_flag, to_unit_range=unit_flag,
                                        mean=mean_for_denorm, std=std_for_denorm)
            y_hat_phys = denorm_to_physical(y_hat, normalize=norm_flag, to_unit_range=unit_flag,
                                            mean=mean_for_denorm, std=std_for_denorm)
            mse_full_phys = torch.mean((y_hat_phys - y_phys) ** 2)
            all_mse_full_phys.append(mse_full_phys.item())

            # CRPS (optional)
            if args.crps_samples and args.crps_samples > 0:
                samples = sample_ensemble(diffusion, ddim_sampler, y.size(0), sparse, cond_mask, args.crps_samples)

                crps_full, crps_unknown = empirical_crps_components(samples, y, target_mask)
                all_crps_full.append(crps_full.item())
                all_crps_unknown.append(crps_unknown.item())

                samples_phys = denorm_to_physical(samples, normalize=norm_flag, to_unit_range=unit_flag,
                                                  mean=mean_for_denorm, std=std_for_denorm)
                y_phys = denorm_to_physical(y, normalize=norm_flag, to_unit_range=unit_flag,
                                            mean=mean_for_denorm, std=std_for_denorm)
                crps_full_phys, crps_unknown_phys = empirical_crps_components(samples_phys, y_phys, target_mask)
                all_crps_full_phys.append(crps_full_phys.item())
                all_crps_unknown_phys.append(crps_unknown_phys.item())

            # save grid (first batch only)
            if args.save_grids and (not first_batch_saved):
                out_dir = os.path.join(args.save_dir, "grids")
                os.makedirs(out_dir, exist_ok=True)
                B = min(8, y.size(0))
                comp = torch.cat([
                    apply_colormap(y[:B]),
                    apply_colormap((y[:B] * cond_mask[:B])),
                    apply_colormap(cond_mask[:B]),
                    apply_colormap(y_hat[:B]),
                    apply_colormap(target_mask[:B]),
                ], dim=0)
                save_image(make_grid(comp, nrow=B), os.path.join(out_dir, f"grid_{tag}.png"))
                first_batch_saved = True

        results = {
            "mse_full": float(np.mean(all_mse_full)) if all_mse_full else float("nan"),
            "mse_full_phys": float(np.mean(all_mse_full_phys)) if all_mse_full_phys else float("nan"),
            "num_batches": len(all_mse_full)
        }
        if all_crps_full:
            results.update({
                "crps_full": float(np.mean(all_crps_full)),
                "crps_unknown": float(np.mean(all_crps_unknown)),
                "crps_full_phys": float(np.mean(all_crps_full_phys)),
                "crps_unknown_phys": float(np.mean(all_crps_unknown_phys)),
                "crps_S": int(args.crps_samples)
            })
        return results

    os.makedirs(args.save_dir, exist_ok=True)

    res_full = None
    if loader_full is not None:
        print("[Eval] Running FULL split (this can be large).")
        res_full = run_eval(loader_full, tag="full")

    print("[Eval] Running SMALL split.")
    res_small = run_eval(loader_small, tag="small")

    # Save + print
    summary = {
        "small_split": res_small,
        "notes": {
            "small_items": int(K),
            "unseen_items_total": int(len(test_items_all)),
            "T": int(T),
            "train_max_items": int(args.train_max_items),
            "seed": int(args.seed),
            "sampler": "DDIM" if args.use_ddim else "DDPM",
            "ddim_eta": float(args.ddim_eta),
            "crps_samples": int(args.crps_samples),
            "only_small": bool(args.only_small),
        }
    }
    if res_full is not None:
        summary["full_split"] = res_full

    out_path = os.path.join(args.save_dir, "metrics_ns_eval.json")
    import json
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved metrics to: {out_path}")
    if args.save_grids:
        print(f"Saved qualitative grids under: {os.path.join(args.save_dir, 'grids')}")

if __name__ == "__main__":
    main()
