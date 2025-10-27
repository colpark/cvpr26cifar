import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from src.model_original import Unet
from src.diffusion import GaussianDiffusion
from src.dataset import dataset_wrapper
from src.utils import FID

def make_sparse_random(img_tensor, percent):
    B, C, H, W = img_tensor.shape
    total_pixels = H * W
    num_known = int(percent * total_pixels)

    mask = torch.zeros(H * W, device=img_tensor.device)
    idx = torch.randperm(H * W, device=img_tensor.device)[:num_known]
    mask[idx] = 1
    mask = mask.view(1, 1, H, W).repeat(B, 1, 1, 1)

    sparse = img_tensor * mask
    return sparse, mask


def make_sparse_block(img_tensor, percent, block_size=5):
    """
    Create a block-wise sparse mask matching the desired percent of visible pixels.
    All blocks are used for conditioning (unlike training where only half are used).
    """
    B, C, H, W = img_tensor.shape
    device = img_tensor.device

    total_pixels = H * W
    visible_pixels = int(percent * total_pixels)
    pixels_per_block = block_size * block_size
    num_blocks = visible_pixels // pixels_per_block

    cond_mask = torch.zeros((B, 1, H, W), device=device)

    rng = np.random.default_rng()
    block_coords = []
    tries = 0
    max_tries = 1000

    while len(block_coords) < num_blocks and tries < max_tries:
        x = rng.integers(0, H - block_size + 1)
        y = rng.integers(0, W - block_size + 1)
        overlap = any(abs(bx - x) < block_size and abs(by - y) < block_size for bx, by in block_coords)
        if not overlap:
            block_coords.append((x, y))
        tries += 1

    for bx, by in block_coords:
        cond_mask[:, :, bx:bx + block_size, by:by + block_size] = 1

    cond_mask = cond_mask.repeat(1, C, 1, 1)
    sparse = img_tensor * cond_mask
    return sparse, cond_mask



@torch.no_grad()
def evaluate_fid_by_sparsity(
    checkpoint_path, dataset_path, mask_type, sparsity_levels,
    num_samples=30000, batch_size=128, save_dir="./fid_sparse_eval", num_composite=8
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "composites"), exist_ok=True)

    dataset = dataset_wrapper(dataset_path, image_size=32, augment_horizontal_flip=False, min1to1=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loader_iter = iter(dataloader)

    unet = Unet(
        dim=64,
        image_size=32,
        dim_multiply=(1, 2, 2, 2),
        channel=3,
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.1,
        device=device
    ).to(device)

    model = GaussianDiffusion(
        model=unet,
        image_size=32,
        time_step=1000,
        loss_type='l2'
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    fid_logger = {}

    for sparsity in sparsity_levels:
        print(f"\nEvaluating sparsity {int(sparsity*100)}%")

        
        mse_in_total = 0.0
        mse_out_total = 0.0
        mse_count = 0
        fid_calc = FID(
            batch_size=batch_size,
            dataLoader=dataloader,
            dataset_name="cifar10",
            device=device
        )

        def dynamic_sampler(batch_size, clip=True, min1to1=False):
            nonlocal loader_iter, mse_in_total, mse_out_total, mse_count
            try:
                real_batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                real_batch = next(loader_iter)

            real_imgs = real_batch[0] if isinstance(real_batch, (list, tuple)) else real_batch
            real_imgs = real_imgs.to(device)

            if mask_type == "random":
                sparse_input, mask = make_sparse_random(real_imgs, sparsity)
            elif mask_type == "block":
                sparse_input, mask = make_sparse_block(real_imgs, sparsity)
            else:
                raise ValueError(f"Unknown mask_type: {mask_type}")

                
      
            min_B = min(real_imgs.shape[0], sparse_input.shape[0], mask.shape[0])
            real_imgs = real_imgs[:min_B]
            sparse_input = sparse_input[:min_B]
            mask = mask[:min_B]

 
            samples, _ = model.sample(batch_size=min_B, sparse_input=sparse_input, mask=mask, clip=True)

            samples = samples * 2.0 - 1.0


            with torch.no_grad():
                mse_total = (samples - real_imgs) ** 2
                mse_in_hint = mse_total * mask
                mse_out_hint = mse_total * (1 - mask)

                mse_in = mse_in_hint.sum() / mask.sum()
                mse_out = mse_out_hint.sum() / (mask.numel() - mask.sum())

                mse_in_total += mse_in.item()
                mse_out_total += mse_out.item()
                mse_count += 1

            
            # Save composite grid (first batch only)
            if dynamic_sampler.first_call:
                composite = torch.cat([
                    make_grid(real_imgs[:num_composite], nrow=num_composite),
                    make_grid(sparse_input[:num_composite], nrow=num_composite),
                    make_grid(samples[:num_composite], nrow=num_composite)
                ], dim=1)
                save_path = os.path.join(save_dir, "composites", f"composite_sparsity_{int(sparsity*100)}.png")
                save_image(composite, save_path)
                dynamic_sampler.first_call = False

            
            
            return samples

        dynamic_sampler.first_call = True

        fid_score_val, _ = fid_calc.fid_score(dynamic_sampler, num_samples=num_samples, return_sample_image=False)
        fid_logger[int(sparsity * 100)] = fid_score_val
        print(f"FID@{int(sparsity*100)}%: {fid_score_val:.4f}")

    # Save FID log
    with open(os.path.join(save_dir, "fid_scores.txt"), "w") as f:
        for k, v in fid_logger.items():
            f.write(f"FID@{k}%: {v:.4f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--mask_type", type=str, choices=["random", "block"], required=True)
    parser.add_argument("--save_dir", type=str, default="./fid_sparse_eval_sweep")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_composite", type=int, default=8)
    args = parser.parse_args()

    sparsity_levels = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    evaluate_fid_by_sparsity(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        mask_type=args.mask_type,
        sparsity_levels=sparsity_levels,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        num_composite=args.num_composite
    )
