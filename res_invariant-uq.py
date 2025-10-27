import os
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from src.diffusion import GaussianDiffusion
from src.dataset import dataset_wrapper
from src.model_original import Unet
import argparse
from torch.nn.functional import mse_loss




def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def make_split_sparse(img_tensor, percent_total=0.2):
    """Split sparse pixels into 50% conditioning and 50% loss"""
    B, C, H, W = img_tensor.shape
    total_pixels = H * W
    num_total = int(percent_total * total_pixels)
    num_cond = num_total // 2
    num_target = num_total - num_cond

    mask = torch.zeros(H * W, device=img_tensor.device)
    idx = torch.randperm(H * W)
    cond_idx = idx[:num_cond]
    target_idx = idx[num_cond:num_total]

    cond_mask = torch.zeros_like(mask)
    cond_mask[cond_idx] = 1

    target_mask = torch.zeros_like(mask)
    target_mask[target_idx] = 1

    cond_mask = cond_mask.view(1, 1, H, W).repeat(B, C, 1, 1)
    target_mask = target_mask.view(1, 1, H, W).repeat(B, C, 1, 1)

    sparse = img_tensor * cond_mask
    return sparse, cond_mask, target_mask



def test_sparse_reconstruction(model, dataset_path, checkpoint_path, sparsity_levels, batch_size=16, num_samples=16, save_dir="./sparse_eval"):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    dataset = dataset_wrapper(dataset_path, image_size=32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images, *_ = next(iter(dataloader))  # Unpack image from (image, label)
    images = images[:num_samples].to(device)

    # Load model
    unet = Unet(
        dim=64,
        image_size=32,
        dim_multiply=(1, 2, 2, 2),  # must match training
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

    with torch.inference_mode():
        for sparsity in sparsity_levels:
            sparse_input, cond_mask, target_mask = make_split_sparse(images, percent_total=sparsity)
            batches = num_to_groups(num_samples, batch_size)

            M = 5  # Number of sampling runs
            all_samples = []

            for _ in range(M):
                start_idx = 0
                samples = []

                for n in batches:
                    sample_batch, _ = model.sample(
                        batch_size=n,
                        sparse_input=sparse_input[start_idx:start_idx + n],
                        mask=cond_mask[start_idx:start_idx + n],
                        clip=True,estimate_uq_per_step=False
                    )
                    samples.append(sample_batch)
                    start_idx += n

                all_samples.append(torch.cat(samples, dim=0))  # shape (B, C, H, W)

            all_samples = torch.stack(all_samples, dim=0)  # shape (M, B, C, H, W)
            U_final = all_samples.var(dim=0, unbiased=False)  # shape (B, C, H, W)
            imgs = all_samples.mean(dim=0)  # final averaged image

            # Visualization grid
            composite = torch.cat([
                (images + 1) / 2,
                (sparse_input + 1) / 2,
                cond_mask,
                (imgs)
            ], dim=0)
    

         
            # MSE on target (held-out) pixels only
            eps = 1e-8
            gt_flat = ((images + 1) / 2).view(images.size(0), -1)
            pred_flat = ((imgs + 1) / 2).view(imgs.size(0), -1)
            target_mask_flat = target_mask.view(target_mask.size(0), -1)

            mse_unknown = ((gt_flat - pred_flat) ** 2 * target_mask_flat).sum() / (target_mask_flat.sum() + eps)
            print(f"[{int(sparsity * 100)}% total sparse] MSE @ held-out target pixels: {mse_unknown.item():.6f}")

            # Save results
            grid = make_grid(composite, nrow=num_samples)
            save_image(grid, os.path.join(save_dir, f"sparsity_grid_{int(sparsity*100)}.png"))
            save_image(imgs, nrow=4, fp=os.path.join(save_dir, f"sparsity_{int(sparsity*100)}.png"))
            
            u_grid = make_grid(U_final[:, :1], nrow=4, normalize=True, scale_each=True)
            save_image(u_grid, os.path.join(save_dir, f"uncertainty_{int(sparsity*100)}.png"))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset (folder or cifar10)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--save_dir", type=str, default="./sparse_res_eval_uq_test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=32)
    args = parser.parse_args()

    # Sparsity levels from 2% to 50%
    sparsity_levels = [ 0.1, 0.2, 0.3, 0.5,0.6]

    test_sparse_reconstruction(
        model=None,
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint,
        sparsity_levels=sparsity_levels,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )