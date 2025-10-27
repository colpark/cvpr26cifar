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
from src.utils import FID




def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def make_sparse(img_tensor, percent):
    B, C, H, W = img_tensor.shape
    total_pixels = H * W
    num_known = int(percent * total_pixels)

    mask = torch.zeros(B, 1, H, W, device=img_tensor.device)

    for b in range(B):
        idx = torch.randperm(H * W, device=img_tensor.device)[:num_known]
        mask[b, 0].view(-1)[idx] = 1.0

    sparse = img_tensor * mask  # mask will broadcast to (B, C, H, W)
    return sparse, mask  # ← mask still shape (B, 1, H, W)



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
            sparse_input, mask = make_sparse(images, percent=sparsity)
            batches = num_to_groups(num_samples, batch_size)

            
            
            imgs = []
            start_idx = 0
            
            for n in batches:
                mask_chunk = mask[start_idx:start_idx + n]
                mask_expanded = mask_chunk.expand(-1, 3, -1, -1)
                imgs.append(model.sample(
                    batch_size=n,
                    sparse_input=sparse_input[start_idx:start_idx + n],
                    perceiver_input = None,
                    mask=mask_expanded,
                    clip=True   # <-- use only True
                ))
                start_idx += n
            imgs = torch.cat(imgs, dim=0)
            
            
            # === Log mean RGB and min/max values ===
            samples_vis = (imgs )         # [0, 1] for visualization
            real_vis = (images + 1) / 2          # [0, 1] GT

            gen_mean = samples_vis.mean(dim=[0, 2, 3])
            gt_mean = real_vis.mean(dim=[0, 2, 3])

            gen_min = samples_vis.min().item()
            gen_max = samples_vis.max().item()
            gt_min = real_vis.min().item()
            gt_max = real_vis.max().item()

        
        
        
            def compute_rgb_saturation(images):
                
                return images.mean(dim=[1, 2]) 

           
            samples_vis = imgs.clamp(0, 1)  
            gt_vis = ((images + 1) / 2).clamp(0, 1)  

            
            gt_sat = compute_rgb_saturation(gt_vis)
            gen_sat = compute_rgb_saturation(samples_vis)

            
            sat_diff = (gt_sat - gen_sat).abs()
            avg_sat_diff = sat_diff.mean().item()

            print(f"[{int(sparsity * 100)}% sparsity] Avg RGB saturation diff: {avg_sat_diff:.6f}")

        
           
            
            composite = torch.cat([
                (images + 1) / 2,
                (sparse_input + 1) / 2,
                mask.expand(-1, 3, -1, -1),
                (imgs )  # use imgs here
            ], dim=0)
      
            # Compute MSE for known and unknown pixels
            gt = images
            pred = imgs

            gt_flat = gt.view(gt.size(0), -1)
            pred_flat = pred.view(pred.size(0), -1)
            mask_expanded = mask.expand(-1, 3, -1, -1)  # (B, 3, H, W)
            mask_flat = mask_expanded.reshape(mask_expanded.size(0), -1)  # (B, 3072)

            # Avoid division by zero in case mask is empty
            eps = 1e-8

            mse_known = ((gt_flat - pred_flat) ** 2 * mask_flat).sum() / (mask_flat.sum() + eps)
            mse_unknown = ((gt_flat - pred_flat) ** 2 * (1 - mask_flat)).sum() / ((1 - mask_flat).sum() + eps)

            print(f"[{int(sparsity * 100)}% sparsity] MSE @ known pixels: {mse_known.item():.6f} | unknown pixels: {mse_unknown.item():.6f}")

            
            
            
            grid = make_grid(composite, nrow=num_samples)
            save_image(grid, os.path.join(save_dir, f"sparsity_grid_{int(sparsity*100)}.png"))
            print(f"Saved sparsity {int(sparsity*100)}% result to {save_dir}")

            save_path = os.path.join(save_dir, f"sparsity_{int(sparsity*100)}.png")
            save_image(imgs, nrow=4, fp=save_path)
            print(f"Saved: {save_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset (folder or cifar10)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--save_dir", type=str, default="./sparse_eval_test2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=16)
    args = parser.parse_args()

    # Sparsity levels from 2% to 50%
    sparsity_levels = [0.1]

    test_sparse_reconstruction(
        model=None,
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint,
        sparsity_levels=sparsity_levels,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
