import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from src.dataset import dataset_wrapper
from src.model_original import Unet
from src.diffusion import GaussianDiffusion

from src.perceiver import *


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


_shared_fixed_all_controller = None  # initialize as None

@torch.no_grad()
def make_sparse(img_tensor, sparsity_level, mode="fixed_all", pattern="random", block_size=5):
    global _shared_fixed_all_controller  # make sure to use global version

    B, C, H, W = img_tensor.shape
    device = img_tensor.device

    if pattern == "random":
        total_pixels = H * W
        num_known = int(sparsity_level * total_pixels)

        if mode == "fixed_instance":
            mask = torch.zeros(B, 1, H, W, device=device)
            for b in range(B):
                idx = torch.randperm(H * W)[:num_known]
                mask[b].view(-1)[idx] = 1
            mask = mask.repeat(1, C, 1, 1)

        elif mode == "fixed_all":
            if _shared_fixed_all_controller is None:
                from src.sparsity import SparsityController
                _shared_fixed_all_controller = SparsityController(
                    image_size=H,
                    mode='fixed_all',
                    pattern='random',
                    sparsity=sparsity_level,
                    block_size=block_size
                )
            cond_masks, _ = _shared_fixed_all_controller.get_masks(B, C)
            mask = torch.stack(cond_masks).to(device)

        else:
            raise ValueError("Invalid mode for random pattern")

    elif pattern == "block":
        if mode == "fixed_all":
            if _shared_fixed_all_controller is None:
                from src.sparsity import SparsityController
                _shared_fixed_all_controller = SparsityController(
                    image_size=H,
                    mode='fixed_all',
                    pattern='block',
                    sparsity=sparsity_level,
                    block_size=block_size,
                    num_blocks=int(sparsity_level)  # assumes `sparsity_level` is block count here
                )
            cond_masks, _ = _shared_fixed_all_controller.get_masks(B, C)
            mask = torch.stack(cond_masks).to(device)

        elif mode == "fixed_instance":
            num_blocks = int(sparsity_level)
            mask = torch.zeros(B, C, H, W, device=device)

            def generate_block_coords():
                coords = []
                placed, tries = 0, 0
                max_tries = 1000
                while placed < num_blocks and tries < max_tries:
                    x = torch.randint(0, H - block_size + 1, (1,)).item()
                    y = torch.randint(0, W - block_size + 1, (1,)).item()
                    if not any(abs(bx - x) < block_size and abs(by - y) < block_size for bx, by in coords):
                        coords.append((x, y))
                        placed += 1
                    tries += 1
                assert placed == num_blocks, f"Could not place {num_blocks} non-overlapping blocks"
                return coords

            for b in range(B):
                coords = generate_block_coords()
                for x, y in coords:
                    mask[b, :, x:x+block_size, y:y+block_size] = 1

        else:
            raise ValueError("Invalid mode for block pattern")

    else:
        raise ValueError("Unknown pattern type")

    sparse = img_tensor * mask
    return sparse, mask


@torch.no_grad()
def compute_uncertainty_map(checkpoint_path, dataset_path, sparsity, num_samples=10,
                             save_dir="./uq_map_eval", mask_mode="fixed_instance",
                             pattern="random", block_size=5, upscale=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Format folder name: e.g., "random_fixed_instance_5p" or "block_fixed_batch_10blocks"
    if pattern == "random":
        sparsity_label = f"{int(sparsity * 100)}p"
    else:
        sparsity_label = f"{int(sparsity)}blocks"

    folder_name = f"{pattern}_{mask_mode}_{sparsity_label}"
    save_path = os.path.join(save_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    
    perceiver = CascadedPerceiverIO(
            input_dim=3 + 192,  # 3 channels + 96*2 Fourier
            queries_dim=192,
            logits_dim=3,
            latent_dims=(256, 384, 512),
            num_latents=(256, 256, 256),
            decoder_ff=True
        ).to(device)
    perceiver.load_state_dict(torch.load("./perceiver_cifar10_sparseinvariant.pth"))
    perceiver.eval()

    fourier = GaussianFourierFeatures(in_features=2, mapping_size=96, scale=15.0).to(device)
    fourier.load_state_dict(torch.load("./fourier_encoderinvariant.pth"))
    fourier.eval()

    # Create coordinate grid
    coords = create_coordinate_grid(32, 32, device=device)
    
    # Load 1 image from dataset
    dataset = dataset_wrapper(dataset_path, image_size=32,augment_horizontal_flip=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    gt_img, *_ = next(iter(dataloader))
    gt_img = gt_img.to(device)

    
    
    sparse_input, mask = make_sparse(gt_img, sparsity_level=sparsity, mode=mask_mode, pattern=pattern, block_size=block_size)

    unet = Unet(
        dim=64, image_size=32, dim_multiply=(1, 2, 2, 2),
        channel=3, num_res_blocks=2, attn_resolutions=(16,), dropout=0.1, device=device
    ).to(device)

    model = GaussianDiffusion(model=unet, image_size=32, time_step=1000, loss_type='l2').to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()

    recon = run_perceiver_reconstruction(
                    sparse_img=sparse_input.to(device),
                    mask=mask.to(device),
                    model=perceiver,
                    fourier_encoder=fourier,
                    coords=coords
                ).to(device)
    # Generate multiple samples
    samples = []
    for _ in range(num_samples):
        out = model.sample(batch_size=1, sparse_input=sparse_input, perceiver_input = recon, mask=mask, clip=True)
        samples.append(out)
    samples = torch.stack(samples, dim=0)  # [N, 1, 3, 32, 32]

    mean = samples.mean(dim=0).squeeze(0)      # [3, H, W]
    var = samples.var(dim=0).squeeze(0)        # [3, H, W]
    var_gray = var.mean(dim=0, keepdim=True)   # [1, H, W]

    # Upscale for better visualization
    gt_up = F.interpolate((gt_img + 1) / 2, scale_factor=upscale, mode='bilinear', align_corners=False)
    mean_up = F.interpolate((mean.unsqueeze(0) ) , scale_factor=upscale, mode='bilinear', align_corners=False)
    sparse_up = F.interpolate((sparse_input + 1) / 2, scale_factor=upscale, mode='nearest')  # [1, 3, H*, W*]
    var_up = F.interpolate(var_gray.unsqueeze(0), scale_factor=upscale, mode='bilinear', align_corners=False)  # [1, 1, H*, W*]

    mse_gt_mean = F.mse_loss(mean, gt_img.squeeze(0)).item()
    print(f"[MSE] GT vs Mean Output: {mse_gt_mean:.6f}")
    
    # Save output images
    save_image((gt_img + 1) / 2, os.path.join(save_path, "gt.png"))
    save_image((sparse_input + 1) / 2, os.path.join(save_path, "sparse_input.png"))
    save_image(mean, os.path.join(save_path, "mean_output.png"))
    save_image(var / var.max(), os.path.join(save_path, "variance_map.png"))

    
    
    save_image(gt_up, os.path.join(save_path, "gt_up.png"))
    save_image(mean_up, os.path.join(save_path, "mean_up.png"))
    # Side-by-side matplotlib plot
    gt_np = gt_up[0].permute(1, 2, 0).cpu().numpy()
    mean_np = mean_up[0].permute(1, 2, 0).cpu().numpy()
    sparse_np = sparse_up[0].permute(1, 2, 0).cpu().numpy()  # HWC
    var_np = var_up[0, 0].cpu().numpy()                      # HW
    
    plt.figure(figsize=(14, 5))
    titles = ["Sparse Input", "Uncertainty Map", "GT", "Mean"]
    images = [sparse_np, var_np, gt_np, mean_np]

    for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 4, i + 1)
            if title == "Uncertainty Map":
                im = plt.imshow(img, cmap='hot')
                plt.colorbar(im, fraction=0.046, pad=0.04, label='Variance')
            else:
                plt.imshow(img)
            plt.title(title)
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "uq_composite.png"))
    plt.close()
    # Save individual samples (up to 8)
    for i in range(min(8, num_samples)):
        save_image(samples[i, 0], os.path.join(save_path, f"sample_{i}.png"))

    print(f"[✓] Saved all UQ results to {save_path}")

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./uq_map_eval2")
    
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--mask_mode", type=str, default="fixed_instance", choices=["fixed_instance", "fixed_all"])
    parser.add_argument("--pattern", type=str, default="random", choices=["random", "block"])
    parser.add_argument("--block_size", type=int, default=5)
    parser.add_argument("--upscale", type=int, default=8)
    args = parser.parse_args()

    sparsity = 0.5
    compute_uncertainty_map(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        sparsity=sparsity,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        mask_mode=args.mask_mode,
        pattern=args.pattern,
        block_size=args.block_size,
        upscale=args.upscale
    )
