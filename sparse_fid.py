import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from src.diffusion import GaussianDiffusion
from src.dataset import dataset_wrapper
from src.model_original import Unet
from src.utils import *

from src.perceiver import *

def make_sampler(model, sparse_input, mask):
    def sampler(batch_size, clip=True, min1to1=False):
        nonlocal sparse_input, mask
        return model.sample(
            batch_size=batch_size,
            sparse_input=sparse_input[:batch_size],
            mask=mask[:batch_size],
            clip=clip
        )
    return sampler


def dynamic_sampler(batch_size, clip=True, min1to1=False):
    try:
        real_batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(dataloader_real)
        real_batch = next(loader_iter)
    
    real_imgs = real_batch if isinstance(real_batch, torch.Tensor) else real_batch[0]
    real_imgs = real_imgs.to(device)[:batch_size]
    sparse_input, mask = make_sparse(real_imgs, sparsity)
    
    return model.sample(batch_size=batch_size, sparse_input=sparse_input, mask=mask, clip=clip)




_shared_fixed_all_controller = None  # initialize as None

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


    
    
def evaluate_fid(checkpoint_path, dataset_path, sparsity_levels, num_samples=30000, batch_size=128, save_dir="./fid_sparse_eval", mask_mode="fixed_instance",pattern ="block"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    dataset = dataset_wrapper(dataset_path, image_size=32, augment_horizontal_flip=False, min1to1=False)
    dataloader_real = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

    output_dir = os.path.join(save_dir, "perceiver_outputs")
    os.makedirs(output_dir, exist_ok=True)

    fid_logger = {}
    for sparsity in sparsity_levels:
        if pattern == "random":
            sparsity_label = f"{int(sparsity * 100)}%"
        elif pattern == "block":
            sparsity_label = f"{int(sparsity)} blocks"
        else:
            sparsity_label = str(sparsity)

        print(f"Evaluating FID for sparsity: {sparsity_label}")

        def dynamic_sampler_factory(sparsity_level, dataloader, mask_mode,pattern):
            dataloader_iter = iter(dataloader)

            def sampler(batch_size, clip=True, min1to1=False):
                nonlocal dataloader_iter
                try:
                    real_batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    real_batch = next(dataloader_iter)

                real_imgs = real_batch[0].to(device)[:batch_size]
                sparse_input, mask = make_sparse(real_imgs, sparsity_level, mode=mask_mode,pattern=pattern)
                
                  
                recon = run_perceiver_reconstruction(
                    sparse_img=sparse_input.to(device),
                    mask=mask.to(device),
                    model=perceiver,
                    fourier_encoder=fourier,
                    coords=coords
                ).to(device)
                
#                 # Denormalize if necessary (assuming recon in [0,1])
#                 recon_vis = recon.clone().detach().cpu()
#                 if recon_vis.min() < 0 or recon_vis.max() > 1:
#                     recon_vis = recon_vis  # assuming recon in [-1, 1] range

#                 # Save only the first N images as a grid (e.g., 8)
                # save_image(recon_vis[:8], os.path.join(output_dir, f"perceiver_recon_{sparsity_level}.png"), nrow=4)
                
                
                
#                 fused_input = recon * (1 - mask) + sparse_input * mask
                
               
                # return model.sample(batch_size=batch_size, sparse_input=sparse_input, mask=mask, clip=clip)
                return model.sample(batch_size=batch_size, sparse_input=sparse_input, perceiver_input = recon, mask=mask, clip=clip)

            return sampler


        fid_calc = FID(
            batch_size=batch_size,
            dataLoader=dataloader_real,
            dataset_name="cifar10",
            device=device
        )

    
        # Create output directory for FID samples
        fid_sample_dir = os.path.join(save_dir, "fid_samples")
        os.makedirs(fid_sample_dir, exist_ok=True)
        
        sampler_fn = dynamic_sampler_factory(sparsity, dataloader_real, mask_mode,pattern)
        fid_score_val, sample_image = fid_calc.fid_score(sampler_fn, num_samples=num_samples, return_sample_image=True)
        
        save_image((sample_image[:8] - 1) /2, os.path.join(fid_sample_dir, f"fid_sample_{sparsity_label}.png"), nrow=4)
        

        fid_logger[sparsity_label] = fid_score_val
        print(f"FID@{sparsity_label}: {fid_score_val:.4f}")

    with open(os.path.join(save_dir, "fid_scores.txt"), "w") as f:
        for k, v in fid_logger.items():
            f.write(f"FID@{k}%: {v:.4f}\n")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./fid_sparse_eval_perc")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--mask_mode", type=str, default="fixed_instance", choices=["fixed_instance", "fixed_all"])
    parser.add_argument("--pattern", type=str, default="random", choices=["random", "block"])

    args = parser.parse_args()

    sparsity_levels = [0.02]
    evaluate_fid(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        sparsity_levels=sparsity_levels,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        mask_mode=args.mask_mode,
        pattern=args.pattern
    )