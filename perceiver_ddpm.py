import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.dataset import dataset_wrapper

from src.model_original import Unet
from src.diffusion import GaussianDiffusion
from src.perceiver import *  # change to your actual import



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




# === Config ===
dataset_path = "cifar10"  # path to dataset
checkpoint_path = "./results/cifar10/2025-07-20_07h45m(percdetach)/model_latest.pt"
perceiver_ckpt = "./perceiver_cifar10_sparse10.pth"
fourier_ckpt = "./fourier_encoder.pth"
save_dir = "results/fused"
pattern = "random"
mask_mode = "fixed_instance"
sparsity = 0.1
block_size = 4  # unused unless pattern == "block"
num_samples = 1
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# === Folder naming ===
if pattern == "random":
    sparsity_label = f"{int(sparsity * 100)}p"
else:
    sparsity_label = f"{int(sparsity)}blocks"
folder_name = f"{pattern}_{mask_mode}_{sparsity_label}"
save_path = os.path.join(save_dir, folder_name)
os.makedirs(save_path, exist_ok=True)

# === Load 1 image ===
dataset = dataset_wrapper(dataset_path, image_size=32, augment_horizontal_flip=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
gt_img, *_ = next(iter(dataloader))
gt_img = gt_img.to(DEVICE)

# === Create sparse input + mask ===
sparse_input, mask = make_sparse(
    gt_img, sparsity_level=sparsity, mode=mask_mode, pattern=pattern, block_size=block_size
)

# === Load Perceiver and generate prior ===

perceiver = CascadedPerceiverIO(
        input_dim=3 + 192,  # 3 channels + 96*2 Fourier
        queries_dim=192,
        logits_dim=3,
        latent_dims=(256, 384, 512),
        num_latents=(256, 256, 256),
        decoder_ff=True
    ).to(DEVICE)
perceiver.load_state_dict(torch.load(perceiver_ckpt))
perceiver.eval()

fourier = GaussianFourierFeatures(in_features=2, mapping_size=96, scale=15.0).to(DEVICE)
fourier.load_state_dict(torch.load(fourier_ckpt))
fourier.eval()

# Create coordinate grid
coords = create_coordinate_grid(32, 32, device=DEVICE)

# Run Perceiver reconstruction
recon = run_perceiver_reconstruction(
    sparse_img=sparse_input.to(DEVICE),
    mask=mask.to(DEVICE),
    model=perceiver,
    fourier_encoder=fourier,
    coords=coords
).to(DEVICE)



with torch.no_grad():
   
    recon = recon.clamp(-1, 1)

# === Fuse sparse + perceiver ===

# === Load DDPM model ===
unet = Unet(
    dim=64, image_size=32, dim_multiply=(1, 2, 2, 2),
    channel=3, num_res_blocks=2, attn_resolutions=(16,), dropout=0.1, device=DEVICE
).to(DEVICE)

model = GaussianDiffusion(model=unet, image_size=32, time_step=1000, loss_type='l2').to(DEVICE)
state = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(state['model'])
model.eval()

# === Generate samples ===
samples = []
for _ in range(num_samples):
    out = model.sample(batch_size=1, sparse_input=sparse_input.to(DEVICE), perceiver_input = recon.to(DEVICE), mask=mask.to(DEVICE), clip=True)
    samples.append(out)
samples = torch.cat(samples, dim=0)  # [num_samples, 3, 32, 32]

# samples_without = []
# for _ in range(num_samples):
#     out = model.sample(batch_size=1, sparse_input=sparse_input.to(DEVICE), mask=mask.to(DEVICE), clip=True)
#     samples_without.append(out)
# samples_without = torch.cat(samples_without, dim=0)  # [num_samples, 3, 32, 32]


# === Save images ===
save_image((gt_img + 1) / 2, os.path.join(save_path, "gt.png"))
save_image((sparse_input + 1) / 2, os.path.join(save_path, "sparse_input.png"))
save_image((recon + 1) / 2, os.path.join(save_path, "perceiver_output.png"))
save_image((samples ) , os.path.join(save_path, "ddpm_outputs.png"), nrow=5)


print(f"Saved results to: {save_path}")
