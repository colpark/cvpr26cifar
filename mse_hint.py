import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.diffusion import GaussianDiffusion
from src.dataset import dataset_wrapper
from src.model_original import Unet


@torch.no_grad()
def make_sparse(img_tensor, sparsity_level, mode="fixed_instance", pattern="random", block_size=5):
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

        elif mode == "fixed_batch":
            idx = torch.randperm(H * W)[:num_known]
            base_mask = torch.zeros(H * W, device=device)
            base_mask[idx] = 1
            mask = base_mask.view(1, 1, H, W).repeat(B, C, 1, 1)

        else:
            raise ValueError("Invalid mask_mode for random pattern")

    elif pattern == "block":
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
            assert placed == num_blocks, f"Could not place {num_blocks} blocks"
            return coords

        if mode == "fixed_instance":
            for b in range(B):
                coords = generate_block_coords()
                for x, y in coords:
                    mask[b, :, x:x+block_size, y:y+block_size] = 1

        elif mode == "fixed_batch":
            coords = generate_block_coords()
            for x, y in coords:
                mask[:, :, x:x+block_size, y:y+block_size] = 1

        else:
            raise ValueError("Invalid mask_mode for block pattern")

    else:
        raise ValueError("Unknown pattern type")

    sparse = img_tensor * mask
    return sparse, mask


@torch.no_grad()
def evaluate_mse(checkpoint_path, dataset_path, sparsity_levels, batch_size=128, mask_mode="fixed_instance",
                 pattern="random", save_dir="./mse_hint_eval", block_size=5, num_batches=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    dataset = dataset_wrapper(dataset_path, image_size=32, augment_horizontal_flip=False, min1to1=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

    logs = {}

    for sparsity in sparsity_levels:
        if pattern == "random":
            label = f"{int(sparsity * 100)}%"
        else:
            label = f"{int(sparsity)} blocks"

        print(f"Evaluating MSE (in-hint / out-hint) for: {label}")

        mse_in_total = 0
        mse_out_total = 0
        count_in, count_out = 0, 0

        loader_iter = iter(dataloader)

        for _ in tqdm(range(num_batches)):
            try:
                real_batch = next(loader_iter)
            except StopIteration:
                break

            real_imgs = real_batch[0].to(device)
            sparse_input, mask = make_sparse(real_imgs, sparsity, mode=mask_mode, pattern=pattern, block_size=block_size)

            recon = model.sample(batch_size=real_imgs.size(0), sparse_input=sparse_input, mask=mask, clip=True)

            diff = (recon - real_imgs) ** 2
            mse_in_total += (diff * mask).sum().item()
            mse_out_total += (diff * (1 - mask)).sum().item()
            count_in += mask.sum().item()
            count_out += (1 - mask).sum().item()

        mse_in = mse_in_total / count_in
        mse_out = mse_out_total / count_out

        logs[label] = {"mse_in": mse_in, "mse_out": mse_out}
        print(f"MSE In-Hint: {mse_in:.6f} | MSE Out-Hint: {mse_out:.6f}")

    # Save results
    with open(os.path.join(save_dir, "mse_hint_scores.txt"), "w") as f:
        for k, v in logs.items():
            f.write(f"{k}: in-hint={v['mse_in']:.6f}, out-hint={v['mse_out']:.6f}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./mse_hint_eval")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--mask_mode", type=str, default="fixed_instance", choices=["fixed_instance", "fixed_batch"])
    parser.add_argument("--pattern", type=str, default="random", choices=["random", "block"])
    parser.add_argument("--block_size", type=int, default=5)
    args = parser.parse_args()

    
    sparsity_levels = [0.01]
    
    evaluate_mse(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        sparsity_levels=sparsity_levels,
        batch_size=args.batch_size,
        mask_mode=args.mask_mode,
        pattern=args.pattern,
        save_dir=args.save_dir,
        block_size=args.block_size,
        num_batches=args.num_batches
    )
