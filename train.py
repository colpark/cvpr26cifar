from src import model_torch
from src import model_original
from src.trainer import Trainer
from src.diffusion import GaussianDiffusion, DDIM_Sampler
from src.dataset import NavierStokesNextStepDDPM
import yaml
import argparse
from mmap_ninja import RaggedMmap
import os
import numpy as np

def build_index_pairs(mem, max_items=100, seed=0):
    """
    Choose up to `max_items` sequences (items) from `mem`, then build (t, t+1) pairs
    for each chosen item. Assumes each chosen item has the same T (e.g., 50).
    """
    rng = np.random.default_rng(seed)

    num_items_total = len(mem)
    pick = min(max_items, num_items_total)

    # choose which items to keep
    chosen_item_idxs = rng.choice(num_items_total, size=pick, replace=False)
    chosen_item_idxs = np.sort(chosen_item_idxs)

    # assume fixed T across chosen items (skip items with T<2 just in case)
    T = int(mem[int(chosen_item_idxs[0])].shape[0])
    assert T >= 2, "Need at least 2 frames per item."

    # build (t, t+1) pairs only from the chosen items
    index_pairs = [(int(it), int(t)) for it in chosen_item_idxs for t in range(T - 1)]
    index_pairs = np.asarray(index_pairs, dtype=np.int64)

    print(f"Using {len(chosen_item_idxs)} items; built {len(index_pairs)} (t→t+1) pairs with T={T}.")
    return index_pairs, T



def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    unet_cfg = config['unet']
    ddim_cfg = config['ddim']
    trainer_cfg = config['trainer']
    image_size = unet_cfg['image_size']

    if config['type'] == 'original':
        unet = model_original.Unet(**unet_cfg).to(args.device)
    elif config['type'] == 'torch':
        unet = model_torch.Unet(**unet_cfg).to(args.device)
    else:
        unet = None
        print("Unet type must be one of ['original', 'torch']")
        exit()


    
    def count_params(m):
        # if wrapped by DDP/DataParallel later, this keeps it robust
        base = getattr(m, 'module', m)
        total = sum(p.numel() for p in base.parameters())
        trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
        return total, trainable
    
    total_params, trainable_params = count_params(unet)
    
    def humanize(n):
        # 1,234,567 -> "1.23M"
        for unit in ['','K','M','B','T']:
            if abs(n) < 1000:
                return f"{n:.0f}{unit}"
            n /= 1000.0
        return f"{n:.2f}P"
    
    print(f"[Model] Total params: {total_params} ({humanize(total_params)}) "
          f"| Trainable: {trainable_params} ({humanize(trainable_params)})")
    diffusion = GaussianDiffusion(unet, image_size=image_size).to(args.device)

    ddim_samplers = list()

    # --- Build mem + index_pairs here (Python), inject via trainer_cfg ---
    BASE_DIR = "/pscratch/sd/d/dpark1/NSData"
    NAME = "100k"
    mmap_path = os.path.join(BASE_DIR, NAME)
    mem = RaggedMmap(mmap_path, mode="r")

    index_pairs, T = build_index_pairs(mem, max_items=100, seed=0)
    print(f"Using {len(index_pairs)} (t→t+1) pairs; sequence length T={T}")

    trainer_cfg.update(dict(
        dataset='navierstokes',
        memmap=mem,
        index_pairs=index_pairs,
        normalize=True,
        mean=0.0,
        std=2.4036,
        to_unit_range=True,       # optional (tanh squashing)
    ))
    for sampler_cfg in ddim_cfg.values():
        ddim_samplers.append(DDIM_Sampler(diffusion, **sampler_cfg))

    trainer = Trainer(diffusion, ddim_samplers=ddim_samplers, exp_name=args.exp_name,
                      cpu_percentage=args.cpu_percentage, **trainer_cfg)
    if args.load is not None:
        trainer.load(args.load, args.tensorboard, args.no_prev_ddim_setting)
    trainer.train()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='DDPM & DDIM')
    parse.add_argument('-c', '--config', type=str, default='./config/cifar10.yaml')
    parse.add_argument('-l', '--load', type=str, default=None)
    parse.add_argument('-t', '--tensorboard', type=str, default=None)
    parse.add_argument('--exp_name', default=None)
    parse.add_argument('--device', type=str, choices=['cuda', 'cpu','cuda:2'], default='cuda')
    parse.add_argument('--cpu_percentage', type=float, default=0)
    parse.add_argument('--no_prev_ddim_setting', action='store_true')
    args = parse.parse_args()
    main(args)
