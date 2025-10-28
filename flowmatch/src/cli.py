"""
Command-line interface for training masked-conditioning generative models.
"""
import argparse
import yaml
import torch
import random
import numpy as np

from .datasets.cifar10 import get_cifar10_dataloader
from .datasets.cifar10_coordinate import get_cifar10_coordinate_dataloader
from .sparsity.controller import SparsityController
from .models.unet_ddpm import UNetDDPM
from .models.unet_fm import UNetFM
from .models.dit_fm import DiTFM
from .models.perceiver_io_fm import PerceiverIOFM
from .models.perceiver_io_fm_v2 import PerceiverIOFMV2
from .models.coordinate_fm import CoordinateBasedFM
from .models.perceiver_coordinate_fm import PerceiverCoordinateFM
from .models.mamba_coordinate_fm import MambaCoordinateFM
from .diffusion.ddpm import GaussianDiffusion
from .diffusion.rectified_flow import RectifiedFlow
from .trainers.trainer_ddpm import DDPMTrainer
from .trainers.trainer_fm import FMTrainer
from .trainers.trainer_coordinate_fm import CoordinateFMTrainer


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config, device):
    """
    Build model based on configuration.

    Returns:
        model: Either GaussianDiffusion or RectifiedFlow
    """
    model_type = config['model']['type']
    model_config = config['model']

    # Build backbone network
    if model_type == 'unet_ddpm':
        backbone = UNetDDPM(
            dim=model_config['dim'],
            image_size=model_config['image_size'],
            dim_multiply=tuple(model_config['dim_multiply']),
            channel=model_config['channel'],
            num_res_blocks=model_config['num_res_blocks'],
            attn_resolutions=tuple(model_config['attn_resolutions']),
            dropout=model_config['dropout'],
            groups=model_config['groups']
        )
    elif model_type == 'unet_fm':
        backbone = UNetFM(
            dim=model_config['dim'],
            image_size=model_config['image_size'],
            dim_multiply=tuple(model_config['dim_multiply']),
            channel=model_config['channel'],
            num_res_blocks=model_config['num_res_blocks'],
            attn_resolutions=tuple(model_config['attn_resolutions']),
            dropout=model_config['dropout'],
            groups=model_config['groups'],
            time_emb_scale=model_config.get('time_emb_scale', 16.0)
        )
    elif model_type == 'dit_fm':
        backbone = DiTFM(
            patch_size=model_config['patch_size'],
            dim=model_config['dim'],
            depth=model_config['depth'],
            num_heads=model_config['num_heads'],
            mlp_ratio=model_config['mlp_ratio'],
            channel=model_config['channel'],
            dropout=model_config['dropout'],
            window_size=model_config['window_size']
        )
    elif model_type == 'perceiver_io_fm':
        backbone = PerceiverIOFM(
            channel=model_config['channel'],
            latent_dim=model_config['latent_dim'],
            num_latents=model_config['num_latents'],
            depth=model_config['depth'],
            num_heads=model_config['num_heads'],
            head_dim=model_config['head_dim'],
            input_fourier_features=model_config['input_fourier_features'],
            query_fourier_features=model_config['query_fourier_features']
        )
    elif model_type == 'perceiver_io_fm_v2':
        backbone = PerceiverIOFMV2(
            channel=model_config['channel'],
            latent_dim=model_config['latent_dim'],
            num_latents=model_config['num_latents'],
            depth=model_config['depth'],
            num_heads=model_config['num_heads'],
            head_dim=model_config['head_dim'],
            input_fourier_features=model_config['input_fourier_features'],
            query_fourier_features=model_config['query_fourier_features']
        )
    elif model_type == 'coordinate_fm':
        backbone = CoordinateBasedFM(
            channel=model_config['channel'],
            dim=model_config['dim'],
            depth=model_config['depth'],
            num_heads=model_config['num_heads'],
            mlp_ratio=model_config['mlp_ratio'],
            num_fourier_freqs=model_config['num_fourier_freqs'],
            fourier_scale=model_config['fourier_scale'],
            dropout=model_config['dropout']
        )
    elif model_type == 'perceiver_coordinate_fm':
        backbone = PerceiverCoordinateFM(
            channel=model_config['channel'],
            latent_dim=model_config['latent_dim'],
            num_latents=model_config['num_latents'],
            depth=model_config['depth'],
            num_heads=model_config['num_heads'],
            head_dim=model_config['head_dim'],
            fourier_mapping_size=model_config['fourier_mapping_size'],
            fourier_scale=model_config['fourier_scale'],
            dropout=model_config['dropout']
        )
    elif model_type == 'mamba_coordinate_fm':
        backbone = MambaCoordinateFM(
            channel=model_config['channel'],
            dim=model_config['dim'],
            depth=model_config['depth'],
            d_state=model_config['d_state'],
            d_conv=model_config['d_conv'],
            expand=model_config['expand'],
            fourier_mapping_size=model_config['fourier_mapping_size'],
            fourier_scale=model_config['fourier_scale'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Wrap in diffusion/flow model
    if config['model_type'] == 'ddpm':
        model = GaussianDiffusion(
            backbone,
            image_size=model_config['image_size'],
            timesteps=config['diffusion']['timesteps'],
            loss_type=config['diffusion']['loss_type']
        )
    elif config['model_type'] == 'flow_matching':
        model = RectifiedFlow(
            backbone,
            image_size=model_config['image_size']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    return model.to(device)


def build_trainer(model, train_loader, config, sparsity_controller):
    """Build trainer based on configuration."""
    training_config = config['training']
    device = config['device']
    save_dir = config['save_dir']
    model_type = config['model']['type']

    optimizer_config = {
        'lr': training_config['lr'],
        'weight_decay': training_config.get('weight_decay', 0.0),
        'betas': tuple(training_config.get('betas', [0.9, 0.999]))
    }

    # Check if using coordinate-based model
    if model_type in ['coordinate_fm', 'perceiver_coordinate_fm', 'mamba_coordinate_fm']:
        trainer = CoordinateFMTrainer(
            flow=model,
            train_loader=train_loader,
            optimizer_config=optimizer_config,
            device=device,
            save_dir=save_dir,
            sampling_steps=training_config.get('sampling_steps', 50),
            clip_sampling=training_config.get('clip_sampling', True),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            eval_resolutions=training_config.get('eval_resolutions', [32, 64, 96])
        )
    elif config['model_type'] == 'ddpm':
        trainer = DDPMTrainer(
            diffusion=model,
            train_loader=train_loader,
            optimizer_config=optimizer_config,
            device=device,
            save_dir=save_dir,
            sparsity_controller=sparsity_controller,
            clip_sampling=training_config.get('clip_sampling', True),
            max_grad_norm=training_config.get('max_grad_norm', 1.0)
        )
    elif config['model_type'] == 'flow_matching':
        trainer = FMTrainer(
            flow=model,
            train_loader=train_loader,
            optimizer_config=optimizer_config,
            device=device,
            save_dir=save_dir,
            sparsity_controller=sparsity_controller,
            sampling_steps=training_config.get('sampling_steps', 50),
            clip_sampling=training_config.get('clip_sampling', True),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            target_size=training_config.get('target_size', None)
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description='Train masked-conditioning generative models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Model type: {config['model_type']}, Model architecture: {config['model']['type']}")

    # Set seed
    set_seed(config.get('seed', 42))

    # Setup device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build dataloader
    dataset_config = config['dataset']
    dataset_name = dataset_config.get('name', 'cifar10')

    if dataset_name == 'cifar10_coordinate':
        # Coordinate-based dataset
        train_loader = get_cifar10_coordinate_dataloader(
            root=dataset_config['root'],
            train=dataset_config['train'],
            batch_size=dataset_config['batch_size'],
            input_ratio=dataset_config.get('input_ratio', 0.2),
            target_ratio=dataset_config.get('target_ratio', 1.0),
            image_size=dataset_config.get('image_size', 32),
            num_workers=dataset_config.get('num_workers', 4),
            download=True,
            seed=config.get('seed', 42)
        )
        print(f"Loaded CIFAR-10 coordinate dataset: {len(train_loader.dataset)} images")
        print(f"Input ratio: {dataset_config.get('input_ratio', 0.2):.1%}, "
              f"Target ratio: {dataset_config.get('target_ratio', 1.0):.1%}")
        sparsity_controller = None  # Not used for coordinate-based
    else:
        # Standard grid-based dataset
        train_loader = get_cifar10_dataloader(
            root=dataset_config['root'],
            train=dataset_config['train'],
            batch_size=dataset_config['batch_size'],
            augment_horizontal_flip=dataset_config.get('augment_horizontal_flip', False),
            num_workers=dataset_config.get('num_workers', 4)
        )
        print(f"Loaded CIFAR-10 dataset: {len(train_loader.dataset)} images")

        # Build sparsity controller
        sparsity_config = config['sparsity']
        sparsity_controller = SparsityController(
            image_size=config['model']['image_size'],
            mode=sparsity_config['mode'],
            pattern=sparsity_config['pattern'],
            sparsity=sparsity_config['sparsity'],
            block_size=sparsity_config.get('block_size', 5),
            num_blocks=sparsity_config.get('num_blocks', 6),
            grid_stride=sparsity_config.get('grid_stride', 2)
        )
        print(f"Sparsity mode: {sparsity_config['mode']}, pattern: {sparsity_config['pattern']}")

    # Build model
    model = build_model(config, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Build trainer
    trainer = build_trainer(model, train_loader, config, sparsity_controller)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    training_config = config['training']
    trainer.train(
        num_steps=training_config['num_steps'],
        log_every=training_config.get('log_every', 100),
        sample_every=training_config.get('sample_every', 1000),
        save_every=training_config.get('save_every', 5000)
    )

    print("Training completed!")


if __name__ == '__main__':
    main()
