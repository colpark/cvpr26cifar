"""
Test sampling API: masks, hard projection, and zero-shot super-resolution.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.unet_ddpm import UNetDDPM
from src.models.unet_fm import UNetFM
from src.models.dit_fm import DiTFM
from src.models.perceiver_io_fm import PerceiverIOFM
from src.diffusion.ddpm import GaussianDiffusion
from src.diffusion.rectified_flow import RectifiedFlow


def test_ddpm_sampling_with_masks():
    """Test DDPM sampling with sparse conditioning masks."""
    print("Testing DDPM sampling with masks...")

    backbone = UNetDDPM(
        dim=64,
        image_size=32,
        dim_multiply=(1, 2, 4),
        channel=3,
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        groups=8
    )

    model = GaussianDiffusion(
        model=backbone,
        image_size=32,
        timesteps=100,  # Fewer steps for testing
        loss_type='l2'
    )

    B = 2
    sparse_input = torch.randn(B, 3, 32, 32)
    mask = torch.zeros(B, 1, 32, 32)
    mask[:, :, ::4, ::4] = 1.0  # Grid pattern

    samples = model.sample(
        batch_size=B,
        sparse_input=sparse_input,
        mask=mask,
        clip=True,
        device='cpu'
    )

    assert samples.shape == (B, 3, 32, 32), f"Expected shape {(B, 3, 32, 32)}, got {samples.shape}"

    # Verify hard projection: known pixels should match sparse_input
    known_pixels = samples * mask
    expected_known = sparse_input * mask
    diff = torch.abs(known_pixels - expected_known).mean()
    assert diff < 1e-5, f"Hard projection failed: diff={diff:.6f}"

    print("✓ DDPM sampling with masks and hard projection correct")


def test_flow_matching_sampling_with_masks():
    """Test Flow Matching sampling with sparse conditioning masks."""
    print("Testing Flow Matching sampling with masks...")

    backbone = UNetFM(
        dim=64,
        image_size=32,
        dim_multiply=(1, 2, 4),
        channel=3,
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        groups=8,
        time_emb_scale=16.0
    )

    model = RectifiedFlow(
        model=backbone,
        image_size=32
    )

    B = 2
    sparse_input = torch.randn(B, 3, 32, 32)
    mask = torch.zeros(B, 1, 32, 32)
    mask[:, :, ::4, ::4] = 1.0  # Grid pattern

    samples = model.sample(
        batch_size=B,
        sparse_input=sparse_input,
        mask=mask,
        steps=10,  # Fewer ODE steps for testing
        method='euler',
        clip=True,
        device='cpu'
    )

    assert samples.shape == (B, 3, 32, 32), f"Expected shape {(B, 3, 32, 32)}, got {samples.shape}"

    # Verify hard projection: known pixels should match sparse_input
    known_pixels = samples * mask
    expected_known = sparse_input * mask
    diff = torch.abs(known_pixels - expected_known).mean()
    assert diff < 1e-5, f"Hard projection failed: diff={diff:.6f}"

    print("✓ Flow Matching sampling with masks and hard projection correct")


def test_zero_shot_super_resolution_unet():
    """Test zero-shot super-resolution with Flow Matching U-Net."""
    print("Testing zero-shot SR with UNetFM...")

    backbone = UNetFM(
        dim=64,
        image_size=32,
        dim_multiply=(1, 2, 4),
        channel=3,
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        groups=8,
        time_emb_scale=16.0
    )

    model = RectifiedFlow(
        model=backbone,
        image_size=32
    )

    B = 1
    # Training resolution: 32x32, sparse pattern
    sparse_input_32 = torch.randn(B, 3, 32, 32)
    mask_32 = torch.zeros(B, 1, 32, 32)
    mask_32[:, :, ::4, ::4] = 1.0  # Grid pattern

    # Sample at 2x resolution (64x64)
    target_size = (64, 64)
    samples_64 = model.sample(
        batch_size=B,
        sparse_input=sparse_input_32,
        mask=mask_32,
        steps=10,
        target_size=target_size,
        method='euler',
        clip=True,
        device='cpu'
    )

    assert samples_64.shape == (B, 3, 64, 64), f"Expected shape {(B, 3, 64, 64)}, got {samples_64.shape}"

    # Verify upsampled sparse pixels are preserved at their locations
    # After upsampling, grid pixels at (i*4, j*4) in 32x32 should map to (i*8, j*8) in 64x64
    sparse_input_64 = torch.nn.functional.interpolate(
        sparse_input_32, size=target_size, mode='nearest'
    )
    mask_64 = torch.nn.functional.interpolate(
        mask_32, size=target_size, mode='nearest'
    )

    known_pixels = samples_64 * mask_64
    expected_known = sparse_input_64 * mask_64
    diff = torch.abs(known_pixels - expected_known).mean()
    assert diff < 1e-5, f"Hard projection failed at 2x resolution: diff={diff:.6f}"

    print("✓ Zero-shot SR with UNetFM correct (32→64)")


def test_zero_shot_super_resolution_dit():
    """Test zero-shot super-resolution with DiT (resolution-agnostic)."""
    print("Testing zero-shot SR with DiTFM...")

    model_wrapper = DiTFM(
        patch_size=4,
        dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        channel=3,
        dropout=0.0,
        window_size=8
    )

    model = RectifiedFlow(
        model=model_wrapper,
        image_size=32
    )

    B = 1
    # Training resolution: 32x32
    sparse_input_32 = torch.randn(B, 3, 32, 32)
    mask_32 = torch.zeros(B, 1, 32, 32)
    mask_32[:, :, ::4, ::4] = 1.0

    # Sample at 2x resolution (64x64) - DiT handles naturally
    target_size = (64, 64)
    samples_64 = model.sample(
        batch_size=B,
        sparse_input=sparse_input_32,
        mask=mask_32,
        steps=10,
        target_size=target_size,
        method='euler',
        clip=True,
        device='cpu'
    )

    assert samples_64.shape == (B, 3, 64, 64), f"Expected shape {(B, 3, 64, 64)}, got {samples_64.shape}"

    print("✓ Zero-shot SR with DiTFM correct (32→64)")


def test_zero_shot_super_resolution_perceiver():
    """Test zero-shot super-resolution with Perceiver IO (naturally handles variable resolution)."""
    print("Testing zero-shot SR with PerceiverIOFM...")

    model_wrapper = PerceiverIOFM(
        channel=3,
        latent_dim=128,
        num_latents=64,
        depth=2,
        num_heads=4,
        head_dim=32,
        input_fourier_features=32,
        query_fourier_features=32
    )

    model = RectifiedFlow(
        model=model_wrapper,
        image_size=32
    )

    B = 1
    # Training resolution: 32x32
    sparse_input_32 = torch.randn(B, 3, 32, 32)
    mask_32 = torch.zeros(B, 1, 32, 32)
    mask_32[:, :, ::4, ::4] = 1.0

    # Sample at 48x48 (Perceiver handles any resolution)
    target_size = (48, 48)
    samples_48 = model.sample(
        batch_size=B,
        sparse_input=sparse_input_32,
        mask=mask_32,
        steps=10,
        target_size=target_size,
        method='euler',
        clip=True,
        device='cpu'
    )

    assert samples_48.shape == (B, 3, 48, 48), f"Expected shape {(B, 3, 48, 48)}, got {samples_48.shape}"

    print("✓ Zero-shot SR with PerceiverIOFM correct (32→48)")


def test_sampling_without_masks():
    """Test sampling without sparse conditioning (unconditional generation)."""
    print("Testing unconditional sampling...")

    backbone = UNetFM(
        dim=64,
        image_size=32,
        dim_multiply=(1, 2, 4),
        channel=3,
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        groups=8,
        time_emb_scale=16.0
    )

    model = RectifiedFlow(
        model=backbone,
        image_size=32
    )

    B = 2
    samples = model.sample(
        batch_size=B,
        sparse_input=None,
        mask=None,
        steps=10,
        clip=True,
        device='cpu'
    )

    assert samples.shape == (B, 3, 32, 32), f"Expected shape {(B, 3, 32, 32)}, got {samples.shape}"

    print("✓ Unconditional sampling correct")


if __name__ == '__main__':
    print("="*60)
    print("Running sampling API tests...")
    print("="*60)

    test_ddpm_sampling_with_masks()
    test_flow_matching_sampling_with_masks()
    test_zero_shot_super_resolution_unet()
    test_zero_shot_super_resolution_dit()
    test_zero_shot_super_resolution_perceiver()
    test_sampling_without_masks()

    print("="*60)
    print("All sampling API tests passed!")
    print("="*60)
