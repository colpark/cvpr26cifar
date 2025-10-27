# Masked-Conditioning Generative Models on CIFAR-10

A PyTorch implementation of masked-conditioning generative models with three variants:
- **DDPM Baseline**: Denoising Diffusion Probabilistic Models with discrete timesteps
- **Flow Matching with U-Net**: Rectified Flow with continuous time
- **Flow Matching with Transformers**: DiT and Perceiver IO for zero-shot super-resolution

## Key Features

- ✅ **Masked Conditioning**: Train with sparse pixel observations (20% sparsity)
- ✅ **Split Masks**: Separate conditioning pixels (input) and target pixels (supervision)
- ✅ **Multiple Patterns**: Random pixels, non-overlapping blocks, grid (for SR)
- ✅ **Zero-Shot Super-Resolution**: Train at 32×32, sample at 64×64 or 128×128
- ✅ **Resolution-Agnostic**: DiT (Fourier positional encoding) and Perceiver IO (query-based)
- ✅ **Hard Projection**: Known pixels stay fixed during sampling

## Installation

```bash
# Create environment
conda create -n flowmatch python=3.9
conda activate flowmatch

# Install dependencies
pip install torch torchvision einops pyyaml scipy tqdm

# Clone and navigate
cd flowmatch
```

## Project Structure

```
flowmatch/
├── configs/                    # YAML configuration files
│   ├── cifar10_ddpm.yaml      # DDPM baseline
│   ├── cifar10_fm_unet.yaml   # Flow Matching with U-Net
│   ├── cifar10_fm_dit.yaml    # Flow Matching with DiT
│   └── cifar10_fm_perceiver.yaml  # Flow Matching with Perceiver IO
├── src/
│   ├── datasets/              # Dataset wrappers
│   │   └── cifar10.py
│   ├── models/                # Model architectures
│   │   ├── unet_ddpm.py      # U-Net for DDPM
│   │   ├── unet_fm.py        # U-Net for Flow Matching
│   │   ├── dit_fm.py         # Diffusion Transformer
│   │   └── perceiver_io_fm.py  # Perceiver IO
│   ├── diffusion/             # Diffusion/Flow modules
│   │   ├── ddpm.py           # Gaussian Diffusion
│   │   └── rectified_flow.py  # Rectified Flow
│   ├── sparsity/              # Mask generation
│   │   └── controller.py
│   ├── trainers/              # Training loops
│   │   ├── trainer_base.py
│   │   ├── trainer_ddpm.py
│   │   └── trainer_fm.py
│   ├── utils/                 # Utilities
│   │   ├── time_emb.py       # Time embeddings
│   │   ├── metrics.py        # FID, PSNR, SSIM
│   │   └── viz.py            # Visualization
│   └── cli.py                 # Command-line interface
├── tests/                     # Unit tests
│   ├── test_shapes.py
│   ├── test_sampling_api.py
│   └── test_train_step.py
└── README.md
```

## Quick Start

### 1. Train DDPM Baseline (32×32)

```bash
python -m src.cli --config configs/cifar10_ddpm.yaml
```

**Configuration highlights:**
- Discrete timesteps: T=1000
- Sparsity: 20% (10% conditioning + 10% target)
- Pattern: random pixels
- Training: 200k steps, batch=128

### 2. Train Flow Matching with U-Net (32×32)

```bash
python -m src.cli --config configs/cifar10_fm_unet.yaml
```

**Configuration highlights:**
- Continuous time: t ∈ [0, 1]
- ODE integration: Euler method
- Sampling steps: 50
- Pattern: random pixels (use "grid" for SR)

### 3. Train Flow Matching with DiT (32×32, SR-ready)

```bash
python -m src.cli --config configs/cifar10_fm_dit.yaml
```

**Configuration highlights:**
- Architecture: Transformer with 2D Fourier positional encoding
- Resolution-agnostic: Can sample at any resolution
- Pattern: grid (recommended for super-resolution)
- Batch: 64 (smaller due to transformer memory)

### 4. Train Flow Matching with Perceiver IO (32×32, SR-ready)

```bash
python -m src.cli --config configs/cifar10_fm_perceiver.yaml
```

**Configuration highlights:**
- Architecture: Latent bottleneck with cross-attention
- Naturally handles variable resolutions
- Pattern: grid (recommended for super-resolution)
- Lower learning rate: 1e-4

## Understanding Masked Conditioning

### How Masks Are Formed

The `SparsityController` generates two disjoint masks:

1. **Conditioning Mask** (50% of sparse pixels):
   - Used as input to guide generation
   - Known pixels that model can see
   - Enforced during sampling via hard projection

2. **Target Mask** (50% of sparse pixels):
   - Used for supervision during training
   - Unknown pixels to be predicted
   - Higher loss weight applied to these pixels

**Total sparsity**: If `sparsity=0.2`, then 20% of pixels are sparse:
- 10% are conditioning pixels (input)
- 10% are target pixels (supervision)

### Sparsity Modes

```yaml
sparsity:
  mode: "random_epoch"  # Options: random_epoch, random_iter, fixed_instance, fixed_all
  pattern: "random"     # Options: random, block, grid
  sparsity: 0.2         # Total sparse pixels (0.0 to 1.0)
  block_size: 5         # For "block" pattern
  num_blocks: 6         # For "block" pattern
  grid_stride: 4        # For "grid" pattern
```

**Modes:**
- `random_epoch`: New random masks each epoch, same within epoch
- `random_iter`: New random masks every iteration
- `fixed_instance`: Fixed per-image masks (consistent for each image)
- `fixed_all`: Same mask for all images

**Patterns:**
- `random`: Random pixel locations
- `block`: Non-overlapping square blocks
- `grid`: Stride-based lattice (best for super-resolution)

### Grid Pattern for Super-Resolution

For zero-shot super-resolution, use the **grid pattern**:

```yaml
sparsity:
  pattern: "grid"
  grid_stride: 4  # Conditioning pixels at (0, 4, 8, ...) in both dimensions
```

This creates a regular lattice of known pixels that can be naturally upsampled to higher resolutions.

## Zero-Shot Super-Resolution

### Training for SR

1. **Use grid pattern** in config:
```yaml
sparsity:
  pattern: "grid"
  grid_stride: 4
```

2. **Train at 32×32**:
```bash
python -m src.cli --config configs/cifar10_fm_dit.yaml
```

### Sampling at Higher Resolution

**Option 1: Modify config and run CLI**

Edit `configs/cifar10_fm_dit.yaml`:
```yaml
training:
  target_size: [64, 64]  # or [128, 128] for 4x SR
```

Then sample:
```bash
python -m src.cli --config configs/cifar10_fm_dit.yaml --resume path/to/checkpoint.pt
```

**Option 2: Direct sampling in code**

```python
from src.models.dit_fm import DiTFM
from src.diffusion.rectified_flow import RectifiedFlow
import torch

# Load trained model (32×32)
backbone = DiTFM(...)
model = RectifiedFlow(backbone, image_size=32)
model.load_state_dict(torch.load('checkpoint.pt'))

# Create 32×32 grid mask
sparse_input_32 = torch.randn(1, 3, 32, 32)
mask_32 = torch.zeros(1, 1, 32, 32)
mask_32[:, :, ::4, ::4] = 1.0  # Grid stride 4

# Sample at 64×64 (2x SR)
samples_64 = model.sample(
    batch_size=1,
    sparse_input=sparse_input_32,
    mask=mask_32,
    steps=50,
    target_size=(64, 64),  # Zero-shot 2x SR
    clip=True,
    device='cuda'
)

# Sample at 128×128 (4x SR)
samples_128 = model.sample(
    batch_size=1,
    sparse_input=sparse_input_32,
    mask=mask_32,
    steps=50,
    target_size=(128, 128),  # Zero-shot 4x SR
    clip=True,
    device='cuda'
)
```

### Which Models Support SR?

| Model | Zero-Shot SR | Implementation |
|-------|-------------|----------------|
| DDPM | ❌ | Fixed resolution architecture |
| U-Net FM | ⚠️ | Via upsampling (not ideal) |
| DiT FM | ✅ | 2D Fourier positional encoding |
| Perceiver IO FM | ✅ | Query-based decoder |

## Model Architectures

### DDPM (Discrete Time)

```python
# Time: integer t ∈ {0, 1, ..., T-1}
# Timesteps: T=1000
# Prediction: noise ε
# Embedding: Sinusoidal positional encoding

x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε
```

### Flow Matching (Continuous Time)

```python
# Time: continuous t ∈ [0, 1]
# Prediction: velocity v
# Embedding: Gaussian Fourier features
# Path: Straight line from x_0 to z*

# Standard noise:
z* = z  (where z ~ N(0, I))

# Mask-aware noise (known pixels fixed):
z* = (1 - mask) * z + mask * x_0

# Interpolation:
x_t = (1 - t) * x_0 + t * z*

# Target velocity:
v = z* - x_0

# ODE: dx/dt = v(x_t, t)
```

### DiT (Resolution-Agnostic Transformer)

```python
# Input: Image → Patches (4×4)
# Positional Encoding: 2D Fourier (not learned absolute positions)
# Architecture: Transformer with adaptive layer norm (AdaLN)
# Time Conditioning: Added to tokens via AdaLN

# Key: Fourier encoding works at any resolution
fourier_encode(h, w) = [sin(2π * freqs * y), cos(2π * freqs * y),
                        sin(2π * freqs * x), cos(2π * freqs * x)]
```

### Perceiver IO (Variable Resolution)

```python
# Input: Pixels + 2D Fourier features
# Latents: Fixed-size bottleneck (e.g., 256 latents)
# Processing: Cross-attention (latents attend to inputs)
# Output: Query grid at target resolution attends to latents

# Key: Decouples input resolution from computation
```

## Training Details

### Masked Loss Computation

During training, the loss is weighted to focus on target pixels:

```python
# Heavier weight on target pixels
loss_on_target = mse_loss * target_mask

# Light penalty on known conditioning pixels
loss_on_cond = mse_loss * cond_mask * 0.05

# Combined loss
loss = (loss_on_target + loss_on_cond).sum() / (target_mask + 0.05 * cond_mask).sum()
```

### Hard Projection During Sampling

Known pixels are enforced at every sampling step:

```python
# After each ODE step or denoising step
x_t = mask * sparse_input + (1 - mask) * x_t
```

This ensures generated samples exactly match the conditioning pixels.

### Gradient Clipping

All trainers use gradient clipping for stability:

```yaml
training:
  max_grad_norm: 1.0
```

## Testing

Run all tests:

```bash
# Test model shapes
python tests/test_shapes.py

# Test sampling API (masks, SR, hard projection)
python tests/test_sampling_api.py

# Test training step (loss decreases)
python tests/test_train_step.py
```

## Configuration Reference

### Common Parameters

```yaml
# Model type
model_type: "ddpm" or "flow_matching"

# Model architecture
model:
  type: "unet_ddpm" | "unet_fm" | "dit_fm" | "perceiver_io_fm"
  # ... architecture-specific params

# Dataset
dataset:
  root: "./data"
  batch_size: 128
  augment_horizontal_flip: false

# Sparsity
sparsity:
  mode: "random_epoch"
  pattern: "random"  # or "block" or "grid"
  sparsity: 0.2

# Training
training:
  num_steps: 200000
  lr: 0.0002
  max_grad_norm: 1.0
  clip_sampling: true
  sampling_steps: 50  # For flow matching only
  target_size: null   # For zero-shot SR: [64, 64]

# Paths
save_dir: "./results/experiment_name"
device: "cuda"
seed: 42
```

### DDPM-Specific

```yaml
diffusion:
  timesteps: 1000
  loss_type: "l2"
```

### Flow Matching-Specific

```yaml
flow:
  integration_method: "euler"  # or "heun" or "rk4"

training:
  sampling_steps: 50  # ODE integration steps
  target_size: null   # For SR: [64, 64] or [128, 128]
```

## Expected Results

After training for 200k steps:

### DDPM Baseline (32×32)
- **FID**: ~20-30 (CIFAR-10 baseline)
- **Sample quality**: Coherent images with masked conditioning
- **Training time**: ~24 hours on single GPU

### Flow Matching U-Net (32×32)
- **FID**: ~20-30 (comparable to DDPM)
- **Sampling speed**: Faster (50 steps vs 1000)
- **Training time**: ~24 hours on single GPU

### Flow Matching DiT (32×32 → 64×64 SR)
- **FID at 32×32**: ~25-35
- **SR quality**: Coherent 64×64 images from 32×32 training
- **PSNR improvement**: +2-3 dB over nearest-neighbor upsampling
- **Training time**: ~36 hours on single GPU

### Flow Matching Perceiver IO (32×32 → arbitrary SR)
- **FID at 32×32**: ~30-40
- **SR flexibility**: Can sample at any resolution
- **Training time**: ~48 hours on single GPU
- **Memory**: More efficient than DiT for large resolutions

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```yaml
dataset:
  batch_size: 64  # or 32
```

### Slow Training

Use mixed precision (requires code modification):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(...)
scaler.scale(loss).backward()
```

### Poor Sample Quality

1. **Check conditioning**: Visualize sparse_input and masks
2. **Verify hard projection**: Known pixels should match exactly
3. **Increase training steps**: 200k may not be enough for complex patterns
4. **Adjust sparsity**: Try 0.3 or 0.4 for easier task

### SR Not Working

1. **Use grid pattern**: Essential for zero-shot SR
2. **Check Fourier encoding**: DiT and Perceiver IO should use relative positions
3. **Verify upsampling**: Masks and sparse_input are upsampled correctly

## Citation

If you use this code, please cite:

```bibtex
@article{flow_matching_sr,
  title={Zero-Shot Super-Resolution with Masked-Conditioning Flow Matching},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- DDPM implementation inspired by [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- Flow Matching based on [Rectified Flow](https://arxiv.org/abs/2209.03003)
- DiT architecture from [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- Perceiver IO from [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)
