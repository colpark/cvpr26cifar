# Perceiver IO V2 Improvements for Super-Resolution

## Problem with V1

The original Perceiver IO had issues with zero-shot super-resolution:

1. **No Target Resolution Support**: V1 model's forward() didn't accept target_size, always outputting same resolution as input
2. **Latent Spatial Unawareness**: Latents were initialized randomly without spatial structure
3. **Query-Latent Mismatch**: When querying at 64×64 but latents processed 32×32 inputs, cross-attention struggled to generalize

## V2 Improvements

### 1. Variable Resolution Support

```python
def forward(self, x, t, sparse_input=None, mask=None, target_size=None):
    # V2 supports different input and output resolutions
    H_in, W_in = x.shape[2:]
    H_out, W_out = target_size if target_size else (H_in, W_in)
```

**How it works:**
- Input encoded at 32×32: 1024 input tokens with Fourier PE
- Latents process input: 256 latent vectors (16×16 grid)
- Output queries at 64×64: 4096 query tokens with Fourier PE
- Cross-attention maps latents → high-res queries

### 2. Spatial Latent Structure

```python
# V1: Random latent initialization
self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

# V2: Spatial grid structure with positional encoding
latent_h = latent_w = int(math.sqrt(num_latents))  # 16×16 grid
self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
self.latent_pos_embed = nn.Parameter(torch.randn(num_latents, latent_dim))
```

**Benefits:**
- Latents arranged in 16×16 spatial grid
- Learned spatial positional encoding preserves structure
- Better generalization to higher resolutions

### 3. Resolution-Normalized Fourier Features

V2 uses the same Fourier feature generation for inputs and queries, ensuring consistency:

```python
def _fourier_features(self, coords, scale):
    # coords in [-1, 1] regardless of resolution
    # Normalized coordinates enable resolution-agnostic learning
    freq_bands = torch.linspace(1.0, scale/2, scale // 2)
    # sin/cos applied to normalized coordinates
```

**Key insight:** By using normalized coordinates [-1, 1], the same frequency bands work across resolutions.

### 4. RectifiedFlow Integration

V2 models are automatically detected and receive target_size:

```python
# In rectified_flow.py:
import inspect
model_params = inspect.signature(self.model.forward).parameters
if 'target_size' in model_params:
    v = self.model(x, t, sparse_input, mask, target_size=(H, W))
```

## Architecture Comparison

| Component | V1 | V2 |
|-----------|----|----|
| Target resolution support | ❌ No | ✅ Yes |
| Latent structure | Random | 16×16 spatial grid |
| Latent positional encoding | Only time | Time + spatial |
| Query generation | Same as input | Variable resolution |
| SR mechanism | Implicit via upsampling | Explicit via queries |

## Model Size Comparison

**V2 variants:**

```yaml
perceiver_v2 (standard):
  latent_dim: 256
  num_latents: 256 (16×16 grid)
  depth: 4
  Parameters: ~8-10M
  Batch size: 128
  Training speed: ~2x faster than V1

perceiver_v2_large:
  latent_dim: 512
  num_latents: 256
  depth: 6
  Parameters: ~30M
  For best quality (similar to V1)
```

## Usage

```bash
# Train V2 model (efficient version)
python -m src.cli --config configs/cifar10_fm_perceiver_v2.yaml

# Original V1 (preserved for comparison)
python -m src.cli --config configs/cifar10_fm_perceiver.yaml

# Light variants (fastest)
python -m src.cli --config configs/cifar10_fm_perceiver_light.yaml
python -m src.cli --config configs/cifar10_fm_perceiver_tiny.yaml
```

## Expected Results

**32×32 conditional generation:**
- V1 and V2 should perform similarly
- Both learn masked conditional generation

**64×64 super-resolution:**
- V1: Poor quality, artifacts (no explicit SR support)
- V2: Better quality, proper spatial structure

**96×96 super-resolution:**
- V1: Severe artifacts
- V2: Reasonable quality, maintains structure

## Why V2 Should Work Better

1. **Explicit Resolution Handling**: Model processes different resolutions in forward pass
2. **Spatial Awareness**: 16×16 latent grid preserves spatial structure
3. **Query-Based Decoding**: Queries at target resolution attend to latents
4. **Normalized Coordinates**: [-1, 1] coordinates work across resolutions

## Limitations

- Still trained only at 32×32, so SR is zero-shot extrapolation
- Very high resolutions (>128×128) may still struggle
- Sparse conditioning needs to be upsampled to target resolution
- Best results with training_steps > 100K

## Debugging

If SR quality is still poor:

1. **Check latent structure**: `num_latents` must be perfect square (256 = 16²)
2. **Verify target_size passing**: Check logs for target_size in model calls
3. **Increase training**: SR quality improves significantly after 50K+ steps
4. **Try larger model**: Use larger latent_dim (512) and depth (6) for best quality
5. **Visualize step 0**: Check untrained model baseline with step 0 samples
