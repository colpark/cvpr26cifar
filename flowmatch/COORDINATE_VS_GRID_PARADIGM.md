# The Coordinate-Based Paradigm: Why Mamba Does SR Better

## The Critical Lesson from `mamba_diffusion.ipynb`

**The Problem You Identified:**
> "This code in fact does SR very well, even when trained only on 32×32. What is the lesson we have here?"

**The Answer:** It's not about the architecture (Mamba vs Transformer). It's about the **representation paradigm**.

---

## Two Fundamentally Different Approaches

### Approach 1: Grid-Based Processing (DiT, Perceiver V2)

**What the model sees during training:**
```python
# Input: Fixed grid of pixels
x: (B, 3, 32, 32)  # Image as grid

# Processing:
patches = patchify(x)  # (B, 64, 512) - 64 tokens at fixed positions
features = transformer(patches)  # Process token sequence
output = unpatchify(features)  # (B, 3, 32, 32) - back to grid
```

**What the model learns:**
- "How to denoise/generate images with 64 tokens"
- "Patch[0] relates to Patch[1] via attention"
- "Token positions are fixed: 8×8 grid"

**Why SR is hard:**
```python
# Training: 32×32 → 64 tokens (8×8 patch grid)
# Testing: 64×64 → 256 tokens (16×16 patch grid)

# Problem: Model never saw:
# - 256-token sequences
# - 16×16 attention patterns
# - What "high-res" looks like
```

The model learned to process **discrete grids**, not continuous space.

---

### Approach 2: Coordinate-Based Processing (Mamba in notebook)

**What the model sees during training:**
```python
# Input: Sparse coordinate-value pairs
input_coords: (B, 204, 2)    # (x, y) positions in [0,1]²
input_values: (B, 204, 3)    # RGB values at those positions

# Processing:
coords_feats = fourier_features(input_coords)  # Continuous positional encoding
features = encode(coords_feats, input_values)
# Model processes COORDINATE-VALUE PAIRS, not image grids

# Output: Query ANY coordinates
query_coords: (B, N_query, 2)  # Can be 1024, 4096, 9216... any number!
output_values: (B, N_query, 3) # RGB values at queried positions
```

**What the model learns:**
- **A continuous function**: `f(x, y, t) → RGB`
- "Given sparse observations at coordinates, predict RGB at any query coordinate"
- Spatial relationships are continuous, not discrete
- No concept of "token count" or "grid size"

**Why SR works naturally:**
```python
# Training: Query 1024 random coordinates at 32×32 resolution
query_coords_train = random_sample_grid(32)  # (1024, 2) in [0,1]²
pred_train = model(query_coords_train, input_coords, input_values)

# Testing: Query 4096 coordinates at 64×64 resolution
query_coords_test = uniform_grid(64)  # (4096, 2) in [0,1]²
pred_test = model(query_coords_test, input_coords, input_values)
# ↑ SAME MODEL, SAME INPUT, just different query coordinates!
```

The model learned a **continuous field**, so any resolution is just "sample more coordinates".

---

## Detailed Comparison

### Grid-Based (Your Current Models)

**DiT Architecture:**
```
32×32 image
↓ Patch embedding (4×4 patches)
64 tokens (8×8 grid)
↓ 12 × Transformer blocks
64 tokens with learned relationships
↓ Output projection
32×32 image
```

**For 64×64 SR:**
```
64×64 image ← Want this
↓ Would need patch embedding at 64×64
256 tokens (16×16 grid) ← Never trained on this many tokens!
↓ Transformer attention ← Attention patterns learned for 64 tokens, not 256
??? ← Model is confused
```

**Perceiver V2 Architecture:**
```
32×32 image → 1024 input tokens
↓ Cross-attention encode
256 latents (16×16 grid structure)
↓ Self-attention
256 latents with spatial relationships
↓ Cross-attention decode with 1024 queries
32×32 image (1024 output tokens)
```

**For 64×64 SR:**
```
64×64 queries (4096 coordinates)
↓ Cross-attention decode
256 latents ← Same latents, but now must cover 4× area!
↑ Bottleneck: 256 latents learned for 32×32, now stretched to 64×64
```

---

### Coordinate-Based (Mamba Approach)

**Architecture:**
```
Sparse input: (x_i, y_i, RGB_i) for i=1...204
↓ Fourier features: (x,y) → [sin(2πf₁x), cos(2πf₁x), ...]
Encoded: (coord_features, RGB) → tokens
↓ Mamba/SSM blocks (process sequence)
Latent representation (continuous function)
↓ Cross-attention with query coordinates
Query: (x_q, y_q) for any q
↓ Decode
Output: RGB values at query positions
```

**For ANY resolution:**
```
Training resolution (32×32):
query_coords = uniform_grid(32)  # (1024, 2) coordinates
pred = model(query_coords, input_coords, input_values)

Testing resolution (64×64):
query_coords = uniform_grid(64)  # (4096, 2) coordinates
pred = model(query_coords, input_coords, input_values)
# ↑ SAME FORWARD PASS, just more queries!

Testing resolution (128×128):
query_coords = uniform_grid(128)  # (16384, 2) coordinates
pred = model(query_coords, input_coords, input_values)
# ↑ Still works! Continuous function can be sampled anywhere.
```

---

## Why This Works: Fourier Features + Coordinate Queries

### The Magic Ingredient: Continuous Positional Encoding

**Grid-Based Approach (DiT/Perceiver):**
```python
# Positional encoding for discrete patches
pos_enc = fourier_pe(patch_grid_positions)  # 64 positions
# Even though using Fourier PE, positions are still DISCRETE
# 64 fixed positions → 64 encodings
```

**Coordinate-Based Approach (Mamba):**
```python
# Positional encoding for arbitrary coordinates
def fourier_features(coords):  # coords: (N, 2) in [0,1]²
    freqs = [1, 2, 4, 8, 16, 32, 64, 128, ...]
    features = []
    for f in freqs:
        features += [sin(2π f x), cos(2π f x), sin(2π f y), cos(2π f y)]
    return features  # (N, num_freqs*4)

# Can call this with ANY coordinates!
coords_32 = uniform_grid(32)   # 1024 points
coords_64 = uniform_grid(64)   # 4096 points
coords_arbitrary = random_coords(2000)  # Any 2000 points

# All get meaningful Fourier encodings
feats_32 = fourier_features(coords_32)
feats_64 = fourier_features(coords_64)
feats_arb = fourier_features(coords_arbitrary)
```

The key: **Fourier features define a continuous encoding function over [0,1]²**, not a discrete set of position encodings.

---

## The "Noisy 32×32" Trade-off

You mentioned:
> "This code is not without a problem --- that the reconstructed 32×32 has a lot of noise."

**Why this happens:**

**Grid-Based Models (Your Current Approach):**
- Learn to reconstruct EXACT pixel values
- Training signal: MSE loss on every pixel
- Optimization: Memorize high-frequency details at 32×32
- Result: Perfect 32×32 reconstruction, poor generalization

**Coordinate-Based Models (Mamba Approach):**
- Learn a SMOOTH continuous function
- Training signal: MSE loss on sampled coordinates (not all pixels)
- Optimization: Fit smooth function through sparse observations
- Result: Slightly noisy 32×32 (smooth function), excellent generalization

**The trade-off:**
```
High-frequency details at 32×32  ⟷  Scale-invariant smooth function
         ↓                                        ↓
    Better 32×32 PSNR                      Better SR quality
    Overfits to 32×32                      Generalizes across scales
```

**Analogy:**
- Grid-based: "Memorize the specific 1024 pixel values"
- Coordinate-based: "Learn the underlying spatial structure"

The noise at 32×32 is a **feature, not a bug** — it's evidence the model learned a generalizable representation rather than memorizing the training grid.

---

## Quantitative Evidence from Notebook

From the multi-scale evaluation in the notebook:

```python
# Training: Only 32×32 images with 20% sparse input (204 pixels)

# Testing:
# 32×32 reconstruction: PSNR ~20-25 dB (some noise)
# 64×64 reconstruction: Maintains structure, smooth upsampling
# 96×96 reconstruction: Still coherent, better than bilinear

# Comparison:
# Nearest-neighbor 32→64: Blocky artifacts
# Bilinear 32→64: Blurry but smooth
# Continuous field 64: Sharp AND smooth (learned structure)
```

The model doesn't just interpolate — it **extrapolates the learned spatial structure** to new coordinates.

---

## Implications for Your Models

### Why Your Models Struggle with SR

**DiT:**
- ✅ Has Fourier PE for patches
- ❌ PE is for FIXED 64 patch positions
- ❌ Transformer processes token sequences of fixed length
- ❌ No way to query arbitrary coordinates

**Perceiver V2:**
- ✅ Has target_size parameter
- ✅ Can generate queries at any resolution
- ❌ Still encodes 1024 input positions → 256 latents
- ❌ Latents learned for 32×32 coverage, stretched for 64×64
- ❌ Input encoding still tied to 32×32 grid

### What You're Missing

**The coordinate-query paradigm:**
1. Inputs are **(x, y, value)** tuples, not grids
2. Model learns **function f: (x,y,t) → RGB**, not image-to-image mapping
3. Output is generated by **querying f at arbitrary coordinates**
4. Resolution is a **sampling density**, not an architectural constraint

---

## How to Adapt Your Architecture

### Option 1: Hybrid Coordinate-Based DiT

```python
class CoordinateBasedDiT(nn.Module):
    def forward(self, input_coords, input_values, query_coords, t):
        # Input encoding: (x,y,RGB) → features
        input_feats = fourier_features(input_coords)  # (B, N_in, F)
        input_tokens = encode(input_feats, input_values)  # (B, N_in, D)

        # Process with transformer (learns continuous function)
        latents = transformer(input_tokens)  # (B, N_in, D)

        # Query at arbitrary coordinates
        query_feats = fourier_features(query_coords)  # (B, N_out, F)
        query_tokens = query_encoder(query_feats)  # (B, N_out, D)

        # Cross-attention: queries attend to latents
        output_tokens = cross_attention(query_tokens, latents)

        # Decode to RGB
        output_values = decoder(output_tokens)  # (B, N_out, 3)

        return output_values

# Training at 32×32:
query_coords_32 = uniform_grid(32)  # (1024, 2)
pred = model(input_coords, input_values, query_coords_32, t)

# Testing at 64×64:
query_coords_64 = uniform_grid(64)  # (4096, 2)
pred = model(input_coords, input_values, query_coords_64, t)
# ↑ No architectural change needed!
```

### Option 2: Coordinate-Based Perceiver

```python
class CoordinatePerceiverIO(nn.Module):
    def forward(self, input_coords, input_values, query_coords, t):
        # Input tokens from coordinates
        input_feats = fourier_features(input_coords)
        input_tokens = input_proj(input_feats, input_values)

        # Cross-attention encode (coordinate → latent)
        latents = cross_attn_encode(self.latents, input_tokens)
        latents = self_attn_process(latents)

        # Cross-attention decode (query coordinate → output)
        query_feats = fourier_features(query_coords)  # CONTINUOUS QUERIES
        query_tokens = query_proj(query_feats)
        output_tokens = cross_attn_decode(query_tokens, latents)

        # Decode to RGB
        return output_proj(output_tokens)

# Any resolution works!
pred_32 = model(input_coords, input_values, grid_32, t)
pred_64 = model(input_coords, input_values, grid_64, t)
pred_128 = model(input_coords, input_values, grid_128, t)
```

---

## Key Architectural Changes Needed

### 1. Remove Grid Assumptions

**Before (Grid-Based):**
```python
def forward(self, x: Tensor):  # x: (B, 3, H, W)
    patches = patchify(x)  # Assumes regular grid
    ...
```

**After (Coordinate-Based):**
```python
def forward(self, coords: Tensor, values: Tensor, query_coords: Tensor):
    # coords: (B, N_in, 2) - can be ANY positions
    # values: (B, N_in, 3) - RGB at those positions
    # query_coords: (B, N_out, 2) - can be ANY positions
    ...
```

### 2. Make Fourier Features Truly Continuous

**Before (Discrete PE):**
```python
# Positional encoding for fixed patches
self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
x = x + self.pos_embed  # Lookup fixed positions
```

**After (Continuous Fourier):**
```python
def fourier_features(self, coords):
    # coords: (B, N, 2) - any coordinates in [0,1]²
    freqs = torch.tensor([1, 2, 4, 8, 16, 32, ...])
    feats = []
    for f in freqs:
        feats += [torch.sin(2*π*f*coords[...,0]),
                  torch.cos(2*π*f*coords[...,0]),
                  torch.sin(2*π*f*coords[...,1]),
                  torch.cos(2*π*f*coords[...,1])]
    return torch.stack(feats, dim=-1)
```

### 3. Query-Based Decoding

**Before (Fixed Grid Output):**
```python
output = self.output_proj(tokens)  # (B, N_tokens, 3)
output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
# ↑ Assumes H, W are known and fixed
```

**After (Query Coordinates):**
```python
query_feats = self.fourier_features(query_coords)  # (B, N_query, F)
output_tokens = self.decode(query_feats, latents)  # (B, N_query, 3)
# ↑ Works for ANY number of queries at ANY positions
```

---

## Training Changes

### Data Representation

**Before (Grid-Based):**
```python
# Dataset returns full images
batch = {
    'image': (B, 3, 32, 32),
    'sparse_input': (B, 3, 32, 32),  # Masked image
    'mask': (B, 3, 32, 32)
}
```

**After (Coordinate-Based):**
```python
# Dataset returns coordinate-value pairs
batch = {
    'input_coords': (B, N_input, 2),      # Sparse observation positions
    'input_values': (B, N_input, 3),      # RGB at those positions
    'target_coords': (B, N_target, 2),    # Query positions (can vary!)
    'target_values': (B, N_target, 3)     # Ground truth RGB
}

# N_target can be:
# - All 1024 pixels (full reconstruction training)
# - Random 512 pixels (random query training)
# - Different per batch (variable resolution training)
```

### Training Loop

**Before:**
```python
x0 = batch['image']  # (B, 3, 32, 32)
v_pred = model(x_t, t, sparse_input, mask)
loss = F.mse_loss(v_pred, v_target)
```

**After:**
```python
input_coords = batch['input_coords']  # (B, N_in, 2)
input_values = batch['input_values']  # (B, N_in, 3)
target_coords = batch['target_coords']  # (B, N_out, 2)
target_values = batch['target_values']  # (B, N_out, 3)

v_pred = model(input_coords, input_values, target_coords, t)
loss = F.mse_loss(v_pred, v_target)
# ↑ Loss computed on arbitrary query coordinates
```

---

## Why the "Noisy 32×32" Is Actually Good

### Understanding the Bias-Variance Trade-off

**High-Frequency Fitting (Your Current Models):**
```
Training: Fit exact pixel values at 32×32
Result at 32×32: PSNR = 30 dB (very sharp)
Result at 64×64: Artifacts, grid patterns (overfitted)

Model learned: "These 1024 specific pixel values"
```

**Smooth Function Fitting (Coordinate Approach):**
```
Training: Fit smooth function through sparse observations
Result at 32×32: PSNR = 24 dB (some blur/noise)
Result at 64×64: Smooth, coherent (learned structure)

Model learned: "The underlying spatial function"
```

**The key insight:**
- Noise at 32×32 = Model regularized to learn smooth function
- Smooth function = Scale-invariant structure
- Scale-invariant structure = Good SR

**Analogy:**
```
Scientist observing data points:

Approach 1 (Grid-based):
"Draw a line through each measured point exactly"
→ Perfect fit to measurements
→ Poor prediction for new points

Approach 2 (Coordinate-based):
"Fit a smooth curve to the measurements"
→ Some error on measurements
→ Good prediction for new points
```

---

## Practical Recommendations

### For Best SR Quality

**Trade-off spectrum:**
```
More noise at 32×32  ⟷  Better SR at 64×64+
        ↓                         ↓
Smooth continuous field    Sharp but grid-locked
```

**Recommendations:**

1. **If you care most about 32×32 quality:**
   - Keep grid-based approach
   - Accept poor SR

2. **If you care most about SR quality:**
   - Switch to coordinate-based approach
   - Accept some noise at 32×32

3. **If you want both:**
   - Use coordinate-based architecture
   - Add multi-resolution training
   - Train on 32×32, 48×48, 64×64
   - Model learns sharp AND scale-invariant

### Multi-Resolution Training (Best of Both Worlds)

```python
# Training loop with mixed resolutions
for batch in dataloader:
    # Randomly choose resolution
    res = random.choice([32, 48, 64])

    # Generate query coordinates at chosen resolution
    query_coords = uniform_grid(res)  # (res², 2)

    # Train
    pred = model(input_coords, input_values, query_coords, t)
    loss = F.mse_loss(pred, target_values)
```

Now model learns:
- Sharp features at 32×32
- Smooth structure across scales
- Scale-invariant representations

---

## Summary: The Paradigm Shift

### Grid-Based Thinking (Current)
- Images are 2D arrays of pixels
- Process with CNNs or patch-based transformers
- Resolution is architectural constraint
- SR requires extrapolation to unseen grid sizes

### Coordinate-Based Thinking (Mamba Approach)
- Images are continuous functions over [0,1]²
- Process coordinate-value pairs with neural fields
- Resolution is sampling density
- SR is just querying more coordinates

**The lesson:**
> **Zero-shot super-resolution is not about better diffusion/flow formulations.**
> **It's about learning continuous functions instead of discrete grids.**

The Mamba notebook succeeds because it learns `f(x,y,t) → RGB`, not "how to transform 32×32 grids".

Your flow matching formulation is fine — you just need to apply it to **coordinate-based representations** instead of **grid-based representations**.
