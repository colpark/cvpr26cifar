Below is a **single, copy-paste prompt** you can give to your coding agent. It tells it *exactly* what to build and how to test it:

---

**ROLE:** You are a senior ML engineer. Implement, train, and test a masked-conditioning generative model on CIFAR-10 with three variants:

1) **Baseline (DDPM) on CIFAR-10** re-implementing the existing code’s behavior.  
2) **Flow-Matching (FM) using the same U-Net architecture** as much as possible.  
3) **Flow-Matching with a transformer-style model** (DiT/U-ViT) *and* an optional **Perceiver IO** variant, designed to scale to higher resolutions and enable zero-shot super-resolution (SR).

**CONTEXT & INVARIANTS (from the reference code):**
- The current pipeline concatenates **three inputs** to the denoiser: `[x_t, sparse_input, mask]`, so the first conv expects `channel*3` input channels (e.g., for RGB CIFAR-10, that’s `3*3=9`). Keep this contract unless otherwise noted. fileciteturn0file1  
- The trainer builds **conditioning masks** and **target masks**, forms `sparse_input = image * cond_mask`, and uses a **masked loss** (focused on the target region, with a small penalty on conditioned pixels). Keep this training semantics. Also, remove the single-channel assumption. fileciteturn0file0  
- The diffusion module currently does DDPM (and DDIM) with the same concatenation and masked loss wiring. Your FM implementation must preserve the same call signature (inputs & outputs) so the trainer can swap modules with minimal code churn. fileciteturn0file2

---

## DELIVERABLES

Produce a self-contained repo (or PR) with this structure:

```
.
├─ configs/
│  ├─ cifar10_ddpm.yaml
│  ├─ cifar10_fm_unet.yaml
│  ├─ cifar10_fm_dit.yaml
│  └─ cifar10_fm_perceiver.yaml
├─ data/                       # auto-downloaded CIFAR-10
├─ src/
│  ├─ datasets/
│  │  └─ cifar10.py
│  ├─ models/
│  │  ├─ unet_ddpm.py          # adapted from reference U-Net
│  │  ├─ unet_fm.py            # same U-Net, for FM
│  │  ├─ dit_fm.py             # transformer (DiT/U-ViT-style)
│  │  └─ perceiver_io_fm.py    # Perceiver IO variant
│  ├─ diffusion/
│  │  ├─ ddpm.py               # refactored from reference
│  │  └─ rectified_flow.py     # FM trainer + sampler (ODE integration)
│  ├─ sparsity/
│  │  └─ controller.py         # CIFAR-ready; adds 'grid' pattern for SR
│  ├─ trainers/
│  │  ├─ trainer_base.py       # generic loops, logging, FID hooks
│  │  ├─ trainer_ddpm.py
│  │  └─ trainer_fm.py
│  ├─ utils/
│  │  ├─ time_emb.py           # discrete (DDPM) + continuous (FM) embeddings
│  │  ├─ metrics.py            # FID/IS/PSNR/SSIM (masked + global)
│  │  └─ viz.py
│  └─ cli.py
├─ tests/
│  ├─ test_shapes.py
│  ├─ test_train_step.py
│  └─ test_sampling_api.py
└─ README.md
```

---

## TASKS

### A) Re-implement the baseline on **CIFAR-10 (32×32, RGB)**

1. **Dataset & transforms**
   - Use `torchvision.datasets.CIFAR10` (train split), auto-download to `./data`.  
   - Transform: `ToTensor() → map to [-1,1]` (`x = x*2-1`), no resize (CIFAR is 32×32).  
   - Return tuples `(img, idx)` to preserve `sample_id` behavior in the trainer (for mask reproducibility).  
   - Provide a config knob for optional augmentations (random horizontal flip).

2. **SparsityController (CIFAR)**
   - Port the existing controller behavior; support modes:
     - `random_epoch` / `random_iter`
     - `pattern in {random, block, grid}`  
       - **grid:** For SR training, reveal pixels on a stride-`s` lattice (`::s, ::s`) with `s ∈ {2, 4}`.
   - Ensure outputs are shape-compatible with **3-channel** RGB masks (either `(B,1,H,W)` broadcast to 3 or `(B,3,H,W)`). The trainer already expects to form `sparse_input = image * cond_mask`. fileciteturn0file0

3. **U-Net (DDPM)**
   - Start from the reference U-Net; keep the **concatenation contract** (`init_conv` input channels = `3*channel`). For CIFAR, set `channel=3`, so `init_conv` gets 9 channels. fileciteturn0file1  
   - Keep GroupNorm+SiLU, attention at 16×16. Time embedding: **discrete** timesteps (int) via positional encoding (as in reference). fileciteturn0file1

4. **Diffusion (DDPM)**
   - Port the reference DDPM and DDIM samplers as `src/diffusion/ddpm.py`. Preserve concatenation `[x_t, sparse, mask]` and masked loss with the small conditioner penalty. fileciteturn0file2  
   - Fix any channel alignment to handle `C=3`.  
   - Remove the single-channel assertion in the trainer (`assert C == 1`) and replace with checks that `C == 3` OR allow any `C` ≥ 1 (preferred). fileciteturn0file0

5. **Trainer & logging**
   - Implement `trainer_ddpm.py` mirroring the reference loop: stepwise loss, periodic image grids, optional FID/IS.  
   - For FID on CIFAR, implement a **conditional FID** path that samples with random masks drawn from the same controller to match training conditions; also provide an “unconditional baseline” by passing zero masks (document the difference). fileciteturn0file0

6. **CLI**
   - `python -m src.cli --config configs/cifar10_ddpm.yaml` should train DDPM on CIFAR-10 and write samples + logs.

**Default hyperparams (config):**
- `image_size=32`, `channel=3`, `dim=128`, `dim_multiply=(1,2,4,4)`, `num_res_blocks=2`, `attn_resolutions=(16,)`.  
- DDPM: `T=1000`, optimizer Adam `lr=2e-4`, batch size `128`, total steps `200k`.  
- Sparsity: default `pattern=random`, `sparsity=0.2`; enable `grid` for SR experiments.

---

### B) **Flow Matching (FM) using the same U-Net**

1. **Rectified-Flow module (`rectified_flow.py`)**
   - Implement mask-aware *straight* path:
     - Draw `t ~ Uniform(0,1)`, noise `z ~ N(0,I)`.  
     - If `mask` is provided, define `z* = (1-mask)·z + mask·x0` (known pixels stay fixed).  
     - Define `x_t = (1-t)·x0 + t·z*` and **target velocity** `v* = z* − x0 = (1-mask)·(z − x0)`.  
   - U-Net predicts `vθ(x_t, t, sparse, mask)`. Loss: **masked MSE** over `loss_mask + λ·mask`, with `λ≈0` for FM because `v*` is already zero on known pixels. Reuse the reference masked-loss pattern. fileciteturn0file2  
   - Continuous **time embedding**: replace integer timesteps with `float32 t∈[0,1]` (Gaussian Fourier features or your existing `PositionalEncoding` adjusted for floats). Keep forward signature unchanged for the trainer. fileciteturn0file1

2. **Sampler (probability-flow ODE)**
   - Euler solver over `steps=S`:  
     ```
     x_{k+1} = x_k - (1/S) · vθ(x_k, t_k, sparse, mask);   t_k = 1 - k/S
     ```
   - After each step, **hard project known pixels**: `x = mask*sparse + (1-mask)*x`.  
   - Accept `H,W` override to sample at **higher resolution** (zero-shot SR). If `sparse`/`mask` spatial size differs, upsample with `nearest`.

3. **U-Net reuse**
   - Keep the CIFAR U-Net intact (same concatenation, same channels, same attention). Only change time embedding to accept floats.

4. **Trainer**
   - `trainer_fm.py` mirrors `trainer_ddpm.py`. Add a config `steps=50` for ODE sampling.  
   - Provide two configs:  
     - `cifar10_fm_unet.yaml` — FM with U-Net.  
     - Enable SR: add a **grid** sparsity pattern (e.g., stride `s=2` or `4`). During sampling, request `H=W=64` or `128`.

---

### C) **Flow Matching with a Transformer / Perceiver IO**

You will implement **two** alternative backbones that *retain the same I/O contract* as the U-Net:

1) **Transformer (DiT/U-ViT style) – `dit_fm.py`**
   - **Inputs:** Stack channels `[x, sparse, mask]` → `(B, 9, H, W)` for RGB. **Patchify** to tokens (e.g., 2×2 or 4×4 patches), linear proj to `d_model`.  
   - **Positional encoding:** Use **2D Fourier features** (or RoPE) that are *resolution-agnostic*. **Avoid absolute learned positions** tied to 32×32; the model must run at 64/128 without resizing errors.  
   - **Conditioning:** Keep concatenation semantics (conditioning already inside tokens via `sparse, mask`). Optionally add a **timestep token** (from continuous time embedding) prepended to the sequence.  
   - **Blocks:** Pre-LN Transformer blocks with MLPs (GELU), residuals, and **windowed attention** (or full if memory allows at 32×32; windowed for 128×128).  
   - **Head:** Linear proj from tokens back to patches → fold to image → predict **velocity** `vθ` with `C=3`.  
   - Expose `forward(x, t, sparse, mask)` exactly like U-Net’s signature (the trainer/flow module hands in concatenated `x` already).

2) **Perceiver IO – `perceiver_io_fm.py`**
   - **Inputs:** Tokens from `[x, sparse, mask]` with 2D Fourier features concatenated.  
   - **Latents:** Use a fixed latent array (e.g., 256–512 latents at `d_latent=512`), apply several cross-attn + latent self-attn blocks.  
   - **Queries (IO):** Use a per-pixel query grid at the *target resolution* (works naturally for upscaling).  
   - **Time conditioning:** Add time embedding to latents and/or as bias terms in cross-attn.  
   - **Output:** Project query outputs to `C=3` velocity per pixel. Hard projection on known pixels during sampling remains in `rectified_flow.py`.

3) **Keep the trainer & FM module unchanged.** Only swap the model class via config and ensure `model(x_concat, t)` returns `vθ` with the same shape as `x`.

---

## ACCEPTANCE CRITERIA

- **API parity:**  
  - All models accept **concatenated** inputs consistent with the reference (`[x, sparse, mask]`), and their **first layer/channel math** is correct for CIFAR (9 input channels). fileciteturn0file1 fileciteturn0file2  
  - Trainers expose identical CLI flags across DDPM and FM.

- **Unit tests (`pytest`)**:
  1. `test_shapes.py` — For U-Net, DiT, and Perceiver IO, verify forward pass with `(B=2,C=9,H=W=32)` returns `(B,3,32,32)`.  
  2. `test_sampling_api.py` — FM sampler with `(sparse,mask)` round-trips shapes at 32, 64; hard projection keeps anchors exact (L2 distance zero on masked pixels).  
  3. `test_train_step.py` — One forward/backward step runs and decreases loss on a tiny dummy set (overfit 8 images).

- **Training smoke tests** (fast):  
  - DDPM on CIFAR-10 for 1k steps produces non-NaN loss and saves grids.  
  - FM-U-Net trains for 1k steps and can sample 32→64 SR with **grid** conditioning.  
  - FM-DiT and FM-Perceiver IO compile and run for 200 steps with valid samples.

- **Metrics (optional but preferred):**  
  - CIFAR-10 FID/IS (document conditional vs unconditional sampling).  
  - For SR, compute **masked PSNR/SSIM** on the unknown region only (using `target_mask`) and report sample grids. fileciteturn0file0

---

## IMPLEMENTATION NOTES & HINTS

- **Remove single-channel asserts** in the trainer; ensure batching, previews, and FID paths work with `C=3`. fileciteturn0file0  
- The **reference diffusion** and **U-Net** already assume concatenation of `[x, sparse, mask]` and set `init_conv` to `channel*3`. Mirror that everywhere so the trainer works across models. fileciteturn0file2 fileciteturn0file1  
- For FM, the **masked velocity** makes the small conditioner penalty unnecessary; set λ≈0 in masked MSE. Keep the *same* `loss_mask` handling as reference. fileciteturn0file2  
- **Zero-shot SR recipe (example):**
  - Training: include `pattern=grid` masks with strides `s∈{2,4}` on 32×32.  
  - Inference: upsample the mask & sparse anchor to 64 or 128 with nearest-neighbor; run FM ODE with hard projection each step.  
- **Memory tips:** DiT window size 8–16; Perceiver latents 256–512; use AMP + gradient clipping (`max_grad_norm=1.0`, as in reference). fileciteturn0file0

---

## COMMANDS

- **DDPM (baseline):**  
  `python -m src.cli --config configs/cifar10_ddpm.yaml`

- **FM (U-Net):**  
  `python -m src.cli --config configs/cifar10_fm_unet.yaml steps=50 sparsity.pattern=grid sparsity.grid_stride=4`

- **FM (DiT):**  
  `python -m src.cli --config configs/cifar10_fm_dit.yaml steps=50`

- **FM (Perceiver IO):**  
  `python -m src.cli --config configs/cifar10_fm_perceiver.yaml steps=50`

Each run must save: training curves, periodic 5×5 grids (GT / sparse / mask / output / target), and (if enabled) FID logs, matching the reference trainer’s visualization approach. fileciteturn0file0

---

## WHAT TO READ/EMULATE FROM THE REFERENCE FILES (do not copy verbatim)

- **`trainer (6) (1).py`** — data loop, mask creation, masked loss usage, sampling hooks, and logging patterns to replicate. fileciteturn0file0  
- **`model_original (7) (1).py`** — U-Net channel math (`channel*3` input), time embedding, attention placements; mirror for CIFAR’s 3-channel case. fileciteturn0file1  
- **`diffusion (8) (1).py`** — concatenation contract `[x_t, sparse_input, mask]`, alignment logic, and DDIM wrapper style; keep the same public method signatures for drop-in compatibility. fileciteturn0file2

---

**OUTPUT:** Submit the repo/PR, plus a short README explaining how to train each variant, how masks are formed, and how to run zero-shot SR (32→64/128). Include sample images and a table with the quick-and-dirty metrics (FID/IS and masked PSNR/SSIM).
