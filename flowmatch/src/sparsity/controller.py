"""
SparsityController for generating conditioning and target masks.
Adapted from reference code with added 'grid' pattern for super-resolution.
"""
import torch
import numpy as np


class SparsityController:
    """
    Central controller for generating conditioning and target masks.

    Args:
        image_size: Image height/width (assumes square)
        mode: One of ['random_epoch', 'random_iter', 'fixed_instance', 'fixed_all']
        pattern: One of ['random', 'block', 'grid']
        sparsity: Total fraction of pixels to use (e.g., 0.2 = 20%)
        block_size: Side length of square blocks (for 'block' pattern)
        num_blocks: Number of blocks to place (for 'block' pattern)
        grid_stride: Stride for grid pattern (e.g., 2 or 4 for SR)
        seed: Random seed for reproducibility
    """
    def __init__(self, image_size, mode='random_epoch', pattern='random',
                 sparsity=0.2, block_size=5, num_blocks=6, grid_stride=2, seed=42):
        self.mode = mode
        self.pattern = pattern
        self.sparsity = sparsity
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.grid_stride = grid_stride
        self.H = self.W = image_size
        self.rng = np.random.default_rng(seed)
        self.cache = {}  # Stores masks for fixed modes

    def get_masks(self, B, C, sample_ids=None):
        """
        Returns B conditioning and target masks.

        Args:
            B: Batch size
            C: Number of channels (will broadcast to this)
            sample_ids: Required for fixed_instance mode

        Returns:
            cond_masks: List of B tensors, each (C, H, W)
            target_masks: List of B tensors, each (C, H, W)
        """
        if self.mode == 'fixed_all':
            if 'fixed_all' not in self.cache:
                # Generate only once and store
                mask_cond, mask_target = self._generate_mask_pair()
                self.cache['fixed_all'] = (mask_cond, mask_target)
            else:
                mask_cond, mask_target = self.cache['fixed_all']

            cond_masks = [mask_cond.repeat(C, 1, 1).clone() for _ in range(B)]
            target_masks = [mask_target.repeat(C, 1, 1).clone() for _ in range(B)]
            return cond_masks, target_masks

        conds, targets = [], []
        for i in range(B):
            if self.mode == 'random_iter':
                # Always generate new mask
                key = self.rng.integers(0, 1e9)
            elif self.mode == 'fixed_instance':
                # Use sample_id as key
                assert sample_ids is not None, "sample_ids required for fixed_instance mode"
                key = sample_ids[i]
            else:  # random_epoch
                # Generate fresh every time
                key = None

            if key is not None and key in self.cache:
                cond, target = self.cache[key]
            else:
                cond, target = self._generate_mask_pair()
                if self.mode == 'fixed_instance' and key is not None:
                    self.cache[key] = (cond, target)

            conds.append(cond.repeat(C, 1, 1))
            targets.append(target.repeat(C, 1, 1))

        return conds, targets

    def _generate_mask_pair(self):
        """
        Generate a pair of masks: one for conditioning, one for supervision.

        Returns:
            cond_mask: (1, H, W)
            target_mask: (1, H, W)
        """
        if self.pattern == 'random':
            return self._generate_random_masks()
        elif self.pattern == 'block':
            return self._generate_block_masks()
        elif self.pattern == 'grid':
            return self._generate_grid_masks()
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def _generate_random_masks(self):
        """Random pixel selection, split 50/50 between cond and target."""
        total = self.H * self.W
        num_total = int(self.sparsity * total)
        num_half = num_total // 2

        # Random pixel indices
        idx = self.rng.choice(total, num_total, replace=False)
        idx_cond, idx_target = idx[:num_half], idx[num_half:]

        mask_cond = torch.zeros(total)
        mask_target = torch.zeros(total)
        mask_cond[idx_cond] = 1.0
        mask_target[idx_target] = 1.0

        mask_cond = mask_cond.view(1, self.H, self.W)
        mask_target = mask_target.view(1, self.H, self.W)

        return mask_cond, mask_target

    def _generate_block_masks(self):
        """Place non-overlapping blocks, split evenly between cond and target."""
        mask_cond = torch.zeros(self.H, self.W)
        mask_target = torch.zeros(self.H, self.W)

        total_blocks = self.num_blocks
        assert total_blocks % 2 == 0, "num_blocks must be even to split into cond/target"

        block_coords = []
        max_tries = 1000
        placed = 0
        tries = 0

        while placed < total_blocks and tries < max_tries:
            x = self.rng.integers(0, self.H - self.block_size + 1)
            y = self.rng.integers(0, self.W - self.block_size + 1)

            # Check for overlap
            overlap = False
            for bx, by in block_coords:
                if abs(bx - x) < self.block_size and abs(by - y) < self.block_size:
                    overlap = True
                    break

            if not overlap:
                block_coords.append((x, y))
                placed += 1
            tries += 1

        assert len(block_coords) == total_blocks, f"Failed to place {total_blocks} blocks"

        # Shuffle and split
        self.rng.shuffle(block_coords)
        cond_blocks = block_coords[:total_blocks // 2]
        target_blocks = block_coords[total_blocks // 2:]

        for x, y in cond_blocks:
            mask_cond[x:x+self.block_size, y:y+self.block_size] = 1

        for x, y in target_blocks:
            mask_target[x:x+self.block_size, y:y+self.block_size] = 1

        return mask_cond.view(1, self.H, self.W), mask_target.view(1, self.H, self.W)

    def _generate_grid_masks(self):
        """
        Grid pattern for super-resolution: pixels on a stride-s lattice.
        Conditioning = grid pixels, target = complement.
        """
        s = self.grid_stride
        mask_cond = torch.zeros(self.H, self.W)

        # Set grid pixels (::s, ::s)
        mask_cond[::s, ::s] = 1.0

        # Target = all other pixels
        mask_target = 1.0 - mask_cond

        return mask_cond.view(1, self.H, self.W), mask_target.view(1, self.H, self.W)
