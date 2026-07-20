"""
FID / Inception Score helpers, built on torchmetrics (which handles the Inception-v3
feature extraction internally). Kept as a thin wrapper so train.py can call a couple of
simple functions without needing to know the torchmetrics API details.

Decoupled from any specific model/sampling mechanism: callers pass a `sample_fn(batch_size)`
callable that returns a (batch_size, 3, H, W) tensor of generated images in [-1, 1]. That way
this module doesn't need to know whether generation happens via a diffusion reverse loop,
a VAE decoder, or anything else.

Both metrics are expensive to compute (they require *generating* samples), so train.py only
calls this periodically -- see configs.yml's EVAL.FID_EVERY_N_EPOCHS -- not every epoch.
"""

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def _to_uint8(images):
    """torchmetrics expects uint8 images in [0, 255]; our images are in [-1, 1]."""
    images = (images.clamp(-1, 1) + 1.0) / 2.0  # -> [0, 1]
    return (images * 255).to(torch.uint8)


@torch.no_grad()
def compute_fid_and_is(sample_fn, real_dataloader, num_samples, device, batch_size=100):
    """
    Generates `num_samples` images via `sample_fn(batch_size) -> (B, 3, H, W)` tensor in
    [-1, 1], and compares them against real images drawn from `real_dataloader` using FID,
    and separately scores the generated images with Inception Score.

    Returns: (fid_value: float, is_mean: float, is_std: float)
    """
    fid = FrechetInceptionDistance(normalize=False).to(device)
    inception_score = InceptionScore(normalize=False).to(device)

    # --- Real images ---
    real_seen = 0
    for real_images, _ in real_dataloader:
        real_images = real_images.to(device)
        fid.update(_to_uint8(real_images), real=True)
        real_seen += real_images.shape[0]
        if real_seen >= num_samples:
            break

    # --- Generated images ---
    generated_seen = 0
    while generated_seen < num_samples:
        cur_batch = min(batch_size, num_samples - generated_seen)
        samples = sample_fn(cur_batch)
        fake_uint8 = _to_uint8(samples)
        fid.update(fake_uint8, real=False)
        inception_score.update(fake_uint8)
        generated_seen += cur_batch

    fid_value = fid.compute().item()
    is_mean, is_std = inception_score.compute()
    return fid_value, is_mean.item(), is_std.item()
