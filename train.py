"""
Main training script for the conditional VAE on CIFAR-10.

Progressive loss schedule (see configs.yml -> LOSS):
    Phase 1 (epochs 1..PHASE1_END_EPOCH):                reconstruction + KL only
    Phase 2 (epochs PHASE1_END_EPOCH+1..PHASE2_END_EPOCH): + perceptual (VGG16)
    Phase 3 (epochs PHASE2_END_EPOCH+1..EPOCHS):           + adversarial (PatchGAN)

Every epoch: decodes a fixed latent grid to visually track progress.
Every EVAL.FID_EVERY_N_EPOCHS epochs: computes an FID/IS estimate on a modest sample count.
At the end of training: computes a final, larger-sample-count FID/IS.
"""

import os

# Silence tqdm progress bars globally (e.g. the VGG16 pretrained-weights
# download inside VGGPerceptualLoss) -- must be set before torch/torchvision
# are imported.
os.environ.setdefault("TQDM_DISABLE", "1")

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont

from cifar10_dataset import CIFAR10Dataset, denormalize
from ConditionalVAE import ConditionalVAE
from losses import VGGPerceptualLoss, PatchDiscriminator, discriminator_loss, generator_adversarial_loss
from metrics import compute_fid_and_is


def load_config(path="configs.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_loss_weights(epoch, loss_cfg):
    """Returns (perceptual_weight, adversarial_weight) active at this epoch (1-indexed)."""
    perceptual_weight = 0.0
    adversarial_weight = 0.0

    if epoch > loss_cfg["PHASE1_END_EPOCH"]:
        perceptual_weight = loss_cfg["PERCEPTUAL_WEIGHT"]
    if epoch > loss_cfg["PHASE2_END_EPOCH"]:
        adversarial_weight = loss_cfg["ADVERSARIAL_WEIGHT"]

    return perceptual_weight, adversarial_weight


def build_fixed_latent_inputs(cfg, device):
    """Fixed latent z + class ids cycling through 0..NUM_CLASSES-1, seeded, for
    consistent per-epoch qualitative samples. NUM_FIXED_SAMPLES can be set higher
    than NUM_CLASSES -- class ids simply wrap around (0,1,...,9,0,1,...,9,...) so
    every class keeps getting represented no matter how high it's set.
    """
    model_cfg = cfg["MODEL"]
    num_classes = model_cfg["NUM_CLASSES"]
    num_samples = cfg["SAMPLING"]["NUM_FIXED_SAMPLES"]
    latent_channels = model_cfg["LATENT_CHANNELS"]
    latent_spatial = model_cfg["IMAGE_SIDE_LENGTH"] // 8  # encoder downsamples 3x (2^3 = 8)

    generator = torch.Generator(device="cpu").manual_seed(cfg["TRAINING"]["SEED"])
    fixed_z = torch.randn(
        (num_samples, latent_channels, latent_spatial, latent_spatial), generator=generator,
    ).to(device)
    fixed_class_ids = torch.arange(num_samples, device=device) % num_classes
    return fixed_z, fixed_class_ids


@torch.no_grad()
def sample_with_fixed_latent(model, fixed_z, fixed_class_ids):
    """Decodes a fixed latent tensor (not resampled each call) through the VAE decoder."""
    model.eval()
    samples = model.decode(fixed_z, fixed_class_ids)
    model.train()
    return samples.clamp(-1.0, 1.0)


CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def save_labeled_sample_grid(images, class_ids, path, nrow, padding=2, upscale=8):
    """
    Saves an image grid (like torchvision.utils.save_image) but also stamps each
    tile's class name in its top-left corner, so it's easy to visually confirm
    samples match the class they were conditioned on.

    Args:
        images: (N, 3, H, W) tensor already denormalized to [0, 1].
        class_ids: (N,) tensor or list of int class indices, same order as images.
        path: output file path.
        nrow: images per row (same meaning as torchvision.utils.make_grid).
        padding: pixel padding between grid cells (matches make_grid's default of 2).
        upscale: integer factor to enlarge the grid before drawing text. CIFAR-10
            images are only 32x32, too small to fit legible text otherwise -- default
            of 8 brings each 32x32 tile up to 256x256.
    """
    grid = make_grid(images, nrow=nrow, padding=padding)
    grid_np = (grid.clamp(0.0, 1.0) * 255).byte().permute(1, 2, 0).cpu().numpy()
    grid_img = Image.fromarray(grid_np)

    if upscale > 1:
        grid_img = grid_img.resize(
            (grid_img.width * upscale, grid_img.height * upscale), resample=Image.NEAREST,
        )

    draw = ImageDraw.Draw(grid_img)
    font = ImageFont.load_default()

    img_h, img_w = images.shape[-2], images.shape[-1]
    ncols = nrow
    for idx in range(images.shape[0]):
        row, col = divmod(idx, ncols)
        cell_x = (padding + col * (img_w + padding)) * upscale
        cell_y = (padding + row * (img_h + padding)) * upscale
        class_id = int(class_ids[idx])
        label = CIFAR10_CLASS_NAMES[class_id] if 0 <= class_id < len(CIFAR10_CLASS_NAMES) else str(class_id)

        text_pos = (cell_x + 2, cell_y + 2)
        text_bbox = draw.textbbox(text_pos, label, font=font)
        box = (text_pos[0] - 1, text_pos[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1)
        # small filled box sized to the text so it stays legible over any tile color
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text(text_pos, label, fill=(255, 255, 0), font=font)

    grid_img.save(path)


def train_one_epoch(
    model, discriminator, perceptual_loss_fn,
    train_loader, model_optimizer, disc_optimizer,
    epoch, cfg, device,
):
    """Runs the per-batch training loop for a single epoch.

    No per-batch logging -- only returns the averaged losses so the caller
    can print a single summary line once the epoch finishes.

    Returns (avg_recon, avg_kl, avg_perceptual, avg_adv, phase_desc).
    """
    perceptual_weight, adversarial_weight = get_loss_weights(epoch, cfg["LOSS"])
    phase_desc = "recon+kl"
    if perceptual_weight > 0:
        phase_desc += "+perceptual"
    if adversarial_weight > 0:
        phase_desc += "+adversarial"

    grad_clip_norm = cfg["TRAINING"]["GRAD_CLIP_NORM"]
    kl_weight = cfg["LOSS"]["KL_WEIGHT"]

    running_recon = 0.0
    running_kl = 0.0
    running_perceptual = 0.0
    running_adv = 0.0

    for x0, class_ids in train_loader:
        x0 = x0.to(device)
        class_ids = class_ids.to(device)

        recon, mu, logvar = model(x0, class_ids)

        # ---- Reconstruction + KL (always active -- this is the base VAE objective) ----
        recon_loss = nn.functional.mse_loss(recon, x0)
        kl_loss = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).sum(dim=[1, 2, 3]).mean()
        total_loss = recon_loss + kl_weight * kl_loss

        perceptual_loss_value = torch.tensor(0.0, device=device)
        adv_gen_loss_value = torch.tensor(0.0, device=device)

        # ---- Perceptual loss (phase 2+) ----
        if perceptual_weight > 0:
            perceptual_loss_value = perceptual_loss_fn(recon, x0)
            total_loss = total_loss + perceptual_weight * perceptual_loss_value

        # ---- Adversarial loss (phase 3+) ----
        if adversarial_weight > 0:
            # 1) Discriminator update (uses a detached fake so gradients don't flow into the model)
            disc_optimizer.zero_grad()
            d_loss = discriminator_loss(discriminator, x0, recon.detach())
            d_loss.backward()
            disc_optimizer.step()

            # 2) Generator (model) adversarial term -- fresh forward through D, not detached
            adv_gen_loss_value = generator_adversarial_loss(discriminator, recon)
            total_loss = total_loss + adversarial_weight * adv_gen_loss_value

        model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        model_optimizer.step()

        running_recon += recon_loss.item()
        running_kl += kl_loss.item()
        running_perceptual += perceptual_loss_value.item()
        running_adv += adv_gen_loss_value.item()

    n_batches = len(train_loader)
    avg_recon = running_recon / n_batches
    avg_kl = running_kl / n_batches
    avg_perceptual = running_perceptual / n_batches
    avg_adv = running_adv / n_batches

    return avg_recon, avg_kl, avg_perceptual, avg_adv, phase_desc


def train(cfg):
    torch.manual_seed(cfg["TRAINING"]["SEED"])

    device = cfg["TRAINING"]["DEVICE"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(cfg["TRAINING"]["CHECKPOINT_DIR"], exist_ok=True)
    os.makedirs(cfg["SAMPLING"]["SAMPLES_DIR"], exist_ok=True)

    model_cfg = cfg["MODEL"]
    embedding_dim = model_cfg["CLASS_EMBEDDING_DIM"]

    train_dataset = CIFAR10Dataset(
        root=cfg["DATA"]["DATA_DIR"], train=True,
        image_side_length=model_cfg["IMAGE_SIDE_LENGTH"], augment=True,
        download=cfg["DATA"].get("DOWNLOAD", True),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["DATA"]["BATCH_SIZE"], shuffle=True,
        num_workers=cfg["DATA"]["NUM_WORKERS"], pin_memory=True, drop_last=True,
    )

    real_eval_dataset = CIFAR10Dataset(
        root=cfg["DATA"]["DATA_DIR"], train=False,
        image_side_length=model_cfg["IMAGE_SIDE_LENGTH"], augment=False,
        download=cfg["DATA"].get("DOWNLOAD", True),
    )
    real_eval_loader = DataLoader(
        real_eval_dataset, batch_size=cfg["DATA"]["BATCH_SIZE"], shuffle=True,
        num_workers=cfg["DATA"]["NUM_WORKERS"],
    )

    # ---- Model ----
    model = ConditionalVAE(
        num_classes=model_cfg["NUM_CLASSES"],
        embedding_dim=embedding_dim,
        num_groups=model_cfg["NUM_GROUPS"],
        channels_per_level=model_cfg["CHANNELS_PER_LEVEL"],
        latent_channels=model_cfg["LATENT_CHANNELS"],
    ).to(device)

    # ---- Progressive loss components (perceptual / adversarial) ----
    perceptual_loss_fn = VGGPerceptualLoss(layer_indices=cfg["LOSS"]["VGG_LAYER_INDICES"]).to(device)
    discriminator = PatchDiscriminator().to(device)

    # ---- Optimizers ----
    betas = tuple(cfg["TRAINING"]["ADAM_BETAS"])
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["TRAINING"]["LR"], betas=betas)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg["TRAINING"]["DISCRIMINATOR_LR"], betas=betas)

    # ---- Fixed latent for per-epoch qualitative sampling ----
    fixed_z, fixed_class_ids = build_fixed_latent_inputs(cfg, device)

    # ---- Sampling closure for FID/IS: random latent + random class -> decoded image ----
    latent_channels = model_cfg["LATENT_CHANNELS"]
    latent_spatial = model_cfg["IMAGE_SIDE_LENGTH"] // 8

    @torch.no_grad()
    def sample_fn(batch_size):
        z = torch.randn(batch_size, latent_channels, latent_spatial, latent_spatial, device=device)
        class_ids = torch.randint(0, model_cfg["NUM_CLASSES"], (batch_size,), device=device)
        model.eval()
        samples = model.decode(z, class_ids).clamp(-1.0, 1.0)
        model.train()
        return samples

    num_epochs = cfg["TRAINING"]["EPOCHS"]

    for epoch in range(1, num_epochs + 1):
        avg_recon, avg_kl, avg_perceptual, avg_adv, phase_desc = train_one_epoch(
            model, discriminator, perceptual_loss_fn,
            train_loader, model_optimizer, disc_optimizer,
            epoch, cfg, device,
        )

        print(
            f"== Epoch {epoch}/{num_epochs} done | phase={phase_desc} | "
            f"avg_recon={avg_recon:.4f} avg_kl={avg_kl:.4f} "
            f"avg_perceptual={avg_perceptual:.4f} avg_adv={avg_adv:.4f} =="
        )

        # ---- Per-epoch qualitative sample grid from a fixed latent ----
        if epoch % cfg["SAMPLING"]["SAMPLE_EVERY_N_EPOCHS"] == 0:
            samples = sample_with_fixed_latent(model, fixed_z, fixed_class_ids)
            grid_path = os.path.join(cfg["SAMPLING"]["SAMPLES_DIR"], f"epoch_{epoch:03d}.png")
            save_labeled_sample_grid(
                denormalize(samples), fixed_class_ids, grid_path,
                nrow=min(model_cfg["NUM_CLASSES"], fixed_z.shape[0]),
            )
            print(f"Saved fixed-latent sample grid -> {grid_path}")

        # ---- Periodic FID / IS estimate ----
        if cfg["EVAL"]["COMPUTE_FID_IS"] and epoch % cfg["EVAL"]["FID_EVERY_N_EPOCHS"] == 0:
            fid_value, is_mean, is_std = compute_fid_and_is(
                sample_fn, real_eval_loader,
                num_samples=cfg["EVAL"]["FID_NUM_SAMPLES"], device=device,
            )
            print(f"[epoch {epoch}] periodic FID={fid_value:.3f} | IS={is_mean:.3f} +/- {is_std:.3f}")

        # ---- Checkpointing ----
        if epoch % cfg["TRAINING"]["CHECKPOINT_EVERY_N_EPOCHS"] == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(cfg["TRAINING"]["CHECKPOINT_DIR"], f"vae_epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "model_optimizer_state_dict": model_optimizer.state_dict(),
                "disc_optimizer_state_dict": disc_optimizer.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")

    # ---- Final, larger-sample-count FID / IS ----
    if cfg["EVAL"]["COMPUTE_FID_IS"]:
        fid_value, is_mean, is_std = compute_fid_and_is(
            sample_fn, real_eval_loader,
            num_samples=cfg["EVAL"]["FINAL_FID_NUM_SAMPLES"], device=device,
        )
        print(f"== FINAL METRICS == FID={fid_value:.3f} | IS={is_mean:.3f} +/- {is_std:.3f}")


def main():
    cfg = load_config("configs.yml")
    train(cfg)


if __name__ == "__main__":
    main()
