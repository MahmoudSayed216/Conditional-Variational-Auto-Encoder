"""
Class-conditional convolutional VAE for CIFAR-10.

Reuses the same GroupNorm/SiLU residual-block + class-conditioning style as the
project's original ConditionalDDPM, minus the timestep embedding (a VAE has no
diffusion timestep to condition on) and minus encoder->decoder skip connections
(a VAE must reconstruct purely from its latent bottleneck -- skip connections
would let the decoder cheat by copying the input instead of actually compressing
it through the latent space).
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """A GroupNorm/SiLU residual block with class-conditioning injection and optional self-attention."""

    def __init__(self, in_channels, out_channels, num_groups, emb_dim, use_attn=False):
        super().__init__()
        self.use_attn = use_attn

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.cond_proj = nn.Linear(emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if use_attn:
            self.attn_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4, batch_first=True)

    def forward(self, x, cond):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        x = h + self.shortcut(x)

        if self.use_attn:
            h = self.attn_norm(x)
            B, C, H, W = h.shape
            h = h.view(B, C, H * W).permute(0, 2, 1)
            attn_out, _ = self.attention(h, h, h)
            attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
            x = x + attn_out

        return x


class DownBlock(nn.Module):
    """ResBlock followed by a strided-conv downsample (halves spatial resolution)."""

    def __init__(self, in_channels, out_channels, num_groups, emb_dim, use_attn=False):
        super().__init__()
        self.block = ResBlock(in_channels, out_channels, num_groups, emb_dim, use_attn)
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, cond):
        x = self.block(x, cond)
        return self.pool(x)


class UpBlock(nn.Module):
    """Nearest-neighbor upsample (doubles spatial resolution) followed by a ResBlock."""

    def __init__(self, in_channels, out_channels, num_groups, emb_dim, use_attn=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = ResBlock(in_channels, out_channels, num_groups, emb_dim, use_attn)

    def forward(self, x, cond):
        x = self.upsample(x)
        return self.block(x, cond)


class ConditionalVAE(nn.Module):
    """
    Encoder maps an image + class label to a Gaussian posterior over a spatial
    latent (mu, logvar); decoder maps a latent sample + class label back to a
    reconstructed image.

    The encoder downsamples 3 times (stride-2 conv each), so for IMAGE_SIDE_LENGTH=32
    the latent spatial size is 32 // 8 = 4. Latent shape is therefore
    (LATENT_CHANNELS, 4, 4) for the default CIFAR-10 config.
    """

    def __init__(self, num_classes, embedding_dim, num_groups, channels_per_level, latent_channels):
        super().__init__()
        c1, c2, c3, c4 = channels_per_level
        self.latent_channels = latent_channels

        self.class_embedder = nn.Embedding(num_classes, embedding_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # ---- Encoder ----
        self.stem_conv = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.down1 = DownBlock(c1, c1, num_groups, embedding_dim, use_attn=False)
        self.down2 = DownBlock(c1, c2, num_groups, embedding_dim, use_attn=True)
        self.down3 = DownBlock(c2, c3, num_groups, embedding_dim, use_attn=True)
        self.bottleneck_enc = ResBlock(c3, c4, num_groups, embedding_dim, use_attn=True)
        self.to_mu = nn.Conv2d(c4, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(c4, latent_channels, kernel_size=1)

        # ---- Decoder ----
        self.from_latent = nn.Conv2d(latent_channels, c4, kernel_size=1)
        self.bottleneck_dec = ResBlock(c4, c4, num_groups, embedding_dim, use_attn=True)
        self.up3 = UpBlock(c4, c3, num_groups, embedding_dim, use_attn=True)
        self.up2 = UpBlock(c3, c2, num_groups, embedding_dim, use_attn=True)
        self.up1 = UpBlock(c2, c1, num_groups, embedding_dim, use_attn=False)

        self.out_layer = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=c1),
            nn.SiLU(),
            nn.Conv2d(c1, 3, kernel_size=3, padding=1),
        )

    def _cond_vector(self, class_id):
        cls_emb = self.class_embedder(class_id).squeeze(1) if len(class_id.shape) > 1 else self.class_embedder(class_id)
        return self.emb_mlp(cls_emb)

    def encode(self, x, class_id):
        cond = self._cond_vector(class_id)
        h = self.stem_conv(x)
        h = self.down1(h, cond)
        h = self.down2(h, cond)
        h = self.down3(h, cond)
        h = self.bottleneck_enc(h, cond)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, class_id):
        cond = self._cond_vector(class_id)
        h = self.from_latent(z)
        h = self.bottleneck_dec(h, cond)
        h = self.up3(h, cond)
        h = self.up2(h, cond)
        h = self.up1(h, cond)
        return self.out_layer(h)

    def forward(self, x, class_id):
        mu, logvar = self.encode(x, class_id)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, class_id)
        return recon, mu, logvar
