"""
Perceptual loss (pretrained VGG16 features) and a PatchGAN discriminator used to
progressively add perceptual + adversarial terms on top of the base reconstruction
loss, operating on the model's reconstructed image vs. the real image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGPerceptualLoss(nn.Module):
    """
    LPIPS-style perceptual loss using a frozen, pretrained VGG16.

    Images are expected in [-1, 1] (the model's native range). They get converted to
    [0, 1], resized to 224x224 (VGG's expected input size), and normalized with
    ImageNet statistics before being passed through VGG's feature layers.
    """

    def __init__(self, layer_indices=(3, 8, 15, 22), resize_to=224):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        self.layer_indices = sorted(layer_indices)
        self.resize_to = resize_to

        self.register_buffer("mean", IMAGENET_MEAN)
        self.register_buffer("std", IMAGENET_STD)

    def _preprocess(self, x):
        x = (x.clamp(-1, 1) + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def _extract_features(self, x):
        feats = []
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.layer_indices:
                feats.append(h)
            if i == self.layer_indices[-1]:
                break
        return feats

    def forward(self, pred_x0, real_x0):
        pred = self._preprocess(pred_x0)
        real = self._preprocess(real_x0)

        pred_feats = self._extract_features(pred)
        with torch.no_grad():
            real_feats = self._extract_features(real)

        loss = 0.0
        for pf, rf in zip(pred_feats, real_feats):
            loss = loss + F.l1_loss(pf, rf)
        return loss / len(pred_feats)


class PatchDiscriminator(nn.Module):
    """
    Small PatchGAN-style discriminator for 32x32 images. Classifies overlapping
    patches as real/fake rather than the whole image at once (outputs a small
    spatial map of logits instead of a single scalar).
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        def block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base_channels, stride=2, norm=False),   # 32 -> 16
            *block(base_channels, base_channels * 2, stride=2),         # 16 -> 8
            *block(base_channels * 2, base_channels * 4, stride=2),     # 8 -> 4
            nn.Conv2d(base_channels * 4, 1, kernel_size=3, stride=1, padding=1),  # 4x4 patch logits
        )

    def forward(self, x):
        return self.net(x)  # raw logits, shape (B, 1, 4, 4)


def discriminator_loss(discriminator, real_x0, fake_x0_detached):
    """Standard non-saturating GAN discriminator loss (real -> 1, fake -> 0)."""
    real_logits = discriminator(real_x0)
    fake_logits = discriminator(fake_x0_detached)

    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return real_loss + fake_loss


def generator_adversarial_loss(discriminator, fake_x0):
    """Non-saturating generator loss: encourage discriminator to call fakes 'real'."""
    fake_logits = discriminator(fake_x0)
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
