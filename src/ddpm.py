"""
Task: GenCV003 — Generative Models Comparison
Dataset: MNIST
Paper Reference: Ho et al., 2020 (https://arxiv.org/abs/2006.11239)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os
import math


# SINUSOIDAL TIME EMBEDDING
class SinusoidalPositionEmbedding(nn.Module):
    """
    Encodes the diffusion timestep t as a sinusoidal embedding (Transformer-style).
    Allows the U-Net to condition on the noise level at each forward pass.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args   = t[:, None].float() * freqs[None]
        emb    = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb


# BUILDING BLOCKS

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Time embedding projection → scale+shift for AdaGN
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )

        # Skip connection
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # Inject time: split into scale and shift
        t_proj = self.time_mlp(t_emb)          # (B, 2*out_ch)
        t_proj = t_proj[:, :, None, None]       # broadcast spatial dims
        scale, shift = t_proj.chunk(2, dim=1)

        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention over spatial locations (used at low-resolution feature maps).
    Scaled dot-product attention with residual connection.
    """
    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.reshape(B, 3, C, H * W).unbind(1)  # each: (B, C, HW)

        scale = C ** -0.5
        attn  = torch.einsum("bci,bcj->bij", q, k) * scale  # (B, HW, HW)
        attn  = attn.softmax(dim=-1)

        out = torch.einsum("bij,bcj->bci", attn, v)         # (B, C, HW)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, use_attn=False):
        super().__init__()
        self.res1  = ResidualBlock(in_ch,  out_ch, time_dim)
        self.res2  = ResidualBlock(out_ch, out_ch, time_dim)
        self.attn  = AttentionBlock(out_ch) if use_attn else nn.Identity()
        self.down  = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x)
        return self.down(x), x   # (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, use_attn=False):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res1  = ResidualBlock(in_ch + skip_ch, out_ch, time_dim)
        self.res2  = ResidualBlock(out_ch,          out_ch, time_dim)
        self.attn  = AttentionBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t):
        x = self.up(x)
        # Fixed the mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]))
        
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x)
        return x


# 3. U-NET NOISE PRED
class UNet(nn.Module):
    """
    U-Net that predicts the noise ε_θ(x_t, t) added at step t.
    Encoder: downsample with residual blocks.
    Bottleneck: residual + attention + residual.
    Decoder: upsample with skip connections.
    """
    def __init__(
        self,
        in_ch:    int = 1,
        base_ch:  int = 64,
        ch_mults: tuple = (1, 2, 4),
        time_dim: int = 256,
    ):
        super().__init__()
        # Time embedding pipeline
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = [base_ch * m for m in ch_mults]

        # Stem
        self.stem = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        in_c = base_ch
        for i, out_c in enumerate(channels):
            use_attn = (i == len(channels) - 1)   # attention only at deepest
            self.downs.append(DownBlock(in_c, out_c, time_dim, use_attn))
            in_c = out_c

        # Bottleneck
        self.mid_res1 = ResidualBlock(in_c, in_c, time_dim)
        self.mid_attn = AttentionBlock(in_c)
        self.mid_res2 = ResidualBlock(in_c, in_c, time_dim)

        # Decoder
        self.ups = nn.ModuleList()
        for i, out_c in enumerate(reversed(channels)):
            use_attn = (i == 0)
            next_c = channels[-i - 2] if i < len(channels) - 1 else base_ch
            self.ups.append(UpBlock(in_c, out_c, next_c, time_dim, use_attn))
            in_c = next_c

        # Output head
        self.out_norm = nn.GroupNorm(8, in_c)
        self.out_conv = nn.Conv2d(in_c, 1, 1)   # predict 1-channel noise

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)      # (B, time_dim)

        h = self.stem(x)
        skips = []
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)

        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip, t_emb)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# DIFFUSION PROCESS (FORWARD + REVERSE)
class DDPM(nn.Module):

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.T = T

        betas  = torch.linspace(beta_start, beta_end, T)      # (T,)
        alphas = 1.0 - betas                                   # α_t
        alpha_bar = torch.cumprod(alphas, dim=0)               # α_bar_t

        # Register as buffers so they move with .to(device)
        self.register_buffer("betas",          betas)
        self.register_buffer("alphas",         alphas)
        self.register_buffer("alpha_bar",      alpha_bar)
        self.register_buffer("sqrt_ab",        alpha_bar.sqrt())
        self.register_buffer("sqrt_one_m_ab",  (1 - alpha_bar).sqrt())

        self.unet = UNet()

    # Forward process
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None):
        """Sample x_t given x_0 and t (closed-form)."""
        if eps is None:
            eps = torch.randn_like(x0)
        sqrt_ab     = self.sqrt_ab[t][:, None, None, None]
        sqrt_1m_ab  = self.sqrt_one_m_ab[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_1m_ab * eps, eps

    # Training loss
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Simple denoising score-matching objective."""
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, eps     = self.q_sample(x0, t)
        eps_pred      = self.unet(x_t, t)
        return F.mse_loss(eps_pred, eps)

    # Reverse process
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One denoising step: x_t → x_{t-1}."""
        t_tensor = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        eps_pred = self.unet(x_t, t_tensor)

        beta_t       = self.betas[t]
        alpha_t      = self.alphas[t]
        alpha_bar_t  = self.alpha_bar[t]

        # Mean of p(x_{t-1} | x_t)
        coeff   = beta_t / (1 - alpha_bar_t).sqrt()
        x_prev  = (1 / alpha_t.sqrt()) * (x_t - coeff * eps_pred)

        if t > 0:
            noise  = torch.randn_like(x_t)
            x_prev = x_prev + beta_t.sqrt() * noise

        return x_prev

    @torch.no_grad()
    def sample(self, n: int, image_size: int = 28, device: str = "cpu") -> torch.Tensor:
        """Full ancestral sampling: x_T ~ N(0,I) → x_0."""
        x = torch.randn(n, 1, image_size, image_size, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t)
        return x.clamp(0, 1)


# TRAINING
def train_ddpm(
    epochs:     int   = 100,
    batch_size: int   = 128,
    lr:         float = 2e-4,
    T:          int   = 1000,
    save_dir:   str   = "results/ddpm",
    device:     str   = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # Data
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),   # scale to [-1, 1]
    ])
    train_ds     = datasets.MNIST("data", train=True,  download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)

    model  = DDPM(T=T).to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    print(f"Training DDPM on {device} | T={T} | epochs={epochs}")
    print(f"  U-Net params: {sum(p.numel() for p in model.unet.parameters()):,}")

    history = {"train_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss = model.loss(x)

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.unet.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")
            # Generate samples (rescale from [-1,1] to [0,1])
            model.eval()
            samples = model.sample(64, device=device)
            save_image(samples, f"{save_dir}/samples_epoch_{epoch:03d}.png", nrow=8)

    torch.save(model.state_dict(), f"{save_dir}/ddpm_final.pth")
    np.save(f"{save_dir}/history.npy", history)
    print(f"\nDDPM training complete. Saved to {save_dir}/")
    return model, history


if __name__ == "__main__":
    train_ddpm()
