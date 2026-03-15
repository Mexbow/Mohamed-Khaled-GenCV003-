"""
Task: GenCV003 — Generative Models Comparison
Dataset: MNIST
Paper Reference:Kingma, 2014 (https://arxiv.org/abs/1312.6114)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os

# Encoder
class Encoder(nn.Module):

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 28→14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14→7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 7→7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # Flatten: 128 * 7 * 7 = 6272
        self.flatten_dim = 128 * 7 * 7
        self.fc_mu      = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)          # flatten
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


# Reparameterize
def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    z = mu + eps * std,  eps ~ N(0, I)
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# Decoder
class Decoder(nn.Module):
    """
    Maps latent vector z -> reconstructed image x_hat.
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # 7→7
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 7→14
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 14→28
            nn.Sigmoid(),  # pixel values ∈ [0, 1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 128, 7, 7)
        x_hat = self.deconv(h)
        return x_hat


# Full Model
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128, beta: float = 1.0):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.beta    = beta

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = reparameterize(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var

    def loss(self, x, x_hat, mu, log_var):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)

        # KL divergence: -0.5 * SUM(1 + log σ² - μ² - σ²)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)

        return recon + self.beta * kl, recon, kl

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu'):
        z = torch.randn(n, self.encoder.latent_dim, device=device)
        return self.decoder(z)


# Training
def train_vae(
    epochs: int     = 50,
    batch_size: int = 128,
    latent_dim: int = 128,
    lr: float       = 1e-3,
    beta: float     = 1.0,
    save_dir: str   = "results/vae",
    device: str     = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # Data 
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = VAE(latent_dim=latent_dim, beta=beta).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    history = {"train_loss": [], "test_loss": [], "recon": [], "kl": []}

    print(f"Training VAE on {device} | latent_dim={latent_dim} | β={beta}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = recon_sum = kl_sum = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, log_var = model(x)
            loss, recon, kl    = model.loss(x, x_hat, mu, log_var)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            train_loss += loss.item()
            recon_sum  += recon.item()
            kl_sum     += kl.item()

        scheduler.step()
        n_batches = len(train_loader)
        history["train_loss"].append(train_loss / n_batches)
        history["recon"].append(recon_sum / n_batches)
        history["kl"].append(kl_sum / n_batches)

        # Eval
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_hat, mu, log_var = model(x)
                loss, _, _ = model.loss(x, x_hat, mu, log_var)
                test_loss += loss.item()
        history["test_loss"].append(test_loss / len(test_loader))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {history['train_loss'][-1]:.2f} | "
                  f"Test: {history['test_loss'][-1]:.2f} | "
                  f"Recon: {history['recon'][-1]:.2f} | "
                  f"KL: {history['kl'][-1]:.2f}")

            samples = model.sample(64, device=device)
            save_image(samples, f"{save_dir}/samples_epoch_{epoch:03d}.png", nrow=8)

    # Final save
    torch.save(model.state_dict(), f"{save_dir}/vae_final.pth")
    np.save(f"{save_dir}/history.npy", history)
    print(f"\nVAE training complete. Model saved to {save_dir}/")
    return model, history


if __name__ == "__main__":
    train_vae()
