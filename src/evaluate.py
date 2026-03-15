import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import linalg
import os
import warnings
warnings.filterwarnings("ignore")


# 1. INCEPTION NETWORK (LeNet-style for MNIST — Inception not suited to 28×28)
class MNISTClassifier(nn.Module):
    """
    Lightweight CNN trained on MNIST.
    Used as a proxy for the Inception network when computing FID/IS on MNIST.
    Extracts penultimate-layer features (dim=256) for FID computation.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128,3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def get_features(self, x):
        """Return 256-dim feature vector (penultimate layer)."""
        h = self.features(x)
        h = h.view(h.size(0), -1)
        # Run through first linear only
        h = F.relu(self.classifier[1](h))   # (B, 256)
        return h

    def forward(self, x):
        h = self.features(x)
        return self.classifier(h)


def get_or_train_classifier(device, save_path="data/mnist_clf.pth", epochs=5):
    """Load or train the MNIST proxy classifier."""
    clf = MNISTClassifier().to(device)

    if os.path.exists(save_path):
        clf.load_state_dict(torch.load(save_path, map_location=device))
        clf.eval()
        print("  Loaded pre-trained MNIST classifier.")
        return clf

    print("  Training MNIST classifier for FID/IS evaluation …")
    tf     = transforms.ToTensor()
    loader = DataLoader(datasets.MNIST("data", train=True, download=True, transform=tf),
                        batch_size=256, shuffle=True)
    opt    = torch.optim.Adam(clf.parameters())
    clf.train()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"    Clf epoch {ep+1}/{epochs}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(clf.state_dict(), save_path)
    clf.eval()
    return clf


# FEATURE EXTRACTION
@torch.no_grad()
def extract_features(images: torch.Tensor, clf: MNISTClassifier, device: str,
                     batch_size: int = 256) -> np.ndarray:
    """Extract 256-dim feature vectors for a set of images."""
    clf.eval()
    feats = []
    ds = TensorDataset(images)
    loader = DataLoader(ds, batch_size=batch_size)
    for (x,) in loader:
        x = x.to(device)
        if x.shape[1] == 1:           # grayscale
            x = x.repeat(1, 1, 1, 1)  # keep 1-channel (clf is 1-ch)
        feats.append(clf.get_features(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def extract_logits(images: torch.Tensor, clf: MNISTClassifier, device: str,
                   batch_size: int = 256) -> np.ndarray:
    """Extract softmax probabilities for Inception Score calculation."""
    clf.eval()
    probs = []
    ds = TensorDataset(images)
    loader = DataLoader(ds, batch_size=batch_size)
    for (x,) in loader:
        x = x.to(device)
        p = torch.softmax(clf(x), dim=-1)
        probs.append(p.cpu().numpy())
    return np.concatenate(probs, axis=0)


# FID SCORE
def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """
    Fréchet Inception Distance:
      FID = || μ_r - μ_g ||² + Tr(Σ_r + Σ_g - 2*(Σ_r Σ_g)^0.5)
    Lower is better; 0 = identical distributions.
    """
    mu_r, sig_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, sig_g = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

    diff   = mu_r - mu_g
    # Matrix square root via eigendecomposition (numerically stable)
    covmean, _ = linalg.sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sig_r + sig_g - 2 * covmean)
    return float(fid)


# INCEPTION SCORE
def compute_inception_score(probs: np.ndarray, splits: int = 10) -> tuple[float, float]:
    """
    Inception Score: IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )
    Uses split-average for variance estimation.
    Higher is better.
    """
    N    = probs.shape[0]
    part = N // splits
    scores = []
    for k in range(splits):
        part_probs = probs[k * part: (k + 1) * part]
        p_y        = part_probs.mean(0, keepdims=True)   # marginal
        kl         = part_probs * (np.log(part_probs + 1e-10) - np.log(p_y + 1e-10))
        kl_mean    = kl.sum(1).mean()
        scores.append(np.exp(kl_mean))
    return float(np.mean(scores)), float(np.std(scores))


#  MAIN EVALUATION FUNCTION
def evaluate_models(
    vae_model=None,
    ddpm_model=None,
    n_samples: int = 10_000,
    save_dir:  str = "results/eval",
    device:    str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # Load proxy classifier
    clf = get_or_train_classifier(device)

    print("\nExtracting real data features …")
    tf_real   = transforms.ToTensor()
    real_ds   = datasets.MNIST("data", train=False, download=True, transform=tf_real)
    real_imgs = torch.stack([real_ds[i][0] for i in range(min(n_samples, len(real_ds)))])
    real_feats = extract_features(real_imgs, clf, device)
    real_probs = extract_logits(real_imgs, clf, device)
    real_is, real_is_std = compute_inception_score(real_probs)
    print(f"  Real data  IS: {real_is:.3f} ± {real_is_std:.3f}")

    results = {}

    #VAE evaluation
    if vae_model is not None:
        print("\nEvaluating VAE …")
        vae_model.eval()
        vae_imgs = []
        with torch.no_grad():
            for _ in range(n_samples // 64 + 1):
                samples = vae_model.sample(64, device=device)
                vae_imgs.append(samples.cpu())
        vae_imgs = torch.cat(vae_imgs, 0)[:n_samples]

        vae_feats = extract_features(vae_imgs, clf, device)
        vae_probs = extract_logits(vae_imgs, clf, device)
        fid_vae   = compute_fid(real_feats, vae_feats)
        is_vae, is_vae_std = compute_inception_score(vae_probs)

        results["VAE"] = {"FID": fid_vae, "IS": is_vae, "IS_std": is_vae_std}
        print(f"  VAE — FID: {fid_vae:.2f}  |  IS: {is_vae:.3f} ± {is_vae_std:.3f}")

        # Save qualitative grid
        grid = make_grid(vae_imgs[:64], nrow=8, normalize=True)
        save_image(grid, f"{save_dir}/vae_samples_eval.png")

    # DDPM evaluation 
    if ddpm_model is not None:
        print("\nEvaluating DDPM … (this may take several minutes)")
        ddpm_model.eval()
        ddpm_imgs = []
        with torch.no_grad():
            for _ in range(n_samples // 16 + 1):
                samples = ddpm_model.sample(16, device=device)
                ddpm_imgs.append(samples.cpu())
        ddpm_imgs = torch.cat(ddpm_imgs, 0)[:n_samples]

        ddpm_feats = extract_features(ddpm_imgs, clf, device)
        ddpm_probs = extract_logits(ddpm_imgs, clf, device)
        fid_ddpm   = compute_fid(real_feats, ddpm_feats)
        is_ddpm, is_ddpm_std = compute_inception_score(ddpm_probs)

        results["DDPM"] = {"FID": fid_ddpm, "IS": is_ddpm, "IS_std": is_ddpm_std}
        print(f"  DDPM — FID: {fid_ddpm:.2f}  |  IS: {is_ddpm:.3f} ± {is_ddpm_std:.3f}")

        grid = make_grid(ddpm_imgs[:64], nrow=8, normalize=True)
        save_image(grid, f"{save_dir}/ddpm_samples_eval.png")

    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for model_name, scores in results.items():
        print(f"  {model_name:5s}  FID={scores['FID']:.2f}   "
              f"IS={scores['IS']:.3f} ± {scores['IS_std']:.3f}")
    print("=" * 50)

    np.save(f"{save_dir}/metrics.npy", results)
    return results


# INTERPOLATION DEMO (latent space)

@torch.no_grad()
def latent_interpolation(vae_model, n_steps: int = 10, device: str = "cpu",
                         save_path: str = "results/vae_interpolation.png"):
    """Spherical interpolation between two random latent vectors."""
    vae_model.eval()
    z1 = torch.randn(1, vae_model.encoder.latent_dim, device=device)
    z2 = torch.randn(1, vae_model.encoder.latent_dim, device=device)

    alphas = torch.linspace(0, 1, n_steps, device=device)
    z_interp = []
    for a in alphas:
        z = (1 - a) * z1 + a * z2
        z = F.normalize(z, dim=-1) * z1.norm()
        z_interp.append(z)
    z_interp = torch.cat(z_interp, dim=0)

    imgs = vae_model.decoder(z_interp)
    save_image(imgs, save_path, nrow=n_steps)
    print(f"  Saved latent interpolation → {save_path}")


if __name__ == "__main__":
    from src.vae import VAE
    from src.ddpm import DDPM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    vae = VAE(latent_dim=128).to(device)
    vae.load_state_dict(torch.load("results/vae/vae_final.pth", map_location=device))

    ddpm = DDPM().to(device)
    ddpm.load_state_dict(torch.load("results/ddpm/ddpm_final.pth", map_location=device))

    evaluate_models(vae, ddpm, n_samples=5000, device=device)
    latent_interpolation(vae, device=device)
