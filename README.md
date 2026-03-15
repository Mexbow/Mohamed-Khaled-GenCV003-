# GenCV003 — VAE vs DDPM: Generative Models from Scratch

A from-scratch implementation and comparison of **Variational Autoencoders (VAE)** and **Denoising Diffusion Probabilistic Models (DDPM)** on the MNIST dataset.

---

## Repository Structure

```
.
├── src/
│   ├── vae.py          # VAE encoder, decoder, reparameterization, training loop
│   ├── ddpm.py         # U-Net noise predictor, forward/reverse diffusion, training loop
│   └── evaluate.py     # FID, Inception Score, latent interpolation
├── train.py            # Main CLI entrypoint
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Mexbow/Mohamed-Khaled-GenCV003-.git
cd Mohamed-Khaled-GenCV003-

# 2. Create environment (Python 3.10+)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

GPU is strongly recommended for DDPM training. CUDA 11.8+ or ROCm 5.4+ supported.

---

## Reproducing Results

### Train VAE (≈ 10 min on GPU)
```bash
python train.py --model vae --epochs_vae 50 --batch_size 128 --latent_dim 128
```

### Train DDPM (≈ 60–90 min on GPU)
```bash
python train.py --model ddpm --epochs_ddpm 100 --batch_size 128 --T 1000
```

### Train Both + Evaluate
```bash
python train.py --model both --epochs_vae 50 --epochs_ddpm 100
```

### Evaluate Pre-trained Models
```bash
python train.py --model evaluate_only --n_eval_samples 5000
```

Expected output locations:
```
results/
├── vae/
│   ├── samples_epoch_*.png    # training progress grids
│   ├── vae_final.pth          # saved weights
│   └── history.npy            # loss curves
├── ddpm/
│   ├── samples_epoch_*.png
│   ├── ddpm_final.pth
│   └── history.npy
└── eval/
    ├── vae_samples_eval.png   # 64-image qualitative grid
    ├── ddpm_samples_eval.png
    └── metrics.npy            # FID and IS scores
```

---

## Results (Measured — MNIST, 5,000 samples)

| Model | FID ↓ | IS ↑ | Epochs | Training Time |
|-------|-------|------|--------|---------------|
| Real MNIST (baseline) | — | 9.323 ± 0.067 | — | — |
| VAE (latent=32, β=1) | 101.90 | 6.683 ± 0.096 | 50 | ~10 min (GPU) |
| DDPM (T=1000) | 31.08 | 8.975 ± 0.122 | 100 | ~75 min (GPU) |

**Key observations:**
- DDPM IS of 8.975 is 96% of real data quality (9.323) — excellent mode coverage and sharpness
- VAE FID of 101.90 reflects blurry reconstructions from the BCE loss and mild posterior collapse (KL plateaued at ~25 nats). Reducing `latent_dim` from 128 → 32 improved FID from 124.88 → 101.90
- Further VAE improvement: train with `--epochs_vae 100 --beta 0.5` to push FID toward 30–50
- Further DDPM improvement: DDPM loss converged at 0.0200 by epoch 80 — a cosine noise schedule (Nichol & Dhariwal 2021) would push FID below 15

---

## Architecture Overview

### VAE
- **Encoder**: 3-layer Conv → BatchNorm → LeakyReLU → two linear heads (μ, log σ²)
- **Reparameterization**: z = μ + ε·σ, ε ~ N(0,I)
- **Decoder**: Linear → Transposed-Conv stack → Sigmoid
- **Loss**: BCE reconstruction + β·KL divergence

### DDPM
- **Forward process**: Adds Gaussian noise over T=1000 steps via linear β schedule
- **Reverse process**: U-Net predicts noise ε_θ(x_t, t) at each step
- **U-Net**: Residual blocks + sinusoidal time embedding + self-attention at bottleneck
- **Sampling**: Ancestral sampling x_T → x_0 (1000 denoising steps)

---

## Key Differences

| Aspect | VAE | DDPM |
|---|---|---|
| Inference speed | Fast (single forward pass) | Slow (1000 denoising steps) |
| Sample quality | Moderate (blurrier) | High (sharper, more diverse) |
| Latent space | Explicit, structured, interpolable | Implicit (Gaussian noise) |
| Training stability | Stable | Stable (score-matching) |
| Mode coverage | May drop modes | Excellent mode coverage |

---

## Citation

```bibtex
@article{ho2020ddpm,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={NeurIPS}, year={2020}
}
@article{kingma2013vae,
  title={Auto-Encoding Variational Bayes},
  author={Kingma, Diederik P and Welling, Max},
  journal={ICLR}, year={2014}
}
```
