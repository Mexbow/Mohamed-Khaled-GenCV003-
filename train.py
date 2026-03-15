import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.vae      import train_vae, VAE
from src.ddpm     import train_ddpm, DDPM
from src.evaluate import evaluate_models, latent_interpolation


def parse_args():
    p = argparse.ArgumentParser(description="GenCV003 — VAE & DDPM Trainer")
    p.add_argument("--model",          type=str, default="both",
                   choices=["vae", "ddpm", "both", "evaluate_only"])
    p.add_argument("--epochs_vae",     type=int, default=50)
    p.add_argument("--epochs_ddpm",    type=int, default=100)
    p.add_argument("--batch_size",     type=int, default=128)
    p.add_argument("--latent_dim",     type=int, default=128)
    p.add_argument("--lr_vae",         type=float, default=1e-3)
    p.add_argument("--lr_ddpm",        type=float, default=2e-4)
    p.add_argument("--beta",           type=float, default=1.0,
                   help="KL weight for β-VAE (default=1 → vanilla VAE)")
    p.add_argument("--T",              type=int, default=1000,
                   help="Diffusion timesteps")
    p.add_argument("--n_eval_samples", type=int, default=5000)
    p.add_argument("--device",         type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  GenCV003 — Generative Models: VAE vs DDPM")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    vae_model  = None
    ddpm_model = None

    # ── Train VAE ─────────────────────────────────────────────────────────
    if args.model in ("vae", "both"):
        print("[1/2] Training VAE …")
        vae_model, _ = train_vae(
            epochs     = args.epochs_vae,
            batch_size = args.batch_size,
            latent_dim = args.latent_dim,
            lr         = args.lr_vae,
            beta       = args.beta,
            device     = device,
        )

    # ── Train DDPM ────────────────────────────────────────────────────────
    if args.model in ("ddpm", "both"):
        print("\n[2/2] Training DDPM …")
        ddpm_model, _ = train_ddpm(
            epochs     = args.epochs_ddpm,
            batch_size = args.batch_size,
            lr         = args.lr_ddpm,
            T          = args.T,
            device     = device,
        )

    # ── Load existing models if evaluate_only ─────────────────────────────
    if args.model == "evaluate_only":
        vae_path  = "results/vae/vae_final.pth"
        ddpm_path = "results/ddpm/ddpm_final.pth"
        if os.path.exists(vae_path):
            vae_model = VAE(latent_dim=args.latent_dim).to(device)
            vae_model.load_state_dict(torch.load(vae_path, map_location=device))
        if os.path.exists(ddpm_path):
            ddpm_model = DDPM(T=args.T).to(device)
            ddpm_model.load_state_dict(torch.load(ddpm_path, map_location=device))

    # ── Evaluation ────────────────────────────────────────────────────────
    if vae_model is not None or ddpm_model is not None:
        print("\n[3/3] Running evaluation …")
        results = evaluate_models(
            vae_model  = vae_model,
            ddpm_model = ddpm_model,
            n_samples  = args.n_eval_samples,
            device     = device,
        )

        if vae_model is not None:
            latent_interpolation(vae_model, device=device)

    print("\nAll done! Results saved to results/")


if __name__ == "__main__":
    main()
