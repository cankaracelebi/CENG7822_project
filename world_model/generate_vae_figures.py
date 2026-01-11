#!/usr/bin/env python
"""
Generate publication-quality VAE reconstruction visualizations.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.vae import VAE


def generate_report_reconstructions(model_dir: str, output_dir: str, device: str = "cpu"):
    """Generate high-quality reconstruction images for report."""
    
    # Load VAE
    vae_path = os.path.join(model_dir, "vae.pt")
    vae = VAE(in_channels=3, latent_dim=32)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.to(device).eval()
    print(f"Loaded VAE from {vae_path}")
    
    # Load data
    data_path = os.path.join(model_dir, "collected_data.npz")
    if not os.path.exists(data_path):
        data_path = "./world_model_checkpoints/collected_data.npz"
    data = np.load(data_path)
    frames = data['frames']
    print(f"Loaded {len(frames)} frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select diverse frames (different timesteps)
    n_samples = 8
    indices = np.linspace(0, len(frames)-1, n_samples, dtype=int)
    sample_frames = frames[indices]
    
    # Convert to tensor
    frames_t = torch.tensor(sample_frames, dtype=torch.float32, device=device)
    frames_t = frames_t.permute(0, 3, 1, 2)  # (N, C, H, W)
    
    with torch.no_grad():
        recon, mu, logvar = vae(frames_t)
    
    mse = ((recon - frames_t) ** 2).mean().item()
    
    # === Figure 1: Side-by-side comparison grid ===
    fig, axes = plt.subplots(2, n_samples, figsize=(16, 4))
    
    for i in range(n_samples):
        # Original
        orig = frames_t[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(orig)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12, rotation=0, ha='right', va='center')
        
        # Reconstruction  
        rec = recon[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(rec)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12, rotation=0, ha='right', va='center')
    
    plt.suptitle(f'VAE Reconstruction Results (MSE: {mse:.6f})', fontsize=14)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "vae_reconstruction_grid.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path1}")
    
    # === Figure 2: Latent space interpolation ===
    z1 = vae.encode(frames_t[0:1], deterministic=True)
    z2 = vae.encode(frames_t[-1:], deterministic=True)
    
    n_interp = 8
    alphas = np.linspace(0, 1, n_interp)
    
    fig, axes = plt.subplots(1, n_interp + 2, figsize=(14, 2))
    
    # Start frame
    axes[0].imshow(frames_t[0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Start', fontsize=10)
    axes[0].axis('off')
    
    # Interpolated
    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * z1 + alpha * z2
        with torch.no_grad():
            decoded = vae.decode(z_interp)
        axes[i + 1].imshow(decoded[0].permute(1, 2, 0).cpu().numpy())
        axes[i + 1].set_title(f'α={alpha:.2f}', fontsize=9)
        axes[i + 1].axis('off')
    
    # End frame
    axes[-1].imshow(frames_t[-1].permute(1, 2, 0).cpu().numpy())
    axes[-1].set_title('End', fontsize=10)
    axes[-1].axis('off')
    
    plt.suptitle('Latent Space Interpolation', fontsize=12)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "vae_latent_interpolation.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path2}")
    
    # === Figure 3: Latent space visualization (t-SNE) ===
    try:
        from sklearn.manifold import TSNE
        
        # Encode more frames for t-SNE
        n_tsne = 1000
        indices_tsne = np.random.choice(len(frames), n_tsne, replace=False)
        frames_tsne = torch.tensor(frames[indices_tsne], dtype=torch.float32, device=device)
        frames_tsne = frames_tsne.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            latents = vae.encode(frames_tsne, deterministic=True).cpu().numpy()
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        latents_2d = tsne.fit_transform(latents)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=indices_tsne, 
                            cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax, label='Frame Index (Time)')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('VAE Latent Space Visualization (t-SNE)')
        
        path3 = os.path.join(output_dir, "vae_latent_tsne.png")
        plt.savefig(path3, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path3}")
    except ImportError:
        print("sklearn not available, skipping t-SNE visualization")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    return mse


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./world_model_agent_long")
    parser.add_argument("--output-dir", default="./report_figures/world_model")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    generate_report_reconstructions(args.model_dir, args.output_dir, args.device)
