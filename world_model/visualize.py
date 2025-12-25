#!/usr/bin/env python
"""
Visualization utilities for World Model.
- View VAE reconstructions (original vs decoded)
- Visualize latent space
- Imagination rollouts
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def visualize_vae_reconstruction(
    vae,
    frames: np.ndarray,
    n_samples: int = 8,
    save_path: str = None,
    show: bool = True,
):
    """
    Show original frames vs VAE reconstructions side by side.
    
    Args:
        vae: Trained VAE model
        frames: Array of frames (N, H, W, C) in [0, 1] range
        n_samples: Number of samples to show
        save_path: Path to save figure
        show: Whether to display the figure
    """
    vae.eval()
    device = next(vae.parameters()).device
    
    # Select random samples
    indices = np.random.choice(len(frames), min(n_samples, len(frames)), replace=False)
    samples = frames[indices]
    
    # Convert to tensor (N, C, H, W)
    samples_tensor = torch.tensor(samples, dtype=torch.float32).permute(0, 3, 1, 2)
    samples_tensor = samples_tensor.to(device)
    
    # Encode and decode
    with torch.no_grad():
        recon, mu, logvar = vae(samples_tensor)
        z = vae.encode(samples_tensor, deterministic=True)
    
    # Convert back to numpy
    recon_np = recon.cpu().permute(0, 2, 3, 1).numpy()
    
    # Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(np.clip(samples[i], 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstruction
        axes[1, i].imshow(np.clip(recon_np[i], 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.suptitle(f'VAE Reconstruction (Latent dim: {vae.latent_dim})', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    # Return metrics
    mse = np.mean((samples - recon_np) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    return mse


def visualize_latent_interpolation(
    vae,
    frame1: np.ndarray,
    frame2: np.ndarray,
    n_steps: int = 10,
    save_path: str = None,
    show: bool = True,
):
    """
    Interpolate between two frames in latent space.
    Shows smooth transition from frame1 to frame2.
    """
    vae.eval()
    device = next(vae.parameters()).device
    
    # Preprocess frames
    def preprocess(frame):
        if frame.shape[:2] != (64, 64):
            frame = cv2.resize(frame, (64, 64))
        if frame.max() > 1:
            frame = frame.astype(np.float32) / 255.0
        t = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return t.to(device)
    
    f1 = preprocess(frame1)
    f2 = preprocess(frame2)
    
    # Encode both
    with torch.no_grad():
        z1 = vae.encode(f1, deterministic=True)
        z2 = vae.encode(f2, deterministic=True)
    
    # Interpolate in latent space
    alphas = np.linspace(0, 1, n_steps)
    interpolated = []
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            recon = vae.decode(z_interp)
            interpolated.append(recon.cpu().squeeze().permute(1, 2, 0).numpy())
    
    # Plot
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2))
    
    for i, img in enumerate(interpolated):
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].axis('off')
        axes[i].set_title(f'α={alphas[i]:.1f}', fontsize=8)
    
    plt.suptitle('Latent Space Interpolation', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def visualize_imagination_rollout(
    vae,
    dynamics,
    start_frame: np.ndarray,
    actions: np.ndarray,
    save_path: str = None,
    show: bool = True,
):
    """
    Show imagined rollout: start from real frame, predict future in latent space.
    
    Args:
        vae: Trained VAE
        dynamics: Trained dynamics model
        start_frame: Initial frame (H, W, C)
        actions: Sequence of actions (T, action_dim)
    """
    vae.eval()
    dynamics.eval()
    device = next(vae.parameters()).device
    
    # Preprocess start frame
    if start_frame.shape[:2] != (64, 64):
        start_frame = cv2.resize(start_frame, (64, 64))
    if start_frame.max() > 1:
        start_frame = start_frame.astype(np.float32) / 255.0
    
    frame_tensor = torch.tensor(start_frame, dtype=torch.float32)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Encode start frame
    with torch.no_grad():
        z = vae.encode(frame_tensor, deterministic=True)
    
    # Rollout in imagination
    actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
    T = len(actions)
    
    imagined_frames = [start_frame]
    
    with torch.no_grad():
        for t in range(T):
            a = actions_tensor[t:t+1]
            z, _, _ = dynamics(z, a)
            recon = vae.decode(z)
            img = recon.cpu().squeeze().permute(1, 2, 0).numpy()
            imagined_frames.append(img)
    
    # Plot
    n_frames = len(imagined_frames)
    fig, axes = plt.subplots(1, n_frames, figsize=(2 * n_frames, 2))
    
    for i, img in enumerate(imagined_frames):
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title('Start', fontsize=8)
        else:
            axes[i].set_title(f't+{i}', fontsize=8)
    
    plt.suptitle('Imagination Rollout', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def test_with_environment():
    """Test visualization with actual game frames"""
    from game.g2D import ShooterEnv
    from world_model.vae import VAE
    
    print("Creating environment and VAE...")
    env = ShooterEnv(render_mode="rgb_array")
    vae = VAE(in_channels=3, latent_dim=32)
    
    print("Collecting sample frames...")
    frames = []
    obs, _ = env.reset()
    
    for _ in range(20):
        frame = env.render()
        # Resize and normalize
        frame = cv2.resize(frame, (64, 64))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
        
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
    frames = np.array(frames)
    
    print(f"Collected {len(frames)} frames, shape: {frames[0].shape}")
    
    # Test reconstruction (with untrained VAE - will be blurry)
    print("\nNote: VAE is untrained, reconstructions will be poor.")
    print("Train the VAE first for meaningful reconstructions.\n")
    
    visualize_vae_reconstruction(
        vae, 
        frames, 
        n_samples=min(8, len(frames)),
        save_path="./world_model_checkpoints/reconstruction_test.png",
        show=False,
    )
    
    # Test interpolation
    if len(frames) >= 2:
        visualize_latent_interpolation(
            vae,
            frames[0],
            frames[-1],
            n_steps=8,
            save_path="./world_model_checkpoints/interpolation_test.png",
            show=False,
        )
    
    print("Visualizations saved to ./world_model_checkpoints/")


if __name__ == "__main__":
    import os
    os.makedirs("./world_model_checkpoints", exist_ok=True)
    test_with_environment()
