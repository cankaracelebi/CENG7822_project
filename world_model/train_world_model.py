#!/usr/bin/env python
"""
Train World Model: VAE + Dynamics
1. Collect frames using random policy
2. Train VAE on collected frames
3. Train dynamics model on latent transitions
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.vae import VAE
from world_model.dynamics import DynamicsModel
from world_model.data_collector import FrameDataCollector
from game.g2D import ShooterEnv


def train_vae(
    vae: VAE,
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Train VAE on collected frames"""
    vae = vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    history = {'loss': [], 'recon': [], 'kl': []}
    
    for epoch in range(epochs):
        vae.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        n_batches = 0
        
        for batch in dataloader:
            frames = batch[0].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = vae(frames)
            loss, metrics = VAE.loss_function(recon, frames, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            
            epoch_loss += metrics['loss']
            epoch_recon += metrics['recon_loss']
            epoch_kl += metrics['kl_loss']
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        
        history['loss'].append(avg_loss)
        history['recon'].append(avg_recon)
        history['kl'].append(avg_kl)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")
    
    return history


def train_dynamics(
    dynamics: DynamicsModel,
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """Train dynamics model on latent transitions"""
    dynamics = dynamics.to(device)
    optimizer = optim.Adam(dynamics.parameters(), lr=lr)
    
    history = {'total_loss': [], 'z_loss': [], 'reward_loss': [], 'done_loss': []}
    
    for epoch in range(epochs):
        dynamics.train()
        epoch_metrics = {k: 0.0 for k in history.keys()}
        n_batches = 0
        
        for batch in dataloader:
            z_t, actions, rewards, z_next, dones = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            z_pred, r_pred, d_pred = dynamics(z_t, actions)
            loss, metrics = dynamics.loss_function(
                z_pred, z_next,
                r_pred, rewards,
                d_pred, dones,
            )
            loss.backward()
            optimizer.step()
            
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]
            n_batches += 1
        
        for k in history:
            avg = epoch_metrics[k] / n_batches if n_batches > 0 else 0
            history[k].append(avg)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={history['total_loss'][-1]:.4f}, Z={history['z_loss'][-1]:.4f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train World Model")
    parser.add_argument("--collect-frames", type=int, default=10000, help="Number of frames to collect")
    parser.add_argument("--n-episodes", type=int, default=100, help="Episodes for collection")
    parser.add_argument("--vae-epochs", type=int, default=50, help="VAE training epochs")
    parser.add_argument("--dynamics-epochs", type=int, default=50, help="Dynamics training epochs")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="./world_model_checkpoints")
    parser.add_argument("--skip-collect", action="store_true", help="Skip data collection, load existing")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create environment
    env = ShooterEnv(render_mode="rgb_array")
    action_dim = int(np.prod(env.action_space.nvec))  # 5 * 2 * 8 = 80
    
    print(f"Environment action space: {env.action_space}")
    print(f"Flattened action dim: {action_dim}")
    print(f"Device: {args.device}")
    
    # Step 1: Collect data
    collector = FrameDataCollector(
        save_dir=args.save_dir,
        frame_size=(64, 64),
        max_frames=args.collect_frames,
    )
    
    data_file = os.path.join(args.save_dir, "collected_data.npz")
    if args.skip_collect and os.path.exists(data_file):
        print("\n=== Loading existing data ===")
        collector.load("collected_data.npz")
    else:
        print("\n=== Collecting frames with random policy ===")
        collector.collect_random_episodes(
            env=env,
            n_episodes=args.n_episodes,
            max_steps_per_episode=500,
            verbose=True,
        )
        collector.save("collected_data.npz")
    
    env.close()
    
    print(f"Total frames: {len(collector.frames)}")
    
    # Step 2: Train VAE
    print("\n=== Training VAE ===")
    vae = VAE(in_channels=3, latent_dim=args.latent_dim)
    vae_dataset = collector.get_vae_dataset()
    vae_loader = DataLoader(vae_dataset, batch_size=args.batch_size, shuffle=True)
    
    vae_history = train_vae(
        vae=vae,
        dataloader=vae_loader,
        epochs=args.vae_epochs,
        lr=args.lr,
        device=args.device,
    )
    
    torch.save(vae.state_dict(), os.path.join(args.save_dir, "vae.pt"))
    print(f"VAE saved to {args.save_dir}/vae.pt")
    
    # Step 3: Train Dynamics
    print("\n=== Training Dynamics Model ===")
    vae = vae.to(args.device)
    dynamics = DynamicsModel(
        latent_dim=args.latent_dim,
        action_dim=3,  # Raw action before flattening [move, shoot, direction]
        predict_reward=True,
        predict_done=True,
    )
    
    dynamics_dataset = collector.get_dynamics_dataset(vae.cpu())
    dynamics_loader = DataLoader(dynamics_dataset, batch_size=args.batch_size, shuffle=True)
    
    dynamics_history = train_dynamics(
        dynamics=dynamics,
        dataloader=dynamics_loader,
        epochs=args.dynamics_epochs,
        lr=args.lr,
        device=args.device,
    )
    
    torch.save(dynamics.state_dict(), os.path.join(args.save_dir, "dynamics.pt"))
    print(f"Dynamics saved to {args.save_dir}/dynamics.pt")
    
    print("\n=== Training Complete ===")
    print(f"VAE final loss: {vae_history['loss'][-1]:.4f}")
    print(f"Dynamics final loss: {dynamics_history['total_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
