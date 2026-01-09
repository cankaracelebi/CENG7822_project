#!/usr/bin/env python
"""
Train World Model Agent with PPO Controller
Based on Ha & Schmidhuber 2018 "World Models" architecture.

Pipeline:
1. Collect experience data (frames, actions, rewards)  
2. Train VAE on frames
3. Train MDN-RNN on latent sequences
4. Train Controller with PPO (using SB3) in latent/dream environment
5. Evaluate with video recording

Usage:
    source ~/miniconda3/bin/activate rlan
    cd /home/guava/Desktop/CENG7822_project
    python -m world_model.train_world_model_ppo
"""

import os
import sys
import argparse
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.vae import VAE
from world_model.mdn_rnn import MDNRNN
from world_model.data_collector import FrameDataCollector
from game.g2D import ShooterEnv


# ==============================================================================
# DREAM ENVIRONMENT (for PPO Controller Training)
# ==============================================================================

def preprocess_frame(obs, device):
    """Preprocess observation for VAE encoding."""
    import cv2
    # Handle different input shapes
    if len(obs.shape) == 2:
        # Grayscale - convert to RGB
        obs = np.stack([obs, obs, obs], axis=-1)
    
    frame = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    frame_tensor = torch.tensor(frame, dtype=torch.float32, device=device)
    
    if len(frame_tensor.shape) == 2:
        # Still 2D, add channel dim
        frame_tensor = frame_tensor.unsqueeze(-1).repeat(1, 1, 3)
    
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return frame_tensor


class DreamEnv(gym.Env):
    """
    Dream environment that uses VAE + MDN-RNN to simulate the game.
    """
    
    def __init__(
        self, 
        vae: VAE, 
        mdn_rnn: MDNRNN, 
        real_env,
        max_steps: int = 500,
        device: str = "cpu",
        use_real_env: bool = False,
    ):
        super().__init__()
        
        self.vae = vae.to(device)
        self.mdn_rnn = mdn_rnn.to(device)
        self.real_env = real_env
        self.max_steps = max_steps
        self.device = device
        self.use_real_env = use_real_env
        
        self.vae.eval()
        self.mdn_rnn.eval()
        
        latent_dim = vae.latent_dim
        hidden_dim = mdn_rnn.hidden_dim
        obs_dim = latent_dim + hidden_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_dim = int(np.prod(real_env.action_space.nvec))
        self.action_space = spaces.Discrete(self.action_dim)
        self.nvec = real_env.action_space.nvec
        
        self.z = None
        self.hidden = None
        self.step_count = 0
    
    def _get_obs(self):
        h = self.hidden[0].squeeze(0).squeeze(0)
        z = self.z.squeeze(0)
        obs = torch.cat([z, h], dim=-1).cpu().numpy()
        return obs.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.real_env.reset(seed=seed)
        self.step_count = 0
        
        frame_tensor = preprocess_frame(obs, self.device)
        
        with torch.no_grad():
            self.z = self.vae.encode(frame_tensor, deterministic=True)
            self.hidden = self.mdn_rnn.init_hidden(1, self.device)
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.step_count += 1
        
        action_onehot = F.one_hot(
            torch.tensor([action], device=self.device),
            num_classes=self.action_dim
        ).float()
        
        if self.use_real_env:
            env_action = self._decode_action(action)
            obs, reward, terminated, truncated, info = self.real_env.step(env_action)
            
            frame_tensor = preprocess_frame(obs, self.device)
            
            with torch.no_grad():
                self.z = self.vae.encode(frame_tensor, deterministic=True)
                _, _, _, _, _, self.hidden = self.mdn_rnn(self.z, action_onehot, self.hidden)
        else:
            with torch.no_grad():
                pi, mu, sigma, reward_pred, done_pred, self.hidden = self.mdn_rnn(
                    self.z, action_onehot, self.hidden
                )
                self.z = self.mdn_rnn.get_deterministic(pi, mu)
                reward = reward_pred.squeeze().item() if reward_pred is not None else 0.0
                done_prob = done_pred.squeeze().item() if done_pred is not None else 0.0
                terminated = done_prob > 0.5
                truncated = False
        
        if self.step_count >= self.max_steps:
            truncated = True
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _decode_action(self, action_idx):
        indices = []
        remaining = action_idx
        for n in reversed(self.nvec):
            indices.append(remaining % n)
            remaining //= n
        return np.array(list(reversed(indices)), dtype=np.int64)


# ==============================================================================
# METRICS CALLBACK
# ==============================================================================

class WorldModelMetricsCallback(BaseCallback):
    """Save training metrics to CSV for plotting."""
    
    def __init__(self, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "world_model_ppo_metrics.csv")
        self.episode_rewards = []
        self.episode_lengths = []
        
        os.makedirs(log_dir, exist_ok=True)
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'episode', 'reward', 'length', 'avg_reward_10', 'avg_length_10'])
        
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    infos = self.locals.get('infos', [{}])
                    if i < len(infos):
                        ep_info = infos[i].get('episode')
                        if ep_info:
                            self.episode_count += 1
                            reward = ep_info['r']
                            length = ep_info['l']
                            self.episode_rewards.append(reward)
                            self.episode_lengths.append(length)
                            
                            avg_r = np.mean(self.episode_rewards[-10:])
                            avg_l = np.mean(self.episode_lengths[-10:])
                            
                            with open(self.csv_path, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    self.num_timesteps, self.episode_count,
                                    reward, length, avg_r, avg_l
                                ])
                            
                            if self.episode_count % 50 == 0 and self.verbose:
                                print(f"[World Model] Episode {self.episode_count}, "
                                      f"Timestep {self.num_timesteps}, Avg Reward (10 ep): {avg_r:.2f}")
        return True


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_vae(vae, dataloader, epochs, lr, beta, device, log_dir):
    """Train VAE on frames with logging."""
    vae = vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    csv_path = os.path.join(log_dir, "vae_training.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'recon_loss', 'kl_loss'])
    
    for epoch in range(epochs):
        vae.train()
        epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
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
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, avg_recon, avg_kl])
        
        if (epoch + 1) % 10 == 0:
            print(f"VAE Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")
    
    return vae


def train_mdn_rnn(mdn_rnn, vae, frames, actions, rewards, dones, 
                  seq_len, epochs, batch_size, lr, device, log_dir):
    """Train MDN-RNN on latent sequences with logging."""
    mdn_rnn = mdn_rnn.to(device)
    vae = vae.to(device)
    vae.eval()
    optimizer = optim.Adam(mdn_rnn.parameters(), lr=lr)
    
    csv_path = os.path.join(log_dir, "mdn_rnn_training.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'mdn_loss', 'total_loss'])
    
    print("Encoding frames...")
    with torch.no_grad():
        frames_t = torch.tensor(frames, dtype=torch.float32, device=device)
        frames_t = frames_t.permute(0, 3, 1, 2)  # (N, C, H, W)
        latents = []
        for i in range(0, len(frames_t), 128):
            z = vae.encode(frames_t[i:i+128], deterministic=True)
            latents.append(z.cpu())
        latents = torch.cat(latents, dim=0).numpy()
    
    print(f"Latent shape: {latents.shape}")
    
    n_seq = len(latents) - seq_len
    z_seqs = np.array([latents[i:i+seq_len] for i in range(n_seq)])
    a_seqs = np.array([actions[i:i+seq_len] for i in range(n_seq)])
    r_seqs = np.array([rewards[i:i+seq_len] for i in range(n_seq)])
    d_seqs = np.array([dones[i:i+seq_len] for i in range(n_seq)])
    z_next = np.array([latents[i+1:i+1+seq_len] for i in range(n_seq)])
    
    dataset = TensorDataset(
        torch.tensor(z_seqs, dtype=torch.float32),
        torch.tensor(a_seqs, dtype=torch.float32),
        torch.tensor(r_seqs, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(d_seqs, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(z_next, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        mdn_rnn.train()
        epoch_mdn, epoch_total = 0, 0
        n_batches = 0
        
        for z_seq, a_seq, r_seq, d_seq, z_next_seq in loader:
            z_seq, a_seq = z_seq.to(device), a_seq.to(device)
            r_seq, d_seq = r_seq.to(device), d_seq.to(device)
            z_next_seq = z_next_seq.to(device)
            
            optimizer.zero_grad()
            pi, mu, sigma, r_pred, d_pred, _ = mdn_rnn(z_seq, a_seq)
            loss, metrics = mdn_rnn.loss_function(pi, mu, sigma, z_next_seq, r_pred, r_seq, d_pred, d_seq)
            loss.backward()
            optimizer.step()
            epoch_mdn += metrics['mdn_loss']
            epoch_total += metrics['total_loss']
            n_batches += 1
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_mdn / n_batches, epoch_total / n_batches])
        
        if (epoch + 1) % 10 == 0:
            print(f"MDN-RNN Epoch {epoch+1}/{epochs}: Loss={epoch_total/n_batches:.4f}")
    
    return mdn_rnn


def evaluate_with_video(model, vae, mdn_rnn, real_env, n_episodes, video_dir, device):
    """Evaluate model in real environment and record videos."""
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"\n=== Recording {n_episodes} Evaluation Episodes ===")
    
    results = []
    
    for ep in range(n_episodes):
        ep_video_dir = os.path.join(video_dir, f"episode_{ep}")
        
        # Create recording environment
        env = ShooterEnv(render_mode="rgb_array")
        env = RecordVideo(env, ep_video_dir, episode_trigger=lambda x: True,
                         name_prefix=f"world_model_ep{ep}")
        
        # Create dream env wrapper for observation encoding
        obs, _ = env.reset()
        frame_tensor = preprocess_frame(obs, device)
        
        with torch.no_grad():
            z = vae.encode(frame_tensor, deterministic=True)
            hidden = mdn_rnn.init_hidden(1, device)
        
        done = False
        ep_reward = 0
        steps = 0
        
        while not done:
            # Get observation for model
            h = hidden[0].squeeze(0).squeeze(0)
            z_flat = z.squeeze(0)
            model_obs = torch.cat([z_flat, h], dim=-1).cpu().numpy().astype(np.float32)
            
            # Predict action
            action, _ = model.predict(model_obs, deterministic=True)
            
            # Decode to env action
            action_int = int(action.item()) if hasattr(action, 'item') else int(action)
            indices = []
            remaining = action_int
            for n in reversed(real_env.action_space.nvec):
                indices.append(remaining % n)
                remaining //= n
            env_action = np.array(list(reversed(indices)), dtype=np.int64)
            
            # Step
            obs, reward, terminated, truncated, _ = env.step(env_action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
            
            if not done:
                frame_tensor = preprocess_frame(obs, device)
                action_dim = int(np.prod(real_env.action_space.nvec))
                action_onehot = F.one_hot(
                    torch.tensor([action_int], device=device), num_classes=action_dim
                ).float()
                
                with torch.no_grad():
                    z = vae.encode(frame_tensor, deterministic=True)
                    _, _, _, _, _, hidden = mdn_rnn(z, action_onehot, hidden)
        
        env.close()
        results.append({'episode': ep, 'reward': ep_reward, 'length': steps})
        print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Length={steps}")
    
    # Save results
    csv_path = os.path.join(video_dir, "evaluation_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'reward', 'length'])
        writer.writeheader()
        writer.writerows(results)
    
    avg_reward = np.mean([r['reward'] for r in results])
    print(f"\nAverage Reward: {avg_reward:.2f}")
    print(f"Videos saved to: {video_dir}")
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train World Model with PPO Controller")
    parser.add_argument("--collect-frames", type=int, default=20000)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--vae-epochs", type=int, default=50)
    parser.add_argument("--mdn-epochs", type=int, default=50)
    parser.add_argument("--ppo-timesteps", type=int, default=100000)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="./world_model_agent")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-mdn", action="store_true")
    parser.add_argument("--use-real-env", action="store_true")
    parser.add_argument("--eval-videos", type=int, default=5, help="Number of evaluation videos")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "logs")
    video_dir = os.path.join(args.save_dir, "videos")
    os.makedirs(log_dir, exist_ok=True)
    
    print("="*70)
    print("World Model Training with PPO Controller")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
    print(f"Log directory: {log_dir}")
    
    real_env = ShooterEnv(render_mode="rgb_array")
    action_dim = int(np.prod(real_env.action_space.nvec))
    
    # ==== Step 1: Collect Data ====
    print("\n=== Step 1: Collect Data ===")
    collector = FrameDataCollector(save_dir=args.save_dir, frame_size=(64,64), max_frames=args.collect_frames)
    
    if args.skip_collect and os.path.exists(os.path.join(args.save_dir, "collected_data.npz")):
        print("Loading existing data...")
        collector.load("collected_data.npz")
    else:
        collector.collect_random_episodes(real_env, args.n_episodes, 500, verbose=True)
        collector.save("collected_data.npz")
    
    print(f"Frames: {len(collector.frames)}")
    
    # ==== Step 2: Train VAE ====
    print("\n=== Step 2: Train VAE ===")
    vae = VAE(in_channels=3, latent_dim=args.latent_dim)
    vae_path = os.path.join(args.save_dir, "vae.pt")
    
    if args.skip_vae and os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, map_location=args.device))
        print("Loaded existing VAE")
    else:
        vae_dataset = collector.get_vae_dataset()
        vae_loader = DataLoader(vae_dataset, batch_size=args.batch_size, shuffle=True)
        vae = train_vae(vae, vae_loader, args.vae_epochs, args.lr, 1.0, args.device, log_dir)
        torch.save(vae.state_dict(), vae_path)
    
    # ==== Step 3: Train MDN-RNN ====
    print("\n=== Step 3: Train MDN-RNN ===")
    mdn_rnn = MDNRNN(latent_dim=args.latent_dim, action_dim=action_dim, hidden_dim=args.hidden_dim)
    mdn_path = os.path.join(args.save_dir, "mdn_rnn.pt")
    
    if args.skip_mdn and os.path.exists(mdn_path):
        mdn_rnn.load_state_dict(torch.load(mdn_path, map_location=args.device))
        print("Loaded existing MDN-RNN")
    else:
        frames = np.array(collector.frames)
        actions_raw = np.array(collector.actions)
        
        actions = np.zeros((len(actions_raw), action_dim), dtype=np.float32)
        for i, a in enumerate(actions_raw):
            flat_idx = np.ravel_multi_index(a, real_env.action_space.nvec)
            actions[i, flat_idx] = 1.0
        
        mdn_rnn = train_mdn_rnn(
            mdn_rnn, vae, frames, actions,
            np.array(collector.rewards), np.array(collector.dones),
            32, args.mdn_epochs, args.batch_size, args.lr, args.device, log_dir
        )
        torch.save(mdn_rnn.state_dict(), mdn_path)
    
    # ==== Step 4: Train Controller with PPO ====
    print("\n=== Step 4: Train Controller with PPO ===")
    
    dream_env = DreamEnv(vae=vae, mdn_rnn=mdn_rnn, real_env=real_env,
                         max_steps=500, device=args.device, use_real_env=args.use_real_env)
    dream_env = Monitor(dream_env)
    
    env = DummyVecEnv([lambda: dream_env])
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(args.save_dir, "tb"))
    
    callbacks = [
        WorldModelMetricsCallback(log_dir=log_dir, verbose=1),
    ]
    
    print(f"Training PPO controller for {args.ppo_timesteps} timesteps...")
    model.learn(total_timesteps=args.ppo_timesteps, callback=callbacks)
    model.save(os.path.join(args.save_dir, "ppo_controller_final"))
    
    # ==== Step 5: Evaluate with Videos ====
    print("\n=== Step 5: Evaluate with Video Recording ===")
    evaluate_with_video(model, vae, mdn_rnn, real_env, args.eval_videos, video_dir, args.device)
    
    real_env.close()
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Models saved to: {args.save_dir}")
    print(f"Training logs: {log_dir}")
    print(f"Evaluation videos: {video_dir}")


if __name__ == "__main__":
    main()
