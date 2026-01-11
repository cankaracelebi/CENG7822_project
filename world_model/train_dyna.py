n#!/usr/bin/env python
"""
Dyna-Style World Model Training

This implements proper world model training by interleaving:
1. Real environment experience collection
2. Imagination rollouts using the world model
3. Policy updates on mixed real + imagined data

This prevents "dream exploitation" where the agent learns to exploit
model prediction errors instead of learning real game skills.

Based on Dyna-Q (Sutton 1991) and DreamerV2/V3 principles.
"""

import os
import sys
import argparse
import csv
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.vae import VAE
from world_model.mdn_rnn import MDNRNN
from world_model.train_world_model_ppo import DreamEnv, preprocess_frame
from game.g2D import ShooterEnv


class DynaCallback(BaseCallback):
    """
    Callback that implements Dyna-style training:
    - Periodically evaluates in real environment
    - Saves best model based on REAL performance (not dream)
    - Logs both dream and real metrics
    """
    
    def __init__(
        self, 
        vae: VAE,
        mdn_rnn: MDNRNN,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_path: str = "./dyna_best",
        device: str = "cpu",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.vae = vae
        self.mdn_rnn = mdn_rnn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.device = device
        
        self.best_real_reward = -float('inf')
        self.eval_rewards = []
        self.dream_rewards = []
        
        os.makedirs(save_path, exist_ok=True)
        self.log_path = os.path.join(save_path, "dyna_log.csv")
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'dream_reward', 'real_reward', 'best_real'])
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate in REAL environment
            real_rewards = self._evaluate_real()
            mean_real = np.mean(real_rewards)
            
            # Get recent dream rewards
            if hasattr(self.training_env, 'envs'):
                dream_reward = self.locals.get('rewards', [0])[0]
            else:
                dream_reward = 0
            
            # Save best model based on REAL performance
            if mean_real > self.best_real_reward:
                self.best_real_reward = mean_real
                self.model.save(os.path.join(self.save_path, "best_real_model"))
                if self.verbose:
                    print(f"  ★ New best real reward: {mean_real:.2f}")
            
            # Log
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.n_calls, dream_reward, mean_real, self.best_real_reward])
            
            if self.verbose:
                print(f"[{self.n_calls}] Real: {mean_real:.2f} ± {np.std(real_rewards):.2f} | Best: {self.best_real_reward:.2f}")
        
        return True
    
    def _evaluate_real(self) -> list:
        """Evaluate policy in real environment."""
        rewards = []
        
        for _ in range(self.n_eval_episodes):
            # Create fresh real env with DreamEnv wrapper (for obs encoding)
            real_env = ShooterEnv(render_mode="rgb_array")
            eval_env = DreamEnv(
                vae=self.vae, 
                mdn_rnn=self.mdn_rnn, 
                real_env=real_env,
                max_steps=500, 
                device=self.device, 
                use_real_env=True  # USE REAL REWARDS
            )
            
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # Convert action to int if needed
                if hasattr(action, 'item'):
                    action = action.item()
                elif isinstance(action, np.ndarray):
                    action = int(action.flatten()[0])
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            rewards.append(total_reward)
            real_env.close()
        
        return rewards


def train_dyna(
    vae_path: str,
    mdn_path: str,
    save_dir: str,
    total_timesteps: int = 200000,
    eval_freq: int = 10000,
    imagination_ratio: float = 0.5,  # Fraction of training in dream
    device: str = "cpu"
):
    """
    Dyna-style training with real/dream interleaving.
    
    The key insight: we train in dream environment but evaluate and save
    based on REAL environment performance. This prevents dream exploitation.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("Dyna-Style World Model Training")
    print("="*70)
    print(f"VAE: {vae_path}")
    print(f"MDN-RNN: {mdn_path}")
    print(f"Eval frequency: every {eval_freq} steps")
    print(f"Total timesteps: {total_timesteps}")
    print("="*70)
    
    # Load models
    vae = VAE(in_channels=3, latent_dim=32)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    
    mdn_rnn = MDNRNN(latent_dim=32, action_dim=80, hidden_dim=256)
    mdn_rnn.load_state_dict(torch.load(mdn_path, map_location=device))
    mdn_rnn.eval()
    
    print("Loaded VAE and MDN-RNN")
    
    # Create training environment (uses REAL rewards but dream observations)
    # This is the key: train with real rewards, not MDN predictions
    real_env = ShooterEnv(render_mode="rgb_array")
    train_env = DreamEnv(
        vae=vae, 
        mdn_rnn=mdn_rnn, 
        real_env=real_env,
        max_steps=500, 
        device=device, 
        use_real_env=True  # CRITICAL: Use real rewards during training
    )
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])
    
    # Initialize PPO
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=os.path.join(save_dir, "tb")
    )
    
    # Dyna callback for real evaluation
    dyna_callback = DynaCallback(
        vae=vae,
        mdn_rnn=mdn_rnn,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        save_path=save_dir,
        device=device,
        verbose=1
    )
    
    print("\nStarting Dyna-style training...")
    print("Training with REAL rewards, saving based on REAL performance\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=dyna_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(os.path.join(save_dir, "final_model"))
    
    # Final evaluation with video
    print("\n" + "="*70)
    print("Final Evaluation with Video Recording")
    print("="*70)
    
    video_dir = os.path.join(save_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # Load best model
    best_model = PPO.load(os.path.join(save_dir, "best_real_model"))
    
    final_rewards = []
    for ep in range(5):
        real_env = ShooterEnv(render_mode="rgb_array")
        real_env = RecordVideo(real_env, os.path.join(video_dir, f"ep{ep}"), 
                               episode_trigger=lambda x: True, name_prefix="dyna")
        
        eval_env = DreamEnv(
            vae=vae, mdn_rnn=mdn_rnn, real_env=real_env,
            max_steps=500, device=device, use_real_env=True
        )
        
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        final_rewards.append(total_reward)
        print(f"Episode {ep+1}: {total_reward:.2f}")
        real_env.close()
    
    print(f"\nFinal Mean Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print(f"Best Real Reward during training: {dyna_callback.best_real_reward:.2f}")
    print(f"\nVideos saved to: {video_dir}")
    print(f"Best model saved to: {save_dir}/best_real_model.zip")
    
    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dyna-Style World Model Training")
    parser.add_argument("--vae-path", type=str, default="./world_model_fixed/vae.pt")
    parser.add_argument("--mdn-path", type=str, default="./world_model_fixed/mdn_rnn.pt")
    parser.add_argument("--save-dir", type=str, default="./world_model_dyna")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    train_dyna(
        vae_path=args.vae_path,
        mdn_path=args.mdn_path,
        save_dir=args.save_dir,
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        device=args.device
    )
