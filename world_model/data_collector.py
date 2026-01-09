"""
Data collector for gathering frames, actions, and rewards from the environment.
Used to train the VAE and dynamics model.
"""

import os
import numpy as np
import torch
from typing import Optional, Dict, List
from PIL import Image
import cv2


class FrameDataCollector:
    """
    Collects (frame, action, reward, next_frame, done) tuples from environment.
    """
    
    def __init__(
        self,
        save_dir: str = "./world_model_data",
        frame_size: tuple = (64, 64),
        max_frames: int = 50000,
    ):
        self.save_dir = save_dir
        self.frame_size = frame_size
        self.max_frames = max_frames
        
        self.frames: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize frame to [0, 1]"""
        if frame.shape[:2] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        return frame.astype(np.float32) / 255.0
    
    def add_transition(
        self,
        frame: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Add a single transition to the buffer"""
        if len(self.frames) >= self.max_frames:
            return
        
        processed = self.preprocess_frame(frame)
        self.frames.append(processed)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def collect_random_episodes(
        self,
        env,
        n_episodes: int = 100,
        max_steps_per_episode: int = 500,
        verbose: bool = True,
    ) -> int:
        """Collect frames using random policy"""
        total_frames = 0
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            frame = env.render()  # Get RGB frame
            
            for step in range(max_steps_per_episode):
                if len(self.frames) >= self.max_frames:
                    break
                
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                self.add_transition(frame, action, reward, done)
                total_frames += 1
                
                if done:
                    break
                
                frame = env.render()
            
            if verbose and (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{n_episodes}, Total frames: {len(self.frames)}")
            
            if len(self.frames) >= self.max_frames:
                break
        
        return total_frames
    
    def collect_with_policy(
        self,
        env,
        policy,
        n_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        verbose: bool = True,
    ) -> int:
        """Collect frames using a trained policy (e.g., PPO model from SB3)"""
        total_frames = 0
        total_reward = 0
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            frame = env.render()
            
            for step in range(max_steps_per_episode):
                if len(self.frames) >= self.max_frames:
                    break
                
                # Use trained policy
                action, _ = policy.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                self.add_transition(frame, action, reward, done)
                total_frames += 1
                total_reward += reward
                
                if done:
                    break
                
                obs = next_obs
                frame = env.render()
            
            if verbose and (ep + 1) % 10 == 0:
                avg_r = total_reward / (ep + 1)
                print(f"Episode {ep+1}/{n_episodes}, Total frames: {len(self.frames)}, Avg Reward: {avg_r:.2f}")
            
            if len(self.frames) >= self.max_frames:
                break
        
        return total_frames
    
    def save(self, filename: str = "collected_data.npz"):
        """Save collected data to disk"""
        path = os.path.join(self.save_dir, filename)
        np.savez_compressed(
            path,
            frames=np.array(self.frames),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
            dones=np.array(self.dones),
        )
        print(f"Saved {len(self.frames)} frames to {path}")
    
    def load(self, filename: str = "collected_data.npz"):
        """Load previously collected data"""
        path = os.path.join(self.save_dir, filename)
        data = np.load(path)
        self.frames = list(data['frames'])
        self.actions = list(data['actions'])
        self.rewards = list(data['rewards'])
        self.dones = list(data['dones'])
        print(f"Loaded {len(self.frames)} frames from {path}")
    
    def get_vae_dataset(self) -> torch.utils.data.Dataset:
        """Get dataset for training VAE (just frames)"""
        frames = torch.tensor(np.array(self.frames), dtype=torch.float32)
        # Convert from (N, H, W, C) to (N, C, H, W)
        frames = frames.permute(0, 3, 1, 2)
        return torch.utils.data.TensorDataset(frames)
    
    def get_dynamics_dataset(self, vae) -> torch.utils.data.Dataset:
        """
        Get dataset for training dynamics model.
        Requires trained VAE to encode frames to latent.
        """
        frames = torch.tensor(np.array(self.frames), dtype=torch.float32)
        frames = frames.permute(0, 3, 1, 2)  # (N, C, H, W)
        
        # Encode all frames to latent
        vae.eval()
        with torch.no_grad():
            latents = vae.encode(frames, deterministic=True)
        
        # Create (z_t, action, reward, z_{t+1}, done) tuples
        # Skip last frame and transitions across episodes
        z_t = []
        actions = []
        rewards = []
        z_next = []
        dones = []
        
        for i in range(len(self.frames) - 1):
            if self.dones[i]:  # Skip transitions that cross episode boundaries
                continue
            z_t.append(latents[i])
            actions.append(torch.tensor(self.actions[i], dtype=torch.float32))
            rewards.append(torch.tensor([self.rewards[i]], dtype=torch.float32))
            z_next.append(latents[i + 1])
            dones.append(torch.tensor([float(self.dones[i])], dtype=torch.float32))
        
        return torch.utils.data.TensorDataset(
            torch.stack(z_t),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(z_next),
            torch.stack(dones),
        )


if __name__ == "__main__":
    # Quick test
    collector = FrameDataCollector(max_frames=100)
    
    # Test with dummy data
    for i in range(50):
        frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        action = np.array([1, 0, 3])
        collector.add_transition(frame, action, 0.1, i % 10 == 9)
    
    print(f"Collected {len(collector.frames)} frames")
    print(f"Frame shape: {collector.frames[0].shape}")
