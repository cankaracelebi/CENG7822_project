"""
World Model Agent - combines VAE + Dynamics for latent-space planning.
Two modes: MPC (model predictive control) or learned latent policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class LatentPolicy(nn.Module):
    """Small MLP policy that acts on latent states"""
    
    def __init__(self, latent_dim: int = 32, action_dim: int = 80, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
    
    def get_action(self, z: torch.Tensor, deterministic: bool = True) -> int:
        logits = self.forward(z)
        if deterministic:
            return logits.argmax(dim=-1).item()
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()


class WorldModelAgent:
    """
    Agent that uses VAE + Dynamics for decision making.
    Supports MPC planning or learned latent policy.
    """
    
    def __init__(
        self,
        vae,
        dynamics,
        action_space,
        planning_horizon: int = 10,
        n_candidates: int = 100,
        use_mpc: bool = True,
        device: str = "cpu",
    ):
        self.vae = vae.to(device)
        self.dynamics = dynamics.to(device)
        self.action_space = action_space
        self.planning_horizon = planning_horizon
        self.n_candidates = n_candidates
        self.use_mpc = use_mpc
        self.device = device
        
        # For MPC: get action dimensions
        if hasattr(action_space, 'nvec'):
            self.n_actions = int(np.prod(action_space.nvec))
            self.action_dims = action_space.nvec
        else:
            self.n_actions = action_space.n
            self.action_dims = None
        
        # For learned policy mode
        self.policy = None
        if not use_mpc:
            self.policy = LatentPolicy(
                latent_dim=vae.latent_dim,
                action_dim=self.n_actions,
            ).to(device)
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess raw frame for VAE"""
        # Resize to 64x64 if needed
        import cv2
        if frame.shape[:2] != (64, 64):
            frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] and convert to tensor
        frame = frame.astype(np.float32) / 255.0
        frame = torch.tensor(frame, dtype=torch.float32, device=self.device)
        frame = frame.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return frame
    
    def encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Encode frame to latent using VAE"""
        self.vae.eval()
        with torch.no_grad():
            frame_tensor = self.preprocess_frame(frame)
            z = self.vae.encode(frame_tensor, deterministic=True)
        return z
    
    def decode_action(self, action_idx: int) -> np.ndarray:
        """Convert flat action index to multi-discrete action"""
        if self.action_dims is None:
            return action_idx
        
        indices = []
        remaining = action_idx
        for n in reversed(self.action_dims):
            indices.append(remaining % n)
            remaining //= n
        return np.array(list(reversed(indices)), dtype=np.int64)
    
    def mpc_planning(self, z: torch.Tensor) -> int:
        """
        Model Predictive Control: sample random action sequences,
        roll out in imagination, pick best.
        """
        self.dynamics.eval()
        
        with torch.no_grad():
            best_reward = float('-inf')
            best_action = 0
            
            for _ in range(self.n_candidates):
                # Sample random action sequence
                actions = torch.randint(
                    0, self.n_actions, 
                    (self.planning_horizon,),
                    device=self.device
                )
                
                # Convert to one-hot for dynamics
                action_onehot = F.one_hot(actions, self.n_actions).float()
                action_onehot = action_onehot.unsqueeze(0)  # (1, T, action_dim)
                
                # Rollout
                z_seq, rewards, dones = self.dynamics.rollout(z, action_onehot)
                
                # Sum predicted rewards (discounted)
                if rewards is not None:
                    gamma = 0.99
                    discounts = torch.tensor([gamma ** t for t in range(self.planning_horizon)], device=self.device)
                    total_reward = (rewards.squeeze() * discounts).sum().item()
                else:
                    # If no reward prediction, use negative latent distance as proxy
                    total_reward = -torch.norm(z_seq[:, -1] - z_seq[:, 0]).item()
                
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_action = actions[0].item()  # Take first action
        
        return best_action
    
    def get_action(self, frame: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action given raw frame observation"""
        z = self.encode_frame(frame)
        
        if self.use_mpc:
            action_idx = self.mpc_planning(z)
        else:
            action_idx = self.policy.get_action(z, deterministic)
        
        return self.decode_action(action_idx)
    
    def save(self, path: str):
        """Save agent models"""
        torch.save({
            'vae': self.vae.state_dict(),
            'dynamics': self.dynamics.state_dict(),
            'policy': self.policy.state_dict() if self.policy else None,
        }, path)
    
    def load(self, path: str):
        """Load agent models"""
        checkpoint = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae'])
        self.dynamics.load_state_dict(checkpoint['dynamics'])
        if self.policy and checkpoint['policy']:
            self.policy.load_state_dict(checkpoint['policy'])


if __name__ == "__main__":
    from vae import VAE
    from dynamics import DynamicsModel
    
    # Quick test
    vae = VAE(latent_dim=32)
    dynamics = DynamicsModel(latent_dim=32, action_dim=80)
    
    class DummyActionSpace:
        nvec = np.array([5, 2, 8])
    
    agent = WorldModelAgent(
        vae=vae,
        dynamics=dynamics,
        action_space=DummyActionSpace(),
        planning_horizon=5,
        n_candidates=10,
    )
    
    # Test with random frame
    frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    action = agent.get_action(frame)
    print(f"Frame shape: {frame.shape} -> Action: {action}")
