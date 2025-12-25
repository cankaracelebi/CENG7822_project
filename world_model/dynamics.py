"""
Latent Dynamics Model - predicts next latent state given current state and action.
Also optionally predicts reward and done flag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DynamicsModel(nn.Module):
    """
    MLP that predicts z_{t+1} from (z_t, action).
    Can also predict reward and done signal.
    """
    
    def __init__(
        self, 
        latent_dim: int = 32, 
        action_dim: int = 3,  # MultiDiscrete [5, 2, 8] flattened
        hidden_dim: int = 256,
        predict_reward: bool = True,
        predict_done: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.predict_reward = predict_reward
        self.predict_done = predict_done
        
        input_dim = latent_dim + action_dim
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Predict next latent state (deterministic or distribution)
        self.z_head = nn.Linear(hidden_dim, latent_dim)
        
        # Optional reward prediction
        if predict_reward:
            self.reward_head = nn.Linear(hidden_dim, 1)
        
        # Optional done prediction
        if predict_done:
            self.done_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self, 
        z: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            z: Current latent state (B, latent_dim)
            action: Action taken (B, action_dim) - can be one-hot or continuous
        Returns:
            z_next: Predicted next latent (B, latent_dim)
            reward: Predicted reward (B, 1) or None
            done: Predicted done probability (B, 1) or None
        """
        # Concatenate z and action
        x = torch.cat([z, action], dim=-1)
        h = self.trunk(x)
        
        # Predict next latent
        z_next = self.z_head(h)
        
        # Predict reward
        reward = self.reward_head(h) if self.predict_reward else None
        
        # Predict done (sigmoid for probability)
        done = torch.sigmoid(self.done_head(h)) if self.predict_done else None
        
        return z_next, reward, done
    
    def loss_function(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        reward_pred: Optional[torch.Tensor] = None,
        reward_target: Optional[torch.Tensor] = None,
        done_pred: Optional[torch.Tensor] = None,
        done_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute dynamics loss.
        """
        # Latent prediction loss (MSE)
        z_loss = F.mse_loss(z_pred, z_target)
        total_loss = z_loss
        
        metrics = {'z_loss': z_loss.item()}
        
        # Reward prediction loss
        if reward_pred is not None and reward_target is not None:
            r_loss = F.mse_loss(reward_pred, reward_target)
            total_loss = total_loss + r_loss
            metrics['reward_loss'] = r_loss.item()
        
        # Done prediction loss (BCE)
        if done_pred is not None and done_target is not None:
            d_loss = F.binary_cross_entropy(done_pred, done_target)
            total_loss = total_loss + d_loss
            metrics['done_loss'] = d_loss.item()
        
        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics
    
    def rollout(
        self, 
        z_start: torch.Tensor, 
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Roll out dynamics for multiple steps in imagination.
        
        Args:
            z_start: Initial latent (B, latent_dim)
            actions: Sequence of actions (B, T, action_dim)
        Returns:
            z_seq: Sequence of latent states (B, T+1, latent_dim)
            rewards: Predicted rewards (B, T) or None
            dones: Predicted dones (B, T) or None
        """
        B, T, _ = actions.shape
        
        z_seq = [z_start]
        rewards = [] if self.predict_reward else None
        dones = [] if self.predict_done else None
        
        z = z_start
        for t in range(T):
            z_next, r, d = self(z, actions[:, t])
            z_seq.append(z_next)
            if rewards is not None:
                rewards.append(r)
            if dones is not None:
                dones.append(d)
            z = z_next
        
        z_seq = torch.stack(z_seq, dim=1)
        if rewards is not None:
            rewards = torch.cat(rewards, dim=1)
        if dones is not None:
            dones = torch.cat(dones, dim=1)
        
        return z_seq, rewards, dones


if __name__ == "__main__":
    # Quick test
    model = DynamicsModel(latent_dim=32, action_dim=3)
    z = torch.randn(4, 32)
    a = torch.randn(4, 3)
    z_next, r, d = model(z, a)
    print(f"z: {z.shape}, a: {a.shape} -> z_next: {z_next.shape}, r: {r.shape}, d: {d.shape}")
    
    # Test rollout
    actions = torch.randn(4, 10, 3)  # 10 steps
    z_seq, rewards, dones = model.rollout(z, actions)
    print(f"Rollout: z_seq: {z_seq.shape}, rewards: {rewards.shape}, dones: {dones.shape}")
