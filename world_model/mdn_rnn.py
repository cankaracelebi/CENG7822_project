"""
MDN-RNN (Mixture Density Network - Recurrent Neural Network) for World Models.
Based on Ha & Schmidhuber 2018 "World Models" paper.

The MDN-RNN captures temporal dynamics by:
1. Using LSTM to maintain hidden state across time steps
2. Outputting a Mixture of Gaussians (MDN) for stochastic next-state prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MDNRNN(nn.Module):
    """
    MDN-RNN: LSTM with Mixture Density Network output.
    
    Predicts P(z_{t+1} | z_t, a_t, h_t) as a mixture of Gaussians,
    plus optional reward and done predictions.
    
    Architecture matches the original World Models paper:
    - LSTM for temporal modeling
    - MDN output layer for stochastic prediction
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 80,  # Flattened action space
        hidden_dim: int = 256,
        n_gaussians: int = 5,  # Number of mixture components
        n_layers: int = 1,
        predict_reward: bool = True,
        predict_done: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        self.predict_reward = predict_reward
        self.predict_done = predict_done
        
        # Input: concatenation of latent z and action
        input_dim = latent_dim + action_dim
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        
        # MDN output heads for next latent prediction
        # For each Gaussian: mu (latent_dim), log_sigma (latent_dim), log_pi (1)
        # Total per Gaussian: 2 * latent_dim + 1
        # Total output: n_gaussians * (2 * latent_dim + 1)
        self.mdn_output_dim = n_gaussians * (2 * latent_dim + 1)
        self.mdn_head = nn.Linear(hidden_dim, self.mdn_output_dim)
        
        # Optional reward prediction
        if predict_reward:
            self.reward_head = nn.Linear(hidden_dim, 1)
        
        # Optional done prediction
        if predict_done:
            self.done_head = nn.Linear(hidden_dim, 1)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
    
    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
               Optional[torch.Tensor], Optional[torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through MDN-RNN.
        
        Args:
            z: Current latent state (B, latent_dim) or (B, T, latent_dim)
            action: Action taken (B, action_dim) or (B, T, action_dim)
            hidden: Optional LSTM hidden state tuple (h, c)
            
        Returns:
            pi: Mixture weights (B, [T,] n_gaussians)
            mu: Mixture means (B, [T,] n_gaussians, latent_dim)
            sigma: Mixture stds (B, [T,] n_gaussians, latent_dim)
            reward: Predicted reward (B, [T,] 1) or None
            done: Predicted done probability (B, [T,] 1) or None
            hidden: Updated LSTM hidden state (h, c)
        """
        # Handle both single step and sequence inputs
        single_step = z.dim() == 2
        if single_step:
            z = z.unsqueeze(1)  # (B, 1, latent_dim)
            action = action.unsqueeze(1)  # (B, 1, action_dim)
        
        B, T, _ = z.shape
        device = z.device
        
        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.init_hidden(B, device)
        
        # Concatenate z and action
        x = torch.cat([z, action], dim=-1)  # (B, T, latent_dim + action_dim)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)  # (B, T, hidden_dim)
        
        # MDN output
        mdn_out = self.mdn_head(lstm_out)  # (B, T, mdn_output_dim)
        
        # Parse MDN outputs
        pi, mu, sigma = self._parse_mdn_output(mdn_out)
        
        # Optional predictions
        reward = self.reward_head(lstm_out) if self.predict_reward else None
        done = torch.sigmoid(self.done_head(lstm_out)) if self.predict_done else None
        
        # Remove time dimension if single step
        if single_step:
            pi = pi.squeeze(1)
            mu = mu.squeeze(1)
            sigma = sigma.squeeze(1)
            if reward is not None:
                reward = reward.squeeze(1)
            if done is not None:
                done = done.squeeze(1)
        
        return pi, mu, sigma, reward, done, hidden
    
    def _parse_mdn_output(self, mdn_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse MDN output into mixture components.
        
        Args:
            mdn_out: Raw MDN output (B, T, mdn_output_dim)
            
        Returns:
            pi: Mixture weights (B, T, n_gaussians) - softmaxed
            mu: Mixture means (B, T, n_gaussians, latent_dim)
            sigma: Mixture stds (B, T, n_gaussians, latent_dim) - exponentiated
        """
        B, T, _ = mdn_out.shape
        
        # Split output
        # Layout: [log_pi_1, ..., log_pi_K, mu_1, ..., mu_K, log_sigma_1, ..., log_sigma_K]
        log_pi = mdn_out[:, :, :self.n_gaussians]  # (B, T, K)
        mu_flat = mdn_out[:, :, self.n_gaussians:self.n_gaussians * (1 + self.latent_dim)]
        log_sigma_flat = mdn_out[:, :, self.n_gaussians * (1 + self.latent_dim):]
        
        # Reshape mu and sigma
        mu = mu_flat.view(B, T, self.n_gaussians, self.latent_dim)
        log_sigma = log_sigma_flat.view(B, T, self.n_gaussians, self.latent_dim)
        
        # Apply softmax to get mixture weights
        pi = F.softmax(log_pi, dim=-1)
        
        # Exponentiate and clamp sigma for numerical stability
        sigma = torch.exp(log_sigma).clamp(min=1e-6, max=10.0)
        
        return pi, mu, sigma
    
    def sample(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample from the mixture of Gaussians.
        
        Args:
            pi: Mixture weights (B, n_gaussians)
            mu: Mixture means (B, n_gaussians, latent_dim)
            sigma: Mixture stds (B, n_gaussians, latent_dim)
            temperature: Sampling temperature (1.0 = normal, <1 = more deterministic)
            
        Returns:
            z_next: Sampled next latent (B, latent_dim)
        """
        B = pi.shape[0]
        device = pi.device
        
        # Apply temperature to mixture weights
        if temperature != 1.0:
            pi = F.softmax(torch.log(pi + 1e-8) / temperature, dim=-1)
        
        # Sample mixture component
        component_idx = torch.multinomial(pi, 1).squeeze(-1)  # (B,)
        
        # Gather selected mu and sigma
        batch_idx = torch.arange(B, device=device)
        selected_mu = mu[batch_idx, component_idx]  # (B, latent_dim)
        selected_sigma = sigma[batch_idx, component_idx]  # (B, latent_dim)
        
        # Apply temperature to sigma
        if temperature != 1.0:
            selected_sigma = selected_sigma * temperature
        
        # Sample from selected Gaussian
        eps = torch.randn_like(selected_mu)
        z_next = selected_mu + selected_sigma * eps
        
        return z_next
    
    def get_deterministic(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get deterministic prediction (mixture mean or most likely component mean).
        
        Args:
            pi: Mixture weights (B, n_gaussians)
            mu: Mixture means (B, n_gaussians, latent_dim)
            
        Returns:
            z_next: Predicted next latent (B, latent_dim)
        """
        # Weighted mean of all components
        pi_expanded = pi.unsqueeze(-1)  # (B, n_gaussians, 1)
        z_next = (pi_expanded * mu).sum(dim=1)  # (B, latent_dim)
        return z_next
    
    def loss_function(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        z_target: torch.Tensor,
        reward_pred: Optional[torch.Tensor] = None,
        reward_target: Optional[torch.Tensor] = None,
        done_pred: Optional[torch.Tensor] = None,
        done_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute MDN-RNN loss (negative log-likelihood of mixture).
        
        Args:
            pi: Mixture weights (B, [T,] n_gaussians)
            mu: Mixture means (B, [T,] n_gaussians, latent_dim)
            sigma: Mixture stds (B, [T,] n_gaussians, latent_dim)
            z_target: Target next latent (B, [T,] latent_dim)
            
        Returns:
            loss: Total loss
            metrics: Dict with individual loss components
        """
        # Handle both single step and sequence
        if z_target.dim() == 2:
            z_target = z_target.unsqueeze(1)
            pi = pi.unsqueeze(1)
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        
        # Expand z_target for broadcasting with mixture components
        z_target_expanded = z_target.unsqueeze(2)  # (B, T, 1, latent_dim)
        
        # Compute log probability for each Gaussian component
        # log N(z | mu, sigma) = -0.5 * (log(2*pi) + 2*log(sigma) + ((z-mu)/sigma)^2)
        log_prob = -0.5 * (
            math.log(2 * math.pi) + 
            2 * torch.log(sigma + 1e-8) + 
            ((z_target_expanded - mu) / (sigma + 1e-8)) ** 2
        )
        log_prob = log_prob.sum(dim=-1)  # Sum over latent_dim: (B, T, n_gaussians)
        
        # Add log mixture weights
        log_pi = torch.log(pi + 1e-8)  # (B, T, n_gaussians)
        log_prob_weighted = log_prob + log_pi
        
        # Log-sum-exp for mixture log probability
        mdn_loss = -torch.logsumexp(log_prob_weighted, dim=-1).mean()
        
        total_loss = mdn_loss
        metrics = {'mdn_loss': mdn_loss.item()}
        
        # Reward prediction loss (weighted by reward_weight if set)
        if reward_pred is not None and reward_target is not None:
            r_loss = F.mse_loss(reward_pred, reward_target)
            reward_weight = getattr(self, 'reward_weight', 1.0)
            total_loss = total_loss + reward_weight * r_loss
            metrics['reward_loss'] = r_loss.item()
        
        # Done prediction loss
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
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor]]:
        """
        Roll out dynamics for multiple steps in imagination.
        
        Args:
            z_start: Initial latent (B, latent_dim)
            actions: Sequence of actions (B, T, action_dim)
            hidden: Initial LSTM hidden state
            temperature: Sampling temperature
            deterministic: If True, use deterministic prediction
            
        Returns:
            z_seq: Sequence of latent states (B, T+1, latent_dim)
            rewards: Predicted rewards (B, T, 1) or None
            dones: Predicted dones (B, T, 1) or None
            hidden: Final LSTM hidden state
        """
        B, T, _ = actions.shape
        device = z_start.device
        
        if hidden is None:
            hidden = self.init_hidden(B, device)
        
        z_seq = [z_start]
        rewards = [] if self.predict_reward else None
        dones = [] if self.predict_done else None
        
        z = z_start
        for t in range(T):
            pi, mu, sigma, r, d, hidden = self(z, actions[:, t], hidden)
            
            # Get next latent
            if deterministic:
                z_next = self.get_deterministic(pi, mu)
            else:
                z_next = self.sample(pi, mu, sigma, temperature)
            
            z_seq.append(z_next)
            if rewards is not None:
                rewards.append(r)
            if dones is not None:
                dones.append(d)
            z = z_next
        
        z_seq = torch.stack(z_seq, dim=1)  # (B, T+1, latent_dim)
        if rewards is not None:
            rewards = torch.stack(rewards, dim=1)  # (B, T, 1)
        if dones is not None:
            dones = torch.stack(dones, dim=1)  # (B, T, 1)
        
        return z_seq, rewards, dones, hidden


# Backward compatible alias
DynamicsModel = MDNRNN


if __name__ == "__main__":
    # Quick test
    print("Testing MDN-RNN...")
    
    model = MDNRNN(latent_dim=32, action_dim=80, n_gaussians=5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test single step
    z = torch.randn(4, 32)
    a = torch.randn(4, 80)
    pi, mu, sigma, r, d, hidden = model(z, a)
    print(f"Single step: z={z.shape}, a={a.shape}")
    print(f"  -> pi={pi.shape}, mu={mu.shape}, sigma={sigma.shape}")
    print(f"  -> reward={r.shape}, done={d.shape}")
    
    # Test sampling
    z_next = model.sample(pi, mu, sigma)
    print(f"  -> sampled z_next={z_next.shape}")
    
    # Test deterministic
    z_next_det = model.get_deterministic(pi, mu)
    print(f"  -> deterministic z_next={z_next_det.shape}")
    
    # Test loss
    loss, metrics = model.loss_function(pi, mu, sigma, z_next)
    print(f"  -> loss={loss.item():.4f}, metrics={metrics}")
    
    # Test sequence
    z_seq = torch.randn(4, 10, 32)
    a_seq = torch.randn(4, 10, 80)
    pi, mu, sigma, r, d, hidden = model(z_seq, a_seq)
    print(f"Sequence: z_seq={z_seq.shape}, a_seq={a_seq.shape}")
    print(f"  -> pi={pi.shape}, mu={mu.shape}, sigma={sigma.shape}")
    
    # Test rollout
    z_start = torch.randn(4, 32)
    actions = torch.randn(4, 10, 80)
    z_rolled, rewards, dones, hidden = model.rollout(z_start, actions)
    print(f"Rollout: z_start={z_start.shape}, actions={actions.shape}")
    print(f"  -> z_rolled={z_rolled.shape}, rewards={rewards.shape}, dones={dones.shape}")
    
    print("\nMDN-RNN test passed!")
