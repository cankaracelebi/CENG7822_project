#!/usr/bin/env python
"""
Train an agent using the World Model (VAE + Dynamics).
Two training modes:
1. MPC (Model Predictive Control) - sample actions, roll out in imagination
2. Latent Policy - train a policy network on latent states using imagined rollouts
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.vae import VAE
from world_model.dynamics import DynamicsModel
from world_model.world_agent import WorldModelAgent, LatentPolicy
from game.g2D import ShooterEnv


def evaluate_agent(
    agent: WorldModelAgent,
    env: ShooterEnv,
    n_episodes: int = 10,
) -> dict:
    """Evaluate agent performance"""
    rewards = []
    lengths = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        frame = env.render()
        total_reward = 0
        steps = 0
        
        done = False
        while not done:
            action = agent.get_action(frame)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            if not done:
                frame = env.render()
        
        rewards.append(total_reward)
        lengths.append(steps)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'n_episodes': n_episodes,
    }


def train_latent_policy(
    vae: VAE,
    dynamics: DynamicsModel,
    env: ShooterEnv,
    policy: LatentPolicy,
    epochs: int = 100,
    rollout_length: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    gamma: float = 0.99,
    device: str = "cpu",
) -> List[float]:
    """
    Train latent policy using imagined rollouts.
    Similar to Dreamer/World Models approach.
    """
    policy = policy.to(device)
    vae = vae.to(device)
    dynamics = dynamics.to(device)
    
    vae.eval()
    dynamics.eval()
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    history = []
    
    for epoch in range(epochs):
        # Collect initial frames from environment
        obs, _ = env.reset()
        frame = env.render()
        
        # Preprocess and encode
        frame_proc = preprocess_frame(frame).to(device)
        
        with torch.no_grad():
            z = vae.encode(frame_proc, deterministic=True)
        
        # Imagined rollout with policy
        z_batch = z.repeat(batch_size, 1)  # (B, latent_dim)
        
        total_imagined_reward = 0
        log_probs = []
        rewards = []
        
        for t in range(rollout_length):
            # Get action from policy
            logits = policy(z_batch)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            log_probs.append(log_prob)
            
            # Convert to one-hot for dynamics
            action_onehot = torch.zeros(batch_size, 80, device=device)
            action_onehot.scatter_(1, action_idx.unsqueeze(1), 1)
            
            # Predict next state using dynamics (need to adapt action dim)
            # Simplify: use first 3 dims as action representation
            action_simple = action_onehot[:, :3]  # Rough approximation
            
            with torch.no_grad():
                z_next, r_pred, d_pred = dynamics(z_batch, action_simple)
            
            if r_pred is not None:
                rewards.append(r_pred.squeeze())
            
            z_batch = z_next
        
        # Calculate returns with discount
        if rewards:
            returns = []
            G = torch.zeros(batch_size, device=device)
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.stack(returns)  # (T, B)
            
            # Policy gradient loss
            log_probs_stack = torch.stack(log_probs)  # (T, B)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            loss = -(log_probs_stack * returns.detach()).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            mean_return = returns.mean().item()
            history.append(mean_return)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Imagined Return = {mean_return:.3f}")
    
    return history


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Preprocess frame for VAE"""
    import cv2
    if frame.shape[:2] != (64, 64):
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    if frame.max() > 1:
        frame = frame.astype(np.float32) / 255.0
    tensor = torch.tensor(frame, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return tensor


def main():
    parser = argparse.ArgumentParser(description="Train World Model Agent")
    parser.add_argument("--mode", type=str, choices=["mpc", "policy"], default="mpc",
                       help="Training mode: 'mpc' for Model Predictive Control, 'policy' for learned latent policy")
    parser.add_argument("--checkpoint-dir", type=str, default="./world_model_checkpoints",
                       help="Directory with trained VAE and dynamics")
    parser.add_argument("--policy-epochs", type=int, default=100,
                       help="Training epochs for latent policy")
    parser.add_argument("--eval-episodes", type=int, default=20,
                       help="Number of evaluation episodes")
    parser.add_argument("--planning-horizon", type=int, default=10,
                       help="MPC planning horizon")
    parser.add_argument("--n-candidates", type=int, default=100,
                       help="Number of action sequence candidates for MPC")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="./world_model_agent")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 50)
    print("World Model Agent Training")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    
    # Load VAE
    print("\nLoading VAE...")
    vae = VAE(in_channels=3, latent_dim=32)
    vae_path = os.path.join(args.checkpoint_dir, "vae.pt")
    vae.load_state_dict(torch.load(vae_path, map_location=args.device))
    vae.eval()
    print(f"Loaded VAE from {vae_path}")
    
    # Load Dynamics
    print("Loading Dynamics Model...")
    dynamics = DynamicsModel(latent_dim=32, action_dim=3)
    dynamics_path = os.path.join(args.checkpoint_dir, "dynamics.pt")
    dynamics.load_state_dict(torch.load(dynamics_path, map_location=args.device))
    dynamics.eval()
    print(f"Loaded Dynamics from {dynamics_path}")
    
    # Create environment
    print("Creating environment...")
    env = ShooterEnv(render_mode="rgb_array")
    
    # Create agent
    if args.mode == "mpc":
        print(f"\nUsing MPC with horizon={args.planning_horizon}, candidates={args.n_candidates}")
        agent = WorldModelAgent(
            vae=vae,
            dynamics=dynamics,
            action_space=env.action_space,
            planning_horizon=args.planning_horizon,
            n_candidates=args.n_candidates,
            use_mpc=True,
            device=args.device,
        )
    else:
        print("\nTraining latent policy...")
        policy = LatentPolicy(latent_dim=32, action_dim=80)
        
        # Train policy in imagination
        history = train_latent_policy(
            vae=vae,
            dynamics=dynamics,
            env=env,
            policy=policy,
            epochs=args.policy_epochs,
            device=args.device,
        )
        
        # Create agent with learned policy
        agent = WorldModelAgent(
            vae=vae,
            dynamics=dynamics,
            action_space=env.action_space,
            use_mpc=False,
            device=args.device,
        )
        agent.policy = policy.to(args.device)
        
        # Save training history
        with open(os.path.join(args.save_dir, "policy_training.json"), 'w') as f:
            json.dump({'returns': history}, f, indent=2)
    
    # Evaluate
    print(f"\nEvaluating agent over {args.eval_episodes} episodes...")
    results = evaluate_agent(agent, env, n_episodes=args.eval_episodes)
    
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f}")
    
    # Save results
    results['mode'] = args.mode
    results['timestamp'] = datetime.now().isoformat()
    results['planning_horizon'] = args.planning_horizon
    results['n_candidates'] = args.n_candidates
    
    results_path = os.path.join(args.save_dir, f"eval_results_{args.mode}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save agent
    agent.save(os.path.join(args.save_dir, f"agent_{args.mode}.pt"))
    print(f"Agent saved to {args.save_dir}/agent_{args.mode}.pt")
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
