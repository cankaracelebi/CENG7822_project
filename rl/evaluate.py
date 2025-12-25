"""
Evaluation script for trained RL agents
"""

import argparse
import numpy as np
from typing import Optional

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from game.g2D import ShooterEnv
from rl.configs.shooter_config import ENV_CONFIG


class MultiDiscreteToBoxWrapper(gym.ActionWrapper):
    """Wrapper to convert MultiDiscrete action space to Box for SAC evaluation."""
    
    def __init__(self, env):
        super().__init__(env)
        self.orig_action_space = env.action_space
        self.n_actions = len(env.action_space.nvec)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actions,),
            dtype=np.float32
        )
        self._nvec = env.action_space.nvec
        
    def action(self, action):
        discrete_action = []
        for i, (a, n) in enumerate(zip(action, self._nvec)):
            scaled = (a + 1) / 2
            idx = int(np.clip(scaled * n, 0, n - 1))
            discrete_action.append(idx)
        return np.array(discrete_action, dtype=np.int64)


def evaluate_model(
    model_path: str,
    algo: str = "ppo",
    n_episodes: int = 10,
    render: bool = True,
    seed: Optional[int] = None,
    vec_normalize_path: Optional[str] = None,
):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to the saved model
        algo: Algorithm used ('ppo', 'dqn', or 'sac')
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        seed: Random seed for evaluation
        vec_normalize_path: Path to VecNormalize stats (for PPO)
    """
    
    # Load model
    if algo == "ppo":
        model = PPO.load(model_path)
    elif algo == "dqn":
        model = DQN.load(model_path)
    elif algo == "sac":
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Create environment
    render_mode = "human" if render else None
    env = ShooterEnv(render_mode=render_mode, **ENV_CONFIG)
    
    # Wrap for SAC if needed
    if algo == "sac":
        env = MultiDiscreteToBoxWrapper(env)
    
    # Wrap in DummyVecEnv for compatibility
    env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize if provided (typically for PPO)
    if vec_normalize_path:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    
    import time
    for episode in range(n_episodes):
        obs = env.reset()
        if seed is not None:
            env.seed(seed + episode)
        
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            # If rendering, process window events and display frames
            if render:
                # Get the actual environment (unwrap from vectorization)
                actual_env = env.venv.envs[0]
                if hasattr(actual_env, '_window') and actual_env._window:
                    actual_env._window.dispatch_events()
                    actual_env._window.on_draw()
                    actual_env._window.flip()
                    time.sleep(0.05)  # Control frame rate for watchability
            
            # Handle episode termination
            if done[0]:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Reward = {total_reward:.2f}, Length = {steps}")
    
    env.close()
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print("\n" + "="*50)
    print(f"Evaluation Results ({n_episodes} episodes):")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print("="*50)
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def compare_with_random(n_episodes: int = 10, seed: Optional[int] = None):
    """
    Evaluate a random policy baseline
    """
    print("Evaluating random policy baseline...")
    
    env = ShooterEnv(render_mode=None, **ENV_CONFIG)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode if seed else None)
        
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    env.close()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nRandom Policy Results ({n_episodes} episodes):")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dqn", "sac"],
        help="Algorithm used to train the model (default: ppo)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Path to VecNormalize stats file (for PPO)",
    )
    parser.add_argument(
        "--compare-random",
        action="store_true",
        help="Also evaluate random policy for comparison",
    )
    
    args = parser.parse_args()
    
    # Evaluate trained model
    results = evaluate_model(
        model_path=args.model_path,
        algo=args.algo,
        n_episodes=args.n_episodes,
        render=not args.no_render,
        seed=args.seed,
        vec_normalize_path=args.vec_normalize,
    )
    
    # Compare with random policy if requested
    if args.compare_random:
        print("\n")
        random_results = compare_with_random(
            n_episodes=args.n_episodes,
            seed=args.seed,
        )
        
        improvement = results["mean_reward"] - random_results["mean_reward"]
        print(f"\nImprovement over random: {improvement:.2f}")


if __name__ == "__main__":
    main()
