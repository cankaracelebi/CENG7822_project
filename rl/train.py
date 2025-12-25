"""
Training script for the shooter environment using Stable-Baselines3
Supports PPO, DQN, and SAC algorithms with comprehensive metrics tracking.
"""

import os
import argparse
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from game.g2D import ShooterEnv
from rl.configs.shooter_config import ENV_CONFIG, PPO_CONFIG, DQN_CONFIG, SAC_CONFIG, TRAINING_CONFIG
from rl.metrics_callback import MetricsCallback, TensorboardMetricsCallback


class MultiDiscreteToBoxWrapper(gym.ActionWrapper):
    """
    Wrapper to convert MultiDiscrete action space to Box for SAC.
    SAC outputs continuous actions which are then discretized.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.orig_action_space = env.action_space
        # MultiDiscrete([5, 2, 8]) -> Box with shape (3,)
        self.n_actions = len(env.action_space.nvec)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actions,),
            dtype=np.float32
        )
        self._nvec = env.action_space.nvec
        
    def action(self, action):
        """Convert continuous action to discrete."""
        # Map continuous [-1, 1] to discrete [0, n-1]
        discrete_action = []
        for i, (a, n) in enumerate(zip(action, self._nvec)):
            # Map [-1, 1] to [0, n-1]
            scaled = (a + 1) / 2  # [0, 1]
            idx = int(np.clip(scaled * n, 0, n - 1))
            discrete_action.append(idx)
        return np.array(discrete_action, dtype=np.int64)


class MultiDiscreteToDiscreteWrapper(gym.ActionWrapper):
    """
    Wrapper to convert MultiDiscrete action space to Discrete for DQN.
    Flattens MultiDiscrete([5, 2, 8]) to Discrete(5*2*8=80).
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.orig_action_space = env.action_space
        self._nvec = env.action_space.nvec
        # Total number of flattened actions: 5 * 2 * 8 = 80
        self.n_total = int(np.prod(self._nvec))
        self.action_space = spaces.Discrete(self.n_total)
        
    def action(self, action):
        """Convert flat discrete action to MultiDiscrete."""
        # Decode flat action index to multi-dimensional indices
        indices = []
        remaining = action
        for n in reversed(self._nvec):
            indices.append(remaining % n)
            remaining //= n
        return np.array(list(reversed(indices)), dtype=np.int64)


def make_env(render_mode: Optional[str] = None, seed: Optional[int] = None, 
             wrap_for_sac: bool = False, wrap_for_dqn: bool = False):
    """Factory function to create the environment"""
    def _init():
        env = ShooterEnv(**ENV_CONFIG)
        if wrap_for_sac:
            env = MultiDiscreteToBoxWrapper(env)
        elif wrap_for_dqn:
            env = MultiDiscreteToDiscreteWrapper(env)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def train_ppo(
    total_timesteps: int = None,
    save_dir: str = "./models/ppo",
    log_dir: str = "./logs/ppo",
    tensorboard_log: str = "./tensorboard_logs/ppo",
    n_envs: int = 4,
    experiment_name: str = "ppo",
):
    """Train PPO agent on the shooter environment"""
    
    if total_timesteps is None:
        total_timesteps = TRAINING_CONFIG["total_timesteps"]
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training PPO for {total_timesteps:,} timesteps...")
    print(f"Using {n_envs} parallel environments")
    print(f"{'='*60}\n")
    
    # Create vectorized environments
    env = DummyVecEnv([make_env(seed=i) for i in range(n_envs)])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(seed=100)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG["save_freq"] // n_envs,
        save_path=save_dir,
        name_prefix="ppo_shooter",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=TRAINING_CONFIG.get("eval_freq", 5000) // n_envs,
        deterministic=True,
        render=False,
    )
    
    metrics_callback = MetricsCallback(
        log_dir=log_dir,
        algo_name="ppo",
        verbose=1,
    )
    
    tb_callback = TensorboardMetricsCallback(verbose=0)
    
    # Create PPO model
    model = PPO(
        env=env,
        tensorboard_log=tensorboard_log,
        **PPO_CONFIG
    )
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, metrics_callback, tb_callback],
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "ppo_shooter_final")
    model.save(final_path)
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print(f"\n{'='*60}")
    print(f"PPO Training complete! Model saved to {final_path}")
    summary = metrics_callback.get_summary()
    if summary:
        print(f"Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"Total Episodes: {summary['total_episodes']}")
    print(f"{'='*60}\n")
    
    return model, metrics_callback


def train_dqn(
    total_timesteps: int = None,
    save_dir: str = "./models/dqn",
    log_dir: str = "./logs/dqn",
    tensorboard_log: str = "./tensorboard_logs/dqn",
    experiment_name: str = "dqn",
):
    """Train DQN agent on the shooter environment"""
    
    if total_timesteps is None:
        total_timesteps = TRAINING_CONFIG["total_timesteps"]
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training DQN for {total_timesteps:,} timesteps...")
    print(f"Using MultiDiscrete->Discrete action wrapper (80 actions)")
    print(f"{'='*60}\n")
    
    # Create environment (DQN uses single env with discrete action wrapper)
    env = DummyVecEnv([make_env(seed=0, wrap_for_dqn=True)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(seed=100, wrap_for_dqn=True)])
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=save_dir,
        name_prefix="dqn_shooter",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=TRAINING_CONFIG.get("eval_freq", 10000),
        deterministic=True,
        render=False,
    )
    
    metrics_callback = MetricsCallback(
        log_dir=log_dir,
        algo_name="dqn",
        verbose=1,
    )
    
    tb_callback = TensorboardMetricsCallback(verbose=0)
    
    # Create DQN model
    model = DQN(
        env=env,
        tensorboard_log=tensorboard_log,
        **DQN_CONFIG
    )
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, metrics_callback, tb_callback],
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "dqn_shooter_final")
    model.save(final_path)
    
    print(f"\n{'='*60}")
    print(f"DQN Training complete! Model saved to {final_path}")
    summary = metrics_callback.get_summary()
    if summary:
        print(f"Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"Total Episodes: {summary['total_episodes']}")
    print(f"{'='*60}\n")
    
    return model, metrics_callback


def train_sac(
    total_timesteps: int = None,
    save_dir: str = "./models/sac",
    log_dir: str = "./logs/sac",
    tensorboard_log: str = "./tensorboard_logs/sac",
    n_envs: int = 1,  # SAC typically uses single env
    experiment_name: str = "sac",
):
    """Train SAC agent on the shooter environment (with continuous action wrapper)"""
    
    if total_timesteps is None:
        total_timesteps = TRAINING_CONFIG["total_timesteps"]
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training SAC for {total_timesteps:,} timesteps...")
    print(f"Using MultiDiscrete->Box action wrapper for SAC compatibility")
    print(f"{'='*60}\n")
    
    # Create environment with SAC wrapper
    env = DummyVecEnv([make_env(seed=0, wrap_for_sac=True)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(seed=100, wrap_for_sac=True)])
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=save_dir,
        name_prefix="sac_shooter",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=TRAINING_CONFIG.get("eval_freq", 5000),
        deterministic=True,
        render=False,
    )
    
    metrics_callback = MetricsCallback(
        log_dir=log_dir,
        algo_name="sac",
        verbose=1,
    )
    
    tb_callback = TensorboardMetricsCallback(verbose=0)
    
    # Create SAC model
    model = SAC(
        env=env,
        tensorboard_log=tensorboard_log,
        **SAC_CONFIG
    )
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, metrics_callback, tb_callback],
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "sac_shooter_final")
    model.save(final_path)
    
    print(f"\n{'='*60}")
    print(f"SAC Training complete! Model saved to {final_path}")
    summary = metrics_callback.get_summary()
    if summary:
        print(f"Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"Total Episodes: {summary['total_episodes']}")
    print(f"{'='*60}\n")
    
    return model, metrics_callback


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on shooter environment")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dqn", "sac", "all"],
        help="RL algorithm to use (default: ppo)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help=f"Total timesteps to train (default: {TRAINING_CONFIG['total_timesteps']})",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments for PPO (default: 4)",
    )
    
    args = parser.parse_args()
    
    if args.algo == "ppo":
        train_ppo(total_timesteps=args.timesteps, n_envs=args.n_envs)
    elif args.algo == "dqn":
        train_dqn(total_timesteps=args.timesteps)
    elif args.algo == "sac":
        train_sac(total_timesteps=args.timesteps)
    elif args.algo == "all":
        print("Training all algorithms sequentially...")
        train_dqn(total_timesteps=args.timesteps)
        train_ppo(total_timesteps=args.timesteps, n_envs=args.n_envs)
        train_sac(total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
