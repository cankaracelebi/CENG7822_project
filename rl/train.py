"""
Training script for the shooter environment using Stable-Baselines3
"""

import os
import argparse
from typing import Optional

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from game.g2D import ShooterEnv
from rl.configs.shooter_config import ENV_CONFIG, PPO_CONFIG, DQN_CONFIG, TRAINING_CONFIG


def make_env(render_mode: Optional[str] = None, seed: Optional[int] = None):
    """Factory function to create the environment"""
    def _init():
        env = ShooterEnv(**ENV_CONFIG) # config kwargs is sufficient  do not add render here 
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
):
    """Train PPO agent on the shooter environment"""
    
    if total_timesteps is None:
        total_timesteps = TRAINING_CONFIG["total_timesteps"]
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Training PPO for {total_timesteps} timesteps...")
    print(f"Using {n_envs} parallel environments")
    
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
        eval_freq=5000 // n_envs,
        deterministic=True,
        render=False,
    )
    
    # Create PPO model
    model = PPO(
        env=env,
        tensorboard_log=tensorboard_log,
        **PPO_CONFIG
    )
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "ppo_shooter_final")
    model.save(final_path)
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print(f"\nTraining complete! Model saved to {final_path}")
    
    return model


def train_dqn(
    total_timesteps: int = None,
    save_dir: str = "./models/dqn",
    log_dir: str = "./logs/dqn",
    tensorboard_log: str = "./tensorboard_logs/dqn",
):
    """Train DQN agent on the shooter environment"""
    
    if total_timesteps is None:
        total_timesteps = TRAINING_CONFIG["total_timesteps"]
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Training DQN for {total_timesteps} timesteps...")
    
    # Create environment
    env = DummyVecEnv([make_env(seed=0)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(seed=100)])
    
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
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    
    # Create DQN model
    model = DQN(
        env=env,
        tensorboard_log=tensorboard_log,
        **DQN_CONFIG
    )
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "dqn_shooter_final")
    model.save(final_path)
    
    print(f"\nTraining complete! Model saved to {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on shooter environment")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
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


if __name__ == "__main__":
    main()
