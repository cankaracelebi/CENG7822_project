#!/usr/bin/env python
"""
Extended Experiment Runner for 2D Shooter RL
Runs all combinations of: algorithms × reward configs × timesteps
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.g2D import ShooterEnv
from rl.configs.shooter_config import (
    ENV_CONFIG, PPO_CONFIG, DQN_CONFIG, SAC_CONFIG,
    REWARD_CONFIGS, TIMESTEP_CONFIGS, EXPERIMENT_CONFIG,
    get_experiment_matrix
)
from rl.metrics_callback import MetricsCallback, TensorboardMetricsCallback

import gymnasium as gym
from gymnasium import spaces


class MultiDiscreteToBoxWrapper(gym.ActionWrapper):
    """Wrapper for SAC - converts MultiDiscrete to Box"""
    def __init__(self, env):
        super().__init__(env)
        self._nvec = env.action_space.nvec
        self.n_actions = len(self._nvec)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
        )
    
    def action(self, action):
        discrete_action = []
        for a, n in zip(action, self._nvec):
            scaled = (a + 1) / 2
            idx = int(np.clip(scaled * n, 0, n - 1))
            discrete_action.append(idx)
        return np.array(discrete_action, dtype=np.int64)


class MultiDiscreteToDiscreteWrapper(gym.ActionWrapper):
    """Wrapper for DQN - flattens MultiDiscrete to Discrete"""
    def __init__(self, env):
        super().__init__(env)
        self._nvec = env.action_space.nvec
        self.n_total = int(np.prod(self._nvec))
        self.action_space = spaces.Discrete(self.n_total)
    
    def action(self, action):
        indices = []
        remaining = action
        for n in reversed(self._nvec):
            indices.append(remaining % n)
            remaining //= n
        return np.array(list(reversed(indices)), dtype=np.int64)


def make_env(seed: int = 0, reward_config: Dict = None, 
             wrap_for_sac: bool = False, wrap_for_dqn: bool = False):
    """Factory to create env with reward config"""
    def _init():
        env_kwargs = ENV_CONFIG.copy()
        if reward_config:
            env_kwargs["reward_config"] = reward_config
        env = ShooterEnv(**env_kwargs)
        
        if wrap_for_sac:
            env = MultiDiscreteToBoxWrapper(env)
        elif wrap_for_dqn:
            env = MultiDiscreteToDiscreteWrapper(env)
        
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def run_single_experiment(
    experiment: Dict[str, Any],
    base_dir: str = "./experiments",
    n_envs: int = 4,
) -> Dict[str, Any]:
    """Run a single experiment configuration"""
    
    exp_name = experiment["name"]
    algo = experiment["algorithm"]
    timesteps = experiment["timesteps"]
    reward_params = experiment["reward_params"]
    
    # Create experiment directories
    exp_dir = os.path.join(base_dir, exp_name)
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")
    # Use main tensorboard_logs dir so TensorBoard can see all experiments
    tb_dir = os.path.join("./tensorboard_logs", exp_name)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  Algorithm: {algo.upper()}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Reward Config: {experiment['reward_config']}")
    print(f"{'='*70}\n")
    
    # Save experiment config
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "name": exp_name,
            "algorithm": algo,
            "timesteps": timesteps,
            "reward_config": experiment["reward_config"],
            "reward_params": reward_params,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    # Create environments
    if algo == "dqn":
        env = DummyVecEnv([make_env(seed=0, reward_config=reward_params, wrap_for_dqn=True)])
        eval_env = DummyVecEnv([make_env(seed=100, reward_config=reward_params, wrap_for_dqn=True)])
    elif algo == "sac":
        env = DummyVecEnv([make_env(seed=0, reward_config=reward_params, wrap_for_sac=True)])
        eval_env = DummyVecEnv([make_env(seed=100, reward_config=reward_params, wrap_for_sac=True)])
    else:  # ppo
        env = DummyVecEnv([make_env(seed=i, reward_config=reward_params) for i in range(n_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        eval_env = DummyVecEnv([make_env(seed=100, reward_config=reward_params)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Calculate eval/save frequencies
    eval_freq = max(5000, timesteps // 100)
    save_freq = max(10000, timesteps // 50)
    if algo == "ppo":
        eval_freq = eval_freq // n_envs
        save_freq = save_freq // n_envs
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix=f"{algo}_shooter",
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    
    metrics_cb = MetricsCallback(
        log_dir=log_dir,
        algo_name=algo,
        verbose=1,
    )
    
    tb_cb = TensorboardMetricsCallback(verbose=0)
    
    # Create model
    if algo == "dqn":
        model = DQN(env=env, tensorboard_log=tb_dir, **DQN_CONFIG)
    elif algo == "ppo":
        model = PPO(env=env, tensorboard_log=tb_dir, **PPO_CONFIG)
    elif algo == "sac":
        model = SAC(env=env, tensorboard_log=tb_dir, **SAC_CONFIG)
    
    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb, tb_cb],
    )
    
    # Save final model
    final_path = os.path.join(model_dir, f"{algo}_final")
    model.save(final_path)
    
    if algo == "ppo":
        env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    
    # Get summary
    summary = metrics_cb.get_summary()
    
    print(f"\n{'='*70}")
    print(f"COMPLETED: {exp_name}")
    if summary:
        print(f"  Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"  Total Episodes: {summary['total_episodes']}")
    print(f"{'='*70}\n")
    
    return {
        "name": exp_name,
        "completed": True,
        "summary": summary,
    }


def run_all_experiments(
    base_dir: str = "./experiments",
    filter_algo: Optional[str] = None,
    filter_reward: Optional[str] = None,
    filter_timestep: Optional[str] = None,
    n_envs: int = 4,
):
    """Run all (or filtered) experiments"""
    
    experiments = get_experiment_matrix()
    
    # Apply filters
    if filter_algo:
        experiments = [e for e in experiments if e["algorithm"] == filter_algo]
    if filter_reward:
        experiments = [e for e in experiments if e["reward_config"] == filter_reward]
    if filter_timestep:
        experiments = [e for e in experiments if e["timestep_config"] == filter_timestep]
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT BATCH")
    print(f"  Total experiments to run: {len(experiments)}")
    print(f"  Output directory: {base_dir}")
    print(f"{'='*70}")
    
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp['name']:40} | {exp['timesteps']:>12,} steps")
    
    total_steps = sum(e['timesteps'] for e in experiments)
    print(f"\n  Total timesteps: {total_steps:,}")
    print(f"  Estimated time: ~{total_steps / 500 / 3600:.1f} hours (at 500 steps/sec)")
    print(f"{'='*70}\n")
    
    results = []
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Starting {exp['name']}...")
        result = run_single_experiment(exp, base_dir, n_envs)
        results.append(result)
    
    # Save overall results
    results_path = os.path.join(base_dir, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "total_experiments": len(results),
            "completed": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {results_path}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run RL experiments")
    parser.add_argument(
        "--output-dir", type=str, default="./experiments",
        help="Base directory for experiment outputs"
    )
    parser.add_argument(
        "--algo", type=str, choices=["dqn", "ppo", "sac"],
        help="Filter to specific algorithm"
    )
    parser.add_argument(
        "--reward", type=str, choices=["baseline", "survival", "aggressive"],
        help="Filter to specific reward config"
    )
    parser.add_argument(
        "--timestep", type=str, choices=["short", "medium", "long"],
        help="Filter to specific timestep config"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments for PPO"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List experiments without running"
    )
    
    args = parser.parse_args()
    
    if args.list:
        experiments = get_experiment_matrix()
        print(f"\nAll experiments ({len(experiments)}):")
        for exp in experiments:
            print(f"  {exp['name']:40} | {exp['timesteps']:>12,} steps")
        total = sum(e['timesteps'] for e in experiments)
        print(f"\nTotal: {total:,} steps (~{total/500/3600:.1f} hours)")
        return
    
    run_all_experiments(
        base_dir=args.output_dir,
        filter_algo=args.algo,
        filter_reward=args.reward,
        filter_timestep=args.timestep,
        n_envs=args.n_envs,
    )


if __name__ == "__main__":
    main()
