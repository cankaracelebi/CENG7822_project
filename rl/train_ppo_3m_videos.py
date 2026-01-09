"""
PPO Training Script for 3M steps with Aggressive Reward Configuration
Includes checkpoint saving and video recording at milestones for presentation.

Usage:
    source ~/miniconda3/bin/activate rlan
    cd /home/guava/Desktop/CENG7822_project
    python -m rl.train_ppo_3m_videos
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.g2D import ShooterEnv
from rl.configs.shooter_config import (
    ENV_CONFIG, PPO_CONFIG,
    REWARD_CONFIG_AGGRESSIVE,
)
from rl.metrics_callback import MetricsCallback, TensorboardMetricsCallback


# ==============================================================================
# CONFIGURATION
# ==============================================================================

TOTAL_TIMESTEPS = 3_000_000  # 3M steps

# Checkpoint milestones for saving and video recording
# Includes early milestone (50K) for quick testing
CHECKPOINT_MILESTONES = [50_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000]

# Number of episodes to record at each milestone
N_VIDEO_EPISODES = 3

# Experiment directories
EXPERIMENT_NAME = "ppo_aggressive_3m"
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", EXPERIMENT_NAME)
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
LOG_DIR = os.path.join(BASE_DIR, "logs")
TB_LOG_DIR = os.path.join(BASE_DIR, "tensorboard")


# ==============================================================================
# CUSTOM CALLBACKS
# ==============================================================================

class VideoRecordingCallback(BaseCallback):
    """
    Custom callback to record videos at specific training milestones.
    """
    
    def __init__(
        self, 
        milestones: List[int],
        video_dir: str,
        n_episodes: int = 3,
        env_config: dict = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.milestones = sorted(milestones)
        self.video_dir = video_dir
        self.n_episodes = n_episodes
        self.env_config = env_config or ENV_CONFIG
        self.recorded_milestones = set()
        
        os.makedirs(video_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Check if we've hit a milestone
        current_timesteps = self.num_timesteps
        
        for milestone in self.milestones:
            if milestone not in self.recorded_milestones and current_timesteps >= milestone:
                self._record_videos(milestone)
                self.recorded_milestones.add(milestone)
        
        return True
    
    def _record_videos(self, milestone: int):
        """Record evaluation videos at the given milestone."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Recording videos at milestone {milestone:,} timesteps...")
            print(f"{'='*60}")
        
        milestone_str = f"{milestone // 1000}k"
        video_folder = os.path.join(self.video_dir, f"milestone_{milestone_str}")
        os.makedirs(video_folder, exist_ok=True)
        
        # Create a fresh environment for recording
        def make_eval_env():
            env = ShooterEnv(render_mode="rgb_array", **self.env_config)
            return env
        
        # Record episodes
        for ep_idx in range(self.n_episodes):
            episode_video_folder = os.path.join(video_folder, f"episode_{ep_idx}")
            os.makedirs(episode_video_folder, exist_ok=True)
            
            # Create environment with video recording
            env = make_eval_env()
            env = RecordVideo(
                env, 
                episode_video_folder,
                episode_trigger=lambda x: True,  # Record every episode
                name_prefix=f"ppo_{milestone_str}_ep{ep_idx}"
            )
            
            # Run one episode
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                # Get action from trained model
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            env.close()
            
            if self.verbose:
                print(f"  Episode {ep_idx + 1}/{self.n_episodes}: "
                      f"Reward = {total_reward:.2f}, Length = {steps}")
        
        if self.verbose:
            print(f"Videos saved to: {video_folder}")
            print(f"{'='*60}\n")


class MilestoneCheckpointCallback(BaseCallback):
    """
    Save model at specific timestep milestones.
    """
    
    def __init__(
        self, 
        milestones: List[int],
        save_dir: str,
        name_prefix: str = "ppo_aggressive",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.milestones = sorted(milestones)
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.saved_milestones = set()
        
        os.makedirs(save_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        current_timesteps = self.num_timesteps
        
        for milestone in self.milestones:
            if milestone not in self.saved_milestones and current_timesteps >= milestone:
                self._save_model(milestone)
                self.saved_milestones.add(milestone)
        
        return True
    
    def _save_model(self, milestone: int):
        """Save model at the given milestone."""
        milestone_str = f"{milestone // 1000}k"
        save_path = os.path.join(self.save_dir, f"{self.name_prefix}_{milestone_str}")
        
        self.model.save(save_path)
        
        if self.verbose:
            print(f"\n*** Saved model checkpoint at {milestone:,} timesteps: {save_path} ***\n")


# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train_ppo_3m_videos(
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: int = 4,
    resume_from: Optional[str] = None,
):
    """
    Train PPO with aggressive reward configuration for 3M timesteps.
    Records videos at milestones for presentation.
    """
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PPO Training with Aggressive Reward Configuration")
    print(f"{'='*70}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Checkpoint milestones: {[f'{m//1000}k' for m in CHECKPOINT_MILESTONES]}")
    print(f"Videos per milestone: {N_VIDEO_EPISODES}")
    print(f"\nAggressive Reward Config:")
    for key, value in REWARD_CONFIG_AGGRESSIVE.items():
        print(f"  {key}: {value}")
    print(f"\nOutput directories:")
    print(f"  Models: {MODEL_DIR}")
    print(f"  Videos: {VIDEO_DIR}")
    print(f"  Logs: {LOG_DIR}")
    print(f"{'='*70}\n")
    
    # Create environment factory
    def make_env(seed: int = 0):
        def _init():
            env = ShooterEnv(**ENV_CONFIG)
            env = Monitor(env)
            return env
        return _init
    
    # Create vectorized environments
    env = DummyVecEnv([make_env(seed=i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(seed=100)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Create or load model
    if resume_from:
        print(f"Resuming training from: {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        model = PPO(
            env=env,
            tensorboard_log=TB_LOG_DIR,
            **PPO_CONFIG
        )
    
    # Setup callbacks
    callbacks = [
        # Milestone checkpoints
        MilestoneCheckpointCallback(
            milestones=CHECKPOINT_MILESTONES,
            save_dir=MODEL_DIR,
            name_prefix="ppo_aggressive",
            verbose=1,
        ),
        # Video recording at milestones
        VideoRecordingCallback(
            milestones=CHECKPOINT_MILESTONES,
            video_dir=VIDEO_DIR,
            n_episodes=N_VIDEO_EPISODES,
            env_config=ENV_CONFIG,
            verbose=1,
        ),
        # Evaluation callback
        EvalCallback(
            eval_env,
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=50_000 // n_envs,  # Evaluate every 50k steps
            deterministic=True,
            render=False,
            verbose=1,
        ),
        # Metrics tracking
        MetricsCallback(
            log_dir=LOG_DIR,
            algo_name="ppo_aggressive_3m",
            verbose=1,
        ),
        # Tensorboard metrics
        TensorboardMetricsCallback(verbose=0),
    ]
    
    # Start training
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model...")
        model.save(os.path.join(MODEL_DIR, "ppo_aggressive_interrupted"))
        env.save(os.path.join(MODEL_DIR, "vec_normalize_interrupted.pkl"))
        print("Model saved. Videos recorded up to current milestone.")
    
    # Save final model
    final_path = os.path.join(MODEL_DIR, "ppo_aggressive_final")
    model.save(final_path)
    env.save(os.path.join(MODEL_DIR, "vec_normalize_final.pkl"))
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Duration: {duration}")
    print(f"Final model saved to: {final_path}")
    print(f"Videos saved to: {VIDEO_DIR}")
    print(f"{'='*70}\n")
    
    return model


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train PPO for 3M steps with video recording at milestones"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    train_ppo_3m_videos(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
