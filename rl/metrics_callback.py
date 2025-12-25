"""
Custom callback for tracking task-specific metrics during training.
Records: prizes collected, enemies killed, damage taken, survival time.
"""

import os
import csv
from typing import Dict, List, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """
    Callback to track and log task-specific metrics per episode.
    Saves to CSV for easy plotting.
    """
    
    def __init__(
        self,
        log_dir: str,
        algo_name: str,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.algo_name = algo_name
        
        # Episode tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_prizes: List[float] = []
        self.episode_kills: List[float] = []
        self.episode_damage: List[float] = []
        
        # Current episode accumulators
        self._current_prizes = 0.0
        self._current_kills = 0.0
        self._current_damage = 0.0
        
        # CSV file
        self.csv_path: Optional[str] = None
        self.csv_file = None
        self.csv_writer = None
        
    def _on_training_start(self) -> None:
        """Initialize CSV file for logging."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, f"{self.algo_name}_metrics.csv")
        
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestep", "episode", "reward", "length", 
            "prizes", "kills", "damage", "survival_rate"
        ])
        self.csv_file.flush()
        
        if self.verbose > 0:
            print(f"[MetricsCallback] Logging to {self.csv_path}")
    
    def _on_step(self) -> bool:
        """Called after each step."""
        # Get info from environments
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for i, (info, done) in enumerate(zip(infos, dones)):
            # Check for episode end (Monitor wrapper adds episode info)
            if done and "episode" in info:
                ep_info = info["episode"]
                ep_reward = ep_info["r"]
                ep_length = ep_info["l"]
                
                # Extract task metrics from terminal observation info
                # Note: These are estimated from reward events
                prizes = info.get("prizes_collected", 0)
                kills = info.get("enemies_killed", 0)
                damage = info.get("damage_taken", 0)
                
                # Survival rate: 1.0 if survived to truncation, based on health
                health = info.get("health", 0)
                survival = 1.0 if health > 0 else 0.0
                
                # Store episode stats
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_prizes.append(prizes)
                self.episode_kills.append(kills)
                self.episode_damage.append(damage)
                
                # Write to CSV
                if self.csv_writer:
                    self.csv_writer.writerow([
                        self.num_timesteps,
                        len(self.episode_rewards),
                        ep_reward,
                        ep_length,
                        prizes,
                        kills,
                        damage,
                        survival
                    ])
                    self.csv_file.flush()
                
                if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                    avg_reward = sum(self.episode_rewards[-10:]) / 10
                    print(f"[{self.algo_name}] Episode {len(self.episode_rewards)}, "
                          f"Timestep {self.num_timesteps}, "
                          f"Avg Reward (10 ep): {avg_reward:.2f}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Cleanup CSV file."""
        if self.csv_file:
            self.csv_file.close()
            if self.verbose > 0:
                print(f"[MetricsCallback] Saved {len(self.episode_rewards)} episodes to {self.csv_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {}
        
        import numpy as np
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "total_episodes": len(self.episode_rewards),
            "mean_prizes": np.mean(self.episode_prizes) if self.episode_prizes else 0,
            "mean_kills": np.mean(self.episode_kills) if self.episode_kills else 0,
        }


class TensorboardMetricsCallback(BaseCallback):
    """
    Extended callback that logs task-specific metrics to TensorBoard.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for info, done in zip(infos, dones):
            if done and "episode" in info:
                ep = info["episode"]
                self._episode_rewards.append(ep["r"])
                self._episode_lengths.append(ep["l"])
                
                # Log to tensorboard
                if self.logger:
                    self.logger.record("custom/episode_reward", ep["r"])
                    self.logger.record("custom/episode_length", ep["l"])
                    
                    # Log health at end (survival indicator)
                    if "health" in info:
                        self.logger.record("custom/final_health", info["health"])
        
        return True
