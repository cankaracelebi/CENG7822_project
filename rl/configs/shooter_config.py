"""
Training configuration for shooter environment
Extended experiment configurations with multiple reward shaping settings
"""

# Environment parameters
ENV_CONFIG = {
    # "render_mode": None,  # Don't render during training - it's too slow with parallel envs
    "width": 800,
    "height": 600,
    "dt": 1/30,
    "max_steps": 1800,  # 60 seconds at 30 FPS
    "k_enemies": 5,
    "m_prizes": 3,
    "max_enemies": 10,
    "enemy_spawn_interval": 1.5,
    "prize_spawn_interval": 1.0,
    "agent_speed": 140.0,
    "bullet_speed": 260.0,
    "shoot_cooldown_steps": 6,
    "damage_on_contact": 0.2,
}

# ==============================================================================
# REWARD SHAPING CONFIGURATIONS
# These define different reward balancing strategies for experiments
# ==============================================================================

# Reward Config 1: BASELINE (Original settings - balanced)
REWARD_CONFIG_BASELINE = {
    "name": "baseline",
    "description": "Original balanced reward shaping",
    "R_PRIZE": 1.0,      # Reward for collecting prize
    "R_HIT": 0.3,        # Reward for hitting enemy
    "R_KILL": 1.0,       # Reward for killing enemy
    "R_DAMAGE": 1.0,     # Penalty multiplier for taking damage
    "R_SHOT": 0.02,      # Penalty for shooting (encourage efficiency)
    "R_TIME": 0.001,     # Small time penalty
    "R_DEATH": 5.0,      # Death penalty
}

# Reward Config 2: SURVIVAL_FOCUS (Prioritize staying alive and dodging)
REWARD_CONFIG_SURVIVAL = {
    "name": "survival",
    "description": "Prioritize survival - higher damage/death penalties, lower combat rewards",
    "R_PRIZE": 0.5,      # Reduced prize importance
    "R_HIT": 0.1,        # Lower combat reward
    "R_KILL": 0.5,       # Lower kill reward
    "R_DAMAGE": 3.0,     # MUCH higher damage penalty - encourages dodging
    "R_SHOT": 0.05,      # Higher shot cost
    "R_TIME": 0.0005,    # Lower time penalty (reward survival)
    "R_DEATH": 10.0,     # MUCH higher death penalty
}

# Reward Config 3: AGGRESSIVE (Prioritize combat and prize collection)
REWARD_CONFIG_AGGRESSIVE = {
    "name": "aggressive",
    "description": "Prioritize combat and prizes - higher rewards, lower penalties",
    "R_PRIZE": 2.0,      # MUCH higher prize reward
    "R_HIT": 0.5,        # Higher hit reward
    "R_KILL": 2.0,       # MUCH higher kill reward
    "R_DAMAGE": 0.5,     # Lower damage penalty - accept risk
    "R_SHOT": 0.01,      # Lower shot cost - encourage shooting
    "R_TIME": 0.002,     # Higher time penalty - encourage action
    "R_DEATH": 3.0,      # Lower death penalty - accept risk
}

# All reward configs for easy iteration
REWARD_CONFIGS = {
    "baseline": REWARD_CONFIG_BASELINE,
    "survival": REWARD_CONFIG_SURVIVAL,
    "aggressive": REWARD_CONFIG_AGGRESSIVE,
}

# ==============================================================================
# TIMESTEP CONFIGURATIONS
# ==============================================================================

TIMESTEP_CONFIGS = {
    "short": 50_000,       # Quick evaluation (50k)
    "medium": 500_000,     # Standard training (500k)
    "long": 1_600_000,     # Extended training (1.6M)
}

# ==============================================================================
# ALGORITHM HYPERPARAMETERS
# ==============================================================================

# PPO hyperparameters
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}

# DQN hyperparameters
DQN_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 128,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "verbose": 1,
}

# SAC hyperparameters (for continuous action approximation)
SAC_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_entropy": "auto",
    "verbose": 1,
}

# ==============================================================================
# TRAINING SETTINGS
# ==============================================================================

TRAINING_CONFIG = {
    "total_timesteps": 500_000,
    "save_freq": 10_000,
    "eval_freq": 5_000,
    "log_dir": "./logs",
    "model_dir": "./models",
    "tensorboard_log": "./tensorboard_logs",
}

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

EXPERIMENT_CONFIG = {
    "seeds": [42, 123, 456],
    "n_eval_episodes": 10,
    "algorithms": ["dqn", "ppo", "sac"],
    "reward_configs": ["baseline", "survival", "aggressive"],
    "timestep_configs": ["short", "medium", "long"],
}

# ==============================================================================
# EXPERIMENT MATRIX
# Generate all experiment combinations
# ==============================================================================

def get_experiment_matrix():
    """
    Generate all experiment configurations.
    Returns list of dicts with: algo, reward_config, timesteps, experiment_name
    """
    experiments = []
    
    for reward_name in EXPERIMENT_CONFIG["reward_configs"]:
        for timestep_name in EXPERIMENT_CONFIG["timestep_configs"]:
            for algo in EXPERIMENT_CONFIG["algorithms"]:
                exp_name = f"{algo}_{reward_name}_{timestep_name}"
                experiments.append({
                    "name": exp_name,
                    "algorithm": algo,
                    "reward_config": reward_name,
                    "reward_params": REWARD_CONFIGS[reward_name],
                    "timestep_config": timestep_name,
                    "timesteps": TIMESTEP_CONFIGS[timestep_name],
                })
    
    return experiments


# Print experiment summary when loaded
if __name__ == "__main__":
    experiments = get_experiment_matrix()
    print(f"Total experiments: {len(experiments)}")
    print("\nExperiment Matrix:")
    print("-" * 70)
    for exp in experiments:
        print(f"  {exp['name']:35} | {exp['timesteps']:>10,} steps")
    print("-" * 70)
    
    # Time estimates (rough)
    total_steps = sum(exp['timesteps'] for exp in experiments)
    print(f"\nTotal timesteps across all experiments: {total_steps:,}")
    print(f"Estimated time (at ~500 steps/sec): {total_steps / 500 / 3600:.1f} hours")
