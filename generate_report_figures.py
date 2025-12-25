#!/usr/bin/env python
"""
Generate plots and analysis for RL experiments report.
Creates learning curves, comparison plots, and summary statistics.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

# Set style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def load_all_metrics(experiments_dir: str = "./experiments") -> Dict[str, pd.DataFrame]:
    """Load metrics CSV files for all experiments"""
    metrics = {}
    
    for exp_dir in sorted(glob.glob(os.path.join(experiments_dir, "*"))):
        if not os.path.isdir(exp_dir):
            continue
        
        exp_name = os.path.basename(exp_dir)
        
        # Find metrics CSV
        for algo in ["dqn", "ppo", "sac"]:
            csv_path = os.path.join(exp_dir, "logs", f"{algo}_metrics.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    metrics[exp_name] = df
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
                break
    
    return metrics


def plot_learning_curves(metrics: Dict[str, pd.DataFrame], output_dir: str = "./report_figures"):
    """Plot learning curves for each algorithm grouped by reward config"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by algorithm
    algos = ["dqn", "ppo", "sac"]
    rewards = ["baseline", "survival", "aggressive"]
    timesteps = ["short", "medium", "long"]
    colors = {'short': 'blue', 'medium': 'green', 'long': 'red'}
    
    for algo in algos:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        for i, reward in enumerate(rewards):
            ax = axes[i]
            
            for ts in timesteps:
                exp_name = f"{algo}_{reward}_{ts}"
                if exp_name in metrics:
                    df = metrics[exp_name]
                    if 'reward' in df.columns and len(df) > 0:
                        # Rolling mean for smoother curve
                        window = max(1, len(df) // 20)
                        smooth = df['reward'].rolling(window=window, min_periods=1).mean()
                        ax.plot(df['timestep'], smooth, label=ts.capitalize(), 
                               alpha=0.8, linewidth=2, color=colors[ts])
            
            ax.set_title(f'{reward.capitalize()} Reward')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Episode Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{algo.upper()} Learning Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{algo}_learning_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {algo}_learning_curves.png")


def plot_algorithm_comparison(metrics: Dict[str, pd.DataFrame], output_dir: str = "./report_figures"):
    """Compare algorithms on same reward config"""
    os.makedirs(output_dir, exist_ok=True)
    
    algos = ["dqn", "ppo", "sac"]
    rewards = ["baseline", "survival", "aggressive"]
    colors = {'DQN': 'blue', 'PPO': 'green', 'SAC': 'red'}
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for i, reward in enumerate(rewards):
        ax = axes[i]
        
        for algo in algos:
            # Use medium timestep for comparison
            exp_name = f"{algo}_{reward}_medium"
            if exp_name in metrics:
                df = metrics[exp_name]
                if 'reward' in df.columns and len(df) > 0:
                    window = max(1, len(df) // 20)
                    smooth = df['reward'].rolling(window=window, min_periods=1).mean()
                    ax.plot(df['timestep'], smooth, label=algo.upper(), 
                           alpha=0.8, linewidth=2, color=colors[algo.upper()])
        
        ax.set_title(f'{reward.capitalize()} Reward (500k steps)')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Algorithm Comparison by Reward Config', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved algorithm_comparison.png")


def plot_reward_ablation(metrics: Dict[str, pd.DataFrame], output_dir: str = "./report_figures"):
    """Ablation study: effect of reward shaping"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate final performance for each config
    results = []
    
    for exp_name, df in metrics.items():
        parts = exp_name.split('_')
        if len(parts) >= 3:
            algo = parts[0]
            reward = parts[1]
            timestep = parts[2]
            
            if 'reward' in df.columns and len(df) > 0:
                # Last 10% of episodes
                n = max(1, len(df) // 10)
                final_reward = df['reward'].tail(n).mean()
                final_std = df['reward'].tail(n).std()
                
                results.append({
                    'Algorithm': algo.upper(),
                    'Reward Config': reward.capitalize(),
                    'Timesteps': timestep,
                    'Final Reward': final_reward,
                    'Std': final_std,
                })
    
    if not results:
        print("No results to plot for ablation")
        return
    
    df_results = pd.DataFrame(results)
    
    # Filter to medium timestep for cleaner comparison
    df_medium = df_results[df_results['Timesteps'] == 'medium']
    
    if len(df_medium) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Grouped bar chart
        x = np.arange(3)  # 3 reward configs
        width = 0.25
        colors = {'DQN': 'blue', 'PPO': 'green', 'SAC': 'red'}
        
        for i, algo in enumerate(['DQN', 'PPO', 'SAC']):
            algo_data = df_medium[df_medium['Algorithm'] == algo]
            rewards_list = []
            stds = []
            for rc in ['Baseline', 'Survival', 'Aggressive']:
                row = algo_data[algo_data['Reward Config'] == rc]
                if len(row) > 0:
                    rewards_list.append(row['Final Reward'].values[0])
                    stds.append(row['Std'].values[0])
                else:
                    rewards_list.append(0)
                    stds.append(0)
            
            bars = ax.bar(x + i * width, rewards_list, width, label=algo, 
                         yerr=stds, capsize=3, color=colors[algo], alpha=0.8)
        
        ax.set_xlabel('Reward Configuration')
        ax.set_ylabel('Final Episode Reward')
        ax.set_title('Reward Shaping Ablation (500k timesteps)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Baseline', 'Survival', 'Aggressive'])
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_ablation.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved reward_ablation.png")


def plot_timestep_scaling(metrics: Dict[str, pd.DataFrame], output_dir: str = "./report_figures"):
    """Show how performance scales with training time"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    timestep_map = {'short': 50000, 'medium': 500000, 'long': 1600000}
    
    for exp_name, df in metrics.items():
        parts = exp_name.split('_')
        if len(parts) >= 3:
            algo = parts[0]
            reward = parts[1]
            timestep = parts[2]
            
            if 'reward' in df.columns and len(df) > 0:
                n = max(1, len(df) // 10)
                final_reward = df['reward'].tail(n).mean()
                
                results.append({
                    'Algorithm': algo.upper(),
                    'Reward': reward,
                    'Timesteps': timestep_map.get(timestep, 0),
                    'Timestep Label': timestep,
                    'Final Reward': final_reward,
                })
    
    if not results:
        return
    
    df_results = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot for baseline reward config
    for algo in ['DQN', 'PPO', 'SAC']:
        algo_data = df_results[(df_results['Algorithm'] == algo) & (df_results['Reward'] == 'baseline')]
        algo_data = algo_data.sort_values('Timesteps')
        if len(algo_data) > 0:
            ax.plot(algo_data['Timesteps'], algo_data['Final Reward'], 
                   marker='o', label=algo, linewidth=2, markersize=8)
    
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Final Episode Reward')
    ax.set_title('Performance Scaling with Training Time (Baseline Reward)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timestep_scaling.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved timestep_scaling.png")


def generate_summary_table(metrics: Dict[str, pd.DataFrame], output_dir: str = "./report_figures"):
    """Generate LaTeX table with summary statistics"""
    os.makedirs(output_dir, exist_ok=True)
    
    rows = []
    for exp_name, df in sorted(metrics.items()):
        if 'reward' in df.columns and len(df) > 0:
            n = max(1, len(df) // 10)
            final_reward = df['reward'].tail(n).mean()
            final_std = df['reward'].tail(n).std()
            eps = len(df)
            
            parts = exp_name.split('_')
            algo = parts[0].upper() if len(parts) > 0 else exp_name
            reward = parts[1].capitalize() if len(parts) > 1 else ""
            ts = parts[2] if len(parts) > 2 else ""
            
            rows.append({
                'Algorithm': algo,
                'Reward': reward,
                'Steps': ts,
                'Episodes': eps,
                'Mean Reward': f"{final_reward:.2f}",
                'Std': f"{final_std:.2f}",
            })
    
    df_summary = pd.DataFrame(rows)
    
    # Save as CSV
    df_summary.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)
    
    # Generate LaTeX table
    latex = df_summary.to_latex(index=False, caption='Experiment Results Summary', label='tab:results')
    with open(os.path.join(output_dir, 'summary_table.tex'), 'w') as f:
        f.write(latex)
    
    print("Saved summary_table.csv and summary_table.tex")
    return df_summary


def main():
    print("Loading experiment metrics...")
    metrics = load_all_metrics("./experiments")
    print(f"Loaded {len(metrics)} experiments")
    
    if len(metrics) == 0:
        print("No metrics found!")
        return
    
    output_dir = "./report_figures"
    
    print("\nGenerating plots...")
    plot_learning_curves(metrics, output_dir)
    plot_algorithm_comparison(metrics, output_dir)
    plot_reward_ablation(metrics, output_dir)
    plot_timestep_scaling(metrics, output_dir)
    
    print("\nGenerating summary table...")
    summary = generate_summary_table(metrics, output_dir)
    print("\nSummary:")
    print(summary.to_string())
    
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
