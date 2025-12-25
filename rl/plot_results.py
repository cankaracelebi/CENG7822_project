"""
Plotting script for RL experiment results.
Generates learning curves and comparison plots.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path


def load_metrics(log_dir: str, algo: str) -> Optional[pd.DataFrame]:
    """Load metrics CSV for an algorithm."""
    csv_path = os.path.join(log_dir, algo, f"{algo}_metrics.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    
    # Also check in the direct log_dir
    csv_path = os.path.join(log_dir, f"{algo}_metrics.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    
    return None


def load_evaluations(log_dir: str, algo: str) -> Optional[np.ndarray]:
    """Load evaluation results from EvalCallback."""
    eval_path = os.path.join(log_dir, algo, "evaluations.npz")
    if os.path.exists(eval_path):
        data = np.load(eval_path)
        return {
            "timesteps": data["timesteps"],
            "results": data["results"],
            "ep_lengths": data["ep_lengths"],
        }
    return None


def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply rolling average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_learning_curve(
    df: pd.DataFrame,
    algo: str,
    output_dir: str,
    window: int = 50,
):
    """Plot learning curve for a single algorithm."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{algo.upper()} Learning Curves", fontsize=16, fontweight="bold")
    
    # Episode reward
    ax = axes[0, 0]
    rewards = df["reward"].values
    smoothed = smooth(rewards, window)
    ax.plot(df["timestep"].values[:len(smoothed)], smoothed, linewidth=2, label=f"{algo} (smoothed)")
    ax.fill_between(
        df["timestep"].values[:len(smoothed)],
        smooth(rewards - np.std(rewards), window),
        smooth(rewards + np.std(rewards), window),
        alpha=0.2
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Reward vs Timesteps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Episode length
    ax = axes[0, 1]
    lengths = df["length"].values
    smoothed_len = smooth(lengths, window)
    ax.plot(df["timestep"].values[:len(smoothed_len)], smoothed_len, linewidth=2, color="orange")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length vs Timesteps")
    ax.grid(True, alpha=0.3)
    
    # Survival rate (rolling average)
    ax = axes[1, 0]
    if "survival_rate" in df.columns:
        survival = df["survival_rate"].values
        smoothed_surv = smooth(survival, window)
        ax.plot(df["timestep"].values[:len(smoothed_surv)], smoothed_surv, linewidth=2, color="green")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Survival Rate")
        ax.set_title("Survival Rate vs Timesteps")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    # Reward distribution histogram
    ax = axes[1, 1]
    ax.hist(rewards, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(rewards), color="red", linestyle="--", label=f"Mean: {np.mean(rewards):.2f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Frequency")
    ax.set_title("Reward Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{algo}_learning_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved {algo} learning curve to {save_path}")
    return save_path


def plot_comparison(
    data: Dict[str, pd.DataFrame],
    output_dir: str,
    window: int = 50,
):
    """Plot comparison of all algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Algorithm Comparison", fontsize=16, fontweight="bold")
    
    colors = {"dqn": "#2ecc71", "ppo": "#3498db", "sac": "#e74c3c"}
    
    # Reward comparison
    ax = axes[0, 0]
    for algo, df in data.items():
        if df is not None and len(df) > 0:
            rewards = df["reward"].values
            smoothed = smooth(rewards, window)
            timesteps = df["timestep"].values[:len(smoothed)]
            ax.plot(timesteps, smoothed, linewidth=2, label=algo.upper(), color=colors.get(algo, None))
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Reward Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Episode length comparison
    ax = axes[0, 1]
    for algo, df in data.items():
        if df is not None and len(df) > 0:
            lengths = df["length"].values
            smoothed = smooth(lengths, window)
            timesteps = df["timestep"].values[:len(smoothed)]
            ax.plot(timesteps, smoothed, linewidth=2, label=algo.upper(), color=colors.get(algo, None))
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final reward box plot
    ax = axes[1, 0]
    final_rewards = []
    labels = []
    for algo, df in data.items():
        if df is not None and len(df) > 0:
            # Take last 100 episodes for final performance
            final = df["reward"].tail(100).values
            final_rewards.append(final)
            labels.append(algo.upper())
    
    if final_rewards:
        bp = ax.boxplot(final_rewards, labels=labels, patch_artist=True)
        for patch, algo in zip(bp["boxes"], data.keys()):
            patch.set_facecolor(colors.get(algo, "#888888"))
            patch.set_alpha(0.6)
    ax.set_ylabel("Episode Reward")
    ax.set_title("Final Performance (Last 100 Episodes)")
    ax.grid(True, alpha=0.3)
    
    # Summary statistics table
    ax = axes[1, 1]
    ax.axis("off")
    
    summary_data = []
    for algo, df in data.items():
        if df is not None and len(df) > 0:
            final = df["reward"].tail(100)
            summary_data.append([
                algo.upper(),
                f"{final.mean():.2f}",
                f"{final.std():.2f}",
                f"{final.max():.2f}",
                f"{len(df)}"
            ])
    
    if summary_data:
        table = ax.table(
            cellText=summary_data,
            colLabels=["Algorithm", "Mean", "Std", "Max", "Episodes"],
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
    ax.set_title("Summary Statistics (Last 100 Episodes)", pad=20)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "algorithm_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved comparison plot to {save_path}")
    return save_path


def generate_summary_report(data: Dict[str, pd.DataFrame], output_dir: str):
    """Generate a text summary report."""
    report_lines = [
        "=" * 60,
        "RL EXPERIMENT SUMMARY REPORT",
        "=" * 60,
        "",
    ]
    
    for algo, df in data.items():
        if df is not None and len(df) > 0:
            report_lines.append(f"\n{algo.upper()} Results:")
            report_lines.append("-" * 40)
            report_lines.append(f"  Total Episodes: {len(df)}")
            report_lines.append(f"  Total Timesteps: {df['timestep'].max():,}")
            report_lines.append(f"  Mean Reward: {df['reward'].mean():.2f} ± {df['reward'].std():.2f}")
            report_lines.append(f"  Max Reward: {df['reward'].max():.2f}")
            report_lines.append(f"  Min Reward: {df['reward'].min():.2f}")
            report_lines.append(f"  Mean Episode Length: {df['length'].mean():.1f}")
            
            # Final 100 episodes
            final = df.tail(100)
            report_lines.append(f"\n  Final Performance (last 100 episodes):")
            report_lines.append(f"    Mean Reward: {final['reward'].mean():.2f} ± {final['reward'].std():.2f}")
            report_lines.append(f"    Mean Length: {final['length'].mean():.1f}")
            
            if "survival_rate" in df.columns:
                report_lines.append(f"    Survival Rate: {final['survival_rate'].mean():.2%}")
    
    report_lines.append("\n" + "=" * 60)
    
    report = "\n".join(report_lines)
    print(report)
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "experiment_summary.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\nSaved summary report to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Plot RL experiment results")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory containing log files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Smoothing window size (default: 50)",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["dqn", "ppo", "sac"],
        help="Algorithms to plot",
    )
    
    args = parser.parse_args()
    
    print(f"Loading metrics from {args.log_dir}...")
    
    # Load all available data
    data = {}
    for algo in args.algos:
        df = load_metrics(args.log_dir, algo)
        if df is not None:
            print(f"  Loaded {algo}: {len(df)} episodes")
            data[algo] = df
        else:
            print(f"  No data found for {algo}")
            data[algo] = None
    
    if not any(d is not None for d in data.values()):
        print("\nNo data found! Make sure training has generated metrics files.")
        return
    
    # Generate individual plots
    for algo, df in data.items():
        if df is not None:
            plot_learning_curve(df, algo, args.output_dir, args.window)
    
    # Generate comparison plot
    if sum(1 for d in data.values() if d is not None) > 1:
        plot_comparison(data, args.output_dir, args.window)
    
    # Generate summary report
    generate_summary_report(data, args.output_dir)
    
    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
