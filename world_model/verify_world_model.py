#!/usr/bin/env python
"""
Verification script for World Model components (VAE and MDN-RNN).
Diagnoses common issues before PPO training.

Usage:
    python -m world_model.verify_world_model --model-dir ./world_model_agent_long
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.vae import VAE
from world_model.mdn_rnn import MDNRNN


def load_models(model_dir: str, device: str = "cpu"):
    """Load VAE and MDN-RNN from directory."""
    vae_path = os.path.join(model_dir, "vae.pt")
    mdn_path = os.path.join(model_dir, "mdn_rnn.pt")
    
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"VAE not found: {vae_path}")
    if not os.path.exists(mdn_path):
        raise FileNotFoundError(f"MDN-RNN not found: {mdn_path}")
    
    vae = VAE(in_channels=3, latent_dim=32)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.to(device).eval()
    
    mdn_rnn = MDNRNN(latent_dim=32, action_dim=80, hidden_dim=256)
    mdn_rnn.load_state_dict(torch.load(mdn_path, map_location=device))
    mdn_rnn.to(device).eval()
    
    return vae, mdn_rnn


def load_data(model_dir: str):
    """Load collected data for testing."""
    data_path = os.path.join(model_dir, "collected_data.npz")
    if not os.path.exists(data_path):
        # Try parent checkpoints directory
        data_path = os.path.join(os.path.dirname(model_dir), "world_model_checkpoints", "collected_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found in {model_dir}")
    
    data = np.load(data_path)
    return {
        'frames': data['frames'],
        'actions': data['actions'],
        'rewards': data['rewards'],
        'dones': data['dones'],
    }


def test_vae_reconstruction(vae, frames, device, save_dir):
    """Test VAE reconstruction quality."""
    print("\n" + "="*60)
    print("VAE RECONSTRUCTION TEST")
    print("="*60)
    
    # Sample random frames
    n_samples = min(100, len(frames))
    indices = np.random.choice(len(frames), n_samples, replace=False)
    sample_frames = frames[indices]
    
    # Convert to tensor (already normalized to [0,1])
    frames_t = torch.tensor(sample_frames, dtype=torch.float32, device=device)
    frames_t = frames_t.permute(0, 3, 1, 2)  # (N, C, H, W)
    
    with torch.no_grad():
        recon, mu, logvar = vae(frames_t)
    
    # Compute metrics
    mse = ((recon - frames_t) ** 2).mean().item()
    mae = (recon - frames_t).abs().mean().item()
    
    # VAE latent stats
    mu_mean = mu.mean().item()
    mu_std = mu.std().item()
    logvar_mean = logvar.mean().item()
    logvar_std = logvar.std().item()
    sigma_mean = torch.exp(0.5 * logvar).mean().item()
    
    print(f"\nReconstruction Quality:")
    print(f"  MSE:  {mse:.6f}  {'✓ Good' if mse < 0.02 else '✗ HIGH - VAE may be undertrained'}")
    print(f"  MAE:  {mae:.6f}")
    
    print(f"\nLatent Space Statistics:")
    print(f"  mu mean/std:     {mu_mean:.4f} / {mu_std:.4f}")
    print(f"  logvar mean/std: {logvar_mean:.4f} / {logvar_std:.4f}")
    print(f"  sigma mean:      {sigma_mean:.4f}")
    
    # Check for posterior collapse
    if sigma_mean < 0.1:
        print(f"\n  ⚠️  WARNING: VAE sigma very low ({sigma_mean:.4f})")
        print(f"      This may indicate posterior collapse (KL too strong)")
    elif sigma_mean > 5.0:
        print(f"\n  ⚠️  WARNING: VAE sigma very high ({sigma_mean:.4f})")
        print(f"      The VAE may not be learning meaningful latents")
    else:
        print(f"\n  ✓ Latent distribution looks reasonable")
    
    # Save sample reconstructions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        # Original
        orig = frames_t[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis('off')
        
        # Reconstruction
        rec = recon[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(rec)
        axes[1, i].set_title(f"Reconstructed")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "vae_reconstruction_test.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"\nSaved reconstruction samples to: {save_path}")
    
    return {
        'mse': mse,
        'mae': mae,
        'mu_mean': mu_mean,
        'mu_std': mu_std,
        'sigma_mean': sigma_mean,
    }


def test_mdn_prediction(mdn_rnn, vae, frames, actions, rewards, dones, device, save_dir, action_dim=80):
    """Test MDN-RNN prediction quality and sigma statistics."""
    print("\n" + "="*60)
    print("MDN-RNN PREDICTION TEST")
    print("="*60)
    
    # Encode frames to latents
    frames_t = torch.tensor(frames, dtype=torch.float32, device=device)
    frames_t = frames_t.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        latents = []
        for i in range(0, len(frames_t), 128):
            z = vae.encode(frames_t[i:i+128], deterministic=True)
            latents.append(z.cpu())
        latents = torch.cat(latents, dim=0).numpy()
    
    print(f"\nEncoded {len(latents)} frames to latents")
    
    # One-hot encode actions
    action_flat = np.array([np.ravel_multi_index(a, [5, 2, 8]) for a in actions])
    actions_onehot = np.zeros((len(actions), action_dim), dtype=np.float32)
    actions_onehot[np.arange(len(actions)), action_flat] = 1.0
    
    # Test predictions on sequences
    seq_len = 16
    n_sequences = 100
    
    all_sigma_min = []
    all_sigma_max = []
    all_sigma_mean = []
    reward_preds = []
    reward_targets = []
    prediction_errors = []
    
    # Find valid sequence starts (not crossing episode boundaries)
    valid_starts = []
    for i in range(len(latents) - seq_len - 1):
        if not any(dones[i:i+seq_len]):
            valid_starts.append(i)
    
    if len(valid_starts) < n_sequences:
        print(f"  Warning: Only {len(valid_starts)} valid sequences (needed {n_sequences})")
        n_sequences = len(valid_starts)
    
    if n_sequences == 0:
        print("  ERROR: No valid sequences found! Episodes may be too short.")
        return None
    
    sample_starts = np.random.choice(valid_starts, min(n_sequences, len(valid_starts)), replace=False)
    
    for start in sample_starts:
        z_seq = torch.tensor(latents[start:start+seq_len], dtype=torch.float32, device=device).unsqueeze(0)
        a_seq = torch.tensor(actions_onehot[start:start+seq_len], dtype=torch.float32, device=device).unsqueeze(0)
        z_next = torch.tensor(latents[start+1:start+seq_len+1], dtype=torch.float32, device=device).unsqueeze(0)
        r_seq = rewards[start:start+seq_len]
        
        with torch.no_grad():
            pi, mu, sigma, r_pred, d_pred, _ = mdn_rnn(z_seq, a_seq)
        
        # Sigma statistics
        sigma_np = sigma.cpu().numpy()
        all_sigma_min.append(sigma_np.min())
        all_sigma_max.append(sigma_np.max())
        all_sigma_mean.append(sigma_np.mean())
        
        # Reward predictions
        if r_pred is not None:
            reward_preds.extend(r_pred.squeeze().cpu().numpy().tolist())
            reward_targets.extend(r_seq.tolist())
        
        # Prediction error (using deterministic prediction)
        # pi: (B, T, K), mu: (B, T, K, latent_dim) - need to reshape for get_deterministic
        B_seq, T_seq, K = pi.shape
        pi_flat = pi.view(B_seq * T_seq, K)
        mu_flat = mu.view(B_seq * T_seq, K, -1)
        z_pred = mdn_rnn.get_deterministic(pi_flat, mu_flat)  # (B*T, latent_dim)
        z_pred = z_pred.view(B_seq, T_seq, -1)  # (B, T, latent_dim)
        error = ((z_pred - z_next) ** 2).mean().item()
        prediction_errors.append(error)
    
    # Report statistics
    sigma_min = np.min(all_sigma_min)
    sigma_max = np.max(all_sigma_max)
    sigma_mean = np.mean(all_sigma_mean)
    
    print(f"\nSigma Statistics (from {n_sequences} sequences):")
    print(f"  Min sigma:  {sigma_min:.6f}  {'✗ COLLAPSED!' if sigma_min < 1e-4 else '✓ OK'}")
    print(f"  Max sigma:  {sigma_max:.4f}")
    print(f"  Mean sigma: {sigma_mean:.4f}")
    
    if sigma_min < 1e-4:
        print(f"\n  ⚠️  WARNING: MDN sigma collapsed to near-zero!")
        print(f"      Model is overconfident. Consider:")
        print(f"      - Adding sigma floor (e.g., sigma = sigma + 0.01)")
        print(f"      - Reducing training epochs")
        print(f"      - Adding regularization")
    
    pred_error_mean = np.mean(prediction_errors)
    print(f"\nLatent Prediction Error:")
    print(f"  Mean MSE: {pred_error_mean:.6f}")
    
    if reward_preds:
        reward_preds = np.array(reward_preds)
        reward_targets = np.array(reward_targets)
        reward_corr = np.corrcoef(reward_preds, reward_targets)[0, 1]
        reward_mse = ((reward_preds - reward_targets) ** 2).mean()
        
        print(f"\nReward Prediction:")
        print(f"  MSE:         {reward_mse:.6f}")
        print(f"  Correlation: {reward_corr:.4f}  {'✓ Good' if reward_corr > 0.3 else '✗ Weak prediction'}")
        print(f"  Pred range:  [{reward_preds.min():.3f}, {reward_preds.max():.3f}]")
        print(f"  True range:  [{reward_targets.min():.3f}, {reward_targets.max():.3f}]")
        
        if np.abs(reward_preds).max() < 0.1:
            print(f"\n  ⚠️  WARNING: Reward predictions near zero!")
            print(f"      MDN-RNN may not be learning reward dynamics.")
    
    # Save sigma distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(all_sigma_mean, bins=30, edgecolor='black')
    axes[0].axvline(x=sigma_mean, color='r', linestyle='--', label=f'Mean: {sigma_mean:.4f}')
    axes[0].set_xlabel('Mean Sigma per Sequence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('MDN Sigma Distribution')
    axes[0].legend()
    
    if reward_preds is not None and len(reward_preds) > 0:
        axes[1].scatter(reward_targets, reward_preds, alpha=0.5, s=10)
        axes[1].plot([reward_targets.min(), reward_targets.max()], 
                     [reward_targets.min(), reward_targets.max()], 'r--', label='Perfect')
        axes[1].set_xlabel('True Reward')
        axes[1].set_ylabel('Predicted Reward')
        axes[1].set_title(f'Reward Prediction (corr={reward_corr:.3f})')
        axes[1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "mdn_rnn_test.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"\nSaved MDN-RNN analysis to: {save_path}")
    
    return {
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'sigma_mean': sigma_mean,
        'pred_error': pred_error_mean,
        'reward_corr': reward_corr if reward_preds is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify World Model Components")
    parser.add_argument("--model-dir", type=str, default="./world_model_agent_long",
                        help="Directory containing VAE and MDN-RNN models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae-only", action="store_true", help="Only test VAE")
    parser.add_argument("--mdn-only", action="store_true", help="Only test MDN-RNN")
    args = parser.parse_args()
    
    print("="*60)
    print("WORLD MODEL VERIFICATION")
    print("="*60)
    print(f"Model directory: {args.model_dir}")
    print(f"Device: {args.device}")
    
    # Load models
    try:
        vae, mdn_rnn = load_models(args.model_dir, args.device)
        print(f"✓ Loaded VAE and MDN-RNN")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return 1
    
    # Load data
    try:
        data = load_data(args.model_dir)
        print(f"✓ Loaded {len(data['frames'])} frames")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return 1
    
    # Print reward statistics
    print(f"\nData Statistics:")
    print(f"  Rewards: min={data['rewards'].min():.3f}, max={data['rewards'].max():.3f}, "
          f"mean={data['rewards'].mean():.3f}, std={data['rewards'].std():.3f}")
    print(f"  Non-zero rewards: {(data['rewards'] != 0).sum()} / {len(data['rewards'])} "
          f"({100*(data['rewards'] != 0).mean():.1f}%)")
    
    save_dir = args.model_dir
    
    # Run tests
    results = {}
    
    if not args.mdn_only:
        results['vae'] = test_vae_reconstruction(vae, data['frames'], args.device, save_dir)
    
    if not args.vae_only:
        results['mdn'] = test_mdn_prediction(
            mdn_rnn, vae, data['frames'], data['actions'], 
            data['rewards'], data['dones'], args.device, save_dir
        )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    issues = []
    
    if 'vae' in results:
        if results['vae']['mse'] > 0.02:
            issues.append("VAE reconstruction MSE is high")
        if results['vae']['sigma_mean'] < 0.1:
            issues.append("VAE sigma collapsed (posterior collapse)")
    
    if 'mdn' in results and results['mdn']:
        if results['mdn']['sigma_min'] < 1e-4:
            issues.append("MDN sigma collapsed")
        if results['mdn']['reward_corr'] and results['mdn']['reward_corr'] < 0.3:
            issues.append("MDN reward prediction is weak")
    
    if issues:
        print("\n⚠️  Issues Found:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n  Recommendation: Use --use-real-env for PPO training")
    else:
        print("\n✓ No major issues detected")
        print("  If PPO still fails, consider using --use-real-env anyway")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
