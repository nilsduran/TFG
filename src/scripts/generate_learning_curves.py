import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to sys.path so we can import our project modules
try:
    scripts_dir = Path(__file__).resolve().parent
    src_dir = scripts_dir.parent
except NameError:
    src_dir = Path.cwd() / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from br_analysis import detect_project_paths

def main():
    # Detect Google Drive project paths
    paths = detect_project_paths("cross_model_comparison", in_colab=False)
    analysis_root = paths.analysis_root
    
    eegnet_path = analysis_root / "c_vs_i_eegnet" / "tables" / "cohort_fold_metrics_c_vs_ip_eegnet.csv"
    riemann_path = analysis_root / "c_vs_i_riemann" / "tables" / "epochs_grid_riemann.csv"
    lr_path = analysis_root / "c_vs_i" / "tables" / "epochs_grid_c_vs_i_FCz_theta_Oz_alpha.csv"
    
    figures_dir = analysis_root / "cross_model_comparison" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if not (eegnet_path.exists() and riemann_path.exists() and lr_path.exists()):
        print(f"Error: Missing results files in {analysis_root}")
        print(f"eegnet_path exists: {eegnet_path.exists()} ({eegnet_path})")
        print(f"riemann_path exists: {riemann_path.exists()} ({riemann_path})")
        print(f"lr_path exists: {lr_path.exists()} ({lr_path})")
        return
        
    # Load data
    df_eeg = pd.read_csv(eegnet_path)
    df_rie = pd.read_csv(riemann_path)
    df_lr = pd.read_csv(lr_path)
    
    # 1. Map total trials per subject from Riemann epochs grid (which is the session size)
    subject_trials = df_rie.groupby('subject')['n_epochs_total'].max().to_dict()
    df_eeg['total_trials'] = df_eeg['subject'].map(subject_trials)
    
    # In 5-split chronological CV:
    # training size (train + val) for fold 'fold' is exactly (fold / 6) * total_trials.
    df_eeg['train_trials'] = (df_eeg['fold'] / 6.0) * df_eeg['total_trials']
    
    # 2. For Riemann & LR epochs grid:
    # average training size across folds 1-5 is 0.5 * n_epochs_total
    df_rie['train_trials'] = 0.5 * df_rie['n_epochs_total']
    df_lr['train_trials'] = 0.5 * df_lr['n_epochs_total']
    
    # Compute averages across subjects
    eeg_agnostic = df_eeg[df_eeg['model'] == 'Agnostic'].groupby('fold')[['train_trials', 'test_acc']].mean().reset_index()
    eeg_clustering = df_eeg[df_eeg['model'] == 'Clustering'].groupby('fold')[['train_trials', 'test_acc']].mean().reset_index()
    
    riemann = df_rie.groupby('n_epochs_total')[['train_trials', 'acc_real_mean']].mean().reset_index()
    
    lr_theta = df_lr[df_lr['electrode'] == 'FCz-theta'].groupby('n_epochs_total')[['train_trials', 'acc_real_mean']].mean().reset_index()
    lr_alpha = df_lr[df_lr['electrode'] == 'Oz-alpha'].groupby('n_epochs_total')[['train_trials', 'acc_real_mean']].mean().reset_index()
    
    # Setup plotting style
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        'font.sans-serif': 'Inter, Outfit, sans-serif',
        'font.family': 'sans-serif',
        'grid.color': '#eaeaea',
        'grid.linestyle': '--'
    })
    
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    # Plot curves
    # Deep Learning models (EEGNet)
    ax.plot(eeg_agnostic['train_trials'], eeg_agnostic['test_acc'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='EEGNet Agnostic (60 ch)')
    ax.plot(eeg_clustering['train_trials'], eeg_clustering['test_acc'], 's--', color='#aec7e8', linewidth=2.0, markersize=7, label='EEGNet Clustering (29 ch)')
    
    # Linear/Shallow models (Riemann, LR)
    ax.plot(riemann['train_trials'], riemann['acc_real_mean'], 'd-', color='#2ca02c', linewidth=2.5, markersize=8, label='Riemannian TS-LDA (60 ch)')
    ax.plot(lr_alpha['train_trials'], lr_alpha['acc_real_mean'], '^-', color='#ff7f0e', linewidth=2.0, markersize=7, label='Logistic Reg. (Oz-alpha)')
    ax.plot(lr_theta['train_trials'], lr_theta['acc_real_mean'], 'v-', color='#ffbb78', linewidth=1.8, markersize=7, label='Logistic Reg. (FCz-theta)')
    
    # Plot chance line
    ax.axhline(0.5, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.8, label='Chance (0.50)')
    
    # Labels & Title
    ax.set_xlabel('Average Training Set Size (number of trials)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Cohort-Wide Mean Accuracy', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Cohort-Wide Decoding Learning Curves (Task: Congruent vs Incongruent Pure)', fontsize=13, fontweight='bold', pad=15)
    
    # Limits & Grid
    ax.set_xlim(0, 220)
    ax.set_ylim(0.44, 0.62)
    ax.grid(True)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#eaeaea', fontsize=10)
    
    sns.despine(trim=True)
    fig.tight_layout()
    
    output_path = figures_dir / "cohort_learning_curves_comparison_c_vs_ip.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved plot to: {output_path}")
    plt.close()

if __name__ == '__main__':
    main()
