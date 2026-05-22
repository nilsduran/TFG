from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def agg_by_subject(df_epochs: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df_epochs.groupby(["subject", "n_epochs_total"], as_index=False)[metric]
        .mean()
        .sort_values(["subject", "n_epochs_total"])
    )


def plot_subject_lines(
    df_plot: pd.DataFrame,
    *,
    y_col: str,
    title: str,
    y_label: str,
    subject_order: Sequence[str] | None = None,
    output_file: Path | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=df_plot,
        x="n_epochs_total",
        y=y_col,
        hue="subject",
        hue_order=subject_order,
        palette="tab20",
        marker="o",
        linewidth=2,
    )
    if y_col in ["delta_acc", "delta_acc_q95"]:
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("n_epochs_total")
    plt.ylabel(y_label)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(alpha=0.25)
    plt.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()


def plot_accuracy_distributions(
    df_epochs: pd.DataFrame,
    *,
    output_file: Path | None = None,
) -> None:
    plt.figure(figsize=(12, 5))
    sns.histplot(df_epochs, x="acc_real_mean", color="tab:blue", label="Acc real", kde=True, stat="density", bins=30)
    sns.histplot(
        df_epochs,
        x="acc_chance_mean",
        color="tab:orange",
        label="Chance mean",
        kde=True,
        stat="density",
        bins=30,
    )
    sns.histplot(
        df_epochs,
        x="acc_chance_q95",
        color="tab:green",
        label="Chance q95",
        kde=True,
        stat="density",
        bins=30,
    )
    plt.xlabel("Accuracy")
    plt.ylabel("Density")
    plt.title("Global distribution: real accuracy vs chance")
    plt.legend()
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()


def plot_metric_by_label(
    df_epochs: pd.DataFrame,
    *,
    metric: str,
    title: str,
    y_label: str,
    output_file: Path | None = None,
) -> None:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_epochs, x="n_epochs_total", y=metric, hue="electrode", marker="o", linewidth=2)
    if metric in {"delta_acc", "delta_acc_q95"}:
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("n_epochs_total")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()


def plot_surpass_rate(
    df_epochs: pd.DataFrame,
    *,
    output_file: Path | None = None,
) -> pd.DataFrame:
    df_plot = df_epochs.copy()
    df_plot["surpass_q95"] = (df_plot["acc_real_mean"] > df_plot["acc_chance_q95"]).astype(float)
    surpass_tbl = (
        df_plot.groupby(["electrode", "n_epochs_total"])["surpass_q95"]
        .mean()
        .reset_index()
        .sort_values(["electrode", "n_epochs_total"])
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=surpass_tbl, x="n_epochs_total", y="surpass_q95", hue="electrode", marker="o", linewidth=2)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("n_epochs_total")
    plt.ylabel("Fraction subjects above q95")
    plt.title("Exceedance rate of the q95 threshold")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()
    return surpass_tbl


def plot_time_map(
    values: np.ndarray,
    *,
    labels: Sequence[str],
    window_centers: np.ndarray,
    title: str,
    colorbar_label: str,
    output_file: Path | None = None,
    cmap: str = "RdBu_r",
    center: float | None = 0.0,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(window_centers) * 0.45), max(2.8, len(labels) * 0.65)))
    sns.heatmap(
        values,
        ax=ax,
        cmap=cmap,
        center=center,
        xticklabels=[f"{t:.2f}" for t in window_centers],
        yticklabels=list(labels),
        cbar_kws={"label": colorbar_label},
    )
    ax.set_xlabel("Window center (s)")
    ax.set_ylabel("Label")
    ax.set_title(title)
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.show()


def classifier_sensor_weights(model, n_sensors: int) -> np.ndarray:
    final_estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    coef = np.asarray(getattr(final_estimator, "coef_", np.array([]))).ravel()
    if coef.size == 0:
        return np.zeros(n_sensors, dtype=float)
    if coef.size == n_sensors:
        return coef.astype(float)
    tri_size = n_sensors * (n_sensors + 1) // 2
    if coef.size == tri_size:
        weights = np.zeros(n_sensors, dtype=float)
        k = 0
        for i in range(n_sensors):
            for j in range(i, n_sensors):
                value = float(abs(coef[k]))
                if i == j:
                    weights[i] += value
                else:
                    weights[i] += 0.5 * value
                    weights[j] += 0.5 * value
                k += 1
        return weights
    return np.abs(np.resize(coef, n_sensors)).astype(float)


def compute_gradient_saliency(
    model: "torch.nn.Module",
    X: np.ndarray,
    y: np.ndarray,
    device: str = "cpu",
    mode: str = "input_grad",
) -> np.ndarray:
    """Calcula la rellevància temporal-espacial usant els gradients de PyTorch."""
    import torch

    model.eval()
    
    # Ens assegurem que el model estigui al dispositiu correcte
    device_obj = torch.device(device)
    model.to(device_obj)

    # Convertim a tensor i activem gradients
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device_obj).requires_grad_(True)
    
    # Forward pass
    logits = model(X_tensor)
    
    # Agafem les puntuacions de la classe correcte/desitjada
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device_obj)
    scores = logits[torch.arange(len(y_tensor)), y_tensor]
    
    # Backpropagation per obtenir gradients respecte a l'entrada X
    model.zero_grad()
    scores.sum().backward()
    
    grads = X_tensor.grad.detach().cpu().numpy()
    
    if mode == "grad":
        saliency = np.abs(grads)
    elif mode == "input_grad":
        saliency = grads * X
    else:
        raise ValueError(f"Saliency mode not supported: {mode}")
        
    return saliency


def plot_eeg_topomap(
    weights: np.ndarray,
    ch_names: list[str],
    *,
    info: "mne.Info" | None = None,
    title: str = "",
    cmap: str = "RdBu_r",
    output_file: Path | None = None,
    show: bool = True,
) -> None:
    """Dibuixa un topomap 2D del cap a partir de pesos de canals usant MNE."""
    import mne
    import matplotlib.pyplot as plt

    if info is None:
        # Creem info mínima amb montage estàndard
        info = mne.create_info(ch_names=list(ch_names), sfreq=128.0, ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1005")
        info.set_montage(montage, on_missing="ignore")

    fig, ax = plt.subplots(figsize=(5, 5))
    im, _ = mne.viz.plot_topomap(weights, info, axes=ax, cmap=cmap, show=False)
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_attention_heatmap(
    attention: np.ndarray,
    *,
    title: str = "Attention Map",
    xlabel: str = "Key tokens",
    ylabel: str = "Query tokens",
    xticklabels: list[str] | list[bool] | None = None,
    yticklabels: list[str] | list[bool] | None = None,
    output_file: Path | None = None,
    show: bool = True,
) -> None:
    """Dibuixa un heatmap dels pesos d'atenció d'un Transformer."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        attention,
        ax=ax,
        cmap="viridis",
        xticklabels=xticklabels if xticklabels is not None else True,
        yticklabels=yticklabels if yticklabels is not None else True,
        cbar_kws={"label": "Attention weight"},
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        
    if show:
        plt.show()
    else:
        plt.close(fig)

