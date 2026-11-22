"""Pooled deep-learning benchmark for the binocular-rivalry EEG data (GPU).

Self-contained: needs only numpy, scipy, scikit-learn, torch and braindecode.
Designed to run on a Kaggle GPU notebook, but works anywhere.

Protocol (identical split to the local benchmark):
  * every subject's trials are kept in chronological order;
  * the first 80% is the development period, the last 20% the untouched holdout;
  * inside development, the last 12.5% (trials 70-80%) is a chronological
    validation set used for early stopping and per-subject fine-tuning choice;
  * each subject is Euclidean-aligned with a reference computed, without labels,
    from its own development trials only;
  * ONE model is trained on all subjects' training trials; it is then optionally
    fine-tuned per subject. Holdout trials are never used for anything but the
    final score.

Outputs, per run, ``<run>.csv`` + ``<run>.npz`` + ``<run>.json`` in the same
format as ``python -m br_analysis.research`` so they can be imported directly.

Example:
  python deep_pooled.py --data-dir /kaggle/input/br-eeg --out results --suite
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.metrics import roc_auc_score
from torch import nn

HOLDOUT_FRACTION = 0.2
VALIDATION_FRACTION_OF_DEV = 0.125


# ----------------------------------------------------------------------------- data


@dataclass
class Subject:
    id: str
    X: np.ndarray  # (trials, channels, times), aligned float32
    y: np.ndarray
    train: np.ndarray
    val: np.ndarray
    holdout: np.ndarray


def balanced_accuracy(y_true, y_pred):
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]))


def ea_reference(X):
    centered = X - X.mean(axis=-1, keepdims=True)
    cov = np.einsum("nct,ndt->cd", centered, centered) / (len(X) * X.shape[-1])
    cov += 1e-6 * np.trace(cov) / len(cov) * np.eye(len(cov))
    return np.real(fractional_matrix_power(cov, -0.5)).astype(np.float32)


def load_subjects(path: Path, *, sfreq_out: float | None, highpass: float | None, pad_to: int | None, align: bool):
    data = np.load(path, allow_pickle=False)
    X_all = data["X"].astype(np.float32)  # microvolts
    y_all = data["y"].astype(np.int64)
    subjects_all = data["subjects"].astype(str)
    sfreq = float(data["sfreq"])
    if highpass:
        sos = butter(4, highpass, btype="highpass", fs=sfreq, output="sos")
        X_all = sosfiltfilt(sos, X_all, axis=-1).astype(np.float32)
    if sfreq_out and abs(sfreq_out - sfreq) > 1e-6:
        from fractions import Fraction

        ratio = Fraction(sfreq_out / sfreq).limit_denominator(100)
        X_all = resample_poly(X_all, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)
        sfreq = sfreq_out
    if pad_to and X_all.shape[-1] < pad_to:
        X_all = np.pad(X_all, ((0, 0), (0, 0), (0, pad_to - X_all.shape[-1])))
    subjects = []
    for sid in dict.fromkeys(subjects_all.tolist()):
        mask = subjects_all == sid
        X, y = X_all[mask], y_all[mask]
        n = len(y)
        split = max(1, min(int(np.floor((1 - HOLDOUT_FRACTION) * n)), n - 1))
        val_start = int(np.floor(split * (1 - VALIDATION_FRACTION_OF_DEV)))
        train, val, holdout = np.arange(val_start), np.arange(val_start, split), np.arange(split, n)
        if any(len(np.unique(y[idx])) < 2 for idx in (train, val, holdout)):
            continue
        if align:
            X = np.einsum("cd,ndt->nct", ea_reference(X[:split]), X).astype(np.float32)
        else:
            X = X / (X[:split].std() + 1e-6)
        subjects.append(Subject(sid, X, y, train, val, holdout))
    return subjects, sfreq, list(data["channel_names"].astype(str))


# ----------------------------------------------------------------------------- models


class CBraModClassifier(nn.Module):
    """Pretrained CBraMod encoder + mean-pooled linear head (works for any channel count)."""

    def __init__(self, n_chans: int, n_times: int, freeze: bool = False):
        super().__init__()
        from braindecode.models import CBraMod

        self.encoder = CBraMod.from_pretrained("braindecode/cbramod-pretrained", return_encoder_output=True)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        with torch.no_grad():
            features = self.encoder(torch.zeros(2, n_chans, n_times))
        dim = features.shape[-1]
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.3), nn.Linear(dim, 2))

    def forward(self, x):
        # CBraMod expects ~0.01 mV units: our inputs are aligned/whitened, scale is learnt by the head.
        features = self.encoder(x)
        pooled = features.reshape(features.shape[0], -1, features.shape[-1]).mean(dim=1)
        return self.head(pooled)


def build_model(name: str, n_chans: int, n_times: int, sfreq: float) -> nn.Module:
    from braindecode import models as bd

    common = {"n_chans": n_chans, "n_outputs": 2, "n_times": n_times}
    if name == "eegnet":
        return bd.EEGNet(**common, sfreq=sfreq)
    if name == "eegnex":
        return bd.EEGNeX(**common, sfreq=sfreq)
    if name == "shallow":
        return bd.ShallowFBCSPNet(**common, sfreq=sfreq, final_conv_length="auto")
    if name == "eeginception":
        return bd.EEGInceptionERP(**common, sfreq=sfreq)
    if name == "conformer":
        return bd.EEGConformer(**common, sfreq=sfreq)
    if name == "cbramod":
        return CBraModClassifier(n_chans, n_times)
    if name == "cbramod_frozen":
        return CBraModClassifier(n_chans, n_times, freeze=True)
    raise ValueError(f"unknown model {name}")


# ----------------------------------------------------------------------------- training


@dataclass
class Config:
    task: str
    model: str
    data_file: str
    align: bool = True
    augment: bool = True
    sfreq: float | None = None
    highpass: float | None = None
    pad_to: int | None = None
    epochs: int = 60
    patience: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2
    finetune_epochs: int = 10
    finetune_lr: float = 2e-4
    seeds: int = 3
    max_subjects: int | None = None

    @property
    def run_id(self) -> str:
        tags = [self.model, "ea" if self.align else "noea", "aug" if self.augment else "noaug"]
        if self.finetune_epochs:
            tags.append(f"ft{self.finetune_epochs}")
        return "gpu_" + "_".join(tags)


def augment_batch(x: torch.Tensor, sfreq: float) -> torch.Tensor:
    # Small circular time shifts (<= 50 ms), Gaussian noise and random channel dropout.
    max_shift = int(0.05 * sfreq)
    if max_shift:
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        x = torch.roll(x, shifts=shift, dims=-1)
    x = x + 0.1 * x.std() * torch.randn_like(x)
    keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > 0.1).float()
    return x * keep


def predict(model, X, device, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def mean_val_bacc(model, subjects, device):
    return float(np.mean([balanced_accuracy(s.y[s.val], predict(model, s.X[s.val], device).argmax(1)) for s in subjects]))


def train_epochs(model, X, y, cfg, device, sfreq, lr, epochs, eval_fn=None, patience=None):
    counts = np.bincount(y, minlength=2)
    weights = torch.tensor(len(y) / (2.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_score, best_state, best_epoch, stale = -np.inf, copy.deepcopy(model.state_dict()), 0, 0
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y_t))
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            xb, yb = X_t[idx].to(device), y_t[idx].to(device)
            if cfg.augment:
                xb = augment_batch(xb, sfreq)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        scheduler.step()
        if eval_fn is None:
            continue
        score = eval_fn(model)
        if score > best_score:
            best_score, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if patience and stale >= patience:
                break
    if eval_fn is not None:
        model.load_state_dict(best_state)
    return best_score, best_epoch


def run(cfg: Config, data_dir: Path, out_dir: Path, device: str) -> None:
    t0 = time.time()
    subjects, sfreq, _ = load_subjects(
        data_dir / cfg.data_file, sfreq_out=cfg.sfreq, highpass=cfg.highpass, pad_to=cfg.pad_to, align=cfg.align
    )
    if cfg.max_subjects:
        subjects = subjects[: cfg.max_subjects]
    n_chans, n_times = subjects[0].X.shape[1:]
    X_train = np.concatenate([s.X[s.train] for s in subjects])
    y_train = np.concatenate([s.y[s.train] for s in subjects])
    print(f"[{cfg.task}/{cfg.run_id}] {len(subjects)} subjects, train={len(y_train)}, shape=({n_chans},{n_times}), sfreq={sfreq}", flush=True)

    holdout_proba = {s.id: np.zeros((len(s.holdout), 2)) for s in subjects}
    val_scores = {s.id: [] for s in subjects}
    for seed in range(cfg.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(cfg.model, n_chans, n_times, sfreq).to(device)
        score, epoch = train_epochs(
            model, X_train, y_train, cfg, device, sfreq, cfg.lr, cfg.epochs,
            eval_fn=lambda m: mean_val_bacc(m, subjects, device), patience=cfg.patience,
        )
        print(f"  seed {seed}: pooled val bacc={score:.3f} at epoch {epoch} ({time.time() - t0:.0f}s)", flush=True)
        pooled_state = copy.deepcopy(model.state_dict())
        for s in subjects:
            model.load_state_dict(pooled_state)
            if cfg.finetune_epochs:
                s_score, _ = train_epochs(
                    model, s.X[s.train], s.y[s.train], cfg, device, sfreq, cfg.finetune_lr, cfg.finetune_epochs,
                    eval_fn=lambda m, s=s: balanced_accuracy(s.y[s.val], predict(m, s.X[s.val], device).argmax(1)),
                )
                # Keep fine-tuning only if it beats the pooled model on this subject's validation trials.
                tuned_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(pooled_state)
                base_score = balanced_accuracy(s.y[s.val], predict(model, s.X[s.val], device).argmax(1))
                if s_score > base_score:
                    model.load_state_dict(tuned_state)
                val_scores[s.id].append(max(s_score, base_score))
            else:
                val_scores[s.id].append(balanced_accuracy(s.y[s.val], predict(model, s.X[s.val], device).argmax(1)))
            holdout_proba[s.id] += predict(model, s.X[s.holdout], device) / cfg.seeds

    rows, arrays = [], {}
    rng = np.random.default_rng(0)
    for s in subjects:
        y_h, proba = s.y[s.holdout], holdout_proba[s.id]
        pred = proba.argmax(1)
        observed = balanced_accuracy(y_h, pred)
        null = np.array([observed] + [balanced_accuracy(rng.permutation(y_h), pred) for _ in range(1000)])
        rows.append({
            "subject": s.id,
            "dev_bacc": float(np.mean(val_scores[s.id])),
            "holdout_bacc": observed,
            "holdout_auc": float(roc_auc_score(y_h, proba[:, 1])),
            "holdout_perm_p": float(np.mean(null >= observed)),
            "n_dev": int(len(s.train) + len(s.val)),
            "n_holdout": int(len(s.holdout)),
        })
        arrays[f"{s.id}__null"] = null
        arrays[f"{s.id}__holdout_y"] = y_h
        arrays[f"{s.id}__holdout_proba"] = proba
    task_dir = out_dir / cfg.task
    task_dir.mkdir(parents=True, exist_ok=True)
    import csv

    with open(task_dir / f"{cfg.run_id}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(task_dir / f"{cfg.run_id}.npz", **arrays)
    meta = asdict(cfg) | {
        "candidate": cfg.run_id,
        "description": f"GPU pooled {cfg.model} (EA={cfg.align}, augment={cfg.augment}, fine-tune={cfg.finetune_epochs})",
        "n_subjects": len(rows),
        "minutes": (time.time() - t0) / 60,
        "note": "dev_bacc here is the chronological validation score (trials 70-80%), not blocked CV.",
    }
    (task_dir / f"{cfg.run_id}.json").write_text(json.dumps(meta, indent=2))
    mean = np.mean([r["holdout_bacc"] for r in rows])
    print(f"[{cfg.task}/{cfg.run_id}] holdout bacc={mean:.3f} val={np.mean([r['dev_bacc'] for r in rows]):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)


SUITE = [
    # (model, data variant, sfreq, highpass, pad_to, epochs, lr)
    ("eegnet", 128.0, 1.0, None, 60, 1e-3),
    ("eegnex", 128.0, 1.0, None, 60, 1e-3),
    ("eeginception", 128.0, None, None, 60, 1e-3),
    ("shallow", 128.0, 1.0, None, 60, 1e-3),
    ("cbramod", 200.0, 0.3, 400, 20, 5e-5),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--tasks", nargs="+", default=["c_vs_ip", "im_vs_ip"])
    parser.add_argument("--suite", action="store_true", help="run every model in SUITE")
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-subjects", type=int, default=None, help="debug: limit subjects")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)
    entries = SUITE if args.suite else [e for e in SUITE if e[0] == args.model] or [(args.model, 128.0, 1.0, None, 60, 1e-3)]
    for task in args.tasks:
        for model, sfreq, highpass, pad_to, epochs, lr in entries:
            cfg = Config(
                task=task, model=model, data_file=f"{task}.npz", align=not args.no_align,
                augment=not args.no_augment, sfreq=sfreq, highpass=highpass, pad_to=pad_to,
                epochs=args.epochs or epochs, lr=lr, seeds=args.seeds,
                finetune_epochs=args.finetune_epochs, max_subjects=args.max_subjects,
            )
            try:
                run(cfg, args.data_dir, args.out, device)
            except Exception as exc:  # keep the suite going; the log shows what failed
                print(f"[{task}/{cfg.run_id}] FAILED: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
