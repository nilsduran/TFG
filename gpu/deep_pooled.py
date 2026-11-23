"""Pooled deep-learning benchmark and screening for the binocular-rivalry EEG data (GPU).

Self-contained: needs only numpy, scipy, scikit-learn, torch and braindecode.
Designed to run on a Kaggle GPU notebook, but works anywhere.

Protocol (identical split to the local benchmark):
  * every subject's trials are kept in chronological order;
  * the first 80% is the development period, the last 20% the untouched holdout;
  * inside development, the last 12.5% (trials 70-80%; 25% while screening) is a chronological
    validation set used for early stopping, per-subject fine-tuning choice and
    (in ``--screen`` mode) for choosing between configurations;
  * each subject is Euclidean-aligned with a reference computed, without labels,
    from its own development trials only;
  * ONE model is trained on all subjects' training trials; it is then optionally
    fine-tuned per subject. Holdout trials are never used for anything but the
    final score.

Two modes:

``--suite``   the fixed model list in ``SUITE`` (3 seeds, fine-tuning, holdout).

``--screen``  "throw spaghetti at the wall": successive halving over ~40
              configurations (architectures, filtering, time windows, channels,
              alignment, augmentation, optimisation, auxiliary third class).
              Rungs 1-2 look at validation trials only (early stopping on the
              even validation trials, scoring on the odd ones), so the holdout
              stays untouched until the few survivors of rung 3 get the full
              protocol. A leaderboard is written after every run.

Outputs, per final run, ``<run>.csv`` + ``<run>.npz`` + ``<run>.json`` in the same
format as ``python -m br_analysis.research`` so they can be imported directly.

Examples:
  python deep_pooled.py --data-dir /kaggle/input/br-eeg --out results --suite
  python deep_pooled.py --data-dir /kaggle/input/br-eeg --out results --screen --tasks im_vs_ip
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.metrics import roc_auc_score
from torch import nn

HOLDOUT_FRACTION = 0.2
VALIDATION_FRACTION_OF_DEV = 0.125
PARTNER_TASK = {"c_vs_ip": "im_vs_ip", "im_vs_ip": "c_vs_ip"}
CHANNEL_SETS = {"posterior": r"^(P|PO|O|Iz)"}


# ----------------------------------------------------------------------------- data


@dataclass
class Subject:
    id: str
    X: np.ndarray  # (trials, channels, times), aligned float32
    y: np.ndarray
    train: np.ndarray
    val: np.ndarray
    holdout: np.ndarray
    X_aux: np.ndarray | None = None  # extra training trials of the auxiliary class (multi-task)
    y_aux: np.ndarray | None = None


def balanced_accuracy(y_true, y_pred):
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]))


def ea_reference(X):
    centered = X - X.mean(axis=-1, keepdims=True)
    cov = np.einsum("nct,ndt->cd", centered, centered) / (len(X) * X.shape[-1])
    cov += 1e-6 * np.trace(cov) / len(cov) * np.eye(len(cov))
    return np.real(fractional_matrix_power(cov, -0.5)).astype(np.float32)


_RAW: dict[Path, dict] = {}


def _raw(path: Path) -> dict:
    if path not in _RAW:
        data = np.load(path, allow_pickle=False)
        _RAW[path] = {
            "X": data["X"].astype(np.float32),  # microvolts, stimulus window starting at 0 s
            "y": data["y"].astype(np.int64),
            "subjects": data["subjects"].astype(str),
            "trials": data["trial_indices"].astype(np.int64),
            "sfreq": float(data["sfreq"]),
            "channels": list(data["channel_names"].astype(str)),
            "classes": list(data["class_ids"].astype(str)),
        }
    return _RAW[path]


def preprocess(X, sfreq, *, highpass, lowpass, sfreq_out, window, channel_idx, pad_to):
    if channel_idx is not None:
        X = X[:, channel_idx]
    if highpass:
        X = sosfiltfilt(butter(4, highpass, btype="highpass", fs=sfreq, output="sos"), X, axis=-1)
    if lowpass:
        X = sosfiltfilt(butter(4, lowpass, btype="lowpass", fs=sfreq, output="sos"), X, axis=-1)
    if sfreq_out and abs(sfreq_out - sfreq) > 1e-6:
        ratio = Fraction(sfreq_out / sfreq).limit_denominator(100)
        X = resample_poly(X, ratio.numerator, ratio.denominator, axis=-1)
        sfreq = sfreq_out
    if window:
        start, stop = int(round(window[0] * sfreq)), int(round(window[1] * sfreq))
        X = X[..., start:stop]
    if pad_to and X.shape[-1] < pad_to:
        X = np.pad(X, ((0, 0), (0, 0), (0, pad_to - X.shape[-1])))
    return np.ascontiguousarray(X, dtype=np.float32), sfreq


def load_subjects(
    path: Path,
    *,
    sfreq_out: float | None,
    highpass: float | None,
    pad_to: int | None = None,
    align: bool = True,
    lowpass: float | None = None,
    window: tuple[float, float] | None = None,
    channels: str | None = None,
    aux_path: Path | None = None,
    val_fraction: float = VALIDATION_FRACTION_OF_DEV,
):
    raw = _raw(path)
    channel_idx = None
    if channels:
        pattern = re.compile(CHANNEL_SETS.get(channels, channels))
        channel_idx = [i for i, name in enumerate(raw["channels"]) if pattern.match(name)]
    prep = dict(highpass=highpass, lowpass=lowpass, sfreq_out=sfreq_out, window=window, channel_idx=channel_idx, pad_to=pad_to)
    X_all, sfreq = preprocess(raw["X"], raw["sfreq"], **prep)

    aux = None
    if aux_path is not None:
        # The partner task contributes the one class this task lacks (C for IM-vs-IP, IM for C-vs-IP)
        # as a third training class. Only partner trials that happened before this subject's
        # validation period are used, so no validation or holdout period leaks into training.
        other = _raw(aux_path)
        aux_class = next(i for i, c in enumerate(other["classes"]) if c not in raw["classes"])
        keep = other["y"] == aux_class
        X_aux_all, _ = preprocess(other["X"][keep], other["sfreq"], **prep)
        aux = (X_aux_all, other["subjects"][keep], other["trials"][keep])

    subjects = []
    for sid in dict.fromkeys(raw["subjects"].tolist()):
        mask = raw["subjects"] == sid
        X, y, trials = X_all[mask], raw["y"][mask], raw["trials"][mask]
        n = len(y)
        split = max(1, min(int(np.floor((1 - HOLDOUT_FRACTION) * n)), n - 1))
        val_start = int(np.floor(split * (1 - val_fraction)))
        train, val, holdout = np.arange(val_start), np.arange(val_start, split), np.arange(split, n)
        if any(len(np.unique(y[idx])) < 2 for idx in (train, val, holdout)):
            continue
        transform = ea_reference(X[:split]) if align else None
        scale = 1.0 if align else 1.0 / (X[:split].std() + 1e-6)

        def apply(A):
            return (np.einsum("cd,ndt->nct", transform, A) if align else A * scale).astype(np.float32)

        subject = Subject(sid, apply(X), y, train, val, holdout)
        if aux is not None:
            X_aux, aux_subjects, aux_trials = aux
            sel = (aux_subjects == sid) & (aux_trials < trials[val_start])
            if sel.any():
                subject.X_aux = apply(X_aux[sel])
                subject.y_aux = np.full(int(sel.sum()), 2, dtype=np.int64)
        subjects.append(subject)
    return subjects, sfreq, raw["channels"] if channel_idx is None else [raw["channels"][i] for i in channel_idx]


# ----------------------------------------------------------------------------- models


class CBraModClassifier(nn.Module):
    """Pretrained CBraMod encoder + mean-pooled linear head (works for any channel count)."""

    def __init__(self, n_chans: int, n_times: int, n_outputs: int = 2, freeze: bool = False):
        super().__init__()
        from braindecode.models import CBraMod

        self.encoder = CBraMod.from_pretrained("braindecode/cbramod-pretrained", return_encoder_output=True)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        with torch.no_grad():
            features = self.encoder(torch.zeros(2, n_chans, n_times))
        dim = features.shape[-1]
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.3), nn.Linear(dim, n_outputs))

    def forward(self, x):
        features = self.encoder(x)
        pooled = features.reshape(features.shape[0], -1, features.shape[-1]).mean(dim=1)
        return self.head(pooled)


BRAINDECODE_MODELS = {
    "eegnet": "EEGNet",
    "eegnex": "EEGNeX",
    "eeginception": "EEGInceptionERP",
    "conformer": "EEGConformer",
    "atcnet": "ATCNet",
    "deep4": "Deep4Net",
    "simpleconv": "EEGSimpleConv",
    "tcnet": "EEGTCNet",
    "itnet": "EEGITNet",
    "tidnet": "TIDNet",
    "sccnet": "SCCNet",
    "ctnet": "CTNet",
    "msvtnet": "MSVTNet",
    "ifnet": "IFNet",
    "fbcnet": "FBCNet",
    "tsception": "TSception",
}


def build_model(name: str, n_chans: int, n_times: int, sfreq: float, n_outputs: int = 2) -> nn.Module:
    from braindecode import models as bd

    common = {"n_chans": n_chans, "n_outputs": n_outputs, "n_times": n_times, "sfreq": sfreq}
    if name == "shallow":
        return bd.ShallowFBCSPNet(**common, final_conv_length="auto")
    if name in BRAINDECODE_MODELS:
        return getattr(bd, BRAINDECODE_MODELS[name])(**common)
    if name == "cbramod":
        return CBraModClassifier(n_chans, n_times, n_outputs)
    if name == "cbramod_frozen":
        return CBraModClassifier(n_chans, n_times, n_outputs, freeze=True)
    raise ValueError(f"unknown model {name}")


# ----------------------------------------------------------------------------- training


@dataclass
class Config:
    task: str
    model: str
    data_file: str
    align: bool = True
    augment: bool = True
    aug_strength: float = 1.0
    sfreq: float | None = None
    highpass: float | None = None
    lowpass: float | None = None
    window: tuple[float, float] | None = None
    channels: str | None = None
    pad_to: int | None = None
    aux3: bool = False
    val_fraction: float = VALIDATION_FRACTION_OF_DEV
    epochs: int = 60
    patience: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2
    mixup: float = 0.0
    label_smoothing: float = 0.0
    finetune_epochs: int = 10
    finetune_lr: float = 2e-4
    seeds: int = 3
    max_subjects: int | None = None
    name: str | None = None

    @property
    def run_id(self) -> str:
        if self.name:
            return self.name
        tags = [self.model, "ea" if self.align else "noea", "aug" if self.augment else "noaug"]
        if self.finetune_epochs:
            tags.append(f"ft{self.finetune_epochs}")
        return "gpu_" + "_".join(tags)

    @property
    def n_outputs(self) -> int:
        return 3 if self.aux3 else 2

    def data_key(self) -> tuple:
        return (self.data_file, self.sfreq, self.highpass, self.lowpass, self.window, self.channels, self.pad_to, self.align, self.aux3, self.val_fraction)


def augment_batch(x: torch.Tensor, sfreq: float, strength: float = 1.0) -> torch.Tensor:
    # Small circular time shifts (<= 50 ms), Gaussian noise and random channel dropout.
    max_shift = int(0.05 * strength * sfreq)
    if max_shift:
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        x = torch.roll(x, shifts=shift, dims=-1)
    x = x + 0.1 * strength * x.std() * torch.randn_like(x)
    keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > 0.1 * strength).float()
    return x * keep


def predict(model, X, device, batch_size=512):
    """Probabilities of the two task classes (an auxiliary third class is renormalised away)."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            out.append(torch.softmax(model(xb), dim=1)[:, :2].cpu().numpy())
    proba = np.concatenate(out)
    return proba / proba.sum(1, keepdims=True)


def subject_scores(model, subjects, device, part: str = "val", parity: int | None = None):
    """Mean per-subject balanced accuracy and AUC on validation trials (optionally even/odd half)."""
    baccs, aucs = [], []
    for s in subjects:
        idx = getattr(s, part)
        if parity is not None:
            idx = idx[parity::2]
        y = s.y[idx]
        if len(np.unique(y)) < 2:
            continue
        proba = predict(model, s.X[idx], device)
        baccs.append(balanced_accuracy(y, proba.argmax(1)))
        aucs.append(roc_auc_score(y, proba[:, 1]))
    return float(np.mean(baccs)), float(np.mean(aucs))


def train_epochs(model, X, y, cfg, device, sfreq, lr, epochs, eval_fn=None, patience=None):
    n_classes = max(cfg.n_outputs, int(y.max()) + 1)
    counts = np.bincount(y, minlength=n_classes)
    weights = torch.tensor(len(y) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
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
            if len(idx) < 2:
                continue
            xb, yb = X_t[idx].to(device), y_t[idx].to(device)
            if cfg.augment:
                xb = augment_batch(xb, sfreq, cfg.aug_strength)
            optimizer.zero_grad()
            if cfg.mixup:
                lam = float(np.random.beta(cfg.mixup, cfg.mixup))
                perm = torch.randperm(len(yb), device=device)
                out = model(lam * xb + (1 - lam) * xb[perm])
                loss = lam * criterion(out, yb) + (1 - lam) * criterion(out, yb[perm])
            else:
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


_SUBJECT_CACHE: dict[tuple, tuple] = {}


def get_subjects(cfg: Config, data_dir: Path):
    key = cfg.data_key()
    if key not in _SUBJECT_CACHE:
        if len(_SUBJECT_CACHE) >= 2:  # keep memory bounded
            _SUBJECT_CACHE.pop(next(iter(_SUBJECT_CACHE)))
        aux_path = data_dir / f"{PARTNER_TASK[cfg.task]}.npz" if cfg.aux3 else None
        _SUBJECT_CACHE[key] = load_subjects(
            data_dir / cfg.data_file, sfreq_out=cfg.sfreq, highpass=cfg.highpass, pad_to=cfg.pad_to,
            align=cfg.align, lowpass=cfg.lowpass, window=cfg.window, channels=cfg.channels, aux_path=aux_path,
            val_fraction=cfg.val_fraction,
        )
    subjects, sfreq, channels = _SUBJECT_CACHE[key]
    if cfg.max_subjects:
        subjects = subjects[: cfg.max_subjects]
    return subjects, sfreq, channels


def training_arrays(subjects):
    X = [s.X[s.train] for s in subjects] + [s.X_aux for s in subjects if s.X_aux is not None]
    y = [s.y[s.train] for s in subjects] + [s.y_aux for s in subjects if s.y_aux is not None]
    return np.concatenate(X), np.concatenate(y)


def screen_once(cfg: Config, data_dir: Path, device: str, seed: int) -> dict:
    """One pooled training run scored on validation trials only (no fine-tuning, no holdout).

    Early stopping uses the even validation trials of each subject and the reported
    score uses the odd ones, so the selection score is not inflated by early stopping.
    """
    t0 = time.time()
    subjects, sfreq, _ = get_subjects(cfg, data_dir)
    X_train, y_train = training_arrays(subjects)
    n_chans, n_times = subjects[0].X.shape[1:]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(cfg.model, n_chans, n_times, sfreq, cfg.n_outputs).to(device)
    _, epoch = train_epochs(
        model, X_train, y_train, cfg, device, sfreq, cfg.lr, cfg.epochs,
        eval_fn=lambda m: subject_scores(m, subjects, device, parity=0)[1], patience=cfg.patience,
    )
    bacc, auc = subject_scores(model, subjects, device, parity=1)
    return {"val_bacc": bacc, "val_auc": auc, "epoch": epoch, "minutes": (time.time() - t0) / 60, "shape": f"{n_chans}x{n_times}"}


def run(cfg: Config, data_dir: Path, out_dir: Path, device: str) -> dict[str, np.ndarray]:
    """Full protocol: pooled training, per-subject fine-tuning, holdout scoring. Returns holdout probabilities."""
    t0 = time.time()
    subjects, sfreq, _ = get_subjects(cfg, data_dir)
    n_chans, n_times = subjects[0].X.shape[1:]
    X_train, y_train = training_arrays(subjects)
    print(f"[{cfg.task}/{cfg.run_id}] {len(subjects)} subjects, train={len(y_train)}, shape=({n_chans},{n_times}), sfreq={sfreq}", flush=True)

    def val_bacc(m, s):
        return balanced_accuracy(s.y[s.val], predict(m, s.X[s.val], device).argmax(1))

    holdout_proba = {s.id: np.zeros((len(s.holdout), 2)) for s in subjects}
    val_scores = {s.id: [] for s in subjects}
    for seed in range(cfg.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(cfg.model, n_chans, n_times, sfreq, cfg.n_outputs).to(device)
        score, epoch = train_epochs(
            model, X_train, y_train, cfg, device, sfreq, cfg.lr, cfg.epochs,
            eval_fn=lambda m: subject_scores(m, subjects, device)[0], patience=cfg.patience,
        )
        print(f"  seed {seed}: pooled val bacc={score:.3f} at epoch {epoch} ({time.time() - t0:.0f}s)", flush=True)
        pooled_state = copy.deepcopy(model.state_dict())
        for s in subjects:
            model.load_state_dict(pooled_state)
            if cfg.finetune_epochs:
                s_score, _ = train_epochs(
                    model, s.X[s.train], s.y[s.train], cfg, device, sfreq, cfg.finetune_lr, cfg.finetune_epochs,
                    eval_fn=lambda m, s=s: val_bacc(m, s),
                )
                # Keep fine-tuning only if it beats the pooled model on this subject's validation trials.
                tuned_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(pooled_state)
                base_score = val_bacc(model, s)
                if s_score > base_score:
                    model.load_state_dict(tuned_state)
                val_scores[s.id].append(max(s_score, base_score))
            else:
                val_scores[s.id].append(val_bacc(model, s))
            holdout_proba[s.id] += predict(model, s.X[s.holdout], device) / cfg.seeds

    write_results(cfg.task, cfg.run_id, subjects, holdout_proba, val_scores, out_dir,
                  asdict(cfg) | {"description": f"GPU pooled {cfg.model} ({cfg.run_id})"}, t0)
    return holdout_proba


def write_results(task, run_id, subjects, holdout_proba, val_scores, out_dir, meta, t0):
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
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / f"{run_id}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(task_dir / f"{run_id}.npz", **arrays)
    meta = meta | {
        "candidate": run_id,
        "n_subjects": len(rows),
        "minutes": (time.time() - t0) / 60,
        "note": "dev_bacc here is the chronological validation score (trials 70-80%), not blocked CV.",
    }
    (task_dir / f"{run_id}.json").write_text(json.dumps(meta, indent=2, default=str))
    mean = np.mean([r["holdout_bacc"] for r in rows])
    print(f"[{task}/{run_id}] holdout bacc={mean:.3f} val={np.mean([r['dev_bacc'] for r in rows]):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ----------------------------------------------------------------------------- fixed suite


SUITE = [
    # (model, sfreq, highpass, pad_to, epochs, lr)
    ("eegnet", 128.0, 1.0, None, 60, 1e-3),
    ("eegnex", 128.0, 1.0, None, 60, 1e-3),
    ("eeginception", 128.0, None, None, 60, 1e-3),
    ("shallow", 128.0, 1.0, None, 60, 1e-3),
    ("cbramod", 200.0, 0.3, 400, 20, 5e-5),
]


# ----------------------------------------------------------------------------- screening

# Screening validates on the last 25% of the development period (trials 60-80%) instead of 12.5%:
# twice the validation trials makes the ranking less noisy. Finalists go back to the standard split.
SCREEN_BASE = dict(model="eegnet", sfreq=128.0, highpass=1.0, epochs=40, patience=8, val_fraction=0.25)

# (name, group, overrides). Groups other than "model" are pipeline tweaks that can be combined.
SCREEN: list[tuple[str, str, dict]] = [
    ("base_eegnet", "base", {}),
    # architectures (all with the base pipeline)
    *[(f"m_{m}", "model", {"model": m}) for m in (
        "eeginception", "eegnex", "shallow", "atcnet", "conformer", "deep4", "simpleconv", "tcnet",
        "itnet", "tidnet", "sccnet", "ctnet", "msvtnet", "ifnet", "fbcnet", "tsception",
    )],
    # filtering and resolution
    ("hp_none", "highpass", {"highpass": None}),
    ("hp_0.5", "highpass", {"highpass": 0.5}),
    ("hp_2", "highpass", {"highpass": 2.0}),
    ("lp_15", "lowpass", {"lowpass": 15.0}),
    ("sf_64", "sfreq", {"sfreq": 64.0}),
    ("sf_250", "sfreq", {"sfreq": 250.0}),
    # time windows inside the stimulus (rivalry is resolved early; the C-vs-IP alpha effect is late)
    ("win_0-0.5", "window", {"window": (0.0, 0.5)}),
    ("win_0.15-0.65", "window", {"window": (0.15, 0.65)}),
    ("win_0-1.0", "window", {"window": (0.0, 1.0)}),
    ("win_0.2-1.0", "window", {"window": (0.2, 1.0)}),
    ("win_0.5-1.5", "window", {"window": (0.5, 1.5)}),
    # spatial / alignment
    ("ch_posterior", "channels", {"channels": "posterior"}),
    ("no_ea", "align", {"align": False}),
    # augmentation and regularisation
    ("aug_none", "augment", {"augment": False}),
    ("aug_strong", "augment", {"aug_strength": 2.0}),
    ("mixup", "mixup", {"mixup": 0.4}),
    ("label_smooth", "label_smoothing", {"label_smoothing": 0.1}),
    # optimisation
    ("lr_3e-4", "lr", {"lr": 3e-4}),
    ("bs_64", "batch_size", {"batch_size": 64}),
    ("wd_0.1", "weight_decay", {"weight_decay": 0.1}),
    # multi-task: the partner task's extra class (C or IM) as a third training class
    ("aux3", "aux3", {"aux3": True}),
]


class Leaderboard:
    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict) -> None:
        self.rows.append(row)
        with open(self.path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def mean_auc(self, name: str) -> float:
        values = [r["val_auc"] for r in self.rows if r["config"] == name and r["status"] == "ok"]
        return float(np.mean(values)) if values else -np.inf

    def ranking(self, names) -> list[str]:
        return sorted(names, key=self.mean_auc, reverse=True)


def screen(task: str, data_dir: Path, out_dir: Path, device: str, budget_hours: float, keep2: int, keep3: int, max_subjects=None):
    t_start = time.time()

    def hours_left():
        return budget_hours - (time.time() - t_start) / 3600

    board = Leaderboard(out_dir / task / "_screen_leaderboard.csv")
    configs: dict[str, tuple[str, dict]] = {name: (group, ov) for name, group, ov in SCREEN}

    def make(name: str, **extra) -> Config:
        _, overrides = configs[name]
        return Config(task=task, data_file=f"{task}.npz", max_subjects=max_subjects, name=f"scr_{name}",
                      **(SCREEN_BASE | overrides | extra))

    def evaluate(name: str, rung: int, seed: int) -> None:
        cfg = make(name)
        try:
            res = screen_once(cfg, data_dir, device, seed)
            status = "ok"
        except Exception as exc:  # a broken config must not stop the screen
            res, status = {"val_bacc": np.nan, "val_auc": np.nan, "epoch": 0, "minutes": 0.0, "shape": ""}, f"FAILED {type(exc).__name__}: {str(exc)[:80]}"
        board.add({"task": task, "config": name, "group": configs[name][0], "rung": rung, "seed": seed, **res, "status": status})
        print(f"[{task}] rung{rung} {name:<18} seed{seed} val_auc={res['val_auc']:.3f} bacc={res['val_bacc']:.3f} "
              f"ep={res['epoch']} {res['minutes']:.1f}min {status if status != 'ok' else ''} | {hours_left():.1f}h left", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- rung 1: every configuration, one seed
    for name in list(configs):
        if hours_left() < 2.0:
            print(f"[{task}] budget: stopping rung 1 early", flush=True)
            break
        evaluate(name, 1, 0)

    # ---- combinations: best architectures + every tweak group that beat the base on its own
    base_auc = board.mean_auc("base_eegnet")
    tweaks: dict = {}
    for group in dict.fromkeys(g for g, _ in configs.values()):
        if group in ("base", "model"):
            continue
        members = [n for n, (g, _) in configs.items() if g == group]
        best = board.ranking(members)[0]
        if board.mean_auc(best) > base_auc + 0.01:
            tweaks |= configs[best][1]
    print(f"[{task}] tweaks that beat the base on their own: {tweaks}", flush=True)
    models = [configs[n][1].get("model", "eegnet") for n in board.ranking([n for n, (g, _) in configs.items() if g in ("model", "base")])][:2]
    for model in models:
        if tweaks:
            configs[f"combo_{model}"] = ("combo", tweaks | {"model": model})
            if hours_left() > 2.0:
                evaluate(f"combo_{model}", 1, 0)

    # ---- rung 2: survivors get two more seeds
    survivors = board.ranking(list(configs))[:keep2]
    print(f"[{task}] rung 2 survivors: {survivors}", flush=True)
    for name in survivors:
        for seed in (1, 2):
            if hours_left() < 1.5:
                break
            evaluate(name, 2, seed)

    # ---- rung 3: full protocol (3 seeds, fine-tuning, holdout) for the finalists + their ensemble
    finalists = board.ranking(survivors)[:keep3]
    print(f"[{task}] finalists: {[(n, round(board.mean_auc(n), 3)) for n in finalists]}", flush=True)
    probas = []
    for name in finalists:
        if hours_left() < 0.75:
            print(f"[{task}] budget: skipping finalist {name}", flush=True)
            continue
        cfg = make(name, epochs=60, patience=10, val_fraction=VALIDATION_FRACTION_OF_DEV)
        try:
            probas.append((cfg, run(cfg, data_dir, out_dir, device)))
        except Exception as exc:
            print(f"[{task}/{cfg.run_id}] FAILED: {type(exc).__name__}: {exc}", flush=True)
    if len(probas) >= 2:
        subjects, _, _ = get_subjects(probas[0][0], data_dir)
        common = set.intersection(*(set(p) for _, p in probas))
        subjects = [s for s in subjects if s.id in common]
        ens = {s.id: np.mean([p[s.id] for _, p in probas], axis=0) for s in subjects}
        write_results(task, "scr_ens_finalists", subjects, ens, {s.id: [np.nan] for s in subjects}, out_dir,
                      {"description": "mean probability of the screening finalists (chosen on validation)",
                       "members": [c.run_id for c, _ in probas]}, time.time())
    print(f"[{task}] screening finished in {(time.time() - t_start) / 3600:.1f} h", flush=True)


# ----------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--tasks", nargs="+", default=["c_vs_ip", "im_vs_ip"])
    parser.add_argument("--suite", action="store_true", help="run every model in SUITE")
    parser.add_argument("--screen", action="store_true", help="successive-halving screen over SCREEN")
    parser.add_argument("--budget-hours", type=float, default=10.0, help="screen: wall-clock budget per process")
    parser.add_argument("--keep2", type=int, default=6, help="screen: configs promoted to rung 2")
    parser.add_argument("--keep3", type=int, default=3, help="screen: finalists given the full protocol")
    parser.add_argument("--only", nargs="*", default=None, help="screen: restrict to these config names (debug)")
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

    if args.screen:
        global SCREEN
        if args.only:
            SCREEN = [entry for entry in SCREEN if entry[0] in set(args.only) | {"base_eegnet"}]
        if args.epochs:
            SCREEN_BASE.update(epochs=args.epochs, patience=max(2, args.epochs // 5))
        for task in args.tasks:
            screen(task, args.data_dir, args.out, device, args.budget_hours, args.keep2, args.keep3, args.max_subjects)
        return

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
