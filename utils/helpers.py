import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from engine.continual import _fine_to_coarse, register_sessions_until


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def create_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters. Check training-stage configuration.")
    return torch.optim.AdamW(
        params,
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )


def create_scheduler(optimizer, cfg, total_epochs):
    name = cfg["train"].get("scheduler", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=cfg["train"].get("eta_min", 1e-6),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(model, session_idx, save_path):
    ensure_dir(os.path.dirname(save_path))
    torch.save({"model_state": model.state_dict(), "session_idx": session_idx}, save_path)
    print(f"[Saved CKPT] {save_path}")


def prepare_model_structure_for_ckpt_load(model, session_manager, session_idx, device, cfg=None):
    opened_fine = session_manager.opened_fine_labels(session_idx)
    opened_coarse = sorted({_fine_to_coarse(model.taxonomy, f) for f in opened_fine})
    model.activate_nodes(opened_coarse, opened_fine, device)
    model.expand_classifiers(opened_fine, device)
    register_sessions_until(model, session_manager, session_idx, device, cfg or getattr(model, "cfg", {}))
    model.to(device)


def resolve_ckpt_dir(ckpt_path):
    if ckpt_path is None:
        raise ValueError("--ckpt is required in test mode.")
    if os.path.isdir(ckpt_path):
        return ckpt_path
    if os.path.isfile(ckpt_path):
        return os.path.dirname(ckpt_path)
    raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")


def find_session_ckpt_path(ckpt_root, session_idx):
    ckpt_dir = resolve_ckpt_dir(ckpt_root)
    candidates = [
        os.path.join(ckpt_dir, f"ckpt_sess_{session_idx:02d}_last.pth"),
        os.path.join(ckpt_dir, f"ckpt_sess_{session_idx}_last.pth"),
        os.path.join(ckpt_dir, f"ckpt_session_{session_idx:02d}.pth"),
        os.path.join(ckpt_dir, f"ckpt_session_{session_idx}.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def save_confusion_matrix(y_true, y_pred, labels, out_path, title, normalize=None):
    if len(labels) == 0:
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))), normalize=normalize)
    fig_w = max(8, len(labels) * 0.9)
    fig_h = max(6, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fmt = ".2f" if normalize is not None else "d"
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax,
        xticks_rotation=45,
        values_format=fmt,
        colorbar=False,
    )
    ax.set_title(title)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    np.save(out_path.replace(".png", ".npy"), cm)


def save_eval_confusion_matrices(details, coarse_labels, fine_labels, out_dir, prefix):
    ensure_dir(out_dir)
    fine_true = details["fine_true"].detach().cpu().numpy()
    fine_pred = details["fine_pred"].detach().cpu().numpy()
    save_confusion_matrix(
        fine_true,
        fine_pred,
        fine_labels,
        os.path.join(out_dir, f"{prefix}_subtype_cm_count.png"),
        f"{prefix} Subtype Confusion Matrix (Count)",
        normalize=None,
    )
    save_confusion_matrix(
        fine_true,
        fine_pred,
        fine_labels,
        os.path.join(out_dir, f"{prefix}_subtype_cm_norm.png"),
        f"{prefix} Subtype Confusion Matrix (Row-Normalized)",
        normalize="true",
    )
