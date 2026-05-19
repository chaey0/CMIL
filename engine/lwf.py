"""Learning without Forgetting (LwF) baseline.

Li & Hoiem, "Learning without Forgetting", TPAMI 2017.

Before each incremental session, a frozen copy of the model (teacher) is saved.
During training, a knowledge distillation loss on the OLD classes keeps the
student's output distribution close to the teacher's, preventing forgetting.

Key hyper-parameters (set in train config):
    lwf_temperature: float = 2.0   # KD temperature
    lwf_alpha:       float = 1.0   # KD loss weight
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .continual import (
    RunningStats,
    _safe_item,
    infer_task_keys,
    _forward_global_from_z,
    _forward_task_from_z,
    _mask_logits_by_task_keys,
    _masked_accuracy,
    _use_merged_organ_task_adaptors,
)
from .losses import bank_contrastive_loss


# ------------------------------------------------------------------ helpers ---

def snapshot_teacher(model):
    """Create a frozen copy of the model for distillation."""
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _kd_loss(student_logits, teacher_logits, temperature: float = 2.0):
    """Knowledge distillation loss (KL divergence with temperature scaling).

    KD = T^2 · KL( softmax(teacher/T)  ||  softmax(student/T) )
    """
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    return (temperature ** 2) * F.kl_div(
        log_p_student, p_teacher, reduction="batchmean",
    )


def _build_old_to_new_map(old_labels, new_model):
    """Map old model's active fine label indices to new model's local indices.

    Returns:
        idx_map: LongTensor [n_old] or None if mapping failed.
    """
    mapping = []
    for lab in old_labels:
        if lab in new_model.active_fine_to_local:
            mapping.append(new_model.active_fine_to_local[lab])
        else:
            return None  # label disappeared (shouldn't happen in CL)
    return torch.tensor(mapping, dtype=torch.long)


# ------------------------------------------------------------- train loop ---

def train_session_lwf(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    session_idx: int,
    cfg: dict,
    teacher=None,
    lambda_fine: float = 1.0,
    debug_print_freq: int = 20,
    **kwargs,
):
    """LwF training loop for one epoch.

    Args:
        model:      current (student) model — trainable.
        teacher:    frozen snapshot from previous session (None for session 0).
        lambda_fine: default CE weight.
    """
    model.train()
    train_cfg = cfg.get("train", {})
    stats = RunningStats()

    T = float(train_cfg.get("lwf_temperature", 2.0))
    alpha = float(train_cfg.get("lwf_alpha", 1.0))
    lam_task = float(train_cfg.get("lambda_cls_task", lambda_fine))
    lam_global = float(train_cfg.get("lambda_cls_global", 0.3))
    lam_align = float(train_cfg.get("lambda_align", 0.0))

    opened_task_keys = model.get_opened_task_keys()

    # Mapping from teacher's label space → student's label space
    old_to_new = None
    if teacher is not None:
        old_to_new = _build_old_to_new_map(teacher.active_fine_labels, model)
        if old_to_new is not None:
            old_to_new = old_to_new.to(device)

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"[LwF] S{session_idx} E{epoch + 1}",
        leave=False,
    )

    for step, batch in pbar:
        feats = [f.to(device) for f in batch["feats_list"]]
        fine_global = batch["fine_ids"].to(device)
        fine_local = model.global_fine_to_local(fine_global)
        task_keys, _, _ = infer_task_keys(model, batch, fine_global)

        optimizer.zero_grad(set_to_none=True)

        z_shared, _ = model.encode_shared(feats)

        # --- student forward ---
        out_global = _forward_global_from_z(model, z_shared, opened_task_keys, cfg)
        global_logits = out_global["fine_logits"]

        if _use_merged_organ_task_adaptors(model):
            task_logits = _mask_logits_by_task_keys(
                model, global_logits, task_keys, targets_local=fine_local,
            )
            out_task = dict(out_global)
            out_task["fine_logits"] = task_logits
        else:
            out_task = _forward_task_from_z(model, z_shared, task_keys, cfg)
            task_logits = out_task["fine_logits"]

        # Classification losses
        loss_task = F.cross_entropy(task_logits, fine_local)
        loss_global = (
            F.cross_entropy(global_logits, fine_local)
            if lam_global > 0 and len(opened_task_keys) > 1
            else loss_task.new_tensor(0.0)
        )

        # --- KD loss on OLD class logits ---
        loss_kd = loss_task.new_tensor(0.0)
        if teacher is not None and old_to_new is not None and session_idx > 0:
            with torch.no_grad():
                old_z, _ = teacher.encode_shared(feats)
                teacher_keys = teacher.get_opened_task_keys()
                old_out = _forward_global_from_z(teacher, old_z, teacher_keys, cfg)
                old_logits = old_out["fine_logits"]  # [B, n_old]

            # Student's logits for the SAME old classes
            new_logits_old = global_logits[:, old_to_new]  # [B, n_old]
            loss_kd = _kd_loss(new_logits_old, old_logits, temperature=T)

        # --- Text alignment loss ---
        if lam_align > 0:
            fine_bank = model.get_active_fine_bank(device)
            loss_align = bank_contrastive_loss(
                out_task["h_f"], fine_bank, fine_local,
                tau=train_cfg.get("align_temperature", 0.07),
            )
        else:
            loss_align = loss_task.new_tensor(0.0)

        loss = lam_task * loss_task + lam_global * loss_global + alpha * loss_kd + lam_align * loss_align

        loss.backward()
        optimizer.step()

        task_acc = _masked_accuracy(task_logits, fine_local)
        global_acc = _masked_accuracy(global_logits, fine_local)

        stats.update(
            loss_total=loss,
            loss_task=loss_task,
            loss_global=loss_global,
            loss_kd=loss_kd,
            loss_align=loss_align,
            task_acc=task_acc,
            global_acc=global_acc,
        )

        if (step + 1) % max(1, debug_print_freq) == 0:
            pbar.set_postfix(
                {"L": f"{loss.item():.3f}", "KD": f"{loss_kd.item():.3f}", "A": f"{loss_align.item():.3f}"}
            )

    avg = stats.as_dict()
    print(
        f"[LwF S{session_idx} E{epoch + 1:02d}] "
        f"L={avg.get('loss_total', 0):.4f} "
        f"TaskCE={avg.get('loss_task', 0):.4f} "
        f"GlobalCE={avg.get('loss_global', 0):.4f} "
        f"KD={avg.get('loss_kd', 0):.4f} "
        f"Align={avg.get('loss_align', 0):.4f} "
        f"TaskAcc={avg.get('task_acc', 0) * 100:.2f}% "
        f"GlobalAcc={avg.get('global_acc', 0) * 100:.2f}%"
    )
    return avg
