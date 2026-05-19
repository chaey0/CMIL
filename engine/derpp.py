"""Dark Experience Replay++ (DER++) baseline.

Buzzega et al., "Dark Experience for General Continual Learning:
a Strong, Simple Baseline", NeurIPS 2020.

Combines three loss terms:
  1. CE on current batch (standard classification)
  2. CE on replay samples drawn from a flat reservoir buffer
  3. MSE between current logits and stored logits (dark experience)

Buffer design (faithful to original paper + ConSlide):
  - Single flat list, total capacity = derpp_buffer_size
  - Reservoir sampling (each seen sample has equal probability of being kept)
  - Each entry stores the full patch feature tensor [N_patches, D] on CPU
  - Replay: sample a random mini-batch of derpp_replay_batch_size from the flat buffer
  - No per-class balancing

Key hyper-parameters (set in train config):
    derpp_alpha:             float = 0.2   # weight for replay CE loss
    derpp_beta:              float = 0.2   # weight for dark-experience MSE loss
    derpp_buffer_size:       int = 1000    # total flat buffer capacity (WSI 기준)
    derpp_replay_batch_size: int = 32      # replay batch size
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .continual import (
    RunningStats,
    infer_task_keys,
    _forward_global_from_z,
    _forward_task_from_z,
    _mask_logits_by_task_keys,
    _masked_accuracy,
    _use_merged_organ_task_adaptors,
)
from .losses import bank_contrastive_loss


# ---------------------------------------------------------- replay buffer ---

class DERReplayBuffer:
    """Flat reservoir buffer storing (feats_tensor, label, logits).

    Each entry holds the full patch feature tensor [N_patches, D] on CPU,
    matching the ConSlide design. Reservoir sampling ensures uniform coverage.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: List[Dict] = []
        self._n_seen = 0

    def __len__(self):
        return len(self.buffer)

    @torch.no_grad()
    def add_batch(
        self,
        feats_list: List[torch.Tensor],
        fine_ids: torch.Tensor,
        logits: torch.Tensor,
        task_keys: List[str],
    ):
        """Reservoir-sample a batch of WSIs into the flat buffer.

        Args:
            feats_list: list of [N_patches, D] tensors (one per WSI).
            fine_ids:   [B] global fine label indices.
            logits:     [B, C] model output logits.
            task_keys:  list of task key strings (length B).
        """
        for i in range(len(feats_list)):
            self._n_seen += 1
            entry = {
                "feats": feats_list[i].detach().cpu(),   # [N_patches, D]
                "fine_id": int(fine_ids[i].item()),
                "logits": logits[i].detach().cpu(),
                "n_active": logits.size(1),
                "task_key": task_keys[i],
            }
            if len(self.buffer) < self.max_size:
                self.buffer.append(entry)
            else:
                j = random.randint(0, self._n_seen - 1)
                if j < self.max_size:
                    self.buffer[j] = entry

    def sample(self, batch_size: int) -> Optional[Dict]:
        """Sample a random mini-batch from the flat buffer.

        Returns dict with:
            feats_list:    list of [N_patches, D] tensors (cpu)
            fine_ids:      LongTensor [N]
            stored_logits: FloatTensor [N, C_max]  (padded with -1e9)
            n_active:      LongTensor [N]
            task_keys:     list of str
        """
        if not self.buffer:
            return None

        n = min(batch_size, len(self.buffer))
        chosen = random.sample(self.buffer, n)

        max_c = max(e["n_active"] for e in chosen)

        feats_list, ys, logits_list, n_actives, task_keys = [], [], [], [], []
        for e in chosen:
            feats_list.append(e["feats"])
            ys.append(e["fine_id"])
            pad = torch.full((max_c,), -1e9)
            pad[: e["n_active"]] = e["logits"]
            logits_list.append(pad)
            n_actives.append(e["n_active"])
            task_keys.append(e["task_key"])

        return {
            "feats_list": feats_list,
            "fine_ids": torch.tensor(ys, dtype=torch.long),
            "stored_logits": torch.stack(logits_list),
            "n_active": torch.tensor(n_actives, dtype=torch.long),
            "task_keys": task_keys,
        }


def _dark_experience_loss(current_logits, stored_logits, n_active):
    """MSE loss between current and stored logits, only on overlapping dims."""
    loss = current_logits.new_tensor(0.0)
    count = 0
    for i in range(current_logits.size(0)):
        n = min(int(n_active[i].item()), current_logits.size(1))
        loss = loss + F.mse_loss(current_logits[i, :n], stored_logits[i, :n])
        count += 1
    return loss / max(count, 1)


# ---------------------------------------------------------- buffer update ---

@torch.no_grad()
def update_der_buffer(model, loader, buffer, device, cfg):
    """After training a session, reservoir-sample training WSIs into buffer."""
    model.eval()
    opened_task_keys = model.get_opened_task_keys()

    for batch in loader:
        feats_list = [f.to(device) for f in batch["feats_list"]]
        fine_global = batch["fine_ids"].to(device)
        task_keys, _, _ = infer_task_keys(model, batch, fine_global)

        z_shared, _ = model.encode_shared(feats_list)
        out = _forward_global_from_z(model, z_shared, opened_task_keys, cfg)
        logits = out["fine_logits"]

        buffer.add_batch(batch["feats_list"], fine_global, logits, task_keys)


# ------------------------------------------------------------- train loop ---

def train_session_derpp(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    session_idx: int,
    cfg: dict,
    replay_buffer: Optional[DERReplayBuffer] = None,
    lambda_fine: float = 1.0,
    debug_print_freq: int = 20,
    **kwargs,
):
    """DER++ training loop for one epoch.

    Loss = CE_current + α · CE_replay + β · MSE_dark
    """
    model.train()
    train_cfg = cfg.get("train", {})
    stats = RunningStats()

    alpha = float(train_cfg.get("derpp_alpha", 0.2))
    beta = float(train_cfg.get("derpp_beta", 0.2))
    lam_task = float(train_cfg.get("lambda_cls_task", lambda_fine))
    lam_global = float(train_cfg.get("lambda_cls_global", 0.3))
    lam_align = float(train_cfg.get("lambda_align", 0.0))
    replay_batch_size = int(train_cfg.get("derpp_replay_batch_size",
                                          train_cfg.get("batch_size", 32)))

    opened_task_keys = model.get_opened_task_keys()

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"[DER++] S{session_idx} E{epoch + 1}",
        leave=False,
    )

    for step, batch in pbar:
        feats = [f.to(device) for f in batch["feats_list"]]
        fine_global = batch["fine_ids"].to(device)
        fine_local = model.global_fine_to_local(fine_global)
        task_keys, _, _ = infer_task_keys(model, batch, fine_global)

        optimizer.zero_grad(set_to_none=True)

        z_shared, _ = model.encode_shared(feats)

        # --- current batch forward ---
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

        loss_task = F.cross_entropy(task_logits, fine_local)
        loss_global = (
            F.cross_entropy(global_logits, fine_local)
            if lam_global > 0 and len(opened_task_keys) > 1
            else loss_task.new_tensor(0.0)
        )

        if lam_align > 0:
            fine_bank = model.get_active_fine_bank(device)
            loss_align = bank_contrastive_loss(
                out_task["h_f"], fine_bank, fine_local,
                tau=train_cfg.get("align_temperature", 0.07),
            )
        else:
            loss_align = loss_task.new_tensor(0.0)

        # --- replay ---
        loss_replay_ce = loss_task.new_tensor(0.0)
        loss_replay_mse = loss_task.new_tensor(0.0)

        if replay_buffer is not None and len(replay_buffer) > 0 and session_idx > 0:
            rep = replay_buffer.sample(batch_size=replay_batch_size)
            if rep is not None:
                rep_feats = [f.to(device) for f in rep["feats_list"]]
                rep_fine_ids = rep["fine_ids"].to(device)
                rep_local = model.global_fine_to_local(rep_fine_ids)
                stored_logits = rep["stored_logits"].to(device)
                n_active = rep["n_active"].to(device)

                rep_z, _ = model.encode_shared(rep_feats)
                rep_out = _forward_global_from_z(model, rep_z, opened_task_keys, cfg)
                rep_logits = rep_out["fine_logits"]

                loss_replay_ce = F.cross_entropy(rep_logits, rep_local)
                loss_replay_mse = _dark_experience_loss(rep_logits, stored_logits, n_active)

        loss = (
            lam_task * loss_task
            + lam_global * loss_global
            + lam_align * loss_align
            + alpha * loss_replay_ce
            + beta * loss_replay_mse
        )

        loss.backward()
        optimizer.step()

        task_acc = _masked_accuracy(task_logits, fine_local)
        global_acc = _masked_accuracy(global_logits, fine_local)

        stats.update(
            loss_total=loss,
            loss_task=loss_task,
            loss_global=loss_global,
            loss_align=loss_align,
            loss_replay_ce=loss_replay_ce,
            loss_replay_mse=loss_replay_mse,
            task_acc=task_acc,
            global_acc=global_acc,
        )

        if (step + 1) % max(1, debug_print_freq) == 0:
            pbar.set_postfix({
                "L": f"{loss.item():.3f}",
                "A": f"{loss_align.item():.3f}",
                "RCE": f"{loss_replay_ce.item():.3f}",
                "MSE": f"{loss_replay_mse.item():.3f}",
            })

    avg = stats.as_dict()
    print(
        f"[DER++ S{session_idx} E{epoch + 1:02d}] "
        f"L={avg.get('loss_total', 0):.4f} "
        f"TaskCE={avg.get('loss_task', 0):.4f} "
        f"GlobalCE={avg.get('loss_global', 0):.4f} "
        f"Align={avg.get('loss_align', 0):.4f} "
        f"ReplayCE={avg.get('loss_replay_ce', 0):.4f} "
        f"DarkMSE={avg.get('loss_replay_mse', 0):.4f} "
        f"TaskAcc={avg.get('task_acc', 0) * 100:.2f}% "
        f"GlobalAcc={avg.get('global_acc', 0) * 100:.2f}%"
    )
    return avg
