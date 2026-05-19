from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Poincaré ball primitives (used only in hyperbolic refinement path)
# ---------------------------------------------------------------------------

def _expmap0_poincare(u: torch.Tensor, c: float = 1.0, max_norm: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    sc = c ** 0.5
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    u = u * torch.clamp(max_norm / norm, max=1.0)
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    x = torch.tanh(sc * norm) * u / (sc * norm)
    max_ball = (1.0 - 1e-5) / sc
    return x * torch.clamp(max_ball / x.norm(dim=-1, keepdim=True).clamp_min(eps), max=1.0)


def _poincare_dist_pw(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Pairwise Poincaré distance: [N, D] x [M, D] -> [N, M]."""
    c = float(c)
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True).T
    diff2 = torch.cdist(x, y, p=2).pow(2)
    denom = (1.0 - c * x2).clamp_min(eps) * (1.0 - c * y2).clamp_min(eps)
    return torch.acosh((1.0 + 2.0 * c * diff2 / denom).clamp_min(1.0 + eps)) / (c ** 0.5)


def _poincare_dist_elem(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Element-wise Poincaré distance: d(x_i, y_i) for [N, D] x [N, D] -> [N]."""
    c = float(c)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    diff2 = ((x - y) ** 2).sum(-1)
    denom = (1.0 - c * x2).clamp_min(eps) * (1.0 - c * y2).clamp_min(eps)
    return torch.acosh((1.0 + 2.0 * c * diff2 / denom).clamp_min(1.0 + eps)) / (c ** 0.5)


@dataclass
class TextRefineConfig:
    """Delta-only fine-anchor refinement.

    Sibling separation is task-aware when ``fine_to_sibling_group`` is provided.
    Parent attraction still uses ``fine_to_coarse``.
    """

    steps: int = 300
    lr: float = 1e-2
    sib_cos_upper: float = 0.6        # Euclidean: max cosine sim between siblings
    sib_hyp_margin: float = 2.0       # Hyperbolic: min d_H between siblings
    parent_margin: float = 0.8        # Hyperbolic: hinge threshold for parent attraction

    new_delta_scale: float = 1.0
    old_delta_scale: float = 0.0

    lambda_sib: float = 1.0
    lambda_par: float = 0.3
    lambda_dir: float = 0.05          # KEEP direction preservation
    lambda_delta: float = 0.01        # delta L2 penalty

    lambda_cross: float = 0.0         # cross-coarse hard-negative loss (0 = off)
    cross_cos_upper: float = 0.2      # max allowed cosine sim for cross-coarse pairs
    cross_topk: int = 3               # hard negatives per new fine label

    print_every: int = 50
    eps: float = 1e-8


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (normalize(a, eps) * normalize(b, eps)).sum(dim=-1).clamp(-1.0, 1.0)


def angle_deg(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.rad2deg(torch.acos(cosine(a, b, eps).clamp(-1.0, 1.0)))


def unique_keep_order(xs: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(list(xs)))


class DeltaAnchorRefiner(nn.Module):
    """Frozen full anchor bank + learnable deltas for selected fine labels."""

    def __init__(
        self,
        base_bank: torch.Tensor,
        fine_labels: Sequence[str],
        train_labels: Sequence[str],
        old_labels: Sequence[str],
        cfg: TextRefineConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.fine_labels = list(fine_labels)
        self.label_to_idx = {label: i for i, label in enumerate(self.fine_labels)}
        self.train_labels = unique_keep_order(train_labels)
        self.old_labels = set(old_labels)

        missing = [x for x in self.train_labels if x not in self.label_to_idx]
        if missing:
            raise KeyError(f"train_labels not found in fine_labels: {missing}")

        base = normalize(base_bank.detach().clone().float(), eps=cfg.eps)
        self.register_buffer("base_bank", base)
        self.register_buffer(
            "train_idx",
            torch.tensor(
                [self.label_to_idx[x] for x in self.train_labels],
                dtype=torch.long,
                device=base.device,
            ),
        )
        self.register_buffer(
            "delta_scale",
            torch.tensor(
                [cfg.old_delta_scale if x in self.old_labels else cfg.new_delta_scale for x in self.train_labels],
                dtype=base.dtype,
                device=base.device,
            ).unsqueeze(1),
        )
        self.delta = nn.Parameter(torch.zeros(len(self.train_labels), base.shape[1], device=base.device))

    def current_bank(self) -> torch.Tensor:
        bank = self.base_bank.clone()
        bank[self.train_idx] = normalize(
            bank[self.train_idx] + self.delta_scale * self.delta,
            eps=self.cfg.eps,
        )
        return bank


def _sibling_pair_mask(
    labels: Sequence[str],
    group_of_fine: Dict[str, str],
    device: torch.device,
) -> torch.Tensor:
    labels = list(labels)
    mask = torch.zeros(len(labels), len(labels), dtype=torch.bool, device=device)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if group_of_fine.get(labels[i]) == group_of_fine.get(labels[j]):
                mask[i, j] = True
    return mask


def _stats(local: torch.Tensor, sibling_mask: torch.Tensor, parent_local: torch.Tensor) -> Dict[str, float]:
    sim = normalize(local) @ normalize(local).T
    sib = sim[sibling_mask]
    par = cosine(local, parent_local)
    return {
        "sibling_mean_cos": float(sib.mean().detach().cpu()) if sib.numel() else float("nan"),
        "sibling_max_cos": float(sib.max().detach().cpu()) if sib.numel() else float("nan"),
        "parent_mean_cos": float(par.mean().detach().cpu()) if par.numel() else float("nan"),
        "parent_min_cos": float(par.min().detach().cpu()) if par.numel() else float("nan"),
    }


def refine_text_anchors(
    base_bank: torch.Tensor,
    fine_labels: Sequence[str],
    fine_to_coarse: Dict[str, str],
    train_labels: Sequence[str],
    parent_bank: torch.Tensor,
    coarse_labels: Sequence[str],
    old_labels: Optional[Sequence[str]] = None,
    fine_to_sibling_group: Optional[Dict[str, str]] = None,
    cfg: Optional[TextRefineConfig] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Refine selected fine anchors and return the full refined fine bank.

    Args:
        base_bank: Running fine-anchor bank before current session, shape [F, D].
        fine_labels: Row order of ``base_bank``.
        fine_to_coarse: Fine label -> parent organ/coarse label.
        train_labels: Fine labels locally optimized in the current refinement step.
        parent_bank: Fixed parent anchor bank, shape [C, D].
        coarse_labels: Row order of ``parent_bank``.
        old_labels: Previous labels among ``train_labels``. They are fixed when old_delta_scale=0.
        fine_to_sibling_group: Optional fine label -> organ-task group. If omitted, uses fine_to_coarse.
        cfg: Refinement hyperparameters.
    """
    cfg = cfg or TextRefineConfig()
    fine_labels = list(fine_labels)
    coarse_labels = list(coarse_labels)
    label_to_idx = {label: i for i, label in enumerate(fine_labels)}
    coarse_to_idx = {label: i for i, label in enumerate(coarse_labels)}

    train_labels = [x for x in unique_keep_order(train_labels) if x in label_to_idx]
    old_labels = [x for x in unique_keep_order(old_labels or []) if x in set(train_labels)]

    if base_bank.shape[0] != len(fine_labels):
        raise ValueError(f"base_bank rows={base_bank.shape[0]} but len(fine_labels)={len(fine_labels)}")
    if not train_labels:
        return normalize(base_bank, eps=cfg.eps), {"config": asdict(cfg), "history": [], "final": {"loss": 0.0}}

    missing_parent = sorted({fine_to_coarse[x] for x in train_labels if fine_to_coarse.get(x) not in coarse_to_idx})
    if missing_parent:
        raise KeyError(f"Missing parent anchors in coarse_labels/parent_bank: {missing_parent}")

    device = base_bank.device
    base_bank = normalize(base_bank.float(), eps=cfg.eps)
    parent_bank = normalize(parent_bank.to(device=device, dtype=torch.float32), eps=cfg.eps)

    train_idx = torch.tensor([label_to_idx[x] for x in train_labels], dtype=torch.long, device=device)
    parent_idx = torch.tensor([coarse_to_idx[fine_to_coarse[x]] for x in train_labels], dtype=torch.long, device=device)

    old_set = set(old_labels)
    movable_mask = torch.tensor(
        [(x not in old_set) or (float(cfg.old_delta_scale) != 0.0) for x in train_labels],
        dtype=torch.bool,
        device=device,
    )
    movable_labels = [x for x, m in zip(train_labels, movable_mask.detach().cpu().tolist()) if m]

    model = DeltaAnchorRefiner(base_bank, fine_labels, train_labels, old_labels, cfg).to(device)
    opt = torch.optim.AdamW([model.delta], lr=cfg.lr, weight_decay=0.0)

    base_local = base_bank[train_idx].detach()
    parent_local = parent_bank[parent_idx].detach()
    group_of_fine = fine_to_sibling_group or fine_to_coarse
    sibling_mask = _sibling_pair_mask(train_labels, group_of_fine, device)
    sibling_loss_mask = sibling_mask & (movable_mask[:, None] | movable_mask[None, :])

    # Fixed cross-coarse reference anchors: opened fine labels not in train_labels
    # that belong to a different coarse than any movable (new) label.
    new_coarses = {fine_to_coarse.get(x) for x in movable_labels}
    use_cross = cfg.lambda_cross > 0 and bool(movable_labels)
    other_fine_ref: Optional[torch.Tensor] = None
    cross_topk: int = 0
    if use_cross:
        other_idx = [
            label_to_idx[lab] for lab in fine_labels
            if lab not in set(train_labels) and fine_to_coarse.get(lab) not in new_coarses
        ]
        if other_idx:
            other_fine_ref = base_bank[torch.tensor(other_idx, dtype=torch.long, device=device)].detach()
            cross_topk = min(int(cfg.cross_topk), other_fine_ref.size(0))

    before_stats = _stats(base_local, sibling_mask, parent_local)
    zero = base_local.new_tensor(0.0)

    if movable_mask.sum().item() == 0:
        logs = {
            "config": asdict(cfg),
            "train_labels": train_labels,
            "old_labels": old_labels,
            "movable_labels": [],
            "parent_labels": [fine_to_coarse[x] for x in train_labels],
            "sibling_groups": {x: group_of_fine.get(x) for x in train_labels},
            "final": {"loss": 0.0, "sib": 0.0, "par": 0.0},
            "history": [],
            "stats_before": before_stats,
            "stats_after": before_stats,
        }
        return normalize(base_bank, eps=cfg.eps), logs

    history: List[Dict[str, float]] = []
    for step in range(1, cfg.steps + 1):
        bank = model.current_bank()
        cur = bank[train_idx]
        cur_movable = cur[movable_mask]
        parent_movable = parent_local[movable_mask]
        sim = cur @ cur.T

        loss_sib = F.relu(sim[sibling_loss_mask] - cfg.sib_cos_upper).pow(2).mean() if sibling_loss_mask.any() else zero
        loss_par = (1.0 - cosine(cur_movable, parent_movable, cfg.eps)).mean() if cur_movable.numel() > 0 else zero

        if other_fine_ref is not None and cur_movable.numel() > 0:
            # Hard cross-coarse negatives: top-k closest other-coarse fine per new label
            sim_cross = cur_movable @ other_fine_ref.T           # [M_new, K]
            topk_sim = sim_cross.topk(cross_topk, dim=-1).values  # [M_new, top_k]
            loss_cross = F.relu(topk_sim - cfg.cross_cos_upper).pow(2).mean()
        else:
            loss_cross = zero

        loss = cfg.lambda_sib * loss_sib + cfg.lambda_par * loss_par + cfg.lambda_cross * loss_cross

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        row = {
            "step": float(step),
            "loss": float(loss.detach().cpu()),
            "sib": float(loss_sib.detach().cpu()),
            "par": float(loss_par.detach().cpu()),
            "cross": float(loss_cross.detach().cpu()),
        }
        history.append(row)

        if verbose and (step == 1 or step % max(int(cfg.print_every), 1) == 0 or step == cfg.steps):
            msg = " ".join(f"{k}={v:.5f}" for k, v in row.items() if k != "step")
            print(f"[refine {step:04d}/{cfg.steps}] {msg}")

    refined_bank = model.current_bank().detach()
    after_stats = _stats(refined_bank[train_idx], sibling_mask, parent_local)
    logs: Dict[str, object] = {
        "config": asdict(cfg),
        "train_labels": train_labels,
        "old_labels": old_labels,
        "movable_labels": movable_labels,
        "parent_labels": [fine_to_coarse[x] for x in train_labels],
        "sibling_groups": {x: group_of_fine.get(x) for x in train_labels},
        "final": history[-1] if history else {"loss": 0.0},
        "history": history,
        "stats_before": before_stats,
        "stats_after": after_stats,
    }
    return normalize(refined_bank, eps=cfg.eps), logs


# ---------------------------------------------------------------------------
# Hyperbolic text refinement (Poincaré ball)
# ---------------------------------------------------------------------------

class HyperbolicDeltaRefiner(nn.Module):
    """Refine fine-label anchor **directions** in Poincaré ball.

    Radius is fixed per hierarchy level by ``fine_scale``.
    Only the unit direction is optimized via a learnable delta.
    """

    def __init__(
        self,
        base_directions: torch.Tensor,  # [F, D] unit vectors for all fine labels
        movable_idx: torch.Tensor,       # [M] indices of labels to optimize
        fine_scale: float,
        c: float = 1.0,
        max_norm: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.fine_scale = fine_scale
        self.c = c
        self.max_norm = max_norm
        self.eps = eps
        self.register_buffer("base_dir", F.normalize(base_directions.float(), dim=-1))
        self.register_buffer("movable_idx", movable_idx)
        self.delta = nn.Parameter(torch.zeros(len(movable_idx), base_directions.shape[1]))

    def _movable_dirs(self) -> torch.Tensor:
        return F.normalize(self.base_dir[self.movable_idx] + self.delta, dim=-1, eps=self.eps)

    def all_directions(self) -> torch.Tensor:
        dirs = self.base_dir.clone()
        dirs[self.movable_idx] = self._movable_dirs()
        return dirs

    def movable_poincare(self) -> torch.Tensor:
        """Poincaré ball points for the M movable labels only."""
        u = self._movable_dirs() * self.fine_scale
        return _expmap0_poincare(u, c=self.c, max_norm=self.max_norm, eps=self.eps)


def refine_text_anchors_hyperbolic(
    base_directions: torch.Tensor,             # [F, D] unit vectors (KEEP fine embeds)
    fine_labels: Sequence[str],
    fine_to_immediate_parent: Dict[str, str],  # fine -> task_key (immediate parent)
    train_labels: Sequence[str],               # fine labels to optimize
    parent_unit_dirs: Dict[str, torch.Tensor], # task_key / coarse_key -> [D] unit vector
    hyp_cfg: dict,
    old_labels: Optional[Sequence[str]] = None,
    cfg: Optional[TextRefineConfig] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """Refine fine anchor directions using Poincaré ball distances.

    Hierarchy (3-level):
        coarse  (fixed, coarse_tangent_scale)
        task    (fixed, task_tangent_scale)   ← immediate parent of fine
        fine    (optimized, text_tangent_scale)

    loss_par : minimize d_H(fine_i, task_parent_i)  → pull toward task parent
    loss_sib : max(0, margin - d_H(fine_i, fine_j)) for same-task siblings → push apart

    Returns refined tangent vectors [F, D] (fine_scale × direction),
    suitable for anchor_space="poincare_tangent".
    """
    cfg = cfg or TextRefineConfig()
    c = float(hyp_cfg.get("curvature", 1.0))
    max_norm = float(hyp_cfg.get("max_tangent_norm", 1.0))
    fine_scale = float(hyp_cfg.get("fine_tangent_scale", hyp_cfg.get("text_tangent_scale", 1.0)))
    parent_scale = float(hyp_cfg.get("task_tangent_scale", 0.6))
    sib_margin = float(cfg.sib_hyp_margin)
    par_margin = float(cfg.parent_margin)

    fine_labels = list(fine_labels)
    label_to_idx = {l: i for i, l in enumerate(fine_labels)}
    old_set = set(old_labels or [])

    train_labels = [l for l in unique_keep_order(train_labels) if l in label_to_idx]
    movable = [l for l in train_labels if l not in old_set or float(cfg.old_delta_scale) != 0.0]

    if not movable:
        return F.normalize(base_directions.float(), dim=-1) * fine_scale, {}

    device = base_directions.device
    movable_idx = torch.tensor([label_to_idx[l] for l in movable], dtype=torch.long, device=device)

    refiner = HyperbolicDeltaRefiner(
        base_directions, movable_idx, fine_scale, c=c, max_norm=max_norm, eps=cfg.eps
    ).to(device)
    opt = torch.optim.AdamW([refiner.delta], lr=cfg.lr, weight_decay=0.0)

    # Pre-compute fixed parent Poincaré points
    parent_poin: Dict[str, torch.Tensor] = {}
    for key, unit_dir in parent_unit_dirs.items():
        u = F.normalize(unit_dir.float(), dim=-1) * parent_scale
        parent_poin[key] = _expmap0_poincare(u.to(device), c=c, max_norm=max_norm)

    # Sibling mask among movable labels (same immediate parent = siblings)
    M = len(movable)
    sib_mask = torch.zeros(M, M, dtype=torch.bool, device=device)
    for i in range(M):
        for j in range(i + 1, M):
            if fine_to_immediate_parent.get(movable[i]) == fine_to_immediate_parent.get(movable[j]):
                sib_mask[i, j] = True
                sib_mask[j, i] = True

    # Parent list for each movable label
    movable_parent_keys = [fine_to_immediate_parent.get(l) for l in movable]

    zero = base_directions.new_tensor(0.0)
    history: List[Dict[str, float]] = []

    for step in range(1, cfg.steps + 1):
        m_poin = refiner.movable_poincare()  # [M, D]

        # loss_sib: push siblings apart if closer than sib_margin
        if sib_mask.any():
            dists = _poincare_dist_pw(m_poin, m_poin, c=c)          # [M, M]
            loss_sib = F.relu(sib_margin - dists[sib_mask]).pow(2).mean()
        else:
            loss_sib = zero

        # loss_par: hinge on d_H(fine, task_parent) — pull toward parent but not collapse
        par_pts = [parent_poin[k] for k in movable_parent_keys if k in parent_poin]
        par_idx = [i for i, k in enumerate(movable_parent_keys) if k in parent_poin]
        if par_pts:
            par_tensor = torch.stack(par_pts)                         # [V, D]
            fine_pts = m_poin[torch.tensor(par_idx, device=device)]  # [V, D]
            d_par = _poincare_dist_elem(fine_pts, par_tensor, c=c)
            loss_par = F.relu(d_par - par_margin).pow(2).mean()
        else:
            loss_par = zero

        # loss_dir: KEEP direction preservation (cosine)
        base_dirs = refiner.base_dir[refiner.movable_idx]  # [M, D]
        refined_dirs = refiner._movable_dirs()              # [M, D]
        loss_dir = (1.0 - (base_dirs * refined_dirs).sum(dim=-1)).mean()

        # loss_delta: L2 penalty on delta
        loss_delta = refiner.delta.pow(2).mean()

        loss = (
            cfg.lambda_sib * loss_sib
            + cfg.lambda_par * loss_par
            + cfg.lambda_dir * loss_dir
            + cfg.lambda_delta * loss_delta
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        row = {
            "step": float(step),
            "loss": float(loss.detach()),
            "sib": float(loss_sib.detach()),
            "par": float(loss_par.detach()),
            "dir": float(loss_dir.detach()),
            "delta": float(loss_delta.detach()),
        }
        history.append(row)
        if verbose and (step == 1 or step % max(int(cfg.print_every), 1) == 0 or step == cfg.steps):
            print(f"[hyp_refine {step:04d}/{cfg.steps}] " +
                  " ".join(f"{k}={v:.5f}" for k, v in row.items() if k != "step"))

    refined_dirs = refiner.all_directions().detach()          # [F, D] unit vectors
    refined_tangent = refined_dirs * fine_scale               # [F, D] tangent vectors

    return refined_tangent, {
        "history": history,
        "final": history[-1] if history else {},
        "movable_labels": movable,
    }
