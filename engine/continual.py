from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .losses import (
    bank_contrastive_loss,
    sibling_margin_loss,
    hyperbolic_hierarchy_loss,
)

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _safe_item(x):
    if x is None:
        return 0.0
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _to_int_list(x) -> List[int]:
    if x is None:
        return []
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().cpu().tolist()]
    return [int(v) for v in list(x)]


def _as_list(x) -> List[Any]:
    if x is None:
        return []
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _get_batch_value(batch: Dict[str, Any], keys: Sequence[str], default=None):
    for k in keys:
        if k in batch:
            return batch[k]
    return default


def _taxonomy_idx_to_fine(taxonomy, idx: int) -> str:
    if hasattr(taxonomy, "idx_to_fine"):
        return taxonomy.idx_to_fine[int(idx)]
    return taxonomy.fine_labels[int(idx)]


def _taxonomy_idx_to_coarse(taxonomy, idx: int) -> str:
    if hasattr(taxonomy, "idx_to_coarse"):
        return taxonomy.idx_to_coarse[int(idx)]
    return taxonomy.coarse_labels[int(idx)]


def _fine_to_coarse(taxonomy, fine_lab: str) -> str:
    if hasattr(taxonomy, "fine_to_coarse"):
        return taxonomy.fine_to_coarse[fine_lab]
    return taxonomy.fine_nodes[fine_lab]["parent"]


def _fine_to_task(taxonomy, fine_lab: str) -> str:
    if hasattr(taxonomy, "fine_to_task"):
        return taxonomy.fine_to_task.get(fine_lab, "default")
    if hasattr(taxonomy, "fine_nodes"):
        return taxonomy.fine_nodes.get(fine_lab, {}).get("task", "default")
    return "default"


def make_task_key(organ, task="default") -> str:
    organ = str(organ)
    task = str(task)
    if task in {"", "None", "none", "null", "default"}:
        return organ
    return f"{organ}/{task}"


def _infer_context_from_fine_global_ids(model, fine_ids_global) -> Tuple[List[str], List[str]]:
    organs, tasks = [], []
    for fine_id in _to_int_list(fine_ids_global):
        fine_lab = _taxonomy_idx_to_fine(model.taxonomy, fine_id)
        organs.append(_fine_to_coarse(model.taxonomy, fine_lab))
        tasks.append(_fine_to_task(model.taxonomy, fine_lab))
    return organs, tasks


def _infer_batch_context(model, batch, fine_ids_global) -> Tuple[List[str], List[str]]:
    organs = _get_batch_value(batch, ["coarse_labels", "organ_labels", "organs", "organ"])
    tasks = _get_batch_value(batch, ["task_labels", "tasks", "task"])

    if organs is None:
        coarse_ids = _get_batch_value(batch, ["coarse_ids", "organ_ids"])
        if coarse_ids is None:
            organs, _ = _infer_context_from_fine_global_ids(model, fine_ids_global)
        else:
            organs = [_taxonomy_idx_to_coarse(model.taxonomy, idx) for idx in _to_int_list(coarse_ids)]
    else:
        organs = [str(x) for x in _as_list(organs)]

    if tasks is None:
        _, tasks = _infer_context_from_fine_global_ids(model, fine_ids_global)
    else:
        tasks = [str(x) for x in _as_list(tasks)]

    b = int(fine_ids_global.numel())
    if len(organs) != b or len(tasks) != b:
        raise ValueError(f"Context size mismatch: organs={len(organs)}, tasks={len(tasks)}, batch={b}")
    return organs, tasks


def infer_task_keys(model, batch, fine_ids_global) -> Tuple[List[str], List[str], List[str]]:
    organs, tasks = _infer_batch_context(model, batch, fine_ids_global)
    task_keys = [make_task_key(o, t) for o, t in zip(organs, tasks)]
    return task_keys, organs, tasks


def _session_new_fine(session_manager, session_idx: int) -> List[str]:
    return list(session_manager.new_fine_labels(session_idx))


def _session_opened_fine(session_manager, session_idx: int) -> List[str]:
    return list(session_manager.opened_fine_labels(session_idx))


def _session_task_key(model, session_manager, session_idx: int, cfg=None) -> str:
    session = (cfg or {}).get("sessions", [{}])[session_idx] if cfg is not None else {}
    new_fine = _session_new_fine(session_manager, session_idx)

    organ = session.get("organ_id", session.get("organ", None))
    task = session.get("task_id", session.get("task", None))

    if organ is None:
        organs = sorted({_fine_to_coarse(model.taxonomy, f) for f in new_fine})
        organ = organs[0] if len(organs) == 1 else session.get("name", f"session_{session_idx}")
    if task is None:
        tasks = sorted({_fine_to_task(model.taxonomy, f) for f in new_fine})
        task = tasks[0] if len(tasks) == 1 else session.get("name", f"session_{session_idx}")

    return make_task_key(organ, task)


def _session_coarse_labels(model, fine_labels: Sequence[str]) -> List[str]:
    return sorted({_fine_to_coarse(model.taxonomy, f) for f in fine_labels})


def _safe_session_call(session_manager, method_name: str, session_idx: int, device=None):
    if not hasattr(session_manager, method_name):
        return None
    method = getattr(session_manager, method_name)
    try:
        return method(session_idx, device) if device is not None else method(session_idx)
    except TypeError:
        return method(session_idx)


class RunningStats:
    def __init__(self):
        self.data = defaultdict(list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(_safe_item(v))

    def mean(self, key: str) -> float:
        vals = self.data.get(key, [])
        return sum(vals) / len(vals) if vals else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {k: self.mean(k) for k in self.data}


# -----------------------------------------------------------------------------
# Hyperbolic helpers
# -----------------------------------------------------------------------------
_HYP_CLASSIFIER_MODES = {
    "classifier",
    "full",
    "prototype",
    "hyp_proto",
    "hyperbolic_classifier",
}


def _get_hyp_cfg(cfg) -> dict:
    return (cfg or {}).get("hyperbolic", {}) if isinstance(cfg, dict) else {}


def _get_hyp_mode(cfg) -> str:
    hyp_cfg = _get_hyp_cfg(cfg)
    if not bool(hyp_cfg.get("enabled", False)):
        return "off"
    return str(hyp_cfg.get("mode", "aux")).lower()


def _use_hyperbolic_classifier(model, cfg) -> bool:
    if hasattr(model, "use_hyperbolic_classifier"):
        return bool(model.use_hyperbolic_classifier())
    return _get_hyp_mode(cfg) in _HYP_CLASSIFIER_MODES


def _use_merged_organ_task_adaptors(model) -> bool:
    if hasattr(model, "use_merged_organ_task_adaptors"):
        return bool(model.use_merged_organ_task_adaptors())

    mode = str(getattr(model, "organ_task_adaptor_mode", "branch")).lower()
    return mode in {"merge", "merged", "sum", "all", "all_active"}

def _clamp_tangent_norm(u, max_norm=1.0, eps=1e-8):
    if max_norm is None or float(max_norm) <= 0:
        return u
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    factor = torch.clamp(float(max_norm) / norm, max=1.0)
    return u * factor


def _expmap0_lorentz(u, c=1.0, max_tangent_norm=1.0, eps=1e-8):
    c = float(c)
    sqrt_c = c ** 0.5
    u = _clamp_tangent_norm(u, max_tangent_norm, eps=eps)
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    theta = sqrt_c * norm
    time = torch.cosh(theta) / sqrt_c
    space = torch.sinh(theta) * u / (sqrt_c * norm)
    return torch.cat([time, space], dim=-1)


def _expmap0_poincare(u, c=1.0, max_tangent_norm=1.0, eps=1e-8):
    c = float(c)
    sqrt_c = c ** 0.5
    u = _clamp_tangent_norm(u, max_tangent_norm, eps=eps)
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    x = torch.tanh(sqrt_c * norm) * u / (sqrt_c * norm)
    max_ball_norm = (1.0 - 1e-5) / sqrt_c
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x * torch.clamp(max_ball_norm / x_norm, max=1.0)


def _lorentz_distance(x, y, c=1.0, eps=1e-6):
    ip = -x[:, :1] * y[:, :1].T + x[:, 1:] @ y[:, 1:].T
    z = (-float(c) * ip).clamp_min(1.0 + eps)
    return torch.acosh(z) / (float(c) ** 0.5)


def _poincare_distance(x, y, c=1.0, eps=1e-6):
    c = float(c)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True).T
    diff2 = torch.cdist(x, y, p=2).pow(2)
    denom = (1.0 - c * x2).clamp_min(eps) * (1.0 - c * y2).clamp_min(eps)
    z = 1.0 + 2.0 * c * diff2 / denom
    return torch.acosh(z.clamp_min(1.0 + eps)) / (c ** 0.5)


def _hyperbolic_distance_from_tangent(u_img, u_bank, hyp_cfg):
    geometry = str(hyp_cfg.get("geometry", "poincare")).lower()
    c = float(hyp_cfg.get("curvature", 1.0))
    max_norm = float(hyp_cfg.get("max_tangent_norm", 1.0))

    if geometry == "lorentz":
        x = _expmap0_lorentz(u_img, c=c, max_tangent_norm=max_norm)
        t = _expmap0_lorentz(u_bank, c=c, max_tangent_norm=max_norm)
        return _lorentz_distance(x, t, c=c)

    if geometry == "poincare":
        x = _expmap0_poincare(u_img, c=c, max_tangent_norm=max_norm)
        t = _expmap0_poincare(u_bank, c=c, max_tangent_norm=max_norm)
        return _poincare_distance(x, t, c=c)

    raise ValueError(f"Unsupported hyperbolic geometry: {geometry}")


def _hyperbolic_logits_from_tangent(u_img, u_bank, hyp_cfg):
    dist = _hyperbolic_distance_from_tangent(u_img, u_bank, hyp_cfg)
    tau = float(hyp_cfg.get("tau", 0.07))
    if bool(hyp_cfg.get("squared_distance", True)):
        return -dist.pow(2) / tau
    return -dist / tau


def _get_active_fine_bank_tangent(model, device, cfg):
    if hasattr(model, "get_active_fine_bank_tangent"):
        return model.get_active_fine_bank_tangent(device)

    hyp_cfg = _get_hyp_cfg(cfg)
    bank = model.get_active_fine_bank(device)

    anchor_space = str(getattr(model, "anchor_space", "euclidean")).lower()
    if anchor_space in {"hyperbolic_tangent", "tangent", "poincare_tangent", "lorentz_tangent"}:
        return bank

    scale = float(hyp_cfg.get("radius_fine", hyp_cfg.get("fine_tangent_scale", hyp_cfg.get("text_tangent_scale", 1.0))))
    return F.normalize(bank, dim=-1) * scale


def hyperbolic_bank_contrastive_loss(u, bank_tangent, targets, cfg):
    hyp_cfg = _get_hyp_cfg(cfg)
    logits = _hyperbolic_logits_from_tangent(u, bank_tangent, hyp_cfg)
    if bank_tangent.size(0) <= 1:
        dist = _hyperbolic_distance_from_tangent(u, bank_tangent, hyp_cfg)
        return dist.pow(2).mean()
    return F.cross_entropy(logits, targets)


def hyperbolic_sibling_margin_loss(u, bank_tangent, targets, parent_ids, cfg, margin=None):
    if bank_tangent.size(0) <= 1:
        return u.new_tensor(0.0)

    hyp_cfg = _get_hyp_cfg(cfg)
    dist = _hyperbolic_distance_from_tangent(u, bank_tangent, hyp_cfg)

    target_dist = dist.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_parent = parent_ids[targets]
    sibling_mask = parent_ids.unsqueeze(0).eq(target_parent.unsqueeze(1))
    sibling_mask.scatter_(1, targets.unsqueeze(1), False)

    valid = sibling_mask.any(dim=1)
    if not valid.any():
        return u.new_tensor(0.0)

    if margin is None:
        margin = float(hyp_cfg.get("sibling_margin", 0.5))

    sib_dist = dist.masked_fill(~sibling_mask, 1e9).min(dim=1).values
    return F.relu(target_dist - sib_dist + float(margin))[valid].mean()


# -----------------------------------------------------------------------------
# Refined anchors
# -----------------------------------------------------------------------------
@torch.no_grad()
def apply_refined_text_anchors_to_model(model, refined_anchor_path, opened_fine_labels, opened_coarse_labels, device):
    if refined_anchor_path is None:
        return

    artifact = torch.load(refined_anchor_path, map_location="cpu")
    anchor_space = str(artifact.get("anchor_space", "euclidean")).lower()
    model.anchor_space = anchor_space

    keep_norm = anchor_space in {
        "hyperbolic_tangent",
        "tangent",
        "poincare_tangent",
        "lorentz_tangent",
    }

    tree = getattr(model, "tree", None) or getattr(model, "knowledge_tree", None)
    if tree is None:
        raise AttributeError("Cannot find model.tree or model.knowledge_tree.")

    fine_src = list(artifact["fine_labels"])
    fine_bank = artifact["refined_fine_bank"].float()
    if not keep_norm:
        fine_bank = F.normalize(fine_bank, dim=-1)

    for lab in opened_fine_labels:
        if lab in fine_src:
            dst = model.taxonomy.fine_labels.index(lab)
            src = fine_src.index(lab)
            tree.fine_bank[dst].copy_(fine_bank[src].to(device))

    coarse_key = "fixed_parent_bank" if "fixed_parent_bank" in artifact else "refined_coarse_bank"
    if coarse_key in artifact and "coarse_labels" in artifact:
        coarse_src = list(artifact["coarse_labels"])
        coarse_bank = artifact[coarse_key].float()
        if not keep_norm:
            coarse_bank = F.normalize(coarse_bank, dim=-1)

        for lab in opened_coarse_labels:
            if lab in coarse_src:
                dst = model.taxonomy.coarse_labels.index(lab)
                src = coarse_src.index(lab)
                tree.coarse_bank[dst].copy_(coarse_bank[src].to(device))

    print(
        f"[TextRefine] Applied refined anchors: {refined_anchor_path} | "
        f"anchor_space={anchor_space} | normalize={not keep_norm}"
    )


# -----------------------------------------------------------------------------
# Masks / losses
# -----------------------------------------------------------------------------
def _get_candidate_fine_labels(model, organ, task):
    if hasattr(model.taxonomy, "get_candidate_fine_labels"):
        out = model.taxonomy.get_candidate_fine_labels(organ=organ, task=task)
        if out:
            return list(out)

    labels = []
    for lab in model.taxonomy.fine_labels:
        organ_ok = organ is None or _fine_to_coarse(model.taxonomy, lab) == organ
        task_ok = task is None or _fine_to_task(model.taxonomy, lab) == task
        if organ_ok and task_ok:
            labels.append(lab)
    return labels



def _task_mask_from_task_keys(model, task_keys, device, targets_local=None):
    mask = torch.zeros(len(task_keys), len(model.active_fine_labels), dtype=torch.bool, device=device)
    for i, task_key in enumerate(task_keys):
        task_key = str(task_key)
        if task_key not in model.task_to_local_fine:
            raise KeyError(f"Task key has no active labels: {task_key}")
        cols = model.task_to_local_fine[task_key].to(device)
        mask[i, cols] = True

    if targets_local is not None:
        rows = torch.arange(targets_local.numel(), device=device)
        mask[rows, targets_local] = True
    return mask


def _mask_logits_by_task_keys(model, logits, task_keys, targets_local=None):
    mask = _task_mask_from_task_keys(model, task_keys, logits.device, targets_local=targets_local)
    return logits.masked_fill(~mask, -1e9)


def _masked_argmax(logits, mask=None):
    if mask is None:
        return logits.argmax(dim=1)
    if mask.dim() == 1:
        return logits.masked_fill(~mask.unsqueeze(0), -1e9).argmax(dim=1)
    return logits.masked_fill(~mask, -1e9).argmax(dim=1)


def _masked_accuracy(logits, targets, mask=None) -> float:
    pred = _masked_argmax(logits, mask)
    return float((pred == targets).float().mean().item()) if targets.numel() else 0.0


def _build_active_fine_parent_ids(model, device):
    parent_to_id, ids = {}, []
    for lab in model.active_fine_labels:
        parent = _fine_to_coarse(model.taxonomy, lab)
        parent_to_id.setdefault(parent, len(parent_to_id))
        ids.append(parent_to_id[parent])
    return torch.tensor(ids, dtype=torch.long, device=device)


def classifier_weight_anchor_loss(model):
    if not hasattr(model, "fine_head"):
        return next(model.parameters()).new_tensor(0.0)
    w = F.normalize(model.fine_head.weight, dim=-1)
    bank = model.get_active_fine_bank(w.device)
    return (w - bank).pow(2).sum(dim=1).mean()

def _forward_task_from_z(model, z_shared, task_keys, cfg):
    task_keys = [str(k) for k in task_keys]

    if _use_merged_organ_task_adaptors(model):
        return model.forward_task_from_z(z_shared, task_keys)

    if not _use_hyperbolic_classifier(model, cfg):
        return model.forward_task_from_z(z_shared, task_keys)

    b = z_shared.size(0)
    c = len(model.active_fine_labels)
    logits = z_shared.new_full((b, c), -1e9)
    raw_h = z_shared.new_zeros((b, model.embed_dim))
    h = z_shared.new_zeros((b, model.embed_dim))
    bank_tangent = _get_active_fine_bank_tangent(model, z_shared.device, cfg)

    for task_key in sorted(set(task_keys)):
        if task_key not in model.task_to_local_fine:
            raise KeyError(f"Task key has no active labels: {task_key}")

        rows = torch.tensor(
            [i for i, k in enumerate(task_keys) if k == task_key],
            dtype=torch.long,
            device=z_shared.device,
        )

        out = model._branch_from_z(z_shared.index_select(0, rows), task_key)

        branch_logits = _hyperbolic_logits_from_tangent(
            out["raw_h_f"],
            bank_tangent,
            _get_hyp_cfg(cfg),
        )
        cols = model.task_to_local_fine[task_key].to(z_shared.device)

        logits[rows[:, None], cols[None, :]] = branch_logits[:, cols]
        raw_h[rows] = out["raw_h_f"]
        h[rows] = out["h_f"]

    return {
        "z": z_shared,
        "z_shared": z_shared,
        "raw_h_f": raw_h,
        "h_f": h,
        "fine_logits": logits,
        "fine_bank": model.get_active_fine_bank(z_shared.device),
        "fine_bank_tangent": bank_tangent,
    }


def _forward_global_from_z(model, z_shared, opened_task_keys, cfg):
    opened_task_keys = [str(k) for k in opened_task_keys]

    if _use_merged_organ_task_adaptors(model):
        return model.forward_global_from_z(z_shared, opened_task_keys)

    if not _use_hyperbolic_classifier(model, cfg):
        return model.forward_global_from_z(z_shared, opened_task_keys)

    b = z_shared.size(0)
    c = len(model.active_fine_labels)
    logits = z_shared.new_full((b, c), -1e9)
    bank_tangent = _get_active_fine_bank_tangent(model, z_shared.device, cfg)

    for task_key in opened_task_keys:
        if task_key not in model.task_to_local_fine:
            raise KeyError(f"Task key has no active labels: {task_key}")

        out = model._branch_from_z(z_shared, task_key)
        branch_logits = _hyperbolic_logits_from_tangent(
            out["raw_h_f"],
            bank_tangent,
            _get_hyp_cfg(cfg),
        )
        cols = model.task_to_local_fine[task_key].to(z_shared.device)
        logits[:, cols] = branch_logits[:, cols]

    return {
        "z": z_shared,
        "z_shared": z_shared,
        "fine_logits": logits,
        "fine_bank": model.get_active_fine_bank(z_shared.device),
        "fine_bank_tangent": bank_tangent,
    }


# -----------------------------------------------------------------------------
# Replay
# -----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, feature_dim=512, reg_covar=1e-4):
        self.feature_dim = int(feature_dim)
        self.reg_covar = float(reg_covar)
        self.stats_dict = {}

    @property
    def stats(self):
        return self.stats_dict

    @torch.no_grad()
    def _fit(self, z):
        if z.size(0) == 1:
            return z[0].clone(), torch.full_like(z[0], math.sqrt(self.reg_covar))
        return z.mean(0), z.var(0, unbiased=False).clamp_min(self.reg_covar).sqrt()

    @torch.no_grad()
    def update(self, task_key, fine_id, z):
        mu, sigma = self._fit(z.detach())
        self.stats_dict[(str(task_key), int(fine_id))] = {
            "mu": mu.cpu(),
            "sigma": sigma.cpu(),
        }

    @torch.no_grad()
    def sample(self, device, num_per_class=4):
        if not self.stats_dict:
            return None

        zs, ys, task_keys = [], [], []
        for (task_key, fine_id), stat in sorted(self.stats_dict.items()):
            mu = stat["mu"].to(device)
            sigma = stat["sigma"].to(device)
            z = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(
                num_per_class,
                self.feature_dim,
                device=device,
            )
            zs.append(z)
            ys.append(torch.full((num_per_class,), fine_id, dtype=torch.long, device=device))
            task_keys.extend([task_key] * num_per_class)

        return {
            "z_shared": torch.cat(zs, 0),
            "fine_ids": torch.cat(ys, 0),
            "task_keys": task_keys,
        }


@torch.no_grad()
def update_replay_buffer(model, loader, replay_buffer, device):
    model.eval()
    z_dict = defaultdict(list)

    for batch in loader:
        feats = [f.to(device) for f in batch["feats_list"]]
        fine_ids = batch["fine_ids"].to(device)
        task_keys, _, _ = infer_task_keys(model, batch, fine_ids)
        z_shared, _ = model.encode_shared(feats)

        for i, fine_id in enumerate(_to_int_list(fine_ids)):
            z_dict[(task_keys[i], fine_id)].append(z_shared[i:i + 1].detach())

    for (task_key, fine_id), zs in z_dict.items():
        replay_buffer.update(task_key, fine_id, torch.cat(zs, dim=0))


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
def register_sessions_until(model, session_manager, session_idx, device, cfg):
    for i in range(session_idx + 1):
        key = _session_task_key(model, session_manager, i, cfg)
        fine = _session_new_fine(session_manager, i)
        try:
            model.add_organ_task(key, fine, device)
        except TypeError:
            model.add_organ_task(key, fine)
    model.to(device)


@torch.no_grad()
def extract_base_subspace(model):
    """Session 0 (MIL full tuning) 후, base model의 principal subspace를 추출.

    visual_alignment_projector[0].weight (768×512)를 SVD하여
    base model이 z-space에서 의존하는 top-rank 방향을 얻습니다.
    이 방향들은 이후 LoRA adaptor의 orthogonal loss 기준으로 사용됩니다.
    """
    if not hasattr(model, "organ_task_adaptors"):
        return
    ref_weight = model.visual_alignment_projector[0].weight.data
    model.organ_task_adaptors.register_base_subspace(ref_weight)


def setup_session_model(model, session_manager, session_idx, device, cfg, refined_anchor_path=None):
    new_fine = _session_new_fine(session_manager, session_idx)
    opened_fine = _session_opened_fine(session_manager, session_idx)
    new_coarse = _session_coarse_labels(model, new_fine)
    opened_coarse = _session_coarse_labels(model, opened_fine)

    model.activate_nodes(opened_coarse, opened_fine, device, refresh_existing=False)
    apply_refined_text_anchors_to_model(model, refined_anchor_path, opened_fine, opened_coarse, device)
    model.expand_classifiers(opened_fine, device)

    current_task_key = _session_task_key(model, session_manager, session_idx, cfg)
    register_sessions_until(model, session_manager, session_idx, device, cfg)

    model._current_task_key = current_task_key
    model._current_session_idx = session_idx
    model.to(device)

    return {
        "task_key": current_task_key,
        "fine_mask": _safe_session_call(session_manager, "full_fine_mask", session_idx, device),
        "old_fine_mask": _safe_session_call(session_manager, "old_fine_mask", session_idx, device),
        "new_fine": new_fine,
        "new_coarse": new_coarse,
        "opened_fine": opened_fine,
        "opened_coarse": opened_coarse,
    }


def setup_session_eval(model, session_manager, session_idx, device, cfg=None):
    opened_fine = _session_opened_fine(session_manager, session_idx)
    opened_coarse = _session_coarse_labels(model, opened_fine)

    model.active_fine_labels = list(opened_fine)
    model.active_fine_to_local = {lab: i for i, lab in enumerate(model.active_fine_labels)}
    model.active_coarse_labels = list(opened_coarse)
    model.active_coarse_to_local = {lab: i for i, lab in enumerate(model.active_coarse_labels)}

    # Reset task registry so get_opened_task_keys() returns only sessions 0..session_idx.
    # Without this, a prior full-registration (e.g. joint_test setup) leaves stale entries
    # that cause KeyError when branch-mode forward tries to look up tasks whose labels are
    # not in active_fine_to_local.
    model.task_to_fine_labels = {}
    model.task_to_local_fine = {}
    model._current_session_idx = session_idx

    register_sessions_until(model, session_manager, session_idx, device, cfg or getattr(model, "cfg", {}))
    if hasattr(model, "_refresh_task_local_indices"):
        model._refresh_task_local_indices(device)
    model.to(device)

    return {
        "fine_mask": None,
        "opened_fine": opened_fine,
        "opened_coarse": opened_coarse,
    }


def configure_session_training_stage(model, current_task_key, session_idx, epoch, cfg):
    train_cfg = cfg.get("train", {})
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    stage = "base" if session_idx == 0 else ("warmup" if epoch < warmup_epochs else "adapt")
    prev_stage = getattr(model, "_session_stage", None)

    model.freeze_all_params()

    if stage == "base":
        if train_cfg.get("full_tune_base_session", True):
            model.unfreeze_shared_mil()
            model.unfreeze_visual_alignment_projector()
            if not model.use_hyperbolic_classifier():
                model.unfreeze_classifier()

    else:
        use_adaptor = getattr(model, "use_organ_task_adaptor", True)

        if use_adaptor:
            model.train_only_current_organ_task_adaptor(current_task_key)
        else:
            # Baselines (use_mil_lora: false): full-tune shared MIL for all sessions
            model.unfreeze_shared_mil()

        if not model.use_hyperbolic_classifier():
            model.unfreeze_classifier()

        if stage == "warmup":
            if train_cfg.get("train_projector_in_warmup", True):
                model.unfreeze_visual_alignment_projector()
        else:
            model.unfreeze_visual_alignment_projector()
            if use_adaptor and train_cfg.get("train_shared_mil_after_base", False):
                model.unfreeze_shared_mil()

    model._session_stage = stage
    return prev_stage != stage, stage
    

# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
def train_session(
    model,
    loader,
    optimizer,
    device,
    epoch,
    session_idx,
    replay_buffer=None,
    lambda_fine=1.0,
    lambda_replay=1.0,
    debug_print_freq=20,
    lambda_align=1.0,
    train_cfg=None,
    cfg=None,
    use_amp=False,
):
    train_cfg = train_cfg or {}
    cfg = cfg or {"train": train_cfg, "hyperbolic": {}}

    use_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    stats = RunningStats()

    lam_task = float(train_cfg.get("lambda_cls_task", lambda_fine))
    lam_global = float(train_cfg.get("lambda_cls_global", 0.3))
    lam_align = float(train_cfg.get("lambda_align", lambda_align))
    lam_sibling = float(train_cfg.get("lambda_sib", train_cfg.get("lambda_sibling", 0.0)))
    lam_weight = float(train_cfg.get("lambda_weight_anchor", 0.0))
    lam_replay = float(train_cfg.get("lambda_replay", lambda_replay))
    lam_orth = float(train_cfg.get("lambda_orth", 0.0))
    lam_hierarchy = float(train_cfg.get("lambda_hierarchy", 0.0))

    hyp_enabled = bool(_get_hyp_cfg(cfg).get("enabled", False))
    hyp_aux = hyp_enabled and not _use_hyperbolic_classifier(model, cfg)  # aux mode

    if hyp_enabled and not hyp_aux:
        lam_weight = 0.0  # full hyp mode: no linear head

    opened_task_keys = model.get_opened_task_keys()
    parent_ids = _build_active_fine_parent_ids(model, device)

    # Hierarchy info for coarse/task alignment (unchanged within epoch)
    hierarchy_info = None
    if lam_hierarchy > 0 and hyp_enabled:
        hierarchy_info = model.get_hierarchy_info(device)

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"[Sess {session_idx} | Epoch {epoch + 1:02d}]",
        leave=False,
    )

    for step, batch in pbar:
        feats = [f.to(device) for f in batch["feats_list"]]
        fine_global = batch["fine_ids"].to(device)
        fine_local = model.global_fine_to_local(fine_global)
        task_keys, _, _ = infer_task_keys(model, batch, fine_global)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            z_shared, _ = model.encode_shared(feats)

            out_global = _forward_global_from_z(model, z_shared, opened_task_keys, cfg)
            global_logits = out_global["fine_logits"]
            task_logits = _mask_logits_by_task_keys(model, global_logits, task_keys, targets_local=fine_local)

            loss_task = F.cross_entropy(task_logits, fine_local)
            if lam_global > 0 and len(opened_task_keys) > 1:
                loss_global = F.cross_entropy(global_logits, fine_local)
            else:
                loss_global = loss_task.new_tensor(0.0)

            if _use_merged_organ_task_adaptors(model):
                out_task = dict(out_global)
                out_task["fine_logits"] = task_logits
            else:
                out_task = _forward_task_from_z(model, z_shared, task_keys, cfg)

            fine_bank = model.get_active_fine_bank(device)
            fine_bank_tangent = _get_active_fine_bank_tangent(model, device, cfg)

            if lam_align > 0:
                if hyp_enabled:
                    loss_align = hyperbolic_bank_contrastive_loss(
                        out_task["raw_h_f"], fine_bank_tangent, fine_local, cfg,
                    )
                else:
                    loss_align = bank_contrastive_loss(
                        out_task["h_f"], fine_bank, fine_local,
                        tau=train_cfg.get("align_temperature", 0.07),
                    )
            else:
                loss_align = loss_task.new_tensor(0.0)

            if lam_sibling > 0:
                if hyp_enabled:
                    loss_sib = hyperbolic_sibling_margin_loss(
                        out_task["raw_h_f"], fine_bank_tangent, fine_local,
                        parent_ids, cfg,
                        margin=_get_hyp_cfg(cfg).get("sib_hyp_margin", 0.5),
                    )
                else:
                    loss_sib = sibling_margin_loss(
                        out_task["h_f"], fine_bank, fine_local, parent_ids,
                        margin=train_cfg.get("sib_cos_margin", train_cfg.get("sibling_margin", 0.1)),
                    )
            else:
                loss_sib = loss_task.new_tensor(0.0)

            if lam_weight > 0:
                loss_weight = classifier_weight_anchor_loss(model)
            else:
                loss_weight = loss_task.new_tensor(0.0)

            loss_replay = loss_task.new_tensor(0.0)
            loss_replay_cls = loss_task.new_tensor(0.0)
            loss_replay_align = loss_task.new_tensor(0.0)

            if replay_buffer is not None and replay_buffer.stats and lam_replay > 0:
                rep = replay_buffer.sample(
                    device,
                    num_per_class=train_cfg.get("replay_num_per_class", 4),
                )
                if rep is not None:
                    rep_local = model.global_fine_to_local(rep["fine_ids"])
                    rep_global = _forward_global_from_z(model, rep["z_shared"], opened_task_keys, cfg)
                    loss_replay_cls = F.cross_entropy(rep_global["fine_logits"], rep_local)

                    if _use_merged_organ_task_adaptors(model):
                        rep_task_logits = _mask_logits_by_task_keys(
                            model,
                            rep_global["fine_logits"],
                            rep["task_keys"],
                            targets_local=rep_local,
                        )
                        rep_task = dict(rep_global)
                        rep_task["fine_logits"] = rep_task_logits
                    else:
                        rep_task = _forward_task_from_z(model, rep["z_shared"], rep["task_keys"], cfg)

                    replay_align_w = float(train_cfg.get("lambda_replay_align", 1.0))

                    if replay_align_w > 0:
                        if hyp_enabled:
                            loss_replay_align = hyperbolic_bank_contrastive_loss(
                                rep_task["raw_h_f"], fine_bank_tangent, rep_local, cfg,
                            )
                        else:
                            loss_replay_align = bank_contrastive_loss(
                                rep_task["h_f"], fine_bank, rep_local,
                                tau=train_cfg.get("align_temperature", 0.07),
                            )

                    loss_replay = (
                        float(train_cfg.get("lambda_replay_cls", 1.0)) * loss_replay_cls
                        + replay_align_w * loss_replay_align
                    )

            if lam_orth > 0 and hasattr(model, "organ_task_adaptors"):
                base_key = opened_task_keys[0] if opened_task_keys else None
                loss_orth = model.organ_task_adaptors.orthogonal_loss(
                    exclude_keys=[base_key] if base_key is not None else None,
                )
            else:
                loss_orth = loss_task.new_tensor(0.0)

            if hierarchy_info is not None and lam_hierarchy > 0:
                loss_hier = hyperbolic_hierarchy_loss(
                    out_task["raw_h_f"], fine_local, hierarchy_info, _get_hyp_cfg(cfg),
                )
            else:
                loss_hier = loss_task.new_tensor(0.0)

            loss = (
                lam_task * loss_task
                + lam_global * loss_global
                + lam_align * loss_align
                + lam_sibling * loss_sib
                + lam_weight * loss_weight
                + lam_replay * loss_replay
                + lam_orth * loss_orth
                + lam_hierarchy * loss_hier
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        task_acc = _masked_accuracy(task_logits, fine_local)
        global_acc = _masked_accuracy(global_logits, fine_local)

        stats.update(
            loss_total=loss,
            loss_task=loss_task,
            loss_global=loss_global,
            loss_align=loss_align,
            loss_sib=loss_sib,
            loss_weight=loss_weight,
            loss_replay=loss_replay,
            loss_replay_cls=loss_replay_cls,
            loss_replay_align=loss_replay_align,
            loss_orth=loss_orth,
            loss_hier=loss_hier,
            task_acc=task_acc,
            global_acc=global_acc,
        )

        if (step + 1) % max(1, debug_print_freq) == 0:
            pbar.set_postfix(
                {
                    "L": f"{loss.item():.3f}",
                    "T": f"{loss_task.item():.2f}",
                    "G": f"{loss_global.item():.2f}",
                    "A": f"{loss_align.item():.2f}",
                    "S": f"{loss_sib.item():.2f}",
                    "R": f"{loss_replay.item():.2f}",
                }
            )

    avg = stats.as_dict()
    print(
        f"[Session {session_idx} | Epoch {epoch + 1:02d}] "
        f"L={avg.get('loss_total', 0):.4f} "
        f"TaskCE={avg.get('loss_task', 0):.4f} "
        f"GlobalCE={avg.get('loss_global', 0):.4f} "
        f"Align={avg.get('loss_align', 0):.4f} "
        f"Sib={avg.get('loss_sib', 0):.4f} "
        f"Replay={avg.get('loss_replay', 0):.4f} "
        f"Orth={avg.get('loss_orth', 0):.4f} "
        f"Hier={avg.get('loss_hier', 0):.4f} "
        f"TaskAcc={avg.get('task_acc', 0) * 100:.2f}% "
        f"GlobalAcc={avg.get('global_acc', 0) * 100:.2f}%"
    )
    return avg


@torch.no_grad()
def evaluate(model, loader, device, fine_mask=None, coarse_mask=None, return_details=False, eval_scope="task"):
    model.eval()
    cfg = getattr(model, "cfg", {})
    scope = str(eval_scope).lower()
    if scope in {"context", "local", "masked"}:
        scope = "task"
    if scope in {"opened", "open", "all"}:
        scope = "global"
    if scope not in {"task", "global"}:
        raise ValueError(f"Unknown eval_scope: {eval_scope}")

    true_all, pred_all, organs_all, tasks_all, datasets_all = [], [], [], [], []
    opened_task_keys = model.get_opened_task_keys()

    for batch in tqdm(loader, total=len(loader), desc=f"Evaluate[{scope}]", leave=False):
        feats = [f.to(device) for f in batch["feats_list"]]
        fine_global = batch["fine_ids"].to(device)
        fine_local = model.global_fine_to_local(fine_global)
        task_keys, organs, tasks = infer_task_keys(model, batch, fine_global)
        datasets = batch.get("datasets", ["unknown"] * len(feats))

        z_shared, _ = model.encode_shared(feats)
        out_global = _forward_global_from_z(model, z_shared, opened_task_keys, cfg)
        logits = out_global["fine_logits"]

        if scope == "task":
            logits = _mask_logits_by_task_keys(model, logits, task_keys, targets_local=fine_local)

        pred = logits.argmax(dim=1)

        true_all.append(fine_local.cpu())
        pred_all.append(pred.cpu())
        organs_all.extend(organs)
        tasks_all.extend(tasks)
        datasets_all.extend(datasets)

    y_true = torch.cat(true_all).long() if true_all else torch.empty(0, dtype=torch.long)
    y_pred = torch.cat(pred_all).long() if pred_all else torch.empty(0, dtype=torch.long)
    acc = 100.0 * float((y_true == y_pred).float().mean().item()) if y_true.numel() else 0.0

    if not return_details:
        return acc, 0.0

    return acc, 0.0, {
        "eval_scope": scope,
        "fine_true": y_true,
        "fine_pred": y_pred,
        "fine_labels": list(model.active_fine_labels),
        "context_organs": organs_all,
        "context_tasks": tasks_all,
        "context_datasets": datasets_all,
    }


def print_eval_details(details, title="Eval Details"):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)
    y_true = details["fine_true"].tolist()
    y_pred = details["fine_pred"].tolist()
    labels = details["fine_labels"]

    for i, name in enumerate(labels):
        idx = [k for k, t in enumerate(y_true) if t == i]
        if not idx:
            print(f"  {name:20s}: 0/0 (0.00%)")
            continue
        correct = sum(int(y_pred[k] == i) for k in idx)
        print(f"  {name:20s}: {correct:4d}/{len(idx):4d} ({100.0 * correct / len(idx):6.2f}%)")
    print("-" * 80)
