import torch
import torch.nn.functional as F

from models.hyperbolic import hyperbolic_logits_from_tangent, hyperbolic_distance_from_tangent



def bank_contrastive_loss(h, bank, targets, tau=0.07):
    h = F.normalize(h, dim=-1)
    bank = F.normalize(bank, dim=-1)
    if bank.size(0) == 1:
        return (1.0 - torch.sum(h * bank[0].unsqueeze(0), dim=-1)).mean()
    return F.cross_entropy(torch.matmul(h, bank.t()) / tau, targets)


def sibling_margin_loss(h, bank, targets, parent_ids, margin=0.1):
    if bank.size(0) <= 1:
        return h.new_tensor(0.0)

    h = F.normalize(h, dim=-1)
    bank = F.normalize(bank, dim=-1)
    sims = torch.matmul(h, bank.t())
    target_sims = sims.gather(1, targets.unsqueeze(1))
    sibling_mask = parent_ids.unsqueeze(0).eq(parent_ids[targets].unsqueeze(1))
    sibling_mask.scatter_(1, targets.unsqueeze(1), False)

    valid = sibling_mask.any(dim=1)
    if not valid.any():
        return h.new_tensor(0.0)

    loss = F.relu(sims - target_sims + float(margin)) * sibling_mask.float()
    loss = loss.sum(dim=1) / sibling_mask.float().sum(dim=1).clamp_min(1.0)
    return loss[valid].mean()

def hyperbolic_bank_contrastive_loss(u, bank_tangent, targets, hyp_cfg):
    if bank_tangent.size(0) <= 1:
        dist = hyperbolic_distance_from_tangent(u, bank_tangent, hyp_cfg)
        return dist.pow(2).mean()

    logits = hyperbolic_logits_from_tangent(u, bank_tangent, hyp_cfg)
    return F.cross_entropy(logits, targets)


def hyperbolic_sibling_margin_loss(u, bank_tangent, targets, parent_ids, hyp_cfg, margin=None):
    if bank_tangent.size(0) <= 1:
        return u.new_tensor(0.0)

    dist = hyperbolic_distance_from_tangent(u, bank_tangent, hyp_cfg)

    target_dist = dist.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_parent = parent_ids[targets]

    sibling_mask = parent_ids.unsqueeze(0).eq(target_parent.unsqueeze(1))
    sibling_mask.scatter_(1, targets.unsqueeze(1), False)

    valid = sibling_mask.any(dim=1)
    if not valid.any():
        return u.new_tensor(0.0)

    sib_dist = dist.masked_fill(~sibling_mask, 1e9).min(dim=1).values

    if margin is None:
        margin = float(hyp_cfg.get("sib_hyp_margin", 0.5))

    return F.relu(target_dist - sib_dist + margin)[valid].mean()


def hyperbolic_hierarchy_loss(u_img, fine_targets, hierarchy_info, hyp_cfg):
    """Align image embeddings with ancestral coarse/task nodes.

    Uses distance-based CE at each hierarchy level:
      image → parent coarse node  (coarse-level classification)
      image → parent task node    (task-level classification)

    Args:
        u_img: [B, D] image tangent vectors (projector output).
        fine_targets: [B] local fine label indices.
        hierarchy_info: dict from model.get_hierarchy_info().
        hyp_cfg: dict or config-like with hyperbolic settings.
    """
    coarse_tangent = hierarchy_info["coarse_tangent"]
    task_tangent = hierarchy_info["task_tangent"]
    fine_to_coarse = hierarchy_info["fine_to_coarse_idx"]
    fine_to_task = hierarchy_info["fine_to_task_idx"]

    loss = u_img.new_tensor(0.0)
    n = 0

    # Coarse-level: image → parent organ
    if coarse_tangent.size(0) > 1:
        coarse_logits = hyperbolic_logits_from_tangent(u_img, coarse_tangent, hyp_cfg)
        coarse_targets = fine_to_coarse[fine_targets]
        loss = loss + F.cross_entropy(coarse_logits, coarse_targets)
        n += 1

    # Task-level: image → parent task pseudo-node
    if task_tangent.size(0) > 1:
        task_logits = hyperbolic_logits_from_tangent(u_img, task_tangent, hyp_cfg)
        task_targets = fine_to_task[fine_targets]
        loss = loss + F.cross_entropy(task_logits, task_targets)
        n += 1

    return loss / max(n, 1)