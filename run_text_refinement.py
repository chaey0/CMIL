from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from text_anchor_refinement import (
    TextRefineConfig,
    normalize,
    refine_text_anchors,
    refine_text_anchors_hyperbolic,
)
from models.text_encoder import KEEPEncoder
from utils.taxonomy_parser import TaxonomyParser

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["__config_path__"] = str(path)
    return cfg


def pick(cfg: dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def unique(xs: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(list(xs)))


def _as_list(sess: dict, keys: Sequence[str]) -> List[str]:
    for key in keys:
        if key in sess and sess[key] is not None:
            return unique(list(sess[key]))
    raise KeyError(f"Session needs one of {list(keys)}")


def normalize_task(task) -> str:
    task = "default" if task is None else str(task)
    return "default" if task in {"", "None", "none", "null"} else task


def make_task_group(organ: str, task: str) -> str:
    task = normalize_task(task)
    return str(organ) if task == "default" else f"{organ}/{task}"


def setup_device(gpu: Optional[str], device_arg: str) -> str:
    if gpu is not None and str(gpu).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_arg


# -----------------------------------------------------------------------------
# Config / session helpers
# -----------------------------------------------------------------------------

def load_sessions(cfg: dict) -> List[dict]:
    """Load organ-task sessions and cumulative opened fine labels."""
    raw_sessions = cfg.get("sessions", [])
    if not raw_sessions:
        raise ValueError("YAML must contain sessions with fine_labels or label_ids.")

    out: List[dict] = []
    cumulative_opened: List[str] = []
    for i, sess in enumerate(raw_sessions):
        name = sess.get("name", sess.get("session_name", f"session_{i}"))
        organ = sess.get("organ", sess.get("organ_id", sess.get("coarse", sess.get("coarse_label"))))
        task = normalize_task(sess.get("task", sess.get("task_id", "default")))
        new_labels = _as_list(sess, ["fine_labels", "label_ids", "new_fine_labels", "opened_fine_labels"])

        if organ is None:
            organ = sess.get("coarse_label", None)
        if organ is None:
            organ = "UNKNOWN"

        for label in new_labels:
            if label not in cumulative_opened:
                cumulative_opened.append(label)

        out.append(
            {
                "idx": i,
                "name": name,
                "organ": str(organ),
                "task": task,
                "task_group": make_task_group(str(organ), task),
                "new": unique(new_labels),
                "opened": unique(cumulative_opened),
                "format": "organ_task_incremental",
            }
        )
    return out


def make_fine_to_task_group(sessions: Sequence[dict], taxonomy) -> Dict[str, str]:
    """Fine label -> organ-task group for sibling separation."""
    mapping: Dict[str, str] = {}
    for sess in sessions:
        for label in sess["new"]:
            parent = taxonomy.fine_to_coarse.get(label, sess["organ"])
            mapping.setdefault(label, make_task_group(parent, sess.get("task", "default")))
    return mapping


def build_refine_cfg(cfg: dict) -> TextRefineConfig:
    raw = cfg.get("text_refine", cfg.get("text_refinement", {})) or {}
    aliases = {
        "num_steps": "steps",
        "n_steps": "steps",
        "learning_rate": "lr",
        "log_every": "print_every",
        # sibling key backwards-compat aliases (old name → new name)
        "sibling_upper": "sib_cos_upper",
        "sibling_max_cos": "sib_cos_upper",
        "sibling_margin_hyp": "sib_hyp_margin",
        "sibling_margin": "sib_hyp_margin",
        "lambda_sibling": "lambda_sib",
        "lambda_parent": "lambda_par",
        # cross-coarse hard-negative aliases
        "lambda_cross_coarse": "lambda_cross",
        "cross_upper": "cross_cos_upper",
        "cross_top_k": "cross_topk",
    }
    valid = {f.name for f in fields(TextRefineConfig)}
    clean = {aliases.get(k, k): v for k, v in raw.items() if aliases.get(k, k) in valid}
    return TextRefineConfig(**clean)


def select_run_sessions(
    sessions: Sequence[dict],
    session_idx: Optional[int],
    has_prev_refined: bool,
    from_scratch_refinement: bool,
) -> List[dict]:
    if session_idx is None:
        return list(sessions)
    if session_idx < 0 or session_idx >= len(sessions):
        raise IndexError(f"session_idx={session_idx} out of range [0, {len(sessions)-1}]")
    if from_scratch_refinement or not has_prev_refined:
        return list(sessions[: session_idx + 1])
    return [sessions[session_idx]]


# -----------------------------------------------------------------------------
# Text encoding / taxonomy helpers
# -----------------------------------------------------------------------------

def get_text(taxonomy, label: str, kind: str) -> str:
    method = f"get_{kind}_text"
    table = f"{kind}_text"
    if hasattr(taxonomy, method):
        return getattr(taxonomy, method)(label)
    if hasattr(taxonomy, table) and isinstance(getattr(taxonomy, table), dict):
        return getattr(taxonomy, table).get(label, label.replace("_", " "))
    return label.replace("_", " ")


def make_encoder(cfg: dict, device: str):
    name = pick(cfg, "text_encoder.model_name", pick(cfg, "text_encoder.name", "Astaxanthin/KEEP"))
    for kwargs in ({"model_name": name, "device": device}, {"model_name": name}, {"name": name}, {}):
        try:
            return KEEPEncoder(**kwargs)
        except TypeError:
            pass
    return KEEPEncoder(name)


def encode_texts(encoder, texts: Sequence[str], device: str) -> torch.Tensor:
    for method in ("encode", "encode_texts", "encode_labels", "get_text_features"):
        if hasattr(encoder, method):
            x = getattr(encoder, method)(list(texts))
            if isinstance(x, tuple):
                x = x[0]
            return normalize(torch.as_tensor(x, dtype=torch.float32, device=device))
    raise AttributeError("KEEPEncoder has no encode-like method")


def load_previous_refined_bank(
    prev_path: Optional[str],
    base_bank: torch.Tensor,
    used_fine: Sequence[str],
) -> tuple[torch.Tensor, bool]:
    if prev_path is None or str(prev_path).strip() == "":
        return base_bank.clone(), False

    prev_path = Path(prev_path)
    if not prev_path.exists():
        raise FileNotFoundError(f"--prev_refined_anchor_path not found: {prev_path}")

    artifact = torch.load(prev_path, map_location=base_bank.device)
    if "fine_labels" not in artifact or "refined_fine_bank" not in artifact:
        raise KeyError("Previous artifact must contain 'fine_labels' and 'refined_fine_bank'.")

    prev_labels = list(artifact["fine_labels"])
    prev_bank = normalize(torch.as_tensor(artifact["refined_fine_bank"], dtype=torch.float32, device=base_bank.device))
    bank = base_bank.clone()
    copied = 0
    for dst_idx, label in enumerate(used_fine):
        if label in prev_labels:
            bank[dst_idx].copy_(prev_bank[prev_labels.index(label)])
            copied += 1

    print(f"[PrevRefined] loaded: {prev_path}")
    print(f"[PrevRefined] copied anchors: {copied}/{len(used_fine)}")
    return normalize(bank), True


# -----------------------------------------------------------------------------
# Plot / save helpers
# -----------------------------------------------------------------------------

def cos_matrix(x: torch.Tensor) -> np.ndarray:
    x = normalize(x).detach().cpu()
    return (x @ x.T).numpy()


def save_csv_matrix(path: Path, labels: Sequence[str], mat: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + list(labels))
        for label, row in zip(labels, mat):
            w.writerow([label] + [float(v) for v in row])


def save_similarity_report(labels: Sequence[str], keep: torch.Tensor, refined: torch.Tensor, out_dir: Path) -> None:
    keep_sim = cos_matrix(keep)
    refined_sim = cos_matrix(refined)
    diff = refined_sim - keep_sim
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(labels) * 0.75), 5), constrained_layout=True)
    items = [
        ("KEEP initial similarity", keep_sim, 1.0),
        ("Final refined similarity", refined_sim, 1.0),
        ("Final refined - KEEP initial", diff, max(0.05, float(np.abs(diff).max()))),
    ]
    for ax, (title, mat, lim) in zip(axes, items):
        im = ax.imshow(mat, cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_dir / "final_similarity_keep_vs_refined.png", dpi=220)
    plt.close(fig)
    save_csv_matrix(out_dir / "final_similarity_keep.csv", labels, keep_sim)
    save_csv_matrix(out_dir / "final_similarity_refined.csv", labels, refined_sim)
    save_csv_matrix(out_dir / "final_similarity_diff.csv", labels, diff)


def reduce_mds(emb: np.ndarray, dim: int, seed: int = 0) -> np.ndarray:
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    dist = np.clip(1.0 - emb @ emb.T, 0.0, 2.0)
    try:
        from sklearn.manifold import MDS

        try:
            mds = MDS(
                n_components=dim,
                dissimilarity="precomputed",
                random_state=seed,
                n_init=4,
                max_iter=600,
                normalized_stress="auto",
            )
        except TypeError:
            mds = MDS(n_components=dim, dissimilarity="precomputed", random_state=seed, n_init=4, max_iter=600)
        return mds.fit_transform(dist)
    except Exception:
        n = dist.shape[0]
        j = np.eye(n) - np.ones((n, n)) / n
        b = -0.5 * j @ (dist ** 2) @ j
        vals, vecs = np.linalg.eigh(b)
        order = np.argsort(vals)[::-1][:dim]
        return vecs[:, order] * np.sqrt(np.maximum(vals[order], 0.0))


def save_mds_side_by_side(
    labels: Sequence[str],
    parents: Sequence[str],
    keep: torch.Tensor,
    refined: torch.Tensor,
    out_png: Path,
    dim: int,
) -> None:
    all_emb = torch.cat([keep, refined], dim=0).detach().cpu().numpy()
    coords = reduce_mds(all_emb, dim=dim)
    p0, p1 = coords[: len(labels)], coords[len(labels) :]
    parent_list = unique(parents)
    color_idx = {p: i for i, p in enumerate(parent_list)}
    colors = [color_idx[p] for p in parents]

    if dim == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        for ax, title, pts in [(axes[0], "KEEP initial", p0), (axes[1], "Final refined", p1)]:
            ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap="tab20", s=45)
            for x, y, label in zip(pts[:, 0], pts[:, 1], labels):
                ax.text(x, y, label, fontsize=7)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        fig = plt.figure(figsize=(15, 6), constrained_layout=True)
        all_pts = np.concatenate([p0, p1], axis=0)
        mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        pad = 0.08 * span
        lims = [(mins[i] - pad[i], maxs[i] + pad[i]) for i in range(3)]
        for k, (title, pts) in enumerate([("KEEP initial", p0), ("Final refined", p1)], start=1):
            ax = fig.add_subplot(1, 2, k, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, cmap="tab20", s=40, depthshade=True)
            for x, y, z, label in zip(pts[:, 0], pts[:, 1], pts[:, 2], labels):
                ax.text(x, y, z, label, fontsize=6)
            ax.set_title(title)
            ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.grid(True)
            ax.view_init(elev=22, azim=-60)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_rows(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_agglomerative_report(
    labels: Sequence[str],
    parents: Sequence[str],
    keep: torch.Tensor,
    refined: torch.Tensor,
    out_dir: Path,
    method: str = "average",
) -> None:
    try:
        from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
        from scipy.spatial.distance import squareform
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except Exception as e:
        print(f"[Warn] Skip agglomerative tree clustering: {e}")
        return

    def condensed(x: torch.Tensor) -> np.ndarray:
        z = normalize(x).detach().cpu().numpy().astype(np.float64)
        dist = np.clip(1.0 - z @ z.T, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        return squareform(dist, checks=False)

    tree_dir = out_dir / "agglomerative_tree"
    tree_dir.mkdir(parents=True, exist_ok=True)
    parent_ids = np.array([{p: i for i, p in enumerate(unique(parents))}[p] for p in parents])
    metrics = {"method": method, "distance": "cosine_distance = 1 - cosine_similarity", "keep": {}, "refined": {}}

    fig, axes = plt.subplots(1, 2, figsize=(max(15, len(labels) * 0.75), 6), constrained_layout=True)
    for ax, name, emb in [(axes[0], "keep", keep), (axes[1], "refined", refined)]:
        z = linkage(condensed(emb), method=method)
        dendrogram(z, labels=list(labels), leaf_rotation=90, leaf_font_size=7, ax=ax, color_threshold=None)
        ax.set_title(f"{name.upper()} agglomerative tree")
        ax.set_ylabel("cosine distance")
        cluster_ids = fcluster(z, t=max(1, len(unique(parents))), criterion="maxclust")
        metrics[name] = {
            "ari_vs_parent": float(adjusted_rand_score(parent_ids, cluster_ids)),
            "nmi_vs_parent": float(normalized_mutual_info_score(parent_ids, cluster_ids)),
        }
    fig.savefig(tree_dir / "agglomerative_tree_keep_vs_refined.png", dpi=220)
    plt.close(fig)
    with open(tree_dir / "agglomerative_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# -----------------------------------------------------------------------------
# Hyperbolic visualizations
# -----------------------------------------------------------------------------

def _expmap0_poincare_np(u: np.ndarray, c: float = 1.0, max_norm: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """Tangent vector at origin → Poincaré ball point (numpy)."""
    sc = c ** 0.5
    norm = np.linalg.norm(u, axis=-1, keepdims=True).clip(eps)
    factor = np.clip(max_norm / norm, a_min=None, a_max=1.0)
    u = u * factor
    norm = np.linalg.norm(u, axis=-1, keepdims=True).clip(eps)
    x = np.tanh(sc * norm) * u / (sc * norm)
    max_ball_norm = (1.0 - 1e-5) / sc
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True).clip(eps)
    return x * np.clip(max_ball_norm / x_norm, a_min=None, a_max=1.0)


def _poincare_distance_matrix(x: np.ndarray, c: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    """Pairwise geodesic distance on Poincaré ball (numpy)."""
    x2 = (x * x).sum(axis=-1, keepdims=True)           # [N, 1]
    diff2 = np.sum((x[:, None] - x[None]) ** 2, axis=-1)  # [N, N]
    denom = np.clip(1.0 - c * x2, eps, None) * np.clip(1.0 - c * x2.T, eps, None)
    z = 1.0 + 2.0 * c * diff2 / denom
    return np.arccosh(np.clip(z, 1.0 + eps, None)) / (c ** 0.5)


def _to_tangent(bank: torch.Tensor, scale: float = 0.7) -> np.ndarray:
    """Normalized bank → tangent vector."""
    b = torch.nn.functional.normalize(bank, dim=-1) * scale
    return b.detach().cpu().numpy().astype(np.float64)


def save_hyperbolic_distance_report(
    labels: Sequence[str],
    keep: torch.Tensor,
    refined: torch.Tensor,
    out_dir: Path,
    hyp_cfg: dict,
) -> None:
    """Hyperbolic distance matrix heatmap (analogous to cosine similarity report)."""
    c = float(hyp_cfg.get("curvature", 1.0))
    scale = float(hyp_cfg.get("fine_tangent_scale", hyp_cfg.get("text_tangent_scale", 0.7)))
    max_norm = float(hyp_cfg.get("max_tangent_norm", 1.0))

    keep_t = _to_tangent(keep, scale)
    ref_t = _to_tangent(refined, scale)

    keep_poin = _expmap0_poincare_np(keep_t, c=c, max_norm=max_norm)
    ref_poin = _expmap0_poincare_np(ref_t, c=c, max_norm=max_norm)

    keep_dist = _poincare_distance_matrix(keep_poin, c=c)
    ref_dist = _poincare_distance_matrix(ref_poin, c=c)
    diff = ref_dist - keep_dist

    vmax = max(float(np.max(keep_dist)), float(np.max(ref_dist)), 0.1)
    diff_lim = max(0.05, float(np.abs(diff).max()))

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(labels) * 0.75), 5), constrained_layout=True)
    items = [
        ("KEEP hyperbolic distance", keep_dist, 0.0, vmax, "YlOrRd"),
        ("Refined hyperbolic distance", ref_dist, 0.0, vmax, "YlOrRd"),
        ("Refined - KEEP (distance diff)", diff, -diff_lim, diff_lim, "coolwarm"),
    ]
    for ax, (title, mat, vmin, vm, cmap) in zip(axes, items):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vm)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out_dir / "final_hyperbolic_distance_keep_vs_refined.png", dpi=220)
    plt.close(fig)
    save_csv_matrix(out_dir / "final_hyperbolic_distance_keep.csv", labels, keep_dist)
    save_csv_matrix(out_dir / "final_hyperbolic_distance_refined.csv", labels, ref_dist)
    print(f"  [Hyp] Saved hyperbolic distance report")


def save_poincare_disk_plot(
    labels: Sequence[str],
    parents: Sequence[str],
    keep: torch.Tensor,
    refined: torch.Tensor,
    out_path: Path,
    hyp_cfg: dict,
    taxonomy=None,
    sessions: Sequence[dict] = None,
) -> None:
    """Poincaré disk visualization with 3-level hierarchy (coarse/task/fine)."""
    c = float(hyp_cfg.get("curvature", 1.0))
    fine_scale = float(hyp_cfg.get("fine_tangent_scale", hyp_cfg.get("text_tangent_scale", 0.7)))
    task_scale = float(hyp_cfg.get("task_tangent_scale", 0.5))
    coarse_scale = float(hyp_cfg.get("coarse_tangent_scale", 0.3))
    max_norm = float(hyp_cfg.get("max_tangent_norm", 1.0))

    # --- Build 3 levels ---
    # Fine nodes
    ref_t = _to_tangent(refined, fine_scale)
    ref_poin = _expmap0_poincare_np(ref_t, c=c, max_norm=max_norm)

    # Coarse (organ) nodes
    coarse_list = unique(parents)
    coarse_embs = []
    for organ in coarse_list:
        child_idx = [i for i, p in enumerate(parents) if p == organ]
        child_bank = torch.stack([refined[i] for i in child_idx])
        coarse_embs.append(child_bank.mean(dim=0))
    coarse_bank = torch.stack(coarse_embs)
    coarse_t = _to_tangent(coarse_bank, coarse_scale)
    coarse_poin = _expmap0_poincare_np(coarse_t, c=c, max_norm=max_norm)

    # Task pseudo-nodes (one per organ/task group)
    task_labels, task_embs, task_parents = [], [], []
    if sessions:
        for sess in sessions:
            tg = sess.get("task_group", sess.get("name", ""))
            organ = sess.get("organ", "")
            fine_labs = sess.get("new", sess.get("fine_labels", []))
            child_idx = [i for i, l in enumerate(labels) if l in fine_labs]
            if not child_idx:
                continue
            child_bank = torch.stack([refined[i] for i in child_idx])
            task_labels.append(tg)
            task_embs.append(child_bank.mean(dim=0))
            task_parents.append(organ)

    if task_embs:
        task_bank = torch.stack(task_embs)
        task_t = _to_tangent(task_bank, task_scale)
        task_poin = _expmap0_poincare_np(task_t, c=c, max_norm=max_norm)
    else:
        task_poin = np.empty((0, ref_poin.shape[1]))

    # --- PCA for direction, actual Poincaré norm for radius ---
    from sklearn.decomposition import PCA
    all_poin = np.concatenate([coarse_poin, task_poin, ref_poin], axis=0)

    # Actual Poincaré norms (= true hyperbolic radius indicator)
    all_norms = np.linalg.norm(all_poin, axis=-1, keepdims=True).clip(1e-8)

    # PCA on unit-direction vectors only → preserves angular structure
    all_unit = all_poin / all_norms
    pca = PCA(n_components=2, random_state=0)
    all_dir_2d = pca.fit_transform(all_unit)

    # Normalize 2D directions to unit vectors, then scale by actual Poincaré norm
    dir_norms = np.linalg.norm(all_dir_2d, axis=-1, keepdims=True).clip(1e-8)
    all_2d = (all_dir_2d / dir_norms) * all_norms

    n_c, n_t, n_f = len(coarse_list), len(task_labels), len(labels)
    pts_coarse = all_2d[:n_c]
    pts_task = all_2d[n_c:n_c + n_t]
    pts_fine = all_2d[n_c + n_t:]

    # Rescale so max norm < 0.95 (visual fit inside unit disk)
    max_r = max(np.linalg.norm(all_2d, axis=1).max(), 1e-6)
    sf = 0.92 / max_r
    pts_coarse, pts_task, pts_fine = pts_coarse * sf, pts_task * sf, pts_fine * sf

    # --- Colors ---
    cmap = plt.cm.get_cmap("tab20", len(coarse_list))
    organ_color = {p: cmap(i) for i, p in enumerate(coarse_list)}

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), constrained_layout=True)
    fig.suptitle("Poincaré Disk — Hierarchical Text Anchors", fontsize=13)

    # Disk boundary + hierarchy rings at actual scaled radii
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#888888", lw=1.5, ls="--"))

    # Compute visual radii for each level (Poincaré norm * scale_factor)
    def _poincare_norm(tangent_scale):
        """Poincaré ball norm for a tangent vector of norm = tangent_scale."""
        u = np.ones((1, 2)) * tangent_scale / (2**0.5)
        return float(np.linalg.norm(_expmap0_poincare_np(u, c=c, max_norm=max_norm)))

    r_coarse = _poincare_norm(coarse_scale) * sf
    r_task = _poincare_norm(task_scale) * sf
    r_fine = _poincare_norm(fine_scale) * sf

    for r, lbl, clr in [(r_coarse, f"coarse r={coarse_scale}", "#aaaaff"),
                         (r_task, f"task r={task_scale}", "#aaffaa"),
                         (r_fine, f"fine r={fine_scale}", "#ffaaaa")]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=clr, lw=1.0, ls=":"))

    # Coarse nodes (diamonds, large)
    for i, (x, y) in enumerate(pts_coarse):
        ax.scatter(x, y, marker="D", c=[organ_color[coarse_list[i]]], s=160,
                   zorder=5, edgecolors="black", linewidths=1.0)
        ax.annotate(coarse_list[i], (x, y), fontsize=8, fontweight="bold",
                    ha="center", va="bottom", xytext=(0, 8), textcoords="offset points")

    # Task pseudo-nodes (squares, medium)
    for i, (x, y) in enumerate(pts_task):
        parent = task_parents[i] if i < len(task_parents) else coarse_list[0]
        ax.scatter(x, y, marker="s", c=[organ_color.get(parent, "gray")], s=90,
                   zorder=4, edgecolors="black", linewidths=0.7, alpha=0.85)
        short_name = task_labels[i].split("/")[-1] if "/" in task_labels[i] else task_labels[i]
        ax.annotate(short_name, (x, y), fontsize=6, fontstyle="italic",
                    ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")

    # Fine nodes (circles, small)
    for i, (x, y) in enumerate(pts_fine):
        ax.scatter(x, y, marker="o", c=[organ_color.get(parents[i], "gray")], s=50,
                   zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(labels[i], (x, y), fontsize=5.5, ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points")

    # Origin
    ax.scatter(0, 0, marker="+", c="black", s=40, zorder=6)

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend: organs + hierarchy level markers
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=organ_color[p],
               markersize=8, label=p) for p in coarse_list]
    handles += [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="gray", markeredgecolor="black",
                   markersize=9, label=f"Coarse (r={coarse_scale})"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="black",
                   markersize=7, label=f"Task (r={task_scale})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="white",
                   markersize=6, label=f"Fine (r={fine_scale})"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"  [Hyp] Saved Poincaré disk plot (3-level hierarchy): {out_path.name}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="YAML-driven KEEP text-anchor refinement")
    ap.add_argument("--gpu", type=str, default=None, help="GPU id(s) to use")
    ap.add_argument("--config", type=str, default="configs/ALL_15epoch.yaml")
    ap.add_argument("--taxonomy_json", type=str, default="data/taxonomy_all.json")
    ap.add_argument("--session_idx", type=int, default=None)
    ap.add_argument("--prev_refined_anchor_path", type=str, default=None)
    ap.add_argument("--from_scratch_refinement", action="store_true")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_verbose", action="store_true")
    ap.add_argument("--no_tree_clustering", action="store_true")
    ap.add_argument("--tree_linkage", type=str, default="average", choices=["average", "complete", "single"])
    ap.add_argument("--freeze_old_anchors", action="store_true")
    ap.add_argument("--refine_new_only", action="store_true")

    # Hyperbolic refinement override (overrides hyperbolic.enabled in YAML)
    hyp_grp = ap.add_mutually_exclusive_group()
    hyp_grp.add_argument("--hyp", dest="hyp_enabled", action="store_true", default=None,
                         help="Force hyperbolic (Poincaré) text refinement regardless of YAML")
    hyp_grp.add_argument("--no_hyp", dest="hyp_enabled", action="store_false",
                         help="Force Euclidean text refinement regardless of YAML")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    # CLI override: --hyp / --no_hyp override hyperbolic.enabled in YAML
    if getattr(args, "hyp_enabled", None) is not None:
        cfg.setdefault("hyperbolic", {})["enabled"] = bool(args.hyp_enabled)

    refine_cfg = build_refine_cfg(cfg)
    device = setup_device(args.gpu, args.device)

    if args.freeze_old_anchors or args.refine_new_only:
        refine_cfg.old_delta_scale = 0.0

    out_dir = Path(args.out_dir or pick(cfg, "text_refine.out_dir", "outputs_text_refinement"))
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = TaxonomyParser(args.taxonomy_json)
    sessions = load_sessions(cfg)
    fine_to_task_group = make_fine_to_task_group(sessions, taxonomy)

    used_fine = unique([label for sess in sessions for label in sess["opened"]])
    if hasattr(taxonomy, "fine_labels"):
        used_fine = [label for label in taxonomy.fine_labels if label in set(used_fine)]
    used_coarse = unique([taxonomy.fine_to_coarse[label] for label in used_fine])
    parent_of_fine = [taxonomy.fine_to_coarse[label] for label in used_fine]

    print(f"[YAML] {cfg['__config_path__']}")
    print(f"[Taxonomy] {args.taxonomy_json}")
    print(f"[Labels] coarse={len(used_coarse)} fine={len(used_fine)}")
    print(f"[RefineConfig] {refine_cfg}")
    print("[Sibling] task-aware group = organ/task")

    encoder = make_encoder(cfg, device)
    keep_bank = encode_texts(encoder, [get_text(taxonomy, x, "fine") for x in used_fine], device)
    parent_bank = encode_texts(encoder, [get_text(taxonomy, x, "coarse") for x in used_coarse], device)
    bank, has_prev_refined = load_previous_refined_bank(args.prev_refined_anchor_path, keep_bank, used_fine)

    # Task text embeddings: one per unique task group across all sessions
    hyp_cfg = cfg.get("hyperbolic", {})
    use_hyp_refine = bool(hyp_cfg.get("enabled", False))
    task_group_labels = unique([sess["task_group"] for sess in sessions])

    # Build task_group -> fine_labels mapping (for accurate task_axis lookup)
    tg_to_fine_labels: Dict[str, list] = {}
    for sess in sessions:
        tg = sess["task_group"]
        tg_to_fine_labels.setdefault(tg, [])
        for fl in sess.get("new", []):
            if fl not in tg_to_fine_labels[tg]:
                tg_to_fine_labels[tg].append(fl)

    task_bank_dict: Dict[str, torch.Tensor] = {}
    if use_hyp_refine and hasattr(taxonomy, "get_task_text"):
        task_texts = [
            taxonomy.get_task_text(tg, fine_labels=tg_to_fine_labels.get(tg))
            for tg in task_group_labels
        ]
        task_embs = encode_texts(encoder, task_texts, device)   # [T, D]
        for tg, emb in zip(task_group_labels, task_embs):
            task_bank_dict[tg] = emb                            # unit vector

    label_to_idx = {label: i for i, label in enumerate(used_fine)}
    run_sessions = select_run_sessions(
        sessions=sessions,
        session_idx=args.session_idx,
        has_prev_refined=has_prev_refined,
        from_scratch_refinement=args.from_scratch_refinement,
    )

    all_logs: Dict[str, object] = {}
    for sess in run_sessions:
        opened = [x for x in sess["opened"] if x in label_to_idx]
        current_new = [x for x in sess.get("new", []) if x in label_to_idx]
        prev_opened = [] if sess["idx"] == 0 else [x for x in sessions[sess["idx"] - 1]["opened"] if x in label_to_idx]
        prev_set = set(prev_opened)

        new_labels = [x for x in current_new if x not in prev_set]
        if not new_labels:
            new_labels = [x for x in opened if x not in prev_set]
        old_labels = [x for x in opened if x in prev_set]

        if not new_labels:
            print(f"[Skip] {sess['name']}: no new fine labels")
            continue

        affected_groups = {fine_to_task_group[x] for x in new_labels}
        train_labels = [x for x in opened if fine_to_task_group.get(x) in affected_groups]
        old_local = [x for x in train_labels if x in set(old_labels)]

        before = bank.clone()
        print(f"\n[Session {sess['idx']:02d}] {sess['name']}")
        print(f"  organ        : {sess.get('organ')}")
        print(f"  task         : {sess.get('task')}")
        print(f"  task_group   : {sess.get('task_group')}")
        print("  mode         : " + ("incremental-current-only" if has_prev_refined and args.session_idx is not None and not args.from_scratch_refinement else "from-scratch-prefix"))
        print(f"  refine_space : {'hyperbolic (Poincare)' if use_hyp_refine else 'euclidean (cosine)'}")
        print(f"  new          : {new_labels}")
        print(f"  update_local : {train_labels}")
        print(f"  old_local    : {old_local}")

        if use_hyp_refine:
            # Build parent unit direction dict: task_group + coarse keys
            parent_unit_dirs: Dict[str, torch.Tensor] = dict(task_bank_dict)
            for coarse_lab, emb in zip(used_coarse, parent_bank):
                parent_unit_dirs[coarse_lab] = emb  # fallback for coarse
            # fine -> immediate parent = task_group
            bank_tangent, logs = refine_text_anchors_hyperbolic(
                base_directions=normalize(before),
                fine_labels=used_fine,
                fine_to_immediate_parent=fine_to_task_group,
                train_labels=train_labels,
                parent_unit_dirs=parent_unit_dirs,
                hyp_cfg=hyp_cfg,
                old_labels=old_local,
                cfg=refine_cfg,
                verbose=not args.no_verbose,
            )
            bank = bank_tangent  # tangent vectors (fine_scale * direction)
        else:
            bank, logs = refine_text_anchors(
                base_bank=before,
                fine_labels=used_fine,
                fine_to_coarse=taxonomy.fine_to_coarse,
                train_labels=train_labels,
                parent_bank=parent_bank,
                coarse_labels=used_coarse,
                old_labels=old_local,
                fine_to_sibling_group=fine_to_task_group,
                cfg=refine_cfg,
                verbose=not args.no_verbose,
            )
        all_logs[f"session_{sess['idx']:02d}"] = logs

    save_similarity_report(used_fine, keep_bank, bank, out_dir)
    save_mds_side_by_side(used_fine, parent_of_fine, keep_bank, bank, out_dir / "final_mds2d_keep_vs_refined.png", dim=2)
    save_mds_side_by_side(used_fine, parent_of_fine, keep_bank, bank, out_dir / "final_mds3d_keep_vs_refined.png", dim=3)
    if not args.no_tree_clustering:
        save_agglomerative_report(used_fine, parent_of_fine, keep_bank, bank, out_dir, method=args.tree_linkage)

    # Hyperbolic visualizations (auto-detect from config)
    hyp_cfg = cfg.get("hyperbolic", {})
    if bool(hyp_cfg.get("enabled", False)):
        print("\n[Hyperbolic] Generating Poincaré disk & distance visualizations...")
        save_hyperbolic_distance_report(used_fine, keep_bank, bank, out_dir, hyp_cfg)
        try:
            save_poincare_disk_plot(
                used_fine, parent_of_fine, keep_bank, bank,
                out_dir / "final_poincare_disk_hierarchy.png", hyp_cfg,
                taxonomy=taxonomy, sessions=sessions,
            )
        except ImportError:
            print("  [Warn] sklearn not available, skipping Poincaré disk plot")

    with open(out_dir / "refinement_logs.json", "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2)

    anchor_space = "poincare_tangent" if use_hyp_refine else "euclidean"
    torch.save(
        {
            "fine_labels": used_fine,
            "coarse_labels": used_coarse,
            "fine_to_coarse": {x: taxonomy.fine_to_coarse[x] for x in used_fine},
            "fine_to_sibling_group": {x: fine_to_task_group.get(x, taxonomy.fine_to_coarse[x]) for x in used_fine},
            "keep_fine_bank": keep_bank.detach().cpu(),
            "refined_fine_bank": bank.detach().cpu(),
            "fixed_parent_bank": parent_bank.detach().cpu(),
            "anchor_space": anchor_space,
            "config": refine_cfg.__dict__,
            "session_idx": args.session_idx,
            "prev_refined_anchor_path": args.prev_refined_anchor_path,
            "incremental_refinement": bool(has_prev_refined and args.session_idx is not None and not args.from_scratch_refinement),
            "sessions": sessions,
            "run_sessions": run_sessions,
            "session_format": sessions[0].get("format", None) if sessions else None,
            "freeze_old_anchors": bool(args.freeze_old_anchors),
            "refine_new_only": bool(args.refine_new_only),
        },
        out_dir / "final_refined_text_anchors.pt",
    )

    print(f"\n[Done] saved to: {out_dir}")
    print("  - final_refined_text_anchors.pt")
    print("  - refinement_logs.json")
    print("  - final_similarity_keep_vs_refined.png")
    print("  - final_mds2d_keep_vs_refined.png")
    print("  - final_mds3d_keep_vs_refined.png")
    if bool(hyp_cfg.get("enabled", False)):
        print("  - final_hyperbolic_distance_keep_vs_refined.png")
        print("  - final_poincare_disk_hierarchy.png")


if __name__ == "__main__":
    main()
