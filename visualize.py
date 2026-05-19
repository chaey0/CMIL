import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

from data.session_manager import SessionManager
from utils.taxonomy_parser import TaxonomyParser
from models.model import KnowledgeTreeCLModel
from engine.continual import setup_session_eval
from utils.helpers import (
    set_seed,
    ensure_dir,
    resolve_ckpt_dir,
    prepare_model_structure_for_ckpt_load,
    find_session_ckpt_path,
)
from main import make_loader

def load_ckpt_and_get_session_idx(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt
        session_idx = ckpt.get("session_idx", None)
    else:
        state_dict = ckpt
        session_idx = None

    return state_dict, session_idx


def resolve_session_idx_for_ckpt(loaded_session_idx, fallback_session_idx: int):
    if loaded_session_idx is None:
        return fallback_session_idx, f"session_{fallback_session_idx}", False

    if isinstance(loaded_session_idx, (int, np.integer)):
        sidx = int(loaded_session_idx)
        return sidx, str(sidx), False

    s = str(loaded_session_idx).strip()
    s_lower = s.lower()

    # joint checkpoint
    if "joint" in s_lower:
        return fallback_session_idx, s, True

    if s.isdigit():
        sidx = int(s)
        return sidx, s, False

    m = re.search(r"(\d+)", s)
    if m is not None:
        sidx = int(m.group(1))
        return sidx, s, False

    return fallback_session_idx, s, True

def build_eval_model_for_session_ckpt(
    ckpt_path: str,
    taxonomy: TaxonomyParser,
    cfg: dict,
    session_manager: SessionManager,
    fallback_session_idx: int,
    device: torch.device,
):
    state_dict, loaded_session_idx = load_ckpt_and_get_session_idx(ckpt_path, device)

    session_idx, session_tag, is_joint_like = resolve_session_idx_for_ckpt(
        loaded_session_idx,
        fallback_session_idx,
    )

    model = KnowledgeTreeCLModel(taxonomy, cfg).to(device)

    prepare_model_structure_for_ckpt_load(
        model,
        session_manager,
        session_idx,
        device,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if len(missing) > 0:
        print(f"[Warning] Missing keys: {len(missing)}")
    if len(unexpected) > 0:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")

    model.eval_model_session_idx = session_idx
    model.eval_model_session_tag = session_tag
    model.eval_is_joint_like = is_joint_like

    model.to(device)
    model.eval()

    return model, session_idx, session_tag, is_joint_like

def unpack_batch(batch: Any):
    if isinstance(batch, dict):
        feats_list = batch.get("feats_list") or batch.get("features") or batch.get("bags")
        fine_ids = batch.get("fine_ids")
        coarse_ids = batch.get("coarse_ids")
        fine_labels = batch.get("fine_labels")
        coarse_labels = batch.get("coarse_labels")
        return feats_list, fine_ids, coarse_ids, fine_labels, coarse_labels

    if isinstance(batch, (list, tuple)):
        if len(batch) >= 5:
            return batch[0], batch[1], batch[2], batch[3], batch[4]
        if len(batch) == 3:
            return batch[0], batch[1], batch[2], None, None

    raise ValueError("Unsupported batch format from DataLoader/collate_fn.")


# -----------------------------
# Forward / feature extraction
# -----------------------------
def _to_device_feats(feats_list: Iterable[torch.Tensor], device: torch.device):
    out = []
    for x in feats_list:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        out.append(x.to(device, non_blocking=True).float())
    return out


def forward_for_visualization(
    model: torch.nn.Module,
    feats_list: List[torch.Tensor],
    gt_coarse_labels: Optional[torch.Tensor] = None,
):
    call_trials = [
        lambda: model(feats_list, gt_coarse_labels=gt_coarse_labels),
        lambda: model(feats_list, coarse_labels=gt_coarse_labels),
        lambda: model(feats_list, gt_coarse=gt_coarse_labels),
        lambda: model(feats_list),
    ]

    last_error = None
    for fn in call_trials:
        try:
            return fn()
        except TypeError as e:
            last_error = e
            continue

    raise RuntimeError(f"Model forward call failed. Last error: {last_error}")


def pick_embedding_tensor(output: Dict[str, Any], preferred_key: str = "auto") -> Tuple[torch.Tensor, str]:
    if not isinstance(output, dict):
        raise ValueError("Model output must be a dict for this visualization script.")

    if preferred_key != "auto":
        if preferred_key not in output:
            raise KeyError(f"Requested embedding key '{preferred_key}' not found. Available keys: {list(output.keys())}")
        emb = output[preferred_key]
        if not torch.is_tensor(emb) or emb.ndim != 2:
            raise ValueError(f"output['{preferred_key}'] is not a [B, D] tensor.")
        return emb, preferred_key

    candidates = [
        "h_f",
        "fine_proj",
        "fine_feat",
        "fine_embedding",
        "proj",
        "slide_embed",
        "z",
    ]

    for key in candidates:
        value = output.get(key, None)
        if torch.is_tensor(value) and value.ndim == 2:
            return value, key

    for key, value in output.items():
        if torch.is_tensor(value) and value.ndim == 2:
            return value, key

    raise KeyError(f"Could not find a usable [B, D] embedding tensor in output keys: {list(output.keys())}")


def masked_argmax(logits: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if logits is None:
        return None
    if mask is not None:
        logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
    return logits.argmax(dim=1)


def fine_name_from_idx(taxonomy, idx):
    if idx is None:
        return "NA"
    if hasattr(taxonomy, "idx_to_fine"):
        return taxonomy.idx_to_fine[idx]
    return taxonomy.fine_labels[idx]


def coarse_name_from_idx(taxonomy, idx):
    if idx is None:
        return "NA"
    if hasattr(taxonomy, "idx_to_coarse"):
        return taxonomy.idx_to_coarse[idx]
    return taxonomy.coarse_labels[idx]

@torch.no_grad()
def collect_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    taxonomy: TaxonomyParser,
    embedding_key: str = "auto",
    use_gt_coarse: bool = False,
    opened_fine: Optional[Sequence[str]] = None,
    opened_coarse: Optional[Sequence[str]] = None,
):
    all_embs = []
    rows: List[Dict[str, Any]] = []
    chosen_key = None

    opened_fine = list(opened_fine) if opened_fine is not None else list(taxonomy.fine_labels)
    opened_coarse = list(opened_coarse) if opened_coarse is not None else list(taxonomy.coarse_labels)

    def fine_name(idx):
        if idx is None:
            return "NA"
        idx = int(idx)
        if 0 <= idx < len(opened_fine):
            return opened_fine[idx]
        return fine_name_from_idx(taxonomy, idx)

    def coarse_name(idx):
        if idx is None:
            return "NA"
        idx = int(idx)
        if 0 <= idx < len(opened_coarse):
            return opened_coarse[idx]
        return coarse_name_from_idx(taxonomy, idx)

    for batch_idx, batch in enumerate(loader):
        feats_list, fine_ids, coarse_ids, fine_labels, coarse_labels = unpack_batch(batch)
        feats_list = _to_device_feats(feats_list, device)

        if fine_ids is not None and torch.is_tensor(fine_ids):
            fine_ids = fine_ids.to(device)
        if coarse_ids is not None and torch.is_tensor(coarse_ids):
            coarse_ids = coarse_ids.to(device)

        out = forward_for_visualization(
            model,
            feats_list,
            gt_coarse_labels=coarse_ids if use_gt_coarse else None,
        )

        emb, chosen_key = pick_embedding_tensor(out, preferred_key=embedding_key)
        emb = emb.detach().float().cpu()
        all_embs.append(emb)

        fine_logits = out.get("fine_logits", None)
        coarse_logits = out.get("coarse_logits", None)

        # Important:
        # fine_logits/coarse_logits are already active-local after setup_session_eval().
        # Do not apply full-taxonomy masks here.
        pred_fine_ids = fine_logits.argmax(dim=1) if fine_logits is not None else None
        pred_coarse_ids = coarse_logits.argmax(dim=1) if coarse_logits is not None else None

        if pred_fine_ids is not None:
            pred_fine_ids = pred_fine_ids.detach().cpu().tolist()
        else:
            pred_fine_ids = [None] * emb.shape[0]

        if pred_coarse_ids is not None:
            pred_coarse_ids = pred_coarse_ids.detach().cpu().tolist()
        else:
            pred_coarse_ids = [None] * emb.shape[0]

        fine_ids_cpu = fine_ids.detach().cpu().tolist() if torch.is_tensor(fine_ids) else [None] * emb.shape[0]
        coarse_ids_cpu = coarse_ids.detach().cpu().tolist() if torch.is_tensor(coarse_ids) else [None] * emb.shape[0]

        if fine_labels is None:
            fine_labels = [fine_name(i) for i in fine_ids_cpu]
        if coarse_labels is None:
            coarse_labels = [coarse_name(i) for i in coarse_ids_cpu]

        for i in range(emb.shape[0]):
            gt_fine_id = fine_ids_cpu[i]
            gt_coarse_id = coarse_ids_cpu[i]
            pred_fine_id = pred_fine_ids[i]
            pred_coarse_id = pred_coarse_ids[i]

            rows.append(
                {
                    "sample_index": len(rows),
                    "batch_index": batch_idx,
                    "in_batch_index": i,

                    "gt_fine_id": gt_fine_id,
                    "gt_fine": fine_labels[i] if fine_labels is not None else fine_name(gt_fine_id),

                    "gt_coarse_id": gt_coarse_id,
                    "gt_coarse": coarse_labels[i] if coarse_labels is not None else coarse_name(gt_coarse_id),

                    "pred_fine_id": pred_fine_id,
                    "pred_fine": fine_name(pred_fine_id),

                    "pred_coarse_id": pred_coarse_id,
                    "pred_coarse": coarse_name(pred_coarse_id),
                }
            )

    if len(all_embs) == 0:
        raise RuntimeError("No embeddings were collected. Check csv path and dataset filtering.")

    embs = torch.cat(all_embs, dim=0).numpy()
    meta_df = pd.DataFrame(rows)
    return embs, meta_df, chosen_key


# -----------------------------
# Text-anchor overlay helpers
# -----------------------------
def get_active_text_anchors(
    model: torch.nn.Module,
    taxonomy: TaxonomyParser,
):
    coarse_rows = []
    fine_rows = []

    tree = getattr(model, "tree", None)
    if tree is None:
        return None, None

    coarse_bank = getattr(tree, "coarse_bank", None)
    fine_bank = getattr(tree, "fine_bank", None)
    if coarse_bank is None or fine_bank is None:
        return None, None

    coarse_bank = coarse_bank.detach().cpu().float()
    fine_bank = fine_bank.detach().cpu().float()

    active_coarse = list(getattr(model, "active_coarse_labels", taxonomy.coarse_labels))
    active_fine = list(getattr(model, "active_fine_labels", taxonomy.fine_labels))

    for label in active_coarse:
        if label not in taxonomy.coarse_to_idx:
            continue
        idx = taxonomy.coarse_to_idx[label]
        coarse_rows.append(
            {
                "label": label,
                "kind": "coarse_anchor",
                "parent": "",
                "vec": F.normalize(coarse_bank[idx], dim=0).numpy(),
            }
        )

    for label in active_fine:
        if label not in taxonomy.fine_to_idx:
            continue
        idx = taxonomy.fine_to_idx[label]
        fine_rows.append(
            {
                "label": label,
                "kind": "fine_anchor",
                "parent": taxonomy.fine_to_coarse[label],
                "vec": F.normalize(fine_bank[idx], dim=0).numpy(),
            }
        )

    return coarse_rows, fine_rows

def reduce_2d(
    X: np.ndarray,
    method: str = "umap",
    seed: int = 42,
    perplexity: float = 30.0,
    n_neighbors: int = 20,
    min_dist: float = 0.1,
):
    method = method.lower()

    if X.ndim != 2:
        raise ValueError(f"Expected [N, D], got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples for 2D reduction.")

    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        return reducer.fit_transform(X)

    if method == "tsne":
        perp = min(float(perplexity), max(2.0, X.shape[0] - 1.0))
        reducer = TSNE(
            n_components=2,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        return reducer.fit_transform(X)

    if method == "mds":
        reducer = MDS(n_components=2, random_state=seed, normalized_stress="auto")
        return reducer.fit_transform(X)

    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "UMAP is not installed. Install with `pip install umap-learn`, "
                "or use --method pca / tsne / mds."
            ) from e

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, max(2, X.shape[0] - 1)),
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(X)

    raise ValueError(f"Unknown reduction method: {method}")


def stratified_subsample(
    meta_df: pd.DataFrame,
    max_points: int,
    label_col: str = "gt_fine",
    seed: int = 42,
) -> np.ndarray:
    n = len(meta_df)
    if max_points <= 0 or n <= max_points:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    grouped = meta_df.groupby(label_col).indices

    selected = []
    for _, idxs in grouped.items():
        idxs = np.asarray(list(idxs))
        k = max(1, int(round(len(idxs) / n * max_points)))
        k = min(k, len(idxs))
        selected.extend(rng.choice(idxs, size=k, replace=False).tolist())

    selected = np.array(sorted(set(selected)))

    if len(selected) > max_points:
        selected = rng.choice(selected, size=max_points, replace=False)
    elif len(selected) < max_points:
        remain = np.setdiff1d(np.arange(n), selected)
        extra = min(max_points - len(selected), len(remain))
        if extra > 0:
            selected = np.concatenate([selected, rng.choice(remain, size=extra, replace=False)])

    return np.sort(selected)


# -----------------------------
# Plotting
# -----------------------------
def build_color_map(labels: Sequence[str]) -> Dict[str, Any]:
    uniq = [x for x in pd.unique(pd.Series(labels)) if pd.notna(x)]
    cmap_name = "tab20" if len(uniq) <= 20 else "gist_ncar"
    cmap = plt.get_cmap(cmap_name)
    color_map = {lab: cmap(i / max(1, len(uniq) - 1)) for i, lab in enumerate(uniq)}
    return color_map


def plot_scatter(
    coords: np.ndarray,
    meta_df: pd.DataFrame,
    label_col: str,
    save_path: str,
    title: str,
    coarse_anchor_coords: Optional[np.ndarray] = None,
    coarse_anchor_meta: Optional[List[Dict[str, Any]]] = None,
    fine_anchor_coords: Optional[np.ndarray] = None,
    fine_anchor_meta: Optional[List[Dict[str, Any]]] = None,
    annotate_anchors: bool = True,
    annotate_centroids: bool = False,
):
    plt.figure(figsize=(10, 8))
    color_map = build_color_map(meta_df[label_col].tolist())

    for lab, sub_df in meta_df.groupby(label_col, dropna=False):
        idx = sub_df.index.to_numpy()
        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=16,
            alpha=0.72,
            label=str(lab),
            c=[color_map.get(lab, (0.5, 0.5, 0.5, 1.0))],
            edgecolors="none",
        )

        if annotate_centroids:
            cx = coords[idx, 0].mean()
            cy = coords[idx, 1].mean()
            plt.text(cx, cy, str(lab), fontsize=9)

    if coarse_anchor_coords is not None and coarse_anchor_meta is not None and len(coarse_anchor_meta) > 0:
        plt.scatter(
            coarse_anchor_coords[:, 0],
            coarse_anchor_coords[:, 1],
            marker="s",
            s=180,
            linewidths=1.5,
            facecolors="none",
            edgecolors="black",
            label="coarse anchors",
        )
        if annotate_anchors:
            for xy, m in zip(coarse_anchor_coords, coarse_anchor_meta):
                plt.text(xy[0], xy[1], m["label"], fontsize=10)

    if fine_anchor_coords is not None and fine_anchor_meta is not None and len(fine_anchor_meta) > 0:
        plt.scatter(
            fine_anchor_coords[:, 0],
            fine_anchor_coords[:, 1],
            marker="^",
            s=90,
            linewidths=1.2,
            facecolors="none",
            edgecolors="black",
            label="fine anchors",
        )

        if annotate_anchors:
            for xy, m in zip(fine_anchor_coords, fine_anchor_meta):
                plt.text(xy[0], xy[1], m["label"], fontsize=8)

    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.legend(loc="best", fontsize=8, markerscale=1.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

def _none_if_blank(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in {"none", "null", "-"}:
        return None
    return x


def setup_device_from_arg(gpu_arg):
    """
    Device is controlled only by parser argument --gpu.

    Examples:
      --gpu 0    -> use cuda:0
      --gpu 2    -> use cuda:2
      --gpu cpu  -> use CPU
      --gpu -1   -> use CPU
      no --gpu   -> use cuda:0 if available
    """
    gpu_arg = _none_if_blank(gpu_arg)

    if gpu_arg is None:
        gpu_arg = "0"

    gpu_arg = str(gpu_arg).strip()

    if gpu_arg.lower() in {"cpu", "none", "null", "-1"}:
        print("[Device] use CPU")
        return torch.device("cpu")

    if "," in gpu_arg:
        raise ValueError(
            "This visualization script currently supports single-GPU execution only. "
            "Use one GPU id, e.g. --gpu 0 or --gpu 2."
        )

    try:
        gpu_id = int(gpu_arg)
    except ValueError as e:
        raise ValueError(
            f"Invalid --gpu value: {gpu_arg}. "
            "Use examples like --gpu 0, --gpu 1, --gpu 2, or --gpu cpu."
        ) from e

    if not torch.cuda.is_available():
        print("[Device] CUDA is not available. Use CPU.")
        return torch.device("cpu")

    num_gpus = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= num_gpus:
        raise ValueError(
            f"Invalid GPU id: {gpu_id}. Available CUDA device count = {num_gpus}"
        )

    torch.cuda.set_device(gpu_id)
    print(f"[Device] use cuda:{gpu_id}")
    return torch.device(f"cuda:{gpu_id}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/TCGA.yaml")
    parser.add_argument("--gpu", type=str, default=None, help="GPU id to use, e.g. 0 or 1. Use 'cpu' for CPU.")
    parser.add_argument("--taxonomy_json", type=str, default="data/taxonomy_tcga.json")
    parser.add_argument("--csv", type=str, default="data/tcga_continual.csv")
    parser.add_argument("--out_dir", type=str, default="visualization_out")

    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne", "pca", "mds"])
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--min_dist", type=float, default=0.1)

    parser.add_argument("--embedding_key", type=str, default="auto")
    parser.add_argument("--use_gt_coarse", action="store_true")
    parser.add_argument("--max_points", type=int, default=3000)
    parser.add_argument(
        "--subsample_label",
        type=str,
        default="gt_fine",
        choices=["gt_fine", "gt_coarse", "pred_fine", "pred_coarse"],
    )

    parser.add_argument("--plot_gt_fine", action="store_true")
    parser.add_argument("--plot_gt_coarse", action="store_true")
    parser.add_argument("--plot_pred_fine", action="store_true")
    parser.add_argument("--plot_pred_coarse", action="store_true")
    parser.add_argument("--overlay_text_anchors", action="store_true")
    parser.add_argument("--annotate_anchors", action="store_true")
    parser.add_argument("--annotate_centroids", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = setup_device_from_arg(args.gpu)

    taxonomy = TaxonomyParser(args.taxonomy_json)
    session_manager = SessionManager(cfg, taxonomy)

    print(f"[Taxonomy source JSON] {args.taxonomy_json}")
    print("[Taxonomy] use full taxonomy for checkpoint compatibility")
    print(f"[Coarse labels] {len(taxonomy.coarse_labels)}: {taxonomy.coarse_labels}")
    print(f"[Fine labels] {len(taxonomy.fine_labels)}: {taxonomy.fine_labels}")

    fallback_session_idx = len(cfg.get("sessions", [])) - 1
    if fallback_session_idx < 0:
        raise ValueError("config must contain at least one session to reconstruct the checkpoint structure.")

    if args.ckpt is not None:
        ckpt_path = args.ckpt
    else:
        if args.ckpt_dir is None:
            raise ValueError("Provide either --ckpt or --ckpt_dir.")
        resolved_dir = resolve_ckpt_dir(args.ckpt_dir, fallback_session_idx)
        ckpt_path = find_session_ckpt_path(resolved_dir, fallback_session_idx)

    print(f"[Checkpoint] {ckpt_path}")

    model, session_idx, session_tag, is_joint_like = build_eval_model_for_session_ckpt(
        ckpt_path=ckpt_path,
        taxonomy=taxonomy,
        cfg=cfg,
        session_manager=session_manager,
        fallback_session_idx=fallback_session_idx,
        device=device,
    )

    if is_joint_like:
        print(f"[Loaded checkpoint mode] {session_tag} -> using final session structure ({session_idx})")
    else:
        print(f"[Loaded session_idx] {session_idx}")

    eval_info = setup_session_eval(model, session_manager, session_idx, device)
    opened_fine = eval_info.get("opened_fine", [])
    opened_coarse = eval_info.get("opened_coarse", [])

    print(f"[Opened fine] {opened_fine}")
    print(f"[Opened coarse] {opened_coarse}")

    loader = make_loader(
        csv_path=args.csv,
        taxonomy=taxonomy,
        fine_labels=opened_fine if len(opened_fine) > 0 else taxonomy.fine_labels,
        batch_size=args.batch_size,
        split="test",
        shuffle=False,
    )

    image_embs, meta_df, chosen_key = collect_embeddings(
        model=model,
        loader=loader,
        device=device,
        taxonomy=taxonomy,
        embedding_key=args.embedding_key,
        use_gt_coarse=args.use_gt_coarse,
        opened_fine=opened_fine,
        opened_coarse=opened_coarse,
    )

    print(f"[Embedding key] {chosen_key}")
    print(f"[Collected image embeddings] {image_embs.shape}")

    keep_idx = stratified_subsample(
        meta_df=meta_df,
        max_points=args.max_points,
        label_col=args.subsample_label,
        seed=args.seed,
    )
    image_embs_sub = image_embs[keep_idx]
    meta_sub = meta_df.iloc[keep_idx].reset_index(drop=True)

    print(f"[Subsampled points] {len(meta_sub)} / {len(meta_df)}")

    coarse_anchor_rows, fine_anchor_rows = (None, None)
    if args.overlay_text_anchors:
        coarse_anchor_rows, fine_anchor_rows = get_active_text_anchors(
            model=model,
            taxonomy=taxonomy,
        )

    if args.overlay_text_anchors and coarse_anchor_rows is not None and fine_anchor_rows is not None:
        anchor_vecs = [x["vec"] for x in coarse_anchor_rows] + [x["vec"] for x in fine_anchor_rows]
        if len(anchor_vecs) > 0:
            anchor_mat = np.stack(anchor_vecs, axis=0)
            if anchor_mat.shape[1] != image_embs_sub.shape[1]:
                print(
                    f"[Warning] text/image embedding dims differ: {anchor_mat.shape[1]} vs {image_embs_sub.shape[1]}. "
                    f"Anchor overlay will be skipped."
                )
                coords = reduce_2d(
                    image_embs_sub,
                    method=args.method,
                    seed=args.seed,
                    perplexity=args.perplexity,
                    n_neighbors=args.n_neighbors,
                    min_dist=args.min_dist,
                )
                coarse_anchor_coords = None
                fine_anchor_coords = None
                coarse_anchor_rows = None
                fine_anchor_rows = None
            else:
                joint = np.concatenate([image_embs_sub, anchor_mat], axis=0)
                joint_coords = reduce_2d(
                    joint,
                    method=args.method,
                    seed=args.seed,
                    perplexity=args.perplexity,
                    n_neighbors=args.n_neighbors,
                    min_dist=args.min_dist,
                )
                n_img = len(image_embs_sub)
                n_coarse = len(coarse_anchor_rows)
                coords = joint_coords[:n_img]
                coarse_anchor_coords = joint_coords[n_img:n_img + n_coarse] if n_coarse > 0 else None
                fine_anchor_coords = joint_coords[n_img + n_coarse:] if len(fine_anchor_rows) > 0 else None
        else:
            coords = reduce_2d(
                image_embs_sub,
                method=args.method,
                seed=args.seed,
                perplexity=args.perplexity,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
            )
            coarse_anchor_coords = None
            fine_anchor_coords = None
            coarse_anchor_rows = None
            fine_anchor_rows = None
    else:
        coords = reduce_2d(
            image_embs_sub,
            method=args.method,
            seed=args.seed,
            perplexity=args.perplexity,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )
        coarse_anchor_coords = None
        fine_anchor_coords = None
        coarse_anchor_rows = None
        fine_anchor_rows = None

    out_coords = meta_sub.copy()
    out_coords["x"] = coords[:, 0]
    out_coords["y"] = coords[:, 1]
    out_coords.to_csv(os.path.join(args.out_dir, f"coords_{args.method}.csv"), index=False)

    summary = {
        "ckpt_path": ckpt_path,
        "session_idx_for_structure": int(session_idx),
        "session_tag": session_tag,
        "is_joint_like": bool(is_joint_like),
        "csv": args.csv,
        "method": args.method,
        "embedding_key": chosen_key,
        "num_points_total": int(len(meta_df)),
        "num_points_visualized": int(len(meta_sub)),
        "opened_fine": list(opened_fine),
        "opened_coarse": list(opened_coarse),
        "overlay_text_anchors": bool(args.overlay_text_anchors),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_flags = {
        "gt_fine": args.plot_gt_fine,
        "gt_coarse": args.plot_gt_coarse,
        "pred_fine": args.plot_pred_fine,
        "pred_coarse": args.plot_pred_coarse,
    }

    if not any(plot_flags.values()):
        plot_flags["gt_fine"] = True
        plot_flags["pred_fine"] = True
        plot_flags["gt_coarse"] = True

    for col, enabled in plot_flags.items():
        if not enabled:
            continue

        save_name = f"vis_{args.method}_{col}.png"
        plot_scatter(
            coords=coords,
            meta_df=out_coords,
            label_col=col,
            save_path=os.path.join(args.out_dir, save_name),
            title=f"{args.method.upper()} | {col} | emb={chosen_key}",
            coarse_anchor_coords=coarse_anchor_coords,
            coarse_anchor_meta=coarse_anchor_rows,
            fine_anchor_coords=fine_anchor_coords,
            fine_anchor_meta=fine_anchor_rows,
            annotate_anchors=args.annotate_anchors,
            annotate_centroids=args.annotate_centroids,
        )
        print(f"[Saved] {os.path.join(args.out_dir, save_name)}")
    print("[Done]")

if __name__ == "__main__":
    main()