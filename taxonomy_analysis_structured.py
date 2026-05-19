import os
import json
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE

from models.text_encoder import KEEPEncoder
from models.knowledge_tree import KnowledgeTree
from utils.taxonomy_parser import TaxonomyParser

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


# -----------------------------
# basic utilities
# -----------------------------
def load_taxonomy_raw(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_dict(emb_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in emb_dict.items():
        out[k] = F.normalize(v.float(), dim=0)
    return out


def _normalize_np(x: np.ndarray):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def sanitize_filename(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_", "."]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def cosine_sim_and_dist(embs: np.ndarray):
    embs = _normalize_np(embs)
    sim = embs @ embs.T
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)
    return sim, dist


def pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - pairwise_cosine(a, b))


# -----------------------------
# taxonomy / label ordering helpers
# -----------------------------
def get_coarse_order(taxonomy) -> List[str]:
    return list(taxonomy.coarse_labels)


def get_fine_order_grouped_by_coarse(taxonomy_raw: dict, taxonomy) -> Tuple[List[str], Dict[str, List[str]]]:
    coarse_order = get_coarse_order(taxonomy)
    grouped = {c: [] for c in coarse_order}
    for f in taxonomy.fine_labels:
        c = taxonomy_raw["fine_nodes"][f]["parent"]
        grouped.setdefault(c, []).append(f)

    fine_order = []
    for c in coarse_order:
        fine_order.extend(grouped.get(c, []))
    return fine_order, grouped


def build_name_and_emb_list(coarse_embs, fine_embs):
    coarse_names = list(coarse_embs.keys())
    fine_names = list(fine_embs.keys())
    names = coarse_names + fine_names
    embs = torch.stack(
        [coarse_embs[k] for k in coarse_names] + [fine_embs[k] for k in fine_names],
        dim=0,
    ).cpu().numpy()
    return names, embs, coarse_names, fine_names


def build_actual_node_texts(taxonomy) -> Tuple[Dict[str, str], Dict[str, str]]:
    coarse_texts = {label: taxonomy.get_coarse_text(label) for label in taxonomy.coarse_labels}
    fine_texts = {label: taxonomy.get_fine_text(label) for label in taxonomy.fine_labels}
    return coarse_texts, fine_texts


@torch.no_grad()
def extract_initialized_banks(tree, taxonomy):
    coarse_dict = {}
    fine_dict = {}

    for label in taxonomy.coarse_labels:
        idx = taxonomy.coarse_to_idx[label]
        if bool(tree.coarse_initialized[idx].item()):
            coarse_dict[label] = F.normalize(tree.coarse_bank[idx].detach().cpu().float(), dim=0)

    for label in taxonomy.fine_labels:
        idx = taxonomy.fine_to_idx[label]
        if bool(tree.fine_initialized[idx].item()):
            fine_dict[label] = F.normalize(tree.fine_bank[idx].detach().cpu().float(), dim=0)

    return coarse_dict, fine_dict


# -----------------------------
# text inspection
# -----------------------------

def _contains_any(text: str, candidates: List[str]) -> bool:
    text_l = (text or "").lower()
    for c in candidates:
        c = (c or "").strip()
        if c and c.lower() in text_l:
            return True
    return False


def inspect_actual_text_fields(
    taxonomy_raw: dict,
    coarse_texts: Dict[str, str],
    fine_texts: Dict[str, str],
) -> pd.DataFrame:
    coarse_nodes = taxonomy_raw["coarse_nodes"]
    fine_nodes = taxonomy_raw["fine_nodes"]

    rows = []

    for coarse_name, text in coarse_texts.items():
        info = coarse_nodes[coarse_name]
        display_name = info.get("display_name", coarse_name)
        definition = info.get("definition", "")
        synonyms = info.get("synonyms", [])

        rows.append(
            {
                "node_type": "coarse",
                "label": coarse_name,
                "parent": "",
                "text": text,
                "contains_display_name": display_name.lower() in text.lower() if display_name else False,
                "contains_parent_name": False,
                "contains_definition": definition.lower() in text.lower() if definition else False,
                "contains_any_synonym": _contains_any(text, synonyms),
                "contains_any_morphology": False,
                "contains_any_same_parent_name": False,
                "num_synonyms": len(synonyms),
                "num_morphology": 0,
                "num_same_parent": 0,
            }
        )

    for fine_name, text in fine_texts.items():
        info = fine_nodes[fine_name]
        display_name = info.get("display_name", fine_name)
        parent = info["parent"]
        parent_name = coarse_nodes[parent].get("display_name", parent)
        definition = info.get("definition", "")
        synonyms = info.get("synonyms", [])
        morphology = info.get("morphology", [])

        same_parent_names = []
        for other_name, other_info in fine_nodes.items():
            if other_name == fine_name:
                continue
            if other_info["parent"] == parent:
                same_parent_names.append(other_info.get("display_name", other_name))

        rows.append(
            {
                "node_type": "fine",
                "label": fine_name,
                "parent": parent,
                "text": text,
                "contains_display_name": display_name.lower() in text.lower() if display_name else False,
                "contains_parent_name": parent_name.lower() in text.lower() if parent_name else False,
                "contains_definition": definition.lower() in text.lower() if definition else False,
                "contains_any_synonym": _contains_any(text, synonyms),
                "contains_any_morphology": _contains_any(text, morphology),
                "contains_any_same_parent_name": _contains_any(text, same_parent_names),
                "num_synonyms": len(synonyms),
                "num_morphology": len(morphology),
                "num_same_parent": len(same_parent_names),
            }
        )

    return pd.DataFrame(rows)


def summarize_field_usage(df: pd.DataFrame) -> dict:
    summary = {}

    for node_type in ["coarse", "fine"]:
        sub = df[df["node_type"] == node_type].copy()
        if len(sub) == 0:
            continue

        summary[node_type] = {
            "count": int(len(sub)),
            "display_name_used": int(sub["contains_display_name"].sum()),
            "parent_name_used": int(sub["contains_parent_name"].sum()) if "contains_parent_name" in sub else 0,
            "definition_used": int(sub["contains_definition"].sum()),
            "synonym_used": int(sub["contains_any_synonym"].sum()),
            "morphology_used": int(sub["contains_any_morphology"].sum()) if "contains_any_morphology" in sub else 0,
            "same_parent_name_used": int(sub["contains_any_same_parent_name"].sum()) if "contains_any_same_parent_name" in sub else 0,
        }

    return summary


# -----------------------------
# reports / metrics
# -----------------------------

def distance_statistics(taxonomy_raw: dict, coarse_embs: Dict[str, torch.Tensor], fine_embs: Dict[str, torch.Tensor]):
    fine_nodes = taxonomy_raw["fine_nodes"]
    fine_names = list(fine_embs.keys())

    parent_dists = []
    same_parent_dists = []
    cross_parent_dists = []

    for f in fine_names:
        parent = fine_nodes[f]["parent"]
        parent_dists.append(cosine_distance(fine_embs[f], coarse_embs[parent]))

        same_parent = []
        cross_parent = []
        for g in fine_names:
            if f == g:
                continue
            d = cosine_distance(fine_embs[f], fine_embs[g])
            if fine_nodes[g]["parent"] == parent:
                same_parent.append(d)
            else:
                cross_parent.append(d)

        if len(same_parent) > 0:
            same_parent_dists.append(float(np.mean(same_parent)))
        if len(cross_parent) > 0:
            cross_parent_dists.append(float(np.mean(cross_parent)))

    return {
        "parent_mean": float(np.mean(parent_dists)) if len(parent_dists) > 0 else math.nan,
        "parent_std": float(np.std(parent_dists)) if len(parent_dists) > 0 else math.nan,
        "same_parent_mean": float(np.mean(same_parent_dists)) if len(same_parent_dists) > 0 else math.nan,
        "same_parent_std": float(np.std(same_parent_dists)) if len(same_parent_dists) > 0 else math.nan,
        "cross_parent_mean": float(np.mean(cross_parent_dists)) if len(cross_parent_dists) > 0 else math.nan,
        "cross_parent_std": float(np.std(cross_parent_dists)) if len(cross_parent_dists) > 0 else math.nan,
    }


def pairwise_similarity_report(
    taxonomy_raw: dict,
    coarse_embs: Dict[str, torch.Tensor],
    fine_embs: Dict[str, torch.Tensor],
) -> List[dict]:
    fine_nodes = taxonomy_raw["fine_nodes"]
    fine_names = list(fine_embs.keys())

    rows = []
    for f in fine_names:
        parent = fine_nodes[f]["parent"]
        t_f = fine_embs[f]
        t_c = coarse_embs[parent]

        parent_sim = pairwise_cosine(t_f, t_c)

        same_parent_sims = []
        cross_parent_sims = []
        for g in fine_names:
            if g == f:
                continue
            sim = pairwise_cosine(t_f, fine_embs[g])
            if fine_nodes[g]["parent"] == parent:
                same_parent_sims.append(sim)
            else:
                cross_parent_sims.append(sim)

        mean_same_parent = float(np.mean(same_parent_sims)) if len(same_parent_sims) > 0 else math.nan
        mean_cross_parent = float(np.mean(cross_parent_sims)) if len(cross_parent_sims) > 0 else math.nan

        rows.append(
            {
                "fine": f,
                "parent": parent,
                "sim_to_parent": parent_sim,
                "mean_sim_to_same_parent": mean_same_parent,
                "mean_sim_to_cross_parent": mean_cross_parent,
                "same_minus_cross": (
                    mean_same_parent - mean_cross_parent
                    if not math.isnan(mean_same_parent) and not math.isnan(mean_cross_parent)
                    else math.nan
                ),
            }
        )
    return rows


def parent_retrieval_accuracy(
    taxonomy_raw: dict,
    coarse_embs: Dict[str, torch.Tensor],
    fine_embs: Dict[str, torch.Tensor],
) -> float:
    fine_nodes = taxonomy_raw["fine_nodes"]
    coarse_names = list(coarse_embs.keys())

    correct = 0
    total = 0

    for fine_name, fine_info in fine_nodes.items():
        true_parent = fine_info["parent"]
        sims = []
        for c in coarse_names:
            sim = pairwise_cosine(fine_embs[fine_name], coarse_embs[c])
            sims.append((c, sim))
        pred_parent = sorted(sims, key=lambda x: x[1], reverse=True)[0][0]
        correct += int(pred_parent == true_parent)
        total += 1

    return correct / max(total, 1)


def nearest_same_parent_accuracy(taxonomy_raw: dict, fine_embs: Dict[str, torch.Tensor]) -> float:
    fine_nodes = taxonomy_raw["fine_nodes"]
    fine_names = list(fine_embs.keys())

    correct = 0
    total = 0

    for f in fine_names:
        sims = []
        for g in fine_names:
            if f == g:
                continue
            sim = pairwise_cosine(fine_embs[f], fine_embs[g])
            sims.append((g, sim))

        if len(sims) == 0:
            continue

        pred_neighbor = sorted(sims, key=lambda x: x[1], reverse=True)[0][0]
        same_parent = fine_nodes[pred_neighbor]["parent"] == fine_nodes[f]["parent"]
        correct += int(same_parent)
        total += 1

    return correct / max(total, 1)


def fine_to_coarse_margin_report(
    taxonomy_raw: dict,
    coarse_embs: Dict[str, torch.Tensor],
    fine_embs: Dict[str, torch.Tensor],
) -> pd.DataFrame:
    fine_nodes = taxonomy_raw["fine_nodes"]
    coarse_names = list(coarse_embs.keys())

    rows = []
    for f, emb in fine_embs.items():
        true_parent = fine_nodes[f]["parent"]
        sims = [(c, pairwise_cosine(emb, coarse_embs[c])) for c in coarse_names]
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
        best_pred, best_pred_sim = sims_sorted[0]
        true_parent_sim = dict(sims)[true_parent]

        best_wrong_parent = None
        best_wrong_sim = -1.0
        for c, s in sims_sorted:
            if c != true_parent:
                best_wrong_parent = c
                best_wrong_sim = s
                break

        rows.append(
            {
                "fine": f,
                "parent": true_parent,
                "true_parent_sim": true_parent_sim,
                "best_wrong_parent": best_wrong_parent,
                "best_wrong_parent_sim": best_wrong_sim,
                "parent_margin": true_parent_sim - best_wrong_sim,
                "top_pred_parent": best_pred,
                "top_pred_parent_sim": best_pred_sim,
                "parent_correct": int(best_pred == true_parent),
            }
        )

    return pd.DataFrame(rows)


def hard_negative_report(
    taxonomy_raw: dict,
    fine_embs: Dict[str, torch.Tensor],
    topk: int = 5,
) -> pd.DataFrame:
    fine_nodes = taxonomy_raw["fine_nodes"]
    fine_names = list(fine_embs.keys())
    rows = []

    for f in fine_names:
        parent = fine_nodes[f]["parent"]
        sims = []
        for g in fine_names:
            if g == f:
                continue
            sim = pairwise_cosine(fine_embs[f], fine_embs[g])
            relation = "same_parent" if fine_nodes[g]["parent"] == parent else "cross_parent"
            sims.append((g, fine_nodes[g]["parent"], sim, relation))

        sims_sorted = sorted(sims, key=lambda x: x[2], reverse=True)
        same_parent_sorted = [x for x in sims_sorted if x[3] == "same_parent"][:topk]
        cross_parent_sorted = [x for x in sims_sorted if x[3] == "cross_parent"][:topk]

        for rank, (g, g_parent, sim, _) in enumerate(same_parent_sorted, start=1):
            rows.append(
                {
                    "anchor_fine": f,
                    "anchor_parent": parent,
                    "neighbor_type": "same_parent",
                    "rank": rank,
                    "neighbor_fine": g,
                    "neighbor_parent": g_parent,
                    "similarity": sim,
                }
            )

        for rank, (g, g_parent, sim, _) in enumerate(cross_parent_sorted, start=1):
            rows.append(
                {
                    "anchor_fine": f,
                    "anchor_parent": parent,
                    "neighbor_type": "cross_parent",
                    "rank": rank,
                    "neighbor_fine": g,
                    "neighbor_parent": g_parent,
                    "similarity": sim,
                }
            )

    return pd.DataFrame(rows)


# -----------------------------
# plotting helpers
# -----------------------------

def _adaptive_figsize(n_rows: int, n_cols: int, base: float = 0.38, min_size: float = 8.0, max_size: float = 28.0):
    w = min(max(n_cols * base, min_size), max_size)
    h = min(max(n_rows * base, min_size), max_size)
    return (w, h)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    save_path: str,
    title: str,
    row_boundaries: List[int] = None,
    col_boundaries: List[int] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
):
    n_rows, n_cols = matrix.shape
    figsize = _adaptive_figsize(n_rows, n_cols)
    plt.figure(figsize=figsize)
    im = plt.imshow(matrix, vmin=vmin, vmax=vmax, aspect="auto", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    if n_cols <= 120:
        plt.xticks(range(n_cols), col_labels, rotation=90, fontsize=max(5, 10 - n_cols // 15))
    else:
        plt.xticks([])

    if n_rows <= 120:
        plt.yticks(range(n_rows), row_labels, fontsize=max(5, 10 - n_rows // 15))
    else:
        plt.yticks([])

    if row_boundaries:
        for b in row_boundaries:
            plt.axhline(b - 0.5, linewidth=1.2)
    if col_boundaries:
        for b in col_boundaries:
            plt.axvline(b - 0.5, linewidth=1.2)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_clustered_similarity_matrix(
    coarse_embs,
    fine_embs,
    save_path,
    title="Clustered Similarity Matrix",
):
    names, embs, _, _ = build_name_and_emb_list(coarse_embs, fine_embs)
    sim, dist = cosine_sim_and_dist(embs)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)

    sim_ord = sim[order][:, order]
    names_ord = [names[i] for i in order]

    plot_heatmap(
        matrix=sim_ord,
        row_labels=names_ord,
        col_labels=names_ord,
        save_path=save_path,
        title=title,
    )


def reduce_2d(embs: np.ndarray, method: str = "mds", random_state: int = 0):
    embs = _normalize_np(embs)
    sim = embs @ embs.T
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)

    if method == "mds":
        z = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=8,
        ).fit_transform(dist)
        return z
    elif method == "tsne":
        n = len(embs)
        perplexity = min(8, max(2, n // 4))
        z = TSNE(
            n_components=2,
            metric="precomputed",
            init="random",
            perplexity=perplexity,
            random_state=random_state,
        ).fit_transform(dist)
        return z
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_2d_embedding(
    taxonomy_raw,
    coarse_embs,
    fine_embs,
    save_path,
    title="2D Embedding",
    method="mds",
):
    names, embs, coarse_names, fine_names = build_name_and_emb_list(coarse_embs, fine_embs)
    z = reduce_2d(embs, method=method)
    coarse_set = set(coarse_names)

    plt.figure(figsize=(9, 7))
    for i, name in enumerate(names):
        x, y = z[i, 0], z[i, 1]
        if name in coarse_set:
            plt.scatter(x, y, marker="s", s=170)
            plt.text(x, y, name, fontsize=10)
        else:
            parent = taxonomy_raw["fine_nodes"][name]["parent"]
            plt.scatter(x, y, marker="o", s=90)
            plt.text(x, y, f"{name}({parent})", fontsize=9)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def build_similarity_dataframe_from_dicts(row_dict: Dict[str, torch.Tensor], col_dict: Dict[str, torch.Tensor]) -> pd.DataFrame:
    row_names = list(row_dict.keys())
    col_names = list(col_dict.keys())
    matrix = np.zeros((len(row_names), len(col_names)), dtype=np.float32)

    for i, r in enumerate(row_names):
        for j, c in enumerate(col_names):
            matrix[i, j] = pairwise_cosine(row_dict[r], col_dict[c])

    return pd.DataFrame(matrix, index=row_names, columns=col_names)


@torch.no_grad()
def save_similarity_matrix_csv(coarse_embs, fine_embs, save_path):
    names = list(coarse_embs.keys()) + list(fine_embs.keys())
    embs = [coarse_embs[k] for k in coarse_embs.keys()] + [fine_embs[k] for k in fine_embs.keys()]
    embs = torch.stack(embs, dim=0)
    sims = (embs @ embs.t()).cpu().numpy()

    df = pd.DataFrame(sims, index=names, columns=names)
    df.to_csv(save_path, index=True)


def save_text_dump(coarse_texts, fine_texts, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== COARSE TEXTS (ACTUAL) ===\n")
        for k, v in coarse_texts.items():
            f.write(f"[{k}]\n{v}\n\n")

        f.write("=== FINE TEXTS (ACTUAL) ===\n")
        for k, v in fine_texts.items():
            f.write(f"[{k}]\n{v}\n\n")


def save_coarse_only_views(coarse_embs: Dict[str, torch.Tensor], out_dir: str):
    df = build_similarity_dataframe_from_dicts(coarse_embs, coarse_embs)
    df.to_csv(os.path.join(out_dir, "coarse_only_similarity.csv"), index=True)
    plot_heatmap(
        matrix=df.values,
        row_labels=list(df.index),
        col_labels=list(df.columns),
        save_path=os.path.join(out_dir, "coarse_only_similarity.png"),
        title="Coarse-to-Coarse Similarity",
    )


def save_fine_to_coarse_views(
    taxonomy_raw: dict,
    taxonomy,
    coarse_embs: Dict[str, torch.Tensor],
    fine_embs: Dict[str, torch.Tensor],
    out_dir: str,
):
    fine_order, grouped = get_fine_order_grouped_by_coarse(taxonomy_raw, taxonomy)
    coarse_order = get_coarse_order(taxonomy)

    row_dict = {f: fine_embs[f] for f in fine_order if f in fine_embs}
    col_dict = {c: coarse_embs[c] for c in coarse_order if c in coarse_embs}
    df = build_similarity_dataframe_from_dicts(row_dict, col_dict)
    df.to_csv(os.path.join(out_dir, "fine_to_coarse_similarity.csv"), index=True)

    row_boundaries = []
    cursor = 0
    for c in coarse_order:
        n = sum(1 for f in grouped.get(c, []) if f in fine_embs)
        cursor += n
        if 0 < cursor < len(df.index):
            row_boundaries.append(cursor)

    plot_heatmap(
        matrix=df.values,
        row_labels=list(df.index),
        col_labels=list(df.columns),
        save_path=os.path.join(out_dir, "fine_to_coarse_similarity.png"),
        title="Fine-to-Coarse Similarity",
        row_boundaries=row_boundaries,
    )


def save_fine_block_views(
    taxonomy_raw: dict,
    taxonomy,
    fine_embs: Dict[str, torch.Tensor],
    out_dir: str,
):
    fine_order, grouped = get_fine_order_grouped_by_coarse(taxonomy_raw, taxonomy)
    fine_order = [f for f in fine_order if f in fine_embs]
    fine_dict = {f: fine_embs[f] for f in fine_order}
    df = build_similarity_dataframe_from_dicts(fine_dict, fine_dict)
    df.to_csv(os.path.join(out_dir, "fine_similarity_by_coarse_order.csv"), index=True)

    coarse_order = get_coarse_order(taxonomy)
    boundaries = []
    cursor = 0
    for c in coarse_order:
        n = sum(1 for f in grouped.get(c, []) if f in fine_embs)
        cursor += n
        if 0 < cursor < len(fine_order):
            boundaries.append(cursor)

    plot_heatmap(
        matrix=df.values,
        row_labels=list(df.index),
        col_labels=list(df.columns),
        save_path=os.path.join(out_dir, "fine_similarity_by_coarse_order.png"),
        title="Fine-to-Fine Similarity (Grouped by Coarse)",
        row_boundaries=boundaries,
        col_boundaries=boundaries,
    )

    # coarse-block average matrix over fine-fine similarities
    grouped_present = {c: [f for f in grouped.get(c, []) if f in fine_embs] for c in coarse_order}
    block_mat = np.zeros((len(coarse_order), len(coarse_order)), dtype=np.float32)
    for i, c1 in enumerate(coarse_order):
        for j, c2 in enumerate(coarse_order):
            rows = grouped_present.get(c1, [])
            cols = grouped_present.get(c2, [])
            vals = []
            for f1 in rows:
                for f2 in cols:
                    vals.append(pairwise_cosine(fine_embs[f1], fine_embs[f2]))
            block_mat[i, j] = float(np.mean(vals)) if len(vals) > 0 else np.nan

    block_df = pd.DataFrame(block_mat, index=coarse_order, columns=coarse_order)
    block_df.to_csv(os.path.join(out_dir, "coarse_block_average_from_fine.csv"), index=True)
    plot_heatmap(
        matrix=np.nan_to_num(block_df.values, nan=0.0),
        row_labels=list(block_df.index),
        col_labels=list(block_df.columns),
        save_path=os.path.join(out_dir, "coarse_block_average_from_fine.png"),
        title="Average Fine Similarity Between Coarse Blocks",
    )


def save_within_coarse_views(
    taxonomy_raw: dict,
    taxonomy,
    fine_embs: Dict[str, torch.Tensor],
    out_dir: str,
):
    within_dir = os.path.join(out_dir, "within_coarse")
    os.makedirs(within_dir, exist_ok=True)

    _, grouped = get_fine_order_grouped_by_coarse(taxonomy_raw, taxonomy)
    summary_rows = []

    for coarse_name in get_coarse_order(taxonomy):
        fine_names = [f for f in grouped.get(coarse_name, []) if f in fine_embs]
        if len(fine_names) == 0:
            continue

        fine_dict = {f: fine_embs[f] for f in fine_names}
        df = build_similarity_dataframe_from_dicts(fine_dict, fine_dict)
        safe_name = sanitize_filename(coarse_name)
        df.to_csv(os.path.join(within_dir, f"{safe_name}_fine_similarity.csv"), index=True)

        plot_heatmap(
            matrix=df.values,
            row_labels=list(df.index),
            col_labels=list(df.columns),
            save_path=os.path.join(within_dir, f"{safe_name}_fine_similarity.png"),
            title=f"Within-Coarse Fine Similarity: {coarse_name}",
        )

        vals = df.values.copy()
        if len(fine_names) > 1:
            mask = ~np.eye(len(fine_names), dtype=bool)
            off_diag = vals[mask]
            mean_off_diag = float(np.mean(off_diag)) if len(off_diag) > 0 else math.nan
            max_off_diag = float(np.max(off_diag)) if len(off_diag) > 0 else math.nan
            min_off_diag = float(np.min(off_diag)) if len(off_diag) > 0 else math.nan
        else:
            mean_off_diag = math.nan
            max_off_diag = math.nan
            min_off_diag = math.nan

        summary_rows.append(
            {
                "coarse": coarse_name,
                "num_fine": len(fine_names),
                "mean_offdiag_similarity": mean_off_diag,
                "max_offdiag_similarity": max_off_diag,
                "min_offdiag_similarity": min_off_diag,
            }
        )

    if len(summary_rows) > 0:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "within_coarse_summary.csv"), index=False
        )


def save_global_and_local_views(
    taxonomy_raw: dict,
    taxonomy,
    coarse_embs: Dict[str, torch.Tensor],
    fine_embs: Dict[str, torch.Tensor],
    out_dir: str,
    save_2d: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    # legacy/global views
    plot_clustered_similarity_matrix(
        coarse_embs=coarse_embs,
        fine_embs=fine_embs,
        save_path=os.path.join(out_dir, "sim_matrix_clustered.png"),
        title="Clustered Similarity Matrix (All Coarse + Fine)",
    )

    # recommended structured views
    save_coarse_only_views(coarse_embs, out_dir)
    save_fine_to_coarse_views(taxonomy_raw, taxonomy, coarse_embs, fine_embs, out_dir)
    save_fine_block_views(taxonomy_raw, taxonomy, fine_embs, out_dir)
    save_within_coarse_views(taxonomy_raw, taxonomy, fine_embs, out_dir)

    if save_2d:
        plot_2d_embedding(
            taxonomy_raw=taxonomy_raw,
            coarse_embs=coarse_embs,
            fine_embs=fine_embs,
            save_path=os.path.join(out_dir, "mds2d.png"),
            title="MDS 2D",
            method="mds",
        )
        plot_2d_embedding(
            taxonomy_raw=taxonomy_raw,
            coarse_embs=coarse_embs,
            fine_embs=fine_embs,
            save_path=os.path.join(out_dir, "tsne2d.png"),
            title="t-SNE 2D",
            method="tsne",
        )


# -----------------------------
# main analysis
# -----------------------------

@torch.no_grad()
def run_taxonomy_analysis(
    taxonomy_raw,
    taxonomy,
    encoder,
    out_dir,
    device,
    embed_dim=768,
    parent_projection_alpha=0.0,
    sibling_projection_alpha=0.0,
    hard_negative_topk=5,
    save_2d=False,
):
    os.makedirs(out_dir, exist_ok=True)

    tree = KnowledgeTree(
        taxonomy=taxonomy,
        embed_dim=embed_dim,
        parent_projection_alpha=parent_projection_alpha,
    ).to(device)

    all_coarse = list(taxonomy.coarse_labels)
    all_fine = list(taxonomy.fine_labels)

    print("\n" + "=" * 80)
    print("[Full Taxonomy Analysis]")
    print(f"all_coarse: {all_coarse}")
    print(f"all_fine  : {all_fine}")

    tree.initialize_nodes(
        text_encoder=encoder,
        coarse_labels=all_coarse,
        fine_labels=all_fine,
        device=device,
        refresh_existing=False,
        sibling_projection_alpha=sibling_projection_alpha,
    )

    coarse_embs, fine_embs = extract_initialized_banks(tree, taxonomy)

    sim_csv_path = os.path.join(out_dir, "sim_matrix.csv")
    text_dump_path = os.path.join(out_dir, "actual_texts.txt")
    field_csv_path = os.path.join(out_dir, "text_field_usage.csv")
    pairwise_csv_path = os.path.join(out_dir, "pairwise_similarity_report.csv")
    parent_margin_csv_path = os.path.join(out_dir, "fine_to_coarse_margin_report.csv")
    hard_negative_csv_path = os.path.join(out_dir, "hard_negative_report.csv")
    summary_json_path = os.path.join(out_dir, "summary.json")

    save_similarity_matrix_csv(coarse_embs, fine_embs, sim_csv_path)

    save_global_and_local_views(
        taxonomy_raw=taxonomy_raw,
        taxonomy=taxonomy,
        coarse_embs=coarse_embs,
        fine_embs=fine_embs,
        out_dir=out_dir,
        save_2d=save_2d,
    )

    coarse_texts, fine_texts = build_actual_node_texts(taxonomy)
    save_text_dump(coarse_texts, fine_texts, text_dump_path)

    field_df = inspect_actual_text_fields(taxonomy_raw, coarse_texts, fine_texts)
    field_df.to_csv(field_csv_path, index=False)

    pairwise_rows = pairwise_similarity_report(taxonomy_raw, coarse_embs, fine_embs)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_csv_path, index=False)

    margin_df = fine_to_coarse_margin_report(taxonomy_raw, coarse_embs, fine_embs)
    margin_df.to_csv(parent_margin_csv_path, index=False)

    hard_negative_df = hard_negative_report(taxonomy_raw, fine_embs, topk=hard_negative_topk)
    hard_negative_df.to_csv(hard_negative_csv_path, index=False)

    stats = distance_statistics(taxonomy_raw, coarse_embs, fine_embs)
    parent_acc = parent_retrieval_accuracy(taxonomy_raw, coarse_embs, fine_embs)
    nearest_same_parent_acc = nearest_same_parent_accuracy(taxonomy_raw, fine_embs)
    field_summary = summarize_field_usage(field_df)

    summary = {
        "num_coarse": len(coarse_embs),
        "num_fine": len(fine_embs),
        "distance_statistics": stats,
        "parent_retrieval_accuracy": parent_acc,
        "nearest_same_parent_accuracy": nearest_same_parent_acc,
        "fine_to_coarse_margin_mean": float(margin_df["parent_margin"].mean()) if len(margin_df) > 0 else math.nan,
        "fine_to_coarse_margin_min": float(margin_df["parent_margin"].min()) if len(margin_df) > 0 else math.nan,
        "field_usage_summary": field_summary,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy_json", type=str, default="./data/taxonomy_full.json")
    parser.add_argument("--out_dir", type=str, default="./anchor_analysis_taxonomy")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--parent_projection_alpha", type=float, default=0.0)
    parser.add_argument("--sibling_projection_alpha", type=float, default=0.0)
    parser.add_argument("--hard_negative_topk", type=int, default=5)
    parser.add_argument("--save_2d", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    taxonomy_raw = load_taxonomy_raw(args.taxonomy_json)
    taxonomy = TaxonomyParser(args.taxonomy_json)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = KEEPEncoder().to(device)
    encoder.eval()

    run_taxonomy_analysis(
        taxonomy_raw=taxonomy_raw,
        taxonomy=taxonomy,
        encoder=encoder,
        out_dir=args.out_dir,
        device=device,
        embed_dim=768,
        parent_projection_alpha=args.parent_projection_alpha,
        sibling_projection_alpha=args.sibling_projection_alpha,
        hard_negative_topk=args.hard_negative_topk,
        save_2d=args.save_2d,
    )


if __name__ == "__main__":
    main()
