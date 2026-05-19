import os
import json
import math
import argparse
from typing import Dict, List

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


def build_name_and_emb_list(coarse_embs, fine_embs):
    coarse_names = list(coarse_embs.keys())
    fine_names = list(fine_embs.keys())
    names = coarse_names + fine_names
    embs = torch.stack(
        [coarse_embs[k] for k in coarse_names] + [fine_embs[k] for k in fine_names],
        dim=0
    ).cpu().numpy()
    return names, embs, coarse_names, fine_names


def cosine_sim_and_dist(embs: np.ndarray):
    embs = _normalize_np(embs)
    sim = embs @ embs.T
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)
    return sim, dist

def build_actual_node_texts(taxonomy) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    KnowledgeTree.initialize_nodes()가 실제로 사용하는 경로와 동일하게
    taxonomy.get_coarse_text(), taxonomy.get_fine_text()를 호출해서 텍스트 생성
    """
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


def _contains_any(text: str, candidates: List[str]) -> bool:
    text_l = (text or "").lower()
    for c in candidates:
        c = (c or "").strip()
        if c and c.lower() in text_l:
            return True
    return False


def inspect_actual_text_fields(taxonomy_raw: dict, coarse_texts: Dict[str, str], fine_texts: Dict[str, str]) -> pd.DataFrame:
    """
    실제 생성된 텍스트 문자열에 어떤 정보가 포함됐는지 체크
    """
    coarse_nodes = taxonomy_raw["coarse_nodes"]
    fine_nodes = taxonomy_raw["fine_nodes"]

    rows = []

    # coarse
    for coarse_name, text in coarse_texts.items():
        info = coarse_nodes[coarse_name]
        display_name = info.get("display_name", coarse_name)
        definition = info.get("definition", "")
        synonyms = info.get("synonyms", [])

        rows.append({
            "node_type": "coarse",
            "label": coarse_name,
            "parent": "",
            "text": text,
            "contains_display_name": display_name.lower() in text.lower() if display_name else False,
            "contains_parent_name": False,
            "contains_definition": definition.lower() in text.lower() if definition else False,
            "contains_any_synonym": _contains_any(text, synonyms),
            "contains_any_morphology": False,
            "contains_any_sibling_name": False,
            "num_synonyms": len(synonyms),
            "num_morphology": 0,
            "num_siblings": 0,
        })

    # fine
    for fine_name, text in fine_texts.items():
        info = fine_nodes[fine_name]
        display_name = info.get("display_name", fine_name)
        parent = info["parent"]
        parent_name = coarse_nodes[parent].get("display_name", parent)
        definition = info.get("definition", "")
        synonyms = info.get("synonyms", [])
        morphology = info.get("morphology", [])

        sibling_names = []
        for other_name, other_info in fine_nodes.items():
            if other_name == fine_name:
                continue
            if other_info["parent"] == parent:
                sibling_names.append(other_info.get("display_name", other_name))

        rows.append({
            "node_type": "fine",
            "label": fine_name,
            "parent": parent,
            "text": text,
            "contains_display_name": display_name.lower() in text.lower() if display_name else False,
            "contains_parent_name": parent_name.lower() in text.lower() if parent_name else False,
            "contains_definition": definition.lower() in text.lower() if definition else False,
            "contains_any_synonym": _contains_any(text, synonyms),
            "contains_any_morphology": _contains_any(text, morphology),
            "contains_any_sibling_name": _contains_any(text, sibling_names),
            "num_synonyms": len(synonyms),
            "num_morphology": len(morphology),
            "num_siblings": len(sibling_names),
        })

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
            "sibling_used": int(sub["contains_any_sibling_name"].sum()) if "contains_any_sibling_name" in sub else 0,
        }

    return summary


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

def distance_statistics(taxonomy_raw: dict, coarse_embs: Dict[str, torch.Tensor], fine_embs: Dict[str, torch.Tensor]):
    fine_nodes = taxonomy_raw["fine_nodes"]
    fine_names = list(fine_embs.keys())

    parent_dists = []
    sibling_dists = []
    other_dists = []

    for f in fine_names:
        parent = fine_nodes[f]["parent"]
        parent_dists.append(cosine_distance(fine_embs[f], coarse_embs[parent]))

        sibs = []
        others = []

        for g in fine_names:
            if f == g:
                continue
            d = cosine_distance(fine_embs[f], fine_embs[g])
            if fine_nodes[g]["parent"] == parent:
                sibs.append(d)
            else:
                others.append(d)

        if len(sibs) > 0:
            sibling_dists.append(float(np.mean(sibs)))
        if len(others) > 0:
            other_dists.append(float(np.mean(others)))

    return {
        "parent_mean": float(np.mean(parent_dists)) if len(parent_dists) > 0 else math.nan,
        "parent_std": float(np.std(parent_dists)) if len(parent_dists) > 0 else math.nan,
        "sibling_mean": float(np.mean(sibling_dists)) if len(sibling_dists) > 0 else math.nan,
        "sibling_std": float(np.std(sibling_dists)) if len(sibling_dists) > 0 else math.nan,
        "other_mean": float(np.mean(other_dists)) if len(other_dists) > 0 else math.nan,
        "other_std": float(np.std(other_dists)) if len(other_dists) > 0 else math.nan,
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

        parent_sim = F.cosine_similarity(t_f.unsqueeze(0), t_c.unsqueeze(0)).item()

        sib_sims = []
        other_sims = []
        for g in fine_names:
            if g == f:
                continue
            sim = F.cosine_similarity(t_f.unsqueeze(0), fine_embs[g].unsqueeze(0)).item()
            if fine_nodes[g]["parent"] == parent:
                sib_sims.append(sim)
            else:
                other_sims.append(sim)

        mean_sib = float(np.mean(sib_sims)) if len(sib_sims) > 0 else math.nan
        mean_other = float(np.mean(other_sims)) if len(other_sims) > 0 else math.nan

        rows.append({
            "fine": f,
            "parent": parent,
            "sim_to_parent": parent_sim,
            "mean_sim_to_siblings": mean_sib,
            "mean_sim_to_others": mean_other,
            "sib_minus_other": mean_sib - mean_other if not math.isnan(mean_sib) and not math.isnan(mean_other) else math.nan,
        })
    return rows


def parent_retrieval_accuracy(taxonomy_raw: dict, coarse_embs: Dict[str, torch.Tensor], fine_embs: Dict[str, torch.Tensor]) -> float:
    fine_nodes = taxonomy_raw["fine_nodes"]
    coarse_names = list(coarse_embs.keys())

    correct = 0
    total = 0

    for fine_name, fine_info in fine_nodes.items():
        true_parent = fine_info["parent"]
        sims = []
        for c in coarse_names:
            sim = F.cosine_similarity(fine_embs[fine_name].unsqueeze(0), coarse_embs[c].unsqueeze(0)).item()
            sims.append((c, sim))
        pred_parent = sorted(sims, key=lambda x: x[1], reverse=True)[0][0]
        correct += int(pred_parent == true_parent)
        total += 1

    return correct / max(total, 1)


def sibling_retrieval_accuracy(taxonomy_raw: dict, fine_embs: Dict[str, torch.Tensor]) -> float:
    fine_nodes = taxonomy_raw["fine_nodes"]
    fine_names = list(fine_embs.keys())

    correct = 0
    total = 0

    for f in fine_names:
        sims = []
        for g in fine_names:
            if f == g:
                continue
            sim = F.cosine_similarity(fine_embs[f].unsqueeze(0), fine_embs[g].unsqueeze(0)).item()
            sims.append((g, sim))

        if len(sims) == 0:
            continue

        pred_neighbor = sorted(sims, key=lambda x: x[1], reverse=True)[0][0]
        same_parent = fine_nodes[pred_neighbor]["parent"] == fine_nodes[f]["parent"]
        correct += int(same_parent)
        total += 1

    return correct / max(total, 1)

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

    plt.figure(figsize=(10, 8))
    im = plt.imshow(sim_ord, vmin=-1.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(names_ord)), names_ord, rotation=90)
    plt.yticks(range(len(names_ord)), names_ord)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


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



def save_text_dump(coarse_texts, fine_texts, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== COARSE TEXTS (ACTUAL) ===\n")
        for k, v in coarse_texts.items():
            f.write(f"[{k}]\n{v}\n\n")

        f.write("=== FINE TEXTS (ACTUAL) ===\n")
        for k, v in fine_texts.items():
            f.write(f"[{k}]\n{v}\n\n")

@torch.no_grad()
def save_similarity_matrix_csv(coarse_embs, fine_embs, save_path):
    names = list(coarse_embs.keys()) + list(fine_embs.keys())
    embs = [coarse_embs[k] for k in coarse_embs.keys()] + [fine_embs[k] for k in fine_embs.keys()]
    embs = torch.stack(embs, dim=0)   # [N, D]
    sims = (embs @ embs.t()).cpu().numpy()

    df = pd.DataFrame(sims, index=names, columns=names)
    df.to_csv(save_path, index=True)


def save_structure_views(
    taxonomy_raw,
    coarse_embs,
    fine_embs,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)

    plot_clustered_similarity_matrix(
        coarse_embs=coarse_embs,
        fine_embs=fine_embs,
        save_path=os.path.join(out_dir, "sim_matrix_clustered.png"),
        title="Clustered Similarity Matrix",
    )

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
    sim_png_path = os.path.join(out_dir, "sim_matrix.png")
    text_dump_path = os.path.join(out_dir, "actual_texts.txt")
    field_csv_path = os.path.join(out_dir, "text_field_usage.csv")
    pairwise_csv_path = os.path.join(out_dir, "pairwise_similarity_report.csv")
    summary_json_path = os.path.join(out_dir, "summary.json")

    save_similarity_matrix_csv(coarse_embs, fine_embs, sim_csv_path)

    save_structure_views(
        taxonomy_raw=taxonomy_raw,
        coarse_embs=coarse_embs,
        fine_embs=fine_embs,
        out_dir=out_dir,
    )

    coarse_texts, fine_texts = build_actual_node_texts(taxonomy)
    save_text_dump(coarse_texts, fine_texts, text_dump_path)

    field_df = inspect_actual_text_fields(taxonomy_raw, coarse_texts, fine_texts)
    field_df.to_csv(field_csv_path, index=False)

    pairwise_rows = pairwise_similarity_report(taxonomy_raw, coarse_embs, fine_embs)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_csv_path, index=False)

    stats = distance_statistics(taxonomy_raw, coarse_embs, fine_embs)
    parent_acc = parent_retrieval_accuracy(taxonomy_raw, coarse_embs, fine_embs)
    sibling_acc = sibling_retrieval_accuracy(taxonomy_raw, fine_embs)
    field_summary = summarize_field_usage(field_df)

    summary = {
        "num_coarse": len(coarse_embs),
        "num_fine": len(fine_embs),
        "distance_statistics": stats,
        "parent_retrieval_accuracy": parent_acc,
        "sibling_retrieval_accuracy": sibling_acc,
        "field_usage_summary": field_summary,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy_json", type=str, default="./data/taxonomy_full.json")
    parser.add_argument("--out_dir", type=str, default="./anchor_analysis_taxonomy")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--parent_projection_alpha", type=float, default=0.0)
    parser.add_argument("--sibling_projection_alpha", type=float, default=0.0)
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
    )

if __name__ == "__main__":
    main()
