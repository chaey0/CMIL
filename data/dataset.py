from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter

try:
    import h5py
except ImportError:
    h5py = None


# split-version csv에서 사용하는 feature_path를 우선 사용.
# 기존 separate csv / 다른 전처리 산출물과도 호환되도록 후보를 넓게 둔다.
FEATURE_COL_CANDIDATES = [
    "feature_path",
    "features_path",
    "h5_path",
    "path",
    "file_path",
    "slide_path",
    "wsi_path",
    "UNI_patch_features_path",
    "pt_path",
]
FINE_COL_CANDIDATES = [
    "fine_label",
]
COARSE_COL_CANDIDATES = [
    "coarse_label",
]
TASK_COL_CANDIDATES = ["task_label", "task", "classification_task"]
SLIDE_COL_CANDIDATES = ["slide_id", "case_id", "patient_id", "sample_id"]


def _find_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find any of columns: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _choose_feature_column(df: pd.DataFrame) -> str:
    existing = [c for c in FEATURE_COL_CANDIDATES if c in df.columns]
    if not existing:
        raise KeyError(
            f"Could not find any feature column from {FEATURE_COL_CANDIDATES}. "
            f"Available columns: {list(df.columns)}"
        )

    # 실제 feature 파일 경로처럼 보이는 컬럼을 우선 선택한다.
    best_col = None
    best_score = -1
    for c in existing:
        vals = df[c].dropna().astype(str).head(200)
        score = 0
        for v in vals:
            ext = os.path.splitext(v)[1].lower()
            if ext in [".pt", ".pth"]:
                score += 3
            elif ext in [".h5", ".hdf5"]:
                score += 3
            if os.path.isabs(v):
                score += 1
        if score > best_score:
            best_score = score
            best_col = c

    return best_col if best_col is not None else existing[0]


def _normalize_split_values(split: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(split, str):
        return [split.strip().lower()]
    return [str(s).strip().lower() for s in split]


def _load_feature_file(feat_path: str) -> torch.Tensor:
    feat_path = str(feat_path).strip()

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}")

    ext = os.path.splitext(feat_path)[1].lower()

    # PyTorch tensor / dict
    if ext in [".pt", ".pth", ".bin", ".ckpt", ""]:
        try:
            feats = torch.load(feat_path, map_location="cpu")
        except Exception as e:
            with open(feat_path, "rb") as f:
                first_bytes = f.read(16)
            raise RuntimeError(
                f"Failed to torch.load file: {feat_path}\n"
                f"Extension: {ext}\n"
                f"First 16 bytes: {first_bytes}\n"
                f"Original error: {e}"
            ) from e

        if isinstance(feats, dict):
            if "features" in feats:
                feats = feats["features"]
            elif "feats" in feats:
                feats = feats["feats"]
            elif "embeddings" in feats:
                feats = feats["embeddings"]
            else:
                raise ValueError(
                    f"Unsupported feature dict keys for {feat_path}: {list(feats.keys())}"
                )

        if not torch.is_tensor(feats):
            raise TypeError(f"Loaded object is not a tensor: {type(feats)} from {feat_path}")

        return feats.float()

    # HDF5 feature
    if ext in [".h5", ".hdf5"]:
        if h5py is None:
            raise ImportError("h5py is required to load .h5/.hdf5 feature files.")

        with h5py.File(feat_path, "r") as f:
            if "features" in f:
                feats = f["features"][:]
            elif "feats" in f:
                feats = f["feats"][:]
            elif "embeddings" in f:
                feats = f["embeddings"][:]
            else:
                raise ValueError(
                    f"Unsupported HDF5 keys for {feat_path}: {list(f.keys())}"
                )

        return torch.tensor(feats, dtype=torch.float32)

    raise ValueError(f"Unsupported feature file extension: {ext} ({feat_path})")


class SlideCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        taxonomy,
        opened_fine_labels: Optional[List[str]] = None,
        split: Optional[Union[str, Sequence[str]]] = None,
        split_col: str = "split",
        dataset_filter: Optional[Union[str, Sequence[str]]] = None,
        verbose: bool = True,
    ):
        """
        split-version csv 사용 예시:
            SlideCSVDataset(
                "./data/tcga_continual.csv",
                taxonomy,
                opened_fine_labels=["LUAD", "LUSC"],
                split="train",
                split_col="split",
            )

        기존 train.csv/test.csv처럼 split column이 없는 파일도 split=None이면 그대로 사용 가능.
        """
        self.csv_path = csv_path
        self.taxonomy = taxonomy
        self.split = split
        self.split_col = split_col

        self.df = pd.read_csv(csv_path, low_memory=False)

        if split is not None:
            if split_col not in self.df.columns:
                raise KeyError(
                    f"split='{split}' was requested, but split_col='{split_col}' is not in CSV. "
                    f"Available columns: {list(self.df.columns)}"
                )
            wanted = set(_normalize_split_values(split))
            split_values = self.df[split_col].astype(str).str.strip().str.lower()
            before = len(self.df)
            self.df = self.df[split_values.isin(wanted)].reset_index(drop=True)
            if len(self.df) == 0:
                available = sorted(split_values.unique().tolist())
                raise ValueError(
                    f"No rows left after split filtering. requested={sorted(wanted)}, "
                    f"available={available}, csv={csv_path}"
                )
            if verbose:
                print(f"[SlideCSVDataset] split filter: {split_col} in {sorted(wanted)} -> {len(self.df)}/{before}")

        self.feature_col = _choose_feature_column(self.df)

        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[self.feature_col]).reset_index(drop=True)
        if len(self.df) < initial_count and verbose:
            print(f"[Warning] Dropped {initial_count - len(self.df)} rows with NaN in {self.feature_col}")
        # --------------------------

        self.fine_col = _find_column(self.df, FINE_COL_CANDIDATES, required=True)
        self.coarse_col = _find_column(self.df, COARSE_COL_CANDIDATES, required=True)
        self.task_col = _find_column(self.df, TASK_COL_CANDIDATES, required=False)
        self.slide_col = _find_column(self.df, SLIDE_COL_CANDIDATES, required=False)

        if dataset_filter is not None and "dataset" in self.df.columns:
            allowed_ds = {str(d).strip() for d in ([dataset_filter] if isinstance(dataset_filter, str) else dataset_filter)}
            before = len(self.df)
            self.df = self.df[self.df["dataset"].astype(str).str.strip().isin(allowed_ds)].reset_index(drop=True)
            if verbose:
                print(f"[SlideCSVDataset] dataset filter: {sorted(allowed_ds)} -> {len(self.df)}/{before}")

        if opened_fine_labels is not None:
            allowed = set(map(str, opened_fine_labels))
            before = len(self.df)
            self.df = self.df[self.df[self.fine_col].astype(str).isin(allowed)].reset_index(drop=True)
            if len(self.df) == 0:
                raise ValueError(
                    f"No rows left after opened_fine_labels filtering. "
                    f"allowed={sorted(allowed)}, split={split}, csv={csv_path}"
                )
            if verbose:
                print(f"[SlideCSVDataset] fine filter: {len(self.df)}/{before}")

        if verbose:
            print(f"[SlideCSVDataset] csv         = {csv_path}")
            print(f"[SlideCSVDataset] feature_col = {self.feature_col}")
            print(f"[SlideCSVDataset] fine_col    = {self.fine_col}")
            print(f"[SlideCSVDataset] coarse_col  = {self.coarse_col}")
            print(f"[SlideCSVDataset] task_col    = {self.task_col}")
            print(f"[SlideCSVDataset] split       = {split}")
            print(f"[SlideCSVDataset] num_samples = {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        feat_path = str(row[self.feature_col]).strip()
        fine_label = str(row[self.fine_col]).strip()
        coarse_label = str(row[self.coarse_col]).strip()
        task_label = str(row[self.task_col]).strip() if self.task_col is not None else "default"
        slide_id = str(row[self.slide_col]).strip() if self.slide_col is not None else feat_path

        feats = _load_feature_file(feat_path)

        if feats.ndim != 2:
            raise ValueError(
                f"Feature tensor must have shape [N, D], got {tuple(feats.shape)} from {feat_path}"
            )

        dataset_name = str(row["dataset"]).strip() if "dataset" in self.df.columns else "unknown"

        item = {
            "slide_id": slide_id,
            "feature_path": feat_path,
            "feats": feats,
            "fine_label": fine_label,
            "coarse_label": coarse_label,
            "task_label": task_label,
            "dataset": dataset_name,
        }

        if self.taxonomy is not None:
            item["fine_id"] = self.taxonomy.fine_to_idx[fine_label]
            item["coarse_id"] = self.taxonomy.coarse_to_idx[coarse_label]
        else:
            item["fine_id"] = -1
            item["coarse_id"] = -1

        return item


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "slide_ids": [x["slide_id"] for x in batch],
        "feature_paths": [x["feature_path"] for x in batch],
        "feats_list": [x["feats"] for x in batch],
        "fine_labels": [x["fine_label"] for x in batch],
        "coarse_labels": [x["coarse_label"] for x in batch],
        "task_labels": [x.get("task_label", "default") for x in batch],
        "datasets": [x.get("dataset", "unknown") for x in batch],
        "fine_ids": torch.tensor([x["fine_id"] for x in batch], dtype=torch.long),
        "coarse_ids": torch.tensor([x["coarse_id"] for x in batch], dtype=torch.long),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../data/tcga_continual.csv")
    parser.add_argument("--split_col", type=str, default="split")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"])
    args = parser.parse_args()

    for split in args.splits:
        dataset = SlideCSVDataset(
            args.csv,
            taxonomy=None,
            opened_fine_labels=None,
            split=split,
            split_col=args.split_col,
        )
        print(f"\n[{split.upper()}]")
        coarse_counts = Counter(dataset.df[dataset.coarse_col])
        fine_counts = Counter(dataset.df[dataset.fine_col])
        print("Coarse label distribution:")
        for label, count in coarse_counts.most_common():
            print(f"  {label}: {count}")
        print("Fine label distribution:")
        for label, count in fine_counts.most_common():
            print(f"  {label}: {count}")
