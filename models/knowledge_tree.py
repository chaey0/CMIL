import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class KnowledgeTree(nn.Module):
    def __init__(
        self,
        taxonomy,
        embed_dim=768,
    ):
        super().__init__()
        self.taxonomy = taxonomy
        self.embed_dim = embed_dim

        self.register_buffer(
            "coarse_bank",
            torch.zeros(len(taxonomy.coarse_labels), embed_dim)
        )
        self.register_buffer(
            "fine_bank",
            torch.zeros(len(taxonomy.fine_labels), embed_dim)
        )

        self.register_buffer(
            "coarse_initialized",
            torch.zeros(len(taxonomy.coarse_labels), dtype=torch.bool)
        )
        self.register_buffer(
            "fine_initialized",
            torch.zeros(len(taxonomy.fine_labels), dtype=torch.bool)
        )

        # Task embeddings stored as plain dict (not a fixed-size buffer since
        # task keys come from sessions config, not taxonomy). Re-encoded on load.
        self._task_embs: Dict[str, torch.Tensor] = {}

        self.register_buffer("fine_to_coarse_idx", self._build_fine_to_coarse_idx())
        self.register_buffer("fine_to_coarse_matrix", self._build_fine_to_coarse_matrix())

    def _build_fine_to_coarse_idx(self):
        idx = torch.zeros(len(self.taxonomy.fine_labels), dtype=torch.long)
        for fine in self.taxonomy.fine_labels:
            fi = self.taxonomy.fine_to_idx[fine]
            ci = self.taxonomy.coarse_to_idx[self.taxonomy.fine_to_coarse[fine]]
            idx[fi] = ci
        return idx

    def _build_fine_to_coarse_matrix(self):
        num_fine = len(self.taxonomy.fine_labels)
        num_coarse = len(self.taxonomy.coarse_labels)
        M = torch.zeros(num_fine, num_coarse)
        for fine in self.taxonomy.fine_labels:
            fi = self.taxonomy.fine_to_idx[fine]
            ci = self.taxonomy.coarse_to_idx[self.taxonomy.fine_to_coarse[fine]]
            M[fi, ci] = 1.0
        return M

    @torch.no_grad()
    def initialize_nodes(
        self,
        text_encoder,
        coarse_labels,
        fine_labels,
        device,
        refresh_existing=False,
    ):
        # 1) coarse encode
        coarse_to_encode = []
        for label in coarse_labels:
            idx = self.taxonomy.coarse_to_idx[label]
            already = bool(self.coarse_initialized[idx].item())
            if refresh_existing or (not already):
                coarse_to_encode.append(label)

        if coarse_to_encode:
            texts = [self.taxonomy.get_coarse_text(l) for l in coarse_to_encode]
            embs = text_encoder.encode_texts(texts, device=device)
            embs = F.normalize(embs, dim=-1)

            for label, emb in zip(coarse_to_encode, embs):
                idx = self.taxonomy.coarse_to_idx[label]
                self.coarse_bank[idx].copy_(emb)
                self.coarse_initialized[idx] = True

        # 2) fine encode
        fine_to_encode = []
        for label in fine_labels:
            idx = self.taxonomy.fine_to_idx[label]
            already = bool(self.fine_initialized[idx].item())
            if refresh_existing or (not already):
                fine_to_encode.append(label)

        if fine_to_encode:
            texts = [self.taxonomy.get_fine_text(l) for l in fine_to_encode]
            embs = text_encoder.encode_texts(texts, device=device)
            embs = F.normalize(embs, dim=-1)

            for label, emb in zip(fine_to_encode, embs):
                idx = self.taxonomy.fine_to_idx[label]
                self.fine_bank[idx].copy_(emb)
                self.fine_initialized[idx] = True

    @torch.no_grad()
    def initialize_task_node(self, text_encoder, task_key: str, device, fine_labels=None):
        """Encode task description text and store as unit embedding.

        task_key format: "ORGAN/task_name"  (e.g. "BRAIN/brain_glioma_subtype")
        fine_labels: list of fine label strings for this task; passed to get_task_text so
                     task_axis can be resolved from fine node metadata rather than the key string.
        Skips if already initialized.
        """
        if task_key in self._task_embs:
            return
        if not hasattr(self.taxonomy, "get_task_text"):
            return
        text = self.taxonomy.get_task_text(task_key, fine_labels=fine_labels)
        emb = text_encoder.encode_texts([text], device=device)  # [1, D]
        self._task_embs[task_key] = F.normalize(emb[0].cpu(), dim=-1)

    def get_task_emb(self, task_key: str) -> Optional[torch.Tensor]:
        """Return stored unit embedding for task_key, or None if not initialized."""
        return self._task_embs.get(task_key, None)