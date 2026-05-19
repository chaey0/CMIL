import torch
import torch.nn as nn
import torch.nn.functional as F

from .knowledge_tree import KnowledgeTree
from .mil import SimpleMILAggregator, OrganTaskMILAdaptorBank
from .text_encoder import KEEPEncoder
from .hyperbolic import hyperbolic_logits_from_tangent

class KnowledgeTreeCLModel(nn.Module):
    def __init__(self, taxonomy, cfg):
        super().__init__()
        self.taxonomy = taxonomy
        self.cfg = cfg
        self.embed_dim = int(cfg["model"].get("embed_dim", 768))
        self.mil_dim = int(cfg["model"].get("mil_hidden_dim", 512))
        self.align_temperature = float(cfg.get("train", {}).get("align_temperature", 0.07))

        self.text_encoder = KEEPEncoder()
        self.tree = KnowledgeTree(taxonomy, embed_dim=self.embed_dim)

        self.mil = SimpleMILAggregator(
            dim=self.embed_dim,
            hidden_dim=self.mil_dim,
            dropout=cfg["model"].get("mil_dropout", 0.1),
        )

        model_cfg = cfg.get("model", {})

        self.use_organ_task_adaptor = bool(model_cfg.get("use_mil_lora", True))
        self.organ_task_adaptor_mode = str(model_cfg.get("organ_task_adaptor_mode", "branch")).lower()
        self.organ_task_adaptor_merge = str(model_cfg.get("organ_task_adaptor_merge", "sum")).lower()

        adaptor_rank = int(model_cfg.get("mil_lora_r", 8))
        adaptor_alpha = float(model_cfg.get("mil_lora_alpha", adaptor_rank))
        adaptor_dropout = float(model_cfg.get("mil_lora_dropout", 0.05))
        

        # scale은 특별히 직접 지정할 때만 사용.지정하지 않으면 alpha / rank 사용.
        adaptor_scale = model_cfg.get("organ_task_adaptor_scale", None)
        if adaptor_scale is not None:
            adaptor_scale = float(adaptor_scale)

        self.organ_task_adaptors = OrganTaskMILAdaptorBank(
            dim=self.mil_dim,
            rank=adaptor_rank,
            alpha=adaptor_alpha,
            dropout=adaptor_dropout,
            scale=adaptor_scale,
        )

        print(
            f"[Model] OrganTask MIL Adaptor | "
            f"use={self.use_organ_task_adaptor} | "
            f"mode={self.organ_task_adaptor_mode} | "
            f"merge={self.organ_task_adaptor_merge} | "
            f"rank={adaptor_rank} | "
            f"alpha={adaptor_alpha} | "
            f"scale={(adaptor_scale if adaptor_scale is not None else adaptor_alpha / adaptor_rank):.4f}"
        )

        self.visual_alignment_projector = nn.Sequential(
            nn.Linear(self.mil_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.fine_head = nn.Linear(self.embed_dim, 1)

        self.active_coarse_labels = []
        self.active_fine_labels = []
        self.active_coarse_to_local = {}
        self.active_fine_to_local = {}
        self.task_to_fine_labels = {}
        self.task_to_local_fine = {}

        self.task_logit_log_scale = nn.ParameterDict()
        self.task_logit_bias = nn.ParameterDict()

        hyp_cfg = cfg.get("hyperbolic", {})
        self.hyp_cfg = hyp_cfg
        self.hyp_enabled = bool(hyp_cfg.get("enabled", False))
        self.hyp_mode = str(hyp_cfg.get("mode", "aux")).lower()

        if not self.hyp_enabled:
            self.hyp_mode = "off"

        self.hyp_text_scale = float(hyp_cfg.get("fine_tangent_scale", hyp_cfg.get("text_tangent_scale", 1.0)))
        self.anchor_space = "euclidean"

        

    # ------------------------------------------------------------------
    # parameter control
    # ------------------------------------------------------------------
    def freeze_all_params(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_shared_mil(self):
        for p in self.mil.parameters():
            p.requires_grad = True

    def unfreeze_visual_alignment_projector(self):
        for p in self.visual_alignment_projector.parameters():
            p.requires_grad = True

    def unfreeze_classifier(self):
        if self.use_hyperbolic_classifier():
            return
        for p in self.fine_head.parameters():
            p.requires_grad = True

    def train_only_current_organ_task_adaptor(self, task_key):
        self.organ_task_adaptors.train_only(task_key)

    # ------------------------------------------------------------------
    # text bank / classifier
    # ------------------------------------------------------------------
    @torch.no_grad()
    def activate_nodes(self, coarse_labels, fine_labels, device, refresh_existing=False):
        self.tree.initialize_nodes(
            text_encoder=self.text_encoder,
            coarse_labels=list(coarse_labels),
            fine_labels=list(fine_labels),
            device=device,
            refresh_existing=refresh_existing,
        )
        self.active_coarse_labels = list(coarse_labels)
        self.active_coarse_to_local = {lab: i for i, lab in enumerate(self.active_coarse_labels)}

    def _get_bank_by_labels(self, labels, bank_type="fine", device=None, normalize=True):
        if bank_type == "fine":
            idx = [self.taxonomy.fine_labels.index(lab) for lab in labels]
            bank = self.tree.fine_bank[idx]
        elif bank_type == "coarse":
            idx = [self.taxonomy.coarse_labels.index(lab) for lab in labels]
            bank = self.tree.coarse_bank[idx]
        else:
            raise ValueError(f"Unsupported bank_type: {bank_type}")

        if normalize:
            bank = F.normalize(bank, dim=-1)

        return bank.to(device) if device is not None else bank


    @torch.no_grad()
    def _init_rows_from_text_bank(self, head, labels, row_offset=0, bank_type="fine"):
        text_bank = self._get_bank_by_labels(labels, bank_type=bank_type, device=head.weight.device)
        head.weight[row_offset: row_offset + len(labels)].copy_(text_bank)
        if head.bias is not None:
            head.bias[row_offset: row_offset + len(labels)].zero_()

    def expand_classifiers(self, opened_fine_labels, device):
        opened_fine_labels = list(opened_fine_labels)
        old_labels = list(self.active_fine_labels)
        old_n, new_n = len(old_labels), len(opened_fine_labels)

        if old_n == 0:
            self.fine_head = nn.Linear(self.embed_dim, new_n).to(device)
            self._init_rows_from_text_bank(self.fine_head, opened_fine_labels, 0, "fine")
        elif new_n != old_n:
            old_head = self.fine_head
            new_head = nn.Linear(self.embed_dim, new_n).to(device)
            with torch.no_grad():
                new_head.weight[:old_n].copy_(old_head.weight.data[:old_n])
                new_head.bias[:old_n].copy_(old_head.bias.data[:old_n])
                self._init_rows_from_text_bank(new_head, opened_fine_labels[old_n:], old_n, "fine")
            self.fine_head = new_head.to(device)

        self.active_fine_labels = opened_fine_labels
        self.active_fine_to_local = {lab: i for i, lab in enumerate(self.active_fine_labels)}
        self._refresh_task_local_indices(device)

    def use_hyperbolic_classifier(self):
        """Return True only when hyperbolic is the primary classifier (not aux mode)."""
        return self.hyp_enabled and self.hyp_mode not in {"aux", "off"}
    def use_merged_organ_task_adaptors(self):
        return self.organ_task_adaptor_mode in {
            "merge",
            "merged",
            "sum",
            "all",
            "all_active",
        }
        
    def _compute_fine_logits_full(self, raw_h, h):
        if self.use_hyperbolic_classifier():
            bank_tangent = self.get_active_fine_bank_tangent(raw_h.device)
            return hyperbolic_logits_from_tangent(
                raw_h,
                bank_tangent,
                self.hyp_cfg,
            )

        return self.fine_head(h)

    def _refresh_task_local_indices(self, device):
        self.task_to_local_fine = {}
        for task_key, labels in self.task_to_fine_labels.items():
            valid = [lab for lab in labels if lab in self.active_fine_to_local]
            if valid:
                self.task_to_local_fine[task_key] = torch.tensor(
                    [self.active_fine_to_local[lab] for lab in valid],
                    dtype=torch.long,
                    device=device,
                )

    def add_organ_task(self, task_key, fine_labels, device):
        task_key = str(task_key)
        fine_labels = list(fine_labels)

        self.organ_task_adaptors.add(task_key, device=device)
        self.task_to_fine_labels[task_key] = fine_labels

        if self.hyp_enabled:
            self.tree.initialize_task_node(self.text_encoder, task_key, device, fine_labels=fine_labels)

        self._refresh_task_local_indices(device)

    def get_opened_task_keys(self):
        return list(self.task_to_fine_labels.keys())

    def global_fine_to_local(self, global_ids):
        labels = [self.taxonomy.fine_labels[int(i)] for i in global_ids.detach().cpu().tolist()]
        return torch.tensor(
            [self.active_fine_to_local[l] for l in labels],
            device=global_ids.device,
            dtype=torch.long,
        )

    def get_active_fine_bank(self, device=None):
        device = device or next(self.parameters()).device
        return self._get_bank_by_labels(self.active_fine_labels, bank_type="fine", device=device)

    def get_active_fine_bank_tangent(self, device=None):
        device = device or next(self.parameters()).device

        bank = self._get_bank_by_labels(
            self.active_fine_labels,
            bank_type="fine",
            device=device,
            normalize=False,
        )

        # If text refinement already produced tangent-space anchors, keep radius.
        if str(getattr(self, "anchor_space", "euclidean")).lower() in {
            "hyperbolic_tangent",
            "tangent",
            "poincare_tangent",
            "lorentz_tangent",
        }:
            return bank

        # Otherwise use normalized KEEP direction and assign tangent radius.
        return F.normalize(bank, dim=-1) * self.hyp_text_scale

    def get_active_coarse_bank_tangent(self, device=None):
        """Coarse (organ) embeddings as tangent vectors at coarse_tangent_scale."""
        device = device or next(self.parameters()).device
        coarse_labels = self._hierarchy_coarse_order()
        bank = self._get_bank_by_labels(
            coarse_labels, bank_type="coarse", device=device, normalize=False,
        )
        scale = float(self.hyp_cfg.get("coarse_tangent_scale", 0.3))
        return F.normalize(bank, dim=-1) * scale

    def get_active_task_bank_tangent(self, device=None):
        """Task node tangent vectors using actual task text embeddings.

        Returns (task_keys: List[str], tangent: Tensor [T, D]).
        Falls back to averaging fine children if task embedding is unavailable.
        """
        device = device or next(self.parameters()).device
        scale = float(self.hyp_cfg.get("task_tangent_scale", 0.5))
        active_set = set(self.active_fine_labels)

        task_keys, task_embs = [], []
        for task_key in self.task_to_fine_labels:
            fine_labs = [l for l in self.task_to_fine_labels[task_key] if l in active_set]
            if not fine_labs:
                continue
            task_keys.append(task_key)

            emb = self.tree.get_task_emb(task_key)
            if emb is not None:
                task_embs.append(F.normalize(emb.to(device), dim=-1))
            else:
                # Fallback: average of normalized fine children
                bank = self._get_bank_by_labels(fine_labs, "fine", device=device, normalize=True)
                task_embs.append(F.normalize(bank.mean(dim=0), dim=-1))

        if not task_embs:
            return [], torch.empty(0, self.embed_dim, device=device)

        return task_keys, torch.stack(task_embs) * scale

    # -- hierarchy helpers ------------------------------------------------

    def _hierarchy_coarse_order(self):
        """Unique coarse labels in order of first appearance among active fine labels."""
        order, seen = [], set()
        for lab in self.active_fine_labels:
            c = self.taxonomy.fine_to_coarse.get(lab, lab)
            if c not in seen:
                order.append(c)
                seen.add(c)
        return order

    @torch.no_grad()
    def get_hierarchy_info(self, device=None):
        """Build hierarchical tangent banks and fine→coarse/task index mappings.

        Returns dict with:
          coarse_labels, coarse_tangent  [C, D]
          task_keys,     task_tangent    [T, D]
          fine_to_coarse_idx             [F] — local fine idx → coarse idx
          fine_to_task_idx               [F] — local fine idx → task idx
        """
        device = device or next(self.parameters()).device

        coarse_labels = self._hierarchy_coarse_order()
        coarse_tangent = self.get_active_coarse_bank_tangent(device)

        task_keys, task_tangent = self.get_active_task_bank_tangent(device)

        coarse_to_idx = {c: i for i, c in enumerate(coarse_labels)}
        task_to_idx = {t: i for i, t in enumerate(task_keys)}

        # Build fine → task mapping (which task owns each fine label)
        fine_label_to_task = {}
        for tk, labs in self.task_to_fine_labels.items():
            for l in labs:
                fine_label_to_task[l] = tk

        fine_to_coarse_idx, fine_to_task_idx = [], []
        for lab in self.active_fine_labels:
            c = self.taxonomy.fine_to_coarse.get(lab, lab)
            fine_to_coarse_idx.append(coarse_to_idx.get(c, 0))
            tk = fine_label_to_task.get(lab, "")
            fine_to_task_idx.append(task_to_idx.get(tk, 0))

        return {
            "coarse_labels": coarse_labels,
            "coarse_tangent": coarse_tangent,
            "task_keys": task_keys,
            "task_tangent": task_tangent,
            "fine_to_coarse_idx": torch.tensor(fine_to_coarse_idx, dtype=torch.long, device=device),
            "fine_to_task_idx": torch.tensor(fine_to_task_idx, dtype=torch.long, device=device),
        }

    def get_active_text_banks(self):
        device = next(self.parameters()).device
        coarse = self._get_bank_by_labels(self.active_coarse_labels, "coarse", device)
        fine = self._get_bank_by_labels(self.active_fine_labels, "fine", device)
        return coarse, fine

    # ------------------------------------------------------------------
    # forward branches
    # ------------------------------------------------------------------
    def encode_shared(self, feats_list):
        z_list, attn_list = [], []
        for feats in feats_list:
            z_i, attn_i = self.mil(feats)
            z_list.append(z_i.squeeze(0) if z_i.dim() == 2 and z_i.size(0) == 1 else z_i)
            attn_list.append(attn_i)
        return torch.stack(z_list, dim=0), attn_list

    def _project(self, z):
        raw = self.visual_alignment_projector(z)
        return raw, F.normalize(raw, dim=-1)

    def _branch_from_z(self, z_shared, task_key):
        z_task = self.organ_task_adaptors(z_shared, task_key)
        raw_h, h = self._project(z_task)

        logits = self._compute_fine_logits_full(raw_h, h)

        return {
            "z_task": z_task,
            "raw_h_f": raw_h,
            "h_f": h,
            "fine_logits_full": logits,
            "fine_bank": self.get_active_fine_bank(z_shared.device),
            "fine_bank_tangent": self.get_active_fine_bank_tangent(z_shared.device),
        }

    def _merged_from_z(self, z_shared, opened_task_keys=None):
        current_session_idx = int(getattr(self, "_current_session_idx", -1))

        if current_session_idx == 0 or not self.use_organ_task_adaptor:
            z_task = z_shared
        else:
            opened_task_keys = [
                str(k) for k in (opened_task_keys or self.get_opened_task_keys())
            ]
            z_task = self.organ_task_adaptors.forward_merged(
                z_shared,
                task_keys=opened_task_keys,
                merge=self.organ_task_adaptor_merge,
            )

        raw_h, h = self._project(z_task)
        logits = self._compute_fine_logits_full(raw_h, h)

        return {
            "z": z_shared,
            "z_shared": z_shared,
            "z_task": z_task,
            "raw_h_f": raw_h,
            "h_f": h,
            "fine_logits": logits,
            "fine_logits_full": logits,
            "fine_bank": self.get_active_fine_bank(z_shared.device),
            "fine_bank_tangent": self.get_active_fine_bank_tangent(z_shared.device),
        }

    def _empty_branch_output(self, z_shared):
        b = z_shared.size(0)
        c = len(self.active_fine_labels)
        return (
            z_shared.new_full((b, c), -1e9),
            z_shared.new_zeros((b, self.embed_dim)),
            z_shared.new_zeros((b, self.embed_dim)),
        )
    
    def mask_logits_by_task_keys(self, logits, task_keys):
        task_keys = [str(k) for k in task_keys]
        masked = logits.new_full(logits.shape, -1e9)

        for task_key in sorted(set(task_keys)):
            if task_key not in self.task_to_local_fine:
                raise KeyError(f"Task key has no active labels: {task_key}")

            rows = torch.tensor(
                [i for i, k in enumerate(task_keys) if k == task_key],
                dtype=torch.long,
                device=logits.device,
            )
            cols = self.task_to_local_fine[task_key].to(logits.device)

            masked[rows[:, None], cols[None, :]] = logits[rows[:, None], cols[None, :]]

        return masked
    
    def _safe_task_key(self, task_key):
        return str(task_key).replace("/", "__").replace(".", "_")


    def forward_task_from_z(self, z_shared, task_keys):
        task_keys = [str(k) for k in task_keys]

        if self.use_merged_organ_task_adaptors():
            out = self.forward_global_from_z(
                z_shared,
                opened_task_keys=self.get_opened_task_keys(),
            )
            out["fine_logits"] = self.mask_logits_by_task_keys(
                out["fine_logits"],
                task_keys,
            )
            return out

        logits, raw_h, h = self._empty_branch_output(z_shared)

        for task_key in sorted(set(task_keys)):
            if task_key not in self.task_to_local_fine:
                raise KeyError(f"Task key has no active labels: {task_key}")

            rows = torch.tensor(
                [i for i, k in enumerate(task_keys) if k == task_key],
                dtype=torch.long,
                device=z_shared.device,
            )

            out = self._branch_from_z(z_shared.index_select(0, rows), task_key)
            cols = self.task_to_local_fine[task_key].to(z_shared.device)

            logits[rows[:, None], cols[None, :]] = out["fine_logits_full"][:, cols]
            raw_h[rows] = out["raw_h_f"]
            h[rows] = out["h_f"]

        return {
            "z": z_shared,
            "z_shared": z_shared,
            "raw_h_f": raw_h,
            "h_f": h,
            "fine_logits": logits,
            "fine_bank": self.get_active_fine_bank(z_shared.device),
            "fine_bank_tangent": self.get_active_fine_bank_tangent(z_shared.device),
        }

    def forward_task(self, feats_list, task_keys):
        z_shared, attn = self.encode_shared(feats_list)
        out = self.forward_task_from_z(z_shared, task_keys)
        out["attn"] = attn
        return out

    def forward_global_from_z(self, z_shared, opened_task_keys=None):
        opened_task_keys = [
            str(k) for k in (opened_task_keys or self.get_opened_task_keys())
        ]

        if self.use_merged_organ_task_adaptors():
            return self._merged_from_z(z_shared, opened_task_keys)

        b, c = z_shared.size(0), len(self.active_fine_labels)
        logits = z_shared.new_full((b, c), -1e9)

        for task_key in opened_task_keys:
            if task_key not in self.task_to_local_fine:
                raise KeyError(f"Task key has no active labels: {task_key}")

            out = self._branch_from_z(z_shared, task_key)
            cols = self.task_to_local_fine[task_key].to(z_shared.device)
            logits[:, cols] = out["fine_logits_full"][:, cols]

        return {
            "z": z_shared,
            "z_shared": z_shared,
            "fine_logits": logits,
            "fine_bank": self.get_active_fine_bank(z_shared.device),
            "fine_bank_tangent": self.get_active_fine_bank_tangent(z_shared.device),
        }

    def forward_global(self, feats_list, opened_task_keys=None):
        z_shared, attn = self.encode_shared(feats_list)
        out = self.forward_global_from_z(z_shared, opened_task_keys)
        out["attn"] = attn
        return out

    def forward_replay_task(self, z_shared, task_keys):
        return self.forward_task_from_z(z_shared, task_keys)

    def forward(self, feats_list, task_keys=None, eval_scope="global", opened_task_keys=None):
        z_shared, attn = self.encode_shared(feats_list)

        out = self.forward_global_from_z(
            z_shared=z_shared,
            opened_task_keys=opened_task_keys,
        )
        out["attn"] = attn

        if str(eval_scope).lower() == "task":
            if task_keys is None:
                raise ValueError("task_keys are required for task-scope forward.")
            out["fine_logits"] = self.mask_logits_by_task_keys(
                out["fine_logits"],
                task_keys,
            )

        return out