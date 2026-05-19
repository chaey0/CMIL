import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMILAggregator(nn.Module):
    def __init__(self, dim=768, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.L = hidden_dim
        self.D = 256
        self.K = 1

        self.feature = nn.Sequential(
            nn.Linear(dim, self.L),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_v = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attn_w = nn.Linear(self.D, self.K)
        self.post = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.LayerNorm(self.L),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.L, self.L),
        )

    def forward(self, h):
        h_feat = self.feature(h)
        a = self.attn_w(self.attn_v(h_feat) * self.attn_u(h_feat))
        a = F.softmax(a, dim=0)
        z = torch.sum(a * h_feat, dim=0, keepdim=True)
        return self.post(z), a


class OrganTaskMILAdaptor(nn.Module):
    """Residual low-rank OrganTask MIL Adaptor.

    Same behavior as LoRA-style residual:
        A(z) = (alpha / rank) * W_up W_down z
        z_out = z + A(z)
    """

    def __init__(self, dim, rank=8, alpha=16, dropout=0.05, scale=None):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.alpha = float(alpha)

        if scale is None:
            self.scale = self.alpha / float(self.rank)
        else:
            self.scale = float(scale)

        self.down = nn.Linear(dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, dim, bias=False)
        self.drop = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, z):
        return self.scale * self.up(self.drop(self.down(z)))


class OrganTaskMILAdaptorBank(nn.Module):
    def __init__(self, dim, rank=8, alpha=16, dropout=0.05, scale=None):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.scale = scale
        self.adaptors = nn.ModuleDict()

    def add(self, task_key, device=None):
        task_key = str(task_key)

        if task_key not in self.adaptors:
            adaptor = OrganTaskMILAdaptor(
                dim=self.dim,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                scale=self.scale,
            )

            if device is not None:
                adaptor = adaptor.to(device)

            self.adaptors[task_key] = adaptor

    def has(self, task_key):
        return str(task_key) in self.adaptors

    def keys(self):
        return list(self.adaptors.keys())

    def forward(self, z, task_key):
        task_key = str(task_key)

        if task_key not in self.adaptors:
            raise KeyError(f"Unknown OrganTask MIL Adaptor: {task_key}")

        return z + self.adaptors[task_key](z)

    def forward_merged(self, z, task_keys=None, merge="sum"):
        """
        Merge all active OrganTask MIL Adaptors.

        sum:
            z + Σ A_t(z)

        mean:
            z + mean_t A_t(z)

        sqrt:
            z + Σ A_t(z) / sqrt(num_tasks)
        """
        if task_keys is None:
            task_keys = list(self.adaptors.keys())

        task_keys = [str(k) for k in task_keys if str(k) in self.adaptors]

        if len(task_keys) == 0:
            return z

        delta = z.new_zeros(z.shape)

        for task_key in task_keys:
            delta = delta + self.adaptors[task_key](z)

        merge = str(merge).lower()

        if merge == "sum":
            pass
        elif merge == "mean":
            delta = delta / float(len(task_keys))
        elif merge == "sqrt":
            delta = delta / (float(len(task_keys)) ** 0.5)
        else:
            raise ValueError(f"Unknown adaptor merge mode: {merge}")

        return z + delta

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def register_base_subspace(self, reference_weight):
        """Session 0 (MIL full tuning) 후, reference weight의 SVD로 base subspace 추출.

        Visual alignment projector의 첫 번째 Linear weight를 사용하면,
        base model이 z-space에서 의존하는 principal directions을 얻을 수 있습니다.
        이후 adaptor들은 이 방향과 직교하도록 강제됩니다.

        Args:
            reference_weight: [out_dim, dim] 형태의 weight tensor.
                보통 visual_alignment_projector[0].weight (768×512).
        """
        W = reference_weight.detach().float()
        # SVD: W = U @ S @ Vh
        # Vh의 상위 rank개 행 = z-space에서 가장 중요한 방향
        _, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self.register_buffer("_base_subspace", Vh[: self.rank].clone())
        print(
            f"[OWLoRA] Base subspace registered: "
            f"top-{self.rank} singular values = "
            f"{S[:self.rank].tolist()}"
        )

    def orthogonal_loss(self, exclude_keys=None):
        """Inter-task + base-subspace orthogonal regularization.

        두 가지 항으로 구성됩니다:

        1) Inter-adaptor: 학습된 adaptor 쌍 간 직교
           Σ_{i≠j} ( ||D_i D_j^T||²_F + ||U_i^T U_j||²_F )

        2) Base-subspace: 각 학습된 adaptor가 base subspace와 직교
           Σ_k ( ||B D_k^T||²_F + ||U_k^T B^T||²_F )

           여기서 B = SVD로 추출한 base subspace [rank, dim]

        Args:
            exclude_keys: 제외할 task key 집합 (e.g. session 0 adaptor).

        Returns:
            Scalar tensor.
        """
        exclude = set(exclude_keys or [])
        keys = [k for k in self.adaptors.keys() if k not in exclude]

        has_base = hasattr(self, "_base_subspace") and self._base_subspace is not None

        if len(keys) < 2 and not has_base:
            return next(self.parameters()).new_tensor(0.0)
        if len(keys) == 0:
            return next(self.parameters()).new_tensor(0.0)

        loss = next(self.parameters()).new_tensor(0.0)
        n_pairs = 0

        # (1) Inter-adaptor orthogonality
        for i in range(len(keys)):
            di = self.adaptors[keys[i]].down.weight          # [rank, dim]
            ui = self.adaptors[keys[i]].up.weight             # [dim, rank]

            for j in range(i + 1, len(keys)):
                dj = self.adaptors[keys[j]].down.weight       # [rank, dim]
                uj = self.adaptors[keys[j]].up.weight          # [dim, rank]

                loss = loss + (di @ dj.T).pow(2).sum()
                loss = loss + (ui.T @ uj).pow(2).sum()
                n_pairs += 1

        # (2) Base-subspace orthogonality
        if has_base:
            base = self._base_subspace                         # [rank, dim]
            for k in keys:
                dk = self.adaptors[k].down.weight              # [rank, dim]
                uk = self.adaptors[k].up.weight                # [dim, rank]

                # adaptor의 input selection이 base subspace와 직교
                loss = loss + (base @ dk.T).pow(2).sum()
                # adaptor의 output mapping이 base subspace와 직교
                loss = loss + (uk.T @ base.T).pow(2).sum()
                n_pairs += 1

        return loss / max(n_pairs, 1)

    def train_only(self, task_key):
        self.freeze_all()

        task_key = str(task_key)
        if task_key not in self.adaptors:
            raise KeyError(f"Unknown OrganTask MIL Adaptor: {task_key}")

        for p in self.adaptors[task_key].parameters():
            p.requires_grad = True