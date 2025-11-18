from __future__ import annotations

from torch import Tensor, nn

from ..retrieval import cosine_topk


class IDHead(nn.Module):
    classifier: nn.Linear | None
    lab_bank: Tensor | None
    lab_labels: Tensor | None

    def __init__(self, embed_dim: int, n_classes: int | None = None) -> None:
        super().__init__()
        self.classifier = nn.Linear(embed_dim, n_classes) if n_classes else None
        self.lab_bank = None
        self.lab_labels = None

    def set_lab_bank(self, embeddings: Tensor, labels: Tensor) -> None:
        self.lab_bank = embeddings
        self.lab_labels = labels

    def _require_bank(self) -> tuple[Tensor, Tensor]:
        if self.lab_bank is None or self.lab_labels is None:
            raise RuntimeError("Lab bank has not been initialized")
        return self.lab_bank, self.lab_labels

    def forward(self, z: Tensor, topk: int = 5) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        lab_bank, lab_labels = self._require_bank()
        if z.dim() == 1:
            idx, sim = cosine_topk(z, lab_bank, k=topk)
            return {"idx": idx, "sim": sim, "labels": lab_labels[idx]}
        out: list[dict[str, Tensor]] = []
        for i in range(z.shape[0]):
            idx, sim = cosine_topk(z[i], lab_bank, k=topk)
            out.append({"idx": idx, "sim": sim, "labels": lab_labels[idx]})
        return out
