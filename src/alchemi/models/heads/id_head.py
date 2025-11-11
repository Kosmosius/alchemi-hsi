import torch
import torch.nn as nn

from ..retrieval import cosine_topk


class IDHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int | None = None):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, n_classes) if n_classes else None
        self.lab_bank = None
        self.lab_labels = None

    def set_lab_bank(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.lab_bank, self.lab_labels = embeddings, labels

    def forward(self, z: torch.Tensor, topk: int = 5):
        if z.dim() == 1:
            idx, sim = cosine_topk(z, self.lab_bank, k=topk)
            return {"idx": idx, "sim": sim, "labels": self.lab_labels[idx]}
        out = []
        for i in range(z.shape[0]):
            idx, sim = cosine_topk(z[i], self.lab_bank, k=topk)
            out.append({"idx": idx, "sim": sim, "labels": self.lab_labels[idx]})
        return out
