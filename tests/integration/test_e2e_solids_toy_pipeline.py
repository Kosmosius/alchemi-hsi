import torch

from alchemi.models.heads.solids import SolidsHead
from alchemi.models.retrieval.lab_index import LabIndex


def test_toy_solids_pipeline_one_step():
    torch.manual_seed(0)
    embeddings = torch.eye(3)
    lab_index = LabIndex.build(embeddings)
    prototypes = torch.stack(
        [torch.linspace(0.1, 0.3, 4), torch.linspace(0.4, 0.6, 4), torch.linspace(0.7, 0.9, 4)]
    )
    head = SolidsHead(
        embed_dim=3, hidden_dim=4, k=2, lab_index=lab_index, prototype_spectra=prototypes
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)

    features = torch.randn(2, 3)
    target_abundances = torch.tensor([[0.7, 0.3, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]])

    out = head(features)
    loss = torch.nn.functional.mse_loss(out.abundances.view(2, -1), target_abundances)
    loss.backward()
    optimizer.step()
    assert out.abundances.shape[0] == 2
