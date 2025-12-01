import torch

from alchemi.models.backbone.mae import MAEBackbone
from alchemi.models.heads.gas_head import GasHead
from alchemi.models.heads.solids import SolidsHead
from alchemi.models.retrieval.lab_index import LabIndex


def test_mae_backbone_masks_and_unmasks_tokens():
    model = MAEBackbone(embed_dim=8, depth=1, num_heads=2, decoder_dim=8, decoder_depth=1, masking_ratio=0.5)
    tokens = torch.randn(2, 6, 8)
    output = model.forward_mae(tokens)
    assert output.encoded.shape[1] <= tokens.shape[1]
    assert output.decoded.shape == torch.Size([2, 6, 8])
    assert output.mask.dtype == torch.bool


def test_solids_head_reconstructs_trivial_mixture():
    lab_embeddings = torch.eye(3)
    lab_ids = torch.tensor([0, 1, 2])
    lab_index = LabIndex.build(lab_embeddings, ids=lab_ids)
    prototypes = torch.stack([torch.ones(5) * v for v in [0.2, 0.5, 0.8]])
    head = SolidsHead(embed_dim=3, hidden_dim=4, k=2, lab_index=lab_index, prototype_spectra=prototypes)
    features = torch.tensor([[1.0, 0.0, 0.0]])
    out = head(features)
    assert out.abundances.shape[-1] == head.k + 2
    assert out.reconstruction.shape[-1] == prototypes.shape[-1]
    assert out.dominant_id.item() in lab_ids.tolist()


def test_gas_head_output_shape():
    head = GasHead(embed_dim=4)
    dummy = torch.randn(2, 3, 4)
    out = head(dummy)
    assert out.shape == torch.Size([2, 3])
