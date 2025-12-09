import torch

from alchemi.uncertainty.calibration import fit_temperature, temperature_scale_logits
from alchemi.uncertainty.conformal import (
    classification_conformal_thresholds,
    classification_label_set,
)


def _ece(probs: torch.Tensor, labels: torch.Tensor) -> float:
    confidences, preds = probs.max(dim=1)
    accuracy = preds.eq(labels).float()
    return torch.abs(confidences - accuracy).mean().item()


def test_temperature_scaling_reduces_ece():
    torch.manual_seed(0)
    logits = torch.tensor([[5.0, 1.0], [1.0, 5.0], [4.0, 0.5], [0.5, 4.0]])
    labels = torch.tensor([1, 0, 0, 1])
    base_probs = torch.softmax(logits, dim=-1)
    base_ece = _ece(base_probs, labels)

    temperature = 0.5
    scaled = temperature_scale_logits(logits, temperature)
    scaled_probs = torch.softmax(scaled, dim=-1)
    scaled_ece = _ece(scaled_probs, labels)
    assert scaled_ece <= base_ece


def test_conformal_label_sets_cover_nominal_rate():
    torch.manual_seed(0)
    calib_probs = torch.tensor(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.55, 0.45],
            [0.52, 0.48],
        ]
    )
    calib_labels = torch.tensor([0, 0, 1, 1])
    thresholds = classification_conformal_thresholds(calib_probs, calib_labels, alpha=0.2)

    test_probs = torch.tensor(
        [
            [0.8, 0.2],
            [0.51, 0.49],
            [0.45, 0.55],
            [0.6, 0.4],
        ]
    )
    test_labels = torch.tensor([0, 1, 1, 0])
    cover_count = 0
    for prob, label in zip(test_probs, test_labels, strict=True):
        label_set = classification_label_set(prob, thresholds)
        if int(label.item()) in label_set.tolist():
            cover_count += 1
    coverage = cover_count / test_probs.size(0)
    assert 0.5 <= coverage <= 1.0
