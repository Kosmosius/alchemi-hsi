import torch

from alchemi.registry.acceptance import AcceptanceVerdict
from alchemi.uncertainty.ood import combine_ood_scores


def test_combine_scales_with_acceptance_warning():
    scores = {"s1": torch.tensor([0.4, 0.6])}
    combined, decision = combine_ood_scores(
        scores, threshold=0.5, acceptance=AcceptanceVerdict.ACCEPT_WITH_WARNINGS, warning_scale=2.0
    )

    assert torch.allclose(combined, torch.tensor([0.8, 1.2]))
    assert decision is not None and torch.equal(decision, torch.tensor([True, True]))


def test_combine_forces_ood_when_rejected_sensor():
    scores = {"s1": torch.tensor([0.1, 0.2])}
    combined, decision = combine_ood_scores(
        scores, threshold=0.5, acceptance=AcceptanceVerdict.REJECT, reject_score=5.0
    )

    assert torch.all(combined == torch.tensor([5.0, 5.0]))
    assert decision is not None and torch.equal(decision, torch.tensor([True, True]))

    forced_combined, forced_decision = combine_ood_scores(
        scores, threshold=0.5, acceptance=AcceptanceVerdict.REJECT, force_accept_sensor=True, reject_score=5.0
    )
    assert torch.allclose(forced_combined, torch.tensor([0.1, 0.2]))
    assert forced_decision is not None and torch.equal(forced_decision, torch.tensor([False, False]))
