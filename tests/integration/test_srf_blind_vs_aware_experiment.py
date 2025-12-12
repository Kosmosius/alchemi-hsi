from pathlib import Path

from alchemi.evaluation.srf_blind_vs_aware import run_srf_blind_vs_aware_experiment


def test_srf_blind_vs_aware_toy_experiment_runs_and_reports_drop():
    cfg_path = Path("configs/eval/srf_blind_vs_aware.yaml")
    report = run_srf_blind_vs_aware_experiment(cfg_path)

    assert set(report) == {"aware_accuracy", "blind_accuracy", "relative_drop"}
    assert report["aware_accuracy"] > 0.5
    assert report["blind_accuracy"] > 0.25
    assert 0.0 <= report["relative_drop"] < 0.8
