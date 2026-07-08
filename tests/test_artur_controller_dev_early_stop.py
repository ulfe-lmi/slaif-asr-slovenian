from __future__ import annotations

from slaif_asr.artur_controller_dev import select_earliest_within_tolerance, watcher_contract_valid


def test_early_stop_chooses_earliest_checkpoint_within_tolerance() -> None:
    rows = [
        {"round": 1, "available": True, "wer": 20.4, "cer": 8.1, "empty": 0},
        {"round": 2, "available": True, "wer": 20.0, "cer": 8.0, "empty": 0},
        {"round": 3, "available": True, "wer": 19.9, "cer": 7.9, "empty": 0},
    ]
    selected = select_earliest_within_tolerance(rows, base_empty_count=0)
    assert selected is not None
    assert selected["round"] == 1


def test_early_stop_rejects_empty_hypothesis_regression() -> None:
    rows = [
        {"round": 1, "available": True, "wer": 20.4, "cer": 8.1, "empty": 1},
        {"round": 2, "available": True, "wer": 20.0, "cer": 8.0, "empty": 0},
    ]
    selected = select_earliest_within_tolerance(rows, base_empty_count=0)
    assert selected is not None
    assert selected["round"] == 2


def test_watcher_refuses_same_training_and_evaluation_gpu(tmp_path) -> None:
    checkpoint_dir = tmp_path / "runs" / "checkpoints"
    metrics_dir = tmp_path / "runs" / "metrics"
    try:
        watcher_contract_valid("0", "0", checkpoint_dir, metrics_dir)
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("same-GPU watcher contract was accepted")
