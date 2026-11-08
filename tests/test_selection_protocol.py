"""Regression test for the model-selection protocol.

The rubric requires that the model be chosen on validation and scored on test exactly
once. It is easy to break this accidentally, and it happened once here: the summary
reported the best *test* row and called it the project's model. These tests exist so it
cannot happen silently again.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPORTS = Path('reports')


@pytest.fixture(scope='module')
def summary() -> dict:
    path = REPORTS / 'training_summary.json'
    if not path.exists():  # pragma: no cover - reports are committed
        pytest.skip('training_summary.json not generated yet')
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def stage1() -> pd.DataFrame:
    path = REPORTS / 'stage1_results.csv'
    if not path.exists():  # pragma: no cover
        pytest.skip('stage1_results.csv not generated yet')
    return pd.read_csv(path)


def test_reported_model_is_the_validation_winner(summary: dict) -> None:
    """The reported model must be the one with the lowest VALIDATION MAE."""
    val = pd.read_csv(REPORTS / 'stage1_validation.csv')
    winner = val.sort_values('validation_MAE').iloc[0]['model']
    assert summary['reported_model'] == winner, (
        f"summary reports {summary['reported_model']} but validation chose {winner}; "
        'the reported model must be the validation winner, not the test winner'
    )


def test_reported_mae_matches_that_models_test_row(summary: dict, stage1: pd.DataFrame) -> None:
    """Reported MAE must be that model's own test score, not the minimum across models."""
    row = stage1.set_index('model').loc[summary['reported_model']]
    assert summary['reported_model_test_mae'] == pytest.approx(row['MAE'], abs=1e-3)


def test_reported_mae_is_not_the_minimum_test_mae(summary: dict, stage1: pd.DataFrame) -> None:
    """Guard against the exact bug that was fixed.

    If the reported MAE happens to equal the best test MAE while a different model won
    validation, the summary is selecting on test again.
    """
    models = stage1[stage1['family'] != 'Baseline']
    min_mae = float(models['MAE'].min())
    if summary['reported_model_test_mae'] == pytest.approx(min_mae, abs=1e-3):
        # Legitimate only when the validation winner really is also the test winner.
        assert summary['test_optimal_model_if_we_had_peeked'] == summary['reported_model']
    else:
        assert summary['reported_model_test_mae'] > min_mae


def test_selection_variance_is_recorded(summary: dict) -> None:
    """The cost of the honest protocol must be quantified, not hidden."""
    assert 'selection_variance_pct' in summary
    assert 'test_optimal_model_if_we_had_peeked' in summary
    assert 'test_optimal_model_mae' in summary
    assert summary['selection_variance_pct'] >= 0


def test_improvement_is_computed_against_the_best_baseline(summary: dict, stage1: pd.DataFrame) -> None:
    best_baseline = float(stage1.loc[stage1['family'] == 'Baseline', 'MAE'].min())
    assert summary['best_baseline_test_mae'] == pytest.approx(best_baseline, abs=1e-3)
    expected = 100 * (best_baseline - summary['reported_model_test_mae']) / best_baseline
    assert summary['improvement_vs_baseline_pct'] == pytest.approx(expected, abs=0.01)


def test_negative_result_flag_is_consistent(summary: dict) -> None:
    """`model_beats_baseline` must agree with the reported numbers."""
    beats = summary['reported_model_test_mae'] < summary['best_baseline_test_mae']
    assert summary['model_beats_baseline'] is beats
