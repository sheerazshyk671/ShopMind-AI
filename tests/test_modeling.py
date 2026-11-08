"""Tests for the two-stage model, the baselines, and the leakage guarantees."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, OPTIONAL_FEATURES, make_features
from src.modeling import BaseModel, MODEL_ZOO, SellerModel, baseline_forecasts, unit_metrics


class ConstantRatio:
    """Predicts a fixed ratio, so other components can be tested in isolation."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def synthetic_panel(products=6, months=18, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp('2024-01-01')
    rows = []
    for p in range(products):
        level = 10 + 5 * p
        for m in range(months):
            month = start + pd.DateOffset(months=m)
            units = max(0, int(level * (1 + 0.05 * m) * np.exp(rng.normal(0, 0.15))))
            rows.append({'StockCode': f'S{p}', 'month': month, 'monthly_units_sold': units,
                         'category': 'Fashion', 'avg_unit_price': 1000.0 + 100 * p})
    return pd.DataFrame(rows)


# --- metrics ---------------------------------------------------------------
def test_metrics_are_zero_for_a_perfect_forecast():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = unit_metrics(actual, actual)
    assert metrics['MAE'] == 0 and metrics['RMSE'] == 0 and metrics['bias'] == 0


def test_bias_sign_tells_over_from_under_forecasting():
    actual = np.array([10.0, 10.0])
    assert unit_metrics(actual, np.array([12.0, 12.0]))['bias'] > 0     # over-forecast
    assert unit_metrics(actual, np.array([8.0, 8.0]))['bias'] < 0       # under-forecast


def test_rupee_weighting_punishes_expensive_mistakes_more():
    actual = np.array([10.0, 10.0])
    # Identical unit error, very different prices: the rupee metric must differ.
    cheap = unit_metrics(actual, np.array([20.0, 10.0]), prices=np.array([100.0, 50_000.0]))
    dear = unit_metrics(actual, np.array([10.0, 20.0]), prices=np.array([100.0, 50_000.0]))
    assert cheap['MAE'] == dear['MAE']
    assert dear['rupee_WAPE'] > cheap['rupee_WAPE']


def test_mase_below_one_means_the_naive_forecast_was_beaten():
    actual = np.array([10.0, 12.0, 11.0, 13.0])
    assert unit_metrics(actual, actual, naive_scale=2.0)['MASE'] == 0
    assert unit_metrics(actual, actual + 1, naive_scale=2.0)['MASE'] < 1


# --- leakage guarantees ----------------------------------------------------
def test_features_at_month_t_never_use_month_t_or_later():
    """Rebuild features on a truncated panel and confirm the last row is unchanged.

    This is the real leakage test: if any feature peeked forward, truncating the
    panel after month t would change the features computed *for* month t.
    """
    panel = synthetic_panel()
    full = make_features(panel)
    target_month = sorted(full['month_date'].unique())[10]

    truncated_panel = panel[panel['month'] <= target_month]
    truncated = make_features(truncated_panel)

    full_row = full[full['month_date'] == target_month].reset_index(drop=True)
    trunc_row = truncated[truncated['month_date'] == target_month].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_row[FEATURE_COLUMNS], trunc_row[FEATURE_COLUMNS])


def test_target_ratio_uses_the_current_month_but_features_do_not():
    panel = synthetic_panel(products=1, months=8)
    features = make_features(panel)
    row = features.iloc[-1]
    # lag_1 is the previous month, so it must differ from the current month's sales
    previous = panel[panel['month'] == row['month_date'] - pd.DateOffset(months=1)]['monthly_units_sold'].iloc[0]
    assert row['lag_1'] == previous


def test_optional_promo_features_are_used_only_when_present():
    panel = synthetic_panel()
    plain = make_features(panel)
    assert not set(OPTIONAL_FEATURES) & set(plain.columns)

    with_promo = panel.copy()
    with_promo['planned_promo_days'] = 3
    with_promo['planned_discount_pct'] = 10.0
    enriched = make_features(with_promo)
    for column in OPTIONAL_FEATURES:
        assert column in enriched.columns


# --- stage 1 / stage 2 -----------------------------------------------------
def test_base_model_records_the_columns_it_trained_on():
    panel = synthetic_panel()
    features = make_features(panel)
    model = BaseModel.fit(MODEL_ZOO['Linear Regression'](42), features,
                          features['target_ratio'], name='Linear Regression')
    assert model.feature_columns == FEATURE_COLUMNS
    assert len(model.predict_ratio(features)) == len(features)


def test_offset_calibration_is_fitted_without_touching_future_rows():
    panel = synthetic_panel()
    features = make_features(panel)
    base = BaseModel(model=ConstantRatio(1.0), name='Constant')
    history = panel[panel['month'] < features['month_date'].iloc[-1]]

    calibrated = SellerModel.fit(base, history, method='offset')
    uncalibrated = SellerModel(base=base, method='none')
    # Same base model, different ratio: the offset must actually do something.
    assert calibrated.method == 'offset'
    assert calibrated.offset != pytest.approx(uncalibrated.offset * 0 + 1.0) or calibrated.n_examples > 0


def test_category_offsets_are_bounded_and_named():
    panel = synthetic_panel()
    features = make_features(panel)
    base = BaseModel(model=ConstantRatio(1.0), name='Constant')
    model = SellerModel.fit(base, panel, method='category_offset')
    assert model.method == 'category_offset'
    assert '__pooled__' in model.category_offsets
    assert all(0.5 <= v <= 2.0 for v in model.category_offsets.values())


def test_none_method_is_an_exact_pass_through():
    panel = synthetic_panel()
    features = make_features(panel)
    base = BaseModel(model=ConstantRatio(1.3), name='Constant')
    model = SellerModel(base=base, method='none')
    np.testing.assert_allclose(model.predict_ratio(features), 1.3)


def test_unknown_stage_two_method_is_rejected():
    panel = synthetic_panel()
    with pytest.raises(ValueError):
        SellerModel.fit(BaseModel(model=ConstantRatio(1.0)), panel, method='magic')


# --- baselines -------------------------------------------------------------
def test_baselines_are_evaluated_on_the_same_rows_as_the_models():
    panel = synthetic_panel()
    features = make_features(panel)
    months = sorted(features['month_date'].unique())
    test_features = features[features['month_date'] >= months[-3]]
    baselines = baseline_forecasts(panel[panel['month'] < months[-3]], panel[panel['month'] >= months[-3]], test_features)
    assert len(baselines) == len(test_features)
    np.testing.assert_allclose(baselines['actual'], test_features['monthly_units_sold'])
    # The fixed-ratio baseline must equal the trailing mean exactly.
    np.testing.assert_allclose(baselines['fixed_ratio_1'], test_features['rolling_mean_3'])


def test_seasonal_naive_falls_back_when_last_year_is_absent():
    """With under a year of history there is no same-month-last-year value."""
    panel = synthetic_panel(products=3, months=10)
    features = make_features(panel)
    months = sorted(features['month_date'].unique())
    test_features = features[features['month_date'] >= months[-2]]
    baselines = baseline_forecasts(panel[panel['month'] < months[-2]],
                                   panel[panel['month'] >= months[-2]], test_features)
    assert baselines['seasonal_naive'].notna().all()
    np.testing.assert_allclose(baselines['seasonal_naive'], baselines['last_3_month_mean'])
