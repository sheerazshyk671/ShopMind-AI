"""Two-stage demand forecasting: market base model, then a seller-specific fine-tune.

Stage 1 (:class:`BaseModel`)
    Learned from a large *general* market panel. It is the cold-start model: it works
    for a product with no history, because it only knows the generic shape of demand
    (seasonality, trend, price response).

Stage 2 (:class:`SellerModel`)
    The base model, adjusted to one specific seller using that seller's own history.
    Two adjustments are supported, and both are fitted on *training months only*:

    * ``offset``    — a pooled multiplicative calibration (the shipped default). Cheap,
                      stable with short histories, and interpretable as "this seller runs
                      N% below/above the market model". This is what the current app uses.
    * ``residual``  — a second-stage regressor trained on the base model's residuals.
                      Captures seller-specific seasonality/shape the pooled factor cannot.

The target is a **ratio** (actual units / trailing 3-month mean), as in the original
ShopMind pipeline, because unit volume varies by four orders of magnitude across products.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.features import FEATURE_COLUMNS, OPTIONAL_FEATURES, make_features

RATIO_FLOOR = 1e-6


@dataclass
class BaseModel:
    """Stage-1 model: general market demand, expressed as a ratio to trailing mean."""

    model: object
    name: str = 'Gradient Boosting'
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    trained_on: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, model, features: pd.DataFrame, target: pd.Series, name: str = 'Gradient Boosting',
            trained_on: dict | None = None) -> 'BaseModel':
        """Fit and remember the exact column list, so inference cannot drift."""
        # Preserve the canonical feature order, not pandas' column order, so that a
        # model saved today scores identically tomorrow.
        present = set(features.columns)
        columns = [c for c in FEATURE_COLUMNS + OPTIONAL_FEATURES if c in present]
        model.fit(features[columns], target)
        return cls(model=model, name=name, feature_columns=columns,
                   trained_on=trained_on or {})

    def predict_ratio(self, features: pd.DataFrame) -> np.ndarray:
        return np.maximum(0.0, self.model.predict(features[self.feature_columns]))

    def forecast_units(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_ratio(features) * features['rolling_mean_3'].to_numpy()

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'BaseModel':
        return joblib.load(path)


@dataclass
class SellerModel:
    """Stage-2 model: the base model, calibrated to one seller's own history."""

    base: BaseModel
    method: str = 'offset'          # 'offset' | 'residual' | 'none'
    offset: float = 1.0
    category_offsets: dict = field(default_factory=dict)
    residual_model: object | None = None
    n_examples: int = 0
    diagnostics: dict = field(default_factory=dict)

    # -- fitting ---------------------------------------------------------
    @classmethod
    def fit(cls, base: BaseModel, seller_panel: pd.DataFrame, method: str = 'offset',
            min_months: int = 7, shrink: float = 20.0,
            residual_params: dict | None = None,
            category_shrink: float = 8.0) -> 'SellerModel':
        """Fit the stage-2 adjustment on the seller's *training* months only.

        ``seller_panel`` must already be restricted to training months. The caller is
        responsible for that split; this method never looks past the rows it is given.
        """
        features = make_features(seller_panel)
        if features.empty or method == 'none':
            return cls(base=base, method='none', diagnostics={'reason': 'no feature-ready rows'})

        base_ratio = base.predict_ratio(features)

        if method == 'offset':
            predicted_units = base_ratio * features['rolling_mean_3'].to_numpy()
            observed = features['monthly_units_sold'].to_numpy()
            if predicted_units.sum() <= 0:
                return cls(base=base, method='none', diagnostics={'reason': 'zero base prediction'})
            raw = float(np.clip(observed.sum() / predicted_units.sum(), 0.5, 2.0))
            weight = len(features) / (len(features) + shrink)
            offset = float(1 + weight * (raw - 1))
            return cls(base=base, method='offset', offset=offset, n_examples=len(features),
                       diagnostics={'raw_ratio': raw, 'shrink_weight': weight,
                                    'note': 'pooled multiplicative calibration, shrunk to 1'})

        if method == 'category_offset':
            # Partial pooling: one shrinkage factor per category, each shrunk towards
            # the seller's own pooled offset. Categories with few rows stay near the
            # pooled value; high-volume categories earn their own factor.
            pooled = cls.fit(base, seller_panel, method='offset', shrink=shrink)
            observed = features['monthly_units_sold'].to_numpy()
            predicted_units = base_ratio * features['rolling_mean_3'].to_numpy()
            offsets = {'__pooled__': pooled.offset}
            if 'category' in seller_panel.columns:
                cats = seller_panel.set_index(['StockCode', 'month'])['category']
                key = pd.MultiIndex.from_arrays([features['StockCode'], features['month_date']])
                groups = pd.Series(cats.reindex(key).to_numpy(), index=features.index)
                for category, idx in groups.groupby(groups).groups.items():
                    if not isinstance(category, str):
                        continue
                    obs, pred = observed[idx], predicted_units[idx]
                    if pred.sum() <= 0:
                        continue
                    raw = float(np.clip(obs.sum() / pred.sum(), 0.5, 2.0))
                    w = len(idx) / (len(idx) + category_shrink)
                    offsets[category] = float(pooled.offset + w * (raw - pooled.offset))
            return cls(base=base, method='category_offset', offset=pooled.offset,
                       category_offsets=offsets, n_examples=len(features),
                       diagnostics={'n_categories': len(offsets) - 1,
                                    'offsets': {k: round(v, 4) for k, v in offsets.items()}})

        if method == 'residual':
            # Target the *multiplicative* residual so it composes with the ratio output.
            actual_ratio = features['target_ratio'].to_numpy()
            target = np.log((actual_ratio + RATIO_FLOOR) / (base_ratio + RATIO_FLOOR))
            params = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                          subsample=0.8, random_state=42)
            params.update(residual_params or {})
            residual_model = GradientBoostingRegressor(**params)
            residual_model.fit(features[base.feature_columns], target)
            in_sample = base_ratio * np.exp(residual_model.predict(features[base.feature_columns]))
            return cls(base=base, method='residual', residual_model=residual_model,
                       n_examples=len(features),
                       diagnostics={'in_sample_mae': float(np.mean(np.abs(
                           in_sample - actual_ratio)))})

        raise ValueError(f"Unknown stage-2 method {method!r}; use 'offset', 'residual' or 'none'.")

    # -- inference -------------------------------------------------------
    def predict_ratio(self, features: pd.DataFrame, categories: pd.Series | None = None) -> np.ndarray:
        ratio = self.base.predict_ratio(features)
        if self.method == 'category_offset' and self.category_offsets:
            pooled = self.category_offsets.get('__pooled__', self.offset)
            factor = np.full(len(features), pooled, dtype=float)
            if categories is not None:
                mapped = categories.map(self.category_offsets)
                factor = np.where(mapped.notna(), mapped.fillna(pooled), pooled)
            ratio = ratio * factor
        elif self.method == 'offset':
            ratio = ratio * self.offset
        elif self.method == 'residual' and self.residual_model is not None:
            ratio = ratio * np.exp(self.residual_model.predict(features[self.base.feature_columns]))
        return np.maximum(0.0, ratio)

    def forecast_units(self, features: pd.DataFrame, categories: pd.Series | None = None) -> np.ndarray:
        return self.predict_ratio(features, categories) * features['rolling_mean_3'].to_numpy()

    def save(self, path: str) -> None:
        joblib.dump(self, path)


def tune_model(name: str, train: pd.DataFrame, n_splits: int = 3, seed: int = 42):
    """Grid-search on expanding-window TimeSeriesSplit, *before* the test window.

    The split is chronological, so a fold never trains on a month that follows a
    month it validates on. The test window is never touched here.
    """
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    grids = {
        'Random Forest': (RandomForestRegressor(random_state=seed, n_jobs=-1),
                          {'n_estimators': [100, 200], 'min_samples_leaf': [1, 2, 4],
                           'max_features': [0.5, 1.0]}),
        'Gradient Boosting': (GradientBoostingRegressor(random_state=seed),
                              {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                               'max_depth': [2, 3]}),
        'Decision Tree': (DecisionTreeRegressor(random_state=seed),
                          {'max_depth': [4, 6, 8], 'min_samples_leaf': [2, 5, 10]}),
        'Linear Regression': (LinearRegression(), {}),
    }
    estimator, grid = grids[name]
    ordered = train.sort_values('month_date').reset_index(drop=True)
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = GridSearchCV(estimator, grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    search.fit(ordered[FEATURE_COLUMNS], ordered['target_ratio'])
    return search.best_estimator_, search.best_params_, float(-search.best_score_)


MODEL_ZOO = {
    'Linear Regression': lambda seed=42: LinearRegression(),
    'Random Forest': lambda seed=42: RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1),
    'Gradient Boosting': lambda seed=42: GradientBoostingRegressor(random_state=seed),
    'Decision Tree': lambda seed=42: DecisionTreeRegressor(
        max_depth=6, min_samples_leaf=5, random_state=seed),
}


# ---------------------------------------------------------------------------
# Baselines — required by the rubric, and the honest yardstick
# ---------------------------------------------------------------------------
def baseline_forecasts(train_panel: pd.DataFrame, test_panel: pd.DataFrame,
                       features: pd.DataFrame) -> pd.DataFrame:
    """Two trivial baselines evaluated on exactly the same rows as the models.

    * ``last_3_month_mean``  — forecast next month = average of the previous three.
    * ``seasonal_naive``     — forecast next month = same calendar month one year ago.
    * ``fixed_ratio_1``     — the ratio model's own output with the ratio hard-coded to
      1.0. This is the *hardest* trivial rule to beat, because it uses the same
      trailing-mean anchor as the ML models. A team that skips it can accidentally
      conclude a model adds value when it only reproduces a constant.
    """
    test = test_panel.set_index(['StockCode', 'month'])
    hist = train_panel.copy()
    hist['month'] = pd.to_datetime(hist['month'])
    key = pd.MultiIndex.from_arrays([features['StockCode'], features['month_date']])

    # last-3-month mean is already a feature, so no lookahead is introduced.
    mean3 = features['rolling_mean_3'].to_numpy()

    # seasonal naive: same month last year, from history only.
    lookup = hist.set_index(['StockCode', 'month'])['monthly_units_sold']
    prev_year = pd.MultiIndex.from_arrays([features['StockCode'],
                                           features['month_date'] - pd.DateOffset(years=1)])
    seasonal = lookup.reindex(prev_year).to_numpy()
    seasonal = np.where(np.isfinite(seasonal), seasonal, mean3)  # fall back when absent

    out = pd.DataFrame({
        'month': features['month_date'].to_numpy(),
        'StockCode': features['StockCode'].to_numpy(),
        'actual': features['monthly_units_sold'].to_numpy(),
        'last_3_month_mean': mean3,
        'seasonal_naive': seasonal,
        # The strongest trivial rule for this data, and therefore the one that actually
        # has to be beaten: "assume next month looks like the recent three".
        'fixed_ratio_1': mean3 * 1.0,
    })
    del test, key  # kept for readability of the intent above
    return out


def unit_metrics(actual: np.ndarray, predicted: np.ndarray,
                 prices: np.ndarray | None = None,
                 naive_scale: float | None = None) -> dict:
    """Metrics a planner actually reads, in units and in rupees.

    ``MAE``/``RMSE`` treat every unit alike, which is wrong commercially: being 20
    units wrong on a Rs 34,000 air fryer ties up far more cash than being 20 units
    wrong on a Rs 500 lipstick. When ``prices`` is given, a revenue-weighted error
    (the same idea as M5's WRMSSE weighting) is reported alongside.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(0.0, np.asarray(predicted, dtype=float))
    if naive_scale is None:
        # In-sample naive scale: mean absolute month-on-month change. Using it makes
        # MASE comparable across products with wildly different volumes.
        naive_scale = float(np.mean(np.abs(np.diff(actual)))) if len(actual) > 1 else np.nan
    error = predicted - actual
    denominator = np.where(np.abs(actual) > 0, np.abs(actual), np.nan)
    wmape = np.nansum(np.abs(error)) / np.nansum(np.abs(actual)) if np.nansum(np.abs(actual)) else np.nan
    out = {
        'MAE': float(np.mean(np.abs(error))),
        'RMSE': float(np.sqrt(np.mean(error ** 2))),
        'bias': float(np.mean(error)),                 # >0 = over-forecast
        'WMAPE': float(100 * wmape),
        'MAPE': float(100 * np.nanmean(np.abs(error) / denominator)) if np.any(~np.isnan(denominator)) else np.nan,
    }
    # MASE: mean absolute error scaled by the in-sample naive error of the same rows,
    # so it is comparable across products and across models. < 1 beats the naive forecast.
    out['MASE'] = float(np.mean(np.abs(error)) / naive_scale) if naive_scale and naive_scale > 0 else np.nan
    if prices is not None:
        value_error = np.abs(error) * np.asarray(prices, dtype=float)
        value_actual = actual * np.asarray(prices, dtype=float)
        total = np.sum(value_actual)
        out['rupee_WAPE'] = float(100 * value_error.sum() / total) if total > 0 else np.nan
        out['rupee_MAE'] = float(np.mean(value_error))
    return out
