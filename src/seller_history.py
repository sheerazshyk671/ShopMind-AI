"""Seller history validation, feature generation and conservative calibration."""
import re
import numpy as np
import pandas as pd
from src.features import FEATURE_COLUMNS, make_features


def history_columns(df):
    columns = sorted(c for c in df.columns if re.fullmatch(r'Sales_\d\d', str(c)))
    if columns and (len(columns) < 6 or columns != [f'Sales_{i:02d}' for i in range(1, len(columns)+1)]):
        raise ValueError('Sales history must have at least six consecutive columns Sales_01, Sales_02, ... with no gaps.')
    return columns


def features_from_history(products, forecast_month):
    cols = history_columns(products)
    if not cols:
        raise ValueError('At least six monthly sales columns are required.')
    values = products[cols].to_numpy(dtype=float)
    recent3 = values[:, -3:].mean(axis=1)
    recent6 = values[:, -6:].mean(axis=1)
    std6 = values[:, -6:].std(axis=1, ddof=1)
    month = pd.Timestamp(forecast_month).month
    safe = np.where(recent6 > 0, recent6, 1)
    result = pd.DataFrame({'month': month, 'month_sin': np.sin(2*np.pi*month/12),
        'month_cos': np.cos(2*np.pi*month/12), 'months_since_start': len(cols),
        'lag_1': values[:, -1], 'lag_3': values[:, -3], 'lag_6': values[:, -6],
        'rolling_mean_3': recent3, 'rolling_mean_6': recent6, 'rolling_std_6': std6,
        'lag_1_ratio': values[:, -1]/safe, 'trend': recent3/safe, 'volatility': std6/safe})
    return result[FEATURE_COLUMNS]


def calibrate_seller(products, model, forecast_month):
    """Shrink a pooled seller calibration towards 1 using strictly past months.

    Each historical prediction is generated using only earlier sales. The pooled
    adjustment is bounded to reduce instability with short, sparse histories.
    """
    cols = history_columns(products)
    if len(cols) < 7:
        return 1.0, 0
    dates = pd.date_range(end=pd.Timestamp(forecast_month) - pd.DateOffset(months=1), periods=len(cols), freq='MS')
    values = products[cols].to_numpy(dtype=float)
    panel = pd.DataFrame({'StockCode': np.repeat(np.arange(len(products)), len(cols)),
        'month': np.tile(dates, len(products)), 'monthly_units_sold': values.ravel()})
    features = make_features(panel)
    if features.empty:
        return 1.0, 0
    predicted = np.maximum(0, model.predict(features[FEATURE_COLUMNS])) * features['rolling_mean_3'].to_numpy()
    observed = features['monthly_units_sold'].to_numpy()
    if predicted.sum() <= 0:
        return 1.0, len(features)
    raw = np.clip(observed.sum() / predicted.sum(), 0.5, 2.0)
    weight = len(features) / (len(features) + 20.0)
    return float(1 + weight * (raw - 1)), len(features)
