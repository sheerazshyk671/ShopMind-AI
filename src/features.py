"""Leakage-free features for predicting a month's demand from prior months."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ['month', 'month_sin', 'month_cos', 'months_since_start', 'lag_1', 'lag_3', 'lag_6', 'rolling_mean_3', 'rolling_mean_6', 'rolling_std_6', 'lag_1_ratio', 'trend', 'volatility']


def make_features(panel):
    df = panel.copy()
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values(['StockCode', 'month']).reset_index(drop=True)
    grouped = df.groupby('StockCode')['monthly_units_sold']
    df['months_since_start'] = df.groupby('StockCode').cumcount()
    month = df['month'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    for lag in (1, 3, 6):
        df[f'lag_{lag}'] = grouped.shift(lag)
    for window in (3, 6):
        df[f'rolling_mean_{window}'] = grouped.transform(lambda s: s.shift(1).rolling(window).mean())
    df['rolling_std_6'] = grouped.transform(lambda s: s.shift(1).rolling(6).std())
    denominator = df['rolling_mean_6'].replace(0, np.nan)
    df['lag_1_ratio'] = df['lag_1'] / denominator
    df['trend'] = df['rolling_mean_3'] / denominator
    df['volatility'] = df['rolling_std_6'] / denominator
    df['target_ratio'] = df['monthly_units_sold'] / df['rolling_mean_3'].replace(0, np.nan)
    df['month_date'] = df['month']
    df['month'] = month
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ['target_ratio']).reset_index(drop=True)


def seller_features(products, forecast_month=None):
    """Construct inference features from a flat six-month history proxy.

    A single RecentMonthlySales value cannot identify lags or volatility; this
    approximation is deliberately explicit and yields zero estimated safety stock.
    """
    date = pd.Timestamp(forecast_month or (pd.Timestamp.today().to_period('M') + 1).to_timestamp())
    sales = products['RecentMonthlySales'].astype(float).to_numpy()
    n = len(products)
    m = date.month
    return pd.DataFrame({'month': np.full(n, m), 'month_sin': np.full(n, np.sin(2*np.pi*m/12)),
        'month_cos': np.full(n, np.cos(2*np.pi*m/12)), 'months_since_start': np.full(n, 12),
        'lag_1': sales, 'lag_3': sales, 'lag_6': sales, 'rolling_mean_3': sales,
        'rolling_mean_6': sales, 'rolling_std_6': np.zeros(n),
        'lag_1_ratio': np.where(sales > 0, 1., 0.), 'trend': np.where(sales > 0, 1., 0.),
        'volatility': np.zeros(n)}, columns=FEATURE_COLUMNS)
