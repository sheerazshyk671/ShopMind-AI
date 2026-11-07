"""Value-weighted greedy budget allocation in whole units."""
import numpy as np
import pandas as pd


def optimize_budget(df_forecast, budget, alpha=0.5):
    if not np.isfinite(budget) or budget < 0 or not np.isfinite(alpha) or alpha < 0:
        raise ValueError('Budget and alpha must be finite and non-negative.')
    df = df_forecast.copy().reset_index(drop=True)
    required = ['ForecastedDemand', 'CurrentStock', 'CurrentPrice']
    if not set(required).issubset(df.columns):
        raise ValueError('ForecastedDemand, CurrentStock and CurrentPrice are required.')
    if 'SafetyStock' not in df:
        df['SafetyStock'] = 0.0
    for col in required + ['SafetyStock']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col]).all() or (df[col] < 0).any() or (col == 'CurrentPrice' and (df[col] <= 0).any()):
            raise ValueError(f'{col} must contain valid non-negative numbers (price must be positive).')
    df['RequiredInventory'] = df['ForecastedDemand'] + df['SafetyStock']
    df['ShortageUnits'] = np.ceil(np.maximum(0, df['RequiredInventory'] - df['CurrentStock'])).astype(int)
    # Specification uses shortage_units * (CurrentPrice * ALPHA).
    df['PriorityScore'] = df['ShortageUnits'] * df['CurrentPrice'] * alpha
    df = df.sort_values('PriorityScore', ascending=False, kind='stable').reset_index(drop=True)
    positive = df['PriorityScore'] > 0
    ranks = df.loc[positive, 'PriorityScore'].rank(pct=True, method='max')
    df['Priority'] = 'Low'
    df.loc[ranks.index, 'Priority'] = np.select([ranks >= 2/3, ranks >= 1/3], ['High', 'Medium'], default='Low')
    # Decimal arithmetic avoids floating point overspending, including fractional prices.
    from decimal import Decimal, ROUND_FLOOR
    remaining = Decimal(str(budget))
    purchases, investments = [], []
    for _, row in df.iterrows():
        price = Decimal(str(row['CurrentPrice']))
        units = min(int(row['ShortageUnits']), int((remaining / price).to_integral_value(rounding=ROUND_FLOOR)))
        investment = price * units
        remaining -= investment
        purchases.append(units)
        investments.append(float(investment))
    df['RecommendedPurchase'] = purchases
    df['Investment'] = investments
    return df
