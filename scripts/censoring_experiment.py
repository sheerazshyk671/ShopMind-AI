"""Does forecasting *sales* instead of *demand* bias the model downwards?

In this dataset a stock-out truncates recorded sales: the customer wanted the item,
it was not on the shelf, and the sale simply never happened. A model trained on sales
therefore learns a demand level that is systematically too low for fast movers, which
in turn causes the next stock-out. That feedback loop is a real and well-known
inventory pathology.

The experiment is deliberately simple and falsifiable:

    panel A  monthly_units_sold = units_sold              (what sellers normally record)
    panel B  monthly_units_sold = units_sold + lost_units (true demand, only visible here)

Both panels go through the *identical* pipeline (same features, same split, same model,
same seed). Both models are then scored against true demand, which is the only fair
comparison - otherwise panel A would be rewarded for under-forecasting.

Usage:
    python scripts/censoring_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_COLUMNS, make_features
from src.modeling import MODEL_ZOO, BaseModel, unit_metrics

SEED = 42
HOLDOUT = 4


def build_panel(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={'product_id': 'StockCode', 'units_sold': 'units_sold'})
    df['month'] = pd.to_datetime(df['month'])
    if target == 'demand':
        df['monthly_units_sold'] = df['units_sold'] + df['lost_units']
    else:
        df['monthly_units_sold'] = df['units_sold']
    return df[['StockCode', 'month', 'monthly_units_sold', 'category', 'avg_unit_price']].copy()


def run(panel: pd.DataFrame, label: str) -> dict:
    data = make_features(panel)
    months = sorted(data['month_date'].unique())
    cutoff = months[-HOLDOUT]
    train = data[data['month_date'] < cutoff]
    test = data[data['month_date'] >= cutoff]

    model = MODEL_ZOO['Random Forest'](SEED)
    model.fit(train[FEATURE_COLUMNS], train['target_ratio'])
    base = BaseModel(model=model, name='Random Forest')
    return {'trained_on': label, 'test': test, 'base': base}


def main() -> None:
    demo = 'data/demo/seller_sales_history.csv'
    sales = run(build_panel(demo, 'sales'), 'recorded sales (censored)')
    demand = run(build_panel(demo, 'demand'), 'true demand (uncensored)')

    # Ground truth for both: true demand.
    truth_panel = build_panel(demo, 'demand')
    data = make_features(truth_panel)
    months = sorted(data['month_date'].unique())
    test_truth = data[data['month_date'] >= months[-HOLDOUT]]

    rows = []
    for result in (sales, demand):
        predicted = result['base'].forecast_units(test_truth)
        metrics = unit_metrics(test_truth['monthly_units_sold'].values, predicted,
                               test_truth['avg_unit_price'].values)
        rows.append({'model': result['trained_on'], **metrics})

    table = pd.DataFrame(rows)
    Path('reports').mkdir(exist_ok=True)
    table.to_csv('reports/censoring_experiment.csv', index=False)

    print('Both models scored against TRUE DEMAND on the same holdout months:\n')
    print(table.round(3).to_string(index=False))

    sales_mae = float(table.loc[0, 'MAE'])
    demand_mae = float(table.loc[1, 'MAE'])
    sales_bias = float(table.loc[0, 'bias'])
    demand_bias = float(table.loc[1, 'bias'])

    print('\n--- what the numbers actually say ---')
    print(f'Training on recorded sales costs {100 * (sales_mae - demand_mae) / demand_mae:.0f}% more MAE '
          f'when the thing you care about is demand ({sales_mae:.2f} vs {demand_mae:.2f}).')
    print(f'Bias (forecast - demand): sales-trained {sales_bias:+.2f}, demand-trained {demand_bias:+.2f}.')

    if sales_bias > 0:
        print('\nThe sales-trained model OVER-forecasts. This is NOT the textbook censoring')
        print('effect (which would bias it low); the mechanism here is different and worth')
        print('stating precisely:')
        print('  * the ratio target drifts upward across the fit window, so the model learns')
        print('    a ratio above 1 and keeps applying it;')
        print('  * the trailing mean it multiplies is already inflated by the holiday spike;')
        print('  * multiplying an inflated anchor by a ratio above 1 compounds the error.')
        print('So the finding is "training on sales makes the ratio target unstable and the')
        print('forecast biased high at a seasonal turning point", not "censoring hides demand".')
    else:
        print('\nThe sales-trained model UNDER-forecasts, which is the classic censoring')
        print('effect: it was taught that sold-out months had low demand.')

    print('\nBusiness reading: forecasting *demand* rather than *sales* is the right target')
    print('for a buying decision, because a stock-out destroys the sale rather than the want.')
    print('A seller who plans to sales never breaks the cycle.')


if __name__ == '__main__':
    main()
