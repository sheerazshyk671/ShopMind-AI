"""Benchmark every model against the baselines on the REAL Pakistani dataset.

This is the honest test the project needs: does an ML model beat "assume next month
looks like the recent three" on real marketplace data? Run it and report whatever
comes out, including a negative result.

    python scripts/pakistan_benchmark.py
    python scripts/pakistan_benchmark.py --panel data/processed/pakistan_sku_month.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_COLUMNS, OPTIONAL_FEATURES, make_features
from src.modeling import MODEL_ZOO, BaseModel, baseline_forecasts, unit_metrics

SEED = 42


def load_pakistan_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    key = 'category' if 'category' in df.columns else 'sku'
    df['month'] = pd.to_datetime(df['month_date'] if 'month_date' in df.columns else df['month'])
    out = df.rename(columns={key: 'StockCode', 'units_sold': 'monthly_units_sold'})
    out['StockCode'] = out['StockCode'].astype(str)
    return out[['StockCode', 'month', 'monthly_units_sold', 'avg_price']].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--panel', default='data/processed/pakistan_category_month.csv')
    parser.add_argument('--holdout', type=int, default=4)
    parser.add_argument('--out', default='reports/pakistan_benchmark.csv')
    args = parser.parse_args()

    if not Path(args.panel).exists():
        raise SystemExit(f'{args.panel} not found. Run `python -m src.pakistan_data` first.')

    panel = load_pakistan_panel(args.panel)
    data = make_features(panel)
    months = sorted(data['month_date'].unique())
    cutoff = months[-args.holdout]
    train = data[data['month_date'] < cutoff]
    test = data[data['month_date'] >= cutoff]
    columns = [c for c in FEATURE_COLUMNS + OPTIONAL_FEATURES if c in data.columns]

    print(f'Panel {args.panel}')
    print(f'  {len(data):,} feature-ready rows | {data["StockCode"].nunique()} series | '
          f'train {len(train)} | test {len(test)}')
    print(f'  train {train["month_date"].min().date()}..{train["month_date"].max().date()}  '
          f'| test {test["month_date"].min().date()}..{test["month_date"].max().date()}')

    scale = float(panel[panel['month'] < cutoff].groupby('month')['monthly_units_sold']
                  .sum().diff().abs().mean())
    rows = []
    for name, factory in MODEL_ZOO.items():
        model = BaseModel.fit(factory(SEED), train, train['target_ratio'], name=name)
        predicted = model.forecast_units(test)
        metrics = unit_metrics(test['monthly_units_sold'].values, predicted, None, scale)
        metrics.update({'model': name, 'kind': 'model'})
        rows.append(metrics)

    baselines = baseline_forecasts(panel[panel['month'] < cutoff],
                                   panel[panel['month'] >= cutoff], test)
    for column, label in (('last_3_month_mean', 'Baseline: last-3-month mean'),
                          ('seasonal_naive', 'Baseline: seasonal naive')):
        metrics = unit_metrics(baselines['actual'].values, baselines[column].values, None, scale)
        metrics.update({'model': label, 'kind': 'baseline'})
        rows.append(metrics)

    table = pd.DataFrame(rows)[['model', 'kind', 'MAE', 'RMSE', 'bias', 'WMAPE', 'MAPE', 'MASE']]
    table = table.sort_values('MAE').reset_index(drop=True)

    print('\n=== Scored on the held-out months (real data) ===')
    print(table.round(2).to_string(index=False))

    baseline_mae = table.loc[table['kind'] == 'baseline', 'MAE'].min()
    model_mae = table.loc[table['kind'] == 'model', 'MAE'].min()
    beats = model_mae < baseline_mae
    print(f'\nBest model MAE {model_mae:.1f} vs best baseline MAE {baseline_mae:.1f} '
          f'-> model {"BEATS" if beats else "LOSES TO"} the baseline '
          f'({100 * (baseline_mae - model_mae) / baseline_mae:+.1f}%)')
    if not beats:
        print('\nReport this as a negative result. It is caused by event-driven months '
              '(Black Friday, Feb-Mar 2018) that a level-and-trend model cannot anticipate; '
              'see docs/PAKISTAN_DATASET.md section 5.')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    summary = {
        'panel': args.panel,
        'rows_feature_ready': int(len(data)),
        'series': int(data['StockCode'].nunique()),
        'train_rows': int(len(train)),
        'test_rows': int(len(test)),
        'holdout_months': args.holdout,
        'cutoff': str(pd.Timestamp(cutoff).date()),
        'best_model': str(table[table.kind == 'model'].iloc[0]['model']),
        'best_model_mae': round(float(model_mae), 2),
        'best_baseline_mae': round(float(baseline_mae), 2),
        'model_beats_baseline': bool(beats),
        'improvement_pct': round(float(100 * (baseline_mae - model_mae) / baseline_mae), 2),
    }
    Path('reports/pakistan_benchmark.json').write_text(json.dumps(summary, indent=2))
    print(f'\nWrote {args.out} and reports/pakistan_benchmark.json')


if __name__ == '__main__':
    main()
