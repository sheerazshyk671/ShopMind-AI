"""Turn Pakistan's Largest E-Commerce Dataset into a forecasting panel.

The raw export is 584,524 real order lines from a Pakistani marketplace
(July 2016 – August 2018, 16 categories, ~85k SKUs). This module cleans it and
produces the two panels the project needs:

* a **category x month** panel — the honest, usable unit of analysis (see
  ``docs/PAKISTAN_DATASET.md`` for why SKU-level is not viable on this data), and
* a **SKU x month** panel for the small subset of products that persist long
  enough to forecast (opt-in; it covers only ~22% of units).

Run:
    python -m src.pakistan_data --input data/raw/pakistan_ecommerce_raw.csv \
        --output data/processed --audit reports/pakistan_data_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Statuses that represent a real, kept sale. Cancellations, refunds, fraud and the
# literal string "\N" are excluded: their goods came back, so counting them as demand
# would inflate the target.
SALE_STATUSES = ('complete', 'received', 'cod', 'paid', 'closed')

# A cancelled or refunded line is the *return leg* of a sale: the goods came back.
# Those quantities are measurable and become the return-rate denominator/numerator.
RETURN_STATUSES = ('canceled', 'order_refunded', 'refund', 'fraud', 'exchange')

# Orders that never completed but were also never returned: payment issues, manual
# holds, abandoned checkouts. They are neither sales nor returns, so they are excluded
# from both sides. They also carry no usable date alignment, hence the drop.
UNRESOLVED_STATUSES = ('pending', 'pending_paypal', 'processing', 'holded',
                       'payment_review', '\\N')

# 2016-07 is a partial first month and 2018-08 stops on the 28th, so both are trimmed
# from the modelling window. They stay in the audit.
DROP_MONTHS = ('2016-07', '2018-08')


def load_raw(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read the export and drop the blank rows the Excel export appends."""
    raw = pd.read_csv(path, low_memory=False)
    audit = {
        'raw_rows': int(len(raw)),
        'truncated_at_excel_row_limit': bool(len(raw) >= 1_048_575),
        'excel_row_limit': 1_048_575,
        'note': ('The shared CSV export hits the 1,048,575-row Excel limit, so this is '
                 '584,524 non-empty order lines out of the ~1M the Kaggle page advertises. '
                 'State this in the report; do not claim "half a million" as the whole file.'),
    }
    unnamed = [c for c in raw.columns if str(c).startswith('Unnamed:')]
    df = raw.drop(columns=unnamed, errors='ignore').dropna(how='all').reset_index(drop=True)
    audit['rows_after_dropping_blank'] = int(len(df))
    audit['columns_dropped'] = unnamed
    return df, audit


def clean(df: pd.DataFrame, audit: dict) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['created_at'], errors='coerce', format='mixed')
    audit['unparseable_dates'] = int(df['date'].isna().sum())
    df = df.dropna(subset=['date'])

    status_counts = df['status'].value_counts().to_dict()
    audit['status_counts'] = {str(k): int(v) for k, v in status_counts.items()}

    before_qty_filter = len(df)
    df = df[(df['qty_ordered'] > 0) & (df['grand_total'] >= 0)].copy()
    audit['rows_dropped_bad_quantity_or_value'] = int(before_qty_filter - len(df))

    df['is_sale'] = df['status'].isin(SALE_STATUSES)
    df['is_return'] = df['status'].isin(RETURN_STATUSES)
    df['is_unresolved'] = df['status'].isin(UNRESOLVED_STATUSES)
    audit['rows_kept_as_sales'] = int(df['is_sale'].sum())
    audit['rows_returns'] = int(df['is_return'].sum())
    audit['rows_unresolved_excluded'] = int(df['is_unresolved'].sum())
    audit['rows_excluded'] = int(len(df) - df['is_sale'].sum())
    audit['revenue_kept_pct'] = round(
        100 * df.loc[df['is_sale'], 'grand_total'].sum() / df['grand_total'].sum(), 2
    ) if df['grand_total'].sum() else 0.0
    # Prices of zero remove the revenue signal; keep the row but flag it.
    audit['rows_with_zero_price'] = int((df.loc[df['is_sale'], 'price'] <= 0).sum())

    kept = df
    kept['month'] = kept['date'].dt.to_period('M')
    kept['month_date'] = kept['month'].dt.to_timestamp()
    kept['month_key'] = kept['month'].astype(str)
    audit['date_range'] = [str(kept['date'].min().date()), str(kept['date'].max().date())]
    audit['months'] = int(kept['month_key'].nunique())
    audit['distinct_orders'] = int(kept['increment_id'].nunique())
    audit['distinct_skus'] = int(kept['sku'].nunique())
    audit['categories'] = int(kept['category_name_1'].nunique())
    audit['revenue_total_pkr'] = round(float(kept['grand_total'].sum()), 2)
    audit['units_total'] = float(kept['qty_ordered'].sum())
    audit['average_order_value_pkr'] = round(
        float(kept['grand_total'].sum() / kept['increment_id'].nunique()), 2)
    return kept


def category_panel(sales: pd.DataFrame) -> pd.DataFrame:
    """Category x month: units, revenue and orders. The usable forecasting grain."""
    panel = (sales[sales['is_sale']].groupby(['category_name_1', 'month_date'])
             .agg(units_sold=('qty_ordered', 'sum'),
                  revenue=('grand_total', 'sum'),
                  orders=('increment_id', 'nunique'),
                  avg_price=('price', 'mean'),
                  returned_units=('qty_ordered', 'sum'))
             .reset_index()
             .rename(columns={'category_name_1': 'category'}))
    # A cancelled/refunded line is the return leg of the same month. clean() keeps
    # those rows flagged so returns can be measured instead of silently disappearing.
    refunds = (sales[sales['is_return']]
               .groupby(['category_name_1', 'month_date'])['qty_ordered'].sum()
               .rename('refunded_units').reset_index()
               .rename(columns={'category_name_1': 'category'}))
    panel = panel.drop(columns=['returned_units']).merge(refunds, on=['category', 'month_date'], how='left')
    panel['refunded_units'] = panel['refunded_units'].fillna(0.0)
    # Return rate = returned units / (sold + returned). Published Pakistani COD bands are
    # 25-35%; anything far above that is a data artefact and gets flagged in the audit.
    panel['return_rate'] = (panel['refunded_units'] /
                            (panel['units_sold'] + panel['refunded_units'])).replace([np.inf], np.nan)
    panel = panel[panel['month_date'].dt.strftime('%Y-%m').isin(
        [m for m in panel['month_date'].dt.strftime('%Y-%m').unique() if m not in DROP_MONTHS])]
    return panel.sort_values(['category', 'month_date']).reset_index(drop=True)


def sku_panel(sales: pd.DataFrame, min_months: int = 6) -> tuple[pd.DataFrame, dict]:
    """SKU x month, restricted to products that actually persist.

    This dataset is mostly single-purchase listings: the median SKU appears in one
    month. Only SKUs present in ``min_months`` or more are usable, and the returned
    stats show how much of the business they represent.
    """
    monthly = (sales.groupby(['sku', 'month_date'])
               .agg(units_sold=('qty_ordered', 'sum'),
                    revenue=('grand_total', 'sum'),
                    avg_price=('price', 'mean'),
                    category=('category_name_1', 'first'))
               .reset_index())
    months_present = monthly.groupby('sku')['month_date'].nunique()
    keep = months_present[months_present >= min_months].index

    stats = {
        'median_months_per_sku': float(months_present.median()),
        'mean_months_per_sku': round(float(months_present.mean()), 2),
        'skus_total': int(months_present.size),
        'skus_dropped_single_month': int((months_present == 1).sum()),
        f'skus_with_{min_months}+_months': int(len(keep)),
        'unit_coverage_pct': round(100 * monthly[monthly['sku'].isin(keep)]['units_sold'].sum()
                                   / monthly['units_sold'].sum(), 2),
        'revenue_coverage_pct': round(100 * monthly[monthly['sku'].isin(keep)]['revenue'].sum()
                                      / monthly['revenue'].sum(), 2),
        'warning': ('SKU-level forecasting on this dataset covers only a minority of volume. '
                    'Report it as a secondary analysis, not the headline claim.'),
    }
    return monthly[monthly['sku'].isin(keep)].sort_values(['sku', 'month_date']).reset_index(drop=True), stats


def seasonality_report(sales: pd.DataFrame) -> dict:
    """Measure whether the Pakistani retail calendar is visible in this real data."""
    monthly = sales.groupby('month_date').agg(revenue=('grand_total', 'sum'),
                                              orders=('increment_id', 'nunique'),
                                              units=('qty_ordered', 'sum'))
    ramadan = {'2016-06', '2016-07', '2017-05', '2017-06', '2018-05', '2018-06'}
    keys = monthly.index.strftime('%Y-%m')
    monthly['is_ramadan'] = [k in ramadan for k in keys]
    monthly['is_november'] = [k.endswith('-11') for k in keys]
    valid = monthly[~monthly.index.strftime('%Y-%m').isin(DROP_MONTHS)]

    def ratio(mask):
        if not mask.any() or (~mask).sum() == 0:
            return None
        return round(float(valid.loc[mask, 'revenue'].mean()
                           / valid.loc[~mask, 'revenue'].mean()), 3)

    top = valid['revenue'].nlargest(3)
    return {
        'ramadan_revenue_ratio': ratio(valid['is_ramadan']),
        'november_revenue_ratio': ratio(valid['is_november']),
        'top_3_revenue_months': {str(d.date()): round(float(v), 0) for d, v in top.items()},
        'monthly_revenue_cv_pct': round(float(100 * valid['revenue'].std() / valid['revenue'].mean()), 1),
        'reading': ('A November ratio well above 1 is the Black Friday effect. Ramadan does not '
                    'show up cleanly in this export, which is itself a finding: do not assume the '
                    'festival effect is large in this marketplace without measuring it.'),
    }


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(input_path: str, output_dir: str = 'data/processed',
          audit_path: str = 'reports/pakistan_data_audit.json') -> dict[str, Path]:
    raw, audit = load_raw(input_path)
    sales = clean(raw, audit)
    categories = category_panel(sales)
    skus, sku_stats = sku_panel(sales)
    audit['sku_level'] = sku_stats
    audit['seasonality'] = seasonality_report(sales)
    audit['source_file_sha256'] = hash_file(input_path)
    audit['licence'] = ('Data files (c) Original Authors - Zeeshan-ul-hassan Usmani, '
                        'published on Kaggle (zusmani/pakistans-largest-ecommerce-dataset). '
                        'Educational use; cite the dataset.')

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {'categories': out / 'pakistan_category_month.csv',
             'skus': out / 'pakistan_sku_month.csv'}
    categories.to_csv(paths['categories'], index=False)
    skus.to_csv(paths['skus'], index=False)

    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    Path(audit_path).write_text(json.dumps(audit, indent=2))
    audit['_paths'] = {k: str(v) for k, v in paths.items()}
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default='data/raw/pakistan_ecommerce_raw.csv')
    parser.add_argument('--output', default='data/processed')
    parser.add_argument('--audit', default='reports/pakistan_data_audit.json')
    args = parser.parse_args()

    audit = build(args.input, args.output, args.audit)
    print(json.dumps({k: v for k, v in audit.items() if not k.startswith('_')}, indent=2))
    print('\nWrote:', audit['_paths'])


if __name__ == '__main__':
    main()
