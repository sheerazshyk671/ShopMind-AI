"""Clean Online Retail II and build a complete product-month panel."""
from pathlib import Path
import argparse
import pandas as pd

NON_PRODUCTS = {'POST', 'M', 'D', 'BANK CHARGES', 'DOT', 'C2', 'PADS', 'CRUK', 'AMAZONFEE', 'ADJUST', 'B'}


def clean_transactions(df):
    df = df.rename(columns={'Invoice': 'InvoiceNo', 'Price': 'UnitPrice', 'Customer ID': 'CustomerID'}).copy()
    required = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing transaction columns: {', '.join(sorted(missing))}")
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    codes = df['StockCode'].astype('string').str.strip().str.upper()
    invoices = df['InvoiceNo'].astype('string').str.strip().str.upper()
    mask = (invoices.notna() & ~invoices.str.startswith('C', na=False) & codes.notna()
            & ~codes.isin(NON_PRODUCTS) & df['Description'].notna()
            & df['Description'].astype('string').str.strip().ne('').fillna(False)
            & (df['UnitPrice'] > 0) & (df['Quantity'] > 0) & df['InvoiceDate'].notna())
    result = df.loc[mask, required].copy()
    result['StockCode'] = codes[mask]
    return result


def build_monthly_panel(df):
    df = clean_transactions(df)
    if df.empty:
        raise ValueError('No valid transactions remain after cleaning.')
    df['month'] = df['InvoiceDate'].dt.to_period('M')
    monthly = df.groupby(['StockCode', 'month'], observed=True)['Quantity'].sum()
    # Include zero-sales months across the full observed dataset period.
    months = pd.period_range(df['month'].min(), df['month'].max(), freq='M')
    nonzero = monthly.groupby(level=0).apply(lambda x: (x > 0).sum())
    first = df.groupby('StockCode')['month'].min()
    last = df.groupby('StockCode')['month'].max()
    eligible = nonzero[(nonzero >= 6) & ((last - first).map(lambda x: x.n + 1) >= 12)].index
    if len(eligible) == 0:
        raise ValueError('No products meet the 12-month history and 6-sales-month requirements.')
    idx = pd.MultiIndex.from_product([eligible, months], names=['StockCode', 'month'])
    panel = monthly.reindex(idx, fill_value=0).rename('monthly_units_sold').reset_index()
    # Exclude periods before a product first appeared; retain zero months afterwards.
    panel = panel[panel['month'] >= panel['StockCode'].map(first)].copy()
    panel['month'] = panel['month'].dt.to_timestamp()
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', default='data/raw/online_retail_II.xlsx')
    parser.add_argument('--output', default='data/processed/monthly_panel.csv')
    args = parser.parse_args()
    inputs = [Path(p) for p in args.input.split(',')]
    frames = []
    for path in inputs:
        if path.suffix.lower() == '.csv':
            frames.append(pd.read_csv(path, encoding='latin1', low_memory=False))
        else:
            frames.extend(pd.read_excel(path, sheet_name=None).values())
    panel = build_monthly_panel(pd.concat(frames, ignore_index=True))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    print(f'Saved {len(panel):,} rows to {args.output}')

if __name__ == '__main__':
    main()
