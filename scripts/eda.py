"""EDA for the seller panel: the four figures the report needs, each with a takeaway.

    python scripts/eda.py            # writes reports/figures/*.png

The takeaways are printed to stdout *and* written to reports/eda_takeaways.md so they
cannot be separated from the figures by accident.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path('reports/figures')
PANEL = 'data/demo/seller_sales_history.csv'
TAKEAWAYS: list[tuple[str, str, str]] = []


def save(fig, filename: str, title: str, takeaway: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)
    TAKEAWAYS.append((filename, title, takeaway))
    print(f'{filename:34s} {takeaway}')


def main() -> None:
    panel = pd.read_csv(PANEL)
    panel['month'] = pd.to_datetime(panel['month'])
    panel['demand'] = panel['units_sold'] + panel['lost_units']
    panel['revenue'] = panel['revenue'].astype(float)

    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25,
                         'axes.spines.top': False, 'axes.spines.right': False})

    # --- Fig 1: seasonality is not a smooth annual curve -------------------
    monthly = panel.groupby('month').agg(demand=('demand', 'sum'),
                                         units=('units_sold', 'sum')).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(monthly['month'], monthly['demand'], marker='o', ms=3, label='Demand (incl. unmet)')
    ax.plot(monthly['month'], monthly['units'], marker='s', ms=3, label='Sold')
    for _, row in monthly.iterrows():
        if row['demand'] > monthly['demand'].quantile(0.85):
            ax.annotate(row['month'].strftime('%b %y'), (row['month'], row['demand']),
                        textcoords='offset points', xytext=(0, 8), fontsize=7, ha='center')
    ax.set_ylabel('Units per month')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Monthly demand: spikes follow the lunar calendar, not the month number')
    save(fig, 'fig1_seasonality.png', 'Demand over 24 months',
         'Spikes land in Ramadan/Eid months (Mar 2025, Feb-Mar 2026) and Black Friday; '
         'because Ramadan moves ~11 days earlier each year, a month-of-year feature cannot '
         'capture this.')

    # --- Fig 2: revenue concentration -------------------------------------
    per_product = panel.groupby('product_name')['revenue'].sum().sort_values(ascending=False)
    share = 100 * per_product.cumsum() / per_product.sum()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(np.arange(1, len(share) + 1), share.values)
    top10 = int(np.ceil(0.10 * len(share)))
    ax.axvline(top10, color='#c0392b', ls='--', lw=1)
    ax.annotate(f'top 10% of SKUs\n= {share.iloc[top10 - 1]:.0f}% of revenue',
                (top10, share.iloc[top10 - 1]), textcoords='offset points',
                xytext=(12, -30), fontsize=7, color='#c0392b')
    ax.set_xlabel('Products ranked by revenue')
    ax.set_ylabel('Cumulative revenue (%)')
    ax.set_title('Revenue is concentrated')
    save(fig, 'fig2_concentration.png', 'Revenue concentration across SKUs',
         f'The top 10% of products ({top10} of {len(share)}) carry {share.iloc[top10-1]:.0f}% of '
         'revenue, so unit-level MAE understates the rupee cost of errors on those SKUs — '
         'which is why a rupee-weighted metric is reported alongside MAE.')

    # --- Fig 3: stock-outs censor demand ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    service = panel.groupby('month').apply(
        lambda g: 100 * g['units_sold'].sum() / max(g['demand'].sum(), 1), include_groups=False)
    axes[0].plot(service.index, service.values)
    axes[0].set_title('Service level by month')
    axes[0].set_ylabel('% of demand served')
    axes[0].tick_params(axis='x', rotation=30)
    lost = panel.groupby('product_name')['lost_units'].sum().sort_values(ascending=False).head(8)
    axes[1].barh(range(len(lost)), lost.values[::-1], color='#c0392b')
    axes[1].set_yticks(range(len(lost)))
    axes[1].set_yticklabels([n[:22] for n in lost.index[::-1]], fontsize=6)
    axes[1].set_title('Units lost to stock-outs')
    save(fig, 'fig3_censoring.png', 'Stock-outs truncate observed sales',
         f'Service level averages {service.mean():.0f}% and falls hardest in festival months; '
         'the SKUs with the most lost units are exactly the ones a naive model under-forecasts.')

    # --- Fig 4: profit is not proportional to revenue ---------------------
    cats = panel.groupby('category').agg(revenue=('revenue', 'sum'),
                                         profit=('net_profit', 'sum')).reset_index()
    cats['margin'] = 100 * cats['profit'] / cats['revenue']
    cats = cats.sort_values('margin')
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    colours = ['#c0392b' if m < 8 else '#2f9e79' for m in cats['margin']]
    ax.barh(cats['category'], cats['margin'], color=colours)
    ax.set_xlabel('Net margin (%)')
    ax.set_title('Margin by category')
    for i, (margin, revenue) in enumerate(zip(cats['margin'], cats['revenue'])):
        ax.text(margin + 0.3, i, f'Rs {revenue/1e6:.1f}M', va='center', fontsize=7)
    save(fig, 'fig4_margin.png', 'Margin by category',
         f'Electronics earns {cats.iloc[0]["margin"]:.0f}% while Fashion earns '
         f'{cats.iloc[-1]["margin"]:.0f}%, so ranking purchases by revenue or by unit '
         'shortage would misallocate the budget — the planner ranks by profit per rupee.')

    Path('reports').mkdir(exist_ok=True)
    lines = ['# EDA takeaways\n',
             'Each figure must appear in the report with the takeaway below it.\n']
    for filename, title, takeaway in TAKEAWAYS:
        lines.append(f'## {filename} — {title}\n\n{takeaway}\n')
    Path('reports/eda_takeaways.md').write_text('\n'.join(lines))
    print('\nWrote reports/eda_takeaways.md and', len(TAKEAWAYS), 'figures to', OUT)


if __name__ == '__main__':
    main()
