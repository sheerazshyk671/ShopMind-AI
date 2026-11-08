"""ShopMind AI — demand forecasting and budget-aware buying plan for Pakistani sellers.

Run:  streamlit run app/app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.planning import forecast_request  # noqa: E402

DEMO = ROOT / 'data/demo'
MODELS = ROOT / 'models'

st.set_page_config(page_title='ShopMind AI — Buying plan', page_icon='◈', layout='wide')


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'Inter',sans-serif;color:#14232e}
[data-testid="stAppViewContainer"] {background:#f7f9fa}
[data-testid="stSidebar"] {background:#0f2b33}
[data-testid="stSidebar"] * {color:#dfecef}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] select {color:#12262e}
.block-container {padding-top:2.2rem;max-width:1500px}
h1 {font-size:2rem!important;font-weight:800!important;letter-spacing:-.03em}
h2,h3 {font-weight:700!important;letter-spacing:-.02em}
.eyebrow {font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:#1c8578}
.subhead {color:#617280;font-size:.95rem;margin-top:-.4rem}
.card {background:#fff;border:1px solid #e5edee;border-radius:12px;padding:18px 16px;min-height:112px}
.card .label {font-size:.74rem;color:#6d7f88;font-weight:600}
.card .value {font-size:1.42rem;font-weight:800;color:#142d36;margin:.35rem 0 .15rem;letter-spacing:-.03em}
.card .note {font-size:.71rem;color:#8ba0a4}
.pill {display:inline-block;padding:2px 9px;border-radius:20px;font-size:.72rem;font-weight:700}
.bad {background:#fdecec;color:#a5352f}
.good {background:#e8f6ef;color:#1c7a55}
.warn {background:#fff5e2;color:#8a5c12}
.assume {background:#eef5f7;border:1px solid #d8e7ec;border-radius:10px;padding:14px 16px;font-size:.83rem;color:#3b5a63;line-height:1.6}
</style>''', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str, mtime: float):
    return joblib.load(path)


def resolve_model(choice: str, summary: dict):
    """Return the model object for the selected option, plus a label for the UI."""
    if choice.startswith('Stage 2') and (MODELS / 'seller_model.pkl').exists():
        path = MODELS / 'seller_model.pkl'
        return load_model(str(path), path.stat().st_mtime), 'stage 2 (seller-calibrated)'
    path = MODELS / 'base_model.pkl'
    if path.exists():
        return load_model(str(path), path.stat().st_mtime), 'stage 1 (market model)'
    path = MODELS / 'seller_model.pkl'
    if path.exists():
        return load_model(str(path), path.stat().st_mtime), 'stage 2 (seller-calibrated)'
    return None, 'none'


@st.cache_data
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def read_upload(upload) -> pd.DataFrame:
    if upload.name.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(upload)
    return pd.read_csv(upload)


def validate_request(df: pd.DataFrame) -> pd.DataFrame:
    required = ['ProductID', 'ProductName', 'CurrentPrice', 'CurrentStock']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing required column(s): ' + ', '.join(missing))
    if df.empty:
        raise ValueError('The product file has no rows.')
    df = df.copy()
    df['ProductID'] = df['ProductID'].astype(str).str.strip()
    if df['ProductID'].duplicated().any():
        dupes = ', '.join(df.loc[df['ProductID'].duplicated(), 'ProductID'].head(3))
        raise ValueError(f'ProductID must be unique. Duplicated: {dupes}')
    for column in ('CurrentPrice', 'CurrentStock'):
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if df[column].isna().any():
            raise ValueError(f'{column} must be numeric on every row.')
    if (df['CurrentPrice'] <= 0).any():
        raise ValueError('CurrentPrice must be greater than zero.')
    if (df['CurrentStock'] < 0).any():
        raise ValueError('CurrentStock cannot be negative.')
    return df


def money(value: float) -> str:
    return f'Rs {value:,.0f}'


# ---------------------------------------------------------------------------
# Sidebar — the audience's input
# ---------------------------------------------------------------------------
history_path = DEMO / 'seller_sales_history.csv'
sample_request = DEMO / 'seller_purchase_request.csv'

with st.sidebar:
    st.markdown('<div style="font-size:1.25rem;font-weight:800;letter-spacing:-.03em">'
                '<span style="background:#4fc3ad;color:#0f2b33;padding:3px 8px;border-radius:7px">◈</span>'
                ' ShopMind AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#93aeb3;font-size:.82rem;margin-top:.6rem">'
                'Buy the right stock, with the money you actually have.</p>',
                unsafe_allow_html=True)

    st.markdown('<div style="font-size:.68rem;letter-spacing:.14em;font-weight:700;'
                'color:#7fa3a6;margin-top:1.4rem">THIS MONTH\'S BUDGET</div>', unsafe_allow_html=True)
    budget = st.number_input('Budget (PKR)', min_value=0.0, value=500_000.0, step=50_000.0,
                             format='%.0f', label_visibility='collapsed')

    st.markdown('<div style="font-size:.68rem;letter-spacing:.14em;font-weight:700;'
                'color:#7fa3a6;margin-top:1.2rem">PRODUCTS TO BUY</div>', unsafe_allow_html=True)
    upload = st.file_uploader('Purchase request CSV', type=['csv', 'xlsx'],
                              label_visibility='collapsed',
                              help='Columns: ProductID, ProductName, CurrentPrice, CurrentStock, '
                                   'optionally RetailPrice, Category, UnitsWanted.')
    st.download_button('↓ Download a demo request', sample_request.read_bytes(),
                       'seller_purchase_request.csv', 'text/csv', width="stretch")

    st.markdown('<div style="font-size:.68rem;letter-spacing:.14em;font-weight:700;'
                'color:#7fa3a6;margin-top:1.2rem">SALES HISTORY (FORECASTING)</div>', unsafe_allow_html=True)
    with_history = st.toggle('Use my 12-month sales history', value=True,
                             help='Without history, demand falls back to the quantities you asked for.')
    history_upload = st.file_uploader('History CSV', type=['csv'], label_visibility='collapsed',
                                      help='One row per product per month: product_id, month, '
                                           'units_sold (optionally category, promo_days, discount_pct).')

    st.markdown('<div style="font-size:.68rem;letter-spacing:.14em;font-weight:700;'
                'color:#7fa3a6;margin-top:1.2rem">MODEL</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        'Model', ['Stage 1 only (default)', 'Stage 2: calibrate to my history'],
        label_visibility='collapsed',
        help='Stage 2 was selected on the validation window, but it did not improve '
             'accuracy on the untouched test window. Stage 1 is therefore the default.')


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
summary_path = ROOT / 'reports' / 'training_summary.json'
try:
    TRAINING_SUMMARY = __import__('json').loads(summary_path.read_text())
except (OSError, ValueError):
    TRAINING_SUMMARY = {}
model, model_label = resolve_model(model_choice, TRAINING_SUMMARY)
history = None
history_label = 'none'
if with_history:
    if history_upload is not None:
        history = read_upload(history_upload)
        history_label = history_upload.name
    elif history_path.exists():
        history = load_csv(str(history_path), history_path.stat().st_mtime)
        history_label = 'demo 24-month history'

if upload is not None:
    request = read_upload(upload)
    source = upload.name
else:
    request = load_csv(str(sample_request), sample_request.stat().st_mtime)
    source = 'demo request (36 products)'


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Buying plan · Pakistani e-commerce</div>', unsafe_allow_html=True)
st.title('What should I buy this month?')
st.markdown(f'<p class="subhead">Forecast from your sales history, then spend your budget where '
            f'the margin is. Source: <b>{source}</b> · history: <b>{history_label}</b> · '
            f'model: <b>{model_label}</b></p>', unsafe_allow_html=True)

if model is None:
    st.error('No trained model found. Run `python scripts/train_pipeline.py --demo` first.')
    st.stop()

# ---------------------------------------------------------------------------
# Run the plan
# ---------------------------------------------------------------------------
try:
    request = validate_request(request)
    result = forecast_request(request, model, budget=budget, history=history)
except Exception as exc:  # noqa: BLE001 - surface the reason to the user
    st.error(f'Could not build a plan: {exc}')
    st.stop()

plan, summary = result.plan, result.summary

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
columns = st.columns(5)
cards = [
    ('Budget', money(summary['budget']), 'what you said you have'),
    ('Recommended spend', money(summary['investment']), f"{summary['utilisation_pct']:.0f}% of budget used"),
    ('Expected profit', money(summary['expected_profit']),
     f"ROI {summary['expected_roi_pct']:.1f}% after fees & returns"),
    ('Products funded', f"{summary['funded_products']} / {summary['products']}",
     f"{summary['partial_fill_products']} still short after buying"),
    ('Loss-making items', f"{summary['unprofitable_products']}",
     'negative margin at current prices'),
]
for column, (label, value, note) in zip(columns, cards):
    column.markdown(f'<div class="card"><div class="label">{label}</div>'
                    f'<div class="value">{value}</div><div class="note">{note}</div></div>',
                    unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
left, right = st.columns([1.6, 1], gap='large')

with left:
    st.subheader('Where the money goes')
    funded = plan[plan['Investment'] > 0].sort_values('Investment', ascending=True)
    if funded.empty:
        st.info('Nothing is affordable at this budget. Increase the budget or check the prices.')
    else:
        chart = funded.tail(12)
        fig = px.bar(chart, x='Investment', y='ProductName', orientation='h',
                     color='ExpectedProfit',
                     color_continuous_scale=['#c94f4f', '#f0c419', '#2f9e79'],
                     labels={'Investment': 'Investment (PKR)', 'ProductName': ''})
        fig.update_layout(height=380, margin=dict(l=0, r=10, t=10, b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                          coloraxis_colorbar=dict(title='Expected<br>profit (PKR)', thickness=12),
                          xaxis=dict(gridcolor='#eef3f3'), yaxis=dict(gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, width="stretch",
                        config={'displayModeBar': False})

with right:
    st.subheader('Demand vs stock on hand')
    focus = plan.sort_values('ShortageUnits', ascending=False).head(10).iloc[::-1]
    figure = go.Figure()
    figure.add_bar(name='Current stock', y=focus['ProductName'], x=focus['CurrentStock'],
                   orientation='h', marker_color='#cfe0e2')
    figure.add_bar(name='Forecast + safety stock', y=focus['ProductName'],
                   x=focus['RequiredInventory'], orientation='h', marker_color='#2f9e79')
    figure.update_layout(barmode='group', height=380, margin=dict(l=0, r=10, t=10, b=0),
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                         legend=dict(orientation='h', y=1.12, x=0),
                         xaxis=dict(title='Units', gridcolor='#eef3f3'),
                         yaxis=dict(title=''))
    st.plotly_chart(figure, width="stretch", config={'displayModeBar': False})

# ---------------------------------------------------------------------------
# The buying list
# ---------------------------------------------------------------------------
st.subheader('Buy this')
buy = plan[plan['RecommendedPurchase'] > 0].copy()
if buy.empty:
    st.info('No purchases recommended at this budget.')
else:
    display = buy[['ProductName', 'Category', 'CurrentPrice', 'RetailPrice', 'CurrentStock',
                   'RecommendedPurchase', 'Investment', 'ProfitPerUnit', 'ExpectedProfit']]
    st.dataframe(
        display.style.format({
            'CurrentPrice': '{:,.0f}', 'RetailPrice': '{:,.0f}', 'Investment': '{:,.0f}',
            'ProfitPerUnit': '{:,.0f}', 'ExpectedProfit': '{:,.0f}',
        }).map(lambda v: 'color:#a5352f' if isinstance(v, (int, float)) and v < 0 else '',
               subset=['ProfitPerUnit', 'ExpectedProfit']),
        width="stretch", hide_index=True,
        column_config={
            'ProductName': 'Product', 'CurrentPrice': 'Buy price', 'RetailPrice': 'Sell price',
            'CurrentStock': 'Stock', 'RecommendedPurchase': 'Buy units',
            'Investment': 'Spend (PKR)', 'ProfitPerUnit': 'Profit/unit',
            'ExpectedProfit': 'Expected profit',
        })
    st.caption('Buy price is what you pay per unit; expected profit is after marketplace '
               'commission, shipping, ad spend and the expected return rate for that category.')

    download = buy.to_csv(index=False).encode('utf-8')
    st.download_button('↓ Export this buying list (CSV)', download,
                       'shopmind_buying_list.csv', 'text/csv')

# ---------------------------------------------------------------------------
# Skipped items, with a reason
# ---------------------------------------------------------------------------
st.subheader('Not buying — and why')
skipped = plan[plan['RecommendedPurchase'] == 0].copy()
if skipped.empty:
    st.caption('Every requested product is at least partly funded.')
else:
    def reason(row) -> str:
        if row['ShortageUnits'] <= 0:
            return 'Already enough stock'
        if row['ProfitPerUnit'] < 0:
            return 'Selling below cost after fees'
        if row['CurrentPrice'] > summary['unallocated']:
            return 'Budget ran out before this item'
        return 'Budget ran out before this item'

    skipped['Reason'] = skipped.apply(reason, axis=1)
    shown = skipped[['ProductName', 'Category', 'CurrentPrice', 'CurrentStock', 'Reason']]
    st.dataframe(shown, width="stretch", hide_index=True,
                 column_config={'CurrentPrice': 'Buy price', 'CurrentStock': 'Stock'})

# ---------------------------------------------------------------------------
# Honest assumptions
# ---------------------------------------------------------------------------
with st.expander('How this plan was calculated (read this before acting on it)'):
    rows = ''.join(f'<li>{line}</li>' for line in result.assumptions)
    st.markdown(f'<div class="assume"><b>Model</b><br>'
                f'The demand forecast comes from a two-stage model: a general market model, '
                f'then a calibration to this seller. It predicts next month\'s units as a '
                f'ratio to your trailing three-month average.<br><br>'
                f'<b>Assumptions</b><ul>{rows}</ul>'
                f'<b>Limitations</b><ul>'
                f'<li>The demo history is synthetic and generated from published Pakistani price '
                f'bands and return rates; see <code>data/demo/DATA_CARD.md</code>.</li>'
                f'<li>Category economics are averages, not your own fee schedule.</li>'
                f'<li>Forecast error is material (see the report). Treat this as decision support '
                f'that ranks options, not as a precise prediction.</li>'
                f'<li>Greedy allocation is not a globally optimal knapsack solution.</li>'
                f'</ul></div>', unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)
st.caption('ShopMind AI · Sukkur IBA University ML semester project · '
           'forecasts are estimates, not guarantees')
