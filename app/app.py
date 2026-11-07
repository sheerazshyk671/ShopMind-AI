"""ShopMind AI Streamlit inventory planning dashboard."""
from pathlib import Path
import sys
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.forecasting import forecast_seller
from src.optimization import optimize_budget
from src.seller_history import history_columns

st.set_page_config(page_title='ShopMind AI | Inventory planning', page_icon='◈', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;color:#182b3d}
[data-testid="stAppViewContainer"] {background:#f6f8fa}
[data-testid="stSidebar"] {background:#102d38}
[data-testid="stSidebar"] * {color:#e6f0ef}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {color:#17313d}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {background:#1c3b46;border-radius:10px;padding:8px}
.block-container {padding-top:2rem;padding-bottom:3rem;max-width:1550px}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;letter-spacing:-.035em}
h1 {font-size:2.1rem!important;font-weight:800!important}
h3 {font-size:1.05rem!important;font-weight:700!important}
p {color:#637482}
.eyebrow {font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:#23877e;margin-bottom:.5rem}
.subhead {color:#6d7d89;font-size:.95rem;margin-top:-.5rem;margin-bottom:1.6rem}
.brand {font-family:Manrope,sans-serif;font-size:1.3rem;font-weight:800;color:white;letter-spacing:-.04em}
.brand-icon {display:inline-block;background:#54c3af;color:#10323a;padding:4px 9px;border-radius:8px;margin-right:8px}
.side-label {color:#88a6aa!important;font-size:.68rem;letter-spacing:.13em;font-weight:700;margin-top:2rem;margin-bottom:.5rem}
.sidebar-caption {font-size:.8rem;color:#a7bdbe!important;line-height:1.5}
.metric-card {background:white;border:1px solid #e6ecee;border-radius:12px;padding:20px 18px;min-height:125px;box-shadow:0 2px 8px #14332c06}
.metric-label {font-size:.76rem;color:#73838d;font-weight:600}
.metric-value {font-family:Manrope,sans-serif;color:#18333d;font-weight:800;font-size:1.45rem;margin:9px 0 3px;letter-spacing:-.04em}
.metric-note {font-size:.72rem;color:#88a09e}
.section-kicker {font-size:.73rem;color:#82939b;margin-top:-.5rem}
.insight {background:#eaf6f2;border:1px solid #cfe9df;border-radius:10px;padding:16px 18px;color:#28564c;font-size:.9rem;line-height:1.55}
[data-testid="stButton"] button[kind="primary"] {background:#56c3ad;color:#103b36;border:none;font-weight:700;width:100%}
[data-testid="stDownloadButton"] button {border-radius:8px}
</style>''', unsafe_allow_html=True)

sample_path = ROOT / 'data/sample_seller_products.csv'
model_path = ROOT / 'models/demand_model.pkl'
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-icon">◈</span>ShopMind <span style="color:#70d1bf">AI</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-caption" style="margin-top:12px">Smarter inventory decisions,<br>made with confidence.</p>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">PLAN SETTINGS</div>', unsafe_allow_html=True)
    budget = st.number_input('Monthly budget (Rs.)', min_value=0.0, value=500000.0, step=10000.0, format='%.2f')
    st.markdown('<div class="side-label">YOUR PRODUCT DATA</div>', unsafe_allow_html=True)
    upload = st.file_uploader('Upload seller file', type=['csv', 'xlsx'], help='Required: ProductID, ProductName, CurrentPrice, CurrentStock and at least six monthly sales columns Sales_01, Sales_02, ... (oldest first).')
    st.download_button('↓ Download CSV template', sample_path.read_bytes(), 'sample_seller_products.csv', 'text/csv', use_container_width=True)
    st.markdown('<p class="sidebar-caption">No file? Use the sample data to explore the dashboard.</p>', unsafe_allow_html=True)
    generate = st.button('Generate purchase plan →', type='primary', use_container_width=True)
    st.markdown('<div style="margin-top:3rem;border-top:1px solid #31505a;padding-top:16px" class="sidebar-caption">◉ &nbsp;Built for better decisions<br><span style="color:#779699">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Demand meets your budget.</span></div>', unsafe_allow_html=True)


def validate_products(df):
    required = ['ProductID', 'ProductName', 'CurrentPrice', 'CurrentStock']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    if df.empty:
        raise ValueError('The product file is empty.')
    sales_cols = history_columns(df)
    if not sales_cols and 'RecentMonthlySales' not in df.columns:
        raise ValueError('Provide at least six monthly sales columns (Sales_01 onward) or RecentMonthlySales.')
    df = df[required + (sales_cols if sales_cols else ['RecentMonthlySales'])].copy()
    for col in ['ProductID', 'ProductName']:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{col} cannot be empty.')
    if df['ProductID'].astype(str).duplicated().any():
        raise ValueError('ProductID values must be unique.')
    for col in ['CurrentPrice', 'CurrentStock'] + (sales_cols if sales_cols else ['RecentMonthlySales']):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f'{col} must contain finite, non-negative numbers.')
    if (df['CurrentPrice'] <= 0).any():
        raise ValueError('CurrentPrice must be greater than zero.')
    return df


def build_plan():
    if upload:
        df = pd.read_csv(upload) if upload.name.lower().endswith('.csv') else pd.read_excel(upload)
    else:
        df = pd.read_csv(sample_path)
    df = validate_products(df)
    trained = model_path.exists()
    if trained:
        df = forecast_seller(df, joblib.load(model_path))
    else:
        # Transparent baseline allows exploring the interface without the external dataset.
        sales_cols = history_columns(df)
        df['ForecastedDemand'] = df[sales_cols[-3:]].mean(axis=1) if sales_cols else df['RecentMonthlySales']
        df['SafetyStock'] = 1.28 * df[sales_cols[-6:]].std(axis=1) if sales_cols else 0.0
    calibration = df.attrs.get('calibration', 1.0)
    calibration_samples = df.attrs.get('calibration_samples', 0)
    return optimize_budget(df, budget), trained, float(budget), upload.name if upload else 'Sample product data', calibration, calibration_samples, bool(history_columns(df))

if generate or 'plan' not in st.session_state:
    try:
        st.session_state.plan = build_plan()
    except Exception as exc:
        st.error(f'Could not generate plan: {exc}')
        st.stop()
plan, trained, plan_budget, source, calibration, calibration_samples, has_history = st.session_state.plan
investment = plan['Investment'].sum()
remaining = max(0, plan_budget - investment)
utilization = investment / plan_budget * 100 if plan_budget else 0

st.markdown('<div class="eyebrow">INVENTORY INTELLIGENCE &nbsp; / &nbsp; OVERVIEW</div>', unsafe_allow_html=True)
st.title('Your inventory, optimized.')
st.markdown('<p class="subhead">A clearer picture of what to buy, and where your budget goes.</p>', unsafe_allow_html=True)
if not trained:
    st.warning('Baseline preview: no trained model found. Forecasts use the recent three-month average (or the single recent-sales value). Train a model to enable ML predictions.', icon='ℹ️')
else:
    st.caption(f'ML demand-ratio model active · Next calendar month · Seller calibration: {calibration:.2f}× from {calibration_samples} historical examples')
    st.info('The historical holdout showed weak predictive performance. Validate forecasts against your own sales before making purchasing decisions.')
    st.caption('Sales history is ordered oldest to newest; the last month is assumed to be the current calendar month.')
st.caption(f'PLAN SOURCE  ·  {source}     |     Change settings and select Generate purchase plan to refresh.')
cols = st.columns(5)
metrics = [('Total budget', f'Rs. {plan_budget:,.0f}', 'Available to allocate'), ('Recommended investment', f'Rs. {investment:,.0f}', 'Across your purchase plan'), ('Remaining budget', f'Rs. {remaining:,.0f}', 'Unallocated funds'), ('Budget utilization', f'{utilization:.1f}%', 'Of your monthly budget'), ('Total products', f'{len(plan):,}', 'In your product catalog')]
for col, (label, value, note) in zip(cols, metrics):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)
left, right = st.columns([1.65, 1], gap='large')
config = dict(displayModeBar=False, responsive=True)
with left:
    st.subheader('Demand forecast')
    st.markdown('<p class="section-kicker">Estimated units needed next month, by product</p>', unsafe_allow_html=True)
    chart = plan.sort_values('ForecastedDemand', ascending=True).tail(10)
    fig = px.bar(chart, x='ForecastedDemand', y='ProductName', orientation='h', color_discrete_sequence=['#3fae9b'])
    fig.update_layout(height=310, margin=dict(l=0,r=20,t=10,b=0), plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title=None, font=dict(color='#74858d'), xaxis=dict(showgrid=True,gridcolor='#edf2f1'), yaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True, config=config)
with right:
    st.subheader('Budget allocation')
    st.markdown('<p class="section-kicker">Where your recommended investment goes</p>', unsafe_allow_html=True)
    allocated = plan[plan['Investment'] > 0]
    if allocated.empty:
        st.info('No purchases recommended for this budget.')
    else:
        fig = px.pie(allocated, names='ProductName', values='Investment', hole=.7, color_discrete_sequence=['#168d80','#47b7a3','#8ed2bd','#b8ded2','#2d6267','#d7e9d8','#6d9f9a','#c1d4cf'])
        fig.update_traces(textinfo='none', hovertemplate='%{label}<br>Rs. %{value:,.0f}<extra></extra>')
        fig.update_layout(height=310, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='v', font=dict(size=11)), showlegend=True)
        st.plotly_chart(fig, use_container_width=True, config=config)

st.subheader('Purchase recommendations')
st.markdown('<p class="section-kicker">Your prioritized, budget-aware purchase plan</p>', unsafe_allow_html=True)
display = plan[['ProductName','ForecastedDemand','CurrentStock','CurrentPrice','RecommendedPurchase','Investment','Priority']].rename(columns={'ProductName':'Product name','ForecastedDemand':'Forecasted demand','CurrentStock':'Current stock','CurrentPrice':'Price (Rs.)','RecommendedPurchase':'Purchase units','Investment':'Investment (Rs.)'})
styled = display.style.map(lambda value: {'High': 'color: #ad5c24; background-color: #fff2df', 'Medium': 'color: #98711c; background-color: #fff9dd', 'Low': 'color: #358774; background-color: #eaf6f2'}.get(value, ''), subset=['Priority'])
st.dataframe(styled, use_container_width=True, hide_index=True, column_config={'Forecasted demand':st.column_config.NumberColumn(format='%.1f'), 'Price (Rs.)':st.column_config.NumberColumn(format='%.2f'), 'Investment (Rs.)':st.column_config.NumberColumn(format='%.2f')})
st.download_button('↓ Export recommendations', plan.to_csv(index=False).encode('utf-8'), 'shopmind_purchase_plan.csv', 'text/csv')
st.markdown('<br>', unsafe_allow_html=True)
c1,c2 = st.columns([1.5,1], gap='large')
with c1:
    st.subheader('Stock coverage')
    st.markdown('<p class="section-kicker">Current stock compared with required inventory</p>', unsafe_allow_html=True)
    chart = plan.head(10)
    fig = go.Figure()
    fig.add_bar(name='Current stock', x=chart['ProductName'], y=chart['CurrentStock'], marker_color='#c7dedb')
    fig.add_bar(name='Required inventory', x=chart['ProductName'], y=chart['RequiredInventory'], marker_color='#338f83')
    fig.update_layout(barmode='group', height=300, margin=dict(l=0,r=0,t=15,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', font=dict(color='#74858d'), yaxis=dict(gridcolor='#edf2f1',title='Units'), xaxis=dict(title=None), legend=dict(orientation='h', y=1.15))
    st.plotly_chart(fig, use_container_width=True, config=config)
with c2:
    st.subheader('Planning insights')
    st.markdown('<p class="section-kicker">A little context for your next move</p>', unsafe_allow_html=True)
    if investment > 0:
        top = plan.loc[plan['Investment'].idxmax()]
        st.markdown(f'<div class="insight"><b>✦ &nbsp;Largest allocation</b><br>{str(top["ProductName"]).replace("<", "&lt;").replace(">", "&gt;")} receives Rs. {top["Investment"]:,.0f} for {int(top["RecommendedPurchase"])} units, based on its shortage and price-weighted priority.</div>', unsafe_allow_html=True)
    else:
        st.info('No budget has been allocated. Check stock levels and budget.')
    if has_history:
        st.caption('Safety stock uses 1.28 × the standard deviation of the latest six monthly sales values. Forecasts are estimates, not guarantees.')
    else:
        st.caption('Safety stock is zero because no six-month history was provided. Forecasts are estimates, not guarantees.')
