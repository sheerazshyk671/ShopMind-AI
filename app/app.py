"""
ShopMind AI — Interactive Streamlit Web Application
A Prescriptive Decision Support System for E-Commerce & Retail Sellers.
"""

import os
import sys
from pathlib import Path

# Ensure project root directory is always on sys.path regardless of where streamlit is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import importlib
import src.optimizer
importlib.reload(src.optimizer)
from src.optimizer import allocate_budget

# Page configuration
st.set_page_config(
    page_title="ShopMind AI — Smart Budget Allocation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Corporate Look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #0D9488;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 5px solid #1E3A8A;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #0D9488;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">🛍️ ShopMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Smart E-Commerce Sales Forecasting & Inventory Budget Allocation System (PKR)</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/isometric/100/shopping-cart-loaded.png", width=70)
st.sidebar.header("⚙️ Seller Configuration")

budget_input = st.sidebar.number_input(
    "Total Procurement Budget (PKR / Rs.)",
    min_value=50000,
    max_value=10000000,
    value=500000,
    step=25000,
    format="%d",
    help="Enter the total budget you have available to invest in stock."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Date & Seasonal Context")

procurement_date = st.sidebar.date_input(
    "Planning / Procurement Date",
    value=pd.Timestamp.now().date(),
    help="Current date used to infer Month, Day, Day-of-Week, and Season."
)

special_event = st.sidebar.selectbox(
    "Seasonal / Festive Timing",
    [
        "Regular Business Cycle",
        "Pre-Eid / Ramadan Shopping Rush (High Fashion & Footwear Surge)",
        "Late Winter / End of Season (Anti-Dead-Stock Clearance)",
        "Winter Wedding Season (Formal Suits, Crockery, Bedding)",
        "Summer Peak (T-Shirts, Fans, Juicers)"
    ],
    index=0,
    help="Select market conditions or festive rush to adjust demand multipliers."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Product Catalog Input")

data_source = st.sidebar.radio(
    "Catalog Source:",
    ["Use Preloaded Pakistani Retail Catalog", "Upload Custom Procurement CSV"],
    index=0
)

uploaded_file = None
if data_source == "Upload Custom Procurement CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (Required columns: product_id, product_name, category, wholesale_cost_pkr, selling_price_pkr, current_stock_in_hand)",
        type=["csv"]
    )
    with open("templates/sample_pakistan_procurement.csv", "rb") as f:
        st.sidebar.download_button(
            "📥 Download Sample CSV Template",
            data=f,
            file_name="sample_pakistan_procurement.csv",
            mime="text/csv"
        )

# Load data based on selection
if data_source == "Upload Custom Procurement CSV" and uploaded_file is not None:
    try:
        candidate_df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(candidate_df)} custom products!")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        candidate_df = pd.read_csv("templates/sample_pakistan_procurement.csv")
else:
    candidate_df = pd.read_csv("templates/sample_pakistan_procurement.csv")

# Main Content Area
col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_opt = st.button("⚡ Optimize Purchase Plan", type="primary", use_container_width=True)
with col_info:
    st.caption(f"Ready to replenish **{len(candidate_df)} items** with a budget of **Rs. {budget_input:,.0f}**")

# Run optimization
with st.spinner("Calculating optimal stock replenishment allocation..."):
    results_df, summary = allocate_budget(
        candidate_df, 
        total_budget_pkr=budget_input, 
        procurement_date=procurement_date, 
        special_event=special_event
    )

st.info(f"🗓️ **Active Context:** Date: **{procurement_date.strftime('%B %d, %Y')}** | Scenario: **{special_event}**")

# Display KPI Cards
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Budget</div>
        <div class="kpi-value">Rs. {summary['Total_Budget_PKR']:,.0f}</div>
        <div class="kpi-sub">Available Capital</div>
    </div>
    """, unsafe_allow_html=True)
    
with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Capital Invested</div>
        <div class="kpi-value">Rs. {summary['Total_Spent_PKR']:,.0f}</div>
        <div class="kpi-sub">{(summary['Total_Spent_PKR']/summary['Total_Budget_PKR']*100):.1f}% Budget Utilized</div>
    </div>
    """, unsafe_allow_html=True)
    
with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Unspent Cash</div>
        <div class="kpi-value">Rs. {summary['Remaining_Budget_PKR']:,.0f}</div>
        <div class="kpi-sub">Safeguarded Liquidity</div>
    </div>
    """, unsafe_allow_html=True)
    
with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Expected Profit</div>
        <div class="kpi-value">Rs. {summary['Projected_Net_Profit_PKR']:,.0f}</div>
        <div class="kpi-sub">Projected Net Margin</div>
    </div>
    """, unsafe_allow_html=True)
    
with k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Expected ROI</div>
        <div class="kpi-value">{summary['Expected_ROI_Percent']}%</div>
        <div class="kpi-sub">{summary['Total_Units_Procured']} New Units</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 📋 Recommended Stock Purchase Order (Replenishment Plan)")

# Filter and format table
display_df = results_df.copy()
display_df['Status'] = display_df['Recommended_Purchase_Units'].apply(
    lambda x: "✅ Purchase" if x > 0 else "❌ Skip (Stock Full)"
)

cols_order = [
    'Status', 'product_name', 'category', 'Seasonal_Impact', 'wholesale_cost_pkr', 'selling_price_pkr',
    'current_stock_in_hand', 'Predicted_Demand_Horizon', 'Net_Restock_Needed',
    'Recommended_Purchase_Units', 'Allocated_Cost_PKR', 'Projected_Profit_PKR'
]

formatted_df = display_df[cols_order].rename(columns={
    'product_name': 'Product Name',
    'category': 'Category',
    'Seasonal_Impact': 'Market / Seasonal Dynamics',
    'wholesale_cost_pkr': 'Wholesale Cost (Rs.)',
    'selling_price_pkr': 'Retail Price (Rs.)',
    'current_stock_in_hand': 'Current Stock in Hand',
    'Predicted_Demand_Horizon': 'Forecasted Total Demand',
    'Net_Restock_Needed': 'Net Restock Needed',
    'Recommended_Purchase_Units': 'Recommended Buy Qty',
    'Allocated_Cost_PKR': 'Allocated Spend (Rs.)',
    'Projected_Profit_PKR': 'Projected Profit (Rs.)'
})

st.dataframe(
    formatted_df.style.format({
        'Wholesale Cost (Rs.)': 'Rs. {:,.0f}',
        'Retail Price (Rs.)': 'Rs. {:,.0f}',
        'Allocated Spend (Rs.)': 'Rs. {:,.0f}',
        'Projected Profit (Rs.)': 'Rs. {:,.0f}',
        'Current Stock in Hand': '{:d}',
        'Forecasted Total Demand': '{:d}',
        'Net Restock Needed': '{:d}',
        'Recommended Buy Qty': '{:d}'
    }),
    use_container_width=True,
    height=420
)

# Export CSV Button
csv_data = formatted_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Export Recommended Purchase Order (CSV)",
    data=csv_data,
    file_name="ShopMind_AI_Purchase_Order.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("### 📈 Visual Replenishment Insights")

vcol1, vcol2 = st.columns(2)

with vcol1:
    cat_spending = results_df.groupby('category')['Allocated_Cost_PKR'].sum().reset_index()
    fig_pie = px.pie(
        cat_spending, 
        values='Allocated_Cost_PKR', 
        names='category', 
        title="Budget Allocation by Product Category",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)
    
with vcol2:
    fig_bar = px.bar(
        results_df[results_df['Recommended_Purchase_Units'] > 0],
        x='product_name',
        y=['current_stock_in_hand', 'Recommended_Purchase_Units', 'Predicted_Demand_Horizon'],
        barmode='group',
        title="Current Stock in Hand vs Recommended Buy vs Total Forecasted Demand",
        labels={'value': 'Units', 'product_name': 'Product', 'variable': 'Inventory Metric'},
        color_discrete_map={
            'current_stock_in_hand': '#F59E0B',
            'Recommended_Purchase_Units': '#1E3A8A',
            'Predicted_Demand_Horizon': '#0D9488'
        }
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)
