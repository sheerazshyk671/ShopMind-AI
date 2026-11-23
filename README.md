# 🛍️ ShopMind AI
> **Smart E-Commerce Sales Forecasting & Prescriptive Inventory Budget Allocation System**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

ShopMind AI is an end-to-end decision-support platform designed for e-commerce and retail merchants. It bridges the gap between **Machine Learning (Predictive Analytics)** and **Operations Research (Prescriptive Optimization)**:
1. Forecasts consumer demand rates using historical sales patterns and rolling lag indicators.
2. Solves a bounded **Mixed-Integer Linear Programming (MILP)** optimization problem to prescribe exact unit purchase quantities per product within a merchant's budget, maximizing ROI while strictly eliminating dead stock.

---

## 🌟 Key Features

- **📊 Multi-Model Demand Forecasting:** Benchmarks multiple regression families (Ridge L2, Decision Trees, KNN, Random Forest, XGBoost) on chronological hold-out splits without future lookahead leakage.
- **💰 Prescriptive Budget Knapsack (MILP):** Formulated using `scipy.optimize.milp` to maximize net margin subject to merchant budget ceilings, SKU concentration caps (max 25%), and warehouse stock deduction.
- **🗓️ Dynamic Seasonality & Festive Calendar:** Dynamic date recognition with domain-aware event regimes (Pre-Eid/Ramadan demand spikes, wedding season surges, and late-season clearance decay).
- **🛡️ Liquidity & Dead-Stock Protection:** When catalog restock demand is satisfied, surplus capital is preserved as unspent liquidity rather than tied up in excess inventory.
- **💻 Interactive Streamlit Dashboard:** User-friendly web GUI featuring executive KPI cards, interactive Plotly visualizations, custom procurement CSV upload, and exportable purchase orders.

---

## 🏗️ System Architecture

```text
├── app/
│   └── app.py                            # Interactive Streamlit Web Application
├── src/
│   ├── data_preprocessing.py             # Feature engineering, lag variables, leakage audit
│   ├── train_models.py                   # Model training, benchmarking & serialization
│   └── optimizer.py                      # MILP Bounded Knapsack engine with seasonality
├── models/
│   └── best_sales_model.joblib           # Serialized production Random Forest pipeline
├── notebooks/
│   └── ShopMind_AI_Pipeline.ipynb        # Documented Jupyter Notebook workflow
├── templates/
│   └── sample_pakistan_procurement.csv   # Sample retail procurement catalog (PKR)
└── reports/
    ├── model_comparison_results.csv      # Empirical model evaluation metrics
    └── *.png                             # High-resolution analytical figures
```

---

## ⚡ Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/sheerazshyk671/ShopMind-AI.git
cd ShopMind-AI
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
streamlit run app/app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Run the Research Notebook
```bash
jupyter notebook notebooks/ShopMind_AI_Pipeline.ipynb
```

---

## 📈 Empirical Benchmarking Summary

| Model Architecture | MAE (Error) | RMSE (Error) | $R^2$ Score | Selection Status |
| :--- | :---: | :---: | :---: | :--- |
| **Category Historical Mean (Baseline)** | 1.3420 | 1.5019 | +0.0046 | Baseline Floor |
| **Ridge Regression ($L_2$)** | 1.3395 | 1.5027 | +0.0035 | Linear Baseline |
| **Decision Tree (CART)** | 1.3220 | 1.4989 | +0.0085 | Non-Linear Fit |
| **KNN Regressor ($k=9$)** | 1.3674 | 1.5763 | -0.0965 | Curse of Dimensionality |
| **Random Forest Regressor** | **1.3249** | **1.4931** | **+0.0162** | 🏆 **Production Model** |
| **XGBoost Regressor** | 1.3291 | 1.4963 | +0.0119 | Competitive Ensemble |

---

## 🔬 Benchmark Citation

This work benchmarks against and methodologically extends:
> Amjad, U., Talpur, S. R., Ahmed, S., Farman, H., & Laila, U. (2024). *"Advanced Furniture Sales Forecasting Using Machine Learning and Deep Learning Techniques."* **ILMA Journal of Technology & Software Management (IJTSM)**, 5(2), 11–24.
