# 🛍️ ShopMind AI — Complete End-to-End Project Documentation

**Sukkur IBA University — Department of Computer Science**  
**Course:** Machine Learning (BS-V — CS, SE & AI) — Fall 2026  
**Instructors:** Dr. Muhammad Ismail Mangrio, Dr. Zakriya, Dr. Adil Khan & Dr. Saif Hassan  
**Project Title:** ShopMind AI: Smart E-Commerce Sales Forecasting & Inventory Budget Allocation System  

---

## 📌 Executive Summary
ShopMind AI is a prescriptive decision-support platform designed for e-commerce and retail sellers in Pakistan. It combines **Supervised Machine Learning aligned with the Sukkur IBA Course Outline** for empirical sales demand forecasting with **Operations Research (Bounded Knapsack Integer Linear Programming via SciPy MILP)** to calculate the exact, profit-maximizing purchase quantity per product within a seller-defined budget in Pakistani Rupees (PKR / Rs.).

---

## 🏛️ Academic Literature Benchmark
This project benchmarks against and methodologically extends the following peer-reviewed research paper co-authored by Sukkur IBA University faculty:

> **Benchmark Paper:**  
> Amjad, U., **Talpur, S. R. (Sukkur IBA University)**, Ahmed, S., Farman, H., & Laila, U. (2024).  
> *"Advanced Furniture Sales Forecasting Using Machine Learning and Deep Learning Techniques."*  
> **ILMA Journal of Technology & Software Management (IJTSM)**, Vol. 5, Issue 2, pp. 11–24.

---

## 🎓 Direct Alignment with Sukkur IBA Course Outline Weeks
Every machine learning model evaluated in this project directly maps to the official weekly syllabus of the Machine Learning course:

1. **Weeks 2–3 (Supervised Learning I & II):** *Decision Trees, Entropy, Information Gain, CART, Pruning Strategies.*  
   $\rightarrow$ **Model:** `DecisionTreeRegressor` (Achieved lowest MAE = 1.3220).
2. **Week 5 (Supervised Learning III):** *Linear & Logistic Regression, Cost Functions, L1/L2 Regularization.*  
   $\rightarrow$ **Model:** `Ridge(alpha=10.0)` (L2 Regularized Linear Regression).
3. **Week 10 (Instance-based & Distance Learning):** *k-Nearest Neighbor (KNN) Algorithm.*  
   $\rightarrow$ **Model:** `KNeighborsRegressor(n_neighbors=9)`.
4. **Week 13 (Ensemble Learning):** *Bagging, Boosting, Committees of Hypotheses, Random Forest, XGBoost.*  
   $\rightarrow$ **Models:** `RandomForestRegressor` (Winning Model, Best $R^2$ = +0.0162) & `XGBRegressor`.

---

## 📊 Benchmark Model Results (Sukkur IBA Guideline Section 5)

| Course Outline Week | Model Family | Model Name | MAE (Units) | RMSE | $R^2$ Score | MAPE (%) | Status |
|---|---|---|---|---|---|---|---|
| **Course Baseline** | **Trivial Baseline** | Category Historical Mean | 1.3420 | 1.5019 | 0.0046 | 76.48% | Baseline to Beat |
| **Week 5** | **Linear / Regularized** | Ridge Regression (L2) | 1.3395 | 1.5027 | 0.0035 | 76.77% | Classical Course Model |
| **Weeks 2–3** | **Non-Linear Trees** | **Decision Tree (CART)** | **1.3220** | 1.4989 | 0.0085 | 74.68% | 🏆 **Lowest MAE Error** |
| **Week 10** | **Instance-based** | KNN Regressor ($k=9$) | 1.3674 | 1.5763 | -0.0965 | 79.13% | Curse of Dimensionality |
| **Week 13** | **Ensemble (Bagging)** | **Random Forest Regressor** | **1.3249** | **1.4931** | **+0.0162** | 75.95% | 🏆 **Best Overall $R^2$ & RMSE** |
| **Week 13** | **Ensemble (Boosting)** | XGBoost Regressor | 1.3291 | 1.4963 | 0.0119 | 75.52% | Strong Boosting Result |

---

## 🔍 Error Analysis (Sukkur IBA Guideline Section 6)

| Order Date | Category | Product | Actual Qty | Predicted Qty | Residual | Applied Discount | Diagnostic Reason |
|---|---|---|---|---|---|---|---|
| 2018-12-05 | Fashion | Shirts | 5.0 | 2.08 | 2.92 | 40% | Flash sale promotion spike |
| 2018-12-06 | Fashion | Fossil Watch | 5.0 | 2.15 | 2.85 | 50% | Half-price year-end campaign |
| 2018-11-13 | Fashion | Running Shoes | 5.0 | 2.18 | 2.82 | 20% | Winter fitness promotional demand |
| 2018-11-10 | Fashion | Fossil Watch | 5.0 | 2.19 | 2.81 | 20% | Luxury accessory holiday demand |

---

## 🗓️ Dynamic Seasonality & Festive Calendar Intelligence
The application features live calendar and festive regime modeling tailored to Pakistani e-commerce:
- **Date Picker:** Inactive/active calendar date extraction (Month, Day, DayOfWeek, Quarter, Weekend).
- **Pre-Eid / Ramadan Shopping Rush:** +110% demand surge on ethnic apparel, footwear, watches, and bedding.
- **Late Winter / End of Season:** -55% anti-dead-stock decay penalty on heavy apparel and cooling appliances to prevent stranded liquidity.
- **Winter Wedding Season:** +70% demand surge on 3-piece formal suits, luxury dinner sets, and bedsheets.
- **Summer Peak:** +85% demand surge on cooling fans, blenders/juicers, and summer t-shirts.

---

## 📁 Project Deliverables Structure

```text
├── ML_Semester_Project_Guidelines_Fall2026.pdf   # Official Sukkur IBA Project Guidelines
├── ShopMind_AI_Final_Project_Report.pdf           # 4-Page Publication-Grade Project Report (PDF)
├── ShopMind_AI_Project_Proposal.pdf               # Deliverable 1: Approved Project Proposal
├── notebooks/
│   └── ShopMind_AI_Pipeline.ipynb                # Deliverable 2: End-to-End Documented Jupyter Notebook
├── app/
│   └── app.py                                    # Deliverable 2: Interactive Streamlit Web GUI
├── src/
│   ├── data_preprocessing.py                     # Leakage audit, chronological split, lag features
│   ├── train_models.py                           # Benchmark model training suite & serialization
│   └── optimizer.py                              # Scipy MILP Bounded Knapsack with Seasonality
├── models/
│   └── best_sales_model.joblib                   # Serialized Random Forest Model Artifact
├── templates/
│   └── sample_pakistan_procurement.csv           # 17 Retail SKUs localized in Pakistani Rupees (PKR)
└── reports/
    ├── model_comparison_results.csv              # Benchmark evaluation table (MAE, RMSE, R2, MAPE)
    ├── error_analysis_cases.csv                  # Outlier residual audit cases
    └── *.png                                     # High-resolution figures for report and presentation
```

---

## 🚀 How to Run the Project

### 1. Launch the Live Interactive Streamlit GUI (Deliverable 2):
```bash
streamlit run app/app.py
```
Open your browser at **`http://localhost:8501`**.

### 2. Run the Documented Jupyter Notebook (Deliverable 2):
```bash
jupyter notebook notebooks/ShopMind_AI_Pipeline.ipynb
```

### 3. Re-train & Benchmark Models:
```bash
python3 -m src.data_preprocessing
python3 -m src.train_models
```
