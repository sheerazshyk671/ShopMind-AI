# ShopMind AI

Budget-aware demand forecasting and inventory planning using the UCI Online Retail II dataset, with a seller-specific calibration layer and a Streamlit dashboard.

## Run the dashboard

Python 3.9+ is required. The included model was trained with the pinned versions in `requirements.txt`; scikit-learn model files are not reliably portable between versions.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

The dashboard loads the trained model and a demo seller file by default. Upload a CSV or XLSX, enter a budget, and click **Generate purchase plan**. You can download the seller template and the recommendations from the app. If the model is missing, the app explicitly falls back to a recent-sales baseline.

### Seller file and calibration

The demo file `data/sample_seller_products.csv` has `ProductID`, `ProductName`, `CurrentPrice`, `CurrentStock`, and twelve consecutive monthly sales columns `Sales_01` through `Sales_12`, oldest first. You may provide six or more consecutive `Sales_XX` columns (up to 99); every row must have the same months, with no gaps. The most recent column is assumed to represent the current calendar month, and the model forecasts the next calendar month. Supply complete, comparable monthly data; if the current month is incomplete, the forecast may be biased.

The last six months supply actual lag, rolling mean and standard deviation features. Forecast units equal the predicted ratio times the latest three-month average, adjusted by a seller calibration factor. Safety stock is 1.28 times the sample standard deviation of the last six months. An item with zero three-month sales receives a zero forecast, though safety stock may still be positive.

With seven or more months, the app constructs historical one-step predictions using only preceding months. It compares pooled predicted units to actual seller units, caps the resulting ratio to 0.5–2.0, then shrinks it towards 1 by `n / (n + 20)`, where `n` is the number of eligible product-month examples. This is **pooled post-hoc calibration**, not retraining or fine-tuning the model; it is not independently validated on future seller data and can be unstable or biased for small or changing catalogs. Six months alone produces no calibration. The dashboard shows the factor and example count. Legacy files with only `RecentMonthlySales` are supported via a flat-history proxy and zero safety stock.

## Reproduce training

The [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset contains UK transactions from 2009–2011. The raw dataset is not committed. To reproduce using the pinned CSV exports mirrored in [this GitHub repository](https://github.com/nicolexyu/Team_Project_Online_Retail/tree/main/data/raw), run:

```bash
python scripts/download_data.py
python -m src.preprocessing 'data/raw/online_retail_II(Year 2009-2010).csv,data/raw/online_retail_II(Year 2010-2011).csv'
python -m src.forecasting
pytest -q
```

The downloader verifies SHA-256 checksums. Alternatively, download the original `online_retail_II.xlsx` workbook from UCI, place it at `data/raw/online_retail_II.xlsx`, and run `python -m src.preprocessing`; all workbook sheets are read. Check the dataset's source terms before reuse.

Preprocessing removes cancellations, non-product codes, non-positive prices and quantities, and missing descriptions; it fills zero-sales months after a product's first sale and retains products with at least 12 months between first and last sale and six selling months. Features predict each month's sales relative to the preceding three-month mean, using only prior months. A zero rolling mean cannot define a ratio, so such training rows are excluded. The lowest validation MAE is saved to `models/demand_model.pkl`.

### Training results

The two original annual CSV exports contain 1,067,371 rows. Cleaning and aggregation yielded 73,679 product-month rows across 3,126 products. There were 39,033 training and 10,159 validation examples. The time split uses feature-ready months before September 2011 for training and September–December 2011 for validation. Reproducible results with the pinned dependencies:

| Model | Ratio MAE | Ratio RMSE | R² |
|---|---:|---:|---:|
| Linear regression | 2.9866 | 25.7084 | 0.0052 |
| Random forest | 3.6223 | 26.5023 | -0.0572 |
| Gradient boosting | **2.9534** | 25.9730 | -0.0154 |

The included gradient boosting model had the lowest MAE, but the low/negative R² and high RMSE indicate weak performance and large errors on this historical holdout. Ratio targets can be extreme when the historical mean is small. This model and the seller calibration should be independently validated on current seller data before operational use; they are not guarantees of demand or savings.

## Optimization assumptions

- Prices are treated as unit acquisition costs in rupees. Shortages are rounded up to whole units, then allocated greedily within the budget.
- As written in the specification, priority is `shortage * (price * 0.5)`; a fixed positive alpha scales every score equally and does **not** change product ordering. The heuristic is not a globally optimal knapsack solution.
- Only load trusted model files: joblib artifacts can execute code when loaded.
