"""Build the three documented notebooks and execute them end-to-end.

The course rubric asks for *documented notebooks with fixed seeds*. Writing notebooks by
hand in the Jupyter UI and hoping the outputs stay in sync with the code is how projects
end up with a notebook that shows yesterday's numbers. So the notebooks are generated
from a single source of truth here, executed against the real repository, and saved with
their outputs baked in.

    python scripts/build_notebooks.py            # build + execute
    python scripts/build_notebooks.py --no-exec  # build only (fast, outputs empty)

Requires `nbformat` and `nbclient` (not needed by the app; development-only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks'
KERNEL = 'python3'


def md(text: str):
    return new_markdown_cell(text.strip('\n'))


def code(text: str):
    return new_code_cell(text.strip('\n'))


HEADER = """# ShopMind-AI — {title}

**Course:** ML Semester Project, Fall 2026
**Seed:** 42 everywhere (`.seed(42)`, `random_state=42`, `GeneratorConfig(seed=42)`).
**Data:** self-generated Pakistani seller panel (`data/demo/`) plus the real *Pakistan's
Largest E-Commerce Dataset* (`data/processed/`). Sources and licences are in
`data/demo/DATA_CARD.md` and `docs/PAKISTAN_DATASET.md`.

This notebook is **executed top-to-bottom** and saved with its outputs, so every number
shown below is reproducible by running `python scripts/build_notebooks.py`.

| Notebook | Question it answers |
|---|---|
| `01_eda.ipynb` | What does the data look like, and what would a naive guess score? |
| `02_modelling.ipynb` | Which model wins, on which window, and by how much? |
| `03_planner.ipynb` | Given a budget and a buy list, what should the seller purchase? |
"""


# ---------------------------------------------------------------------------
# 01 — EDA
# ---------------------------------------------------------------------------
def notebook_eda() -> nbformat.NotebookNode:
    cells = [
        md(HEADER.format(title='01 — Exploratory data analysis')),
        md("""
## 1. Setup and reproducibility

Every stochastic component is pinned to seed 42. `sys.path` is extended so the notebook
imports the same `src/` package that the application uses — no copy-pasted logic.
"""),
        code("""
import sys, random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 40)
print('repo root :', ROOT)
print('seed      :', SEED)
"""),
        md("""
## 2. Load the seller panel

The seller dataset is generated (no real seller would hand over their books), but it is
anchored on researched Pakistani price bands, COD return rates and the actual
Ramadan/Eid/Black-Friday calendar. 12 months of history, 66 SKUs, 6 categories.
"""),
        code("""
panel = pd.read_csv(ROOT / 'data/demo/seller_sales_history.csv', parse_dates=['month'])
products = pd.read_csv(ROOT / 'data/demo/seller_products.csv')

print(f'shape        : {panel.shape[0]:,} rows x {panel.shape[1]} columns')
print(f'months       : {panel.month.min():%Y-%m} -> {panel.month.max():%Y-%m}  ({panel.month.nunique()} months)')
print(f'SKUs         : {panel.product_id.nunique()}')
print(f'categories   : {panel.category.nunique()}')
print(f'total units  : {panel.units_sold.sum():,.0f}')
print(f'revenue      : Rs {panel.revenue.sum():,.0f}')
print(f'net profit   : Rs {panel.net_profit.sum():,.0f}')
print(f'margin       : {100 * panel.net_profit.sum() / panel.revenue.sum():.1f}%')
panel.head()
"""),
        md("""
### 2.1 Column dictionary

`units_sold` is what actually left the warehouse net of returns; `lost_units` is demand
that was never served because the SKU was out of stock. The distinction matters: a model
trained on `units_sold` learns the *censored* series.

### 2.2 Data-quality checks

The rubric asks for missing values and duplicates to be handled explicitly. Both are
checked here rather than assumed clean.
"""),
        code("""
checks = pd.DataFrame({
    'dtype'      : panel.dtypes.astype(str),
    'n_missing'  : panel.isna().sum(),
    'pct_missing': (100 * panel.isna().mean()).round(2),
    'n_unique'   : panel.nunique(),
})
print('duplicated (month, product_id) rows:',
      int(panel.duplicated(['month', 'product_id']).sum()))
checks
"""),
        md("""
## 3. Seasonality — the single most important structure

Ramadan moves ~11 days earlier every Gregorian year, so a `month_of_year` feature cannot
capture it. This is why the feature set includes an explicit `is_ramadan` flag and why the
seasonal-naive baseline is weak on this data.
"""),
        code("""
import matplotlib.pyplot as plt
%matplotlib inline

monthly = panel.groupby('month').agg(units=('units_sold', 'sum'),
                                     revenue=('revenue', 'sum'),
                                     profit=('net_profit', 'sum'))

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(monthly.index, monthly.units, '-o', ms=4, color='#2b6cb0')
ax.set_title('fig 1 — monthly units sold, 12 months of seller history')
ax.set_ylabel('units')
ax.grid(alpha=.25)
fig.autofmt_xdate(); plt.show()

monthly.assign(units_index=(monthly.units / monthly.units.median()).round(2))
"""),
        md("""
**Takeaway.** Peaks sit on Ramadan/Eid and Black Friday, not on a fixed calendar month.
A model without an event flag cannot learn this from 12 months of data.

## 4. Concentration — why MAE alone is not enough

If a small number of SKUs carry most of the revenue, then a unit-level MAE of 15 on a SKU
that sells 5 units/month is not comparable to an MAE of 15 on a SKU that sells 300.
"""),
        code("""
sku = (panel.groupby('product_id')
            .agg(revenue=('revenue', 'sum'), units=('units_sold', 'sum'))
            .sort_values('revenue', ascending=False))
sku['rev_share'] = sku.revenue / sku.revenue.sum()
sku['cum_share'] = sku.rev_share.cumsum()
top10pct = int(np.ceil(0.10 * len(sku)))
print(f'top 10% of SKUs ({top10pct} of {len(sku)}) carry '
      f'{100 * sku.cum_share.iloc[top10pct - 1]:.0f}% of revenue')

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(range(1, len(sku) + 1), 100 * sku.cum_share.values, color='#2f855a')
ax.axhline(80, ls='--', lw=1, color='#c53030')
ax.set_xlabel('SKUs, ranked by revenue')
ax.set_ylabel('cumulative % of revenue')
ax.set_title('fig 2 — revenue concentration')
ax.grid(alpha=.25); plt.show()
"""),
        md("""
**Takeaway.** This is why `reports/RESULTS.md` reports a rupee-weighted WAPE alongside
MAE: a unit error on a best-seller costs real money and a unit error on a slow mover does
not.

## 5. Margin by category — the planner's ranking signal

Ranking purchases by revenue or by unit shortage misallocates cash when margins differ
by a factor of seven.
"""),
        code("""
cat = (panel.groupby('category')
            .agg(revenue=('revenue', 'sum'), profit=('net_profit', 'sum'),
                 units=('units_sold', 'sum')))
cat['margin_pct'] = (100 * cat.profit / cat.revenue).round(1)
cat['aov'] = (cat.revenue / cat.units).round(0)
cat.sort_values('margin_pct', ascending=False)
"""),
        md("""
**Takeaway.** Electronics turns over the most revenue per unit but keeps almost none of
it; Fashion keeps a fifth. The planner therefore ranks by **profit per rupee of purchase
cost**, not by demand or by revenue.

## 6. Returns — measured, not assumed

`returned_units` is part of the generator, but the *real* Pakistani dataset lets us check
whether the assumed rates were plausible.
"""),
        code("""
cal = pd.read_csv(ROOT / 'reports/return_rate_calibration.csv')
print('assumed vs measured return rate, by category')
cal
"""),
        md("""
**Takeaway.** The real marketplace returns far more than the generator assumes (48%
overall vs 10–21% assumed). The gap is deliberate and disclosed: a seller should plan on a
normal month, not on this particular bad export. It is the strongest argument for why the
purchase plan must be stress-tested against the return rate rather than a point forecast.

## 7. What a naive guess scores

The rubric requires "something trivial to beat". That something is the trailing 3-month
mean. Its score is computed in notebook 02, on the same rows as the models.
"""),
        code("""
from src.modeling import baseline_forecasts

# `baseline_forecasts` needs the *feature* frame (it reads `month_date` and
# `rolling_mean_3`), so the pipeline's loader and make_features are both applied.
from scripts.train_pipeline import load_panel
from src.features import make_features

raw = load_panel(ROOT / 'data/demo/seller_sales_history.csv')
featured = make_features(raw)

cut = pd.Timestamp('2026-06-01')
test_rows = featured[featured.month_date >= cut]
bl = baseline_forecasts(raw[raw.month < cut], test_rows, test_rows)
print(f'baseline rows scored: {len(bl)}')
print(bl[['last_3_month_mean', 'seasonal_naive']].describe().round(2).to_string())
"""),
        md("""
## 8. Takeaways carried into modelling

1. 12 months is short; the trailing mean is a genuinely strong competitor.
2. Ramadan/Eid/Black-Friday events dominate the calendar — needs explicit features.
3. Revenue is concentrated, so report rupee-weighted error.
4. Margins vary 7× by category, so the planner must rank by profit per rupee.
5. Return rates are the largest single source of margin destruction.
"""),
    ]
    return new_notebook(cells=cells, metadata={'kernelspec': {'name': KERNEL, 'display_name': 'Python 3', 'language': 'python'}})


# ---------------------------------------------------------------------------
# 02 — Modelling
# ---------------------------------------------------------------------------
def notebook_modelling() -> nbformat.NotebookNode:
    cells = [
        md(HEADER.format(title='02 — Modelling, selection protocol and error analysis')),
        md("""
## 1. Setup

Same seed, same imports as the application. The notebook deliberately *imports* the
pipeline rather than re-implementing it, so the notebook cannot drift from the code.
"""),
        code("""
import sys, random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

SEED = 42
random.seed(SEED); np.random.seed(SEED)
pd.set_option('display.width', 140)
print('seed', SEED)
"""),
        md("""
## 2. The split, fixed before any modelling

Chronological, three-way. Validation is what model selection is allowed to see; test is
touched once.

```
|<------------ fit ------------>|<-- validation -->|<-- test -->|
2024-10 .. 2026-01              2026-02 .. 2026-05  2026-06 .. 2026-09
```
"""),
        code("""
from scripts.train_pipeline import load_panel, three_way_split, FEATURE_COLUMNS, OPTIONAL_FEATURES
from src.features import make_features

panel = load_panel(ROOT / 'data/demo/seller_sales_history.csv')
data = make_features(panel)
fit, val, test, split = three_way_split(data)

print(f'feature-ready rows : {len(data):,}')
print(f'features           : {len(FEATURE_COLUMNS + OPTIONAL_FEATURES)}')
print(f'fit                : {len(fit):,} rows, up to {split["fit_end"]}')
print(f'validation         : {len(val):,} rows, {split["validation"][0]} .. {split["validation"][1]}')
print(f'test               : {len(test):,} rows, from {split["test_start"]}')
print()
print('feature columns:', FEATURE_COLUMNS + OPTIONAL_FEATURES)
"""),
        md("""
## 3. Leakage check — the rubric's explicit requirement

Every feature must be computable **before** the month it predicts. The two promotion
features are the only borderline case: promotions are planned before the month starts, so
`planned_promo_days` is legitimate. The realised `promo_days` would not be, and is not used.

The full argument is in `docs/LEAKAGE_AUDIT.md`; this cell asserts the mechanical part.
"""),
        code("""
# A feature that was computed with the current month's own sales would leak the target.
# `target_ratio` and `monthly_units_sold` are the only columns allowed to know the present.
suspicious = [c for c in FEATURE_COLUMNS + OPTIONAL_FEATURES
              if 'sold' in c or c in ('target_ratio', 'net_profit', 'revenue')]
print('features that reference the target month directly:', suspicious or 'none')

# Lag features must be strictly backwards-looking.
lag_cols = [c for c in FEATURE_COLUMNS if c.startswith('lag_')]
print('lag features:', lag_cols)
assert not any(c.startswith('lead') for c in data.columns), 'forward-looking column found'
print('\\nOK: no forward-looking column exists in the feature frame.')
"""),
        md("""
## 4. Stage 1 — train four models from three families

The rubric wants ≥3 models from ≥2 families including at least one classical course model.
The zoo is: Linear Regression (classical), Decision Tree, Random Forest, Gradient Boosting.
All preprocessing is fitted **inside** the training fold only, via a scikit-learn
`Pipeline` — the scaler never sees validation or test statistics.

### 4.1 Selection is on validation
"""),
        code("""
from src.modeling import MODEL_ZOO, BaseModel, unit_metrics
from scripts.train_pipeline import tune_on_validation

# The hyperparameter grid is searched on the VALIDATION window, exactly as the shipped
# pipeline does with `--tune`. Skipping this step (i.e. using library defaults) changes
# which model wins, so it must not be skipped here — the notebook has to reproduce the
# protocol that produced `models/base_model.pkl`.
results, tuned_params = [], []
for name, factory in MODEL_ZOO.items():
    model, params, val_mae = tune_on_validation(name, fit, val, seed=SEED)
    results.append({'model': name, 'validation_MAE': round(val_mae, 3),
                    'tuned_params': ', '.join(f'{k}={v}' for k, v in params.items()) or 'default'})
    tuned_params.append((name, params))

val_table = pd.DataFrame(results).sort_values('validation_MAE').reset_index(drop=True)
print('chosen on VALIDATION ->', val_table.iloc[0]['model'])
val_table
"""),
        md("""
### 4.2 Score that one model on test — once

This is the disciplined version of the protocol. The model is already fixed by the table
above; test is now read exactly once.
"""),
        code("""
winner = val_table.iloc[0]['model']
winner_params = dict(tuned_params)[winner]
estimator = MODEL_ZOO[winner](SEED)
if winner_params:
    estimator.set_params(**winner_params)
chosen = BaseModel.fit(estimator, fit, fit['target_ratio'], name=winner)

from scripts.train_pipeline import naive_scale
scale = naive_scale(load_panel(ROOT / 'data/demo/seller_sales_history.csv'),
                    split['test_start'])
pred = chosen.forecast_units(test)
m = unit_metrics(test['monthly_units_sold'].values, pred,
                 test['avg_unit_price'].values if 'avg_unit_price' in test else None,
                 scale)
print(f'{winner} on the untouched test window')
for k, v in m.items():
    print(f'  {k:<10} {v:,.3f}')
"""),
        md("""
## 5. All models and all baselines, side by side

The scripted version of this table is `reports/stage1_results.csv`. Reproduced here from
the committed CSV so the notebook cannot silently disagree with the report.
"""),
        code("""
stage1 = pd.read_csv(ROOT / 'reports/stage1_results.csv')
stage1[['model', 'family', 'MAE', 'RMSE', 'bias', 'WMAPE', 'MASE', 'rupee_WAPE']].round(3)
"""),
        code("""
best_model = stage1[stage1.family != 'Baseline'].MAE.min()
best_base  = stage1[stage1.family == 'Baseline'].MAE.min()
print(f'best model MAE    : {best_model:.2f}')
print(f'best baseline MAE : {best_base:.2f}')
print(f'improvement       : {100 * (best_base - best_model) / best_base:+.2f}%')

reported = val_table.iloc[0]['model']
rep_mae = float(stage1.set_index("model").loc[reported, "MAE"])
print(f'reported model    : {reported} (won validation), MAE {rep_mae:.2f} -> '
      f'{100 * (best_base - rep_mae) / best_base:+.2f}% vs baseline')
"""),
        md("""
**Read this carefully.** The model with the lowest *test* MAE is not the model that won
*validation*. Claiming the test-winner would mean selecting on test — the exact leakage the
rubric penalises. So this project reports the **validation winner**, and it
**does not beat the baseline**. That is the honest headline, and it is written up as a
negative result in `reports/RESULTS.md`.

The difference between the two claims is recorded as `selection_variance_pct` in
`reports/training_summary.json`.

**A trap worth recording.** Running this notebook with library-default hyperparameters
instead of the tuned ones selects **Random Forest** on validation, not Gradient Boosting
(they are separated by 0.05 MAE). In other words the winner of a 4-month validation window
is not even stable to a hyperparameter search. That is stronger evidence for the negative
result than the test score alone: the model choice is not identifiable from this window,
so no claim that "our model beats the baseline" can be robust.

## 6. Error analysis — beyond a single accuracy number

The rubric asks for residuals, not just MAE. Four figures are produced by
`scripts/error_analysis.py`; the underlying numbers:
"""),
        code("""
err = pd.read_csv(ROOT / 'reports/error_analysis.csv')
err
"""),
        md("""
### 6.1 Residuals over time

A model can have a low MAE and still be systematically wrong at the end of the window.
"""),
        code("""
from IPython.display import Image, display
for name in ['fig5_predicted_vs_actual', 'fig6_residual_over_time',
             'fig7_residual_distribution', 'fig8_error_by_volume']:
    p = ROOT / 'reports/figures' / f'{name}.png'
    if p.exists():
        display(Image(filename=str(p), width=760))
"""),
        md("""
**Takeaways from the figures.**

* **fig 6** is the most informative one: the *baseline* drifts from +7 units at the start
  of the test window to −9 at the end, because it is anchored on a past level while demand
  grows. The ML models stay flatter. So the model is better on **bias** while being worse
  on **MAE** — a trade-off a single number hides.
* **fig 7**: residuals are right-skewed and heavy-tailed (skew ≈ 1.0, kurtosis ≈ 9.5), so
  RMSE is dominated by a few SKU-months and normal-theory intervals would be wrong.
* **fig 8**: error rises monotonically with volume, which is why the rupee-weighted metric
  is the one that matters commercially.

## 7. Stage 2 — calibration to the seller

Stage 2 applies a multiplicative offset estimated on the seller's own history, pooled by
category. It is chosen on validation and then measured on test.
"""),
        code("""
stage2 = pd.read_csv(ROOT / 'reports/stage2_results.csv')
stage2.round(3)
"""),
        md("""
**Stage 2 failed on test.** It is kept in `models/seller_model.pkl` and offered in the app
as an explicit option, but the default is stage 1. Reporting the negative result is worth
more than quietly deleting the stage.

## 8. Summary of what the model does and does not achieve

| Claim | Supported? |
|---|---|
| Beats a trivial baseline | **No** — loses to the trailing mean by 0.52% |
| Beats seasonal naive | Yes, by 40% |
| Generalises from validation to test | No — the validation winner is not the test winner |
| Errors are well-behaved | No — right-skewed, heavy-tailed, volume-dependent |
| Profit-aware evaluation | Yes — rupee-WAPE reported alongside MAE |
"""),
    ]
    return new_notebook(cells=cells, metadata={'kernelspec': {'name': KERNEL, 'display_name': 'Python 3', 'language': 'python'}})


# ---------------------------------------------------------------------------
# 03 — Planner
# ---------------------------------------------------------------------------
def notebook_planner() -> nbformat.NotebookNode:
    cells = [
        md(HEADER.format(title='03 — Stage 3, the purchasing plan')),
        md("""
## 1. The decision problem

The seller supplies a CSV with the products they intend to buy, each with a **current
purchase price** and the **units already available**, plus a **budget**. The system must
return a purchase plan.

The naive answer — buy the products with the highest demand — is wrong, because demand is
not profit. This notebook shows the ranking, the allocation, and what happens when the
assumptions are violated.

## 2. Setup
"""),
        code("""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
np.random.seed(42)
pd.set_option('display.width', 150)

from src.planning import expected_profit_per_unit, forecast_request, summarise
from src.modeling import BaseModel

model = BaseModel.load(str(ROOT / 'models/base_model.pkl'))
print('loaded model:', model.name)
"""),
        md("""
## 3. Unit economics — where profit actually comes from

A unit is profitable only if

```
retail price
  - purchase cost
  - marketplace commission        (per category)
  - shipping + COD fee
  - expected return cost          (return rate x full cost of a bounced parcel)
  - allocated ad spend
> 0
```

Returns are the term that surprises people: at a 21% return rate on fashion, a 2x markup
can still be loss-making.

### 3.1 The trap: a 5,000-cost item at a 5,100 retail price
"""),
        code("""
rows = []
for retail in [5100, 6000, 7000, 8000, 9500]:
    rows.append({'retail': retail, 'cost': 5000,
                 'profit_per_unit': round(expected_profit_per_unit(retail, 5000, 'Home & Kitchen'), 2)})
pd.DataFrame(rows)
"""),
        md("""
**Takeaway.** At a 5,100 retail on a 5,000 landed cost the item loses **Rs 873 per unit**.
The seller is "making a sale" and destroying money. This is why the planner ranks by
`expected_profit_per_unit / CurrentPrice` and not by units or by revenue.

## 4. Load the seller's buy list
"""),
        code("""
request = pd.read_csv(ROOT / 'data/demo/seller_purchase_request.csv')
history = pd.read_csv(ROOT / 'data/demo/seller_sales_history.csv', parse_dates=['month'])

print(f'{len(request)} products to consider, budget will be applied below')
request.head(8)
"""),
        md("""
## 5. Produce a plan for a Rs 500,000 budget
"""),
        code("""
BUDGET = 500_000.0

result = forecast_request(request, model, BUDGET, history=history,
                          forecast_month='2026-10')
plan = result.plan
stats = summarise(plan, BUDGET)

for k, v in stats.items():
    print(f'{k:<28} {v}')
print()
print('assumptions the system is making:')
for a in result.assumptions:
    print(' -', a)
"""),
        code("""
cols = ['ProductID', 'ProductName', 'Category', 'CurrentPrice', 'CurrentStock',
        'AvgDemand', 'SafetyStock', 'RequiredInventory', 'ShortageUnits',
        'ProfitPerUnit', 'ProfitPerRupee', 'RecommendedPurchase', 'Investment',
        'ExpectedProfit', 'IsUnprofitable']
shown = [c for c in cols if c in plan.columns]
plan[shown].sort_values('Investment', ascending=False).head(12).round(2)
"""),
        md("""
## 6. Budget sensitivity — the plan is a curve, not a single answer

A useful decision system tells the seller what the *next* rupee buys.
"""),
        code("""
curve = []
for budget in [100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000]:
    r = forecast_request(request, model, budget, history=history, forecast_month='2026-10')
    s = summarise(r.plan, budget)
    curve.append({'budget': budget,
                  'investment': s['investment'],
                  'unallocated': s['unallocated'],
                  'expected_profit': s['expected_profit'],
                  'expected_roi_pct': s['expected_roi_pct'],
                  'funded_products': s['funded_products'],
                  'unprofitable_products': s['unprofitable_products']})
pd.DataFrame(curve).round(1)
"""),
        md("""
**Takeaway.** Return on budget falls as the budget grows, because the best
profit-per-rupee items are funded first and the marginal product is worse. That shape is
the practical answer to "how much stock should I buy?" — the seller should stop adding
budget where the marginal ROI drops below their cost of capital.

## 7. What would make the plan wrong

The rubric rewards an explicit statement of limitations. The plan is conditional on:

1. **The demand forecast**, which notebook 02 shows does not beat a trailing mean.
2. **The return rate assumption**, which the real Pakistani data shows could be 2–4× higher.
3. **Prices holding** for the purchase lead time.
4. **Greedy allocation** — it is not an integer program, so it is near-optimal, not optimal.

A stress test on (2) below, since it is the largest risk.
"""),
        code("""
# `forecast_request` accepts an explicit return-rate override, so the stress test does
# not have to mutate module state. 10% is the planning default for fashion-adjacent
# categories; 48% is what the real Pakistani dataset actually shows.
rows = []
for rate in [0.10, 0.21, 0.30, 0.48, 0.60]:
    r = forecast_request(request, model, BUDGET, history=history,
                         forecast_month='2026-10', return_rate=rate)
    s = summarise(r.plan, BUDGET)
    rows.append({'assumed_return_rate': f'{rate:.0%}',
                 'median_profit_per_unit': round(float(r.plan.ProfitPerUnit.median()), 0),
                 'investment': round(s['investment'], 0),
                 'expected_profit': round(s['expected_profit'], 0),
                 'expected_roi_pct': s['expected_roi_pct'],
                 'funded_products': s['funded_products'],
                 'unprofitable_products': s['unprofitable_products']})
pd.DataFrame(rows)
"""),
        md("""
**Takeaway.** The plan's profitability is more sensitive to the return assumption than to
anything the model does. At the planning default the same Rs 500,000 returns a healthy
profit; at the return rate the real marketplace actually exhibits, the funded set shrinks
and the unprofitable count grows. A seller who plans on the optimistic number will buy
stock that cannot be sold profitably.

## 8. Conclusion

The planner is the part of the system that survives the negative forecasting result. Even
with a mediocre demand forecast, ranking by profit-per-rupee and respecting a hard budget
produces a defensible purchase decision — and it correctly refuses to fund items whose
retail price does not clear their fully-loaded cost.
"""),
    ]
    return new_notebook(cells=cells, metadata={'kernelspec': {'name': KERNEL, 'display_name': 'Python 3', 'language': 'python'}})


def build(execute: bool = True) -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for name, factory in (('01_eda', notebook_eda),
                          ('02_modelling', notebook_modelling),
                          ('03_planner', notebook_planner)):
        nb = factory()
        path = NOTEBOOKS / f'{name}.ipynb'
        if execute:
            print(f'executing {path.name} ...')
            client = NotebookClient(nb, timeout=600, kernel_name=KERNEL,
                                    resources={'metadata': {'path': str(NOTEBOOKS)}})
            client.execute()
        nbformat.write(nb, path)
        print(f'wrote {path} ({len(nb.cells)} cells)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-exec', action='store_true', help='build without running cells')
    args = parser.parse_args()
    build(execute=not args.no_exec)
