# Leakage Audit

**Requirement (course rubric):** "A leakage audit listing any feature that would not exist at prediction time, and confirming its removal."

This document is that audit, for every stage of the ShopMind AI pipeline. It is written as evidence, not as an assertion: each row states *when* the feature becomes available, the *verdict*, and — where the risk is real — the *test or code* that enforces it.

**Prediction point.** The model runs at the end of month *t*, to place a purchase order for month *t+1*. Anything known at the end of month *t* is legitimate. Anything from month *t+1* or later is leakage.

---

## 1. Feature-by-feature audit

All 15 features used by the model. "Available?" is relative to the end of month *t*.

| # | Feature | What it is | Available at prediction time? | Verdict | Evidence |
|---|---|---|---|---|---|
| 1 | `rolling_mean_3` | Mean units over months t−2…t | **Yes** | Kept | Built with `.shift(1).rolling(3)`; never includes month t+1 |
| 2 | `rolling_mean_6` | Mean units over months t−5…t | **Yes** | Kept | Same construction |
| 3 | `rolling_std_6` | Std. dev. over the same window | **Yes** | Kept | Drives safety stock |
| 4 | `lag_1` | Units in month t | **Yes** | Kept | Direct lag |
| 5 | `lag_3` | Units in month t−2 | **Yes** | Kept | Direct lag |
| 6 | `lag_6` | Units in month t−5 | **Yes** | Kept | Direct lag |
| 7 | `lag_1_ratio` | `lag_1 / rolling_mean_6` | **Yes** | Kept | Ratio of two available features |
| 8 | `trend` | `rolling_mean_3 / rolling_mean_6` | **Yes** | Kept | Ratio of two available features |
| 9 | `volatility` | `rolling_std_6 / rolling_mean_6` | **Yes** | Kept | Ratio of two available features |
| 10 | `months_since_start` | Product age in months | **Yes** | Kept | Counter, known in advance |
| 11 | `month` | Calendar month number of the target month | **Yes** | Kept | Calendar is known |
| 12 | `month_sin`, `month_cos` | Cyclical encoding of #11 | **Yes** | Kept | Deterministic from #11 |
| 13 | `planned_promo_days` | Promotion days the seller has **scheduled** for month t+1 | **Yes** | Kept | Promotions are planned ahead; this is the *plan*, not the outcome |
| 14 | `planned_discount_pct` | Planned discount for month t+1 | **Yes** | Kept | Same |

**Test that enforces the guarantee:** `tests/test_modeling.py::test_features_at_month_t_never_use_month_t_or_later` recomputes features on a panel truncated at month *t* and asserts every feature for month *t* is byte-identical to the full-panel value. If any feature peeked forward, truncation would change it.

---

## 2. Columns that were removed, and why

These exist in the raw data but must never reach the model.

| Column | Why it is leakage or useless | Action |
|---|---|---|
| `units_sold` of the target month | This **is** the target. | Used only to build `target_ratio`. |
| `demand` = `units_sold + lost_units` | `lost_units` is the *counterfactual* demand that could not be served. Known only after the fact. | **Excluded from features.** Used only in the separate censoring experiment, and disclosed there. |
| `lost_units` | Same as above. | Excluded. |
| `gross_units` | Units shipped before returns — includes the returns that are not yet known. | Excluded. |
| `returned_units` | Final returns take days/weeks to settle after the month ends. | Excluded. |
| `revenue`, `cogs`, `commission`, `shipping_cost`, `return_cost`, `ad_spend`, `net_profit`, `margin_pct` | All are downstream *outcomes* of the target month. | Excluded (used only to verify the simulator's own P&L in `tests/test_demo_data.py`). |
| `stock_on_hand`, `stock_out_days`, `purchased_units` | End-of-month states. `stock_out_days` in particular is a *consequence* of the sales outcome. | Excluded from features. |
| `avg_unit_cost` | The realised average cost of units **sold** that month. | Excluded; the seller's purchase price comes from the request file instead. |
| `product_id` / `StockCode` | An identifier; models that see it can memorise per-product levels instead of generalising. | Used as a **grouping key only** (joins, per-product ordering), never as a feature. |
| `product_name` | Free text identifier. | Display only. |
| `category` | Legitimately known — but handled carefully: it enters through the **stage-2 category offset**, which is fitted on training months only. | Not a model input; used only for the pooled adjustment. |
| `month` as a *timestamp* | A raw timestamp lets a tree split on absolute time and effectively extrapolate the split. | Converted to `month` (int 1–12) plus sin/cos, keeping seasonality without the time index. |

---

## 3. Split-level leakage

The most common way a forecasting project leaks is not a bad column, it is a bad split.

| Risk | Status | Evidence |
|---|---|---|
| Random row split on time series | **Avoided.** Three-way chronological split: fit → validation → test. | `scripts/train_pipeline.py::three_way_split` |
| Tuning on the test window | **Avoided.** Grid search scores each candidate on the *validation* window only; the test window is scored once, at the end. | `tune_on_validation()`; `reports/stage1_validation.csv` vs `reports/stage1_results.csv` |
| Stage-2 calibration fitted on future months | **Avoided.** The stage-2 model is fitted on months **before** the test window (`panel[month < test_start]`). | `evaluate_stage2()` |
| Scaler/imputer fitted on all data | **Not applicable, and stated explicitly:** the pipeline uses tree/linear models with no fitted scaler or imputer, so there is nothing to fit outside the training fold. Adding one would require wrapping it in a `Pipeline`. | `src/modeling.py` |
| MASE scale computed on test data | **Avoided.** The MASE denominator is the mean absolute monthly change on *training* months only. | `naive_scale()` |
| Baselines evaluated on different rows | **Avoided.** Baselines are scored on exactly the test rows, from the same `features` frame. | `tests/test_modeling.py::test_baselines_are_evaluated_on_the_same_rows_as_the_models` |
| A product appearing in both fit and test | **Accepted, and disclosed.** This is *within-product forecasting*, not new-product forecasting: the same SKU is forecast into future months. That is the correct setup for "what do I buy next month", but it must be described honestly rather than passed off as cold-start generalisation. | Report §8 |

---

## 4. Two leakage traps specific to this project

**(a) The strongest baseline is disguised leakage.** The model predicts a **ratio** to the trailing three-month mean. If the model learns nothing, it outputs a ratio of ≈1.0, which *is* the baseline. Reporting an improvement over the "mean forecast" baseline while hiding the "same ratio, hard-coded to 1" baseline would be misleading. The baseline is therefore reported explicitly in every results table as `Baseline: last-3-month mean (= fixed ratio 1.0)`, and on the current test window it beats three of the four models.

**(b) `lost_units` is a counterfactual.** It is the demand that could not be met. It is genuinely useful for a *censoring experiment* (`scripts/censoring_experiment.py`), but including it as a feature would mean the model is told next month's unmet demand in advance. It is excluded, and the experiment that uses it is labelled as using information a real seller does not have.

---

## 5. What this pipeline deliberately does **not** claim

- It does not claim to predict a **new** product with no history. Products without enough months fall back to the seller's stated quantity and the app says so.
- It does not claim the demo data is real. See `data/demo/DATA_CARD.md`.
- It does not claim the forecast is accurate in absolute terms: the test WMAPE is ~26% and the improvement over the strongest baseline is a few percent, not a transformation. Both numbers are in `reports/training_summary.json`.
