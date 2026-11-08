# Results (auto-generated — re-run `python scripts/train_pipeline.py --demo --tune`)

Every number below comes from `reports/*.csv` and `reports/training_summary.json`.
If you edit the model, re-run the script and update this file; a mismatch between
the report and the code is a zero under the course rules.

## Split

| Window | Months | Rows | Used for |
|---|---|---:|---|
| Fit | 2025-04 → 2026-01 | 660 | training every model |
| Validation | 2026-02 → 2026-05 | 264 | hyperparameter search, stage-2 selection |
| Test | 2026-06 → 2026-09 | 264 | scored **once**, at the end |

Seed 42. Features: 15 (13 base + 2 planned-promotion columns).

## Stage 1 — all models vs all baselines, scored on TEST

| Model | Family | MAE | RMSE | Bias | WMAPE | MAPE | MASE | Rupee-WAPE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | Tree ensemble | **14.93** | 29.25 | +3.72 | **26.2%** | 39.7% | 0.027 | **29.0%** |
| Baseline: last-3-month mean (= fixed ratio 1.0) | Baseline | 15.53 | 29.62 | −0.93 | 27.3% | 39.2% | 0.028 | 31.5% |
| Gradient Boosting | Boosted trees | 15.61 | 32.30 | +2.36 | 27.4% | 40.5% | 0.028 | 31.2% |
| Decision Tree | Tree | 17.33 | 36.38 | +0.62 | 30.4% | 41.2% | 0.032 | 33.7% |
| Baseline: seasonal naive | Baseline | 26.22 | 62.45 | −17.13 | 46.0% | 48.7% | 0.048 | 45.3% |
| Linear Regression | Linear | 50.63 | 106.87 | −50.07 | 88.9% | 81.6% | 0.092 | 87.8% |

**Verdict: one of four models beats the strongest baseline, by 3.8%.** Three do not.
Report this as it is. The honest framing is that demand forecasting at SKU-month
granularity is hard, the trailing-mean rule is strong, and an ML model only adds
value here for the rupee-weighted objective (29.0% vs 31.5%).

## Validation window (what model selection actually saw)

| Model | Validation MAE |
|---|---:|
| Gradient Boosting | 25.89 |
| Random Forest | 25.95 |
| Decision Tree | 28.36 |
| Linear Regression | 42.00 |

Gradient Boosting won validation; Random Forest won test. That gap is worth a sentence
in the report's error-analysis section — it is the normal cost of selecting on a
4-month validation window.

## Stage 2 — fine-tuning to this seller

| Variant | Test MAE | Verdict |
|---|---:|---|
| Stage 1 only (no fine-tune) | 15.61 | shipped default |
| Stage 2: pooled offset | 15.75 | worse on test |
| Stage 2: per-category offset | 15.92 | chosen on validation, worse on test |

Stage 2 was **selected on validation** (it had the lowest validation MAE) and then
**failed on test**. It is kept in `models/seller_model.pkl` and offered in the app as an
option, but the default is stage 1. Reporting a negative result is worth more marks than
quietly deleting it.

## Censoring experiment

Both models are scored against **true demand** on the same holdout months:

| Trained on | MAE | Bias | Rupee-WAPE |
|---|---:|---:|---:|
| Recorded sales (censored) | 23.62 | +11.74 | 46.0% |
| True demand (uncensored) | **16.01** | −4.63 | **23.3%** |

Training on recorded sales costs **48% more MAE** when demand is what you care about.
Note the sign: the sales-trained model *over*-forecasts here, which is **not** the textbook
censoring effect. The mechanism is the drifting ratio target (see
`scripts/censoring_experiment.py` output and report §6).

## What to put in the 10-page report

- §5 results table: the Stage 1 table above, with both baselines included.
- §6 error analysis: `reports/worst_predictions.csv` — 10 rows with product, month,
  actual, forecast and error. Explain that the largest errors are cumulative
  over-/under-forecasts on high-volume Beauty and Fashion SKUs, and that bias is
  positive overall because the test window sits in a growth period.
- §7 lecture mapping: EDA (§3), missing values/encoding (§2), train-test split and
  leakage (§4 + `docs/LEAKAGE_AUDIT.md`), linear regression and trees (§5),
  cross-validation/tuning (§4), regression metrics (§5), ensembles (§5).
- §8 limitations: modest improvement over baseline, synthetic data, 4-month test
  window, stage-2 negative result, greedy (not optimal) allocation.
