# Real base data: Pakistan's Largest E-Commerce Dataset

This project can now train on **real** Pakistani marketplace data instead of (or before) the
synthetic seller data. This document records exactly what that dataset is, what survived
cleaning, and — most importantly — **what it cannot do**, because that finding is itself a
result worth reporting.

Source: *Pakistan's Largest E-Commerce Dataset — Half a Million Online Orders* (Zeeshan-ul-hassan
Usmani), Kaggle `zusmani/pakistans-largest-ecommerce-dataset`. Licence: data files © Original
Authors; educational use, cite the dataset.

---

## 1. Provenance and integrity of the copy used

| Item | Value |
|---|---|
| File | `data/raw/pakistan_ecommerce_raw.csv` (not committed; 106.5 MB) |
| SHA-256 | `aa1344e8dde4f2ea9038eaafeaf908acbb54e6eee7783837d567be093bdd528e` |
| Zip size | 14,432,262 bytes — matches the size Kaggle's API metadata reports |
| Date range | 2016-07-01 → 2018-08-28 (26 months) |
| Lines in file | 1,048,575 |

**Two things you must state honestly in the report:**

1. **The file is truncated.** 1,048,575 lines is exactly the Excel worksheet limit
   (1,048,576 minus header). The Kaggle page advertises ~1M orders; this CSV export holds
   **584,524 non-empty order lines**. Do not describe it as "half a million orders" as if it
   were complete. It is a 584k-line slice.
2. **There are reseller bulk rows.** November months contain single orders of up to 1,000 units;
   19 rows have quantity ≥ 100, contributing 1.1% of units. Median order value stays around
   Rs 5,000–7,600, so these are outliers, not the norm — but they inflate totals.

## 2. Cleaning (all of it reproducible)

```bash
python -m src.pakistan_data --input data/raw/pakistan_ecommerce_raw.csv
```

| Step | Rows affected |
|---|---|
| Raw lines read | 1,048,575 |
| Blank rows / 5 empty `Unnamed` columns dropped | → **584,524** rows, 21 columns |
| Unparseable dates | 0 |
| **Returns** (`canceled`, `order_refunded`, `refund`, `fraud`, `exchange`) | **268,803 rows — kept and used as the return leg** |
| **Unresolved** (`pending`, `processing`, `holded`, `payment_review`, `\N`) | 180 rows dropped |
| **Kept as real sales** (`complete`, `received`, `cod`, `paid`, `closed`) | **315,450 rows** |
| Bad quantity or negative value | 37 dropped |
| Zero-price lines (kept, flagged) | 842 |
| Trimmed months (partial data) | 2016-07, 2018-08 |

After cleaning: **408,753 orders, 757,679 units, Rs 4.99 billion revenue** across 26 months.

### The headline real-world finding: a 48% return rate

```
returned units 268,803 / (sold 757,679 + returned 268,803) = 48.0%
```

Per category, the real measured rates (not assumed):

| Category | Return rate | Category | Return rate |
|---|---:|---|---:|
| Entertainment | 64.5% | Women's Fashion | 42.6% |
| Mobiles & Tablets | 59.8% | Beauty & Grooming | 33.8% |
| Computing | 56.2% | Soghaat | 32.8% |
| Appliances | 54.0% | Health & Sports | 32.7% |
| Men's Fashion | 44.6% | Books | 27.9% |

This is **higher than the 25–35% published Pakistani COD band**, and it is a genuine property
of this export rather than a pipeline bug: a `canceled` line in a COD marketplace usually means
the parcel came back. Two consequences for the report:

1. **It confirms the ordering the generator assumed** — category-level return rates really do
   differ by a factor of two (64.5% entertainment vs 27.9% books), so a single business-wide
   return rate would be wrong.
2. **It shows the generator's assumptions are too optimistic.** Side by side:

| Category | Generator assumption | Measured on real data | Gap |
|---|---:|---:|---:|
| Fashion | 21.0% | 42.4% | **+21.4 pts** |
| Electronics | 10.0% | 54.4% | **+44.4 pts** |
| Home & Kitchen | 12.0% | 39.3% | +27.3 pts |
| Beauty & Personal Care | 15.0% | 34.6% | +19.6 pts |
| Baby & Kids | 11.0% | 39.0% | +28.0 pts |
| Sports & Fitness | 13.0% | 32.5% | +19.5 pts |

(`reports/return_rate_calibration.csv`.) The synthetic seller is therefore modelled on a
**better-than-real** marketplace: its margins are optimistic by roughly 20 percentage points
of return cost. The assumption is kept deliberately — a seller should plan on a normal month
rather than on this particular bad year — but it is now **measured and disclosed rather
than guessed**, and the app prints it as an assumption.

3. **It explains why the profit estimate is brutal.** At a 48% return rate, a 1.9× markup
   leaves almost nothing. Any buying plan that ignores returns destroys margin — which is
   exactly the argument for the profit-per-rupee ranking in `src/planning.py`.

## 3. What the data looks like

- **16 product categories**, e.g. Mobiles & Tablets (115,710 rows), Men's Fashion (92,221),
  Women's Fashion (59,721), Appliances (52,413), Superstore (43,613).
- **64,132 distinct SKUs** across 584k lines.
- **Monthly volume is violently uneven:**

| Month | Units | Note |
|---|---:|---|
| 2016-11 | 42,327 | Black Friday |
| 2017-09 | 4,375 | an unexplained collapse — treat as a data-quality event |
| 2017-11 | 50,527 | Black Friday, **4.3× the median month** |
| 2018-02 | 18,852 | 2.3× median |
| 2018-03 | 34,261 | 2.3× median |

**The single most important fact about this dataset for forecasting: one month in six is
dominated by a Black Friday event of 4–17× normal depending on category** (Entertainment 17×,
Appliances 16.9×, Mobiles 7.4×, Books 1.8×, Soghaat 0.7×). Monthly revenue CV is **113.5%**.

## 4. Why SKU-level forecasting does not work here — and this is a real finding

The obvious plan is "train per-SKU demand models on the real data". The data refuses:

| Measure | Value |
|---|---:|
| Median months a SKU is present | **1** |
| Mean months per SKU | 1.64 |
| SKUs present in only one month | 43,149 of 64,132 (**67%**) |
| SKUs with ≥ 6 months of history | **1,544** (2.4% of SKUs) |
| Share of units covered by those 1,544 SKUs | **22.5%** |
| Share of revenue covered by them | **13.1%** |

This is a marketplace of one-off catalog listings, not a stable assortment. There is no
per-product time series to learn from — so a "SKU-level demand model on 500k real orders" is
not achievable with this data, however appealing the phrase sounds.

**Correct framing for the report:** the usable unit of analysis is **category × month**
(16 categories × 24 clean months = 377 observations). SKU-level analysis is a secondary,
explicitly-partial experiment on the 1,544 persistent SKUs covering 22% of volume.

## 5. Forecasting result on the real data (the important one)

Category × month panel, chronological split, last 4 months held out (2018-04 → 2018-07):

| Model | MAE (units) | WMAPE | Bias |
|---|---:|---:|---:|
| **Baseline: seasonal naive (same month last year)** | **680.0** | **99.1%** | +460 |
| **Baseline: ratio 1.0 × trailing 3-month mean** | 767.9 | 111.9% | +532 |
| Linear Regression | 1,236.5 | 180.1% | +1,211 |
| Gradient Boosting | 1,430.1 | 208.3% | +1,248 |
| Decision Tree | 1,544.1 | 224.9% | +1,332 |
| Random Forest | 2,330.9 | 339.5% | +2,208 |

Note which baseline wins. On the synthetic data the trailing mean was strongest; on **real**
data the seasonal naive wins, because the dominant signal is a **once-a-year November event**,
which "same month last year" captures and a 3-month average does not. That single row is the
most interesting result in the project.

**Every model loses to the trivial baseline, and all of them over-forecast badly.**

Why: the model is trained on a window (2017-02 → 2018-03) that contains two Novembers, the
Feb–Mar 2018 surge, and lumpy category-level events. It then applies a learned ratio ≈ 1.5–2×
to a trailing mean in a test window that has no such event. The result is a systematic
over-forecast of roughly 1,200–2,200 units per category-month.

**This is a better result for the assignment than a fake win.** It is a clean demonstration
of: an event-driven series breaking a level-and-trend model, why the naive baseline must be
reported, and why 16 series / 24 months is too little data for a tree ensemble to generalise.

## 6. What this means for the two-stage architecture

| Stage | Real data available? | Verdict |
|---|---|---|
| **Stage 1 — market model** | Yes: 16 categories × 24 months of real orders | **Feasible, but weak.** Use it for the honest negative result in §5 |
| **Stage 2 — seller calibration** | **No.** The dataset has no seller identifier | The synthetic seller panel remains the only place stage 2 can be demonstrated |
| Buying plan / GUI demo | No prices-with-stock or budget in this dataset | Synthetic seller request remains necessary |

So the honest architecture is:

```
Stage 1  trained on  REAL Pakistani marketplace data  (categories, 24 months)   → weak, documented
Stage 2  demonstrated on GENERATED seller history      (products, 24 months)    → the app's demo
```

Both are already wired: `src/pakistan_data.py` produces the real panel, `src/demo_data.py`
produces the seller panel, and `src/modeling.py` trains either.

## 7. Dataset limitations to put in report §8

- **Truncated export** (1.05M-line Excel cap), ~584k usable lines.
- **No seller identifier** → cannot separate sellers, so "this seller's demand" is not recoverable.
- **No stock or inventory columns** → no stock-out / censoring analysis on real data.
- **No returns after delivery** field distinct from `canceled`/`order_refunded` status.
- **Massive event months** (Black Friday 4.3×) dominate any month-level metric; MAE is
  therefore reported alongside WMAPE, and the event months are named rather than smoothed away.
- **2017-09 collapse** (4,375 units vs 8,374 in 2017-10) looks like a collection outage, not
  demand. It is left in, and disclosed, rather than quietly deleted.

## 7b. Benchmark script and where the numbers come from

```bash
python scripts/pakistan_benchmark.py                 # category panel (default)
python scripts/pakistan_benchmark.py --panel data/processed/pakistan_sku_month.csv
```

Writes `reports/pakistan_benchmark.csv` and `.json`. The table in §5 is that output,
unedited. Re-run it before the viva; if the numbers move, the report must move with them.

## 8. How to reproduce

```bash
# 1. place the exported CSV (not committed, 106 MB) at:
#    data/raw/pakistan_ecommerce_raw.csv
# 2. verify it is the same file
sha256sum data/raw/pakistan_ecommerce_raw.csv
#    expect: aa1344e8dde4f2ea9038eaafeaf908acbb54e6eee7783837d567be093bdd528e
# 3. build the panels + audit
python -m src.pakistan_data --input data/raw/pakistan_ecommerce_raw.csv
# 4. see the audit
cat reports/pakistan_data_audit.json
```
