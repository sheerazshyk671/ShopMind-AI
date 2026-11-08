# Data Card — ShopMind AI demo datasets

**Status: SYNTHETIC.** Every number in `data/demo/` is produced by `src/demo_data.py`.
No real seller's sales were used, and no live site was scraped. This card exists because
the course guidelines require an honest description of the data, and because a report that
claims synthetic data is real would be a fabrication.

**Why synthetic?** A private seller's product-level monthly history with profit/loss is not
publicly available anywhere. The real alternative is a famous Kaggle notebook someone else
already wrote — explicitly disallowed. So the demo data is *generated from published
Pakistani market parameters*, documented here, and the project is explicit about it.

---

## 1. Files

| File | Rows | Grain | Role in the project |
|---|---:|---|---|
| `base_market_monthly.csv` | 28,800 | product × month | **Stage 1** — general market panel, the "train on the market" data |
| `seller_sales_history.csv` | 1,584 | product × month | **Stage 2** — one seller's 24 months, with a full P&L |
| `seller_products.csv` | 66 | product | Current catalogue: live price + on-hand units |
| `seller_purchase_request.csv` | 36 | product | **Stage 3** — what the seller wants to buy, with prices and budget |

`CurrentPrice` in the request file is the seller's **purchase** cost; `RetailPrice` is what
the item sells for. The app needs both to estimate margin — giving it only the cost makes
every product look unprofitable, which is a mistake worth avoiding in your report.

Regenerate with (seed fixed, byte-identical output):

```bash
python -m src.demo_data --out data/demo --months 24 --seed 42 --start 2024-10-01
```

## 2. What is real and what is simulated

| Element | Status | Basis |
|---|---|---|
| Retail **price bands** per product | **Real, published** | November-2026 Pakistani listings; each archetype in `src/catalog.py` carries its verified band and its source note |
| **Ramadan / Eid / 11.11 / winter** demand timing | **Real calendar** | Ramadan start dates for 2024–2027; monthly seasonal multipliers per category |
| **Category economics** (markup, commission, ad rate, shipping) | **Published ranges** | See the table in §6 |
| **COD return rates** by category | **Published ranges** | Fashion 18–25%, beauty 12–18%, home 10–15%, electronics 8–12% |
| **Unit counts**, monthly noise, spikes, individual seller's stock levels | **Simulated** | Log-normal noise, seeded; not observed data |
| **Profit/loss** | **Simulated, but from a real P&L formula** | Revenue − COGS − commission − shipping − return leg − ad spend (§4) |

The distinction matters for the report: the *structure* is grounded, the *observations* are not.

## 3. How the data is generated (the DGP you must explain in your report)

For product *i* in month *t*:

```
demand_it = level_i × seasonality(category, month) × (1+growth)^t
            × (net_price / base_price)^(−elasticity)
            × campaign_bump × exp(ε_it)        ε ~ N(0, σ²)
            × spike                          spike ~ U(1.4, 2.4) w.p. 0.02

gross_it   = min(demand_it, on_hand + purchased_prev)      # stock-out truncation
lost_it    = demand_it − gross_it                          # unmet demand
returns_it = Binomial-ish(gross_it, category_return_rate)
sold_it    = gross_it − returns_it
```

The **stock-out truncation is the single most important modelling detail in the whole
dataset**: observed sales are a *censored* version of demand. A month where a product sold
out under-reports demand, and a model trained naively on sales will systematically
under-forecast popular items — exactly the mistake that causes repeated stock-outs in
real inventory planning. `lost_units` is what makes the censoring visible.

## 4. The seller's P&L (why a product-month can be loss-making)

```
revenue      = sold_units × net_unit_price
cogs         = sold_units × landed_unit_cost
commission   = revenue × marketplace_commission      # 8–15%
shipping     = (gross_units / items_per_order) × courier_cost
return_leg   = (returned_units / items_per_order) × return_handling_cost
ad_spend     = revenue × ad_rate                     # 3.5–6%
net_profit   = revenue − cogs − commission − shipping − return_leg − ad_spend
```

Realised margins in the generated data (24 months, seed 42). Recompute for your own run —
these move when the seller profile changes:

| Category | Revenue (Rs) | Margin | Return rate | Why |
|---|---:|---:|---:|---|
| Fashion | 40.6 M | **22%** | 22% | highest markup (1.95×), highest returns |
| Home & Kitchen | 28.2 M | **20%** | 12% | heavy items, high shipping |
| Baby & Kids | 14.3 M | **20%** | 11% | high items-per-order spreads shipping |
| Beauty & Personal Care | 21.4 M | **16%** | 15% | high commission + ad spend |
| Sports & Fitness | 10.0 M | **12%** | 13% | seasonal, thin markup |
| Electronics | see note | **~2–3%** | 10% | 1.55× markup, plus an import-cost shock |

That electronics figure is not a bug: a **+9% landed-cost shock** hits electronics from
month 15 onward (a rupee-devaluation scenario). The seller cannot re-price instantly, so
margins collapse. This is the kind of event a real Pakistani importer faces, and it gives
the report a concrete, explainable failure to analyse.

Overall (24 months, seed 42): **Rs 182.9 M revenue, Rs 33.3 M profit, 18.2% net margin**.
Of 1,584 product-months, **1,306 profit / 247 loss / 31 break-even** — 15.6% lose money,
almost all of them deep clearance promotions priced below cost, plus electronics months
after the import-cost shock. Overall service level is **0.849**: about 15% of demand was
lost to stock-outs.

Regenerate and re-read these figures with:

```bash
python -m src.demo_data --out data/demo
python -c "import pandas as pd; p=pd.read_csv('data/demo/seller_sales_history.csv'); print(p.profit_status.value_counts()); print(100*p.net_profit.sum()/p.revenue.sum())"
```

## 5. Deliberate difficulty (so the project is not trivial)

1. **Censored demand** (stock-outs) — a naive sales model is biased low.
2. **Lumpy, zero-heavy series** — many slow movers; skews MAE and R².
3. **Structural break** — import-cost shock in month 15 changes the margin regime.
4. **Lunar-calendar seasonality** — Ramadan moves ~11 days earlier each year, so a
   month-of-year feature cannot capture it; a naive seasonal-naive baseline will miss.
5. **Return-rate heterogeneity** — fashion returns are ~2× electronics.

## 6. Published parameters used (state these in your report)

| Parameter | Value used | Published Pakistani range | Source class |
|---|---|---|---|
| Marketplace commission | 8–15% | "typically 8 to 15 percent depending on category" | industry guide, 2026 |
| COD share of orders | assumed dominant | 55–95% across sources | industry reports, 2026 |
| COD return/RTO rate | 10–22% by category | national 25–35%; fashion 18–25%, electronics 8–12%, beauty 12–18% | industry reports, 2026 |
| Net margin, mixed catalogue | 14.9% | category-dependent, single digits to ~25% | derived |
| Ramadan 2024/2025/2026/2027 start | 12 Mar / 1 Mar / 19 Feb / 8 Feb | published expected dates | calendar |
| Air fryer retail band | Rs 17,900–55,900 | Anex/Westpoint/Kenwood listings, Nov 2026 | retailer listings |
| Lawn suit band | Rs 2,990–7,990 | Gul Ahmed/Khaadi/Burays bands, 2026 | retailer listings |

## 7. Known limitations (put these in report §8)

- **Not real observations.** Any accuracy figure is a statement about this simulator, not
  about Pakistani e-commerce. Say so plainly; do not claim business impact.
- **Single seller, 24 months.** No cross-seller variation, no multi-year trend, so the
  chronological test window is only the last few months.
- **Demand is recoverable by design.** The generator has a smooth, low-dimensional signal,
  so a model *should* beat the baseline. Failing to beat it here means a pipeline bug,
  not a hard problem — which makes this a fair test of the pipeline rather than a hard
  forecasting benchmark.
- **Return rates are constant per category.** Real returns depend on price, city and
  courier; that heterogeneity is absent.
- **Prices are catalogue-consistent.** No competitor undercutting, no price wars.

## 8. If you get real seller data later

The schema in `seller_sales_history.csv` **is** the contract. A seller's export needs:
`month, product_id, product_name, category, units_sold, returned_units, avg_unit_price,
avg_unit_cost`. Profit columns can be recomputed with the formula in §4. Everything
downstream of `src/demo_data.py` reads these two files and does not care whether they came
from the simulator.
