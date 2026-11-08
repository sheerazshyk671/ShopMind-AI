# EDA takeaways

Each figure must appear in the report with the takeaway below it.

## fig1_seasonality.png — Demand over 24 months

Spikes land in Ramadan/Eid months (Mar 2025, Feb-Mar 2026) and Black Friday; because Ramadan moves ~11 days earlier each year, a month-of-year feature cannot capture this.

## fig2_concentration.png — Revenue concentration across SKUs

The top 10% of products (7 of 66) carry 42% of revenue, so unit-level MAE understates the rupee cost of errors on those SKUs — which is why a rupee-weighted metric is reported alongside MAE.

## fig3_censoring.png — Stock-outs truncate observed sales

Service level averages 87% and falls hardest in festival months; the SKUs with the most lost units are exactly the ones a naive model under-forecasts.

## fig4_margin.png — Margin by category

Electronics earns 3% while Fashion earns 23%, so ranking purchases by revenue or by unit shortage would misallocate the budget — the planner ranks by profit per rupee.
