"""Synthetic-but-grounded Pakistani e-commerce datasets for ShopMind AI.

Three datasets are produced, in the order the project trains them:

1. ``base_market_monthly.csv``  — a *general* market panel (many sellers, many
   products, 24 months). This is the "train on the market" stage. Its real-world
   counterpart is Online Retail II / Pakistan's Largest E-Commerce Dataset; this
   generator exists so the whole pipeline runs offline and reproducibly.
2. ``seller_sales_history.csv`` — 12+ months of one seller's own product-level
   history with units sold, returns, revenue, cost, and realised profit/loss.
3. ``seller_products.csv``      — the seller's current catalogue with live price
   and on-hand units, plus a demo purchase request (product + price + stock + budget).

Design principles
-----------------
* Prices come from verified November-2026 Pakistani retail bands (see
  ``src/catalog.py`` and ``data/demo/DATA_CARD.md``), not arbitrary numbers.
* Profit is *computed from a real P&L* — revenue, COGS, commission, outbound
  shipping, return-leg cost, ad spend — not invented, so a loss-making month is
  loss-making for a stated reason.
* Seasonality follows the actual Pakistani retail calendar (Ramadan/Eid shift
  every lunar year, 11.11, winter bedding, back-to-school).
* The data-generating process is fully documented, so the demo is auditable and
  a reader can see exactly which signal the model is expected to recover.

Everything is seeded; the same seed reproduces the same bytes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.catalog import (
    ARCHETYPES,
    VARIANT_DEMAND_MULT,
    Archetype,
    category_profile,
    price_band_midpoint,
    seasonal_index,
)

# ---------------------------------------------------------------------------
# Fixed calendar events used by the generator (documented in DATA_CARD.md)
# ---------------------------------------------------------------------------
# Ramadan start dates (Pakistan, Ruet-e-Hilal announced/expected).
RAMADAN_START = {2024: (3, 12), 2025: (3, 1), 2026: (2, 19), 2027: (2, 8)}
# Single-day campaign spikes layered on top of seasonality.
CAMPAIGN_MONTHS = {'Electronics': (11,), 'Fashion': (3, 5), 'Beauty & Personal Care': (11,)}


@dataclass(frozen=True)
class SellerProfile:
    """How this particular seller differs from the general market.

    Without this the seller would be a random sample of the market, the market base
    model would already be perfectly calibrated, and stage-2 fine-tuning would have
    nothing to learn. Real sellers *are* different - category mix, festival intensity,
    growth rate and product lifecycle all deviate - so the difference is modelled
    explicitly and can then be recovered from the seller's own history.
    """

    category_level: dict = field(default_factory=lambda: {
        'Fashion': 1.45, 'Beauty & Personal Care': 1.20, 'Home & Kitchen': 0.75,
        'Electronics': 0.55, 'Baby & Kids': 1.10, 'Sports & Fitness': 0.70,
    })
    # This seller leans hard into Ramadan and winter weddings.
    ramadan_boost: float = 1.45
    # and grows faster than the market
    growth_premium: float = 0.012
    # share of SKUs that are quietly dying, and share that are trending
    declining_share: float = 0.12
    growing_share: float = 0.12
    declining_drift: float = -0.045
    growing_drift: float = 0.055
    # A real seller ramps into a new catalogue over the first months rather than
    # trading at full scale from month one (listing build-up, review accumulation).
    ramp_months: int = 4
    ramp_start: float = 0.55


@dataclass(frozen=True)
class GeneratorConfig:
    """Knobs for the simulation. Defaults describe a mid-size Daraz seller."""

    start: str = '2024-10-01'
    months: int = 24
    seed: int = 42
    base_products: int = 1_200
    seller_archetypes: int = 22
    market_growth_monthly: float = 0.015
    demand_noise_sigma: float = 0.16
    spike_probability: float = 0.02
    stockout_target: float = 0.12
    fx_shock_month: int = 14          # index into the horizon
    fx_shock_size: float = 0.09       # electronics import-cost jump
    promotions: bool = True
    seller: SellerProfile = field(default_factory=SellerProfile)


def _month_index(start: str, months: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), periods=months, freq='MS')


def _ramadan_months(months: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """Return the month stamps that contain Ramadan (demand ramps before Eid)."""
    flagged = set()
    for year, (month, day) in RAMADAN_START.items():
        try:
            start = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:  # pragma: no cover - leap-year guard
            continue
        for offset in (0, 1):
            stamp = (start + pd.DateOffset(months=offset)).to_period('M').to_timestamp()
            if months[0] <= stamp <= months[-1]:
                flagged.add(stamp)
    return flagged


def _variant_prices(archetype, rng: np.random.Generator) -> list[float]:
    """Three variant prices spread across the verified band on a log scale."""
    low, high = archetype.price_low, archetype.price_high
    log_low, log_high = np.log(low), np.log(high)
    # Variant multipliers shift within the band rather than beyond it.
    anchors = np.clip(
        np.log(high) - (np.log(high) - np.log(low)) * np.array([0.82, 0.52, 0.18]),
        log_low, log_high,
    )
    jitter = rng.normal(0, 0.045, size=3)
    return [float(np.round(np.exp(a + j) / 10) * 10) for a, j in zip(anchors, jitter)]


def _elasticity(category: str) -> float:
    """Price elasticity of demand (units), by category."""
    return {
        'Fashion': 1.25,
        'Electronics': 1.60,
        'Home & Kitchen': 1.15,
        'Beauty & Personal Care': 1.05,
        'Baby & Kids': 1.10,
        'Sports & Fitness': 1.30,
    }[category]


def _campaign_bump(category: str, month: pd.Timestamp) -> float:
    return 1.18 if month.month in CAMPAIGN_MONTHS.get(category, ()) else 1.0


# ---------------------------------------------------------------------------
# Stage 1 — general market panel
# ---------------------------------------------------------------------------
def generate_base_market(config: GeneratorConfig | None = None) -> pd.DataFrame:
    """General market: many products, one row per product-month.

    This is the dataset for the *base* model. It deliberately contains the same
    signal structure a real marketplace export has: level, trend, annual
    seasonality, price movement and demand noise.
    """
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    months = _month_index(cfg.start, cfg.months)

    rows = []
    categories = list(category_profile.__doc__ and {a.category for a in ARCHETYPES})
    for pid in range(cfg.base_products):
        archetype = ARCHETYPES[pid % len(ARCHETYPES)]
        prof = category_profile(archetype.category)
        band_mid = price_band_midpoint(archetype)
        base_price = float(np.exp(rng.normal(np.log(band_mid), 0.28)))
        base_price = float(np.clip(base_price, archetype.price_low * 0.6, archetype.price_high * 1.5))
        base_cost = base_price / (prof['markup'] * float(np.exp(rng.normal(0, 0.08))))
        level = (prof['base_units'] * archetype.demand_scale
                 * float(np.exp(rng.normal(0, 0.35))))
        drift = float(rng.normal(0, 0.012))
        growth = cfg.market_growth_monthly + drift
        sigma = cfg.demand_noise_sigma * float(np.exp(rng.normal(0, 0.25)))
        elastic = _elasticity(archetype.category)
        slug = archetype.name.lower().replace(' ', '-').replace(',', '').replace('+', '')
        pid_str = f'MKT-{pid:05d}'

        for t, month in enumerate(months):
            season = seasonal_index(archetype.category, month.month)
            trend = (1 + growth) ** t
            # Price rises with retail inflation, with occasional promo months.
            price = base_price * (1 + prof['price_inflation']) ** t
            discount = 0.0
            if cfg.promotions and rng.random() < 0.18:
                discount = float(rng.uniform(0.05, 0.28))
            net_price = price * (1 - discount)
            price_effect = (net_price / base_price) ** (-elastic)
            latent = level * season * trend * price_effect * _campaign_bump(archetype.category, month)
            latent *= float(np.exp(rng.normal(0, sigma)))
            if rng.random() < cfg.spike_probability:
                latent *= float(rng.uniform(1.5, 2.6))
            units = max(0.0, latent)
            sold = int(round(units))
            rows.append({
                'StockCode': pid_str,
                'month': month,
                'monthly_units_sold': sold,
                'avg_unit_price': round(net_price, 2),
                'unit_cost': round(base_cost * (1 + prof['cost_inflation']) ** t, 2),
                'category': archetype.category,
                'product_name': f'{archetype.name} ({slug[:18]})',
            })

    panel = pd.DataFrame(rows)
    # A seller-representative price column is what a marketplace export gives you.
    panel['unit_cost'] = panel['unit_cost'].round(2)
    panel.attrs['config'] = cfg
    return panel


# ---------------------------------------------------------------------------
# Stage 2 — one specific seller, with a real P&L
# ---------------------------------------------------------------------------
def generate_seller(config: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one seller's product-level monthly history plus current catalogue.

    Returns
    -------
    panel : monthly rows, one per product-month (>= 12 months by default 24).
    products : one row per product with the *current* price and stock.
    """
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed + 1)
    months = _month_index(cfg.start, cfg.months)
    ramadan = _ramadan_months(months)

    # Pick archetypes round-robin across categories so the demo catalogue spans the
    # market instead of clustering in whatever order the table happens to be in.
    by_category: dict[str, list[Archetype]] = {}
    for archetype in ARCHETYPES:
        by_category.setdefault(archetype.category, []).append(archetype)
    chosen: list[Archetype] = []
    cats = list(by_category)
    i = 0
    while len(chosen) < cfg.seller_archetypes:
        bucket = by_category[cats[i % len(cats)]]
        idx = i // len(cats)
        if idx < len(bucket):
            chosen.append(bucket[idx])
        i += 1
        if i > 500:  # pragma: no cover - runaway guard
            break
    rng.shuffle(chosen)

    panel_rows: list[dict] = []
    product_rows: list[dict] = []

    for a_idx, archetype in enumerate(chosen):
        prof = category_profile(archetype.category)
        variant_labels = archetype.variants
        prices = _variant_prices(archetype, rng)

        for v_idx, (variant, price0) in enumerate(zip(variant_labels, prices)):
            product_id = f'P{len(product_rows) + 1:03d}'
            name = f'{archetype.name} - {variant}'
            cost0 = price0 / (prof['markup'] * float(np.exp(rng.normal(0, 0.07))))
            seller_level = cfg.seller.category_level.get(archetype.category, 1.0)
            level = (prof['base_units'] * archetype.demand_scale * VARIANT_DEMAND_MULT[v_idx]
                     * seller_level * float(np.exp(rng.normal(0, 0.30))))
            # Product lifecycle: a few SKUs are dying, a few are taking off.
            lifecycle = 0.0
            draw = rng.random()
            if draw < cfg.seller.declining_share:
                lifecycle = cfg.seller.declining_drift
            elif draw < cfg.seller.declining_share + cfg.seller.growing_share:
                lifecycle = cfg.seller.growing_drift
            drift = float(rng.normal(0, 0.010)) + lifecycle
            sigma = cfg.demand_noise_sigma * float(np.exp(rng.normal(0, 0.22)))
            elastic = _elasticity(archetype.category)
            stock_sigma = float(rng.uniform(0.6, 1.4))   # seller's ordering tightness

            # A real seller opens the window with stock already on the shelf.
            on_hand = float(round(1.7 * level))
            purchased_prev = 0.0
            recent_demand: list[float] = []

            for t, month in enumerate(months):
                season = seasonal_index(archetype.category, month.month)
                if month in ramadan:
                    base_boost = 1.22 if archetype.category in ('Fashion', 'Home & Kitchen') else 1.10
                    # The seller's own festival intensity sits on top of the market's.
                    season *= base_boost * (cfg.seller.ramadan_boost / 1.22)
                trend = (1 + cfg.market_growth_monthly + drift + cfg.seller.growth_premium) ** t

                # Landed unit cost: category inflation + one import-price shock.
                unit_cost = cost0 * (1 + prof['cost_inflation']) ** t
                if t >= cfg.fx_shock_month and archetype.category == 'Electronics':
                    unit_cost *= (1 + cfg.fx_shock_size)

                # Retail price: inflation, then the seller's own promo decision.
                # Clearance promos are deliberately deep enough to sell below cost -
                # that is how a real seller ends up with a loss-making product-month,
                # and the dataset must contain those cases to be worth analysing.
                retail = price0 * (1 + prof['price_inflation']) ** t
                discount = 0.0
                promo_days = 0
                liquidation = rng.random() < 0.05
                if liquidation:
                    min_margin = 1 - unit_cost / retail * 1.12   # 12% below cost
                    discount = float(np.clip(rng.uniform(min_margin, min_margin + 0.10), 0.05, 0.40))
                    promo_days = int(rng.integers(7, 21))
                elif cfg.promotions and rng.random() < 0.22:
                    discount = float(rng.uniform(0.08, 0.25))
                    promo_days = int(rng.integers(3, 15))
                net_price = retail * (1 - discount)
                price_effect = (net_price / price0) ** (-elastic)

                ramp = min(1.0, cfg.seller.ramp_start
                           + (1 - cfg.seller.ramp_start) * t / max(1, cfg.seller.ramp_months))
                latent = level * ramp * season * trend * price_effect * _campaign_bump(archetype.category, month)
                latent *= float(np.exp(rng.normal(0, sigma)))
                if rng.random() < cfg.spike_probability:
                    latent *= float(rng.uniform(1.4, 2.4))
                latent = max(0.0, latent)

                # --- inventory mechanics ----------------------------------
                available = on_hand + purchased_prev
                demand_units = int(round(latent))
                gross = int(min(demand_units, available))
                lost = max(0, demand_units - gross)
                stock_out_days = int(min(30, round(30 * lost / demand_units))) if demand_units > 0 and lost > 0 else 0

                returns = int(round(gross * prof['return_rate']
                                    * float(np.exp(rng.normal(0, 0.25)))))
                returns = int(np.clip(returns, 0, gross))
                units_sold = gross - returns

                # --- P&L (accrual) ----------------------------------------
                revenue = units_sold * net_price
                cogs = units_sold * unit_cost
                commission = revenue * prof['commission_rate']
                outbound_shipping = (gross / prof['items_per_order']) * prof['shipping_cost']
                return_leg = (returns / prof['items_per_order']) * prof['return_cost']
                ad_spend = revenue * prof['ad_rate']
                net_profit = revenue - cogs - commission - outbound_shipping - return_leg - ad_spend

                panel_rows.append({
                    'month': month,
                    'product_id': product_id,
                    'product_name': name,
                    'category': archetype.category,
                    'variant': variant,
                    'units_sold': units_sold,
                    'gross_units': gross,
                    'returned_units': returns,
                    'lost_units': lost,
                    'avg_unit_price': round(net_price, 2),
                    'avg_unit_cost': round(unit_cost, 2),
                    'revenue': round(revenue, 2),
                    'cogs': round(cogs, 2),
                    'commission': round(commission, 2),
                    'shipping_cost': round(outbound_shipping, 2),
                    'return_cost': round(return_leg, 2),
                    'ad_spend': round(ad_spend, 2),
                    'net_profit': round(net_profit, 2),
                    'margin_pct': round(100 * net_profit / revenue, 2) if revenue > 0 else 0.0,
                    'profit_status': 'Profit' if net_profit > 0 else ('Loss' if net_profit < 0 else 'Break-even'),
                    'promo_days': promo_days,
                    'discount_pct': round(100 * discount, 1),
                    'stock_on_hand': int(round(max(0.0, available - gross))),
                    'stock_out_days': stock_out_days,
                    'purchased_units': int(round(purchased_prev)),
                })

                # --- seller reorders for next month (naive, demand-based) ---
                # The seller anchors on *observed* demand (not just sales), reorders
                # with a bullwhip-prone order-up-to rule, and the goods arrive one
                # month later. Because the order is placed before next month's demand
                # is known, a surprise spike becomes a stock-out the seller cannot
                # cover - which is what makes this a genuine forecasting problem.
                on_hand = available - gross
                recent_demand.append(latent)
                window = recent_demand[-3:]
                expected = float(np.mean(window)) * (1 + 0.15 / stock_sigma)
                next_season = seasonal_index(archetype.category, (month + pd.DateOffset(months=1)).month)
                expected *= next_season / seasonal_index(archetype.category, month.month)
                target = expected * (1.05 + 0.22 * stock_sigma)
                cash_squeeze = rng.random() < 0.10
                ordered = 0.0 if cash_squeeze else max(0.0, target - on_hand)
                # Supplier lead time + lot-size noise make the arrival imperfect.
                purchased_prev = float(round(ordered * float(np.exp(rng.normal(0, 0.30)))))

            # Current catalogue: last month's ending state.
            last = panel_rows[-1]
            units_recent = [r['units_sold'] for r in panel_rows[-6:] if r['product_id'] == product_id]
            product_rows.append({
                'product_id': product_id,
                'product_name': name,
                'category': archetype.category,
                'variant': variant,
                'current_price': round(float(price0 * (1 + prof['price_inflation']) ** (len(months) - 1)), 2),
                'current_unit_cost': round(float(last['avg_unit_cost']), 2),
                'current_stock': int(last['stock_on_hand']),
                'avg_monthly_units': round(float(np.mean(units_recent)), 1) if units_recent else 0.0,
                'return_rate': prof['return_rate'],
                'active': 1,
            })

    panel = pd.DataFrame(panel_rows).sort_values(['product_id', 'month']).reset_index(drop=True)
    products = pd.DataFrame(product_rows)
    panel.attrs['config'] = cfg
    return panel, products


# ---------------------------------------------------------------------------
# Stage 3 — the seller's month-end purchase request (the app's input)
# ---------------------------------------------------------------------------
def build_purchase_request(products: pd.DataFrame, budget: float = 500_000.0,
                           share: float = 1.0, seed: int = 7) -> pd.DataFrame:
    """Build the CSV a seller would hand to the app: what they want to buy, at what price."""
    rng = np.random.default_rng(seed)
    pick = products.sample(frac=share, random_state=seed).copy()
    # The seller asks for a quantity they *want*, which may exceed the budget.
    wanted = np.maximum(5, np.round(pick['avg_monthly_units'] * rng.uniform(0.8, 2.2)))
    request = pd.DataFrame({
        'ProductID': pick['product_id'].to_numpy(),
        'ProductName': pick['product_name'].to_numpy(),
        # CurrentPrice is what the seller pays to buy the item (landed cost). The
        # seller's own retail price is carried separately so the app can estimate margin.
        'CurrentPrice': pick['current_unit_cost'].to_numpy(),
        'RetailPrice': pick['current_price'].to_numpy(),
        'CurrentStock': pick['current_stock'].to_numpy(),
        'Category': pick['category'].to_numpy(),
        'UnitsWanted': wanted.astype(int),
    })
    request.attrs['budget'] = float(budget)
    return request.reset_index(drop=True)


def write_demo_files(outdir: str | Path = 'data/demo', config: GeneratorConfig | None = None) -> dict[str, Path]:
    """Generate every demo artefact and write it to ``outdir``."""
    cfg = config or GeneratorConfig()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    base = generate_base_market(cfg)
    panel, products = generate_seller(cfg)
    request = build_purchase_request(products, budget=500_000.0, share=0.55)

    paths = {
        'base_market': out / 'base_market_monthly.csv',
        'seller_panel': out / 'seller_sales_history.csv',
        'seller_products': out / 'seller_products.csv',
        'purchase_request': out / 'seller_purchase_request.csv',
    }
    base.to_csv(paths['base_market'], index=False)
    panel.to_csv(paths['seller_panel'], index=False)
    products.to_csv(paths['seller_products'], index=False)
    request.to_csv(paths['purchase_request'], index=False)
    return paths


def summarize(panel: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """One-screen health check so a reader can sanity-check the generated data."""
    monthly = panel.groupby('month').agg(
        revenue=('revenue', 'sum'),
        net_profit=('net_profit', 'sum'),
        units=('units_sold', 'sum'),
        lost_units=('lost_units', 'sum'),
    )
    monthly['margin_pct'] = (100 * monthly['net_profit'] / monthly['revenue']).round(1)
    monthly['service_level'] = (1 - monthly['lost_units'] / (monthly['units'] + monthly['lost_units'])).round(3)
    return monthly.round(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/demo')
    parser.add_argument('--months', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start', default='2024-10-01')
    args = parser.parse_args()

    cfg = GeneratorConfig(start=args.start, months=args.months, seed=args.seed)
    paths = write_demo_files(args.out, cfg)
    for name, path in paths.items():
        print(f'{name:16s} -> {path}  ({path.stat().st_size / 1024:.0f} KB)')

    panel = pd.read_csv(paths['seller_panel'])
    products = pd.read_csv(paths['seller_products'])
    print('\nSeller panel:', panel.shape, '| products:', products.shape)
    print(panel['month'].min(), 'to', panel['month'].max())
    print('\nMonthly P&L:')
    print(summarize(panel, products).to_string())


if __name__ == '__main__':
    main()
