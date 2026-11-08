"""Inference layer: turn a seller's purchase request into a budgeted buying plan.

This is what the Streamlit app calls. It is deliberately separate from the UI so it
can be unit-tested and so the same logic can be scripted in a notebook.

The seller supplies (per product): ``ProductID``, ``ProductName``, ``CurrentPrice``,
``CurrentStock``, plus history as either
  * ``Sales_01..Sales_NN`` monthly unit columns (the original ShopMind format), or
  * ``UnitsSold_01..`` / ``units_sold`` panels passed separately.
and a single ``budget`` for the month.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, OPTIONAL_FEATURES
from src.modeling import BaseModel, SellerModel

# Category-level economics used for the profit and loss estimate. These are the
# same figures that generated the demo data, so the app's rupee numbers are
# internally consistent with the training data (see data/demo/DATA_CARD.md).
CATEGORY_ECONOMICS = {
    'Fashion': {'commission': 0.13, 'return_rate': 0.21, 'ad_rate': 0.045,
                'shipping': 210, 'return_cost': 180, 'items_per_order': 1.25},
    'Electronics': {'commission': 0.08, 'return_rate': 0.10, 'ad_rate': 0.055,
                    'shipping': 260, 'return_cost': 320, 'items_per_order': 1.10},
    'Home & Kitchen': {'commission': 0.11, 'return_rate': 0.12, 'ad_rate': 0.040,
                       'shipping': 300, 'return_cost': 260, 'items_per_order': 1.15},
    'Beauty & Personal Care': {'commission': 0.12, 'return_rate': 0.15, 'ad_rate': 0.060,
                               'shipping': 200, 'return_cost': 170, 'items_per_order': 1.45},
    'Baby & Kids': {'commission': 0.10, 'return_rate': 0.11, 'ad_rate': 0.035,
                    'shipping': 210, 'return_cost': 190, 'items_per_order': 1.60},
    'Sports & Fitness': {'commission': 0.09, 'return_rate': 0.13, 'ad_rate': 0.045,
                         'shipping': 280, 'return_cost': 240, 'items_per_order': 1.20},
}
DEFAULT_ECONOMICS = {'commission': 0.11, 'return_rate': 0.13, 'ad_rate': 0.045,
                     'shipping': 240, 'return_cost': 210, 'items_per_order': 1.20}

# Safety-stock service factor: 1.28 standard deviations covers roughly 90% of a
# normal demand distribution. Stated as an assumption, not a guarantee.
SERVICE_FACTOR = 1.28


@dataclass
class PlanResult:
    """Everything the dashboard needs, plus the assumptions it was built on."""

    plan: pd.DataFrame
    summary: dict
    assumptions: list[str]


def _economics(category: str) -> dict:
    return CATEGORY_ECONOMICS.get(str(category), DEFAULT_ECONOMICS)


def expected_profit_per_unit(price: float, cost: float, category: str) -> float:
    """Expected contribution of selling one additional unit, net of marketplace costs.

    Only units that are actually *delivered and kept* earn revenue, so the return
    rate is applied first; the return leg costs money on the units that came back.
    """
    econ = _economics(category)
    keep = 1 - econ['return_rate']
    revenue = price * keep
    fees = revenue * (econ['commission'] + econ['ad_rate'])
    shipping = econ['shipping'] / max(econ['items_per_order'], 0.1)
    returns = econ['return_cost'] * econ['return_rate'] / max(econ['items_per_order'], 0.1)
    return float(revenue - fees - shipping - returns - cost * keep)


def forecast_request(request: pd.DataFrame, model, budget: float,
                     history: pd.DataFrame | None = None,
                     forecast_month: str | None = None) -> PlanResult:
    """Forecast demand for each requested product and allocate the budget.

    ``history`` is a long panel with ``StockCode``, ``month``, ``monthly_units_sold``
    (and optionally ``category``, ``promo_days``, ``discount_pct``). When absent we
    fall back to a transparent moving-average rule so the UI always returns something.
    """
    from src.features import make_features

    request = request.copy()
    request['ProductID'] = request['ProductID'].astype(str)
    notes: list[str] = []

    if history is None or history.empty:
        # No history: this is a pure cold start. Use the seller's stated intent only.
        request['AvgDemand'] = request.get('UnitsWanted', request['CurrentStock'])
        notes.append('No sales history supplied: demand is assumed equal to UnitsWanted '
                     '(or current stock when that column is missing). Forecasts are weak.')
        request['ModelRatio'] = 1.0
        request['SafetyStock'] = 0.0
    else:
        hist = history.copy()
        hist = hist.rename(columns={'product_id': 'StockCode'})
        hist['month'] = pd.to_datetime(hist['month'])
        if 'units_sold' in hist.columns and 'monthly_units_sold' not in hist.columns:
            hist['monthly_units_sold'] = hist['units_sold']
        if 'promo_days' in hist.columns and 'planned_promo_days' not in hist.columns:
            hist['planned_promo_days'] = hist['promo_days']
        if 'discount_pct' in hist.columns and 'planned_discount_pct' not in hist.columns:
            hist['planned_discount_pct'] = hist['discount_pct']

        known = set(hist['StockCode'].astype(str))
        unknown = sorted(set(request['ProductID']) - known)
        if unknown:
            notes.append(f'{len(unknown)} requested product(s) have no history and fall back '
                         'to the moving-average rule: ' + ', '.join(unknown[:5])
                         + ('...' if len(unknown) > 5 else ''))

        features = make_features(hist)
        if features.empty:
            request['AvgDemand'] = request.get('UnitsWanted', 0)
            request['ModelRatio'] = 1.0
            request['SafetyStock'] = 0.0
            notes.append('Not enough history to build features; using UnitsWanted.')
        else:
            # Forecast the month after the last observed month.
            last_month = features['month_date'].max()
            nxt = (last_month + pd.DateOffset(months=1)).to_period('M').to_timestamp()
            rows = []
            for product_id in request['ProductID']:
                series = hist[hist['StockCode'].astype(str) == product_id].sort_values('month')
                if len(series) < 6:
                    rows.append({'ProductID': product_id, 'AvgDemand': np.nan})
                    continue
                recent = series['monthly_units_sold'].iloc[-6:].to_numpy(dtype=float)
                rows.append({'ProductID': product_id,
                             'rolling_mean_3': float(recent[-3:].mean()),
                             'rolling_mean_6': float(recent.mean()),
                             'rolling_std_6': float(recent.std(ddof=1)) if len(recent) > 1 else 0.0,
                             'month': nxt.month})
            anchor = pd.DataFrame(rows).set_index('ProductID')
            request = request.join(anchor, on='ProductID')

            if 'rolling_mean_3' in request.columns:
                base = model.base if isinstance(model, SellerModel) else model
                columns = base.feature_columns
                inference = pd.DataFrame(index=request.index)
                inference['month'] = request['month'].fillna(nxt.month)
                inference['month_sin'] = np.sin(2 * np.pi * inference['month'] / 12)
                inference['month_cos'] = np.cos(2 * np.pi * inference['month'] / 12)
                inference['months_since_start'] = 24
                inference['lag_1'] = request['rolling_mean_3']
                inference['lag_3'] = request['rolling_mean_3']
                inference['lag_6'] = request['rolling_mean_6']
                inference['rolling_mean_3'] = request['rolling_mean_3']
                inference['rolling_mean_6'] = request['rolling_mean_6']
                inference['rolling_std_6'] = request['rolling_std_6'].fillna(0)
                safe6 = request['rolling_mean_6'].replace(0, np.nan)
                inference['lag_1_ratio'] = (inference['lag_1'] / safe6).fillna(1.0)
                inference['trend'] = (request['rolling_mean_3'] / safe6).fillna(1.0)
                inference['volatility'] = (request['rolling_std_6'] / safe6).fillna(0)
                for optional in OPTIONAL_FEATURES:
                    inference[optional] = 0.0

                categories = request['Category'] if 'Category' in request.columns else None
                ratio = (model.predict_ratio(inference[columns], categories)
                         if isinstance(model, SellerModel)
                         else model.predict_ratio(inference[columns]))
                request['ModelRatio'] = ratio
                request['AvgDemand'] = ratio * request['rolling_mean_3'].fillna(0)
                request['SafetyStock'] = SERVICE_FACTOR * request['rolling_std_6'].fillna(0)
                missing = request['AvgDemand'].isna()
                if missing.any():
                    request.loc[missing, 'AvgDemand'] = request.loc[missing, 'UnitsWanted'] \
                        if 'UnitsWanted' in request.columns else 0.0
                    request['SafetyStock'] = request['SafetyStock'].fillna(0)

    request['AvgDemand'] = pd.to_numeric(request['AvgDemand'], errors='coerce').fillna(0).clip(lower=0)
    request['SafetyStock'] = pd.to_numeric(request.get('SafetyStock', 0), errors='coerce').fillna(0).clip(lower=0)
    request['RecommendedPurchase'] = 0
    request['Investment'] = 0.0

    plan = allocate_budget(request, budget)
    summary = summarise(plan, budget)
    assumptions = _assumptions(model)
    return PlanResult(plan=plan, summary=summary, assumptions=assumptions + notes)


def allocate_budget(plan: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Greedy, profit-per-rupee ranked allocation.

    Products are ranked by expected *profit contribution per rupee of purchase*, which
    is the right ranking when the constraint is cash rather than shelf space. Ties are
    broken by the size of the shortage so large gaps are not starved by rounding.
    """
    plan = plan.copy().reset_index(drop=True)
    if not np.isfinite(budget) or budget < 0:
        raise ValueError('Budget must be a finite, non-negative number.')
    price = pd.to_numeric(plan['CurrentPrice'], errors='coerce')
    stock = pd.to_numeric(plan['CurrentStock'], errors='coerce')
    if (price <= 0).any() or price.isna().any():
        raise ValueError('CurrentPrice must be a positive number for every product.')
    if (stock < 0).any() or stock.isna().any():
        raise ValueError('CurrentStock must be zero or a positive number for every product.')

    # Be forgiving for direct callers (notebooks): if no forecast columns are present
    # the seller's own stated quantities are used instead of inventing demand.
    if 'AvgDemand' not in plan.columns:
        plan['AvgDemand'] = pd.to_numeric(
            plan.get('UnitsWanted', pd.Series(0, index=plan.index)), errors='coerce').fillna(0)
    if 'SafetyStock' not in plan.columns:
        plan['SafetyStock'] = 0.0
    required = plan['AvgDemand'] + plan['SafetyStock']
    plan['RequiredInventory'] = required
    # Shortage is a statement about *demand vs stock*: never masked by what the seller
    # said they wanted to buy (that is a separate cap, applied to the purchase only).
    plan['ShortageUnits'] = np.ceil(np.maximum(0.0, required - stock)).astype(int)
    if 'UnitsWanted' in plan.columns:
        wanted = pd.to_numeric(plan['UnitsWanted'], errors='coerce')
        plan['PurchaseCap'] = wanted.fillna(np.inf).clip(lower=0).fillna(np.inf)
    else:
        plan['PurchaseCap'] = np.inf

    category = plan['Category'] if 'Category' in plan.columns else pd.Series('', index=plan.index)
    # ``CurrentPrice`` is the seller's *purchase* cost. The retail price (if supplied)
    # is what the unit sells for, and only that combination yields a margin.
    cost = price
    retail = (pd.to_numeric(plan['RetailPrice'], errors='coerce').fillna(price)
              if 'RetailPrice' in plan.columns else price)
    plan['UnitCost'] = cost
    plan['RetailPrice'] = retail
    plan['ProfitPerUnit'] = [expected_profit_per_unit(r, c, cat)
                             for r, c, cat in zip(retail, cost, category)]
    plan['ProfitPerRupee'] = plan['ProfitPerUnit'] / cost

    # Initialise outputs before the loop so an empty allocation still returns a
    # complete frame (zero budget, or nothing short).
    plan['RecommendedPurchase'] = 0
    plan['Investment'] = 0.0

    ranked = plan[plan['ShortageUnits'] > 0].sort_values(
        ['ProfitPerRupee', 'ShortageUnits'], ascending=[False, False])
    remaining = float(budget)
    for idx in ranked.index:
        unit_cost = float(cost.loc[idx])
        affordable = int(remaining // unit_cost)
        cap = plan.at[idx, 'PurchaseCap']
        units = int(min(plan.at[idx, 'ShortageUnits'], affordable, cap))
        if units > 0:
            plan.loc[idx, 'RecommendedPurchase'] = units
            plan.loc[idx, 'Investment'] = round(units * unit_cost, 2)
            remaining = round(remaining - units * unit_cost, 2)
    plan['ExpectedProfit'] = (plan['RecommendedPurchase'] * plan['ProfitPerUnit']).round(2)
    plan['ExpectedRevenue'] = (plan['RecommendedPurchase'] * retail
                               * (1 - category.map(lambda c: _economics(c)['return_rate']))).round(2)
    plan['RecommendedPurchase'] = plan['RecommendedPurchase'].astype(int)
    plan['IsUnprofitable'] = plan['ProfitPerUnit'] < 0
    return plan.sort_values(['Investment', 'ShortageUnits'], ascending=False).reset_index(drop=True)


def summarise(plan: pd.DataFrame, budget: float) -> dict:
    investment = float(plan['Investment'].sum())
    profit = float(plan['ExpectedProfit'].sum()) if 'ExpectedProfit' in plan else 0.0
    stockouts = int((plan['ShortageUnits'] > plan['RecommendedPurchase']).sum())
    return {
        'budget': float(budget),
        'investment': round(investment, 2),
        'unallocated': round(float(budget) - investment, 2),
        'utilisation_pct': round(100 * investment / budget, 1) if budget else 0.0,
        'expected_profit': round(profit, 2),
        'expected_roi_pct': round(100 * profit / investment, 1) if investment else 0.0,
        'products': int(len(plan)),
        'funded_products': int((plan['RecommendedPurchase'] > 0).sum()),
        'partial_fill_products': stockouts,
        'unprofitable_products': int(plan['IsUnprofitable'].sum()) if 'IsUnprofitable' in plan else 0,
        'skipped_below_margin': int(((plan['ProfitPerUnit'] < 0) & (plan['RecommendedPurchase'] > 0)).sum())
        if 'ProfitPerUnit' in plan else 0,
    }


def _assumptions(model) -> list[str]:
    lines = [
        'Demand is forecast as a ratio to the trailing three-month mean and converted back to units.',
        f'Safety stock = {SERVICE_FACTOR} x the sample standard deviation of the last six months '
        '(about a 90% service level under a normal demand assumption).',
        'Profit per unit = price x (1 - return rate) - marketplace commission - ad spend '
        '- outbound shipping - return handling - unit cost.',
        'Budget is allocated greedily by expected profit per rupee of purchase; it is not a '
        'globally optimal knapsack solution.',
    ]
    if isinstance(model, SellerModel):
        if model.method == 'category_offset':
            lines.append('The market model is adjusted by a per-category calibration factor, each '
                         'shrunk toward the seller-wide factor using that category\'s sample size.')
        elif model.method == 'offset':
            lines.append(f'The market model is adjusted by one pooled seller factor ({model.offset:.3f}x).')
    return lines
