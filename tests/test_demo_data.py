"""Guards on the generated demo data: it must be internally consistent."""
import numpy as np
import pandas as pd
import pytest

from src.catalog import ARCHETYPES, CATEGORY_PROFILES, price_band_midpoint, seasonal_index
from src.demo_data import GeneratorConfig, build_purchase_request, generate_base_market, generate_seller


@pytest.fixture(scope='module')
def seller():
    cfg = GeneratorConfig(months=18, seller_archetypes=8, base_products=60)
    return generate_seller(cfg)


def test_every_archetype_has_three_variants_and_a_valid_band():
    for archetype in ARCHETYPES:
        assert len(archetype.variants) == 3
        assert archetype.price_low < archetype.price_high
        assert archetype.category in CATEGORY_PROFILES
        mid = price_band_midpoint(archetype)
        assert archetype.price_low <= mid <= archetype.price_high


def test_seasonality_is_centred_around_one():
    for category in CATEGORY_PROFILES:
        values = [seasonal_index(category, m) for m in range(1, 13)]
        assert len(values) == 12
        assert all(v > 0 for v in values)
        # Not a flat line: a category with no seasonality would be pointless here.
        assert max(values) / min(values) > 1.2


def test_panel_has_no_negative_or_impossible_values(seller):
    panel, products = seller
    assert (panel['units_sold'] >= 0).all()
    assert (panel['gross_units'] >= panel['units_sold']).all()
    assert (panel['returned_units'] >= 0).all()
    assert (panel['lost_units'] >= 0).all()
    assert (panel['avg_unit_price'] > 0).all()
    assert (panel['avg_unit_cost'] > 0).all()


def test_profit_is_consistent_with_its_own_components(seller):
    panel, _ = seller
    recomputed = (panel['revenue'] - panel['cogs'] - panel['commission']
                  - panel['shipping_cost'] - panel['return_cost'] - panel['ad_spend'])
    assert np.allclose(recomputed, panel['net_profit'], atol=0.05)


def test_revenue_equals_sold_units_times_price(seller):
    panel, _ = seller
    assert np.allclose(panel['units_sold'] * panel['avg_unit_price'], panel['revenue'], rtol=1e-3, atol=0.5)


def test_status_column_matches_the_sign_of_profit(seller):
    panel, _ = seller
    assert set(panel['profit_status'].unique()) <= {'Profit', 'Loss', 'Break-even'}
    assert (panel.loc[panel['profit_status'] == 'Profit', 'net_profit'] > 0).all()
    assert (panel.loc[panel['profit_status'] == 'Loss', 'net_profit'] < 0).all()


def test_seller_history_reaches_twelve_months(seller):
    panel, _ = seller
    assert panel['month'].nunique() >= 12, 'the spec requires at least 12 months of seller history'


def test_demand_is_censored_somewhere(seller):
    """If nothing ever stocked out, the whole 'lost_units' analysis is vacuous."""
    panel, _ = seller
    assert (panel['lost_units'] > 0).any(), 'expected at least one stock-out'
    assert (panel['units_sold'] + panel['lost_units']).ge(panel['units_sold']).all()


def test_purchase_request_is_a_useful_demo_input(seller):
    _, products = seller
    request = build_purchase_request(products, budget=250_000, share=0.5)
    assert {'ProductID', 'ProductName', 'CurrentPrice', 'CurrentStock'} <= set(request.columns)
    assert len(request) > 0
    assert (request['CurrentPrice'] > 0).all()
    assert (request['CurrentStock'] >= 0).all()
    assert request['ProductID'].is_unique
    assert request.attrs['budget'] == 250_000


def test_generation_is_reproducible():
    cfg = GeneratorConfig(months=14, seller_archetypes=4, base_products=40)
    first, _ = generate_seller(cfg)
    second, _ = generate_seller(cfg)
    pd.testing.assert_frame_equal(first, second)


def test_market_panel_is_large_enough_for_stage_one():
    market = generate_base_market(GeneratorConfig(months=14, base_products=80))
    assert len(market) == 80 * 14
    assert market['monthly_units_sold'].ge(0).all()
    assert market['StockCode'].nunique() == 80
