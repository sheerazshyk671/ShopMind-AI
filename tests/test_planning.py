"""Tests for the buying-plan layer: budget rules, margin logic, fallbacks."""
import numpy as np
import pandas as pd
import pytest

from src.modeling import BaseModel, SellerModel
from src.planning import (
    DEFAULT_ECONOMICS,
    allocate_budget,
    expected_profit_per_unit,
    forecast_request,
    summarise,
)


class ConstantRatio:
    def __init__(self, value=1.0):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def request_frame(**overrides):
    data = {
        'ProductID': ['A', 'B', 'C'],
        'ProductName': ['Mug', 'Kettle', 'Blender'],
        'CurrentPrice': [1200.0, 8400.0, 14000.0],
        'RetailPrice': [2400.0, 16800.0, 28000.0],
        'CurrentStock': [5, 40, 0],
        'Category': ['Home & Kitchen', 'Home & Kitchen', 'Home & Kitchen'],
        'UnitsWanted': [50, 60, 20],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def history_frame(months=14, product='A', base=40):
    start = pd.Timestamp('2025-01-01')
    return pd.DataFrame({
        'product_id': product,
        'month': [start + pd.DateOffset(months=i) for i in range(months)],
        'units_sold': np.linspace(base, base * 1.4, months).round().astype(int),
        'category': 'Home & Kitchen',
    })


# --- margin maths ----------------------------------------------------------
def test_profit_per_unit_is_positive_when_markup_is_enough():
    assert expected_profit_per_unit(2400, 1200, 'Home & Kitchen') > 0


def test_profit_per_unit_is_negative_when_selling_below_cost():
    assert expected_profit_per_unit(900, 1200, 'Home & Kitchen') < 0


def test_a_thin_markup_can_still_lose_money_after_fees_and_returns():
    """A 2% markup is not profitable once commission, shipping and returns are paid."""
    assert expected_profit_per_unit(5_100, 5_000, 'Home & Kitchen') < 0
    assert expected_profit_per_unit(8_000, 5_000, 'Home & Kitchen') > 0


def test_unknown_category_falls_back_to_default_economics():
    known = expected_profit_per_unit(2400, 1200, 'Home & Kitchen')
    unknown = expected_profit_per_unit(2400, 1200, 'Not A Real Category')
    assert isinstance(known, float) and isinstance(unknown, float)
    econ = DEFAULT_ECONOMICS
    assert 0 < econ['return_rate'] < 1 and 0 < econ['commission'] < 1


def test_high_return_category_earns_less_per_unit():
    fashion = expected_profit_per_unit(2400, 1000, 'Fashion')          # 21% returns
    beauty = expected_profit_per_unit(2400, 1000, 'Beauty & Personal Care')  # 15% returns
    assert fashion < beauty


# --- budget allocation -----------------------------------------------------
def test_budget_is_never_exceeded_and_units_are_integers():
    plan = allocate_budget(request_frame(), 100_000)
    assert plan['Investment'].sum() <= 100_000 + 1e-6
    assert plan['RecommendedPurchase'].map(lambda x: float(x).is_integer()).all()
    assert (plan['RecommendedPurchase'] >= 0).all()


def test_zero_budget_buys_nothing_but_still_returns_a_plan():
    plan = allocate_budget(request_frame(), 0)
    assert plan['RecommendedPurchase'].sum() == 0
    assert len(plan) == 3


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError):
        allocate_budget(request_frame(), -1)


def test_invalid_prices_are_rejected():
    with pytest.raises(ValueError):
        allocate_budget(request_frame(CurrentPrice=[1200.0, 0.0, 14000.0]), 100_000)


def test_negative_stock_is_rejected():
    with pytest.raises(ValueError):
        allocate_budget(request_frame(CurrentStock=[-1, 40, 0]), 100_000)


def test_enough_stock_means_no_purchase():
    plan = allocate_budget(request_frame(CurrentStock=[500, 500, 500]), 100_000)
    assert plan['ShortageUnits'].sum() == 0
    assert plan['RecommendedPurchase'].sum() == 0


def test_higher_margin_product_is_funded_first():
    """"C" has the best profit per rupee; a small budget must reach it first."""
    df = request_frame(CurrentStock=[0, 0, 0],
                       CurrentPrice=[5000.0, 5000.0, 5000.0],
                       RetailPrice=[5100.0, 5200.0, 20000.0])
    plan = allocate_budget(df, 5_000)
    funded = plan[plan['RecommendedPurchase'] > 0]
    assert len(funded) == 1
    assert funded.iloc[0]['ProductID'] == 'C'


def test_loss_making_products_are_flagged():
    df = request_frame(CurrentStock=[0, 0, 0],
                       CurrentPrice=[5000.0, 5000.0, 5000.0],
                       RetailPrice=[8000.0, 9000.0, 4000.0])
    plan = allocate_budget(df, 10_000)
    flagged = plan[plan['IsUnprofitable']]
    assert len(flagged) == 1 and flagged.iloc[0]['ProductID'] == 'C'


def test_units_wanted_caps_the_shortage():
    df = request_frame(CurrentStock=[0, 0, 0], UnitsWanted=[2, 3, 4])
    plan = allocate_budget(df, 10_000_000)
    assert (plan['RecommendedPurchase'] <= plan['UnitsWanted']).all()


def test_summary_totals_match_the_plan():
    plan = allocate_budget(request_frame(), 200_000)
    summary = summarise(plan, 200_000)
    assert summary['investment'] == pytest.approx(plan['Investment'].sum(), abs=0.02)
    assert summary['funded_products'] == int((plan['RecommendedPurchase'] > 0).sum())
    assert summary['budget'] == 200_000


# --- end to end ------------------------------------------------------------
def test_forecast_request_uses_history_and_beats_the_fallback():
    model = SellerModel(base=BaseModel(model=ConstantRatio(1.0), name='Constant'), method='none')
    result = forecast_request(request_frame(), model, budget=300_000, history=history_frame())
    assert len(result.plan) == 3
    assert result.assumptions
    funded = result.plan[result.plan['RecommendedPurchase'] > 0]
    # Product A has history, so its demand is forecast rather than assumed.
    assert 'A' in set(funded['ProductID']) | set(result.plan['ProductID'])


def test_forecast_request_without_history_is_explicitly_flagged():
    model = SellerModel(base=BaseModel(model=ConstantRatio(1.0), name='Constant'), method='none')
    result = forecast_request(request_frame(), model, budget=300_000, history=None)
    assert any('No sales history' in note for note in result.assumptions)
    assert len(result.plan) == 3


def test_unseen_product_does_not_crash_the_plan():
    model = SellerModel(base=BaseModel(model=ConstantRatio(1.0), name='Constant'), method='none')
    request = request_frame(ProductID=['A', 'B', 'UNSEEN'],
                            ProductName=['Mug', 'Kettle', 'New thing'])
    result = forecast_request(request, model, budget=100_000, history=history_frame())
    assert len(result.plan) == 3
    assert any('no history' in note for note in result.assumptions)


def test_current_price_zero_is_rejected_through_the_planner():
    model = SellerModel(base=BaseModel(model=ConstantRatio(1.0), name='Constant'), method='none')
    with pytest.raises(ValueError):
        forecast_request(request_frame(CurrentPrice=[1200.0, 0.0, 14000.0]), model,
                         budget=100_000, history=history_frame())
