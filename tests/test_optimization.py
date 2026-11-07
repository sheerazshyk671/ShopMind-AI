import pandas as pd
import pytest
from src.optimization import optimize_budget


def test_budget_never_exceeded_and_integer_purchases():
    df = pd.DataFrame({'ForecastedDemand':[20, 12, 5], 'CurrentStock':[1, 0, 9], 'CurrentPrice':[10.1, 4.2, 3]})
    result = optimize_budget(df, 100)
    assert result.Investment.sum() <= 100
    assert (result.RecommendedPurchase <= result.ShortageUnits).all()
    assert result.RecommendedPurchase.map(lambda x: int(x) == x).all()
    assert result.loc[result.CurrentStock == 9, 'RecommendedPurchase'].iloc[0] == 0


def test_zero_budget_and_fractional_shortage():
    df = pd.DataFrame({'ForecastedDemand':[2.1], 'CurrentStock':[1], 'CurrentPrice':[5]})
    assert optimize_budget(df, 0).RecommendedPurchase.iloc[0] == 0
    assert optimize_budget(df, 10).RecommendedPurchase.iloc[0] == 2


def test_invalid_price_rejected():
    with pytest.raises(ValueError):
        optimize_budget(pd.DataFrame({'ForecastedDemand':[2], 'CurrentStock':[0], 'CurrentPrice':[0]}), 10)
