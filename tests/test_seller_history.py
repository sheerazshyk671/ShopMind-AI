import numpy as np
import pandas as pd
import pytest
from src.seller_history import features_from_history, calibrate_seller, history_columns
from src.forecasting import forecast_seller
from src.optimization import optimize_budget


class ConstantModel:
    def predict(self, X):
        return np.ones(len(X))


def sample():
    return pd.DataFrame({'ProductID':['A'], 'ProductName':['A'], 'CurrentPrice':[10], 'CurrentStock':[0], **{f'Sales_{i:02d}':[i] for i in range(1,13)}})


def test_features_and_safety_stock_use_real_history():
    df = sample()
    f = features_from_history(df, '2026-12-01')
    assert f.lag_1.iloc[0] == 12
    assert f.lag_3.iloc[0] == 10
    assert f.lag_6.iloc[0] == 7
    assert f.rolling_mean_3.iloc[0] == 11
    forecast = forecast_seller(df, ConstantModel(), '2026-12-01')
    assert np.isclose(forecast.SafetyStock.iloc[0], 1.28*np.std(np.arange(7,13), ddof=1))
    assert forecast.attrs['calibration_samples'] == 6
    assert optimize_budget(forecast, 100).Investment.sum() <= 100


def test_six_month_history_needs_no_calibration_and_gap_rejected():
    df = sample().drop(columns=[f'Sales_{i:02d}' for i in range(7,13)])
    assert calibrate_seller(df, ConstantModel(), '2026-12-01') == (1.0, 0)
    with pytest.raises(ValueError):
        history_columns(df.drop(columns=['Sales_02']))
