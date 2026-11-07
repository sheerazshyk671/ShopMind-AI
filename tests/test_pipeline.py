import numpy as np
import pandas as pd
from src.preprocessing import build_monthly_panel
from src.features import make_features


def test_panel_zero_fills_and_features_do_not_see_future():
    months = pd.date_range('2020-01-01', periods=14, freq='MS')
    df = pd.DataFrame({'InvoiceNo': ['1']*13, 'StockCode':['A']*13, 'Description':['Product']*13, 'Quantity':list(range(1, 7))+list(range(8, 15)), 'InvoiceDate':months.delete(6), 'UnitPrice':[1]*13})
    panel = build_monthly_panel(df)
    assert panel.loc[panel.month == months[6], 'monthly_units_sold'].iloc[0] == 0
    features = make_features(panel)
    row = features.loc[features.month_date == months[7]].iloc[0]
    assert row.lag_1 == 0
    assert np.isclose(row.rolling_mean_3, (5+6+0)/3)
    assert np.isclose(row.target_ratio, 8/((5+6+0)/3))
