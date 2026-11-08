"""Tests for the real Pakistani dataset pipeline (run on small synthetic fixtures).

The real export is 106 MB and not committed, so these tests build a tiny frame with
the same schema. That keeps the cleaning rules under test without shipping the data.
"""
import numpy as np
import pandas as pd
import pytest

from src.pakistan_data import (
    DROP_MONTHS,
    SALE_STATUSES,
    category_panel,
    clean,
    load_raw,
    seasonality_report,
    sku_panel,
)


def raw_frame() -> pd.DataFrame:
    """Two real months, a few SKUs, mixed statuses, plus Excel junk columns."""
    rows = [
        # kept: complete order
        ('1', 'complete', '7/1/2017', 'SKU_A', 1000, 1, 1000, '1001', 'Mobiles & Tablets', 'cod'),
        ('2', 'complete', '7/5/2017', 'SKU_A', 1000, 2, 2000, '1002', 'Mobiles & Tablets', 'cod'),
        ('3', 'received', '7/9/2017', 'SKU_B', 500, 1, 500, '1003', 'Beauty & Grooming', 'cod'),
        ('4', 'complete', '8/3/2017', 'SKU_A', 1000, 3, 3000, '1004', 'Mobiles & Tablets', 'cod'),
        ('5', 'complete', '8/7/2017', 'SKU_B', 500, 1, 500, '1005', 'Beauty & Grooming', 'cod'),
        # excluded: cancellations and refunds
        ('6', 'canceled', '7/11/2017', 'SKU_A', 1000, 5, 5000, '1006', 'Mobiles & Tablets', 'cod'),
        ('7', 'refund', '7/12/2017', 'SKU_B', 500, 2, 1000, '1007', 'Beauty & Grooming', 'cod'),
        ('8', '\\N', '7/13/2017', 'SKU_C', 700, 1, 700, '1008', 'Superstore', 'cod'),
        # invalid quantity
        ('9', 'complete', '7/14/2017', 'SKU_A', 1000, 0, 0, '1009', 'Mobiles & Tablets', 'cod'),
    ]
    frame = pd.DataFrame(rows, columns=['item_id', 'status', 'created_at', 'sku', 'price',
                                        'qty_ordered', 'grand_total', 'increment_id',
                                        'category_name_1', 'payment_method'])
    frame['Unnamed: 1'] = np.nan
    frame['Unnamed: 2'] = np.nan
    return frame


def test_blank_rows_and_unnamed_columns_are_dropped():
    frame = pd.concat([raw_frame(), pd.DataFrame([{c: np.nan for c in raw_frame().columns}])],
                      ignore_index=True)
    frame.to_csv('/tmp/_pk_fixture.csv', index=False)
    loaded, audit = load_raw('/tmp/_pk_fixture.csv')
    assert len(loaded) == 9                      # the all-NaN row is gone
    assert not [c for c in loaded.columns if c.startswith('Unnamed')]
    assert audit['columns_dropped'] == ['Unnamed: 1', 'Unnamed: 2']


def test_cancelled_and_refunded_rows_are_flagged_not_silently_kept_as_sales():
    """Returns must stay in the frame (to be measured) but never count as sales."""
    audit = {}
    sales = clean(raw_frame(), audit)

    assert set(sales.loc[sales['is_sale'], 'status']).issubset(set(SALE_STATUSES))
    assert not sales.loc[sales['is_sale'], 'status'].isin(['canceled', 'refund', '\\N']).any()
    # They are retained and flagged so the return rate can be computed.
    assert sales['is_return'].sum() == 2                       # canceled + refund
    assert set(sales.loc[sales['is_return'], 'status']) == {'canceled', 'refund'}
    assert sales['is_unresolved'].sum() == 1                   # the \\N row
    assert set(sales.loc[sales['is_unresolved'], 'status']) == {'\\N'}
    assert audit['rows_excluded'] == 3
    assert audit['rows_kept_as_sales'] == 5


def test_zero_quantity_rows_are_dropped():
    audit = {}
    sales = clean(raw_frame(), audit)
    assert (sales['qty_ordered'] > 0).all()
    assert audit['rows_dropped_bad_quantity_or_value'] == 1


def test_category_panel_has_one_row_per_category_month():
    sales = clean(raw_frame(), {})
    panel = category_panel(sales)
    assert panel.duplicated(subset=['category', 'month_date']).sum() == 0
    mobiles = panel[(panel['category'] == 'Mobiles & Tablets') &
                    (panel['month_date'] == pd.Timestamp('2017-07-01'))].iloc[0]
    assert mobiles['units_sold'] == 3            # 1 + 2 units from the two complete orders
    assert mobiles['orders'] == 2


def test_return_rate_is_computable_and_bounded():
    sales = clean(raw_frame(), {})
    panel = category_panel(sales)
    assert panel['return_rate'].between(0, 1).all()
    mobiles_july = panel[(panel['category'] == 'Mobiles & Tablets') &
                         (panel['month_date'] == pd.Timestamp('2017-07-01'))].iloc[0]
    # 5 cancelled units against 3 sold units -> 5/8
    assert mobiles_july['refunded_units'] == 5
    assert mobiles_july['return_rate'] == pytest.approx(5 / 8)


def test_partial_months_are_trimmed_from_the_panel():
    frame = raw_frame()
    extra = frame.iloc[[0]].copy()
    extra['created_at'] = '7/1/2016'             # the partial first month
    sales = clean(pd.concat([frame, extra], ignore_index=True), {})
    panel = category_panel(sales)
    assert panel['month_date'].dt.strftime('%Y-%m').isin(DROP_MONTHS).sum() == 0


def test_sku_panel_keeps_only_persistent_products():
    sales = clean(raw_frame(), {})
    # With min_months=2 only SKU_A and SKU_B survive; both appear in July and August.
    panel, stats = sku_panel(sales, min_months=2)
    assert set(panel['sku']) == {'SKU_A', 'SKU_B'}
    assert stats['skus_with_2+_months'] == 2
    assert 'warning' in stats


def test_sku_panel_reports_coverage_rather_than_hiding_it():
    sales = clean(raw_frame(), {})
    _, stats = sku_panel(sales, min_months=2)
    for key in ('median_months_per_sku', 'unit_coverage_pct', 'revenue_coverage_pct'):
        assert key in stats
    assert 0 <= stats['unit_coverage_pct'] <= 100


def test_seasonality_report_measures_rather_than_asserts():
    sales = clean(raw_frame(), {})
    report = seasonality_report(sales)
    assert 'november_revenue_ratio' in report
    assert 'reading' in report
    # Only two months of fixture data: the report must not crash on missing Novembers.
    assert report['november_revenue_ratio'] is None or isinstance(report['november_revenue_ratio'], float)
