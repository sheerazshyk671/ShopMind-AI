"""
Data Preprocessing, Cleaning, Leakage Audit and Chronological Splitting Module
for ShopMind AI.
Includes rolling lag features and temporal feature engineering.
"""

import pandas as pd
import numpy as np

# Exchange rate / PKR scaling factor: 1 USD ≈ 280 PKR
PKR_SCALE = 280.0

def load_and_preprocess_data(filepath='ecommerce_store_subset.csv'):
    """
    Loads raw dataset, scales monetary fields to PKR, extracts temporal features,
    and engineers rolling historical demand lags without future leakage.
    """
    df = pd.read_csv(filepath)
    
    # Ensure correct datetime parsing
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Time'] = df['Time'].fillna('12:00:00')
    
    # Sort strictly chronologically by Date and Time
    df = df.sort_values(by=['Order_Date', 'Time']).reset_index(drop=True)
    
    # Scale monetary variables to realistic PKR values
    df['Sales_PKR'] = df['Sales'] * PKR_SCALE
    df['Profit_PKR'] = df['Profit'] * PKR_SCALE
    
    # Extract temporal features (No future leakage)
    df['Year'] = df['Order_Date'].dt.year
    df['Month'] = df['Order_Date'].dt.month
    df['Day'] = df['Order_Date'].dt.day
    df['DayOfWeek'] = df['Order_Date'].dt.day_of_week
    df['Quarter'] = df['Order_Date'].dt.quarter
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Categorical handling: fill missing
    df['Device_Type'] = df['Device_Type'].fillna('Web')
    df['Customer_Login_type'] = df['Customer_Login_type'].fillna('Member')
    df['Product_Category'] = df['Product_Category'].fillna('Fashion')
    df['Product'] = df['Product'].fillna('General Product')
    df['Order_Priority'] = df['Order_Priority'].fillna('Medium')
    df['Payment_method'] = df['Payment_method'].fillna('credit_card')
    df['Discount'] = df['Discount'].fillna(0.0)
    
    # -------------------------------------------------------------------------
    # Scientific Feature Engineering: Historical Rolling Lags (Non-leaky)
    # -------------------------------------------------------------------------
    # Lag 1: Shifted expanding average demand per product (past sales rate)
    df['Lag_Product_Demand'] = df.groupby('Product')['Quantity'].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(df['Quantity'].mean())

    # Lag 2: Shifted expanding average demand per category
    df['Lag_Category_Demand'] = df.groupby('Product_Category')['Quantity'].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(df['Quantity'].mean())

    # Historical product typical price tier (using past orders median, shifted)
    product_price_tier = df.groupby('Product')['Sales_PKR'].transform(
        lambda s: s.shift(1).expanding().median()
    ).fillna(15000.0)
    df['Product_Price_Tier'] = product_price_tier
    
    return df


def perform_leakage_audit(df):
    """
    Explicit Leakage Audit required by Sukkur IBA Guidelines.
    Identifies features that CANNOT be known prior to or during order placement.
    """
    leakage_features = {
        'Profit': 'Post-transaction accounting metric; directly defines Sales algebraically if retained.',
        'Profit_PKR': 'Scaled version of post-transaction profit.',
        'Sales': 'Direct monetary total of the current transaction line item.',
        'Sales_PKR': 'Scaled version of direct transaction sales.',
        'Aging': 'Delivery duration in days; only known after product delivery completes.',
        'Shipping_Cost': 'Carrier shipping cost finalized post-fulfillment.'
    }
    
    # Predictor features strictly verified safe from leakage
    predictor_columns = [
        'Month', 'Day', 'DayOfWeek', 'Quarter', 'IsWeekend',
        'Product_Category', 'Product', 'Device_Type', 
        'Customer_Login_type', 'Order_Priority', 'Payment_method', 'Discount',
        'Lag_Product_Demand', 'Lag_Category_Demand', 'Product_Price_Tier'
    ]
    
    target_column = 'Quantity' # Demand / Units sold
    
    return leakage_features, predictor_columns, target_column


def chronological_train_test_split(df, split_ratio=0.8):
    """
    Splits dataset chronologically: first 80% dates for Training, last 20% for Testing.
    Zero random shuffle to satisfy course time-series validation rules.
    """
    df_sorted = df.sort_values(by=['Order_Date', 'Time']).reset_index(drop=True)
    split_idx = int(len(df_sorted) * split_ratio)
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    return train_df, test_df
