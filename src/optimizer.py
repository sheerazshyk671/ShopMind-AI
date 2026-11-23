"""
Smart Inventory Budget Allocation Engine (Prescriptive Layer) for ShopMind AI.
Includes Date Awareness, Seasonality Dynamics, and Festive / Event Modeling:
- Pre-Eid / Ramadan Shopping Rush (Ethnic Wear, Suits, Shoes, Watches, Host Goods Surge)
- Late Season / Winter Clearance (Anti-Dead-Stock Decay Factor)
- Winter Wedding Season (Formal Suits, Crockery, Bedding Surge)
- Summer Peak (Cooling Appliances, T-Shirts Surge)
"""

import datetime
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import milp, LinearConstraint, Bounds

def map_product_title_to_family(title):
    """
    Maps specific commercial titles (e.g. 'Levi's Slim Fit Stretch Denim Jeans')
    to the trained machine learning product family (e.g. 'Jeans').
    """
    t = str(title).lower()
    if 't-shirt' in t or 't -' in t or 'tee' in t:
        return 'T - Shirts'
    elif 'shirt' in t:
        return 'Shirts'
    elif 'jean' in t or 'denim' in t:
        return 'Jeans'
    elif 'suit' in t:
        return 'Suits'
    elif 'fossil' in t or 'watch' in t:
        return 'Fossil Watch'
    elif 'running' in t:
        return 'Running Shoes'
    elif 'shoe' in t or 'sneaker' in t:
        return 'Casula Shoes'
    elif 'sofa' in t:
        return 'Sofas'
    elif 'sheet' in t:
        return 'Bed Sheets'
    elif 'curtain' in t:
        return 'Curtains'
    elif 'crockery' in t or 'dinner' in t:
        return 'Dinner Crockery'
    elif 'speaker' in t:
        return 'Speakers'
    elif 'led' in t or 'tv' in t:
        return 'LED'
    elif 'mobile' in t:
        return 'Samsung Mobile'
    elif 'fan' in t:
        return 'Fans'
    elif 'iron' in t:
        return 'Iron'
    elif 'rack' in t:
        return 'Shoe Rack'
    return 'Shirts'


def calculate_seasonal_event_factors(categories, product_names, month, day, event):
    """
    Computes domain-aware seasonal and festive multipliers for Pakistani e-commerce:
    - Pre-Eid / Ramadan: Dramatic spike in Shalwar Kameez/Suits, Footwear, Watches, Bedding
    - Late Winter / Clearance: Decay penalty to avoid carrying dead winter stock into Spring
    - Wedding Season (Nov-Feb): Heavy demand for formal suits, dinner crockery, bedding
    - Summer Peak: Fans, Juicers, Light T-shirts surge; heavy fabrics decline
    """
    multipliers = []
    descriptions = []
    
    event_str = str(event).lower()
    
    for cat, prod in zip(categories, product_names):
        p_lower = str(prod).lower()
        mult = 1.0
        desc = "Normal Seasonal Baseline (1.0x)"
        
        # 1. Pre-Eid / Ramadan Festival Rush
        if 'eid' in event_str or 'ramadan' in event_str:
            if any(k in p_lower for k in ['suit', 'shirt', 'watch', 'shoe', 'sneaker', 't-shirt']):
                mult = 2.10
                desc = "🔥 Pre-Eid Festival Surge (+110%)"
            elif any(k in p_lower for k in ['bed', 'crockery', 'curtain']):
                mult = 1.50
                desc = "✨ Eid Hospitality & Hosting Prep (+50%)"
            else:
                mult = 1.15
                desc = "📈 General Festive Retail Lift (+15%)"
                
        # 2. Late Season / End of Season Clearance (e.g. Half of winter passed)
        elif 'late' in event_str or 'clearance' in event_str or (month in [1, 2] and day > 15):
            if any(k in p_lower for k in ['suit', 'sofa', 'leather', 'heavy']):
                mult = 0.45
                desc = "❄️ Late Season Decay (-55% Anti-Dead-Stock)"
            elif any(k in p_lower for k in ['fan', 'juicer', 'blender']):
                mult = 0.35
                desc = "❄️ Deep Winter Off-Season (-65%)"
            else:
                mult = 0.85
                desc = "⚠️ Conservative End-of-Cycle Capping (-15%)"
                
        # 3. Winter Wedding Season (Nov - Feb)
        elif 'wedding' in event_str or (month in [11, 12, 1] and 'summer' not in event_str):
            if any(k in p_lower for k in ['suit', 'crockery', 'sheet', 'shoe', 'watch']):
                mult = 1.70
                desc = "💍 Winter Wedding Season Surge (+70%)"
            elif any(k in p_lower for k in ['fan']):
                mult = 0.30
                desc = "❄️ Winter Cooling Inactivity (-70%)"
            else:
                mult = 1.10
                desc = "Winter Retail Trend (+10%)"
                
        # 4. Summer Peak Season (May - Aug)
        elif 'summer' in event_str or month in [5, 6, 7, 8]:
            if any(k in p_lower for k in ['t-shirt', 'fan', 'juicer', 'blender']):
                mult = 1.85
                desc = "☀️ Summer Peak Demand (+85%)"
            elif any(k in p_lower for k in ['suit', 'heavy', 'leather']):
                mult = 0.60
                desc = "☀️ Warm Weather Fabric Slowdown (-40%)"
            else:
                mult = 1.05
                desc = "Summer Baseline (+5%)"
        else:
            mult = 1.0
            desc = "Regular Standard Baseline (1.0x)"
            
        multipliers.append(mult)
        descriptions.append(desc)
        
    return np.array(multipliers), descriptions


def allocate_budget(
    candidate_df, 
    total_budget_pkr, 
    model_pipeline_path='models/best_sales_model.joblib', 
    strategy='balanced', 
    procurement_date=None,
    special_event='Regular Business Cycle',
    horizon_days=30,
    *args, 
    **kwargs
):
    """
    Inputs:
      - candidate_df: DataFrame containing:
          ['product_id', 'product_name', 'category', 'wholesale_cost_pkr', 'selling_price_pkr', 'current_stock_in_hand']
      - total_budget_pkr: float (e.g. 500,000 PKR)
      - model_pipeline_path: path to serialized ML pipeline
      - strategy: 'balanced' (Default)
      - procurement_date: date object or string (defaults to today)
      - special_event: Timing scenario (Eid, Late Winter, Wedding, Summer, Regular)
      - horizon_days: inventory forecasting cycle (default: 30 days)
      
    Outputs:
      - results_df: Recommended purchase units per item, total cost, projected profit, ROI
      - summary_dict: KPI metrics for the executive dashboard
    """
    df = candidate_df.copy().reset_index(drop=True)
    
    # Standardize stock column name defensively
    stock_col = None
    for candidate in ['current_stock_in_hand', 'current_stock', 'supplier_stock', 'stock', 'inventory']:
        if candidate in df.columns:
            stock_col = candidate
            break
            
    if stock_col is not None:
        df['current_stock_in_hand'] = pd.to_numeric(df[stock_col], errors='coerce').fillna(0).astype(int)
    else:
        df['current_stock_in_hand'] = 0
        
    # Date Handling
    if procurement_date is None:
        target_date = pd.Timestamp.now()
    else:
        target_date = pd.to_datetime(procurement_date)
        
    month = int(target_date.month)
    day = int(target_date.day)
    day_of_week = int(target_date.dayofweek)
    quarter = int(target_date.quarter)
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Load ML Model Pipeline
    model = joblib.load(model_pipeline_path)
    
    # Map specific retail product titles to ML trained families
    df['model_product_family'] = df['product_name'].apply(map_product_title_to_family)
    
    # Category average historical demand estimates
    cat_avg = {'Fashion': 2.48, 'Electronic': 2.50, 'Home & Furniture': 2.45}
    lag_cat = df['category'].map(cat_avg).fillna(2.48).values
    
    # Typical price tier based on wholesale cost
    typical_price = df['wholesale_cost_pkr'].values * 1.6
    
    # Prepare inference feature frame with EXACT DYNAMIC CALENDAR ATTRIBUTES
    inference_features = pd.DataFrame({
        'Month': [month] * len(df),
        'Day': [day] * len(df),
        'DayOfWeek': [day_of_week] * len(df),
        'Quarter': [quarter] * len(df),
        'IsWeekend': [is_weekend] * len(df),
        'Product_Category': df['category'],
        'Product': df['model_product_family'],
        'Device_Type': ['Web'] * len(df),
        'Customer_Login_type': ['Member'] * len(df),
        'Order_Priority': ['Medium'] * len(df),
        'Payment_method': ['credit_card'] * len(df),
        'Discount': [0.10] * len(df),
        'Lag_Product_Demand': [2.5] * len(df),
        'Lag_Category_Demand': lag_cat,
        'Product_Price_Tier': typical_price
    })
    
    # Predict daily order demand rate from trained ML model
    predicted_daily_rate = model.predict(inference_features)
    predicted_daily_rate = np.clip(predicted_daily_rate, a_min=0.5, a_max=None)
    
    # Base category velocity (Monthly absorption benchmark)
    category_velocity = {'Fashion': 18.0, 'Electronic': 10.0, 'Home & Furniture': 12.0}
    base_multipliers = df['category'].map(category_velocity).fillna(12.0).values
    
    # Calculate Dynamic Seasonal & Event Multipliers
    event_multipliers, event_descriptions = calculate_seasonal_event_factors(
        df['category'], df['product_name'], month, day, special_event
    )
    
    # Combined Demand Horizon: ML Daily Rate * Base Velocity * Event Multiplier
    df['Seasonal_Event_Multiplier'] = np.round(event_multipliers, 2)
    df['Seasonal_Impact'] = event_descriptions
    
    df['Predicted_Demand_Horizon'] = np.maximum(
        1, 
        np.round(predicted_daily_rate * base_multipliers * event_multipliers).astype(int)
    )
    
    # Calculate Profit per Unit (Selling Price - Wholesale Cost)
    df['Unit_Profit_PKR'] = df['selling_price_pkr'] - df['wholesale_cost_pkr']
    
    # Net Restock Needed = max(0, Forecasted Demand - Current Stock in Hand)
    df['Net_Restock_Needed'] = np.maximum(
        0, 
        df['Predicted_Demand_Horizon'] - df['current_stock_in_hand']
    )
    
    num_items = len(df)
    profits = df['Unit_Profit_PKR'].values.astype(float)
    costs = df['wholesale_cost_pkr'].values.astype(float)
    net_needed = df['Net_Restock_Needed'].values.astype(float)
    
    # -------------------------------------------------------------------------
    # Optimization Strategy Configuration (Balanced Store Replenishment)
    # -------------------------------------------------------------------------
    # 1. No single item can consume more than 25% of total budget
    max_spend_per_item = total_budget_pkr * 0.25
    upper_bounds = np.minimum(net_needed, np.floor(max_spend_per_item / np.maximum(costs, 1))).astype(int)
    
    # 2. Minimum Safe Allocation: At least 20% of net needed restock if cost allows
    min_safe = np.maximum(1, np.floor(net_needed * 0.20)).astype(int)
    lower_bounds = np.minimum(min_safe, upper_bounds).astype(int)
    
    # Adjust lower bounds if they exceed budget
    while np.sum(lower_bounds * costs) > total_budget_pkr and np.any(lower_bounds > 0):
        idx_max = np.argmax(lower_bounds * costs)
        lower_bounds[idx_max] = max(0, lower_bounds[idx_max] - 1)
    
    # Objective: Maximize Profit => Minimize -Profit in SciPy MILP
    c = -profits
    integrality = np.ones(num_items)
    bounds = Bounds(lb=lower_bounds, ub=upper_bounds)
    constraints = LinearConstraint(A=costs.reshape(1, -1), lb=0, ub=float(total_budget_pkr))
    
    # Solve MILP
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    
    if res.success:
        recommended_units = np.round(res.x).astype(int)
        status_str = "Optimal Seasonal Replenishment Found"
    else:
        # Robust Greedy Allocation Fallback
        recommended_units = lower_bounds.copy()
        current_budget = float(total_budget_pkr - np.sum(recommended_units * costs))
        ratio = profits / np.maximum(costs, 1)
        sorted_indices = np.argsort(-ratio)
        for idx in sorted_indices:
            cost_i = costs[idx]
            space_left = int(upper_bounds[idx] - recommended_units[idx])
            if space_left > 0 and current_budget >= cost_i:
                can_buy = int(min(space_left, current_budget // cost_i))
                recommended_units[idx] += can_buy
                current_budget -= can_buy * cost_i
        status_str = "Greedy Seasonal Replenishment"
        
    df['Recommended_Purchase_Units'] = recommended_units
    df['Allocated_Cost_PKR'] = df['Recommended_Purchase_Units'] * df['wholesale_cost_pkr']
    df['Projected_Profit_PKR'] = df['Recommended_Purchase_Units'] * df['Unit_Profit_PKR']
    
    # Calculate Summary KPIs
    total_spent = float(df['Allocated_Cost_PKR'].sum())
    remaining_budget = float(total_budget_pkr - total_spent)
    total_projected_profit = float(df['Projected_Profit_PKR'].sum())
    roi_percentage = (total_projected_profit / total_spent * 100) if total_spent > 0 else 0.0
    total_units_bought = int(df['Recommended_Purchase_Units'].sum())
    
    summary = {
        'Total_Budget_PKR': float(total_budget_pkr),
        'Total_Spent_PKR': total_spent,
        'Remaining_Budget_PKR': remaining_budget,
        'Projected_Net_Profit_PKR': total_projected_profit,
        'Expected_ROI_Percent': round(roi_percentage, 2),
        'Total_Units_Procured': total_units_bought,
        'Status': status_str,
        'Planning_Date': target_date.strftime('%Y-%m-%d'),
        'Special_Event': special_event
    }
    
    return df, summary
