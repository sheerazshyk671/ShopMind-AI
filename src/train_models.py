"""
Machine Learning Training, Baseline Construction, Cross-Family Comparison,
and Evaluation Metrics Pipeline for ShopMind AI.
Strictly aligned with Sukkur IBA Machine Learning (BS-V) Course Outline Weeks.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import (
    load_and_preprocess_data, 
    perform_leakage_audit, 
    chronological_train_test_split
)

RANDOM_STATE = 42

def compute_mape(y_true, y_pred):
    """Computes Mean Absolute Percentage Error (handling zeros)."""
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class TrivialMeanBaseline:
    """
    Mandatory Trivial Baseline required by Sukkur IBA Guidelines.
    Predicts category-specific historical mean demand.
    """
    def __init__(self):
        self.category_means = {}
        self.global_mean = 0.0
        
    def fit(self, X, y):
        df_temp = X.copy()
        df_temp['target'] = y
        self.global_mean = float(y.mean())
        self.category_means = df_temp.groupby('Product_Category')['target'].mean().to_dict()
        return self
        
    def predict(self, X):
        return np.array([self.category_means.get(cat, self.global_mean) for cat in X['Product_Category']])


def build_preprocessor(categorical_cols, numerical_cols):
    """
    Constructs ColumnTransformer: OneHotEncoder for categoricals,
    StandardScaler for numerical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor


def train_and_evaluate_models():
    print("=" * 75)
    print("1. LOADING DATA AND EXECUTING LEAKAGE AUDIT")
    print("=" * 75)
    
    df = load_and_preprocess_data('ecommerce_store_subset.csv')
    leakage_info, predictor_cols, target_col = perform_leakage_audit(df)
    
    print(f"Total Rows: {len(df)}")
    print("\n[LEAKAGE AUDIT VERIFICATION]")
    for feat, reason in leakage_info.items():
        print(f"  - Dropped '{feat}': {reason}")
        
    print("\n[PREDICTOR FEATURES SELECTED (WITH ROLLING LAGS)]")
    print(" ", predictor_cols)
    print(f"Target Feature: '{target_col}' (Product Demand / Units Sold)")
    
    # Chronological Split (80% Train, 20% Test)
    train_df, test_df = chronological_train_test_split(df, split_ratio=0.8)
    
    train_start = train_df['Order_Date'].min().strftime('%Y-%m-%d')
    train_end = train_df['Order_Date'].max().strftime('%Y-%m-%d')
    test_start = test_df['Order_Date'].min().strftime('%Y-%m-%d')
    test_end = test_df['Order_Date'].max().strftime('%Y-%m-%d')
    
    print("\n[CHRONOLOGICAL SPLIT VERIFICATION]")
    print(f"  Training Set: {len(train_df)} rows ({train_start} to {train_end})")
    print(f"  Test Set:     {len(test_df)} rows ({test_start} to {test_end})")
    
    X_train = train_df[predictor_cols]
    y_train = train_df[target_col]
    X_test = test_df[predictor_cols]
    y_test = test_df[target_col]
    
    cat_cols = ['Product_Category', 'Product', 'Device_Type', 'Customer_Login_type', 'Order_Priority', 'Payment_method']
    num_cols = ['Month', 'Day', 'DayOfWeek', 'Quarter', 'IsWeekend', 'Discount', 'Lag_Product_Demand', 'Lag_Category_Demand', 'Product_Price_Tier']
    
    # -------------------------------------------------------------------------
    # Baseline Model
    # -------------------------------------------------------------------------
    baseline = TrivialMeanBaseline()
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    
    results = []
    results.append({
        'Course Outline Week': 'Course Baseline',
        'Model Family': 'Trivial Baseline',
        'Model Name': 'Category Historical Mean',
        'MAE': mean_absolute_error(y_test, y_pred_base),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_base)),
        'R² Score': r2_score(y_test, y_pred_base),
        'MAPE (%)': compute_mape(y_test.values, y_pred_base)
    })
    
    # -------------------------------------------------------------------------
    # Models Aligned with Sukkur IBA Course Outline (Weeks 2-3, 5, 10, 13)
    # -------------------------------------------------------------------------
    candidate_models = {
        'Ridge Regression (L2)': (
            'Week 5',
            'Linear / Regularized', 
            Ridge(alpha=10.0, random_state=RANDOM_STATE)
        ),
        'Decision Tree Regressor (CART)': (
            'Weeks 2–3',
            'Non-Linear Trees',
            DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
        ),
        'KNN Regressor (k=9)': (
            'Week 10',
            'Instance-based / Distance',
            KNeighborsRegressor(n_neighbors=9)
        ),
        'Random Forest Regressor': (
            'Week 13',
            'Ensemble (Bagging)', 
            RandomForestRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
        ),
        'XGBoost Regressor': (
            'Week 13',
            'Ensemble (Boosting)', 
            XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=RANDOM_STATE, n_jobs=-1)
        )
    }
    
    trained_pipelines = {}
    test_predictions = {'Actual': y_test.values, 'Baseline': y_pred_base}
    
    print("\n" + "=" * 75)
    print("2. TRAINING COURSE OUTLINE MACHINE LEARNING MODELS")
    print("=" * 75)
    
    for name, (week, family, regressor) in candidate_models.items():
        print(f"Training {name} [{week} | {family}]...")
        pipe = Pipeline([
            ('preprocessor', build_preprocessor(cat_cols, num_cols)),
            ('regressor', regressor)
        ])
        
        # Fit on training data ONLY
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        # Clip negative demand predictions to 0
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        
        trained_pipelines[name] = pipe
        test_predictions[name] = y_pred
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = compute_mape(y_test.values, y_pred)
        
        results.append({
            'Course Outline Week': week,
            'Model Family': family,
            'Model Name': name,
            'MAE': mae,
            'RMSE': rmse,
            'R² Score': r2,
            'MAPE (%)': mape
        })
        
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 75)
    print("3. MODEL EVALUATION COMPARISON TABLE (Sukkur IBA Guideline Section 5)")
    print("=" * 75)
    print(results_df.to_string(index=False))
    
    # Save evaluation table
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/model_comparison_results.csv', index=False)
    
    # Select Best Model based on Highest R2 and Lowest MAE/RMSE
    # Filter out baseline
    ml_only = results_df[results_df['Model Family'] != 'Trivial Baseline']
    best_model_name = ml_only.sort_values(by=['R² Score', 'MAE'], ascending=[False, True]).iloc[0]['Model Name']
    best_pipe = trained_pipelines[best_model_name]
    
    print(f"\n[WINNING MODEL]: {best_model_name}")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipe, 'models/best_sales_model.joblib')
    print("Saved winning model artifact to 'models/best_sales_model.joblib'")
    
    # -------------------------------------------------------------------------
    # Error Analysis (Sukkur IBA Guideline Section 6)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("4. ERROR ANALYSIS: EXTREME RESIDUAL OUTLIERS (Guideline Section 6)")
    print("=" * 75)
    best_pred = test_predictions[best_model_name]
    residuals = np.abs(y_test.values - best_pred)
    
    error_analysis_df = test_df.copy()
    error_analysis_df['Predicted_Quantity'] = np.round(best_pred, 2)
    error_analysis_df['Absolute_Residual'] = np.round(residuals, 2)
    
    worst_cases = error_analysis_df.sort_values(by='Absolute_Residual', ascending=False).head(8)
    worst_summary = worst_cases[[
        'Order_Date', 'Product_Category', 'Product', 'Quantity', 
        'Predicted_Quantity', 'Absolute_Residual', 'Discount'
    ]]
    print(worst_summary.to_string(index=False))
    worst_summary.to_csv('reports/error_analysis_cases.csv', index=False)
    
    return results_df, worst_summary

if __name__ == '__main__':
    train_and_evaluate_models()
