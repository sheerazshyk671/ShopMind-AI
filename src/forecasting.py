"""Train and evaluate demand-ratio models with a chronological split."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features import FEATURE_COLUMNS, make_features, seller_features


def train_models(panel, output='models/demand_model.pkl'):
    data = make_features(panel)
    months = sorted(data['month_date'].unique())
    if len(months) < 2:
        raise ValueError('At least two feature-ready months are needed for a chronological split.')
    split = max(1, min(len(months)-1, int(len(months)*0.8)))
    train = data[data['month_date'] < months[split]]
    test = data[data['month_date'] >= months[split]]
    models = {'Linear Regression': LinearRegression(), 'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=2), 'Gradient Boosting': GradientBoostingRegressor(random_state=42)}
    scores = {}
    best, best_mae = None, float('inf')
    for name, model in models.items():
        model.fit(train[FEATURE_COLUMNS], train['target_ratio'])
        predicted = model.predict(test[FEATURE_COLUMNS])
        scores[name] = {'MAE': mean_absolute_error(test['target_ratio'], predicted), 'RMSE': np.sqrt(mean_squared_error(test['target_ratio'], predicted)), 'R2': r2_score(test['target_ratio'], predicted) if len(test) > 1 else float('nan')}
        if scores[name]['MAE'] < best_mae:
            best, best_mae = model, scores[name]['MAE']
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, output)
    return scores


def forecast_seller(products, model, forecast_month=None):
    from src.seller_history import history_columns, features_from_history, calibrate_seller
    result = products.copy()
    date = pd.Timestamp(forecast_month or (pd.Timestamp.today().to_period('M') + 1).to_timestamp())
    if history_columns(result):
        features = features_from_history(result, date)
        calibration, samples = calibrate_seller(result, model, date)
        baseline = features['rolling_mean_3'].to_numpy()
    else:
        features = seller_features(result, date)
        calibration, samples = 1.0, 0
        baseline = result['RecentMonthlySales'].to_numpy()
    ratios = np.maximum(0, model.predict(features))
    result['ForecastedDemand'] = ratios * baseline * calibration
    result['SafetyStock'] = 1.28 * features['rolling_std_6'].to_numpy()
    result.attrs['calibration'] = calibration
    result.attrs['calibration_samples'] = samples
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', default='data/processed/monthly_panel.csv')
    parser.add_argument('--output', default='models/demand_model.pkl')
    args = parser.parse_args()
    scores = train_models(pd.read_csv(args.panel), args.output)
    print(pd.DataFrame(scores).T.to_string())
    print(f'Saved best model by validation MAE to {args.output}')

if __name__ == '__main__':
    main()
