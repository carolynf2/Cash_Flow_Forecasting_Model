import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class CashFlowPredictor:
    def __init__(self, db_path="cashflow_forecast.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def build_forecasting_models(self):
        """Build and train multiple forecasting models"""
        # Get historical data
        historical_data = self._prepare_historical_data()
        
        if historical_data.empty:
            return {'models_built': False, 'error': 'Insufficient historical data'}
        
        # Prepare features and targets
        features, targets = self._prepare_features_targets(historical_data)
        
        if len(features) < 10:  # Need minimum data points
            return {'models_built': False, 'error': 'Insufficient data points for modeling'}
        
        # Split data for training and validation
        train_size = int(len(features) * 0.8)
        X_train, X_test = features[:train_size], features[train_size:]
        y_train, y_test = targets[:train_size], targets[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Build multiple models
        models_performance = {}
        
        # 1. Linear Regression Model
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        
        self.models['linear_regression'] = lr_model
        models_performance['linear_regression'] = {
            'mae': float(lr_mae),
            'rmse': float(np.sqrt(mean_squared_error(y_test, lr_pred)))
        }
        
        # 2. Random Forest Model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        
        self.models['random_forest'] = rf_model
        models_performance['random_forest'] = {
            'mae': float(rf_mae),
            'rmse': float(np.sqrt(mean_squared_error(y_test, rf_pred)))
        }
        
        # Store feature importance
        feature_names = self._get_feature_names()
        self.feature_importance['random_forest'] = dict(zip(feature_names, rf_model.feature_importances_))
        
        # 3. Ensemble Model (simple average)
        ensemble_pred = (lr_pred + rf_pred) / 2
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        
        models_performance['ensemble'] = {
            'mae': float(ensemble_mae),
            'rmse': float(np.sqrt(mean_squared_error(y_test, ensemble_pred)))
        }
        
        # Select best model
        best_model = min(models_performance.keys(), key=lambda x: models_performance[x]['mae'])
        
        return {
            'models_built': True,
            'models_performance': models_performance,
            'best_model': best_model,
            'training_data_points': len(X_train),
            'validation_data_points': len(X_test),
            'feature_importance': self.feature_importance
        }
    
    def _prepare_historical_data(self):
        """Prepare historical data for modeling"""
        conn = sqlite3.connect(self.db_path)
        
        # Get historical cash flows aggregated by day
        query = """
            SELECT 
                transaction_date,
                SUM(CASE WHEN transaction_type = 'inflow' THEN amount ELSE 0 END) as daily_inflows,
                SUM(CASE WHEN transaction_type = 'outflow' THEN amount ELSE 0 END) as daily_outflows,
                COUNT(*) as transaction_count,
                SUM(CASE WHEN transaction_type = 'inflow' THEN amount ELSE -amount END) as net_flow
            FROM cash_transactions
            WHERE transaction_date >= date('now', '-365 days')
            GROUP BY transaction_date
            ORDER BY transaction_date
        """
        
        daily_data = pd.read_sql_query(query, conn)
        conn.close()
        
        if daily_data.empty:
            return pd.DataFrame()
        
        daily_data['transaction_date'] = pd.to_datetime(daily_data['transaction_date'])
        
        # Fill missing dates with zeros
        date_range = pd.date_range(start=daily_data['transaction_date'].min(), 
                                 end=daily_data['transaction_date'].max(), 
                                 freq='D')
        
        complete_data = daily_data.set_index('transaction_date').reindex(date_range, fill_value=0)
        complete_data.index.name = 'date'
        complete_data = complete_data.reset_index()
        
        return complete_data
    
    def _prepare_features_targets(self, data):
        """Prepare features and targets for machine learning"""
        # Create time-based features
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        data['quarter'] = data['date'].dt.quarter
        data['is_month_end'] = (data['date'].dt.day > 25).astype(int)
        data['is_quarter_end'] = ((data['date'].dt.month % 3 == 0) & (data['date'].dt.day > 25)).astype(int)
        
        # Create lag features (previous days' values)
        for lag in [1, 2, 3, 7, 14, 30]:
            data[f'inflow_lag_{lag}'] = data['daily_inflows'].shift(lag)
            data[f'outflow_lag_{lag}'] = data['daily_outflows'].shift(lag)
            data[f'net_flow_lag_{lag}'] = data['net_flow'].shift(lag)
        
        # Create rolling average features
        for window in [7, 14, 30]:
            data[f'inflow_ma_{window}'] = data['daily_inflows'].rolling(window=window).mean()
            data[f'outflow_ma_{window}'] = data['daily_outflows'].rolling(window=window).mean()
            data[f'net_flow_ma_{window}'] = data['net_flow'].rolling(window=window).mean()
        
        # Create volatility features
        data['inflow_volatility_7d'] = data['daily_inflows'].rolling(window=7).std()
        data['outflow_volatility_7d'] = data['daily_outflows'].rolling(window=7).std()
        
        # Remove rows with NaN values (due to lag and rolling features)
        data = data.dropna()
        
        if data.empty:
            return np.array([]), np.array([])
        
        # Define feature columns
        feature_cols = [col for col in data.columns if col not in ['date', 'daily_inflows', 'daily_outflows', 'net_flow', 'transaction_count']]
        
        # Prepare features and targets
        X = data[feature_cols].values
        y = data['net_flow'].values  # Predict net cash flow
        
        return X, y
    
    def _get_feature_names(self):
        """Get feature names for interpretation"""
        base_features = ['day_of_week', 'month', 'quarter', 'is_month_end', 'is_quarter_end']
        
        # Lag features
        lag_features = []
        for lag in [1, 2, 3, 7, 14, 30]:
            lag_features.extend([f'inflow_lag_{lag}', f'outflow_lag_{lag}', f'net_flow_lag_{lag}'])
        
        # Moving average features
        ma_features = []
        for window in [7, 14, 30]:
            ma_features.extend([f'inflow_ma_{window}', f'outflow_ma_{window}', f'net_flow_ma_{window}'])
        
        # Volatility features
        vol_features = ['inflow_volatility_7d', 'outflow_volatility_7d']
        
        return base_features + lag_features + ma_features + vol_features
    
    def generate_forecast(self, forecast_days=90, model_type='ensemble'):
        """Generate cash flow forecast for specified number of days"""
        if not self.models:
            model_build_result = self.build_forecasting_models()
            if not model_build_result['models_built']:
                return model_build_result
        
        # Get recent data to use as input for forecasting
        recent_data = self._prepare_historical_data()
        
        if recent_data.empty:
            return {'forecast_generated': False, 'error': 'No historical data available'}
        
        # Prepare features for the most recent data point
        features, _ = self._prepare_features_targets(recent_data)
        
        if len(features) == 0:
            return {'forecast_generated': False, 'error': 'Cannot prepare features for forecasting'}
        
        # Start forecasting from the next day
        start_date = recent_data['date'].max() + timedelta(days=1)
        forecast_dates = [start_date + timedelta(days=i) for i in range(forecast_days)]
        
        forecasts = []
        last_features = features[-1].copy()  # Start with most recent features
        
        for i, forecast_date in enumerate(forecast_dates):
            # Update time-based features for the forecast date
            updated_features = self._update_time_features(last_features, forecast_date)
            
            # Scale features
            scaled_features = self.scalers['main'].transform([updated_features])
            
            # Generate prediction based on model type
            if model_type == 'ensemble':
                # Use ensemble of models
                lr_pred = self.models['linear_regression'].predict(scaled_features)[0]
                rf_pred = self.models['random_forest'].predict(scaled_features)[0]
                net_flow_pred = (lr_pred + rf_pred) / 2
            else:
                net_flow_pred = self.models.get(model_type, self.models['random_forest']).predict(scaled_features)[0]
            
            # Estimate inflows and outflows (simplified approach)
            # Use historical ratios to split net flow
            avg_inflow_ratio = 0.6  # Typical ratio of inflows to total absolute flow
            
            if net_flow_pred > 0:
                # Positive net flow
                estimated_inflows = abs(net_flow_pred) / (2 * avg_inflow_ratio - 1)
                estimated_outflows = estimated_inflows - net_flow_pred
            else:
                # Negative net flow
                estimated_outflows = abs(net_flow_pred) / (2 * (1 - avg_inflow_ratio) - 1)
                estimated_inflows = estimated_outflows + net_flow_pred
            
            # Ensure non-negative values
            estimated_inflows = max(0, estimated_inflows)
            estimated_outflows = max(0, estimated_outflows)
            
            # Calculate confidence level (decreases with forecast horizon)
            confidence = max(0.3, 0.9 - (i / forecast_days) * 0.5)
            
            forecasts.append({
                'date': forecast_date.date(),
                'predicted_inflows': round(estimated_inflows, 2),
                'predicted_outflows': round(estimated_outflows, 2),
                'predicted_net_flow': round(net_flow_pred, 2),
                'confidence_level': round(confidence, 3)
            })
            
            # Update features for next iteration (simple approach)
            last_features = self._update_lag_features(last_features, net_flow_pred, estimated_inflows, estimated_outflows)
        
        # Calculate cumulative cash flow
        cumulative_cash = 0
        for forecast in forecasts:
            cumulative_cash += forecast['predicted_net_flow']
            forecast['cumulative_cash_flow'] = round(cumulative_cash, 2)
        
        # Aggregate forecasts by week and month
        forecasts_df = pd.DataFrame(forecasts)
        forecasts_df['date'] = pd.to_datetime(forecasts_df['date'])
        forecasts_df['week'] = forecasts_df['date'].dt.to_period('W')
        forecasts_df['month'] = forecasts_df['date'].dt.to_period('M')
        
        weekly_summary = forecasts_df.groupby('week').agg({
            'predicted_inflows': 'sum',
            'predicted_outflows': 'sum',
            'predicted_net_flow': 'sum'
        }).round(2)
        
        monthly_summary = forecasts_df.groupby('month').agg({
            'predicted_inflows': 'sum',
            'predicted_outflows': 'sum',
            'predicted_net_flow': 'sum'
        }).round(2)
        
        return {
            'forecast_generated': True,
            'forecast_period': forecast_days,
            'model_used': model_type,
            'start_date': start_date.date(),
            'daily_forecasts': forecasts,
            'weekly_summary': weekly_summary.to_dict('index'),
            'monthly_summary': monthly_summary.to_dict('index'),
            'summary_statistics': {
                'total_predicted_inflows': float(forecasts_df['predicted_inflows'].sum()),
                'total_predicted_outflows': float(forecasts_df['predicted_outflows'].sum()),
                'total_predicted_net_flow': float(forecasts_df['predicted_net_flow'].sum()),
                'average_daily_net_flow': float(forecasts_df['predicted_net_flow'].mean()),
                'final_cumulative_cash': float(forecasts_df['cumulative_cash_flow'].iloc[-1])
            }
        }
    
    def _update_time_features(self, features, date):
        """Update time-based features for a specific date"""
        updated_features = features.copy()
        
        # Assuming the order matches _get_feature_names()
        updated_features[0] = date.weekday()  # day_of_week
        updated_features[1] = date.month      # month
        updated_features[2] = (date.month - 1) // 3 + 1  # quarter
        updated_features[3] = 1 if date.day > 25 else 0  # is_month_end
        updated_features[4] = 1 if (date.month % 3 == 0 and date.day > 25) else 0  # is_quarter_end
        
        return updated_features
    
    def _update_lag_features(self, features, net_flow, inflows, outflows):
        """Update lag features with new prediction (simplified)"""
        # This is a simplified approach - in practice, you'd maintain a more sophisticated feature update mechanism
        updated_features = features.copy()
        
        # Shift lag features (very simplified)
        # In a real implementation, you'd properly maintain the time series
        
        return updated_features
    
    def integrate_ar_ap_forecasts(self, ar_ap_tracker):
        """Integrate accounts receivable and payable forecasts"""
        try:
            ar_analysis = ar_ap_tracker.analyze_receivables_patterns()
            ap_analysis = ar_ap_tracker.analyze_payables_patterns()
            
            # Get payment predictions
            ar_predictions = ar_analysis.get('payment_predictions', {})
            ap_schedule = ap_analysis.get('payment_schedule', {})
            
            # Create integrated forecast
            integrated_forecast = []
            
            # Process AR predictions
            if 'weekly_collections' in ar_predictions:
                for week, amount in ar_predictions['weekly_collections'].items():
                    # Convert period to date range
                    week_start = pd.Period(week).start_time.date()
                    integrated_forecast.append({
                        'date': week_start,
                        'type': 'receivables_collection',
                        'amount': float(amount),
                        'category': 'inflow'
                    })
            
            # Process AP schedule
            if 'weekly_payment_requirements' in ap_schedule:
                for week, amount in ap_schedule['weekly_payment_requirements'].items():
                    week_start = pd.Period(week).start_time.date()
                    integrated_forecast.append({
                        'date': week_start,
                        'type': 'payables_payment',
                        'amount': float(amount),
                        'category': 'outflow'
                    })
            
            return {
                'integration_successful': True,
                'integrated_forecasts': integrated_forecast,
                'ar_summary': ar_analysis.get('summary', {}),
                'ap_summary': ap_analysis.get('summary', {})
            }
            
        except Exception as e:
            return {
                'integration_successful': False,
                'error': str(e)
            }
    
    def generate_scenario_analysis(self, scenarios=None):
        """Generate scenario-based forecasts"""
        if scenarios is None:
            scenarios = {
                'optimistic': {'inflow_multiplier': 1.15, 'outflow_multiplier': 0.95},
                'realistic': {'inflow_multiplier': 1.0, 'outflow_multiplier': 1.0},
                'pessimistic': {'inflow_multiplier': 0.85, 'outflow_multiplier': 1.1}
            }
        
        scenario_results = {}
        
        # Generate base forecast
        base_forecast = self.generate_forecast(forecast_days=90)
        
        if not base_forecast.get('forecast_generated', False):
            return {'scenario_analysis_generated': False, 'error': 'Could not generate base forecast'}
        
        for scenario_name, adjustments in scenarios.items():
            scenario_forecasts = []
            
            for daily_forecast in base_forecast['daily_forecasts']:
                adjusted_forecast = daily_forecast.copy()
                adjusted_forecast['predicted_inflows'] *= adjustments['inflow_multiplier']
                adjusted_forecast['predicted_outflows'] *= adjustments['outflow_multiplier']
                adjusted_forecast['predicted_net_flow'] = (
                    adjusted_forecast['predicted_inflows'] - adjusted_forecast['predicted_outflows']
                )
                scenario_forecasts.append(adjusted_forecast)
            
            # Recalculate cumulative cash flow
            cumulative_cash = 0
            for forecast in scenario_forecasts:
                cumulative_cash += forecast['predicted_net_flow']
                forecast['cumulative_cash_flow'] = round(cumulative_cash, 2)
            
            scenario_results[scenario_name] = {
                'forecasts': scenario_forecasts,
                'total_net_flow': sum(f['predicted_net_flow'] for f in scenario_forecasts),
                'final_cash_position': scenario_forecasts[-1]['cumulative_cash_flow']
            }
        
        return {
            'scenario_analysis_generated': True,
            'scenarios': scenario_results,
            'base_forecast': base_forecast
        }
    
    def save_forecast_to_database(self, forecast_data, model_name='ensemble'):
        """Save forecast results to database"""
        if not forecast_data.get('forecast_generated', False):
            return {'saved': False, 'error': 'No valid forecast to save'}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing forecasts for this period
        cursor.execute("DELETE FROM cash_forecasts WHERE forecast_date >= date('now')")
        
        # Insert new forecasts
        forecast_date = datetime.now().date()
        
        for daily_forecast in forecast_data['daily_forecasts']:
            cursor.execute("""
                INSERT INTO cash_forecasts 
                (forecast_date, forecast_period_start, forecast_period_end, 
                 predicted_inflows, predicted_outflows, net_cash_flow, 
                 cumulative_cash, confidence_level, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                forecast_date,
                daily_forecast['date'],
                daily_forecast['date'],
                daily_forecast['predicted_inflows'],
                daily_forecast['predicted_outflows'],
                daily_forecast['predicted_net_flow'],
                daily_forecast['cumulative_cash_flow'],
                daily_forecast['confidence_level'],
                model_name
            ))
        
        conn.commit()
        conn.close()
        
        return {
            'saved': True,
            'records_saved': len(forecast_data['daily_forecasts']),
            'forecast_date': forecast_date
        }