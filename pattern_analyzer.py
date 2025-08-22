import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class CashFlowPatternAnalyzer:
    def __init__(self, db_path="cashflow_forecast.db"):
        self.db_path = db_path
    
    def analyze_historical_patterns(self, period_days=730):
        """Analyze historical cash flow patterns"""
        conn = sqlite3.connect(self.db_path)
        
        # Get historical transactions
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        query = """
            SELECT 
                transaction_date,
                category,
                subcategory,
                transaction_type,
                amount,
                is_recurring,
                recurrence_pattern
            FROM cash_transactions
            WHERE transaction_date >= ? AND transaction_date <= ?
            ORDER BY transaction_date
        """
        
        df = pd.read_sql_query(query, conn, params=[start_date.date(), end_date.date()])
        conn.close()
        
        if df.empty:
            return self._empty_analysis_result()
        
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Analyze patterns
        analysis = {
            'period_analyzed': {
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'total_days': period_days
            },
            'summary_statistics': self._calculate_summary_stats(df),
            'monthly_patterns': self._analyze_monthly_patterns(df),
            'seasonal_patterns': self._analyze_seasonal_patterns(df),
            'trend_analysis': self._analyze_trends(df),
            'category_analysis': self._analyze_category_patterns(df),
            'recurring_analysis': self._analyze_recurring_patterns(df),
            'volatility_analysis': self._analyze_volatility(df)
        }
        
        return analysis
    
    def _empty_analysis_result(self):
        """Return empty analysis structure"""
        return {
            'period_analyzed': None,
            'summary_statistics': {},
            'monthly_patterns': {},
            'seasonal_patterns': {},
            'trend_analysis': {},
            'category_analysis': {},
            'recurring_analysis': {},
            'volatility_analysis': {}
        }
    
    def _calculate_summary_stats(self, df):
        """Calculate basic summary statistics"""
        inflows = df[df['transaction_type'] == 'inflow']['amount']
        outflows = df[df['transaction_type'] == 'outflow']['amount']
        
        return {
            'total_transactions': len(df),
            'total_inflows': float(inflows.sum()),
            'total_outflows': float(outflows.sum()),
            'net_cash_flow': float(inflows.sum() - outflows.sum()),
            'average_inflow': float(inflows.mean()) if len(inflows) > 0 else 0,
            'average_outflow': float(outflows.mean()) if len(outflows) > 0 else 0,
            'inflow_std': float(inflows.std()) if len(inflows) > 0 else 0,
            'outflow_std': float(outflows.std()) if len(outflows) > 0 else 0,
            'inflow_transactions': len(inflows),
            'outflow_transactions': len(outflows)
        }
    
    def _analyze_monthly_patterns(self, df):
        """Analyze monthly cash flow patterns"""
        df['year_month'] = df['transaction_date'].dt.to_period('M')
        
        monthly_summary = df.groupby(['year_month', 'transaction_type']).agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        
        # Flatten column names
        monthly_summary.columns = ['total_amount', 'transaction_count', 'avg_amount']
        monthly_summary = monthly_summary.reset_index()
        
        # Pivot to get inflows and outflows as separate columns
        monthly_pivot = monthly_summary.pivot(index='year_month', columns='transaction_type', values='total_amount').fillna(0)
        
        if not monthly_pivot.empty:
            monthly_pivot['net_flow'] = monthly_pivot.get('inflow', 0) - monthly_pivot.get('outflow', 0)
            monthly_pivot['cumulative_flow'] = monthly_pivot['net_flow'].cumsum()
            
            # Calculate month-over-month growth
            monthly_pivot['inflow_growth'] = monthly_pivot.get('inflow', pd.Series()).pct_change() * 100
            monthly_pivot['outflow_growth'] = monthly_pivot.get('outflow', pd.Series()).pct_change() * 100
        
        return {
            'monthly_data': monthly_pivot.to_dict('index') if not monthly_pivot.empty else {},
            'avg_monthly_inflow': float(monthly_pivot.get('inflow', pd.Series()).mean()) if not monthly_pivot.empty else 0,
            'avg_monthly_outflow': float(monthly_pivot.get('outflow', pd.Series()).mean()) if not monthly_pivot.empty else 0,
            'avg_monthly_net': float(monthly_pivot.get('net_flow', pd.Series()).mean()) if not monthly_pivot.empty else 0,
            'monthly_volatility': float(monthly_pivot.get('net_flow', pd.Series()).std()) if not monthly_pivot.empty else 0
        }
    
    def _analyze_seasonal_patterns(self, df):
        """Analyze seasonal patterns in cash flows"""
        # Group by month across all years
        df['month'] = df['transaction_date'].dt.month
        seasonal_summary = df.groupby(['month', 'transaction_type'])['amount'].sum().unstack(fill_value=0)
        
        if not seasonal_summary.empty:
            seasonal_summary['net_flow'] = seasonal_summary.get('inflow', 0) - seasonal_summary.get('outflow', 0)
            
            # Calculate seasonal indices (relative to average)
            avg_monthly_net = seasonal_summary['net_flow'].mean()
            seasonal_summary['seasonal_index'] = seasonal_summary['net_flow'] / avg_monthly_net if avg_monthly_net != 0 else 1
        
        # Analyze day of week patterns
        df['day_of_week'] = df['transaction_date'].dt.day_name()
        daily_patterns = df.groupby(['day_of_week', 'transaction_type'])['amount'].agg(['sum', 'count']).round(2)
        
        return {
            'monthly_seasonal': seasonal_summary.to_dict('index') if not seasonal_summary.empty else {},
            'best_month': seasonal_summary['net_flow'].idxmax() if not seasonal_summary.empty else None,
            'worst_month': seasonal_summary['net_flow'].idxmin() if not seasonal_summary.empty else None,
            'seasonal_amplitude': float(seasonal_summary['net_flow'].max() - seasonal_summary['net_flow'].min()) if not seasonal_summary.empty else 0,
            'daily_patterns': daily_patterns.to_dict('index') if not daily_patterns.empty else {}
        }
    
    def _analyze_trends(self, df):
        """Analyze long-term trends in cash flows"""
        # Create daily aggregates
        daily_flows = df.groupby(['transaction_date', 'transaction_type'])['amount'].sum().unstack(fill_value=0)
        
        if daily_flows.empty:
            return {'trend_detected': False}
        
        daily_flows['net_flow'] = daily_flows.get('inflow', 0) - daily_flows.get('outflow', 0)
        daily_flows['cumulative_flow'] = daily_flows['net_flow'].cumsum()
        
        # Perform linear regression on cumulative flows
        X = np.arange(len(daily_flows)).reshape(-1, 1)
        y = daily_flows['cumulative_flow'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate trend statistics
        trend_slope = float(model.coef_[0])
        r_squared = float(model.score(X, y))
        
        # Determine trend direction and strength
        if abs(trend_slope) < 100:  # Small daily change
            trend_direction = 'stable'
        elif trend_slope > 0:
            trend_direction = 'improving'
        else:
            trend_direction = 'declining'
        
        # Calculate recent vs historical average
        recent_days = 30
        if len(daily_flows) >= recent_days:
            recent_avg = daily_flows['net_flow'].tail(recent_days).mean()
            historical_avg = daily_flows['net_flow'].head(-recent_days).mean() if len(daily_flows) > recent_days else daily_flows['net_flow'].mean()
            recent_vs_historical = (recent_avg - historical_avg) / abs(historical_avg) * 100 if historical_avg != 0 else 0
        else:
            recent_vs_historical = 0
        
        return {
            'trend_detected': True,
            'trend_slope_daily': trend_slope,
            'trend_direction': trend_direction,
            'trend_strength': r_squared,
            'recent_vs_historical_pct': float(recent_vs_historical),
            'avg_daily_net_flow': float(daily_flows['net_flow'].mean()),
            'net_flow_volatility': float(daily_flows['net_flow'].std())
        }
    
    def _analyze_category_patterns(self, df):
        """Analyze patterns by category"""
        category_stats = df.groupby(['category', 'transaction_type']).agg({
            'amount': ['sum', 'count', 'mean', 'std']
        }).round(2)
        
        category_stats.columns = ['total', 'count', 'average', 'std_dev']
        category_stats = category_stats.reset_index()
        
        # Calculate category reliability (coefficient of variation)
        category_stats['reliability_score'] = 1 / (1 + category_stats['std_dev'] / category_stats['average'].replace(0, 1))
        
        # Identify top categories by volume
        top_inflow_categories = category_stats[category_stats['transaction_type'] == 'inflow'].nlargest(5, 'total')
        top_outflow_categories = category_stats[category_stats['transaction_type'] == 'outflow'].nlargest(5, 'total')
        
        return {
            'category_statistics': category_stats.to_dict('records'),
            'top_inflow_categories': top_inflow_categories.to_dict('records'),
            'top_outflow_categories': top_outflow_categories.to_dict('records'),
            'most_reliable_categories': category_stats.nlargest(10, 'reliability_score').to_dict('records')
        }
    
    def _analyze_recurring_patterns(self, df):
        """Analyze recurring transaction patterns"""
        recurring_df = df[df['is_recurring'] == 1]
        
        if recurring_df.empty:
            return {'has_recurring': False}
        
        recurring_summary = recurring_df.groupby(['recurrence_pattern', 'transaction_type']).agg({
            'amount': ['sum', 'count', 'mean', 'std']
        }).round(2)
        
        recurring_summary.columns = ['total', 'count', 'average', 'std_dev']
        recurring_summary = recurring_summary.reset_index()
        
        # Calculate predictability score
        recurring_summary['predictability_score'] = 1 / (1 + recurring_summary['std_dev'] / recurring_summary['average'].replace(0, 1))
        
        return {
            'has_recurring': True,
            'recurring_summary': recurring_summary.to_dict('records'),
            'total_recurring_inflows': float(recurring_df[recurring_df['transaction_type'] == 'inflow']['amount'].sum()),
            'total_recurring_outflows': float(recurring_df[recurring_df['transaction_type'] == 'outflow']['amount'].sum()),
            'recurring_percentage': float(len(recurring_df) / len(df) * 100)
        }
    
    def _analyze_volatility(self, df):
        """Analyze cash flow volatility"""
        # Create weekly aggregates for volatility analysis
        df['week'] = df['transaction_date'].dt.to_period('W')
        weekly_flows = df.groupby(['week', 'transaction_type'])['amount'].sum().unstack(fill_value=0)
        
        if weekly_flows.empty:
            return {'volatility_calculated': False}
        
        weekly_flows['net_flow'] = weekly_flows.get('inflow', 0) - weekly_flows.get('outflow', 0)
        
        # Calculate various volatility measures
        net_flows = weekly_flows['net_flow']
        
        volatility_stats = {
            'volatility_calculated': True,
            'weekly_volatility': float(net_flows.std()),
            'coefficient_of_variation': float(net_flows.std() / abs(net_flows.mean())) if net_flows.mean() != 0 else float('inf'),
            'min_weekly_flow': float(net_flows.min()),
            'max_weekly_flow': float(net_flows.max()),
            'volatility_range': float(net_flows.max() - net_flows.min()),
            'negative_weeks': int((net_flows < 0).sum()),
            'positive_weeks': int((net_flows > 0).sum()),
            'weeks_analyzed': len(net_flows)
        }
        
        # Risk assessment
        cv = volatility_stats['coefficient_of_variation']
        if cv < 0.5:
            risk_level = 'Low'
        elif cv < 1.0:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        volatility_stats['risk_level'] = risk_level
        
        return volatility_stats
    
    def predict_future_patterns(self, forecast_days=90):
        """Predict future cash flow patterns based on historical analysis"""
        analysis = self.analyze_historical_patterns()
        
        if not analysis['summary_statistics']:
            return {'prediction_available': False}
        
        # Get current data for time series analysis
        conn = sqlite3.connect(self.db_path)
        
        # Get daily aggregated cash flows
        query = """
            SELECT 
                transaction_date,
                SUM(CASE WHEN transaction_type = 'inflow' THEN amount ELSE 0 END) as inflows,
                SUM(CASE WHEN transaction_type = 'outflow' THEN amount ELSE 0 END) as outflows
            FROM cash_transactions
            WHERE transaction_date >= date('now', '-365 days')
            GROUP BY transaction_date
            ORDER BY transaction_date
        """
        
        daily_data = pd.read_sql_query(query, conn)
        conn.close()
        
        if daily_data.empty:
            return {'prediction_available': False}
        
        daily_data['transaction_date'] = pd.to_datetime(daily_data['transaction_date'])
        daily_data['net_flow'] = daily_data['inflows'] - daily_data['outflows']
        
        # Create complete date range and fill missing dates
        date_range = pd.date_range(start=daily_data['transaction_date'].min(), 
                                 end=daily_data['transaction_date'].max(), 
                                 freq='D')
        daily_data = daily_data.set_index('transaction_date').reindex(date_range, fill_value=0)
        daily_data.index.name = 'date'
        daily_data = daily_data.reset_index()
        
        # Simple trend-based prediction
        recent_data = daily_data.tail(30)  # Last 30 days
        avg_daily_inflow = recent_data['inflows'].mean()
        avg_daily_outflow = recent_data['outflows'].mean()
        avg_daily_net = recent_data['net_flow'].mean()
        
        # Apply seasonal factors
        seasonal_factors = analysis['seasonal_patterns'].get('monthly_seasonal', {})
        
        predictions = []
        start_date = datetime.now() + timedelta(days=1)
        
        for i in range(forecast_days):
            forecast_date = start_date + timedelta(days=i)
            month = forecast_date.month
            
            # Get seasonal factor for the month
            seasonal_factor = 1.0
            if seasonal_factors and month in seasonal_factors:
                seasonal_factor = seasonal_factors[month].get('seasonal_index', 1.0)
            
            # Add some randomness based on historical volatility
            volatility = analysis['volatility_analysis'].get('weekly_volatility', 0) / 7  # Daily volatility
            
            predicted_inflow = avg_daily_inflow * seasonal_factor + np.random.normal(0, volatility * 0.5)
            predicted_outflow = avg_daily_outflow * seasonal_factor + np.random.normal(0, volatility * 0.5)
            predicted_net = predicted_inflow - predicted_outflow
            
            predictions.append({
                'date': forecast_date.date(),
                'predicted_inflows': max(0, round(predicted_inflow, 2)),
                'predicted_outflows': max(0, round(predicted_outflow, 2)),
                'predicted_net_flow': round(predicted_net, 2),
                'confidence_level': max(0.3, 0.9 - (i / forecast_days) * 0.4)  # Decreasing confidence
            })
        
        return {
            'prediction_available': True,
            'forecast_period_days': forecast_days,
            'predictions': predictions,
            'methodology': 'Trend-based with seasonal adjustment',
            'base_assumptions': {
                'avg_daily_inflow': round(avg_daily_inflow, 2),
                'avg_daily_outflow': round(avg_daily_outflow, 2),
                'seasonal_adjustment': 'Applied based on historical monthly patterns'
            }
        }