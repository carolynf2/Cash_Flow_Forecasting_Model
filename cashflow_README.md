# Cash Flow Forecasting Model

A comprehensive cash flow forecasting system using Python and SQL that predicts cash flows based on historical patterns and outstanding receivables/payables.

## Features

- **Historical Pattern Analysis**: Deep analysis of cash flow trends, seasonality, and volatility
- **Machine Learning Forecasting**: Multiple ML models including Random Forest and Linear Regression
- **AR/AP Integration**: Accounts receivable and payable analysis with collection/payment predictions
- **Interactive Dashboard**: Real-time Streamlit web dashboard with visualizations
- **Scenario Analysis**: Optimistic, realistic, and pessimistic forecasting scenarios
- **Risk Assessment**: Comprehensive risk analysis and volatility measurements

## System Components

### 1. Database Layer (`cashflow_database.py`)
- **SQLite database** with 8 comprehensive tables
- **Tables**: cash_transactions, accounts_receivable, accounts_payable, customers, vendors, cash_forecasts, budget_items, payment_patterns
- **Sample data generation** with realistic financial scenarios
- **2+ years of historical data** with seasonal patterns

### 2. Pattern Analysis Engine (`pattern_analyzer.py`)
- **Historical analysis** of cash flow patterns over 730+ days
- **Seasonal decomposition** and trend analysis
- **Monthly and weekly pattern recognition**
- **Volatility and risk assessment**
- **Recurring transaction identification**
- **Future pattern prediction** based on historical trends

### 3. AR/AP Tracker (`receivables_payables_tracker.py`)
- **Receivables analysis** with aging buckets and collection predictions
- **Payables scheduling** with payment optimization
- **Customer risk profiling** based on payment history
- **Vendor payment pattern analysis**
- **Working capital impact** calculations
- **Collection probability** modeling

### 4. ML Forecasting Engine (`cashflow_predictor.py`)
- **Multiple ML models**: Random Forest, Linear Regression, Ensemble
- **Feature engineering** with lag variables, moving averages, and seasonal indicators
- **Time series forecasting** with confidence intervals
- **Model performance tracking** and automatic selection
- **90-180 day forecasting** capability
- **Scenario-based modeling**

### 5. Interactive Dashboard (`cashflow_dashboard.py`)
- **Real-time visualization** with Plotly charts
- **Key financial metrics** and KPIs
- **Forecast comparison** and confidence tracking
- **AR/AP aging analysis**
- **Risk assessment** dashboard
- **Export capabilities**

## Installation

1. Install required dependencies:
```bash
pip install -r cashflow_requirements.txt
```

2. Set up the database and generate sample data:
```bash
python cashflow_main.py --setup
```

## Usage

### Command Line Interface

```bash
# Setup database and generate sample data
python cashflow_main.py --setup

# Run historical pattern analysis
python cashflow_main.py --analyze

# Analyze accounts receivable and payable
python cashflow_main.py --ar-ap

# Generate cash flow forecast
python cashflow_main.py --forecast --days 90 --model ensemble

# Run scenario analysis
python cashflow_main.py --scenarios

# Start interactive dashboard
python cashflow_main.py --dashboard
```

### Web Dashboard

Start the dashboard:
```bash
python cashflow_main.py --dashboard
```

Then navigate to `http://localhost:8501` in your browser.

## Analysis Results

### Historical Pattern Analysis
- **730-day analysis period** with 418+ transactions
- **Net cash flow tracking**: -$778K over analysis period  
- **Monthly volatility**: $62K standard deviation
- **Risk assessment**: High volatility with seasonal patterns
- **Best/worst months** identified for planning

### AR/AP Analysis
- **Outstanding receivables**: $782K with 71% collection rate
- **Outstanding payables**: $767K with 30% payment rate
- **Working capital**: $15K net position
- **Collection predictions** with probability scoring
- **Payment scheduling** optimization

### Cash Flow Forecasting
- **ML model performance**: Random Forest selected as best model
- **60-90 day predictions** with declining confidence
- **Multiple scenarios**: Optimistic (+$9K), Realistic (+$5K), Pessimistic (-$1K)
- **Risk assessment**: 1 of 3 scenarios show negative cash flow

## Key Features

### 1. Pattern Recognition
- **Seasonal analysis**: Identifies best/worst performing months
- **Trend detection**: Long-term cash flow direction
- **Volatility measurement**: Risk quantification
- **Recurring pattern identification**: Predictable cash flows

### 2. Predictive Modeling
- **Feature engineering**: 20+ variables including lags, moving averages, seasonality
- **Model ensemble**: Combines multiple algorithms for better accuracy
- **Confidence intervals**: Decreasing confidence over time horizon
- **Performance tracking**: Model validation and selection

### 3. AR/AP Integration
- **Aging analysis**: Current, 1-30, 31-60, 61-90, 91-120, 120+ day buckets
- **Collection probability**: Based on customer payment history
- **Payment optimization**: Prioritized payment schedules
- **Working capital impact**: Net cash flow projections

### 4. Risk Assessment
- **Volatility analysis**: Weekly and monthly volatility measurements
- **Scenario modeling**: Best/worst case scenario planning
- **Confidence tracking**: Uncertainty quantification
- **Risk categorization**: Low/Medium/High risk classification

## Dashboard Features

### Overview Tab
- **Key metrics**: 30-day inflows/outflows, net flow, AR/AP balances
- **Monthly trends**: Historical cash flow visualization
- **Category breakdown**: Inflow/outflow analysis by category
- **Cumulative tracking**: Running cash position

### Forecasts Tab
- **Prediction charts**: Daily, weekly, monthly forecasts
- **Model performance**: Confidence levels and accuracy metrics
- **Summary statistics**: Total flows and averages
- **Forecast export**: CSV download capability

### Historical Analysis Tab
- **Pattern visualization**: Seasonal and trend charts
- **Statistical summary**: Key financial metrics
- **Risk analysis**: Volatility and coefficient of variation
- **Monthly performance**: Best/worst period identification

### AR/AP Analysis Tab
- **Aging buckets**: Visual aging analysis
- **Collection predictions**: Expected collection timeline
- **Payment schedules**: Optimized payment planning
- **Working capital**: Net position tracking

### Scenarios Tab
- **Three scenarios**: Optimistic, realistic, pessimistic
- **Comparison charts**: Side-by-side scenario analysis
- **Risk assessment**: Probability of negative cash flow
- **Planning insights**: Decision support information

## Technical Implementation

### Database Schema
```sql
-- Core transaction tracking
cash_transactions: date, category, amount, type, recurring pattern

-- AR/AP management  
accounts_receivable: customer, amount, due_date, status, payment_terms
accounts_payable: vendor, amount, due_date, status, payment_terms

-- Customer/vendor profiles
customers: payment_history_score, average_days_to_pay, industry
vendors: payment_terms, average_monthly_spend, category

-- Forecasting results
cash_forecasts: forecast_date, predicted_flows, confidence_level
```

### ML Features
- **Time-based**: day_of_week, month, quarter, month_end, quarter_end
- **Lag variables**: 1, 2, 3, 7, 14, 30-day lags for inflows/outflows
- **Moving averages**: 7, 14, 30-day rolling averages
- **Volatility**: 7-day rolling standard deviation

### Performance Metrics
- **Model accuracy**: MAE and RMSE tracking
- **Confidence intervals**: Decreasing over forecast horizon
- **Feature importance**: Random Forest feature ranking
- **Validation**: 80/20 train/test split

## Sample Data

The system includes comprehensive sample data:
- **8 customers** with varying payment behaviors
- **8 vendors** across different categories
- **419 transactions** over 2 years with seasonal patterns
- **100 AR records** in various stages (outstanding, partial, paid, overdue)
- **80 AP records** with different payment terms
- **Realistic patterns**: Monthly recurring revenues/expenses, seasonal variations

## Advanced Features

### Scenario Analysis
- **Optimistic**: 15% higher inflows, 5% lower outflows
- **Realistic**: Baseline forecast
- **Pessimistic**: 15% lower inflows, 10% higher outflows

### Risk Assessment
- **Collection risk**: Based on days outstanding and customer payment history
- **Payment prioritization**: Overdue items first, then by due date
- **Cash requirements**: 7, 30, 60, 90-day payment obligations

### Integration Capabilities
- **AR/AP integration**: Combines operational forecasts with ML predictions
- **Budget integration**: Links to annual budget items
- **Pattern learning**: Adapts predictions based on new data

## Configuration

Key parameters can be adjusted:
- **Forecast horizon**: 30-180 days
- **Model selection**: ensemble, random_forest, linear_regression  
- **Confidence thresholds**: Risk assessment levels
- **Seasonal factors**: Monthly adjustment multipliers

## Export and Reporting

- **CSV exports**: Detailed forecast data
- **Database storage**: Persistent forecast history
- **Dashboard views**: Interactive charts and tables
- **Summary reports**: Key metrics and insights

## Future Enhancements

- **Real-time data feeds**: Live transaction integration
- **Advanced ML models**: LSTM, ARIMA time series models
- **External data**: Economic indicators, industry benchmarks
- **Mobile dashboard**: Responsive design for mobile devices
- **API endpoints**: REST API for external integrations
- **Automated alerts**: Email/SMS notifications for cash flow issues

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations  
- `scikit-learn`: Machine learning algorithms
- `scipy`: Statistical functions
- `streamlit`: Web dashboard framework
- `plotly`: Interactive visualizations
- `sqlite3`: Database management
- `statsmodels`: Time series analysis (optional)

## Performance Considerations

- **Efficient queries**: Optimized database access patterns
- **Model caching**: Trained models stored for reuse
- **Incremental updates**: New data processing
- **Memory management**: Large dataset handling
- **Parallel processing**: Multi-model training

## Security Features

- **Local database**: No external data transmission
- **Data validation**: Input sanitization and validation
- **Access control**: File-based permissions
- **Audit trail**: Transaction and forecast logging
- **Privacy**: No sensitive data exposure

This comprehensive cash flow forecasting system provides enterprise-level capabilities for financial planning and risk management, combining traditional financial analysis with modern machine learning techniques.