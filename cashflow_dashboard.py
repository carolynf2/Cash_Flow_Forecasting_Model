import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
from cashflow_database import CashFlowDatabase
from pattern_analyzer import CashFlowPatternAnalyzer
from receivables_payables_tracker import ReceivablesPayablesTracker
from cashflow_predictor import CashFlowPredictor

# Page configuration
st.set_page_config(
    page_title="Cash Flow Forecasting Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    .positive-flow {
        color: #28a745;
        font-weight: bold;
    }
    .negative-flow {
        color: #dc3545;
        font-weight: bold;
    }
    .forecast-confidence {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CashFlowDashboard:
    def __init__(self):
        self.db = CashFlowDatabase()
        self.analyzer = CashFlowPatternAnalyzer()
        self.tracker = ReceivablesPayablesTracker()
        self.predictor = CashFlowPredictor()
    
    def run(self):
        st.title("💰 Cash Flow Forecasting Dashboard")
        st.markdown("Advanced cash flow analysis and prediction based on historical patterns and outstanding receivables/payables")
        
        # Sidebar
        with st.sidebar:
            st.header("Controls")
            
            if st.button("🔄 Refresh Data", type="primary"):
                st.rerun()
            
            if st.button("📊 Generate Sample Data"):
                with st.spinner("Generating sample financial data..."):
                    self.db.generate_sample_data()
                    st.success("Sample data generated successfully!")
                    st.rerun()
            
            st.markdown("---")
            st.header("Forecast Settings")
            
            forecast_days = st.slider("Forecast Period (Days)", 30, 180, 90)
            model_type = st.selectbox("Prediction Model", 
                                    ['ensemble', 'random_forest', 'linear_regression'])
            
            if st.button("🔮 Generate Forecast"):
                with st.spinner("Building forecasting models and generating predictions..."):
                    forecast_result = self.predictor.generate_forecast(
                        forecast_days=forecast_days, 
                        model_type=model_type
                    )
                    if forecast_result.get('forecast_generated', False):
                        st.session_state['forecast_result'] = forecast_result
                        st.success(f"Forecast generated for {forecast_days} days!")
                    else:
                        st.error(f"Forecast generation failed: {forecast_result.get('error', 'Unknown error')}")
                    st.rerun()
            
            st.markdown("---")
            st.header("Analysis Options")
            
            if st.button("📈 Run Pattern Analysis"):
                with st.spinner("Analyzing historical patterns..."):
                    st.session_state['pattern_analysis'] = self.analyzer.analyze_historical_patterns()
                    st.success("Pattern analysis completed!")
                    st.rerun()
            
            if st.button("💳 Analyze AR/AP"):
                with st.spinner("Analyzing receivables and payables..."):
                    st.session_state['ar_analysis'] = self.tracker.analyze_receivables_patterns()
                    st.session_state['ap_analysis'] = self.tracker.analyze_payables_patterns()
                    st.success("AR/AP analysis completed!")
                    st.rerun()
        
        # Main dashboard
        self.display_main_dashboard(forecast_days, model_type)
    
    def display_main_dashboard(self, forecast_days, model_type):
        # Check if we have data
        try:
            # Get basic cash flow data
            conn = sqlite3.connect(self.db.db_path)
            cash_check = pd.read_sql_query("SELECT COUNT(*) as count FROM cash_transactions", conn)
            conn.close()
            
            if cash_check['count'].iloc[0] == 0:
                st.warning("No data available. Please generate sample data using the sidebar.")
                return
                
        except Exception as e:
            st.error("Database not initialized. Please generate sample data first.")
            return
        
        # Key Metrics Row
        self.display_key_metrics()
        
        st.markdown("---")
        
        # Main Content Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔮 Forecasts", "📈 Historical Analysis", "💳 AR/AP Analysis", "📋 Scenarios"])
        
        with tab1:
            self.display_overview_tab()
        
        with tab2:
            self.display_forecasts_tab(forecast_days, model_type)
        
        with tab3:
            self.display_historical_analysis_tab()
        
        with tab4:
            self.display_ar_ap_tab()
        
        with tab5:
            self.display_scenarios_tab()
    
    def display_key_metrics(self):
        """Display key financial metrics"""
        # Get recent cash flow data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        conn = sqlite3.connect(self.db.db_path)
        
        # Recent cash flows
        recent_query = """
            SELECT 
                SUM(CASE WHEN transaction_type = 'inflow' THEN amount ELSE 0 END) as total_inflows,
                SUM(CASE WHEN transaction_type = 'outflow' THEN amount ELSE 0 END) as total_outflows
            FROM cash_transactions
            WHERE transaction_date >= date('now', '-30 days')
        """
        recent_flows = pd.read_sql_query(recent_query, conn)
        
        # Outstanding receivables and payables
        ar_query = "SELECT SUM(amount - amount_paid) as outstanding_ar FROM accounts_receivable WHERE status IN ('outstanding', 'partial', 'overdue')"
        ap_query = "SELECT SUM(amount - amount_paid) as outstanding_ap FROM accounts_payable WHERE status IN ('outstanding', 'partial', 'overdue')"
        
        ar_data = pd.read_sql_query(ar_query, conn)
        ap_data = pd.read_sql_query(ap_query, conn)
        
        conn.close()
        
        # Calculate metrics
        total_inflows = recent_flows['total_inflows'].iloc[0] or 0
        total_outflows = recent_flows['total_outflows'].iloc[0] or 0
        net_flow = total_inflows - total_outflows
        outstanding_ar = ar_data['outstanding_ar'].iloc[0] or 0
        outstanding_ap = ap_data['outstanding_ap'].iloc[0] or 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "30-Day Inflows",
                f"${total_inflows:,.0f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "30-Day Outflows", 
                f"${total_outflows:,.0f}",
                delta=None
            )
        
        with col3:
            delta_color = "normal" if net_flow >= 0 else "inverse"
            st.metric(
                "Net Cash Flow",
                f"${net_flow:,.0f}",
                delta=f"{'Positive' if net_flow >= 0 else 'Negative'} Flow"
            )
        
        with col4:
            st.metric(
                "Outstanding A/R",
                f"${outstanding_ar:,.0f}",
                delta=None
            )
        
        with col5:
            st.metric(
                "Outstanding A/P",
                f"${outstanding_ap:,.0f}",
                delta=None
            )
    
    def display_overview_tab(self):
        """Display overview charts and summary"""
        st.subheader("📊 Cash Flow Overview")
        
        # Get historical data for charts
        conn = sqlite3.connect(self.db.db_path)
        
        # Monthly cash flow trends
        monthly_query = """
            SELECT 
                strftime('%Y-%m', transaction_date) as month,
                SUM(CASE WHEN transaction_type = 'inflow' THEN amount ELSE 0 END) as inflows,
                SUM(CASE WHEN transaction_type = 'outflow' THEN amount ELSE 0 END) as outflows
            FROM cash_transactions
            WHERE transaction_date >= date('now', '-12 months')
            GROUP BY strftime('%Y-%m', transaction_date)
            ORDER BY month
        """
        monthly_data = pd.read_sql_query(monthly_query, conn)
        
        if not monthly_data.empty:
            monthly_data['net_flow'] = monthly_data['inflows'] - monthly_data['outflows']
            monthly_data['cumulative_flow'] = monthly_data['net_flow'].cumsum()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly trends chart
                fig = go.Figure()
                fig.add_trace(go.Bar(x=monthly_data['month'], y=monthly_data['inflows'], 
                                   name='Inflows', marker_color='green', opacity=0.7))
                fig.add_trace(go.Bar(x=monthly_data['month'], y=-monthly_data['outflows'], 
                                   name='Outflows', marker_color='red', opacity=0.7))
                fig.add_trace(go.Scatter(x=monthly_data['month'], y=monthly_data['net_flow'], 
                                       name='Net Flow', line=dict(color='blue', width=3)))
                
                fig.update_layout(title="Monthly Cash Flow Trends", 
                                xaxis_title="Month", yaxis_title="Amount ($)",
                                height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cumulative cash flow
                fig = px.line(monthly_data, x='month', y='cumulative_flow',
                            title="Cumulative Cash Flow", 
                            labels={'cumulative_flow': 'Cumulative Flow ($)', 'month': 'Month'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        category_query = """
            SELECT 
                category,
                transaction_type,
                SUM(amount) as total_amount
            FROM cash_transactions
            WHERE transaction_date >= date('now', '-90 days')
            GROUP BY category, transaction_type
            ORDER BY total_amount DESC
        """
        category_data = pd.read_sql_query(category_query, conn)
        conn.close()
        
        if not category_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Inflows by category
                inflow_data = category_data[category_data['transaction_type'] == 'inflow']
                if not inflow_data.empty:
                    fig = px.pie(inflow_data, values='total_amount', names='category',
                               title="Inflows by Category (Last 90 Days)")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Outflows by category
                outflow_data = category_data[category_data['transaction_type'] == 'outflow']
                if not outflow_data.empty:
                    fig = px.pie(outflow_data, values='total_amount', names='category',
                               title="Outflows by Category (Last 90 Days)")
                    st.plotly_chart(fig, use_container_width=True)
    
    def display_forecasts_tab(self, forecast_days, model_type):
        """Display forecasting results"""
        st.subheader("🔮 Cash Flow Forecasts")
        
        if 'forecast_result' in st.session_state:
            forecast_data = st.session_state['forecast_result']
            
            if forecast_data.get('forecast_generated', False):
                # Display forecast summary
                summary = forecast_data['summary_statistics']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predicted Inflows", f"${summary['total_predicted_inflows']:,.0f}")
                
                with col2:
                    st.metric("Total Predicted Outflows", f"${summary['total_predicted_outflows']:,.0f}")
                
                with col3:
                    net_flow = summary['total_predicted_net_flow']
                    st.metric("Net Cash Flow", f"${net_flow:,.0f}", 
                            delta="Positive" if net_flow > 0 else "Negative")
                
                with col4:
                    st.metric("Average Daily Net", f"${summary['average_daily_net_flow']:,.0f}")
                
                # Forecast charts
                forecasts_df = pd.DataFrame(forecast_data['daily_forecasts'])
                forecasts_df['date'] = pd.to_datetime(forecasts_df['date'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Daily forecast chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=forecasts_df['date'], y=forecasts_df['predicted_inflows'],
                                           name='Predicted Inflows', line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=forecasts_df['date'], y=forecasts_df['predicted_outflows'],
                                           name='Predicted Outflows', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=forecasts_df['date'], y=forecasts_df['predicted_net_flow'],
                                           name='Net Flow', line=dict(color='blue', width=3)))
                    
                    fig.update_layout(title="Daily Cash Flow Forecast", 
                                    xaxis_title="Date", yaxis_title="Amount ($)",
                                    height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Cumulative cash flow forecast
                    fig = px.line(forecasts_df, x='date', y='cumulative_cash_flow',
                                title="Cumulative Cash Flow Forecast",
                                labels={'cumulative_cash_flow': 'Cumulative Cash ($)', 'date': 'Date'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Weekly and monthly summaries
                st.subheader("Forecast Summaries")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Weekly Summary**")
                    weekly_df = pd.DataFrame.from_dict(forecast_data['weekly_summary'], orient='index')
                    if not weekly_df.empty:
                        weekly_df = weekly_df.round(0).astype(int)
                        st.dataframe(weekly_df, use_container_width=True)
                
                with col2:
                    st.write("**Monthly Summary**")
                    monthly_df = pd.DataFrame.from_dict(forecast_data['monthly_summary'], orient='index')
                    if not monthly_df.empty:
                        monthly_df = monthly_df.round(0).astype(int)
                        st.dataframe(monthly_df, use_container_width=True)
                
                # Confidence levels
                st.subheader("Forecast Confidence")
                avg_confidence = forecasts_df['confidence_level'].mean()
                st.write(f"Average Confidence Level: {avg_confidence:.1%}")
                
                fig = px.line(forecasts_df, x='date', y='confidence_level',
                            title="Forecast Confidence Over Time",
                            labels={'confidence_level': 'Confidence Level', 'date': 'Date'})
                fig.update_yaxis(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"Forecast generation failed: {forecast_data.get('error', 'Unknown error')}")
        else:
            st.info("Click 'Generate Forecast' in the sidebar to create cash flow predictions.")
    
    def display_historical_analysis_tab(self):
        """Display historical pattern analysis"""
        st.subheader("📈 Historical Pattern Analysis")
        
        if 'pattern_analysis' in st.session_state:
            analysis = st.session_state['pattern_analysis']
            
            if analysis.get('summary_statistics'):
                # Summary statistics
                stats = analysis['summary_statistics']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", f"{stats['total_transactions']:,}")
                
                with col2:
                    st.metric("Average Daily Inflow", f"${stats['average_inflow']:,.0f}")
                
                with col3:
                    st.metric("Average Daily Outflow", f"${stats['average_outflow']:,.0f}")
                
                with col4:
                    net_flow = stats['net_cash_flow']
                    st.metric("Net Cash Flow", f"${net_flow:,.0f}")
                
                # Monthly patterns
                if analysis.get('monthly_patterns'):
                    st.subheader("Monthly Patterns")
                    monthly_data = analysis['monthly_patterns']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Average Monthly Inflow:** ${monthly_data['avg_monthly_inflow']:,.0f}")
                        st.write(f"**Average Monthly Outflow:** ${monthly_data['avg_monthly_outflow']:,.0f}")
                        st.write(f"**Average Monthly Net:** ${monthly_data['avg_monthly_net']:,.0f}")
                        st.write(f"**Monthly Volatility:** ${monthly_data['monthly_volatility']:,.0f}")
                    
                    with col2:
                        if monthly_data.get('monthly_data'):
                            monthly_df = pd.DataFrame.from_dict(monthly_data['monthly_data'], orient='index')
                            if not monthly_df.empty:
                                fig = px.line(x=monthly_df.index, y=monthly_df['net_flow'],
                                            title="Monthly Net Cash Flow Trend")
                                st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal patterns
                if analysis.get('seasonal_patterns'):
                    st.subheader("Seasonal Patterns")
                    seasonal = analysis['seasonal_patterns']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if seasonal.get('best_month'):
                            st.write(f"**Best Month:** {seasonal['best_month']}")
                        if seasonal.get('worst_month'):
                            st.write(f"**Worst Month:** {seasonal['worst_month']}")
                        if seasonal.get('seasonal_amplitude'):
                            st.write(f"**Seasonal Range:** ${seasonal['seasonal_amplitude']:,.0f}")
                    
                    with col2:
                        if seasonal.get('monthly_seasonal'):
                            seasonal_df = pd.DataFrame.from_dict(seasonal['monthly_seasonal'], orient='index')
                            if not seasonal_df.empty and 'seasonal_index' in seasonal_df.columns:
                                fig = px.bar(x=seasonal_df.index, y=seasonal_df['seasonal_index'],
                                           title="Seasonal Index by Month")
                                st.plotly_chart(fig, use_container_width=True)
                
                # Risk analysis
                if analysis.get('volatility_analysis'):
                    st.subheader("Risk & Volatility Analysis")
                    vol = analysis['volatility_analysis']
                    
                    if vol.get('volatility_calculated'):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Weekly Volatility:** ${vol['weekly_volatility']:,.0f}")
                            st.write(f"**Risk Level:** {vol['risk_level']}")
                        
                        with col2:
                            st.write(f"**Coefficient of Variation:** {vol['coefficient_of_variation']:.2f}")
                            st.write(f"**Negative Weeks:** {vol['negative_weeks']}")
                        
                        with col3:
                            st.write(f"**Min Weekly Flow:** ${vol['min_weekly_flow']:,.0f}")
                            st.write(f"**Max Weekly Flow:** ${vol['max_weekly_flow']:,.0f}")
            
        else:
            st.info("Click 'Run Pattern Analysis' in the sidebar to analyze historical patterns.")
    
    def display_ar_ap_tab(self):
        """Display accounts receivable and payable analysis"""
        st.subheader("💳 Accounts Receivable & Payable Analysis")
        
        if 'ar_analysis' in st.session_state and 'ap_analysis' in st.session_state:
            ar_analysis = st.session_state['ar_analysis']
            ap_analysis = st.session_state['ap_analysis']
            
            # AR/AP Summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Accounts Receivable")
                if ar_analysis.get('summary'):
                    ar_summary = ar_analysis['summary']
                    st.metric("Total Outstanding", f"${ar_summary['total_outstanding']:,.0f}")
                    st.metric("Collection Rate", f"{ar_summary['collection_rate']:.1f}%")
                    st.metric("Average Days Outstanding", f"{ar_summary['average_days_outstanding']:.0f} days")
                    
                    # AR Aging
                    if ar_analysis.get('aging_analysis'):
                        aging_data = ar_analysis['aging_analysis']
                        aging_df = pd.DataFrame.from_dict(aging_data, orient='index')
                        if not aging_df.empty:
                            fig = px.pie(values=aging_df['amount'], names=aging_df.index,
                                       title="AR Aging Distribution")
                            st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### Accounts Payable")
                if ap_analysis.get('summary'):
                    ap_summary = ap_analysis['summary']
                    st.metric("Total Outstanding", f"${ap_summary['total_outstanding']:,.0f}")
                    st.metric("Payment Rate", f"{ap_summary['payment_rate']:.1f}%")
                    st.metric("Due in 30 Days", f"${ap_summary['due_in_30_days']:,.0f}")
                    
                    # AP Aging
                    if ap_analysis.get('aging_analysis'):
                        aging_data = ap_analysis['aging_analysis']
                        aging_df = pd.DataFrame.from_dict(aging_data, orient='index')
                        if not aging_df.empty:
                            fig = px.pie(values=aging_df['amount'], names=aging_df.index,
                                       title="AP Aging Distribution")
                            st.plotly_chart(fig, use_container_width=True)
            
            # Collection predictions
            if ar_analysis.get('payment_predictions'):
                st.subheader("Collection Predictions")
                predictions = ar_analysis['payment_predictions']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Total Expected Collections:** ${predictions['total_expected_collections']:,.0f}")
                    st.write(f"**Average Collection Probability:** {predictions['average_collection_probability']:.1%}")
                
                with col2:
                    if predictions.get('weekly_collections'):
                        weekly_collections = pd.Series(predictions['weekly_collections'])
                        fig = px.bar(x=weekly_collections.index, y=weekly_collections.values,
                                   title="Predicted Weekly Collections")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Payment schedule
            if ap_analysis.get('payment_schedule'):
                st.subheader("Payment Schedule")
                schedule = ap_analysis['payment_schedule']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Total Payment Required:** ${schedule['total_payment_required']:,.0f}")
                    st.write(f"**High Priority Amount:** ${schedule['high_priority_amount']:,.0f}")
                
                with col2:
                    if schedule.get('weekly_payment_requirements'):
                        weekly_payments = pd.Series(schedule['weekly_payment_requirements'])
                        fig = px.bar(x=weekly_payments.index, y=weekly_payments.values,
                                   title="Weekly Payment Requirements")
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Click 'Analyze AR/AP' in the sidebar to analyze receivables and payables.")
    
    def display_scenarios_tab(self):
        """Display scenario analysis"""
        st.subheader("📋 Scenario Analysis")
        
        if st.button("Generate Scenario Analysis"):
            with st.spinner("Generating scenario-based forecasts..."):
                scenario_result = self.predictor.generate_scenario_analysis()
                
                if scenario_result.get('scenario_analysis_generated', False):
                    scenarios = scenario_result['scenarios']
                    
                    # Scenario comparison
                    scenario_summary = []
                    for scenario_name, scenario_data in scenarios.items():
                        scenario_summary.append({
                            'Scenario': scenario_name.title(),
                            'Total Net Flow': f"${scenario_data['total_net_flow']:,.0f}",
                            'Final Cash Position': f"${scenario_data['final_cash_position']:,.0f}"
                        })
                    
                    summary_df = pd.DataFrame(scenario_summary)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Scenario charts
                    fig = go.Figure()
                    
                    for scenario_name, scenario_data in scenarios.items():
                        forecasts_df = pd.DataFrame(scenario_data['forecasts'])
                        forecasts_df['date'] = pd.to_datetime(forecasts_df['date'])
                        
                        fig.add_trace(go.Scatter(
                            x=forecasts_df['date'],
                            y=forecasts_df['cumulative_cash_flow'],
                            name=scenario_name.title(),
                            mode='lines'
                        ))
                    
                    fig.update_layout(
                        title="Scenario Comparison - Cumulative Cash Flow",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Cash Flow ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk assessment
                    st.subheader("Risk Assessment")
                    
                    negative_scenarios = sum(1 for s in scenarios.values() if s['final_cash_position'] < 0)
                    total_scenarios = len(scenarios)
                    
                    if negative_scenarios > 0:
                        st.warning(f"⚠️ {negative_scenarios} out of {total_scenarios} scenarios show negative cash position")
                    else:
                        st.success("✅ All scenarios show positive cash flow")
                
                else:
                    st.error("Failed to generate scenario analysis")
        
        else:
            st.info("Click the button above to generate scenario-based forecasts comparing optimistic, realistic, and pessimistic outcomes.")

def main():
    dashboard = CashFlowDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()