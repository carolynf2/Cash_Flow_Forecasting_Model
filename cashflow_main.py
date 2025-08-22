#!/usr/bin/env python3
"""
Cash Flow Forecasting System - Main Entry Point
"""

import argparse
import sys
from cashflow_database import CashFlowDatabase
from pattern_analyzer import CashFlowPatternAnalyzer
from receivables_payables_tracker import ReceivablesPayablesTracker
from cashflow_predictor import CashFlowPredictor

def setup_database():
    """Initialize database and generate sample data"""
    print("Setting up cash flow forecasting database...")
    db = CashFlowDatabase()
    print("Database initialized successfully!")
    
    print("Generating comprehensive sample financial data...")
    db.generate_sample_data()
    print("Sample data generated successfully!")

def analyze_patterns():
    """Run historical pattern analysis"""
    print("Analyzing historical cash flow patterns...")
    analyzer = CashFlowPatternAnalyzer()
    analysis = analyzer.analyze_historical_patterns()
    
    if analysis['summary_statistics']:
        stats = analysis['summary_statistics']
        print("\n" + "="*60)
        print("HISTORICAL PATTERN ANALYSIS RESULTS")
        print("="*60)
        print(f"Analysis Period: {analysis['period_analyzed']['total_days']} days")
        print(f"Total Transactions: {stats['total_transactions']:,}")
        print(f"Total Inflows: ${stats['total_inflows']:,.2f}")
        print(f"Total Outflows: ${stats['total_outflows']:,.2f}")
        print(f"Net Cash Flow: ${stats['net_cash_flow']:,.2f}")
        print(f"Average Daily Inflow: ${stats['average_inflow']:,.2f}")
        print(f"Average Daily Outflow: ${stats['average_outflow']:,.2f}")
        
        if analysis['monthly_patterns']:
            monthly = analysis['monthly_patterns']
            print(f"\nMonthly Patterns:")
            print(f"Average Monthly Net Flow: ${monthly['avg_monthly_net']:,.2f}")
            print(f"Monthly Volatility: ${monthly['monthly_volatility']:,.2f}")
        
        if analysis['seasonal_patterns']:
            seasonal = analysis['seasonal_patterns']
            if seasonal.get('best_month') and seasonal.get('worst_month'):
                print(f"\nSeasonal Insights:")
                print(f"Best Month: {seasonal['best_month']}")
                print(f"Worst Month: {seasonal['worst_month']}")
        
        if analysis['volatility_analysis'].get('volatility_calculated'):
            vol = analysis['volatility_analysis']
            print(f"\nRisk Assessment:")
            print(f"Risk Level: {vol['risk_level']}")
            print(f"Weekly Volatility: ${vol['weekly_volatility']:,.2f}")
        
        print("="*60)
    else:
        print("No historical data available for analysis.")

def analyze_ar_ap():
    """Analyze accounts receivable and payable"""
    print("Analyzing accounts receivable and payable...")
    tracker = ReceivablesPayablesTracker()
    
    ar_analysis = tracker.analyze_receivables_patterns()
    ap_analysis = tracker.analyze_payables_patterns()
    
    print("\n" + "="*60)
    print("ACCOUNTS RECEIVABLE & PAYABLE ANALYSIS")
    print("="*60)
    
    if ar_analysis.get('summary'):
        ar_summary = ar_analysis['summary']
        print("Accounts Receivable:")
        print(f"  Total Outstanding: ${ar_summary['total_outstanding']:,.2f}")
        print(f"  Collection Rate: {ar_summary['collection_rate']:.1f}%")
        print(f"  Average Days Outstanding: {ar_summary['average_days_outstanding']:.0f} days")
        print(f"  Overdue Amount: ${ar_summary['overdue_amount']:,.2f}")
    
    if ap_analysis.get('summary'):
        ap_summary = ap_analysis['summary']
        print("\nAccounts Payable:")
        print(f"  Total Outstanding: ${ap_summary['total_outstanding']:,.2f}")
        print(f"  Payment Rate: {ap_summary['payment_rate']:.1f}%")
        print(f"  Due in 30 Days: ${ap_summary['due_in_30_days']:,.2f}")
        print(f"  Overdue Amount: ${ap_summary['overdue_amount']:,.2f}")
    
    # Cash flow impact
    cash_flow_impact = tracker.get_cash_flow_impact()
    if cash_flow_impact.get('summary'):
        summary = cash_flow_impact['summary']
        print(f"\nWorking Capital Analysis:")
        print(f"  Net Working Capital: ${summary['net_working_capital']:,.2f}")
        
        if cash_flow_impact.get('cash_flow_impact'):
            impact = cash_flow_impact['cash_flow_impact']
            print(f"\nCash Flow Projections:")
            for period, data in impact.items():
                period_name = period.replace('_', ' ').title()
                print(f"  {period_name}: ${data['net_cash_flow']:,.2f} net")
    
    print("="*60)

def generate_forecast(days=90, model='ensemble'):
    """Generate cash flow forecast"""
    print(f"Generating {days}-day cash flow forecast using {model} model...")
    predictor = CashFlowPredictor()
    
    # Build models first
    print("Building forecasting models...")
    model_result = predictor.build_forecasting_models()
    
    if not model_result.get('models_built', False):
        print(f"Model building failed: {model_result.get('error', 'Unknown error')}")
        return
    
    print(f"Models built successfully. Best model: {model_result['best_model']}")
    
    # Generate forecast
    forecast_result = predictor.generate_forecast(forecast_days=days, model_type=model)
    
    if forecast_result.get('forecast_generated', False):
        summary = forecast_result['summary_statistics']
        
        print("\n" + "="*60)
        print("CASH FLOW FORECAST RESULTS")
        print("="*60)
        print(f"Forecast Period: {days} days")
        print(f"Model Used: {model}")
        print(f"Start Date: {forecast_result['start_date']}")
        
        print(f"\nForecast Summary:")
        print(f"Total Predicted Inflows: ${summary['total_predicted_inflows']:,.2f}")
        print(f"Total Predicted Outflows: ${summary['total_predicted_outflows']:,.2f}")
        print(f"Total Net Cash Flow: ${summary['total_predicted_net_flow']:,.2f}")
        print(f"Average Daily Net Flow: ${summary['average_daily_net_flow']:,.2f}")
        print(f"Final Cumulative Cash: ${summary['final_cumulative_cash']:,.2f}")
        
        # Show weekly summary
        if forecast_result.get('weekly_summary'):
            print(f"\nWeekly Summary (First 4 weeks):")
            week_count = 0
            for week, data in forecast_result['weekly_summary'].items():
                if week_count >= 4:
                    break
                print(f"  Week {week}: ${data['predicted_net_flow']:,.2f} net flow")
                week_count += 1
        
        # Save forecast to database
        save_result = predictor.save_forecast_to_database(forecast_result, model)
        if save_result.get('saved', False):
            print(f"\nForecast saved to database ({save_result['records_saved']} records)")
        
        print("="*60)
    else:
        print(f"Forecast generation failed: {forecast_result.get('error', 'Unknown error')}")

def run_scenario_analysis():
    """Run scenario analysis"""
    print("Running scenario-based cash flow analysis...")
    predictor = CashFlowPredictor()
    
    scenario_result = predictor.generate_scenario_analysis()
    
    if scenario_result.get('scenario_analysis_generated', False):
        scenarios = scenario_result['scenarios']
        
        print("\n" + "="*60)
        print("SCENARIO ANALYSIS RESULTS")
        print("="*60)
        
        for scenario_name, scenario_data in scenarios.items():
            print(f"\n{scenario_name.upper()} SCENARIO:")
            print(f"  Total Net Flow: ${scenario_data['total_net_flow']:,.2f}")
            print(f"  Final Cash Position: ${scenario_data['final_cash_position']:,.2f}")
        
        # Risk assessment
        negative_scenarios = sum(1 for s in scenarios.values() if s['final_cash_position'] < 0)
        total_scenarios = len(scenarios)
        
        print(f"\nRisk Assessment:")
        if negative_scenarios > 0:
            print(f"  WARNING: {negative_scenarios} out of {total_scenarios} scenarios show negative cash position")
            print(f"  Risk Level: HIGH")
        else:
            print(f"  SUCCESS: All scenarios show positive cash flow")
            print(f"  Risk Level: LOW")
        
        print("="*60)
    else:
        print("Scenario analysis failed to generate.")

def run_dashboard():
    """Start the Streamlit dashboard"""
    print("Starting cash flow forecasting dashboard...")
    print("Navigate to http://localhost:8501 in your browser")
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "cashflow_dashboard.py"])

def main():
    parser = argparse.ArgumentParser(description="Cash Flow Forecasting System")
    parser.add_argument("--setup", action="store_true", help="Setup database and generate sample data")
    parser.add_argument("--analyze", action="store_true", help="Run historical pattern analysis")
    parser.add_argument("--ar-ap", action="store_true", help="Analyze accounts receivable and payable")
    parser.add_argument("--forecast", action="store_true", help="Generate cash flow forecast")
    parser.add_argument("--scenarios", action="store_true", help="Run scenario analysis")
    parser.add_argument("--dashboard", action="store_true", help="Start the web dashboard")
    parser.add_argument("--days", type=int, default=90, help="Forecast period in days (default: 90)")
    parser.add_argument("--model", choices=['ensemble', 'random_forest', 'linear_regression'], 
                       default='ensemble', help="Forecasting model to use")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_database()
    elif args.analyze:
        analyze_patterns()
    elif args.ar_ap:
        analyze_ar_ap()
    elif args.forecast:
        generate_forecast(args.days, args.model)
    elif args.scenarios:
        run_scenario_analysis()
    elif args.dashboard:
        run_dashboard()
    else:
        print("Cash Flow Forecasting System")
        print("Usage:")
        print("  python cashflow_main.py --setup       # Setup database and generate sample data")
        print("  python cashflow_main.py --analyze     # Run historical pattern analysis")
        print("  python cashflow_main.py --ar-ap       # Analyze accounts receivable and payable")
        print("  python cashflow_main.py --forecast    # Generate cash flow forecast")
        print("  python cashflow_main.py --scenarios   # Run scenario analysis")
        print("  python cashflow_main.py --dashboard   # Start the web dashboard")
        print("\nOptions:")
        print("  --days N          # Forecast period (default: 90 days)")
        print("  --model MODEL     # Forecasting model: ensemble, random_forest, linear_regression")
        print("\nFor full interactive experience, run: python cashflow_main.py --dashboard")

if __name__ == "__main__":
    main()