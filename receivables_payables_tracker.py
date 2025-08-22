import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from scipy import stats

class ReceivablesPayablesTracker:
    def __init__(self, db_path="cashflow_forecast.db"):
        self.db_path = db_path
    
    def analyze_receivables_patterns(self):
        """Analyze accounts receivable payment patterns"""
        conn = sqlite3.connect(self.db_path)
        
        # Get receivables data with customer information
        query = """
            SELECT 
                ar.*,
                c.payment_history_score,
                c.average_days_to_pay,
                c.industry
            FROM accounts_receivable ar
            LEFT JOIN customers c ON ar.customer_id = c.ROWID
            ORDER BY ar.invoice_date DESC
        """
        
        ar_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if ar_df.empty:
            return self._empty_receivables_analysis()
        
        ar_df['invoice_date'] = pd.to_datetime(ar_df['invoice_date'])
        ar_df['due_date'] = pd.to_datetime(ar_df['due_date'])
        ar_df['days_outstanding'] = (datetime.now() - ar_df['due_date']).dt.days
        ar_df['outstanding_amount'] = ar_df['amount'] - ar_df['amount_paid']
        
        analysis = {
            'summary': self._calculate_receivables_summary(ar_df),
            'aging_analysis': self._analyze_receivables_aging(ar_df),
            'customer_analysis': self._analyze_customer_patterns(ar_df),
            'payment_predictions': self._predict_receivables_collections(ar_df),
            'risk_assessment': self._assess_collection_risk(ar_df)
        }
        
        return analysis
    
    def analyze_payables_patterns(self):
        """Analyze accounts payable payment patterns"""
        conn = sqlite3.connect(self.db_path)
        
        # Get payables data with vendor information
        query = """
            SELECT 
                ap.*,
                v.payment_terms as vendor_payment_terms,
                v.average_monthly_spend,
                v.category as vendor_category
            FROM accounts_payable ap
            LEFT JOIN vendors v ON ap.vendor_id = v.ROWID
            ORDER BY ap.invoice_date DESC
        """
        
        ap_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if ap_df.empty:
            return self._empty_payables_analysis()
        
        ap_df['invoice_date'] = pd.to_datetime(ap_df['invoice_date'])
        ap_df['due_date'] = pd.to_datetime(ap_df['due_date'])
        ap_df['days_until_due'] = (ap_df['due_date'] - datetime.now()).dt.days
        ap_df['outstanding_amount'] = ap_df['amount'] - ap_df['amount_paid']
        
        analysis = {
            'summary': self._calculate_payables_summary(ap_df),
            'aging_analysis': self._analyze_payables_aging(ap_df),
            'vendor_analysis': self._analyze_vendor_patterns(ap_df),
            'payment_schedule': self._create_payment_schedule(ap_df),
            'cash_requirements': self._calculate_cash_requirements(ap_df)
        }
        
        return analysis
    
    def _empty_receivables_analysis(self):
        """Return empty receivables analysis structure"""
        return {
            'summary': {},
            'aging_analysis': {},
            'customer_analysis': {},
            'payment_predictions': {},
            'risk_assessment': {}
        }
    
    def _empty_payables_analysis(self):
        """Return empty payables analysis structure"""
        return {
            'summary': {},
            'aging_analysis': {},
            'vendor_analysis': {},
            'payment_schedule': {},
            'cash_requirements': {}
        }
    
    def _calculate_receivables_summary(self, ar_df):
        """Calculate receivables summary statistics"""
        outstanding_df = ar_df[ar_df['status'].isin(['outstanding', 'partial', 'overdue'])]
        
        return {
            'total_receivables': float(ar_df['amount'].sum()),
            'total_outstanding': float(outstanding_df['outstanding_amount'].sum()),
            'total_collected': float(ar_df['amount_paid'].sum()),
            'collection_rate': float(ar_df['amount_paid'].sum() / ar_df['amount'].sum() * 100) if ar_df['amount'].sum() > 0 else 0,
            'number_of_invoices': len(ar_df),
            'outstanding_invoices': len(outstanding_df),
            'average_invoice_amount': float(ar_df['amount'].mean()),
            'average_days_outstanding': float(outstanding_df['days_outstanding'].mean()) if len(outstanding_df) > 0 else 0,
            'overdue_amount': float(outstanding_df[outstanding_df['days_outstanding'] > 0]['outstanding_amount'].sum()),
            'current_amount': float(outstanding_df[outstanding_df['days_outstanding'] <= 0]['outstanding_amount'].sum())
        }
    
    def _analyze_receivables_aging(self, ar_df):
        """Analyze receivables by aging buckets"""
        outstanding_df = ar_df[ar_df['status'].isin(['outstanding', 'partial', 'overdue'])]
        
        if outstanding_df.empty:
            return {}
        
        # Define aging buckets
        aging_buckets = {
            'current': (outstanding_df['days_outstanding'] <= 0),
            '1_30_days': (outstanding_df['days_outstanding'] > 0) & (outstanding_df['days_outstanding'] <= 30),
            '31_60_days': (outstanding_df['days_outstanding'] > 30) & (outstanding_df['days_outstanding'] <= 60),
            '61_90_days': (outstanding_df['days_outstanding'] > 60) & (outstanding_df['days_outstanding'] <= 90),
            '91_120_days': (outstanding_df['days_outstanding'] > 90) & (outstanding_df['days_outstanding'] <= 120),
            'over_120_days': (outstanding_df['days_outstanding'] > 120)
        }
        
        aging_analysis = {}
        total_outstanding = outstanding_df['outstanding_amount'].sum()
        
        for bucket, condition in aging_buckets.items():
            bucket_data = outstanding_df[condition]
            aging_analysis[bucket] = {
                'amount': float(bucket_data['outstanding_amount'].sum()),
                'percentage': float(bucket_data['outstanding_amount'].sum() / total_outstanding * 100) if total_outstanding > 0 else 0,
                'count': len(bucket_data),
                'average_amount': float(bucket_data['outstanding_amount'].mean()) if len(bucket_data) > 0 else 0
            }
        
        return aging_analysis
    
    def _analyze_customer_patterns(self, ar_df):
        """Analyze payment patterns by customer"""
        customer_stats = ar_df.groupby('customer_id').agg({
            'amount': ['sum', 'count', 'mean'],
            'amount_paid': 'sum',
            'payment_history_score': 'first',
            'average_days_to_pay': 'first',
            'industry': 'first',
            'customer_name': 'first'
        }).round(2)
        
        customer_stats.columns = ['total_invoiced', 'invoice_count', 'avg_invoice', 'total_paid', 
                                'payment_score', 'avg_days_to_pay', 'industry', 'customer_name']
        customer_stats = customer_stats.reset_index()
        
        # Calculate customer-specific metrics
        customer_stats['collection_rate'] = (customer_stats['total_paid'] / customer_stats['total_invoiced'] * 100).round(2)
        customer_stats['outstanding_amount'] = customer_stats['total_invoiced'] - customer_stats['total_paid']
        
        # Rank customers by risk
        customer_stats['risk_score'] = (
            (100 - customer_stats['payment_score']) * 0.4 +
            (customer_stats['avg_days_to_pay'] / 90 * 100) * 0.3 +
            ((100 - customer_stats['collection_rate']) * 0.3)
        ).round(2)
        
        return {
            'customer_statistics': customer_stats.to_dict('records'),
            'top_customers_by_amount': customer_stats.nlargest(5, 'total_invoiced').to_dict('records'),
            'highest_risk_customers': customer_stats.nlargest(5, 'risk_score').to_dict('records'),
            'best_paying_customers': customer_stats.nsmallest(5, 'risk_score').to_dict('records')
        }
    
    def _predict_receivables_collections(self, ar_df):
        """Predict when receivables will be collected"""
        outstanding_df = ar_df[ar_df['status'].isin(['outstanding', 'partial', 'overdue'])]
        
        if outstanding_df.empty:
            return {}
        
        predictions = []
        
        for _, invoice in outstanding_df.iterrows():
            # Base prediction on customer's historical payment behavior
            expected_days_to_pay = invoice['average_days_to_pay'] if pd.notna(invoice['average_days_to_pay']) else 30
            payment_score = invoice['payment_history_score'] if pd.notna(invoice['payment_history_score']) else 75
            
            # Adjust based on how long it's been outstanding
            days_outstanding = invoice['days_outstanding']
            if days_outstanding > 0:
                # Already overdue, extend expected collection time
                expected_days_to_pay = expected_days_to_pay + (days_outstanding * 0.5)
            
            # Calculate collection probability
            if days_outstanding <= 30:
                collection_probability = min(0.95, payment_score / 100)
            elif days_outstanding <= 60:
                collection_probability = min(0.8, payment_score / 100 * 0.8)
            elif days_outstanding <= 90:
                collection_probability = min(0.6, payment_score / 100 * 0.6)
            else:
                collection_probability = min(0.3, payment_score / 100 * 0.4)
            
            # Predicted collection date
            if days_outstanding > 0:
                predicted_collection_date = datetime.now() + timedelta(days=max(7, expected_days_to_pay - days_outstanding))
            else:
                predicted_collection_date = invoice['due_date'] + timedelta(days=expected_days_to_pay - invoice['payment_terms'])
            
            predictions.append({
                'invoice_id': invoice.name,  # Use row index as ID
                'customer_name': invoice['customer_name'],
                'invoice_number': invoice['invoice_number'],
                'outstanding_amount': float(invoice['outstanding_amount']),
                'days_outstanding': int(days_outstanding),
                'predicted_collection_date': predicted_collection_date.date(),
                'collection_probability': round(collection_probability, 3),
                'expected_amount': round(float(invoice['outstanding_amount'] * collection_probability), 2)
            })
        
        # Aggregate predictions by week/month
        predictions_df = pd.DataFrame(predictions)
        predictions_df['predicted_collection_date'] = pd.to_datetime(predictions_df['predicted_collection_date'])
        predictions_df['week'] = predictions_df['predicted_collection_date'].dt.to_period('W')
        predictions_df['month'] = predictions_df['predicted_collection_date'].dt.to_period('M')
        
        weekly_collections = predictions_df.groupby('week')['expected_amount'].sum().round(2)
        monthly_collections = predictions_df.groupby('month')['expected_amount'].sum().round(2)
        
        return {
            'individual_predictions': predictions,
            'weekly_collections': weekly_collections.to_dict(),
            'monthly_collections': monthly_collections.to_dict(),
            'total_expected_collections': float(predictions_df['expected_amount'].sum()),
            'average_collection_probability': float(predictions_df['collection_probability'].mean())
        }
    
    def _assess_collection_risk(self, ar_df):
        """Assess collection risk for outstanding receivables"""
        outstanding_df = ar_df[ar_df['status'].isin(['outstanding', 'partial', 'overdue'])]
        
        if outstanding_df.empty:
            return {}
        
        # Define risk categories
        risk_categories = {
            'low_risk': (outstanding_df['days_outstanding'] <= 30) & (outstanding_df['payment_history_score'] >= 80),
            'medium_risk': ((outstanding_df['days_outstanding'] > 30) & (outstanding_df['days_outstanding'] <= 60)) | 
                          ((outstanding_df['payment_history_score'] >= 60) & (outstanding_df['payment_history_score'] < 80)),
            'high_risk': (outstanding_df['days_outstanding'] > 60) & (outstanding_df['payment_history_score'] < 60),
            'critical_risk': (outstanding_df['days_outstanding'] > 120)
        }
        
        risk_assessment = {}
        total_outstanding = outstanding_df['outstanding_amount'].sum()
        
        for risk_level, condition in risk_categories.items():
            risk_data = outstanding_df[condition]
            risk_assessment[risk_level] = {
                'amount': float(risk_data['outstanding_amount'].sum()),
                'percentage': float(risk_data['outstanding_amount'].sum() / total_outstanding * 100) if total_outstanding > 0 else 0,
                'count': len(risk_data),
                'invoices': risk_data[['customer_name', 'invoice_number', 'outstanding_amount', 'days_outstanding']].to_dict('records')
            }
        
        return risk_assessment
    
    def _calculate_payables_summary(self, ap_df):
        """Calculate payables summary statistics"""
        outstanding_df = ap_df[ap_df['status'].isin(['outstanding', 'partial', 'overdue'])]
        
        return {
            'total_payables': float(ap_df['amount'].sum()),
            'total_outstanding': float(outstanding_df['outstanding_amount'].sum()),
            'total_paid': float(ap_df['amount_paid'].sum()),
            'payment_rate': float(ap_df['amount_paid'].sum() / ap_df['amount'].sum() * 100) if ap_df['amount'].sum() > 0 else 0,
            'number_of_invoices': len(ap_df),
            'outstanding_invoices': len(outstanding_df),
            'average_invoice_amount': float(ap_df['amount'].mean()),
            'due_in_30_days': float(outstanding_df[outstanding_df['days_until_due'] <= 30]['outstanding_amount'].sum()),
            'due_in_60_days': float(outstanding_df[outstanding_df['days_until_due'] <= 60]['outstanding_amount'].sum()),
            'overdue_amount': float(outstanding_df[outstanding_df['days_until_due'] < 0]['outstanding_amount'].sum())
        }
    
    def _analyze_payables_aging(self, ap_df):
        """Analyze payables by due date buckets"""
        outstanding_df = ap_df[ap_df['status'].isin(['outstanding', 'partial', 'overdue'])]
        
        if outstanding_df.empty:
            return {}
        
        # Define aging buckets based on days until due
        aging_buckets = {
            'overdue': (outstanding_df['days_until_due'] < 0),
            'due_this_week': (outstanding_df['days_until_due'] >= 0) & (outstanding_df['days_until_due'] <= 7),
            'due_in_30_days': (outstanding_df['days_until_due'] > 7) & (outstanding_df['days_until_due'] <= 30),
            'due_in_60_days': (outstanding_df['days_until_due'] > 30) & (outstanding_df['days_until_due'] <= 60),
            'due_in_90_days': (outstanding_df['days_until_due'] > 60) & (outstanding_df['days_until_due'] <= 90),
            'due_later': (outstanding_df['days_until_due'] > 90)
        }
        
        aging_analysis = {}
        total_outstanding = outstanding_df['outstanding_amount'].sum()
        
        for bucket, condition in aging_buckets.items():
            bucket_data = outstanding_df[condition]
            aging_analysis[bucket] = {
                'amount': float(bucket_data['outstanding_amount'].sum()),
                'percentage': float(bucket_data['outstanding_amount'].sum() / total_outstanding * 100) if total_outstanding > 0 else 0,
                'count': len(bucket_data),
                'average_amount': float(bucket_data['outstanding_amount'].mean()) if len(bucket_data) > 0 else 0
            }
        
        return aging_analysis
    
    def _analyze_vendor_patterns(self, ap_df):
        """Analyze payment patterns by vendor"""
        vendor_stats = ap_df.groupby('vendor_id').agg({
            'amount': ['sum', 'count', 'mean'],
            'amount_paid': 'sum',
            'payment_terms': 'first',
            'average_monthly_spend': 'first',
            'vendor_category': 'first',
            'vendor_name': 'first'
        }).round(2)
        
        vendor_stats.columns = ['total_invoiced', 'invoice_count', 'avg_invoice', 'total_paid', 
                               'payment_terms', 'avg_monthly_spend', 'category', 'vendor_name']
        vendor_stats = vendor_stats.reset_index()
        
        # Calculate vendor-specific metrics
        vendor_stats['payment_rate'] = (vendor_stats['total_paid'] / vendor_stats['total_invoiced'] * 100).round(2)
        vendor_stats['outstanding_amount'] = vendor_stats['total_invoiced'] - vendor_stats['total_paid']
        
        return {
            'vendor_statistics': vendor_stats.to_dict('records'),
            'top_vendors_by_amount': vendor_stats.nlargest(5, 'total_invoiced').to_dict('records'),
            'largest_outstanding': vendor_stats.nlargest(5, 'outstanding_amount').to_dict('records')
        }
    
    def _create_payment_schedule(self, ap_df):
        """Create optimal payment schedule"""
        outstanding_df = ap_df[ap_df['status'].isin(['outstanding', 'partial', 'overdue'])]
        
        if outstanding_df.empty:
            return {}
        
        # Sort by priority: overdue first, then by due date
        outstanding_df['priority_score'] = outstanding_df.apply(lambda x: 
            1000 - x['days_until_due'] if x['days_until_due'] < 0 else x['days_until_due'], axis=1)
        
        outstanding_df = outstanding_df.sort_values('priority_score')
        
        schedule = []
        for _, invoice in outstanding_df.iterrows():
            payment_date = max(datetime.now().date(), invoice['due_date'].date())
            
            schedule.append({
                'vendor_name': invoice['vendor_name'],
                'invoice_number': invoice['invoice_number'],
                'amount': float(invoice['outstanding_amount']),
                'due_date': invoice['due_date'].date(),
                'recommended_payment_date': payment_date,
                'days_until_due': int(invoice['days_until_due']),
                'priority': 'High' if invoice['days_until_due'] < 7 else 'Medium' if invoice['days_until_due'] < 30 else 'Low'
            })
        
        # Group by week for cash flow planning
        schedule_df = pd.DataFrame(schedule)
        schedule_df['recommended_payment_date'] = pd.to_datetime(schedule_df['recommended_payment_date'])
        schedule_df['week'] = schedule_df['recommended_payment_date'].dt.to_period('W')
        
        weekly_payments = schedule_df.groupby('week')['amount'].sum().round(2)
        
        return {
            'payment_schedule': schedule,
            'weekly_payment_requirements': weekly_payments.to_dict(),
            'total_payment_required': float(schedule_df['amount'].sum()),
            'high_priority_amount': float(schedule_df[schedule_df['priority'] == 'High']['amount'].sum())
        }
    
    def _calculate_cash_requirements(self, ap_df):
        """Calculate future cash requirements"""
        outstanding_df = ap_df[ap_df['status'].isin(['outstanding', 'partial', 'overdue'])]
        
        if outstanding_df.empty:
            return {}
        
        # Calculate requirements by time periods
        today = datetime.now()
        
        requirements = {
            'immediate': float(outstanding_df[outstanding_df['days_until_due'] < 0]['outstanding_amount'].sum()),
            'next_7_days': float(outstanding_df[outstanding_df['days_until_due'].between(0, 7)]['outstanding_amount'].sum()),
            'next_30_days': float(outstanding_df[outstanding_df['days_until_due'].between(0, 30)]['outstanding_amount'].sum()),
            'next_60_days': float(outstanding_df[outstanding_df['days_until_due'].between(0, 60)]['outstanding_amount'].sum()),
            'next_90_days': float(outstanding_df[outstanding_df['days_until_due'].between(0, 90)]['outstanding_amount'].sum()),
            'total_outstanding': float(outstanding_df['outstanding_amount'].sum())
        }
        
        # Calculate cumulative requirements
        requirements['cumulative_30_days'] = requirements['immediate'] + requirements['next_30_days']
        requirements['cumulative_60_days'] = requirements['immediate'] + requirements['next_60_days']
        requirements['cumulative_90_days'] = requirements['immediate'] + requirements['next_90_days']
        
        return requirements
    
    def get_cash_flow_impact(self):
        """Calculate the net cash flow impact from receivables and payables"""
        receivables_analysis = self.analyze_receivables_patterns()
        payables_analysis = self.analyze_payables_patterns()
        
        # Get predicted collections and required payments
        predicted_collections = receivables_analysis.get('payment_predictions', {})
        payment_requirements = payables_analysis.get('cash_requirements', {})
        
        # Calculate net cash flow for different time periods
        periods = ['next_7_days', 'next_30_days', 'next_60_days', 'next_90_days']
        cash_flow_impact = {}
        
        for period in periods:
            # Estimated collections (use prediction probability)
            if period in predicted_collections.get('weekly_collections', {}):
                collections = sum(predicted_collections['weekly_collections'].values()) / 13 * {
                    'next_7_days': 1, 'next_30_days': 4.3, 'next_60_days': 8.6, 'next_90_days': 13
                }[period]
            else:
                collections = receivables_analysis.get('summary', {}).get('total_outstanding', 0) * {
                    'next_7_days': 0.1, 'next_30_days': 0.4, 'next_60_days': 0.7, 'next_90_days': 0.9
                }[period]
            
            # Required payments
            payments = payment_requirements.get(f'cumulative_{period.split("_")[1]}_{period.split("_")[2]}', 0)
            
            cash_flow_impact[period] = {
                'expected_collections': round(collections, 2),
                'required_payments': round(payments, 2),
                'net_cash_flow': round(collections - payments, 2)
            }
        
        return {
            'cash_flow_impact': cash_flow_impact,
            'summary': {
                'total_receivables_outstanding': receivables_analysis.get('summary', {}).get('total_outstanding', 0),
                'total_payables_outstanding': payables_analysis.get('summary', {}).get('total_outstanding', 0),
                'net_working_capital': receivables_analysis.get('summary', {}).get('total_outstanding', 0) - 
                                     payables_analysis.get('summary', {}).get('total_outstanding', 0)
            }
        }