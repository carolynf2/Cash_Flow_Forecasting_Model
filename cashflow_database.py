import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np

class CashFlowDatabase:
    def __init__(self, db_path="cashflow_forecast.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cash_transactions table (historical cash flows)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cash_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_date DATE NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT,
                amount REAL NOT NULL,
                transaction_type TEXT NOT NULL CHECK (transaction_type IN ('inflow', 'outflow')),
                is_recurring INTEGER DEFAULT 0,
                recurrence_pattern TEXT,
                customer_id INTEGER,
                vendor_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create accounts_receivable table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accounts_receivable (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                invoice_number TEXT UNIQUE NOT NULL,
                invoice_date DATE NOT NULL,
                due_date DATE NOT NULL,
                amount REAL NOT NULL,
                amount_paid REAL DEFAULT 0,
                status TEXT DEFAULT 'outstanding' CHECK (status IN ('outstanding', 'partial', 'paid', 'overdue')),
                payment_terms INTEGER DEFAULT 30,
                customer_name TEXT NOT NULL,
                product_service TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create accounts_payable table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accounts_payable (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_id INTEGER NOT NULL,
                invoice_number TEXT NOT NULL,
                invoice_date DATE NOT NULL,
                due_date DATE NOT NULL,
                amount REAL NOT NULL,
                amount_paid REAL DEFAULT 0,
                status TEXT DEFAULT 'outstanding' CHECK (status IN ('outstanding', 'partial', 'paid', 'overdue')),
                payment_terms INTEGER DEFAULT 30,
                vendor_name TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create customers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT UNIQUE NOT NULL,
                contact_email TEXT,
                payment_history_score REAL DEFAULT 75.0,
                average_days_to_pay REAL DEFAULT 30.0,
                credit_limit REAL DEFAULT 10000.0,
                industry TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create vendors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vendors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_name TEXT UNIQUE NOT NULL,
                contact_email TEXT,
                category TEXT NOT NULL,
                payment_terms INTEGER DEFAULT 30,
                average_monthly_spend REAL DEFAULT 1000.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create cash_forecasts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cash_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_date DATE NOT NULL,
                forecast_period_start DATE NOT NULL,
                forecast_period_end DATE NOT NULL,
                predicted_inflows REAL NOT NULL,
                predicted_outflows REAL NOT NULL,
                net_cash_flow REAL NOT NULL,
                cumulative_cash REAL NOT NULL,
                confidence_level REAL DEFAULT 0.8,
                model_used TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create budget_items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budget_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                subcategory TEXT,
                budget_amount REAL NOT NULL,
                period_type TEXT NOT NULL CHECK (period_type IN ('monthly', 'quarterly', 'annual')),
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                is_fixed INTEGER DEFAULT 0,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create payment_patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payment_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                vendor_id INTEGER,
                pattern_type TEXT NOT NULL CHECK (pattern_type IN ('customer_payment', 'vendor_payment')),
                days_to_pay_avg REAL NOT NULL,
                days_to_pay_std REAL NOT NULL,
                payment_probability REAL NOT NULL,
                seasonal_factor REAL DEFAULT 1.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers (id),
                FOREIGN KEY (vendor_id) REFERENCES vendors (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_sample_data(self):
        """Generate comprehensive sample data for cash flow forecasting"""
        conn = sqlite3.connect(self.db_path)
        
        # Generate customers
        customers = [
            {'customer_name': 'ABC Manufacturing', 'contact_email': 'billing@abcmfg.com', 'payment_history_score': 85.0, 'average_days_to_pay': 25.0, 'credit_limit': 50000.0, 'industry': 'Manufacturing'},
            {'customer_name': 'XYZ Retail Corp', 'contact_email': 'ap@xyzretail.com', 'payment_history_score': 92.0, 'average_days_to_pay': 15.0, 'credit_limit': 75000.0, 'industry': 'Retail'},
            {'customer_name': 'Global Services Ltd', 'contact_email': 'payments@globalservices.com', 'payment_history_score': 78.0, 'average_days_to_pay': 35.0, 'credit_limit': 30000.0, 'industry': 'Services'},
            {'customer_name': 'Tech Solutions Inc', 'contact_email': 'finance@techsolutions.com', 'payment_history_score': 95.0, 'average_days_to_pay': 10.0, 'credit_limit': 100000.0, 'industry': 'Technology'},
            {'customer_name': 'Construction Co', 'contact_email': 'billing@constructionco.com', 'payment_history_score': 70.0, 'average_days_to_pay': 45.0, 'credit_limit': 40000.0, 'industry': 'Construction'},
            {'customer_name': 'Healthcare Partners', 'contact_email': 'ap@healthcarepartners.com', 'payment_history_score': 88.0, 'average_days_to_pay': 20.0, 'credit_limit': 60000.0, 'industry': 'Healthcare'},
            {'customer_name': 'Energy Corp', 'contact_email': 'payments@energycorp.com', 'payment_history_score': 82.0, 'average_days_to_pay': 30.0, 'credit_limit': 80000.0, 'industry': 'Energy'},
            {'customer_name': 'Food Processing Ltd', 'contact_email': 'finance@foodprocessing.com', 'payment_history_score': 75.0, 'average_days_to_pay': 40.0, 'credit_limit': 35000.0, 'industry': 'Food'},
        ]
        
        customers_df = pd.DataFrame(customers)
        customers_df.to_sql('customers', conn, if_exists='replace', index=False)
        
        # Generate vendors
        vendors = [
            {'vendor_name': 'Office Supplies Plus', 'contact_email': 'billing@officesupplies.com', 'category': 'Office Expenses', 'payment_terms': 30, 'average_monthly_spend': 2500.0},
            {'vendor_name': 'Industrial Equipment Co', 'contact_email': 'sales@industrialequip.com', 'category': 'Equipment', 'payment_terms': 45, 'average_monthly_spend': 15000.0},
            {'vendor_name': 'Legal Services LLC', 'contact_email': 'billing@legalservices.com', 'category': 'Professional Services', 'payment_terms': 15, 'average_monthly_spend': 5000.0},
            {'vendor_name': 'Utilities Company', 'contact_email': 'billing@utilities.com', 'category': 'Utilities', 'payment_terms': 30, 'average_monthly_spend': 3500.0},
            {'vendor_name': 'Software Solutions', 'contact_email': 'billing@software.com', 'category': 'Technology', 'payment_terms': 30, 'average_monthly_spend': 4000.0},
            {'vendor_name': 'Marketing Agency', 'contact_email': 'billing@marketing.com', 'category': 'Marketing', 'payment_terms': 30, 'average_monthly_spend': 8000.0},
            {'vendor_name': 'Insurance Brokers', 'contact_email': 'billing@insurance.com', 'category': 'Insurance', 'payment_terms': 30, 'average_monthly_spend': 2000.0},
            {'vendor_name': 'Raw Materials Supplier', 'contact_email': 'billing@rawmaterials.com', 'category': 'Materials', 'payment_terms': 60, 'average_monthly_spend': 20000.0},
        ]
        
        vendors_df = pd.DataFrame(vendors)
        vendors_df.to_sql('vendors', conn, if_exists='replace', index=False)
        
        # Generate historical cash transactions
        transactions = []
        start_date = datetime.now() - timedelta(days=730)  # 2 years of history
        
        # Recurring revenue streams
        recurring_revenues = [
            {'description': 'Subscription Revenue', 'category': 'Revenue', 'subcategory': 'Subscriptions', 'base_amount': 45000, 'variation': 0.1},
            {'description': 'Contract Revenue', 'category': 'Revenue', 'subcategory': 'Contracts', 'base_amount': 35000, 'variation': 0.15},
            {'description': 'Product Sales', 'category': 'Revenue', 'subcategory': 'Products', 'base_amount': 60000, 'variation': 0.25},
        ]
        
        # Recurring expenses
        recurring_expenses = [
            {'description': 'Salary Payments', 'category': 'Payroll', 'subcategory': 'Salaries', 'base_amount': 85000, 'variation': 0.05},
            {'description': 'Rent Payment', 'category': 'Facilities', 'subcategory': 'Rent', 'base_amount': 12000, 'variation': 0.0},
            {'description': 'Utilities', 'category': 'Utilities', 'subcategory': 'Electric/Gas/Water', 'base_amount': 3500, 'variation': 0.3},
            {'description': 'Insurance Premiums', 'category': 'Insurance', 'subcategory': 'General', 'base_amount': 2000, 'variation': 0.1},
            {'description': 'Software Licenses', 'category': 'Technology', 'subcategory': 'Software', 'base_amount': 4000, 'variation': 0.1},
        ]
        
        current_date = start_date
        while current_date <= datetime.now():
            # Add monthly recurring revenues
            for revenue in recurring_revenues:
                if current_date.day == 1:  # Beginning of month
                    seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * current_date.month / 12)  # Seasonal variation
                    amount = revenue['base_amount'] * seasonal_factor * (1 + random.uniform(-revenue['variation'], revenue['variation']))
                    transactions.append({
                        'transaction_date': current_date,
                        'description': revenue['description'],
                        'category': revenue['category'],
                        'subcategory': revenue['subcategory'],
                        'amount': round(amount, 2),
                        'transaction_type': 'inflow',
                        'is_recurring': 1,
                        'recurrence_pattern': 'monthly'
                    })
            
            # Add monthly recurring expenses
            for expense in recurring_expenses:
                if current_date.day == 15:  # Mid-month
                    seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * current_date.month / 12)
                    amount = expense['base_amount'] * seasonal_factor * (1 + random.uniform(-expense['variation'], expense['variation']))
                    transactions.append({
                        'transaction_date': current_date,
                        'description': expense['description'],
                        'category': expense['category'],
                        'subcategory': expense['subcategory'],
                        'amount': round(amount, 2),
                        'transaction_type': 'outflow',
                        'is_recurring': 1,
                        'recurrence_pattern': 'monthly'
                    })
            
            # Add random one-time transactions
            if random.random() < 0.3:  # 30% chance of additional transaction
                one_time_transactions = [
                    {'description': 'Equipment Purchase', 'category': 'Capital Expenditure', 'type': 'outflow', 'range': (5000, 50000)},
                    {'description': 'Consulting Revenue', 'category': 'Revenue', 'type': 'inflow', 'range': (10000, 25000)},
                    {'description': 'Marketing Campaign', 'category': 'Marketing', 'type': 'outflow', 'range': (5000, 15000)},
                    {'description': 'Tax Payment', 'category': 'Taxes', 'type': 'outflow', 'range': (8000, 30000)},
                    {'description': 'Investment Income', 'category': 'Investment', 'type': 'inflow', 'range': (1000, 5000)},
                ]
                
                transaction = random.choice(one_time_transactions)
                amount = random.uniform(*transaction['range'])
                transactions.append({
                    'transaction_date': current_date,
                    'description': transaction['description'],
                    'category': transaction['category'],
                    'subcategory': None,
                    'amount': round(amount, 2),
                    'transaction_type': transaction['type'],
                    'is_recurring': 0,
                    'recurrence_pattern': None
                })
            
            current_date += timedelta(days=1)
        
        transactions_df = pd.DataFrame(transactions)
        transactions_df.to_sql('cash_transactions', conn, if_exists='replace', index=False)
        
        # Generate accounts receivable
        receivables = []
        for i in range(100):
            customer_id = random.randint(1, len(customers))
            invoice_date = datetime.now() - timedelta(days=random.randint(0, 180))
            payment_terms = random.choice([15, 30, 45, 60])
            due_date = invoice_date + timedelta(days=payment_terms)
            amount = random.uniform(5000, 50000)
            
            # Determine status based on due date
            days_overdue = (datetime.now() - due_date).days
            if days_overdue > 0:
                status = 'overdue'
            elif random.random() < 0.7:  # 70% chance of being paid
                status = 'paid'
                amount_paid = amount
            elif random.random() < 0.5:  # Some partial payments
                status = 'partial'
                amount_paid = amount * random.uniform(0.3, 0.8)
            else:
                status = 'outstanding'
                amount_paid = 0
            
            receivables.append({
                'customer_id': customer_id,
                'invoice_number': f'INV-{i+1:04d}',
                'invoice_date': invoice_date.date(),
                'due_date': due_date.date(),
                'amount': round(amount, 2),
                'amount_paid': round(amount_paid if 'amount_paid' in locals() else 0, 2),
                'status': status,
                'payment_terms': payment_terms,
                'customer_name': customers[customer_id-1]['customer_name'],
                'product_service': random.choice(['Software License', 'Consulting Services', 'Product Sales', 'Maintenance Contract'])
            })
        
        receivables_df = pd.DataFrame(receivables)
        receivables_df.to_sql('accounts_receivable', conn, if_exists='replace', index=False)
        
        # Generate accounts payable
        payables = []
        for i in range(80):
            vendor_id = random.randint(1, len(vendors))
            invoice_date = datetime.now() - timedelta(days=random.randint(0, 120))
            payment_terms = vendors[vendor_id-1]['payment_terms']
            due_date = invoice_date + timedelta(days=payment_terms)
            amount = random.uniform(1000, 25000)
            
            # Determine status
            days_overdue = (datetime.now() - due_date).days
            if days_overdue > 30:
                status = 'overdue'
                amount_paid = 0
            elif random.random() < 0.6:  # 60% chance of being paid
                status = 'paid'
                amount_paid = amount
            else:
                status = 'outstanding'
                amount_paid = 0
            
            payables.append({
                'vendor_id': vendor_id,
                'invoice_number': f'VINV-{i+1:04d}',
                'invoice_date': invoice_date.date(),
                'due_date': due_date.date(),
                'amount': round(amount, 2),
                'amount_paid': round(amount_paid if 'amount_paid' in locals() else 0, 2),
                'status': status,
                'payment_terms': payment_terms,
                'vendor_name': vendors[vendor_id-1]['vendor_name'],
                'category': vendors[vendor_id-1]['category']
            })
        
        payables_df = pd.DataFrame(payables)
        payables_df.to_sql('accounts_payable', conn, if_exists='replace', index=False)
        
        # Generate budget items
        budget_items = [
            {'category': 'Revenue', 'subcategory': 'Product Sales', 'budget_amount': 720000, 'period_type': 'annual', 'is_fixed': 0},
            {'category': 'Revenue', 'subcategory': 'Service Revenue', 'budget_amount': 480000, 'period_type': 'annual', 'is_fixed': 0},
            {'category': 'Payroll', 'subcategory': 'Salaries', 'budget_amount': 1020000, 'period_type': 'annual', 'is_fixed': 1},
            {'category': 'Facilities', 'subcategory': 'Rent', 'budget_amount': 144000, 'period_type': 'annual', 'is_fixed': 1},
            {'category': 'Marketing', 'subcategory': 'Digital Marketing', 'budget_amount': 96000, 'period_type': 'annual', 'is_fixed': 0},
            {'category': 'Technology', 'subcategory': 'Software', 'budget_amount': 48000, 'period_type': 'annual', 'is_fixed': 1},
            {'category': 'Utilities', 'subcategory': 'General', 'budget_amount': 42000, 'period_type': 'annual', 'is_fixed': 0},
        ]
        
        budget_df = pd.DataFrame(budget_items)
        current_year = datetime.now().year
        budget_df['start_date'] = f'{current_year}-01-01'
        budget_df['end_date'] = f'{current_year}-12-31'
        budget_df.to_sql('budget_items', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"Generated sample data:")
        print(f"- {len(customers)} customers")
        print(f"- {len(vendors)} vendors")
        print(f"- {len(transactions)} historical cash transactions")
        print(f"- {len(receivables)} accounts receivable records")
        print(f"- {len(payables)} accounts payable records")
        print(f"- {len(budget_items)} budget items")
    
    def get_cash_transactions(self, start_date=None, end_date=None):
        """Retrieve cash transactions within date range"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM cash_transactions"
        params = []
        
        if start_date or end_date:
            query += " WHERE"
            conditions = []
            if start_date:
                conditions.append(" transaction_date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append(" transaction_date <= ?")
                params.append(end_date)
            query += " AND".join(conditions)
        
        query += " ORDER BY transaction_date DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_outstanding_receivables(self):
        """Get outstanding accounts receivable"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM accounts_receivable 
            WHERE status IN ('outstanding', 'partial', 'overdue')
            ORDER BY due_date ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_outstanding_payables(self):
        """Get outstanding accounts payable"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM accounts_payable 
            WHERE status IN ('outstanding', 'partial', 'overdue')
            ORDER BY due_date ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df