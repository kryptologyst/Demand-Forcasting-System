"""
Mock Database Generator for Demand Forecasting System
Generates realistic sales data for multiple products across different categories
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import os

class MockDatabaseGenerator:
    def __init__(self, db_path='demand_forecast.db'):
        self.db_path = db_path
        self.products = [
            {'id': 1, 'name': 'Laptop Pro 15"', 'category': 'Electronics', 'price': 1299.99},
            {'id': 2, 'name': 'Wireless Headphones', 'category': 'Electronics', 'price': 199.99},
            {'id': 3, 'name': 'Coffee Maker Deluxe', 'category': 'Appliances', 'price': 89.99},
            {'id': 4, 'name': 'Running Shoes', 'category': 'Sports', 'price': 129.99},
            {'id': 5, 'name': 'Organic Green Tea', 'category': 'Food', 'price': 24.99},
            {'id': 6, 'name': 'Smartphone X', 'category': 'Electronics', 'price': 899.99},
            {'id': 7, 'name': 'Yoga Mat Premium', 'category': 'Sports', 'price': 49.99},
            {'id': 8, 'name': 'Blender Pro', 'category': 'Appliances', 'price': 159.99},
        ]
        
    def generate_sales_data(self, start_date='2022-01-01', periods=730):
        """Generate realistic sales data with seasonal patterns and trends"""
        np.random.seed(42)
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=periods, freq='D')
        sales_data = []
        
        for product in self.products:
            base_demand = self._get_base_demand(product['category'])
            
            for i, date in enumerate(dates):
                # Add seasonal patterns
                seasonal_factor = self._get_seasonal_factor(date, product['category'])
                
                # Add weekly patterns (higher sales on weekends for some categories)
                weekly_factor = self._get_weekly_factor(date, product['category'])
                
                # Add trend (gradual increase over time)
                trend_factor = 1 + (i / periods) * 0.3
                
                # Calculate demand with noise
                daily_demand = (base_demand * seasonal_factor * weekly_factor * 
                              trend_factor * np.random.normal(1, 0.2))
                
                # Ensure non-negative demand
                daily_demand = max(0, int(daily_demand))
                
                sales_data.append({
                    'date': date,
                    'product_id': product['id'],
                    'product_name': product['name'],
                    'category': product['category'],
                    'units_sold': daily_demand,
                    'revenue': daily_demand * product['price'],
                    'price': product['price']
                })
        
        return pd.DataFrame(sales_data)
    
    def _get_base_demand(self, category):
        """Get base daily demand by category"""
        base_demands = {
            'Electronics': 15,
            'Appliances': 8,
            'Sports': 12,
            'Food': 25
        }
        return base_demands.get(category, 10)
    
    def _get_seasonal_factor(self, date, category):
        """Apply seasonal patterns based on category and date"""
        month = date.month
        
        if category == 'Electronics':
            # Higher demand in Nov-Dec (holiday season)
            if month in [11, 12]:
                return 1.8
            elif month in [1, 6, 7]:  # New Year and summer
                return 1.3
            else:
                return 1.0
                
        elif category == 'Sports':
            # Higher demand in spring/summer
            if month in [3, 4, 5, 6, 7, 8]:
                return 1.4
            else:
                return 0.8
                
        elif category == 'Appliances':
            # Higher demand in spring (moving season) and holidays
            if month in [3, 4, 5, 11, 12]:
                return 1.3
            else:
                return 1.0
                
        elif category == 'Food':
            # Relatively stable with slight increase in winter
            if month in [11, 12, 1, 2]:
                return 1.2
            else:
                return 1.0
                
        return 1.0
    
    def _get_weekly_factor(self, date, category):
        """Apply weekly patterns (weekday vs weekend)"""
        weekday = date.weekday()  # 0=Monday, 6=Sunday
        
        if category in ['Electronics', 'Appliances']:
            # Higher sales on weekends
            if weekday >= 5:  # Saturday, Sunday
                return 1.3
            else:
                return 1.0
        else:
            # More consistent throughout the week
            return 1.0
    
    def create_database(self):
        """Create SQLite database with sales data"""
        # Generate sales data
        df = self.generate_sales_data()
        
        # Create database connection
        conn = sqlite3.connect(self.db_path)
        
        # Create products table
        products_df = pd.DataFrame(self.products)
        products_df.to_sql('products', conn, if_exists='replace', index=False)
        
        # Create sales table
        df.to_sql('sales', conn, if_exists='replace', index=False)
        
        # Create aggregated monthly data for easier analysis
        monthly_sales = df.groupby([
            df['date'].dt.to_period('M'), 'product_id', 'product_name', 'category'
        ]).agg({
            'units_sold': 'sum',
            'revenue': 'sum',
            'price': 'first'
        }).reset_index()
        
        monthly_sales['month'] = monthly_sales['date'].astype(str)
        monthly_sales = monthly_sales.drop('date', axis=1)
        monthly_sales.to_sql('monthly_sales', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"Database created successfully: {self.db_path}")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate mock database
    generator = MockDatabaseGenerator('data/demand_forecast.db')
    sales_df = generator.create_database()
    
    # Display sample data
    print("\nSample sales data:")
    print(sales_df.head(10))
    
    print("\nMonthly summary by category:")
    monthly_summary = sales_df.groupby([
        sales_df['date'].dt.to_period('M'), 'category'
    ])['units_sold'].sum().unstack(fill_value=0)
    print(monthly_summary.head())
